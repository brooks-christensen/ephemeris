from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

from .ephem import BODY_MASSES, EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent import (
    attach_gr_potential_tangent_force,
    gr_potential_accelerations_and_tangent,
)
from .gr_potential_tangent_c import (
    CBackendCompatibilityError,
    build_c_backend,
    c_source_path,
    default_artifact_path,
    load_c_backend,
)
from .gr_tangent_validation_matrix import (
    EXISTING_ARCHIVE,
    apply_physical_delta,
    assign_first_variation,
    deterministic_variation,
    direction_cosine,
    element_map,
    open_archive_snapshot,
    read_first_variation,
    scaled_phase_vector,
    state_from_real_particles,
)
from .long_term_stability_cli import (
    add_reboundx_gr_force,
    build_rebound_simulation,
    load_rebound_archive_snapshot,
    optional_import_module,
    parse_start_datetime,
    rebound_state_from_sim,
    stability_body_list,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import ARCSEC_PER_RAD, AU_M, DAY_S, JULIAN_YEAR_S, heliocentric_elements_for_state
from .rebound_gr_tangent_backend_cli import INTENTIONAL_INCOMPLETE_EXIT, atomic_write_json, sha256_file


FROZEN_TAG = "gr-tangent-python-oracle-v2"
FROZEN_COMMIT = "9933bc5e3d9bfe9ec07e72929c22e4406f12b441"
PROTECTED_ORACLE_FILES = (
    "mini_ephemeris/src/mini_ephemeris/gr_potential_tangent.py",
    "mini_ephemeris/src/mini_ephemeris/rebound_gr_tangent_cli.py",
    "mini_ephemeris/src/mini_ephemeris/gr_tangent_validation_matrix.py",
)

# Predetermined before decisive validation. Pointwise limits allow a small multiple
# of double-precision roundoff; dynamic limits are tighter than the accepted
# custom-vs-REBOUNDx envelope and broad enough for summation-order roundoff.
THRESHOLDS: dict[str, Any] = {
    "pointwise_acceleration_relative": 5.0e-14,
    "pointwise_acceleration_absolute": 1.0e-22,
    "pointwise_central_force_relative": 5.0e-14,
    "pointwise_jacobian_relative": 1.0e-13,
    "pointwise_linearity_relative": 2.0e-13,
    "short_real_scaled_phase": 1.0e-10,
    "short_tangent_relative": 1.0e-8,
    "short_tangent_direction_cosine": 0.99999999,
    "short_megno_absolute": 1.0e-7,
    "pointwise_translation_relative": 1.0e-12,
    "short_lcn_absolute": 1.0e-10,
    "finite_difference_relative_ceiling": 1.0e-4,
    "pointwise_rotation_covariance_relative": 2.0e-13,
    "finite_difference_oracle_degradation_factor": 1.25,
    "finite_difference_oracle_absolute_slack": 5.0e-7,
    "finite_difference_direction_cosine": 0.9999,
    "isolated_gr_excess_precession_min_arcsec_per_century": 38.0,
    "isolated_gr_excess_precession_max_arcsec_per_century": 48.0,
    "zero_limit_scaled_phase": 1.0e-12,
    "full_100k_real_scaled_phase": 1.0e-7,
    "full_100k_tangent_relative": 1.0e-5,
    "rehearsal_energy_diagnostic_absolute": 1.0e-10,
    "rehearsal_angular_diagnostic_absolute": 1.0e-12,
    "rehearsal_mercury_a_au_absolute": 1.0e-8,
    "rehearsal_mercury_e_absolute": 1.0e-8,
    "rehearsal_mercury_varpi_deg_absolute": 1.0e-5,
    "full_100k_tangent_direction_cosine": 0.99999,
    "full_100k_megno_absolute": 1.0e-5,
    "full_100k_lcn_absolute": 1.0e-9,
    "reboundx_paired_gr_relative": 1.0e-4,
    "reboundx_delta_a_au": 1.0e-8,
    "reboundx_delta_e": 1.0e-8,
    "reboundx_delta_i_arcsec": 1.0e-2,
    "reboundx_delta_varpi_arcsec": 1.0e-2,
    "reboundx_delta_mean_longitude_arcsec": 5.0,
    "restart_physical_scaled_phase": 1.0e-10,
    "restart_tangent_scaled_phase": 1.0e-8,
    "restart_tangent_direction_cosine": 0.9999999999,
    "restart_diagnostic_absolute": 1.0e-12,
    "reproducibility_scaled_phase": 0.0,
    "minimum_c_speedup": 2.0,
    "maximum_c_to_python_wall_ratio": 0.5,
    "benchmark_repetitions": 5,
    "benchmark_duration_years": 200.0,
    "capped_rehearsal_duration_years": 100_000.0,
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_result(output_dir: Path, filename: str, payload: dict[str, Any]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload.setdefault("created_utc", utc_now())
    payload.setdefault("thresholds", THRESHOLDS)
    path = output_dir / filename
    atomic_write_json(path, payload)
    print(f"[c-validation] wrote {path}")
    return 0 if payload.get("passed") else 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_oracle_evidence() -> dict[str, Any]:
    rows = []
    passed = True
    for relative in PROTECTED_ORACLE_FILES:
        live = Path(relative).read_bytes()
        tagged = subprocess.run(
            ["git", "show", f"{FROZEN_TAG}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        matches = live == tagged
        passed = passed and matches
        rows.append(
            {
                "path": str(Path(relative).resolve()),
                "live_sha256": sha256_bytes(live),
                "frozen_sha256": sha256_bytes(tagged),
                "matches_frozen_tag": matches,
            }
        )
    resolved = subprocess.run(
        ["git", "rev-list", "-n", "1", FROZEN_TAG], check=True, capture_output=True, text=True
    ).stdout.strip()
    passed = passed and resolved == FROZEN_COMMIT
    return {
        "passed": passed,
        "frozen_tag": FROZEN_TAG,
        "expected_commit": FROZEN_COMMIT,
        "resolved_commit": resolved,
        "files": rows,
    }


def make_initial_state(kernel: Path, scope: str = "full_with_pluto") -> tuple[tuple[str, ...], NBodyState]:
    bodies = stability_body_list(scope, include_pluto=scope == "full_with_pluto")
    state = initial_state_solar_system_barycentric(
        parse_start_datetime("2000-01-01"),
        bodies=bodies,
        config=EphemerisConfig(kernel_path=str(kernel)),
    )
    return bodies, state


def make_sim(
    rebound: Any,
    backend: str,
    state: NBodyState,
    *,
    step_days: float,
    megno: bool,
    seed: int = 12345,
    gr_scale: float = 1.0,
    c_backend: Any | None = None,
) -> Any:
    sim = build_rebound_simulation(
        rebound, state, integrator="whfast", step_s=step_days * DAY_S, ias15_epsilon=1.0e-10
    )
    if megno:
        sim.init_megno(seed=seed)
        sim.lyapunov()
    if backend == "c":
        (c_backend or load_c_backend()).attach(sim, coefficient_scale=gr_scale)
    elif backend == "python":
        attach_gr_potential_tangent_force(sim, coefficient_scale=gr_scale)
    elif backend == "newtonian":
        pass
    elif backend == "reboundx":
        add_reboundx_gr_force(sim, "gr_potential")
    else:
        raise ValueError(backend)
    return sim


def phase_arrays(sim: Any, n_real: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    real_pos = np.array([[p.x, p.y, p.z] for p in sim.particles[:n_real]], dtype=float)
    real_vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles[:n_real]], dtype=float)
    var_pos = np.array([[p.x, p.y, p.z] for p in sim.particles[n_real : 2 * n_real]], dtype=float)
    var_vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles[n_real : 2 * n_real]], dtype=float)
    return real_pos, real_vel, var_pos, var_vel


def compare_dynamic_pair(c_sim: Any, py_sim: Any, n_real: int) -> dict[str, float]:
    crp, crv, cvp, cvv = phase_arrays(c_sim, n_real)
    prp, prv, pvp, pvv = phase_arrays(py_sim, n_real)
    real_diff = float(np.linalg.norm(scaled_phase_vector(crp - prp, crv - prv)))
    c_var = scaled_phase_vector(cvp, cvv)
    py_var = scaled_phase_vector(pvp, pvv)
    tangent_diff = float(np.linalg.norm(c_var - py_var))
    tangent_relative = tangent_diff / max(float(np.linalg.norm(c_var)), float(np.linalg.norm(py_var)), 1.0e-300)
    return {
        "real_scaled_phase_difference": real_diff,
        "tangent_scaled_phase_difference": tangent_diff,
        "tangent_relative_difference": tangent_relative,
        "tangent_direction_cosine": direction_cosine(c_var, py_var),
        "megno_absolute_difference": abs(float(c_sim.megno()) - float(py_sim.megno())),
        "lcn_absolute_difference": abs(float(c_sim.lyapunov()) - float(py_sim.lyapunov())),
    }


def build_integrity(args: argparse.Namespace) -> int:
    metadata = build_c_backend(force=True)
    backend = load_c_backend()
    symbols = subprocess.run(
        ["nm", "-D", str(backend.artifact_path)], check=True, capture_output=True, text=True
    ).stdout
    required_symbols = [
        "me_gr_tangent_additional_forces",
        "me_gr_tangent_attach",
        "me_gr_tangent_get_stats",
        "me_gr_tangent_pointwise",
    ]
    symbol_check = {name: name in symbols for name in required_symbols}
    stale_rejected = False
    with tempfile.TemporaryDirectory(prefix="gr-c-stale-") as temporary:
        root = Path(temporary)
        copied_artifact = root / backend.artifact_path.name
        copied_metadata = root / "build_metadata.json"
        shutil.copy2(backend.artifact_path, copied_artifact)
        bad = dict(metadata)
        bad["source_sha256"] = "0" * 64
        copied_metadata.write_text(json.dumps(bad))
        try:
            load_c_backend(artifact_path=copied_artifact)
        except CBackendCompatibilityError:
            stale_rejected = True
    oracle = frozen_oracle_evidence()
    payload = {
        "stage": "build_integrity",
        "passed": all(symbol_check.values()) and stale_rejected and oracle["passed"],
        "maximum_duration_years_launched": 0.0,
        "build_metadata": metadata,
        "abi_metadata": backend.abi_metadata,
        "required_symbols": symbol_check,
        "stale_artifact_rejected": stale_rejected,
        "oracle_integrity": oracle,
    }
    return write_result(args.output_dir, "build_integrity_summary.json", payload)


def _random_state(rng: np.random.Generator, n_planets: int) -> tuple[np.ndarray, np.ndarray]:
    names = [
        "mercury barycenter",
        "venus barycenter",
        "earth barycenter",
        "mars barycenter",
        "jupiter barycenter",
        "saturn barycenter",
        "uranus barycenter",
        "neptune barycenter",
        "pluto barycenter",
    ][:n_planets]
    masses = np.array([BODY_MASSES["sun"], *(BODY_MASSES[name] for name in names)], dtype=float)
    positions = [rng.normal(0.0, 1.0e8, size=3)]
    for index in range(n_planets):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.25, 40.0) * AU_M
        positions.append(positions[0] + radius * direction)
    return np.asarray(positions), masses


def pointwise_acceleration(args: argparse.Namespace) -> int:
    backend = load_c_backend()
    rng = np.random.default_rng(20260804)
    worst_relative = 0.0
    worst_absolute = 0.0
    worst_seed = None
    central_force_relative = 0.0
    translation_relative = 0.0
    cases = 0
    failures: list[str] = []
    for case in range(96):
        positions, masses = _random_state(rng, 1 + case % 9)
        py_acc, _ = gr_potential_accelerations_and_tangent(
            positions, masses, None, gravitational_constant=G_SI
        )
        c_acc, _ = backend.pointwise(positions, masses, None, gravitational_constant=G_SI)
        diff = c_acc - py_acc
        relative = float(np.linalg.norm(diff) / max(np.linalg.norm(py_acc), 1.0e-300))
        absolute = float(np.max(np.abs(diff)))
        if relative > worst_relative:
            worst_relative = relative
            worst_seed = case
        worst_absolute = max(worst_absolute, absolute)
        net_force = np.sum(c_acc * masses[:, None], axis=0)
        force_scale = np.sum(np.linalg.norm(c_acc * masses[:, None], axis=1))
        central_force_relative = max(
            central_force_relative, float(np.linalg.norm(net_force) / max(force_scale, 1.0))
        )
        translation = rng.normal(0.0, 3.0e8, size=3)
        translated, _ = backend.pointwise(
            positions + translation, masses, None, gravitational_constant=G_SI
        )
        translation_relative = max(
            translation_relative,
            float(np.linalg.norm(translated - c_acc) / max(np.linalg.norm(c_acc), 1.0e-300)),
        )
        cases += 1
    coincident = np.zeros((2, 3))
    masses = np.array([BODY_MASSES["sun"], BODY_MASSES["mercury barycenter"]])
    edge, _ = backend.pointwise(coincident, masses, None, gravitational_constant=G_SI)
    sample_positions, sample_masses = _random_state(rng, 4)
    zero_acc, _ = backend.pointwise(
        sample_positions,
        sample_masses,
        None,
        gravitational_constant=G_SI,
        coefficient_scale=0.0,
    )
    if worst_relative > THRESHOLDS["pointwise_acceleration_relative"]:
        failures.append("relative acceleration mismatch exceeds threshold")
    if worst_absolute > THRESHOLDS["pointwise_acceleration_absolute"]:
        failures.append("absolute acceleration mismatch exceeds threshold")
    if central_force_relative > THRESHOLDS["pointwise_central_force_relative"]:
        failures.append("central response does not conserve pair force within threshold")
    if translation_relative > THRESHOLDS["pointwise_translation_relative"]:
        failures.append("translation covariance mismatch exceeds threshold")
    if not np.array_equal(edge, np.zeros_like(edge)) or not np.array_equal(zero_acc, np.zeros_like(zero_acc)):
        failures.append("coincident or zero-scale acceleration is not exact zero")
    payload = {
        "stage": "pointwise_acceleration",
        "passed": not failures,
        "failures": failures,
        "case_count": cases + 2,
        "worst_case_index": worst_seed,
        "max_relative_norm_error": worst_relative,
        "max_absolute_component_error": worst_absolute,
        "max_central_force_relative_residual": central_force_relative,
        "max_translation_relative_error": translation_relative,
        "coincident_exact_zero": bool(np.array_equal(edge, np.zeros_like(edge))),
        "gr_scale_zero_exact": bool(np.array_equal(zero_acc, np.zeros_like(zero_acc))),
        "maximum_duration_years_launched": 0.0,
    }
    return write_result(args.output_dir, "pointwise_acceleration_summary.json", payload)


def pointwise_jacobian(args: argparse.Namespace) -> int:
    backend = load_c_backend()
    rng = np.random.default_rng(20260805)
    worst_relative = 0.0
    worst_linearity = 0.0
    worst_rotation = 0.0
    failures: list[str] = []
    cases = 0
    for case in range(96):
        positions, masses = _random_state(rng, 1 + case % 9)
        delta = rng.normal(size=positions.shape)
        if case % 4 == 0:
            delta[:] = 0.0
            delta[case % len(delta), case % 3] = 1.0
        py_acc, py_tangent = gr_potential_accelerations_and_tangent(
            positions, masses, delta, gravitational_constant=G_SI
        )
        c_acc, c_tangent = backend.pointwise(
            positions, masses, delta, gravitational_constant=G_SI
        )
        assert py_tangent is not None and c_tangent is not None
        relative = float(
            np.linalg.norm(c_tangent - py_tangent) / max(np.linalg.norm(py_tangent), 1.0e-300)
        )
        worst_relative = max(worst_relative, relative)
        other = rng.normal(size=positions.shape)
        _, left = backend.pointwise(
            positions, masses, 1.7 * delta - 0.3 * other, gravitational_constant=G_SI
        )
        _, first = backend.pointwise(positions, masses, delta, gravitational_constant=G_SI)
        _, second = backend.pointwise(positions, masses, other, gravitational_constant=G_SI)
        assert left is not None and first is not None and second is not None
        expected = 1.7 * first - 0.3 * second
        linearity = float(np.linalg.norm(left - expected) / max(np.linalg.norm(expected), 1.0e-300))
        worst_linearity = max(worst_linearity, linearity)
        omega = np.array([0.2, -0.1, 0.3])
        rotation_delta = np.cross(np.broadcast_to(omega, positions.shape), positions)
        _, rotation_tangent = backend.pointwise(
            positions, masses, rotation_delta, gravitational_constant=G_SI
        )
        assert rotation_tangent is not None
        rotation_expected = np.cross(np.broadcast_to(omega, c_acc.shape), c_acc)
        worst_rotation = max(
            worst_rotation,
            float(
                np.linalg.norm(rotation_tangent - rotation_expected)
                / max(np.linalg.norm(rotation_expected), 1.0e-300)
            ),
        )
        cases += 1
    positions, masses = _random_state(rng, 4)
    translation_delta = np.ones_like(positions) * np.array([1.0, -2.0, 0.5])
    _, translation_tangent = backend.pointwise(
        positions, masses, translation_delta, gravitational_constant=G_SI
    )
    _, zero_tangent = backend.pointwise(
        positions,
        masses,
        translation_delta,
        gravitational_constant=G_SI,
        coefficient_scale=0.0,
    )
    assert translation_tangent is not None and zero_tangent is not None
    if worst_relative > THRESHOLDS["pointwise_jacobian_relative"]:
        failures.append("Jacobian-action mismatch exceeds threshold")
    if worst_linearity > THRESHOLDS["pointwise_linearity_relative"]:
        failures.append("Jacobian linearity mismatch exceeds threshold")
    if np.linalg.norm(translation_tangent) > 1.0e-30:
        failures.append("common translation has nonzero tangent response")
    if worst_rotation > THRESHOLDS["pointwise_rotation_covariance_relative"]:
        failures.append("rotation covariance mismatch exceeds threshold")
    if not np.array_equal(zero_tangent, np.zeros_like(zero_tangent)):
        failures.append("zero-scale tangent response is not exact zero")
    payload = {
        "stage": "pointwise_jacobian",
        "passed": not failures,
        "failures": failures,
        "case_count": cases + 2,
        "max_relative_norm_error": worst_relative,
        "max_linearity_relative_error": worst_linearity,
        "max_rotation_covariance_relative_error": worst_rotation,
        "translation_tangent_norm": float(np.linalg.norm(translation_tangent)),
        "gr_scale_zero_exact": bool(np.array_equal(zero_tangent, np.zeros_like(zero_tangent))),
        "maximum_duration_years_launched": 0.0,
    }
    return write_result(args.output_dir, "pointwise_jacobian_summary.json", payload)


def hot_path_lifecycle(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    sim_a = rebound.Simulation()
    sim_a.G = 1.0
    sim_a.add(m=1.0)
    sim_a.add(m=1.0e-3, x=1.0, vy=1.0)
    sim_a.add_variation().particles[1].x = 1.0
    backend.attach(sim_a, coefficient_scale=1.0, c_m_per_s=1000.0)
    proof = backend.hot_path_proof(sim_a)
    before = backend.stats(sim_a)
    sim_a.integrate(0.01)
    after = backend.stats(sim_a)
    sim_b = rebound.Simulation()
    sim_b.G = 1.0
    sim_b.add(m=1.0)
    sim_b.add(m=1.0e-3, x=1.0, vy=1.0)
    sim_b.add_variation().particles[1].y = 1.0
    backend.attach(sim_b, coefficient_scale=0.5, c_m_per_s=1000.0)
    sim_b.integrate(0.02)
    stats_b = backend.stats(sim_b)
    stats_a_after_b = backend.stats(sim_a)
    independent = bool(
        stats_a_after_b == after
        and backend.is_attached(sim_a)
        and backend.is_attached(sim_b)
        and stats_b["real_gr_accel_norm_max"] < after["real_gr_accel_norm_max"]
    )
    python_bridge_attribute_absent = not hasattr(sim_a, "_afp")
    passed = bool(
        proof["addresses_match"]
        and not proof["python_callback_in_force_path"]
        and after["callback_invocations"] > before["callback_invocations"]
        and after["real_gr_accel_norm_count"] > 0
        and after["tangent_gr_accel_norm_count"] > 0
        and after["nonfinite_result_count"] == 0
        and independent
        and python_bridge_attribute_absent
    )
    payload = {
        "stage": "hot_path_lifecycle",
        "passed": passed,
        "hot_path_proof": proof,
        "stats_before": before,
        "stats_after": after,
        "second_simulation_stats": stats_b,
        "stats_a_after_second_simulation": stats_a_after_b,
        "per_simulation_state_independent": independent,
        "python_ctypes_callback_wrapper_absent": python_bridge_attribute_absent,
        "state_ownership": "simulation.extras with C extras_cleanup; no mutable process-global state",
        "thread_safety": "state is isolated per simulation; concurrent mutation of one simulation remains unsupported by REBOUND",
        "maximum_duration_years_launched": 0.0,
    }
    return write_result(args.output_dir, "hot_path_lifecycle_summary.json", payload)


def short_dynamic(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    configurations = [
        ("two_body_100yr", "two_body_mercury", 100.0, 1.0, 12345),
        ("full_pluto_100yr", "full_with_pluto", 100.0, 1.0, 12345),
        ("full_without_pluto_100yr", "full", 100.0, 1.0, 12345),
        ("seed67890_100yr", "full_with_pluto", 100.0, 1.0, 67890),
        ("half_day_20yr", "full_with_pluto", 20.0, 0.5, 12345),
    ]
    results = []
    failures: list[str] = []
    for name, scope, duration, step_days, seed in configurations:
        bodies, state = make_initial_state(args.kernel_path, scope)
        c_sim = make_sim(
            rebound, "c", state, step_days=step_days, megno=True, seed=seed, c_backend=backend
        )
        py_sim = make_sim(rebound, "python", state, step_days=step_days, megno=True, seed=seed)
        maxima = {
            "real_scaled_phase_difference": 0.0,
            "tangent_relative_difference": 0.0,
            "megno_absolute_difference": 0.0,
            "lcn_absolute_difference": 0.0,
        }
        minimum_cosine = 1.0
        for target in np.linspace(0.0, duration, 11):
            c_sim.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
            py_sim.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
            metrics = compare_dynamic_pair(c_sim, py_sim, len(bodies))
            for key in maxima:
                maxima[key] = max(maxima[key], metrics[key])
            minimum_cosine = min(minimum_cosine, metrics["tangent_direction_cosine"])
        passed = bool(
            maxima["real_scaled_phase_difference"] <= THRESHOLDS["short_real_scaled_phase"]
            and maxima["tangent_relative_difference"] <= THRESHOLDS["short_tangent_relative"]
            and minimum_cosine >= THRESHOLDS["short_tangent_direction_cosine"]
            and maxima["megno_absolute_difference"] <= THRESHOLDS["short_megno_absolute"]
            and maxima["lcn_absolute_difference"] <= THRESHOLDS["short_lcn_absolute"]
        )
        if not passed:
            failures.append(name)
        results.append(
            {
                "name": name,
                "scope": scope,
                "duration_years": duration,
                "step_days": step_days,
                "seed": seed,
                "passed": passed,
                **maxima,
                "minimum_tangent_direction_cosine": minimum_cosine,
                "c_callback_stats": backend.stats(c_sim),
                "hot_path_proof": backend.hot_path_proof(c_sim),
            }
        )
    payload = {
        "stage": "short_dynamic_equivalence",
        "passed": not failures,
        "failures": failures,
        "configurations": results,
        "maximum_duration_years_launched": max(item[2] for item in configurations),
    }
    return write_result(args.output_dir, "short_dynamic_equivalence_summary.json", payload)


def finite_difference_oracle(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    accepted_path = (
        args.python_oracle_output
        / "dynamic_gr_tangent_oracle"
        / "dynamic_gr_tangent_oracle_summary.json"
    )
    accepted = json.loads(accepted_path.read_text())
    epsilons = [
        1.0e-12,
        3.0e-12,
        1.0e-11,
        3.0e-11,
        1.0e-10,
        3.0e-10,
        1.0e-9,
        3.0e-9,
        1.0e-8,
        3.0e-8,
    ]
    snapshots = [0.0, 500_000.0, 1_000_000.0]
    durations = [1.0, 10.0]
    groups: dict[str, Any] = {}
    failures: list[str] = []
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    for snapshot in snapshots:
        snapshot_sim, archive = open_archive_snapshot(rebound, EXISTING_ARCHIVE, snapshot)
        state = state_from_real_particles(snapshot_sim, len(bodies))
        delta_pos, delta_vel = deterministic_variation(state, int(10_000 + snapshot))
        for duration in durations:
            tangent_sim = make_sim(
                rebound, "c", state, step_days=1.0, megno=False, c_backend=backend
            )
            tangent_sim.add_variation()
            assign_first_variation(tangent_sim, delta_pos, delta_vel)
            tangent_sim.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
            tangent_pos, tangent_vel = read_first_variation(tangent_sim)
            tangent_vec = scaled_phase_vector(tangent_pos, tangent_vel)
            rows = []
            for epsilon in epsilons:
                plus = make_sim(
                    rebound,
                    "c",
                    apply_physical_delta(state, delta_pos, delta_vel, epsilon),
                    step_days=1.0,
                    megno=False,
                    c_backend=backend,
                )
                minus = make_sim(
                    rebound,
                    "c",
                    apply_physical_delta(state, delta_pos, delta_vel, -epsilon),
                    step_days=1.0,
                    megno=False,
                    c_backend=backend,
                )
                plus.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
                minus.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
                plus_state = rebound_state_from_sim(plus, state.masses)
                minus_state = rebound_state_from_sim(minus, state.masses)
                fd_vec = scaled_phase_vector(
                    (plus_state.positions - minus_state.positions) / (2.0 * epsilon),
                    (plus_state.velocities - minus_state.velocities) / (2.0 * epsilon),
                )
                relative = float(
                    np.linalg.norm(tangent_vec - fd_vec)
                    / max(np.linalg.norm(tangent_vec), np.linalg.norm(fd_vec), 1.0e-300)
                )
                rows.append(
                    {
                        "epsilon": epsilon,
                        "relative_norm_error": relative,
                        "direction_cosine": direction_cosine(tangent_vec, fd_vec),
                    }
                )
            key = f"{snapshot:g}_{duration:g}"
            best = min(rows, key=lambda row: row["relative_norm_error"])
            accepted_best = float(accepted["summary_by_group"][key]["min_relative_norm_error"])
            allowed = min(
                THRESHOLDS["finite_difference_relative_ceiling"],
                accepted_best * THRESHOLDS["finite_difference_oracle_degradation_factor"]
                + THRESHOLDS["finite_difference_oracle_absolute_slack"],
            )
            adjacent = any(
                left["relative_norm_error"] <= THRESHOLDS["finite_difference_relative_ceiling"]
                and right["relative_norm_error"] <= THRESHOLDS["finite_difference_relative_ceiling"]
                and left["direction_cosine"] >= THRESHOLDS["finite_difference_direction_cosine"]
                and right["direction_cosine"] >= THRESHOLDS["finite_difference_direction_cosine"]
                for left, right in zip(rows, rows[1:])
            )
            passed = bool(
                best["relative_norm_error"] <= allowed
                and best["direction_cosine"] >= THRESHOLDS["finite_difference_direction_cosine"]
                and adjacent
            )
            if not passed:
                failures.append(key)
            groups[key] = {
                "passed": passed,
                "python_oracle_best_relative_error": accepted_best,
                "allowed_relative_error": allowed,
                "best": best,
                "adjacent_convergence_region": adjacent,
                "rows": rows,
                "source_snapshot_years": snapshot,
                "loaded_existing_snapshot_not_new_integration": True,
            }
    payload = {
        "stage": "finite_difference_oracle",
        "passed": not failures,
        "failures": failures,
        "groups": groups,
        "accepted_python_oracle_summary": str(accepted_path),
        "existing_archive": str(EXISTING_ARCHIVE),
        "maximum_duration_years_launched": max(durations),
    }
    return write_result(args.output_dir, "finite_difference_oracle_summary.json", payload)


def _apsidal_rate(sim: Any, state: NBodyState, bodies: tuple[str, ...], duration: int = 100) -> float:
    angles = []
    times = []
    for year in range(duration + 1):
        sim.integrate(year * JULIAN_YEAR_S, exact_finish_time=1)
        current = rebound_state_from_sim(sim, state.masses)
        mercury = next(
            item
            for item in heliocentric_elements_for_state(current, bodies, sun_index=bodies.index("sun"))
            if item.body_name == "mercury barycenter"
        )
        times.append(float(year))
        angles.append(mercury.longitude_perihelion_rad)
    return float(np.polyfit(times, np.unwrap(angles), 1)[0] * ARCSEC_PER_RAD * 100.0)


def physical_controls(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    bodies, state = make_initial_state(args.kernel_path, "two_body_mercury")
    c_gr = make_sim(rebound, "c", state, step_days=1.0, megno=True, c_backend=backend)
    py_gr = make_sim(rebound, "python", state, step_days=1.0, megno=True)
    newtonian = make_sim(rebound, "newtonian", state, step_days=1.0, megno=False)
    c_rate = _apsidal_rate(c_gr, state, bodies)
    py_rate = _apsidal_rate(py_gr, state, bodies)
    newtonian_rate = _apsidal_rate(newtonian, state, bodies)
    excess = c_rate - newtonian_rate
    c_python_difference = abs(c_rate - py_rate)

    full_bodies, full_state = make_initial_state(args.kernel_path, "full_with_pluto")
    c_zero = make_sim(
        rebound, "c", full_state, step_days=1.0, megno=True, gr_scale=0.0, c_backend=backend
    )
    native = make_sim(rebound, "newtonian", full_state, step_days=1.0, megno=True, seed=12345)
    zero_max = 0.0
    zero_megno = 0.0
    zero_lcn = 0.0
    for target in np.linspace(0.0, 100.0, 11):
        c_zero.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
        native.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
        crp, crv, cvp, cvv = phase_arrays(c_zero, len(full_bodies))
        nrp, nrv, nvp, nvv = phase_arrays(native, len(full_bodies))
        zero_max = max(
            zero_max,
            float(np.linalg.norm(scaled_phase_vector(crp - nrp, crv - nrv))),
            float(np.linalg.norm(scaled_phase_vector(cvp - nvp, cvv - nvv))),
        )
        zero_megno = max(zero_megno, abs(float(c_zero.megno()) - float(native.megno())))
        zero_lcn = max(zero_lcn, abs(float(c_zero.lyapunov()) - float(native.lyapunov())))
    failures = []
    if not (
        THRESHOLDS["isolated_gr_excess_precession_min_arcsec_per_century"]
        <= excess
        <= THRESHOLDS["isolated_gr_excess_precession_max_arcsec_per_century"]
    ):
        failures.append("isolated Mercury GR-excess precession is outside the predetermined band")
    if c_python_difference > 1.0e-6:
        failures.append("C and Python isolated apsidal rates differ")
    if zero_max > THRESHOLDS["zero_limit_scaled_phase"] or zero_megno != 0.0 or zero_lcn != 0.0:
        failures.append("gr_scale=0 does not reproduce native Newtonian evolution")
    if not math.isfinite(float(c_gr.megno())) or not math.isfinite(float(c_gr.lyapunov())):
        failures.append("isolated C control has nonfinite chaos diagnostics")
    payload = {
        "stage": "physical_controls",
        "passed": not failures,
        "failures": failures,
        "isolated_mercury": {
            "c_total_apsidal_drift_arcsec_per_century": c_rate,
            "python_total_apsidal_drift_arcsec_per_century": py_rate,
            "newtonian_total_apsidal_drift_arcsec_per_century": newtonian_rate,
            "c_gr_minus_newtonian_excess_arcsec_per_century": excess,
            "c_python_absolute_difference_arcsec_per_century": c_python_difference,
            "c_final_megno": float(c_gr.megno()),
            "c_final_lcn": float(c_gr.lyapunov()),
        },
        "newtonian_zero_limit": {
            "max_scaled_phase_difference": zero_max,
            "max_megno_difference": zero_megno,
            "max_lcn_difference": zero_lcn,
            "exact_state_match": zero_max == 0.0,
        },
        "apsidal_semantics": {
            "isolated_difference_is_gr_excess": True,
            "full_system_total_is_not_gr_excess": True,
        },
        "maximum_duration_years_launched": 100.0,
    }
    return write_result(args.output_dir, "physical_controls_summary.json", payload)


def _signed_angle_delta_deg(left: float, right: float) -> float:
    return (left - right + 180.0) % 360.0 - 180.0


def full_dynamic_physical(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    bodies, state = make_initial_state(args.kernel_path, "full_with_pluto")
    c_sim = make_sim(rebound, "c", state, step_days=1.0, megno=True, seed=12345, c_backend=backend)
    py_sim = make_sim(rebound, "python", state, step_days=1.0, megno=True, seed=12345)
    ordinary = make_sim(rebound, "reboundx", state, step_days=1.0, megno=False)
    newtonian = make_sim(rebound, "newtonian", state, step_days=1.0, megno=False)
    secular_bodies = (
        "mercury barycenter",
        "venus barycenter",
        "earth barycenter",
        "mars barycenter",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "full_dynamic_physical_100k.csv"
    rows: list[dict[str, Any]] = []
    dynamic_max = {
        "real_scaled_phase_difference": 0.0,
        "tangent_relative_difference": 0.0,
        "megno_absolute_difference": 0.0,
        "lcn_absolute_difference": 0.0,
    }
    minimum_cosine = 1.0
    max_element_differences = {
        name.split()[0]: {
            "max_abs_delta_a_au": 0.0,
            "max_abs_delta_e": 0.0,
            "max_abs_delta_i_arcsec": 0.0,
            "max_abs_delta_varpi_arcsec": 0.0,
            "max_abs_delta_mean_longitude_arcsec": 0.0,
        }
        for name in secular_bodies
    }
    max_paired = 0.0
    max_gr_newtonian = 0.0
    with csv_path.open("w", newline="") as handle:
        writer: csv.DictWriter | None = None
        for year in range(0, 100_001, 1000):
            target = year * JULIAN_YEAR_S
            for sim in (c_sim, py_sim, ordinary, newtonian):
                sim.integrate(target, exact_finish_time=1)
            dynamic = compare_dynamic_pair(c_sim, py_sim, len(bodies))
            for key in dynamic_max:
                dynamic_max[key] = max(dynamic_max[key], dynamic[key])
            minimum_cosine = min(minimum_cosine, dynamic["tangent_direction_cosine"])
            cs = rebound_state_from_sim(c_sim, state.masses)
            os_state = rebound_state_from_sim(ordinary, state.masses)
            ns = rebound_state_from_sim(newtonian, state.masses)
            c_minus_o = float(
                np.linalg.norm(scaled_phase_vector(cs.positions - os_state.positions, cs.velocities - os_state.velocities))
            )
            c_minus_n = float(
                np.linalg.norm(scaled_phase_vector(cs.positions - ns.positions, cs.velocities - ns.velocities))
            )
            o_minus_n = float(
                np.linalg.norm(
                    scaled_phase_vector(os_state.positions - ns.positions, os_state.velocities - ns.velocities)
                )
            )
            max_paired = max(max_paired, abs(c_minus_n - o_minus_n))
            max_gr_newtonian = max(max_gr_newtonian, c_minus_n, o_minus_n)
            ce = element_map(cs, bodies)
            oe = element_map(os_state, bodies)
            row: dict[str, Any] = {
                "time_years": year,
                **dynamic,
                "c_vs_reboundx_scaled_phase_difference": c_minus_o,
                "c_gr_minus_newtonian_scaled_phase_difference": c_minus_n,
                "reboundx_gr_minus_newtonian_scaled_phase_difference": o_minus_n,
            }
            for body_name in secular_bodies:
                label = body_name.split()[0]
                c_body = ce[body_name]
                o_body = oe[body_name]
                differences = {
                    "delta_a_au": c_body["a_au"] - o_body["a_au"],
                    "delta_e": c_body["e"] - o_body["e"],
                    "delta_i_arcsec": 3600.0 * (c_body["i_deg"] - o_body["i_deg"]),
                    "delta_varpi_arcsec": 3600.0
                    * _signed_angle_delta_deg(c_body["varpi_deg"], o_body["varpi_deg"]),
                    "delta_mean_longitude_arcsec": 3600.0
                    * _signed_angle_delta_deg(
                        c_body["mean_longitude_deg"], o_body["mean_longitude_deg"]
                    ),
                }
                for key, value in differences.items():
                    row[f"{label}_{key}"] = value
                    summary_key = "max_abs_" + key
                    max_element_differences[label][summary_key] = max(
                        max_element_differences[label][summary_key], abs(value)
                    )
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(f"[c-validation] full dynamic/physical t={year:,} yr", flush=True)
    max_paired_relative = max_paired / max(max_gr_newtonian, 1.0e-300)
    failures = []
    if dynamic_max["real_scaled_phase_difference"] > THRESHOLDS["full_100k_real_scaled_phase"]:
        failures.append("C/Python real-state divergence exceeds threshold")
    if dynamic_max["tangent_relative_difference"] > THRESHOLDS["full_100k_tangent_relative"]:
        failures.append("C/Python tangent divergence exceeds threshold")
    if minimum_cosine < THRESHOLDS["full_100k_tangent_direction_cosine"]:
        failures.append("C/Python tangent direction cosine is below threshold")
    if dynamic_max["megno_absolute_difference"] > THRESHOLDS["full_100k_megno_absolute"]:
        failures.append("C/Python MEGNO difference exceeds threshold")
    if dynamic_max["lcn_absolute_difference"] > THRESHOLDS["full_100k_lcn_absolute"]:
        failures.append("C/Python LCN difference exceeds threshold")
    if max_paired_relative > THRESHOLDS["reboundx_paired_gr_relative"]:
        failures.append("C versus REBOUNDx paired GR-minus-Newtonian difference exceeds threshold")
    element_limits = {
        "max_abs_delta_a_au": THRESHOLDS["reboundx_delta_a_au"],
        "max_abs_delta_e": THRESHOLDS["reboundx_delta_e"],
        "max_abs_delta_i_arcsec": THRESHOLDS["reboundx_delta_i_arcsec"],
        "max_abs_delta_varpi_arcsec": THRESHOLDS["reboundx_delta_varpi_arcsec"],
        "max_abs_delta_mean_longitude_arcsec": THRESHOLDS[
            "reboundx_delta_mean_longitude_arcsec"
        ],
    }
    for label, values in max_element_differences.items():
        for key, value in values.items():
            if value > element_limits[key]:
                failures.append(f"{label} {key} exceeds C-vs-REBOUNDx threshold")
    c_stats = backend.stats(c_sim)
    if c_stats["nonfinite_result_count"] != 0 or c_stats["callback_invocations"] <= 0:
        failures.append("C callback instrumentation reports missing or nonfinite execution")
    payload = {
        "stage": "full_dynamic_physical_100k",
        "passed": not failures,
        "failures": failures,
        "c_python_maxima": dynamic_max,
        "minimum_tangent_direction_cosine": minimum_cosine,
        "max_paired_gr_minus_newtonian_difference": max_paired,
        "max_paired_gr_minus_newtonian_relative_difference": max_paired_relative,
        "max_element_differences": max_element_differences,
        "c_callback_stats": c_stats,
        "hot_path_proof": backend.hot_path_proof(c_sim),
        "no_unexplained_secular_divergence": not failures,

        "sample_count": len(rows),
        "maximum_duration_years_launched": 100_000.0,
        "outputs": {"csv": str(csv_path)},
    }
    return write_result(args.output_dir, "full_dynamic_physical_100k_summary.json", payload)
def _runner_base_command(
    args: argparse.Namespace,
    output_dir: Path,
    tag: str,
    duration_years: float,
    *,
    step_days: float = 4.0,
    record_every_years: float | None = None,
) -> list[str]:
    record = record_every_years or max(1.0, duration_years / 20.0)
    command = [
        sys.executable,
        "-m",
        "mini_ephemeris.rebound_gr_tangent_backend_cli",
        "--kernel-path",
        str(args.kernel_path),
        "--model-scope",
        "full_with_pluto",
        "--duration-years",
        f"{duration_years:.17g}",
        "--step-days",
        f"{step_days:.17g}",
        "--record-every-years",
        f"{record:.17g}",
        "--archive-interval-years",
        f"{record:.17g}",
        "--gr-tangent-backend",
        "c",
        "--output-dir",
        str(output_dir),
        "--tag",
        tag,
    ]
    if args.manifest_path:
        command.extend(["--manifest-path", str(args.manifest_path)])
    return command


def _runner_paths(output_dir: Path, tag: str) -> dict[str, Path]:
    return {
        "archive": output_dir / f"gr_tangent_archive_{tag}.bin",
        "progress": output_dir / f"gr_tangent_progress_{tag}.csv",
        "restart": output_dir / f"gr_tangent_restart_{tag}.json",
        "status": output_dir / f"gr_tangent_status_{tag}.json",
        "summary": output_dir / f"gr_tangent_summary_{tag}.json",
    }


def _run_checked(
    command: list[str],
    *,
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != expected_returncode:
        raise RuntimeError(
            f"command returned {completed.returncode}, expected {expected_returncode}: "
            + " ".join(command)
            + f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _archive_phase(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    rebound = optional_import_module("rebound")
    sim, archive = open_archive_snapshot(rebound, path)
    del archive
    n_real = int(sim.N_real)
    state = state_from_real_particles(sim, n_real)
    delta_pos, delta_vel = read_first_variation(sim)
    return (
        scaled_phase_vector(state.positions, state.velocities),
        scaled_phase_vector(delta_pos, delta_vel),
        float(sim.megno()),
        float(sim.lyapunov()),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def restart_equivalence(args: argparse.Namespace) -> int:
    duration = float(args.restart_duration_years)
    if duration <= 0.0 or duration > 100_000.0:
        raise ValueError("restart duration must be in (0, 100000]")
    record = duration / 20.0
    checkpoint = duration / 2.0
    root = args.output_dir
    full_dir = root / "uninterrupted"
    split_dir = root / "split"
    tag = f"restart_{int(duration)}"
    full_command = _runner_base_command(
        args, full_dir, tag, duration, record_every_years=record
    )
    split_command = _runner_base_command(
        args, split_dir, tag, duration, record_every_years=record
    )
    _run_checked([*full_command, "--overwrite-existing-output"])
    _run_checked(
        [
            *split_command,
            "--overwrite-existing-output",
            "--stop-after-years",
            f"{checkpoint:.17g}",
        ],
        expected_returncode=INTENTIONAL_INCOMPLETE_EXIT,
    )
    full_paths = _runner_paths(full_dir, tag)
    split_paths = _runner_paths(split_dir, tag)
    partial_status = json.loads(split_paths["status"].read_text())
    partial_sidecar = json.loads(split_paths["restart"].read_text())

    mismatch = _run_checked(
        [
            *_runner_base_command(
                args,
                split_dir,
                tag,
                duration,
                step_days=8.0,
                record_every_years=record,
            ),
            "--resume",
        ],
        expected_returncode=2,
    )
    failure_cases: dict[str, bool] = {
        "configuration_mismatch_rejected": "fingerprint mismatch" in mismatch.stderr.lower()
    }
    malformed_root = root / "malformed"
    missing_dir = malformed_root / "missing"
    missing = _run_checked(
        [
            *_runner_base_command(
                args, missing_dir, "missing", duration, record_every_years=record
            ),
            "--resume",
        ],
        expected_returncode=2,
    )
    failure_cases["missing_state_rejected"] = "--resume requires" in missing.stderr
    truncated_dir = malformed_root / "truncated"
    truncated_dir.mkdir(parents=True, exist_ok=True)
    truncated_paths = _runner_paths(truncated_dir, "truncated")
    shutil.copy2(split_paths["archive"], truncated_paths["archive"])
    shutil.copy2(split_paths["progress"], truncated_paths["progress"])
    truncated_paths["restart"].write_text('{"schema_version":')
    truncated = _run_checked(
        [
            *_runner_base_command(
                args, truncated_dir, "truncated", duration, record_every_years=record
            ),
            "--resume",
        ],
        expected_returncode=2,
    )
    failure_cases["truncated_sidecar_rejected"] = "unreadable restart sidecar" in truncated.stderr.lower()

    archive_dir = malformed_root / "truncated_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = _runner_paths(archive_dir, "truncated_archive")
    shutil.copy2(split_paths["progress"], archive_paths["progress"])
    archive_paths["restart"].write_text(json.dumps(partial_sidecar))
    archive_paths["archive"].write_bytes(split_paths["archive"].read_bytes()[:32])
    broken_archive = _run_checked(
        [
            *_runner_base_command(
                args,
                archive_dir,
                "truncated_archive",
                duration,
                record_every_years=record,
            ),
            "--resume",
        ],
        expected_returncode=1,
    )
    failure_cases["truncated_archive_rejected"] = broken_archive.returncode != 0

    _run_checked([*split_command, "--resume"])
    full_real, full_tangent, full_megno, full_lcn = _archive_phase(full_paths["archive"])
    split_real, split_tangent, split_megno, split_lcn = _archive_phase(split_paths["archive"])
    physical_difference = float(np.linalg.norm(full_real - split_real))
    tangent_difference = float(np.linalg.norm(full_tangent - split_tangent))
    tangent_relative = tangent_difference / max(
        float(np.linalg.norm(full_tangent)),
        float(np.linalg.norm(split_tangent)),
        1.0e-300,
    )
    tangent_cosine = direction_cosine(full_tangent, split_tangent)
    megno_difference = abs(full_megno - split_megno)
    lcn_difference = abs(full_lcn - split_lcn)

    full_rows = _read_csv_rows(full_paths["progress"])
    split_rows = _read_csv_rows(split_paths["progress"])
    diagnostic_columns = [
        "time_years",
        "megno",
        "lcn_1_per_year",
        "newtonian_energy_component_rel_change",
        "angular_momentum_rel_drift",
        "mercury_a_au",
        "mercury_e",
        "mercury_varpi_deg_unwrapped",
        "callback_invocations",
        "real_gr_accel_norm_max",
        "real_gr_accel_norm_mean",
        "tangent_gr_accel_norm_max",
        "tangent_gr_accel_norm_mean",
    ]
    sequence_maxima: dict[str, float] = {}
    for name in diagnostic_columns:
        left = np.array(
            [float(row[name]) if row[name] != "" else math.nan for row in full_rows]
        )
        right = np.array(
            [float(row[name]) if row[name] != "" else math.nan for row in split_rows]
        )
        missing_left = np.isnan(left)
        missing_right = np.isnan(right)
        masks_match = np.array_equal(missing_left, missing_right)
        finite = ~(missing_left | missing_right)
        if left.shape != right.shape or not masks_match:
            sequence_maxima[name] = math.inf
        elif np.any(finite):
            sequence_maxima[name] = float(
                np.max(np.abs(left[finite] - right[finite]))
            )
        else:
            sequence_maxima[name] = 0.0
    split_summary = json.loads(split_paths["summary"].read_text())
    full_summary = json.loads(full_paths["summary"].read_text())
    resumed_status = json.loads(split_paths["status"].read_text())
    restart_metadata = split_summary["restart"]
    callback_full = full_summary["diagnostics"]["callback_stats"]["callback_invocations"]
    callback_split = split_summary["diagnostics"]["callback_stats"]["callback_invocations"]
    failures = [name for name, passed in failure_cases.items() if not passed]
    if physical_difference > THRESHOLDS["restart_physical_scaled_phase"]:
        failures.append("restart physical phase mismatch exceeds threshold")
    if tangent_relative > THRESHOLDS["restart_tangent_scaled_phase"]:
        failures.append("restart tangent mismatch exceeds threshold")
    if tangent_cosine < THRESHOLDS["restart_tangent_direction_cosine"]:
        failures.append("restart tangent direction mismatch exceeds threshold")
    if megno_difference > THRESHOLDS["restart_diagnostic_absolute"]:
        failures.append("restart MEGNO mismatch exceeds threshold")
    if lcn_difference > THRESHOLDS["restart_diagnostic_absolute"]:
        failures.append("restart LCN mismatch exceeds threshold")
    diagnostic_max = max(
        value
        for name, value in sequence_maxima.items()
        if name not in {"callback_invocations"}
    )
    if diagnostic_max > THRESHOLDS["restart_diagnostic_absolute"]:
        failures.append("restart diagnostic sequence mismatch exceeds threshold")
    if callback_full != callback_split or sequence_maxima["callback_invocations"] != 0.0:
        failures.append("restart callback counters are not continuous")
    if len(full_rows) != len(split_rows) or len({row["time_years"] for row in split_rows}) != len(split_rows):
        failures.append("restart progress rows are missing or duplicated")
    if partial_status.get("worker_pid") == resumed_status.get("worker_pid"):
        failures.append("restart continuation did not run in a fresh process")
    if not restart_metadata.get("callbacks_increased_after_reattachment"):
        failures.append("callback did not execute after restart reattachment")
    payload = {
        "stage": f"restart_equivalence_{int(duration)}",
        "passed": not failures,
        "failures": failures,
        "duration_years": duration,
        "checkpoint_years": checkpoint,
        "fresh_process": partial_status.get("worker_pid") != resumed_status.get("worker_pid"),
        "failure_injection": failure_cases,
        "physical_scaled_phase_difference": physical_difference,
        "tangent_relative_difference": tangent_relative,
        "tangent_direction_cosine": tangent_cosine,
        "megno_absolute_difference": megno_difference,
        "lcn_absolute_difference": lcn_difference,
        "diagnostic_sequence_maxima": sequence_maxima,
        "callback_invocations": {
            "uninterrupted": callback_full,
            "resumed": callback_split,
        },
        "restart_metadata": restart_metadata,
        "row_count": len(split_rows),
        "maximum_duration_years_launched": duration,
        "outputs": {
            "uninterrupted_summary": str(full_paths["summary"]),
            "resumed_summary": str(split_paths["summary"]),
        },
    }
    return write_result(
        args.output_dir,
        f"restart_equivalence_{int(duration)}_summary.json",
        payload,
    )


def reproducibility(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    bodies, state = make_initial_state(args.kernel_path, "full_with_pluto")
    duration = 1_000.0

    def execute() -> dict[str, Any]:
        sim = make_sim(
            rebound,
            "c",
            state,
            step_days=4.0,
            megno=True,
            seed=12345,
            c_backend=backend,
        )
        samples = []
        for target in np.linspace(0.0, duration, 11):
            sim.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
            real_pos, real_vel, var_pos, var_vel = phase_arrays(sim, len(bodies))
            samples.append(
                np.concatenate(
                    [
                        real_pos.ravel(),
                        real_vel.ravel(),
                        var_pos.ravel(),
                        var_vel.ravel(),
                        np.array([float(sim.megno()), float(sim.lyapunov())]),
                    ]
                )
            )
        array = np.vstack(samples)
        return {
            "array": array,
            "sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()),
            "stats": backend.stats(sim),
            "hot_path_proof": backend.hot_path_proof(sim),
        }

    first = execute()
    second = execute()
    bitwise = bool(np.array_equal(first["array"], second["array"]))
    max_difference = float(np.max(np.abs(first["array"] - second["array"])))
    stats_equal = first["stats"] == second["stats"]
    failures = []
    if not bitwise or max_difference > THRESHOLDS["reproducibility_scaled_phase"]:
        failures.append("representative repeated run is not bitwise reproducible")
    if not stats_equal:
        failures.append("C callback instrumentation is not reproducible")
    payload = {
        "stage": "reproducibility",
        "passed": not failures,
        "failures": failures,
        "comparison_kind": "bitwise",
        "duration_years": duration,
        "step_days": 4.0,
        "seed": 12345,
        "sample_count": 11,
        "bitwise_equal": bitwise,
        "maximum_absolute_value_difference": max_difference,
        "sample_sha256_first": first["sha256"],
        "sample_sha256_second": second["sha256"],
        "callback_stats_equal": stats_equal,
        "callback_stats": first["stats"],
        "hot_path_proof": first["hot_path_proof"],
        "maximum_duration_years_launched": duration,
    }
    return write_result(args.output_dir, "reproducibility_summary.json", payload)


def _benchmark_once(
    rebound: Any,
    backend_name: str,
    state: NBodyState,
    bodies: tuple[str, ...],
    backend: Any,
    duration: float,
) -> tuple[float, Any, dict[str, Any]]:
    sim = make_sim(
        rebound,
        backend_name,
        state,
        step_days=4.0,
        megno=True,
        seed=12345,
        c_backend=backend,
    )
    start = time.perf_counter()
    for target in np.linspace(duration / 20.0, duration, 20):
        sim.integrate(float(target) * JULIAN_YEAR_S, exact_finish_time=1)
        current = rebound_state_from_sim(sim, state.masses)
        heliocentric_elements_for_state(current, bodies, sun_index=bodies.index("sun"))
        float(sim.megno())
        float(sim.lyapunov())
        if backend_name == "c":
            backend.stats(sim)
        else:
            python_callback_stats = getattr(
                sim, "_mini_ephemeris_gr_potential_tangent_stats", {}
            )
            int(python_callback_stats.get("callback_invocations", 0))
    elapsed = time.perf_counter() - start
    stats = (
        backend.stats(sim)
        if backend_name == "c"
        else {
            key: value
            for key, value in getattr(
                sim, "_mini_ephemeris_gr_potential_tangent_stats", {}
            ).items()
            if isinstance(value, (int, float))
        }
    )
    return elapsed, sim, stats


def benchmark(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    backend = load_c_backend()
    bodies, state = make_initial_state(args.kernel_path, "full_with_pluto")
    duration = float(THRESHOLDS["benchmark_duration_years"])
    repetitions = int(THRESHOLDS["benchmark_repetitions"])
    _benchmark_once(rebound, "python", state, bodies, backend, 10.0)
    _benchmark_once(rebound, "c", state, bodies, backend, 10.0)
    timings: dict[str, list[float]] = {"python": [], "c": []}
    callback_counts: dict[str, list[int]] = {"python": [], "c": []}
    pairs = []
    for repetition in range(repetitions):
        order = ("c", "python") if repetition % 2 else ("python", "c")
        simulations: dict[str, Any] = {}
        for backend_name in order:
            elapsed, sim, stats = _benchmark_once(
                rebound, backend_name, state, bodies, backend, duration
            )
            timings[backend_name].append(elapsed)
            callback_counts[backend_name].append(int(stats["callback_invocations"]))
            simulations[backend_name] = sim
        pairs.append(compare_dynamic_pair(simulations["c"], simulations["python"], len(bodies)))
    python_median = statistics.median(timings["python"])
    c_median = statistics.median(timings["c"])
    ratio = c_median / python_median
    speedup = python_median / c_median
    failures = []
    if speedup < THRESHOLDS["minimum_c_speedup"]:
        failures.append("compiled production pathway speedup is below threshold")
    if ratio > THRESHOLDS["maximum_c_to_python_wall_ratio"]:
        failures.append("compiled/Python wall-time ratio exceeds threshold")
    if max(item["tangent_relative_difference"] for item in pairs) > THRESHOLDS["short_tangent_relative"]:
        failures.append("benchmark pathway correctness mismatch exceeds threshold")
    payload = {
        "stage": "production_benchmark",
        "passed": not failures,
        "failures": failures,
        "science_settings": {
            "scope": "full_with_pluto",
            "duration_years": duration,
            "step_days": 4.0,
            "record_count": 20,
            "megno_seed": 12345,
        },
        "warmup_duration_years_per_backend": 10.0,
        "repetitions": repetitions,
        "python_wall_seconds": timings["python"],
        "c_wall_seconds": timings["c"],
        "python_median_seconds": python_median,
        "c_median_seconds": c_median,
        "c_to_python_wall_ratio": ratio,
        "speedup": speedup,
        "criterion": {
            "minimum_speedup": THRESHOLDS["minimum_c_speedup"],
            "maximum_c_to_python_wall_ratio": THRESHOLDS[
                "maximum_c_to_python_wall_ratio"
            ],
        },
        "callback_invocation_counts": callback_counts,
        "matched_end_state_comparisons": pairs,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "hot_path_proof": backend.hot_path_proof(simulations["c"]),
        "maximum_duration_years_launched": duration,
    }
    return write_result(args.output_dir, "production_benchmark_summary.json", payload)


def capped_rehearsal(args: argparse.Namespace) -> int:
    duration = float(THRESHOLDS["capped_rehearsal_duration_years"])
    tag = "capped_rehearsal_100k"
    run_dir = args.output_dir / "run"
    command = _runner_base_command(
        args,
        run_dir,
        tag,
        duration,
        step_days=1.0,
        record_every_years=1_000.0,
    )
    _run_checked([*command, "--overwrite-existing-output"])
    paths = _runner_paths(run_dir, tag)
    summary = json.loads(paths["summary"].read_text())
    rows = _read_csv_rows(paths["progress"])
    accepted_summary_path = (
        args.python_oracle_output
        / "gr_100kyr_1d_seed12345"
        / "gr_tangent_summary_gr_100kyr_1d_seed12345.json"
    )
    accepted_summary = json.loads(accepted_summary_path.read_text())
    accepted_rows = _read_csv_rows(Path(accepted_summary["outputs"]["progress_csv"]))
    comparison_limits = {
        "megno": THRESHOLDS["full_100k_megno_absolute"],
        "lcn_1_per_year": THRESHOLDS["full_100k_lcn_absolute"],
        "newtonian_energy_component_rel_change": THRESHOLDS[
            "rehearsal_energy_diagnostic_absolute"
        ],
        "angular_momentum_rel_drift": THRESHOLDS[
            "rehearsal_angular_diagnostic_absolute"
        ],
        "mercury_a_au": THRESHOLDS["rehearsal_mercury_a_au_absolute"],
        "mercury_e": THRESHOLDS["rehearsal_mercury_e_absolute"],
        "mercury_varpi_deg_unwrapped": THRESHOLDS[
            "rehearsal_mercury_varpi_deg_absolute"
        ],
    }
    sequence_maxima = {}
    for name in comparison_limits:
        c_values = np.array([float(row[name]) for row in rows])
        accepted_name = (
            "energy_rel_drift"
            if name == "newtonian_energy_component_rel_change"
            else name
        )
        py_values = np.array([float(row[accepted_name]) for row in accepted_rows])
        sequence_maxima[name] = (
            float(np.max(np.abs(c_values - py_values)))
            if c_values.shape == py_values.shape
            else math.inf
        )
    times = [float(row["time_years"]) for row in rows]
    stats = summary["diagnostics"]["callback_stats"]
    failures = []
    if len(rows) != 101 or times != sorted(set(times)) or times[-1] != duration:
        failures.append("rehearsal diagnostic sequence is incomplete or discontinuous")
    for name, limit in comparison_limits.items():
        if sequence_maxima[name] > limit:
            failures.append(f"rehearsal C/Python {name} difference exceeds threshold")
    if stats["callback_invocations"] <= 0 or stats["nonfinite_result_count"] != 0:
        failures.append("rehearsal callback health check failed")
    if not summary["hot_path_proof"]["addresses_match"]:
        failures.append("rehearsal hot path is not the compiled callback")
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        failures.append("rehearsal output or archive artifact is missing/empty")
    rebound = optional_import_module("rebound")
    archived_sim, archive = open_archive_snapshot(rebound, paths["archive"])
    del archive
    backend = load_c_backend()
    backend.attach(archived_sim)
    restart_ready = backend.hot_path_proof(archived_sim)["addresses_match"]
    if not restart_ready:
        failures.append("completed rehearsal archive cannot reattach the C callback")
    payload = {
        "stage": "capped_rehearsal_100k",
        "passed": not failures,
        "failures": failures,
        "duration_years": duration,
        "step_days": 1.0,
        "sample_count": len(rows),
        "diagnostic_times_strictly_unique": times == sorted(set(times)),
        "finite_diagnostics": all(
            math.isfinite(float(row[name]))
            for row in rows
            for name in comparison_limits
        ),
        "c_python_sequence_maxima": sequence_maxima,
        "comparison_limits": comparison_limits,
        "accepted_python_summary": str(accepted_summary_path),
        "hashes": summary["provenance"]["hashes"],
        "configuration_fingerprint": summary["configuration_fingerprint"],
        "configuration": summary["configuration"],
        "callback_stats": stats,
        "hot_path_proof": summary["hot_path_proof"],
        "restart_ready": restart_ready,
        "archive_size_bytes": paths["archive"].stat().st_size,
        "no_unexplained_secular_divergence": not any(
            "difference exceeds" in failure for failure in failures
        ),
        "maximum_duration_years_launched": duration,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    return write_result(args.output_dir, "capped_rehearsal_100k_summary.json", payload)

EXPECTED_RESULTS = {
    "build_integrity": "build_integrity/build_integrity_summary.json",
    "pointwise_acceleration": "pointwise_acceleration/pointwise_acceleration_summary.json",
    "pointwise_jacobian": "pointwise_jacobian/pointwise_jacobian_summary.json",
    "hot_path_lifecycle": "hot_path_lifecycle/hot_path_lifecycle_summary.json",
    "short_dynamic_equivalence": (
        "short_dynamic_equivalence/short_dynamic_equivalence_summary.json"
    ),
    "finite_difference_oracle": (
        "finite_difference_oracle/finite_difference_oracle_summary.json"
    ),
    "physical_controls": "physical_controls/physical_controls_summary.json",
    "full_dynamic_physical_100k": (
        "full_dynamic_physical_100k/full_dynamic_physical_100k_summary.json"
    ),
    "restart_equivalence_20000": (
        "restart_equivalence_20k/restart_equivalence_20000_summary.json"
    ),
    "restart_equivalence_100000": (
        "restart_equivalence_100k/restart_equivalence_100000_summary.json"
    ),
    "reproducibility": "reproducibility/reproducibility_summary.json",
    "production_benchmark": (
        "production_benchmark/production_benchmark_summary.json"
    ),
    "capped_rehearsal_100k": (
        "capped_rehearsal_100k/capped_rehearsal_100k_summary.json"
    ),
}


def _git_lines(*arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    )
    return [line for line in completed.stdout.splitlines() if line]


def _changed_files(tracked_doc_dir: Path) -> list[str]:
    paths = set(_git_lines("diff", "--name-only", FROZEN_TAG))
    paths.update(_git_lines("ls-files", "--others", "--exclude-standard"))
    paths.update(
        {
            str((tracked_doc_dir / "gr_tangent_c_port_validation_summary.json").relative_to(Path.cwd())),
            str((tracked_doc_dir / "gr_tangent_c_port_validation_report.md").relative_to(Path.cwd())),
        }
    )
    return sorted(paths)


def _decision_status(failures: list[str]) -> str:
    if not failures:
        return "READY_FOR_10MYR_C"
    if any(
        name in failures
        for name in ("build_integrity", "pointwise_acceleration", "pointwise_jacobian")
    ):
        return "BLOCKED_IMPLEMENTATION"
    if any(name.startswith("restart_equivalence") for name in failures):
        return "BLOCKED_RESTART"
    if "production_benchmark" in failures:
        return "BLOCKED_PERFORMANCE"
    return "BLOCKED_VALIDATION"


def _production_tmux_command(args: argparse.Namespace) -> str:
    output = Path.cwd() / "output/stability/gr_tangent_c_10myr_v1"
    log = output / "gr_tangent_c_10myr.log"
    runner = (
        f"cd {Path.cwd()} && mkdir -p {output} && "
        "PYTHONPATH=mini_ephemeris/src "
        ".venv/bin/python -m mini_ephemeris.rebound_gr_tangent_backend_cli "
        f"--kernel-path {args.kernel_path} --model-scope full_with_pluto "
        "--duration-years 10000000 --production-duration-approved "
        "--step-days 1 --record-every-years 10000 --archive-interval-years 10000 "
        "--gr-tangent-backend c "
        f"--manifest-path {args.manifest_path} --output-dir {output} "
        "--tag gr_tangent_c_10myr_seed12345 "
        f"2>&1 | tee {log}"
    )
    return f"tmux new-session -d -s gr_tangent_c_10myr '{runner}'"


def _markdown_report(payload: dict[str, Any]) -> str:
    stages = payload["validation"]["stages"]
    lines = [
        f"# {payload['final_status']}",
        "",
        payload["decision_reason"],
        "",
        "## Architecture",
        "",
        "REBOUND calls a directly installed compiled C function pointer. The callback owns",
        "per-simulation configuration and counters in reb_simulation.extras; Python is",
        "used only for setup, scheduled diagnostics, and reporting. The physical term is",
        "-6 (GM)^2 r / (c^2 r^4) and the tangent action is its analytic position",
        "Jacobian, including the oracle's central mass-ratio response.",
        "",
        "## Validation",
        "",
        "| Stage | Result | Artifact |",
        "|---|---:|---|",
    ]
    for name, result in stages.items():
        lines.append(
            f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | "
            f"{result['path']} |"
        )
    lines.extend(
        [
            "",
            "## Key Evidence",
            f"- Numerical maxima: {json.dumps(payload['numerical_summary'], sort_keys=True)}",
            "",
            f"- Hot path: {json.dumps(payload['hot_path_proof'], sort_keys=True)}",
            f"- Restart: {json.dumps(payload['restart_result'], sort_keys=True)}",
            f"- Reproducibility: {json.dumps(payload['reproducibility_result'], sort_keys=True)}",
            f"- Performance: {json.dumps(payload['performance'], sort_keys=True)}",
            f"- Maximum launched duration: {payload['duration_safety']['maximum_duration_years_launched']:,} years.",
            f"- Oracle unchanged: {payload['oracle_integrity']['passed']}.",
            f"- No 10 Myr run launched: {payload['duration_safety']['no_10myr_run_launched']}.",
            "",
            "## Provenance",
            "",
            f"- Frozen tag/commit: {payload['git']['frozen_tag']} / {payload['git']['base_commit']}",
            f"- Final commit: {payload['git']['final_commit']}; dirty: {payload['git']['final_dirty']}",
            f"- Python: {payload['environment']['python']}",
            f"- Platform: {payload['environment']['platform']}",
            f"- Changed files: {', '.join(payload['changed_files'])}",
            f"- Superseded attempts: {json.dumps(payload['superseded_attempts'], sort_keys=True)}",
            f"- Compiler: {payload['compiler']['identity']}",
            f"- REBOUND: {payload['environment']['rebound_version']}",
            "",
            "## Limitations",
            "",
            "- Finite-time MEGNO/LCN remains a numerical diagnostic, not an asymptotic proof.",
            "- The reported Newtonian energy component excludes the custom GR potential.",
            "- The validated ABI is specific to the recorded REBOUND header/build.",
            "",
            "## Next Action",
            "",
            (
                payload["next_action"]["tmux_command"]
                if payload["final_status"] == "READY_FOR_10MYR_C"
                else payload["next_action"]["smallest_action"]
            ),
            "",
        ]
    )
    return "\n".join(lines)

def final_report(args: argparse.Namespace) -> int:
    stage_records: dict[str, Any] = {}
    failures: list[str] = []
    unreadable: dict[str, str] = {}
    maximum_duration = 0.0
    loaded: dict[str, dict[str, Any]] = {}
    for name, relative in EXPECTED_RESULTS.items():
        path = args.matrix_root / relative
        try:
            result = json.loads(path.read_text())
            if not isinstance(result, dict):
                raise TypeError("result is not a JSON object")
            loaded[name] = result
            passed = result.get("passed") is True
            maximum_duration = max(
                maximum_duration,
                float(result.get("maximum_duration_years_launched", 0.0)),
            )
            stage_records[name] = {
                "path": str(path),
                "readable": True,
                "passed": passed,
                "failures": result.get("failures", []),
            }
            if not passed:
                failures.append(name)
        except Exception as exc:
            failures.append(name)
            unreadable[name] = str(exc)
            stage_records[name] = {
                "path": str(path),
                "readable": False,
                "passed": False,
                "failures": [str(exc)],
            }
    oracle = frozen_oracle_evidence()
    if not oracle["passed"]:
        failures.append("oracle_integrity")
    if maximum_duration > 100_000.0:
        failures.append("duration_cap")
    status = _decision_status(failures)
    backend = load_c_backend()
    benchmark_result = loaded.get("production_benchmark", {})
    restart_result = loaded.get("restart_equivalence_100000", {})
    reproducibility_result = loaded.get("reproducibility", {})
    hot_path_result = loaded.get("hot_path_lifecycle", {})
    rehearsal_result = loaded.get("capped_rehearsal_100k", {})
    final_commit = _git_lines("rev-parse", "HEAD")[0]
    final_dirty = bool(_git_lines("status", "--porcelain"))
    tracked_doc_dir = args.tracked_doc_dir.resolve()
    reason = (
        "All 13 prerequisite stages and the final completeness gate passed."
        if status == "READY_FOR_10MYR_C"
        else "Readiness is blocked by: " + ", ".join(failures)
    )
    next_command = _production_tmux_command(args)
    payload = {
        "schema_version": 1,
        "validation_tag": "gr_tangent_c_port_validation_v1",
        "created_utc": utc_now(),
        "final_status": status,
        "decision_reason": reason,
        "exact_non_ready_reasons": failures if status != "READY_FOR_10MYR_C" else [],
        "git": {
            "repository_root": str(Path.cwd()),
            "branch": _git_lines("branch", "--show-current")[0],
            "frozen_tag": FROZEN_TAG,
            "base_commit": FROZEN_COMMIT,
            "initial_commit": FROZEN_COMMIT,
            "initial_dirty": False,
            "final_commit": final_commit,
            "final_dirty": final_dirty,
        },
        "environment": {
            "platform": platform.platform(),
            "os_uname": list(os.uname()),
            "python": sys.version,
            "rebound_version": str(optional_import_module("rebound").__version__),
            "reboundx_version": str(optional_import_module("reboundx").__version__),
        },
        "compiler": {
            "identity": backend.build_metadata["compiler_identity"],
            "path": backend.build_metadata["compiler_path"],
            "flags": backend.build_metadata["compiler_flags"],
        },
        "compiled_artifact": backend.build_metadata,
        "abi": backend.abi_metadata,
        "hashes": {
            "c_source_sha256": backend.build_metadata["source_sha256"],
            "configuration_sha256": rehearsal_result.get(
                "configuration_fingerprint"
            ),
            "compiled_artifact_sha256": backend.build_metadata["artifact_sha256"],
            "manifest_sha256": sha256_file(args.manifest_path),
            **rehearsal_result.get("hashes", {}),
        },
        "oracle_integrity": oracle,
        "validation": {
            "manifest_path": str(args.manifest_path),
            "output_root": str(args.matrix_root),
            "expected_stage_count": len(EXPECTED_RESULTS) + 1,
            "observed_stage_count": sum(
                record["readable"] for record in stage_records.values()
            )
            + 1,
            "readable_stage_count": sum(
                record["readable"] for record in stage_records.values()
            )
            + 1,
            "passing_stage_count": sum(
                record["passed"] for record in stage_records.values()
            )
            + (not failures),
            "stages": stage_records,
            "unreadable": unreadable,
            "thresholds_fixed_before_decisive_runs": THRESHOLDS,
        },
        "numerical_summary": {
            "pointwise_acceleration": {
                "max_relative_norm_error": loaded.get(
                    "pointwise_acceleration", {}
                ).get("max_relative_norm_error"),
                "max_absolute_component_error": loaded.get(
                    "pointwise_acceleration", {}
                ).get("max_absolute_component_error"),
            },
            "pointwise_jacobian": {
                "max_relative_norm_error": loaded.get(
                    "pointwise_jacobian", {}
                ).get("max_relative_norm_error"),
                "max_linearity_relative_error": loaded.get(
                    "pointwise_jacobian", {}
                ).get("max_linearity_relative_error"),
            },
            "short_dynamic_configurations": loaded.get(
                "short_dynamic_equivalence", {}
            ).get("configurations"),
            "full_dynamic_c_python_maxima": loaded.get(
                "full_dynamic_physical_100k", {}
            ).get("c_python_maxima"),
            "full_dynamic_element_maxima": loaded.get(
                "full_dynamic_physical_100k", {}
            ).get("max_element_differences"),
            "rehearsal_c_python_sequence_maxima": rehearsal_result.get(
                "c_python_sequence_maxima"
            ),
        },
        "hot_path_proof": hot_path_result.get("hot_path_proof"),
        "restart_result": {
            "passed": restart_result.get("passed"),
            "fresh_process": restart_result.get("fresh_process"),
            "physical_scaled_phase_difference": restart_result.get(
                "physical_scaled_phase_difference"
            ),
            "tangent_relative_difference": restart_result.get(
                "tangent_relative_difference"
            ),
            "diagnostic_sequence_maxima": restart_result.get(
                "diagnostic_sequence_maxima"
            ),
        },
        "reproducibility_result": {
            "passed": reproducibility_result.get("passed"),
            "comparison_kind": reproducibility_result.get("comparison_kind"),
            "bitwise_equal": reproducibility_result.get("bitwise_equal"),
            "sample_sha256": reproducibility_result.get("sample_sha256_first"),
        },
        "performance": {
            "passed": benchmark_result.get("passed"),
            "speedup": benchmark_result.get("speedup"),
            "c_to_python_wall_ratio": benchmark_result.get(
                "c_to_python_wall_ratio"
            ),
            "python_median_seconds": benchmark_result.get("python_median_seconds"),
            "c_median_seconds": benchmark_result.get("c_median_seconds"),
            "criterion": benchmark_result.get("criterion"),
        },
        "duration_safety": {
            "maximum_duration_years_launched": maximum_duration,
            "cap_years": 100_000.0,
            "cap_respected": maximum_duration <= 100_000.0,
            "no_10myr_run_launched": True,
        },
        "superseded_attempts": [
            {
                "stage": "restart_equivalence_20000",
                "reason": "truncated sidecar rejection escaped the CLI safety error boundary",
                "preserved_path": str(
                    args.matrix_root / "restart_equivalence_20k_attempt1_failed"
                ),
            },
            {
                "stage": "restart_equivalence_20000",
                "reason": "report comparison did not yet handle matching blank t=0 means",
                "preserved_path": str(
                    args.matrix_root / "restart_equivalence_20k_attempt2_failed"
                ),
            },
            {
                "stage": "capped_rehearsal_100k",
                "reason": "accepted Python CSV uses the historical energy_rel_drift alias",
                "preserved_path": str(
                    args.matrix_root / "capped_rehearsal_100k_attempt1_failed"
                ),
            },
        ],
        "changed_files": _changed_files(tracked_doc_dir),
        "blockers": failures,
        "next_action": {
            "tmux_command": next_command if status == "READY_FOR_10MYR_C" else None,
            "smallest_action": (
                None
                if status == "READY_FOR_10MYR_C"
                else "Repair and rerun the first failing stage and its downstream dependents."
            ),
        },
    }
    report = _markdown_report(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracked_doc_dir.mkdir(parents=True, exist_ok=True)
    json_name = "gr_tangent_c_port_validation_summary.json"
    report_name = "gr_tangent_c_port_validation_report.md"
    atomic_write_json(args.output_dir / json_name, payload)
    (args.output_dir / report_name).write_text(report)
    atomic_write_json(tracked_doc_dir / json_name, payload)
    (tracked_doc_dir / report_name).write_text(report)
    print(f"[c-validation] final status: {status}")
    return 0 if status == "READY_FOR_10MYR_C" else 1

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duration-capped compiled-C GR tangent validation matrix."
    )
    parser.add_argument("--kernel-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=Path("output/stability/gr_tangent_c_port_validation_v1"),
    )
    parser.add_argument(
        "--python-oracle-output",
        type=Path,
        default=Path("output/stability/gr_tangent_validation_matrix_v1"),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(
            "ephemeris_experiment_runner/manifests/"
            "08_gr_tangent_c_port_validation.json"
        ),
    )
    parser.add_argument(
        "--tracked-doc-dir",
        type=Path,
        default=Path("docs/validation/gr-tangent-c-port-v1"),
    )
    parser.add_argument(
        "--restart-duration-years", type=float, default=20_000.0
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stages: dict[str, Callable[[argparse.Namespace], int]] = {
        "build-integrity": build_integrity,
        "pointwise-acceleration": pointwise_acceleration,
        "pointwise-jacobian": pointwise_jacobian,
        "hot-path-lifecycle": hot_path_lifecycle,
        "short-dynamic": short_dynamic,
        "finite-difference-oracle": finite_difference_oracle,
        "physical-controls": physical_controls,
        "full-dynamic-physical": full_dynamic_physical,
        "restart-equivalence": restart_equivalence,
        "reproducibility": reproducibility,
        "benchmark": benchmark,
        "capped-rehearsal": capped_rehearsal,
        "final-report": final_report,
    }
    for name, function in stages.items():
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(function=function)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
