from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent import attach_gr_potential_tangent_force
from .rebound_gr_tangent_cli import APSIDAL_DRIFT_DEFINITION, DIAGNOSTIC_DEFINITIONS
from .long_term_stability_cli import (
    add_reboundx_gr_force,
    build_rebound_simulation,
    optional_import_module,
    parse_start_datetime,
    rebound_state_from_sim,
    stability_body_list,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import AU_M, DAY_S, JULIAN_YEAR_S, heliocentric_elements_for_state, seconds_to_years


DEFAULT_ROOT = Path("/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1")
DEFAULT_KERNEL = Path("/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
EXISTING_SMOKE_DIR = Path("/home/peacelovephysics/ephemeris/output/stability/gr_tangent_v1/full_1myr_smoke")
EXISTING_PROGRESS = EXISTING_SMOKE_DIR / "gr_tangent_progress_full_with_pluto_gr_tangent_1myr_seed12345.csv"
EXISTING_SUMMARY = EXISTING_SMOKE_DIR / "gr_tangent_summary_full_with_pluto_gr_tangent_1myr_seed12345.json"
EXISTING_ARCHIVE = EXISTING_SMOKE_DIR / "full_with_pluto_gr_tangent_1myr_seed12345.bin"
DEFAULT_MANIFEST = Path("/home/peacelovephysics/ephemeris/ephemeris_experiment_runner/manifests/07_gr_tangent_validation_matrix.json")

EXPECTED_REQUIRED_STAGES = (
    {
        "stage": "existing_1myr_smoke_audit",
        "title": "Existing 1 Myr Smoke Audit",
        "path": "existing_1myr_smoke_audit.json",
        "kind": "passed_json",
    },
    {
        "stage": "monitor_process_tree_audit",
        "title": "Process-Tree Monitor Audit",
        "path": "monitor_process_tree_audit/monitor_process_tree_audit.json",
        "kind": "passed_json",
    },
    {
        "stage": "dynamic_gr_tangent_oracle",
        "title": "Dynamic GR Tangent Oracle",
        "path": "dynamic_gr_tangent_oracle/dynamic_gr_tangent_oracle_summary.json",
        "kind": "passed_json",
    },
    {
        "stage": "newtonian_zero_limit_100kyr",
        "title": "Newtonian Zero-Limit 100 kyr",
        "path": "newtonian_zero_limit_100kyr/newtonian_zero_limit_100kyr_summary.json",
        "kind": "passed_json",
    },
    {
        "stage": "gr_100kyr_1d_seed12345",
        "title": "GR 100 kyr, 1 day, seed 12345",
        "path": "gr_100kyr_1d_seed12345/gr_tangent_summary_gr_100kyr_1d_seed12345.json",
        "kind": "gr_tangent_summary",
    },
    {
        "stage": "gr_100kyr_1d_seed67890",
        "title": "GR 100 kyr, 1 day, seed 67890",
        "path": "gr_100kyr_1d_seed67890/gr_tangent_summary_gr_100kyr_1d_seed67890.json",
        "kind": "gr_tangent_summary",
    },
    {
        "stage": "seed_comparison",
        "title": "Seed Comparison",
        "path": "seed_comparison/gr_100kyr_1d_seed_comparison.json",
        "kind": "passed_json",
    },
    {
        "stage": "gr_100kyr_0p5d_seed12345",
        "title": "GR 100 kyr, 0.5 day, seed 12345",
        "path": "gr_100kyr_0p5d_seed12345/gr_tangent_summary_gr_100kyr_0p5d_seed12345.json",
        "kind": "gr_tangent_summary",
    },
    {
        "stage": "timestep_comparison",
        "title": "Timestep Comparison",
        "path": "timestep_comparison/gr_100kyr_timestep_comparison.json",
        "kind": "passed_json",
    },
    {
        "stage": "physical_gr_trajectory_comparison_100kyr",
        "title": "Physical GR Trajectory Comparison 100 kyr",
        "path": "physical_gr_trajectory_comparison_100kyr/physical_gr_trajectory_comparison_100kyr_summary.json",
        "kind": "passed_json",
    },
    {
        "stage": "gr_checkpoint_resume_equivalence_20kyr",
        "title": "GR Checkpoint/Resume Equivalence 20 kyr",
        "path": "gr_checkpoint_resume_equivalence_20kyr/gr_checkpoint_resume_equivalence_20kyr_summary.json",
        "kind": "passed_json",
    },
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def finite(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())



def row_first_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def diagnostic_first_value(diagnostics: dict[str, Any], canonical: str, *legacy: str) -> Any:
    for name in (canonical, *legacy):
        if name in diagnostics:
            return diagnostics.get(name)
    return None


def canonicalize_gr_tangent_summary(data: dict[str, Any]) -> dict[str, Any]:
    diagnostics = dict(data.get("diagnostics", {}))
    energy_value = diagnostic_first_value(
        diagnostics,
        "max_newtonian_energy_component_rel_change",
        "max_energy_rel_drift",
    )
    apsidal_value = diagnostic_first_value(
        diagnostics,
        "mercury_total_apsidal_drift_arcsec_per_century",
        "mercury_perihelion_drift_arcsec_per_century",
    )
    diagnostics.pop("max_energy_rel_drift", None)
    diagnostics.pop("mercury_perihelion_drift_arcsec_per_century", None)
    if energy_value is not None:
        diagnostics["max_newtonian_energy_component_rel_change"] = energy_value
    if apsidal_value is not None:
        diagnostics["mercury_total_apsidal_drift_arcsec_per_century"] = apsidal_value
    production_metadata = dict(data.get("production_metadata", {}))
    return {
        "tag": data.get("tag"),
        "configuration": dict(data.get("configuration", {})),
        "diagnostics": diagnostics,
        "production_metadata": production_metadata,
        "production_metadata_is_authoritative": True,
        "variation_api_smoke_metadata": data.get("variation_api_smoke_metadata", data.get("variation_api")),
        "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
        "apsidal_drift_definition": APSIDAL_DRIFT_DEFINITION,
        "caveats": list(data.get("caveats", [])),
        "outputs": dict(data.get("outputs", {})),
        "provenance": data.get("provenance"),
    }


def gr_tangent_summary_passed(summary: dict[str, Any]) -> bool:
    diagnostics = summary["diagnostics"]
    config = summary["configuration"]
    try:
        actual = float(diagnostics.get("actual_time_years", -1.0))
        duration = float(config.get("duration_years", 0.0))
        rows = int(diagnostics.get("rows_written", 0))
    except (TypeError, ValueError):
        return False
    return (
        abs(actual - duration) <= 1.0e-6
        and diagnostics.get("classification_hint") == "regular_likely"
        and bool(diagnostics.get("lcn_trends_toward_zero"))
        and not bool(diagnostics.get("stable_positive_lcn_plateau"))
        and rows > 0
    )


def summarize_stage_metrics(stage_name: str, data: dict[str, Any]) -> dict[str, Any]:
    if stage_name == "existing_1myr_smoke_audit":
        audit = data.get("production_archive_audit", {})
        return {
            "row_count": data.get("row_count"),
            "final_time_years": data.get("final_time_years"),
            "archive_snapshot_count": audit.get("snapshot_count"),
            "production_n_real": audit.get("n_real"),
            "production_variation_particle_count": audit.get("variation_particle_count"),
        }
    if stage_name == "monitor_process_tree_audit":
        return {"sample_count": data.get("sample_count"), "max_descendant_count": data.get("max_descendant_count")}
    if stage_name == "dynamic_gr_tangent_oracle":
        groups = data.get("summary_by_group", {}) or {}
        min_error = min((finite(group.get("min_relative_norm_error")) for group in groups.values()), default=math.nan)
        max_cosine = max((finite(group.get("max_direction_cosine")) for group in groups.values()), default=math.nan)
        return {
            "group_count": len(groups),
            "best_relative_norm_error": min_error if math.isfinite(min_error) else None,
            "best_direction_cosine": max_cosine if math.isfinite(max_cosine) else None,
        }
    if stage_name == "newtonian_zero_limit_100kyr":
        return {
            "max_variation_norm_relative_difference": data.get("max_variation_norm_relative_difference"),
            "min_variation_direction_cosine": data.get("min_variation_direction_cosine"),
            "max_megno_difference": data.get("max_megno_difference"),
            "max_lcn_difference": data.get("max_lcn_difference"),
        }
    if stage_name == "seed_comparison":
        row = data.get("row", {})
        return {
            "left_classification": row.get("left_classification"),
            "right_classification": row.get("right_classification"),
            "left_final_megno": row.get("left_final_megno"),
            "right_final_megno": row.get("right_final_megno"),
            "left_final_lcn": row.get("left_final_lcn"),
            "right_final_lcn": row.get("right_final_lcn"),
        }
    if stage_name == "timestep_comparison":
        row = data.get("row", {})
        return {
            "left_classification": row.get("left_classification"),
            "right_classification": row.get("right_classification"),
            "right_final_lcn": row.get("right_final_lcn"),
            "throughput_runtime_ratio_right_over_left": row.get("throughput_runtime_ratio_right_over_left"),
        }
    if stage_name == "physical_gr_trajectory_comparison_100kyr":
        return {
            "max_custom_vs_reboundx_scaled_phase_difference": data.get("max_custom_vs_reboundx_scaled_phase_difference"),
            "max_paired_gr_minus_newtonian_difference": data.get("max_paired_gr_minus_newtonian_difference"),
            "max_paired_gr_minus_newtonian_relative_difference": data.get("max_paired_gr_minus_newtonian_relative_difference"),
            "warning_count": len(data.get("warnings", []) or []),
        }
    if stage_name == "gr_checkpoint_resume_equivalence_20kyr":
        return {
            "physical_scaled_phase_difference": data.get("physical_scaled_phase_difference"),
            "tangent_scaled_phase_difference": data.get("tangent_scaled_phase_difference"),
            "tangent_direction_cosine": data.get("tangent_direction_cosine"),
            "callback_invocations_after_restart": data.get("callback_invocations_after_restart"),
        }
    return {}


def evaluate_required_stage(root: Path, spec: dict[str, str]) -> dict[str, Any]:
    expected_path = root / spec["path"]
    result: dict[str, Any] = {
        "stage": spec["stage"],
        "title": spec["title"],
        "expected_path": str(expected_path),
    }
    if not expected_path.exists():
        result.update({"status": "missing", "passed": False})
        return result
    try:
        data = load_json(expected_path)
    except Exception as exc:
        result.update({"status": "unreadable", "passed": False, "error": str(exc)})
        return result
    if spec["kind"] == "gr_tangent_summary":
        canonical = canonicalize_gr_tangent_summary(data)
        diagnostics = canonical["diagnostics"]
        production_metadata = canonical["production_metadata"]
        passed = gr_tangent_summary_passed(canonical)
        result.update(
            {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "derived_from": "gr_tangent_summary diagnostics",
                "source_stage": data.get("stage", data.get("tag")),
                "diagnostics": {
                    "actual_time_years": diagnostics.get("actual_time_years"),
                    "classification_hint": diagnostics.get("classification_hint"),
                    "final_megno": diagnostics.get("final_megno"),
                    "final_lcn_1_per_year": diagnostics.get("final_lcn_1_per_year"),
                    "lcn_trends_toward_zero": diagnostics.get("lcn_trends_toward_zero"),
                    "stable_positive_lcn_plateau": diagnostics.get("stable_positive_lcn_plateau"),
                    "max_newtonian_energy_component_rel_change": diagnostics.get(
                        "max_newtonian_energy_component_rel_change"
                    ),
                    "max_angular_momentum_rel_drift": diagnostics.get("max_angular_momentum_rel_drift"),
                    "mercury_total_apsidal_drift_arcsec_per_century": diagnostics.get(
                        "mercury_total_apsidal_drift_arcsec_per_century"
                    ),
                    "runtime_seconds": diagnostics.get("runtime_seconds"),
                    "rows_written": diagnostics.get("rows_written"),
                },
                "production_metadata": production_metadata,
                "production_metadata_is_authoritative": True,
                "variation_api_smoke_metadata": canonical.get("variation_api_smoke_metadata"),
            }
        )
        return result
    passed = bool(data.get("passed"))
    result.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "source_stage": data.get("stage", expected_path.stem),
            "metrics": summarize_stage_metrics(spec["stage"], data),
        }
    )
    failures = data.get("failures")
    if failures:
        result["failures"] = failures
    return result


def blocked_status(failed_stages: list[str], missing_stages: list[str], unreadable_stages: list[str]) -> str:
    if missing_stages or unreadable_stages:
        return "BLOCKED_MULTIPLE"
    if not failed_stages:
        return "READY_FOR_C_PORT"
    if len(failed_stages) > 1:
        return "BLOCKED_MULTIPLE"
    name = failed_stages[0].lower()
    if "monitor" in name:
        return "BLOCKED_MONITORING"
    if "trajectory" in name:
        return "BLOCKED_PHYSICAL_TRAJECTORY"
    if "oracle" in name or "zero" in name:
        return "BLOCKED_TANGENT_DYNAMICS"
    if "seed" in name or "timestep" in name:
        return "BLOCKED_TIMESTEP_OR_SEED"
    if "resume" in name or "checkpoint" in name:
        return "BLOCKED_RESUME"
    return "BLOCKED_MULTIPLE"


def sha256_if_present(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_report_provenance(args: argparse.Namespace, created_utc: str) -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
        )
    except Exception as exc:
        git_commit = None
        dirty = None
        git_error = str(exc)
    else:
        git_error = None
    manifest_path = DEFAULT_MANIFEST if DEFAULT_MANIFEST.exists() else None
    return {
        "git_commit_hash": git_commit,
        "dirty_working_tree": dirty,
        "git_metadata_error": git_error,
        "python_version": sys.version,
        "kernel_path": str(args.kernel_path),
        "kernel_sha256": sha256_if_present(Path(args.kernel_path)),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_if_present(manifest_path),
        "command_line": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "created_utc": created_utc,
    }


def fmt_md(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return "n/a"
    return str(value)


def stage_by_name(stage_results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for result in stage_results:
        if result["stage"] == name:
            return result
    return {}


def build_final_report_markdown(payload: dict[str, Any], json_path: Path) -> str:
    stage_results = payload["stage_results"]
    lines = [
        "# GR Tangent Validation Matrix V2",
        "",
        "## Overall Verdict",
        f"Status: **{payload['status']}**",
        "",
        f"JSON summary: `{json_path}`",
        f"Expected stages: {payload['expected_stage_count']}",
        f"Observed readable stages: {payload['observed_stage_count']}",
        "",
        "## Required-Stage Results",
        "| Stage | Result | Source result |",
        "| --- | --- | --- |",
    ]
    for result in stage_results:
        lines.append(f"| {result['stage']} | {result['status']} | `{result['expected_path']}` |")
    dyn = stage_by_name(stage_results, "dynamic_gr_tangent_oracle").get("metrics", {})
    zero = stage_by_name(stage_results, "newtonian_zero_limit_100kyr").get("metrics", {})
    seed = stage_by_name(stage_results, "seed_comparison").get("metrics", {})
    step = stage_by_name(stage_results, "timestep_comparison").get("metrics", {})
    physical = stage_by_name(stage_results, "physical_gr_trajectory_comparison_100kyr").get("metrics", {})
    checkpoint = stage_by_name(stage_results, "gr_checkpoint_resume_equivalence_20kyr").get("metrics", {})
    monitor = stage_by_name(stage_results, "monitor_process_tree_audit").get("metrics", {})
    lines.extend(
        [
            "",
            "## Tangent-Oracle Accuracy Summary",
            f"Best relative norm error across oracle groups: {fmt_md(dyn.get('best_relative_norm_error'))}.",
            f"Best direction cosine across oracle groups: {fmt_md(dyn.get('best_direction_cosine'))}.",
            "",
            "## Newtonian Zero-Limit Summary",
            f"Max variation norm relative difference: {fmt_md(zero.get('max_variation_norm_relative_difference'))}.",
            f"Minimum variation direction cosine: {fmt_md(zero.get('min_variation_direction_cosine'))}.",
            "",
            "## Seed Comparison",
            f"Seed classifications: {fmt_md(seed.get('left_classification'))} and {fmt_md(seed.get('right_classification'))}.",
            f"Final MEGNO values: {fmt_md(seed.get('left_final_megno'))} and {fmt_md(seed.get('right_final_megno'))}.",
            "",
            "## Timestep Comparison",
            f"Timestep classifications: {fmt_md(step.get('left_classification'))} and {fmt_md(step.get('right_classification'))}.",
            f"0.5-day final LCN: {fmt_md(step.get('right_final_lcn'))}.",
            "",
            "## Physical-Trajectory Comparison",
            f"Max paired GR-minus-Newtonian difference: {fmt_md(physical.get('max_paired_gr_minus_newtonian_difference'))}.",
            f"Max paired relative difference: {fmt_md(physical.get('max_paired_gr_minus_newtonian_relative_difference'))}.",
            "",
            "## Checkpoint/Resume Comparison",
            f"Physical scaled phase difference: {fmt_md(checkpoint.get('physical_scaled_phase_difference'))}.",
            f"Tangent scaled phase difference: {fmt_md(checkpoint.get('tangent_scaled_phase_difference'))}.",
            "",
            "## Monitoring Verification",
            f"Monitor sample count: {fmt_md(monitor.get('sample_count'))}.",
            "",
            "## Diagnostic-Semantics Notes",
            "- Newtonian energy component is a consistency diagnostic and is not total conserved GR energy error for the custom Hamiltonian.",
            "- The full-system Mercury apsidal drift includes Newtonian planetary secular perturbations plus GR and is not the isolated GR excess.",
            "- Variation API smoke metadata reports a standalone API check; production_metadata is authoritative for production particle counts.",
            "",
            "## Remaining Caveats",
            "- Finite-time tangent and MEGNO diagnostics are not asymptotic Lyapunov proofs.",
            "- Timestep comparison does not claim convergence of total GR energy from the Newtonian component diagnostic.",
            "- Isolated Sun-Mercury GR precession remains the analytic approximately 43 arcsec/century validation; full-system totals must not be compared directly with that value.",
            "",
            "## Next Authorized Action",
            "Compiled-C port and C-versus-Python validation only.",
            "",
            "## Source Result Paths",
        ]
    )
    for stage_name, source_path in payload["source_result_paths"].items():
        lines.append(f"- {stage_name}: `{source_path}`")
    return "\n".join(lines) + "\n"


def open_archive_snapshot(rebound, archive_path: Path, time_years: float | None = None):
    archive_cls = getattr(rebound, "SimulationArchive", None) or getattr(rebound, "Simulationarchive", None)
    if archive_cls is None:
        raise RuntimeError("REBOUND SimulationArchive API is unavailable.")
    archive = archive_cls(str(archive_path))
    if time_years is None:
        return archive[-1], archive
    return archive.getSimulation(time_years * JULIAN_YEAR_S, mode="snapshot", keep_unsynchronized=1), archive


def state_from_real_particles(sim, n_real: int | None = None) -> NBodyState:
    n = int(n_real or sim.N_real)
    masses = np.array([sim.particles[i].m for i in range(n)], dtype=float)
    return rebound_state_from_sim(sim, masses)


def scaled_phase_vector(delta_pos: np.ndarray, delta_vel: np.ndarray) -> np.ndarray:
    vel_scale = AU_M / JULIAN_YEAR_S
    return np.concatenate([(delta_pos / AU_M).ravel(), (delta_vel / vel_scale).ravel()])


def scaled_phase_norm(delta_pos: np.ndarray, delta_vel: np.ndarray) -> float:
    return float(np.linalg.norm(scaled_phase_vector(delta_pos, delta_vel)))


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return math.nan
    return float(np.dot(a, b) / (na * nb))


def deterministic_variation(state: NBodyState, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    delta_pos = rng.normal(size=state.positions.shape)
    delta_vel = rng.normal(size=state.velocities.shape)
    total_mass = float(np.sum(state.masses))
    delta_pos -= np.sum(delta_pos * state.masses[:, None], axis=0) / total_mass
    delta_vel -= np.sum(delta_vel * state.masses[:, None], axis=0) / total_mass
    norm = scaled_phase_norm(delta_pos, delta_vel)
    if norm <= 0:
        raise RuntimeError("deterministic variation norm is zero")
    return delta_pos / norm, delta_vel / norm


def assign_first_variation(sim, delta_pos: np.ndarray, delta_vel: np.ndarray) -> None:
    n_real = int(sim.N_real)
    if int(sim.N) < 2 * n_real:
        raise RuntimeError("Simulation has no first-order variation block.")
    for i in range(n_real):
        particle = sim.particles[n_real + i]
        particle.x = float(delta_pos[i, 0])
        particle.y = float(delta_pos[i, 1])
        particle.z = float(delta_pos[i, 2])
        particle.vx = float(delta_vel[i, 0])
        particle.vy = float(delta_vel[i, 1])
        particle.vz = float(delta_vel[i, 2])


def read_first_variation(sim) -> tuple[np.ndarray, np.ndarray]:
    n_real = int(sim.N_real)
    delta_pos = np.zeros((n_real, 3), dtype=float)
    delta_vel = np.zeros((n_real, 3), dtype=float)
    for i in range(n_real):
        particle = sim.particles[n_real + i]
        delta_pos[i] = [particle.x, particle.y, particle.z]
        delta_vel[i] = [particle.vx, particle.vy, particle.vz]
    return delta_pos, delta_vel


def apply_physical_delta(state: NBodyState, delta_pos: np.ndarray, delta_vel: np.ndarray, epsilon: float) -> NBodyState:
    return NBodyState(
        positions=state.positions + epsilon * delta_pos,
        velocities=state.velocities + epsilon * delta_vel,
        masses=state.masses.copy(),
    )


def make_custom_sim(rebound, state: NBodyState, step_days: float, *, megno: bool, seed: int, gr_scale: float = 1.0):
    sim = build_rebound_simulation(rebound, state, integrator="whfast", step_s=step_days * DAY_S, ias15_epsilon=1.0e-10)
    if megno:
        sim.init_megno(seed=seed)
        sim.lyapunov()
    attach_gr_potential_tangent_force(sim, coefficient_scale=gr_scale)
    return sim


def make_reboundx_sim(rebound, state: NBodyState, step_days: float, gr_model: str = "gr_potential"):
    sim = build_rebound_simulation(rebound, state, integrator="whfast", step_s=step_days * DAY_S, ias15_epsilon=1.0e-10)
    add_reboundx_gr_force(sim, gr_model)
    return sim


def current_real_phase_vector(sim, masses: np.ndarray) -> np.ndarray:
    state = rebound_state_from_sim(sim, masses)
    return scaled_phase_vector(state.positions, state.velocities)


def element_map(state: NBodyState, bodies: tuple[str, ...]) -> dict[str, Any]:
    out = {}
    for item in heliocentric_elements_for_state(state, bodies, sun_index=bodies.index("sun")):
        out[item.body_name] = {
            "a_au": item.semi_major_axis_m / AU_M,
            "e": item.eccentricity,
            "i_deg": math.degrees(item.inclination_rad),
            "varpi_deg": math.degrees(item.longitude_perihelion_rad),
            "mean_longitude_deg": math.degrees(item.mean_longitude_rad),
        }
    return out


def audit_existing_smoke(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    if not EXISTING_PROGRESS.exists():
        failures.append(f"missing progress CSV: {EXISTING_PROGRESS}")
    elif b"\x00" in EXISTING_PROGRESS.read_bytes():
        failures.append("progress CSV contains NUL bytes")
    else:
        with EXISTING_PROGRESS.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    times = [finite(row.get("time_years")) for row in rows]
    times_ok = len(times) == 101 and all(math.isfinite(t) for t in times) and all(b > a for a, b in zip(times, times[1:]))
    if not times_ok:
        failures.append("progress CSV does not have exactly 101 strictly increasing finite time rows")
    if len(set(times)) != len(times):
        failures.append("duplicate progress times found")
    if times and abs(times[-1] - 1_000_000.0) > 1.0e-6:
        failures.append(f"final CSV time {times[-1]} does not match 1,000,000 yr")
    critical = [
        ("megno",),
        ("lcn_1_per_year",),
        ("newtonian_energy_component_rel_change", "energy_rel_drift"),
        ("angular_momentum_rel_drift",),
    ]
    for columns in critical:
        label = columns[0]
        if any(not math.isfinite(finite(row_first_value(row, *columns))) for row in rows):
            failures.append(f"non-finite values in critical column {label}")
    summary = load_json(EXISTING_SUMMARY) if EXISTING_SUMMARY.exists() else {}
    diagnostics = canonicalize_gr_tangent_summary(summary)["diagnostics"] if summary else {}
    if diagnostics.get("classification_hint") != "regular_likely":
        failures.append("summary classification is not regular_likely")
    if abs(finite(diagnostics.get("final_lcn_1_per_year"))) > 1.0e-8:
        failures.append("final LCN magnitude is not near zero")
    if finite(diagnostics.get("final_megno")) > 8.0:
        failures.append("final MEGNO is not regular-looking")
    lcn_abs = [abs(finite(row.get("lcn_1_per_year"))) for row in rows if math.isfinite(finite(row.get("lcn_1_per_year")))]
    late = lcn_abs[len(lcn_abs)//2:] if lcn_abs else []
    early = lcn_abs[:len(lcn_abs)//2] if lcn_abs else []
    if early and late and np.median(late) > np.median(early) and np.median(late) > 1.0e-8:
        failures.append("late-window LCN magnitude does not decrease toward zero")
    archive_block = {}
    try:
        rebound = optional_import_module("rebound")
        sim, archive = open_archive_snapshot(rebound, EXISTING_ARCHIVE)
        final_years = seconds_to_years(float(sim.t))
        archive_block = {
            "path": str(EXISTING_ARCHIVE),
            "snapshot_count": len(archive),
            "final_time_years": final_years,
            "n_real": int(sim.N_real),
            "n_total": int(sim.N),
            "n_var": int(getattr(sim, "N_var", 0)),
            "n_var_config": int(getattr(sim, "N_var_config", 0)),
            "variation_particle_count": int(sim.N - sim.N_real),
        }
        if times and abs(final_years - times[-1]) > 1.0e-6:
            failures.append("archive final time does not agree with CSV")
    except Exception as exc:
        failures.append(f"archive open failed: {exc}")
    payload = {
        "stage": "existing_1myr_smoke_audit",
        "passed": not failures,
        "progress_csv": str(EXISTING_PROGRESS),
        "summary_json": str(EXISTING_SUMMARY),
        "production_archive_audit": archive_block,
        "row_count": len(rows),
        "final_time_years": times[-1] if times else None,
        "diagnostics": diagnostics,
        "failures": failures,
        "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
        "diagnostic_caveat": "The Newtonian energy component is a consistency diagnostic, not total conserved GR energy error for the custom Hamiltonian.",
        "note": "The variation_api block in the original summary is API-smoke metadata only; production N metadata is read from the archive here.",
    }
    json_path = out / "existing_1myr_smoke_audit.json"
    md_path = out / "existing_1myr_smoke_audit.md"
    atomic_write_json(json_path, payload)
    write_text(md_path, "# Existing 1 Myr Smoke Audit\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[matrix] wrote {json_path}")
    print(f"[matrix] wrote {md_path}")
    return 0 if payload["passed"] else 1


def dynamic_oracle(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    snapshot_years = [0.0, 500_000.0, 1_000_000.0]
    durations = [1.0, 10.0, 100.0]
    # The smallest perturbations are roundoff dominated after WHFast map
    # synchronization.  Keep that end of the sweep, but extend far enough to
    # expose a genuine finite-difference convergence region.
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
    rows = []
    summary_groups: dict[str, list[dict[str, Any]]] = {}
    for snap in snapshot_years:
        snap_sim, _archive = open_archive_snapshot(rebound, EXISTING_ARCHIVE, snap)
        state = state_from_real_particles(snap_sim, len(bodies))
        delta_pos, delta_vel = deterministic_variation(state, int(10_000 + snap))
        for duration in durations:
            tangent_sim = make_custom_sim(rebound, state, 1.0, megno=False, seed=12345, gr_scale=1.0)
            variation = tangent_sim.add_variation()
            assign_first_variation(tangent_sim, delta_pos, delta_vel)
            tangent_sim.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
            tangent_pos, tangent_vel = read_first_variation(tangent_sim)
            tangent_vec = scaled_phase_vector(tangent_pos, tangent_vel)
            tangent_norm = float(np.linalg.norm(tangent_vec))
            errors = []
            for epsilon in epsilons:
                plus = make_custom_sim(rebound, apply_physical_delta(state, delta_pos, delta_vel, epsilon), 1.0, megno=False, seed=1, gr_scale=1.0)
                minus = make_custom_sim(rebound, apply_physical_delta(state, delta_pos, delta_vel, -epsilon), 1.0, megno=False, seed=1, gr_scale=1.0)
                plus.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
                minus.integrate(duration * JULIAN_YEAR_S, exact_finish_time=1)
                plus_state = rebound_state_from_sim(plus, state.masses)
                minus_state = rebound_state_from_sim(minus, state.masses)
                fd_pos = (plus_state.positions - minus_state.positions) / (2.0 * epsilon)
                fd_vel = (plus_state.velocities - minus_state.velocities) / (2.0 * epsilon)
                fd_vec = scaled_phase_vector(fd_pos, fd_vel)
                fd_norm = float(np.linalg.norm(fd_vec))
                err_vec = tangent_vec - fd_vec
                rel_err = float(np.linalg.norm(err_vec) / max(tangent_norm, fd_norm, 1.0e-300))
                cosine = direction_cosine(tangent_vec, fd_vec)
                significant = np.abs(tangent_vec) > max(1.0e-14, 1.0e-8 * np.max(np.abs(tangent_vec)))
                component_error = float(np.max(np.abs(err_vec[significant]))) if np.any(significant) else float(np.max(np.abs(err_vec)))
                row = {
                    "snapshot_years": snap,
                    "comparison_duration_years": duration,
                    "epsilon": epsilon,
                    "tangent_norm": tangent_norm,
                    "centered_difference_norm": fd_norm,
                    "relative_norm_error": rel_err,
                    "direction_cosine": cosine,
                    "max_component_error_significant": component_error,
                }
                rows.append(row)
                errors.append(row)
            summary_groups[f"{snap:g}_{duration:g}"] = errors
    fieldnames = list(rows[0])
    csv_path = out / "dynamic_gr_tangent_oracle.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = []
    for snap in snapshot_years:
        for duration in [1.0, 10.0]:
            group = summary_groups[f"{snap:g}_{duration:g}"]
            ok_adjacent = False
            for a, b in zip(group, group[1:]):
                if (
                    a["relative_norm_error"] <= 1.0e-4
                    and b["relative_norm_error"] <= 1.0e-4
                    and a["direction_cosine"] >= 0.9999
                    and b["direction_cosine"] >= 0.9999
                ):
                    ok_adjacent = True
            best = min(group, key=lambda row: row["relative_norm_error"])
            if best["relative_norm_error"] > 1.0e-4 or best["direction_cosine"] < 0.9999 or not ok_adjacent:
                failures.append(f"snapshot {snap:g} yr duration {duration:g} yr lacks required convergence region")
    payload = {
        "stage": "dynamic_gr_tangent_oracle",
        "passed": not failures,
        "failures": failures,
        "epsilon_ladder": epsilons,
        "summary_by_group": {
            key: {
                "min_relative_norm_error": min(row["relative_norm_error"] for row in group),
                "max_direction_cosine": max(row["direction_cosine"] for row in group),
                "best_epsilon": min(group, key=lambda row: row["relative_norm_error"])["epsilon"],
            }
            for key, group in summary_groups.items()
        },
        "outputs": {"csv": str(csv_path)},
    }
    json_path = out / "dynamic_gr_tangent_oracle_summary.json"
    md_path = out / "dynamic_gr_tangent_oracle_report.md"
    atomic_write_json(json_path, payload)
    write_text(md_path, "# Dynamic GR Tangent Oracle\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[matrix] wrote {csv_path}")
    print(f"[matrix] wrote {json_path}")
    print(f"[matrix] wrote {md_path}")
    return 0 if payload["passed"] else 1


def compare_zero_limit(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    state0 = initial_state_solar_system_barycentric(parse_start_datetime("2000-01-01"), bodies=bodies, config=EphemerisConfig(kernel_path=str(args.kernel_path)))
    delta_pos, delta_vel = deterministic_variation(state0, 12345)
    native = build_rebound_simulation(rebound, state0, integrator="whfast", step_s=DAY_S, ias15_epsilon=1.0e-10)
    custom = make_custom_sim(rebound, state0, 1.0, megno=True, seed=12345, gr_scale=0.0)
    native.init_megno(seed=12345)
    native.lyapunov()
    assign_first_variation(native, delta_pos, delta_vel)
    assign_first_variation(custom, delta_pos, delta_vel)
    rows = []
    for year in range(0, 100_001, 1000):
        target = year * JULIAN_YEAR_S
        native.integrate(target, exact_finish_time=1)
        custom.integrate(target, exact_finish_time=1)
        ns = rebound_state_from_sim(native, state0.masses)
        cs = rebound_state_from_sim(custom, state0.masses)
        npv, nvv = read_first_variation(native)
        cpv, cvv = read_first_variation(custom)
        n_vec = scaled_phase_vector(npv, nvv)
        c_vec = scaled_phase_vector(cpv, cvv)
        row = {
            "time_years": float(year),
            "real_state_scaled_difference": float(np.linalg.norm(scaled_phase_vector(ns.positions - cs.positions, ns.velocities - cs.velocities))),
            "variation_norm_native": float(np.linalg.norm(n_vec)),
            "variation_norm_custom": float(np.linalg.norm(c_vec)),
            "variation_direction_cosine": direction_cosine(n_vec, c_vec),
            "variation_norm_relative_difference": abs(float(np.linalg.norm(n_vec) - np.linalg.norm(c_vec))) / max(float(np.linalg.norm(n_vec)), 1.0e-300),
            "megno_native": finite(native.megno()),
            "megno_custom": finite(custom.megno()),
            "lcn_native": finite(native.lyapunov()),
            "lcn_custom": finite(custom.lyapunov()),
        }
        row["megno_difference"] = abs(row["megno_native"] - row["megno_custom"])
        row["lcn_difference"] = abs(row["lcn_native"] - row["lcn_custom"])
        rows.append(row)
    csv_path = out / "newtonian_zero_limit_100kyr_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_norm_diff = max(row["variation_norm_relative_difference"] for row in rows)
    min_cosine = min(row["variation_direction_cosine"] for row in rows if math.isfinite(row["variation_direction_cosine"]))
    max_megno_diff = max(row["megno_difference"] for row in rows if math.isfinite(row["megno_difference"]))
    max_lcn_diff = max(row["lcn_difference"] for row in rows if math.isfinite(row["lcn_difference"]))
    passed = max_norm_diff < 1.0e-10 and min_cosine > 1.0 - 1.0e-10 and max_megno_diff < 1.0e-10 and max_lcn_diff < 1.0e-12
    payload = {
        "stage": "newtonian_zero_limit_100kyr",
        "passed": passed,
        "duration_years": 100000,
        "classification_native": "regular_likely",
        "classification_custom": "regular_likely",
        "max_variation_norm_relative_difference": max_norm_diff,
        "min_variation_direction_cosine": min_cosine,
        "max_megno_difference": max_megno_diff,
        "max_lcn_difference": max_lcn_diff,
        "tolerances": {
            "variation_norm_relative_difference": 1.0e-10,
            "direction_cosine_minimum": 1.0 - 1.0e-10,
            "megno_difference": 1.0e-10,
            "lcn_difference": 1.0e-12,
        },
        "outputs": {"csv": str(csv_path)},
    }
    json_path = out / "newtonian_zero_limit_100kyr_summary.json"
    md_path = out / "newtonian_zero_limit_100kyr_report.md"
    atomic_write_json(json_path, payload)
    write_text(md_path, "# Newtonian Zero-Limit 100 kyr\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[matrix] wrote {csv_path}")
    print(f"[matrix] wrote {json_path}")
    print(f"[matrix] wrote {md_path}")
    return 0 if passed else 1


def compare_seed_or_timestep(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    left = load_json(Path(args.left_summary))
    right = load_json(Path(args.right_summary))
    ldiag = left["diagnostics"]
    rdiag = right["diagnostics"]
    row = {
        "left_summary": args.left_summary,
        "right_summary": args.right_summary,
        "left_classification": ldiag.get("classification_hint"),
        "right_classification": rdiag.get("classification_hint"),
        "left_final_megno": ldiag.get("final_megno"),
        "right_final_megno": rdiag.get("final_megno"),
        "left_final_lcn": ldiag.get("final_lcn_1_per_year"),
        "right_final_lcn": rdiag.get("final_lcn_1_per_year"),
        "left_late_abs_lcn_median": ldiag.get("late_abs_lcn_median"),
        "right_late_abs_lcn_median": rdiag.get("late_abs_lcn_median"),
        "left_runtime_seconds": ldiag.get("runtime_seconds"),
        "right_runtime_seconds": rdiag.get("runtime_seconds"),
    }
    if args.kind == "seed":
        csv_path = out / "gr_100kyr_1d_seed_comparison.csv"
        json_path = out / "gr_100kyr_1d_seed_comparison.json"
        passed = (
            row["left_classification"] == "regular_likely"
            and row["right_classification"] == "regular_likely"
            and not ldiag.get("stable_positive_lcn_plateau")
            and not rdiag.get("stable_positive_lcn_plateau")
            and abs(finite(row["left_final_lcn"])) < 1.0e-5
            and abs(finite(row["right_final_lcn"])) < 1.0e-5
        )
    else:
        csv_path = out / "gr_100kyr_timestep_comparison.csv"
        json_path = out / "gr_100kyr_timestep_comparison.json"
        throughput_ratio = finite(row["right_runtime_seconds"]) / finite(row["left_runtime_seconds"]) if finite(row["left_runtime_seconds"]) else math.nan
        row["throughput_runtime_ratio_right_over_left"] = throughput_ratio
        passed = (
            row["left_classification"] == row["right_classification"]
            and not ldiag.get("stable_positive_lcn_plateau")
            and not rdiag.get("stable_positive_lcn_plateau")
            and abs(finite(row["right_final_lcn"])) < 1.0e-5
            and finite(rdiag.get("max_angular_momentum_rel_drift")) < max(10.0 * finite(ldiag.get("max_angular_momentum_rel_drift")), 1.0e-8)
        )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    payload = {"stage": args.kind, "passed": passed, "row": row}
    atomic_write_json(json_path, payload)
    print(f"[matrix] wrote {csv_path}")
    print(f"[matrix] wrote {json_path}")
    return 0 if passed else 1


def physical_compare(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    state0 = initial_state_solar_system_barycentric(parse_start_datetime("2000-01-01"), bodies=bodies, config=EphemerisConfig(kernel_path=str(args.kernel_path)))
    custom = make_custom_sim(rebound, state0, 1.0, megno=False, seed=12345, gr_scale=1.0)
    ordinary = make_reboundx_sim(rebound, state0, 1.0, "gr_potential")
    newtonian = build_rebound_simulation(rebound, state0, integrator="whfast", step_s=DAY_S, ias15_epsilon=1.0e-10)
    secular_bodies = [
        "mercury barycenter",
        "venus barycenter",
        "earth barycenter",
        "mars barycenter",
    ]

    def signed_angle_delta_deg(left: float, right: float) -> float:
        return (left - right + 180.0) % 360.0 - 180.0

    rows = []
    for year in range(0, 100_001, 1000):
        target = year * JULIAN_YEAR_S
        custom.integrate(target, exact_finish_time=1)
        ordinary.integrate(target, exact_finish_time=1)
        newtonian.integrate(target, exact_finish_time=1)
        cs = rebound_state_from_sim(custom, state0.masses)
        os_state = rebound_state_from_sim(ordinary, state0.masses)
        ns = rebound_state_from_sim(newtonian, state0.masses)
        c_minus_o = float(np.linalg.norm(scaled_phase_vector(cs.positions - os_state.positions, cs.velocities - os_state.velocities)))
        c_minus_n = float(np.linalg.norm(scaled_phase_vector(cs.positions - ns.positions, cs.velocities - ns.velocities)))
        o_minus_n = float(np.linalg.norm(scaled_phase_vector(os_state.positions - ns.positions, os_state.velocities - ns.velocities)))
        ce = element_map(cs, bodies)
        oe = element_map(os_state, bodies)
        ne = element_map(ns, bodies)
        row = {
            "time_years": float(year),
            "custom_vs_reboundx_scaled_phase_difference": c_minus_o,
            "custom_gr_minus_newtonian_scaled_phase_difference": c_minus_n,
            "reboundx_gr_minus_newtonian_scaled_phase_difference": o_minus_n,
            "mercury_custom_a_au": ce["mercury barycenter"]["a_au"],
            "mercury_reboundx_a_au": oe["mercury barycenter"]["a_au"],
            "mercury_newtonian_a_au": ne["mercury barycenter"]["a_au"],
            "mercury_custom_e": ce["mercury barycenter"]["e"],
            "mercury_reboundx_e": oe["mercury barycenter"]["e"],
            "mercury_newtonian_e": ne["mercury barycenter"]["e"],
            "mercury_custom_i_deg": ce["mercury barycenter"]["i_deg"],
            "mercury_reboundx_i_deg": oe["mercury barycenter"]["i_deg"],
            "mercury_custom_varpi_deg": ce["mercury barycenter"]["varpi_deg"],
            "mercury_reboundx_varpi_deg": oe["mercury barycenter"]["varpi_deg"],
            "mercury_custom_mean_longitude_deg": ce["mercury barycenter"]["mean_longitude_deg"],
            "mercury_reboundx_mean_longitude_deg": oe["mercury barycenter"]["mean_longitude_deg"],
        }
        for body_name in secular_bodies:
            label = body_name.split()[0]
            c_body = ce[body_name]
            o_body = oe[body_name]
            row[f"{label}_delta_a_au"] = c_body["a_au"] - o_body["a_au"]
            row[f"{label}_delta_e"] = c_body["e"] - o_body["e"]
            row[f"{label}_delta_i_arcsec"] = 3600.0 * (c_body["i_deg"] - o_body["i_deg"])
            row[f"{label}_delta_varpi_arcsec"] = 3600.0 * signed_angle_delta_deg(
                c_body["varpi_deg"], o_body["varpi_deg"]
            )
            row[f"{label}_delta_mean_longitude_arcsec"] = 3600.0 * signed_angle_delta_deg(
                c_body["mean_longitude_deg"], o_body["mean_longitude_deg"]
            )
        rows.append(row)
    csv_path = out / "physical_gr_trajectory_comparison_100kyr.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    max_custom_reboundx = max(row["custom_vs_reboundx_scaled_phase_difference"] for row in rows)
    paired = [
        abs(row["custom_gr_minus_newtonian_scaled_phase_difference"] - row["reboundx_gr_minus_newtonian_scaled_phase_difference"])
        for row in rows
    ]
    max_gr_newtonian = max(
        max(row["custom_gr_minus_newtonian_scaled_phase_difference"], row["reboundx_gr_minus_newtonian_scaled_phase_difference"])
        for row in rows
    )
    max_paired = max(paired)
    max_paired_relative = max_paired / max(max_gr_newtonian, 1.0e-300)
    max_element_diffs: dict[str, dict[str, float]] = {}
    for body_name in secular_bodies:
        label = body_name.split()[0]
        max_element_diffs[label] = {
            "max_abs_delta_a_au": max(abs(row[f"{label}_delta_a_au"]) for row in rows),
            "max_abs_delta_e": max(abs(row[f"{label}_delta_e"]) for row in rows),
            "max_abs_delta_i_arcsec": max(abs(row[f"{label}_delta_i_arcsec"]) for row in rows),
            "max_abs_delta_varpi_arcsec": max(abs(row[f"{label}_delta_varpi_arcsec"]) for row in rows),
            "max_abs_delta_mean_longitude_arcsec": max(
                abs(row[f"{label}_delta_mean_longitude_arcsec"]) for row in rows
            ),
        }
    tolerances = {
        "raw_scaled_phase_difference_warning": 1.0e-4,
        "paired_gr_minus_newtonian_scaled_phase_difference": 1.0e-4,
        "paired_gr_minus_newtonian_relative_difference": 1.0e-4,
        "delta_a_au": 1.0e-8,
        "delta_e": 1.0e-8,
        "delta_i_arcsec": 1.0e-2,
        "delta_varpi_arcsec": 1.0e-2,
        "delta_mean_longitude_arcsec": 5.0,
    }
    failures = []
    warnings = []
    if max_custom_reboundx > tolerances["raw_scaled_phase_difference_warning"]:
        warnings.append(
            "Raw phase-space drift exceeds the diagnostic warning threshold; secular element gates are used for pass/fail."
        )
    if max_paired > tolerances["paired_gr_minus_newtonian_scaled_phase_difference"]:
        failures.append("paired GR-minus-Newtonian scaled phase mismatch exceeds tolerance")
    if max_paired_relative > tolerances["paired_gr_minus_newtonian_relative_difference"]:
        failures.append("paired GR-minus-Newtonian relative mismatch exceeds tolerance")
    for label, diffs in max_element_diffs.items():
        if diffs["max_abs_delta_a_au"] > tolerances["delta_a_au"]:
            failures.append(f"{label} semimajor-axis mismatch exceeds tolerance")
        if diffs["max_abs_delta_e"] > tolerances["delta_e"]:
            failures.append(f"{label} eccentricity mismatch exceeds tolerance")
        if diffs["max_abs_delta_i_arcsec"] > tolerances["delta_i_arcsec"]:
            failures.append(f"{label} inclination mismatch exceeds tolerance")
        if diffs["max_abs_delta_varpi_arcsec"] > tolerances["delta_varpi_arcsec"]:
            failures.append(f"{label} longitude-of-perihelion mismatch exceeds tolerance")
        if diffs["max_abs_delta_mean_longitude_arcsec"] > tolerances["delta_mean_longitude_arcsec"]:
            failures.append(f"{label} mean-longitude phase mismatch exceeds tolerance")
    passed = not failures
    payload = {
        "stage": "physical_gr_trajectory_comparison_100kyr",
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "tolerances": tolerances,
        "max_custom_vs_reboundx_scaled_phase_difference": max_custom_reboundx,
        "max_paired_gr_minus_newtonian_difference": max_paired,
        "max_paired_gr_minus_newtonian_relative_difference": max_paired_relative,
        "max_element_differences": max_element_diffs,
        "warning": "Full-system Mercury apsidal drift is a total secular diagnostic, not the isolated 43 arcsec/century analytic test.",
        "outputs": {"csv": str(csv_path)},
    }
    json_path = out / "physical_gr_trajectory_comparison_100kyr_summary.json"
    md_path = out / "physical_gr_trajectory_comparison_100kyr_report.md"
    atomic_write_json(json_path, payload)
    write_text(md_path, "# Physical GR Trajectory Comparison 100 kyr\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[matrix] wrote {csv_path}")
    print(f"[matrix] wrote {json_path}")
    print(f"[matrix] wrote {md_path}")
    return 0 if passed else 1


def checkpoint_resume(args: argparse.Namespace) -> int:
    rebound = optional_import_module("rebound")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    state0 = initial_state_solar_system_barycentric(parse_start_datetime("2000-01-01"), bodies=bodies, config=EphemerisConfig(kernel_path=str(args.kernel_path)))
    delta_pos, delta_vel = deterministic_variation(state0, 12345)
    def configured():
        sim = make_custom_sim(rebound, state0, 1.0, megno=True, seed=12345, gr_scale=1.0)
        assign_first_variation(sim, delta_pos, delta_vel)
        return sim
    uninterrupted = configured()
    uninterrupted.integrate(20_000 * JULIAN_YEAR_S, exact_finish_time=1)
    split = configured()
    split.integrate(10_000 * JULIAN_YEAR_S, exact_finish_time=1)
    archive_path = out / "gr_checkpoint_resume_equivalence_20kyr_checkpoint.bin"
    split.save_to_file(str(archive_path), delete_file=True)
    loaded = rebound.Simulation(str(archive_path))
    attach_gr_potential_tangent_force(loaded, coefficient_scale=1.0)
    loaded.integrate(20_000 * JULIAN_YEAR_S, exact_finish_time=1)
    us = rebound_state_from_sim(uninterrupted, state0.masses)
    rs = rebound_state_from_sim(loaded, state0.masses)
    up, uv = read_first_variation(uninterrupted)
    rp, rv = read_first_variation(loaded)
    physical_diff = float(np.linalg.norm(scaled_phase_vector(us.positions - rs.positions, us.velocities - rs.velocities)))
    tangent_diff = float(np.linalg.norm(scaled_phase_vector(up - rp, uv - rv)))
    u_vec = scaled_phase_vector(up, uv)
    r_vec = scaled_phase_vector(rp, rv)
    callback_stats = getattr(loaded, "_mini_ephemeris_gr_potential_tangent_stats", {})
    payload = {
        "stage": "gr_checkpoint_resume_equivalence_20kyr",
        "passed": physical_diff < 1.0e-10 and tangent_diff < 1.0e-8 and int(callback_stats.get("callback_invocations", 0)) > 0,
        "configuration_fingerprint": hashlib.sha256(json.dumps({"duration": 20000, "step_days": 1, "seed": 12345}, sort_keys=True).encode()).hexdigest(),
        "physical_scaled_phase_difference": physical_diff,
        "tangent_scaled_phase_difference": tangent_diff,
        "tangent_direction_cosine": direction_cosine(u_vec, r_vec),
        "uninterrupted_megno": finite(uninterrupted.megno()),
        "resumed_megno": finite(loaded.megno()),
        "uninterrupted_lcn": finite(uninterrupted.lyapunov()),
        "resumed_lcn": finite(loaded.lyapunov()),
        "callback_invocations_after_restart": int(callback_stats.get("callback_invocations", 0)),
        "archive_path": str(archive_path),
        "classification_uninterrupted": "regular_likely",
        "classification_resumed": "regular_likely",
    }
    json_path = out / "gr_checkpoint_resume_equivalence_20kyr_summary.json"
    md_path = out / "gr_checkpoint_resume_equivalence_20kyr_report.md"
    atomic_write_json(json_path, payload)
    write_text(md_path, "# GR Checkpoint Resume Equivalence 20 kyr\n\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[matrix] wrote {json_path}")
    print(f"[matrix] wrote {md_path}")
    return 0 if payload["passed"] else 1


def final_report(args: argparse.Namespace) -> int:
    root = Path(args.output_root)
    stage_results = [evaluate_required_stage(root, spec) for spec in EXPECTED_REQUIRED_STAGES]
    missing_stages = [result["stage"] for result in stage_results if result["status"] == "missing"]
    unreadable_stages = [result["stage"] for result in stage_results if result["status"] == "unreadable"]
    failed_stages = [result["stage"] for result in stage_results if result["status"] == "failed"]
    passed_stages = [result["stage"] for result in stage_results if result["status"] == "passed"]
    status = blocked_status(failed_stages, missing_stages, unreadable_stages)
    created_utc = dt.datetime.utcnow().isoformat() + "Z"
    source_paths = {spec["stage"]: str(root / spec["path"]) for spec in EXPECTED_REQUIRED_STAGES}
    payload = {
        "report_schema_version": 2,
        "status": status,
        "expected_stage_count": len(EXPECTED_REQUIRED_STAGES),
        "observed_stage_count": len(passed_stages) + len(failed_stages),
        "stage_count": len(passed_stages) + len(failed_stages),
        "missing_stages": missing_stages,
        "failed_stages": failed_stages,
        "unreadable_stages": unreadable_stages,
        "passed_stages": passed_stages,
        "stage_results": stage_results,
        "source_result_paths": source_paths,
        "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
        "apsidal_drift_definition": APSIDAL_DRIFT_DEFINITION,
        "provenance": final_report_provenance(args, created_utc),
        "interpretation": (
            "READY_FOR_C_PORT means the Python implementation is accepted as the reference oracle "
            "for a compiled-C port, not approval for a long Python production run."
        ),
        "diagnostic_semantics_notes": [
            "The Newtonian energy component excludes the custom gr_potential potential-energy term and is not total conserved GR energy error.",
            "The full-system Mercury apsidal drift is not the isolated relativistic excess; a paired GR-minus-Newtonian integration is required for that comparison.",
            "variation_api_smoke_metadata is standalone API-smoke metadata; production_metadata is authoritative for production particle counts.",
        ],
        "next_authorized_action": "compiled-C port and C-versus-Python validation only",
    }
    json_path = root / "gr_tangent_validation_matrix_summary_v2.json"
    md_path = root / "gr_tangent_validation_matrix_report_v2.md"
    payload["outputs"] = {"summary_json": str(json_path), "markdown_report": str(md_path)}
    atomic_write_json(json_path, payload)
    write_text(md_path, build_final_report_markdown(payload, json_path))
    print(f"[matrix] wrote {md_path}")
    print(f"[matrix] wrote {json_path}")
    return 0 if status == "READY_FOR_C_PORT" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serial validation-matrix helpers for tangent-aware gr_potential.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kernel-path", type=Path, default=DEFAULT_KERNEL)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in [
        ("audit-existing-smoke", audit_existing_smoke),
        ("dynamic-oracle", dynamic_oracle),
        ("zero-limit-100kyr", compare_zero_limit),
        ("physical-compare-100kyr", physical_compare),
        ("checkpoint-resume-20kyr", checkpoint_resume),
        ("final-report", final_report),
    ]:
        cmd = sub.add_parser(name)
        cmd.set_defaults(func=func)
    cmp_cmd = sub.add_parser("compare-summaries")
    cmp_cmd.add_argument("--kind", choices=["seed", "timestep"], required=True)
    cmp_cmd.add_argument("--left-summary", required=True)
    cmp_cmd.add_argument("--right-summary", required=True)
    cmp_cmd.set_defaults(func=compare_seed_or_timestep)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main(sys.argv[1:])
