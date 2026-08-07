from __future__ import annotations

import argparse
import csv
import datetime as dt
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .long_term_stability_cli import (
    configure_rebound_simulationarchive,
    optional_import_module,
    rebound_state_from_sim,
)
from .m0_energy_precision_diagnosis import compensated_energy, decimal_energy
from .m0_integrator_roundoff_diagnosis import (
    PHYSICAL_STATE_FIELDS,
    PROGRESS_FIELDS,
    DiagnosisError,
    _audit_state_file,
    _block_metrics,
    _build_simulation,
    _energy_statistics,
    _existing_states,
    _git,
    _lane_paths,
    _load_json,
    _matched_state_comparison,
    _progress_row,
    _read_physical_groups,
    _require,
    _runtime_identity,
    _settings,
    _state_energy,
    _state_rows,
    _write_csv_atomic,
    audit as audit_step3d,
    classify_mechanism,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import (
    AU_M,
    DAY_S,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
)
from .rebound_gr_tangent_backend_cli import atomic_write_json, canonical_hash, sha256_file
from .stability_diagnostics import total_angular_momentum_vector


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/16_m0_ias15_phase_reference_v1.json"
)
REFERENCE_STATUSES = {
    "IAS15_REFERENCE_QUALIFIED_AT_ROUNDOFF_PHASE_FLOOR",
    "IAS15_REFERENCE_NOT_QUALIFIED",
    "BLOCKED",
}
ANGLE_FIELDS = (
    "inclination_rad",
    "longitude_ascending_node_rad",
    "argument_perihelion_rad",
    "mean_anomaly_rad",
    "mean_longitude_rad",
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(len(array) > 0 and np.all(np.isfinite(array)), "Bad timestep sample.")
    return {
        "count": len(array),
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def _expanded_configuration(
    manifest: dict[str, Any], definition: dict[str, Any]
) -> dict[str, Any]:
    config = {**manifest["common_configuration"], **definition["configuration"]}
    _require(
        canonical_hash(config) == definition["configuration_fingerprint"],
        f"Configuration fingerprint mismatch: {definition['id']}",
    )
    return config


def _verify_inventory(entries: Any) -> int:
    if isinstance(entries, dict):
        entries = list(entries.values())
    _require(isinstance(entries, list), "Artifact inventory is malformed.")
    count = 0
    for item in entries:
        path = Path(item["path"])
        _require(path.stat().st_size == item["size_bytes"], f"Size mismatch: {path}")
        _require(sha256_file(path) == item["sha256"], f"Hash mismatch: {path}")
        count += 1
    return count


def _audit_new_lane(manifest: dict[str, Any]) -> dict[str, Any] | None:
    lane = manifest["new_lane"]
    paths = _lane_paths(manifest, lane["id"])
    if not paths["directory"].exists():
        return None
    summary = _load_json(paths["summary"], "default IAS15 lane summary")
    _require(summary.get("status") == "COMPLETED", "Default IAS15 lane is incomplete.")
    _require(
        summary.get("manifest_sha256") == sha256_file(Path(manifest["paths"]["manifest"])),
        "Default IAS15 lane manifest mismatch.",
    )
    inventory_count = _verify_inventory(summary["artifact_inventory"])
    config = _expanded_configuration(manifest, lane)
    state = _audit_state_file(
        paths["state"],
        fields=PHYSICAL_STATE_FIELDS,
        body_names=manifest["common_configuration"]["body_names"],
        fingerprint=lane["configuration_fingerprint"],
        expected_samples=config["expected_samples"],
        cadence_years=config["record_every_years"],
    )
    return {"inventory_count": inventory_count, **state}


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "manifest 16")
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Run from the project root.")
    _require(manifest.get("frozen_before_new_integrations") is True, "Manifest 16 is not frozen.")
    head = _git("rev-parse", "HEAD")
    _require(
        _git("merge-base", "--is-ancestor", manifest["provenance"]["starting_commit"], head)
        == "",
        "Starting commit is not an ancestor.",
    )
    source_hashes = {}
    for label in ("manifest_13", "manifest_14", "manifest_15"):
        path = Path(manifest["source_artifacts"][label]["path"])
        actual = sha256_file(path)
        _require(actual == manifest["source_artifacts"][label]["sha256"], f"{label} changed.")
        source_hashes[label] = actual
    source_summary_path = Path(manifest["source_artifacts"]["manifest_15_summary"]["path"])
    _require(
        sha256_file(source_summary_path)
        == manifest["source_artifacts"]["manifest_15_summary"]["sha256"],
        "Manifest 15 summary changed.",
    )
    source_summary = _load_json(source_summary_path, "manifest 15 summary")
    source_report = manifest["source_artifacts"]["manifest_15_report"]
    _require(
        sha256_file(Path(source_report["path"])) == source_report["sha256"],
        "Manifest 15 report changed.",
    )
    _require(
        source_summary["primary_mechanism"] == source_summary["step3_diagnosis_status"] == "BLOCKED",
        "Manifest 15 historical BLOCKED result changed.",
    )
    _require(
        source_summary["ias15"]["tolerance_converged"] is False,
        "Manifest 15 IAS15 historical gate changed.",
    )
    step3d_manifest = Path(manifest["source_artifacts"]["manifest_15"]["path"])
    step3d_audit = audit_step3d(step3d_manifest)
    _require(step3d_audit["status"] == "PASS", "Step 3d audit failed.")
    inventory_count = 0
    for lane in source_summary["new_lanes"].values():
        inventory_count += _verify_inventory(lane["artifact_inventory"])
        inventory_count += _verify_inventory([lane["energy"]["timeseries"]])
    for lane in source_summary["reversibility"].values():
        inventory_count += _verify_inventory(lane["artifact_inventory"])
    inventory_count += _verify_inventory(source_summary["figures"])
    protected = []
    for relative, expected in manifest["protected_files"].items():
        actual = sha256_file(root / relative)
        _require(actual == expected, f"Protected file changed: {relative}")
        protected.append({"path": relative, "sha256": actual})
    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    runtime = _runtime_identity(rebound)
    expected_runtime = manifest["installed_runtime"]["rebound"]
    for key in ("rebound_version", "rebound_build", "rebound_githash"):
        _require(runtime[key] == expected_runtime[key], f"REBOUND identity changed: {key}")
    _require(
        runtime["shared_library_sha256"] == expected_runtime["shared_library_sha256"],
        "REBOUND shared library changed.",
    )
    _require(
        runtime["header_sha256"] == expected_runtime["header_sha256"],
        "REBOUND header changed.",
    )
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C baseline tag is not annotated.")
    _require(
        _git("rev-parse", tag + "^{commit}")
        == manifest["provenance"]["validated_c_baseline_commit"],
        "Compiled-C baseline tag target changed.",
    )
    reboundx = optional_import_module("reboundx")
    _require(reboundx is not None, "REBOUNDx is unavailable.")
    _require(reboundx.__version__ == manifest["installed_runtime"]["reboundx_version"], "REBOUNDx version changed.")
    default_probe = rebound.Simulation()
    default_probe.integrator = "ias15"
    _require(
        float(default_probe.ri_ias15.epsilon)
        == manifest["installed_runtime"]["ias15_default_epsilon"],
        "Installed IAS15 default epsilon changed.",
    )
    _require(
        default_probe.ri_ias15.adaptive_mode
        == manifest["installed_runtime"]["ias15_adaptive_mode"],
        "Installed IAS15 adaptive mode changed.",
    )
    return {
        "status": "PASS",
        "git_head": head,
        "git_dirty_after_preregistration": bool(_git("status", "--porcelain")),
        "manifest_sha256": sha256_file(manifest_path),
        "source_hashes": source_hashes,
        "historical_primary_mechanism": "BLOCKED",
        "historical_step3_status": "BLOCKED",
        "step3d_inventory_entries_verified": inventory_count,
        "protected_files": protected,
        "runtime": runtime,
        "reboundx_version": reboundx.__version__,
        "installed_default_epsilon": float(default_probe.ri_ias15.epsilon),
        "installed_adaptive_mode": default_probe.ri_ias15.adaptive_mode,
        "new_lane": _audit_new_lane(manifest),
    }


def benchmark(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 16")
    audit_payload = audit(manifest_path)
    definition = manifest["benchmark"]
    config = _expanded_configuration(manifest, definition)
    output = Path(manifest["paths"]["output_root"]) / definition["id"]
    _require(not output.exists(), f"Collision-safe benchmark exists: {output}")
    output.mkdir(parents=True)
    rebound, sim, backend, bodies, state0 = _build_simulation(manifest, config)
    initial = rebound_state_from_sim(sim, state0.masses)
    start = time.perf_counter()
    sim.integrate(config["duration_years"] * JULIAN_YEAR_S, exact_finish_time=1)
    elapsed = time.perf_counter() - start
    final = rebound_state_from_sim(sim, state0.masses)
    callback = backend.stats(sim)
    _require(float(sim.t) == config["duration_years"] * JULIAN_YEAR_S, "Benchmark missed endpoint.")
    _require(np.all(np.isfinite(final.positions)) and np.all(np.isfinite(final.velocities)), "Nonfinite benchmark state.")
    _require(int(callback["nonfinite_result_count"]) == 0, "Nonfinite benchmark callback.")
    state_path = output / "physical_state.csv"
    _write_csv_atomic(
        state_path,
        PHYSICAL_STATE_FIELDS,
        [
            *_state_rows(initial, bodies, sample_index=0, time_seconds=0.0, fingerprint=definition["configuration_fingerprint"]),
            *_state_rows(final, bodies, sample_index=1, time_seconds=float(sim.t), fingerprint=definition["configuration_fingerprint"]),
        ],
    )
    projected = elapsed * config["projected_duration_years"] / config["duration_years"]
    payload = {
        "schema_version": 1,
        "status": "PASSED" if projected <= config["projected_runtime_limit_seconds"] else "FAILED",
        "manifest_sha256": sha256_file(manifest_path),
        "configuration": config,
        "configuration_fingerprint": definition["configuration_fingerprint"],
        "command": sys.argv,
        "audit": audit_payload,
        "runtime_seconds": elapsed,
        "projected_10000y_runtime_seconds": projected,
        "projected_runtime_limit_seconds": config["projected_runtime_limit_seconds"],
        "accepted_steps": int(sim.steps_done),
        "rejected_steps": None,
        "rejected_steps_reason": manifest["step_accounting"]["rejected_steps_reason"],
        "callback_stats": callback,
        "settings": _settings(sim),
        "state": _artifact(state_path),
    }
    atomic_write_json(output / "summary.json", payload)
    _require(payload["status"] == "PASSED", "Default IAS15 projected runtime gate failed.")


def run_lane(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 16")
    audit_payload = audit(manifest_path)
    lane = manifest["new_lane"]
    config = _expanded_configuration(manifest, lane)
    benchmark_summary = _load_json(
        Path(manifest["paths"]["output_root"]) / manifest["benchmark"]["id"] / "summary.json",
        "default IAS15 benchmark",
    )
    _require(benchmark_summary.get("status") == "PASSED", "Benchmark gate did not pass.")
    _require(benchmark_summary.get("manifest_sha256") == sha256_file(manifest_path), "Benchmark manifest mismatch.")
    paths = _lane_paths(manifest, lane["id"])
    _require(not paths["directory"].exists(), f"Collision-safe lane exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True)
    events = paths["events"]
    _atomic_text(events, f"{dt.datetime.now(dt.timezone.utc).isoformat()} START command={' '.join(sys.argv)}\n")
    progress_temp = paths["progress"].with_name(paths["progress"].name + ".tmp")
    state_temp = paths["state"].with_name(paths["state"].name + ".tmp")
    rebound, sim, backend, bodies, state0 = _build_simulation(manifest, config)
    configure_rebound_simulationarchive(
        sim,
        paths["archive"],
        interval_s=config["archive_interval_years"] * JULIAN_YEAR_S,
        delete_existing=True,
    )
    energy0 = _state_energy(state0)["corrected"]
    angular0 = float(np.linalg.norm(total_angular_momentum_vector(state0)))
    targets = np.arange(config["expected_samples"], dtype=np.float64) * config["record_every_years"]
    dt_samples = []
    start = time.perf_counter()
    with progress_temp.open("w", newline="") as progress_handle, state_temp.open("w", newline="") as state_handle:
        progress_writer = csv.DictWriter(progress_handle, fieldnames=PROGRESS_FIELDS)
        state_writer = csv.DictWriter(state_handle, fieldnames=PHYSICAL_STATE_FIELDS)
        progress_writer.writeheader()
        state_writer.writeheader()
        for sample_index, target_years in enumerate(targets):
            sim.integrate(float(target_years) * JULIAN_YEAR_S, exact_finish_time=1)
            _require(float(sim.t) == float(target_years) * JULIAN_YEAR_S, "IAS15 lane missed endpoint.")
            state = rebound_state_from_sim(sim, state0.masses)
            _require(np.all(np.isfinite(state.positions)) and np.all(np.isfinite(state.velocities)), "Nonfinite IAS15 state.")
            progress_writer.writerow(
                _progress_row(
                    state,
                    sim=sim,
                    backend=backend,
                    fingerprint=lane["configuration_fingerprint"],
                    sample_index=sample_index,
                    target_years=float(target_years),
                    energy_reference=energy0,
                    angular_reference=angular0,
                    synchronization_mode=config["purpose"],
                )
            )
            state_writer.writerows(
                _state_rows(
                    state,
                    bodies,
                    sample_index=sample_index,
                    time_seconds=float(sim.t),
                    fingerprint=lane["configuration_fingerprint"],
                )
            )
            if sample_index:
                dt_samples.append(
                    {
                        "time_years": float(target_years),
                        "accepted_steps": int(sim.steps_done),
                        "proposed_dt_days": abs(float(sim.dt)) / DAY_S,
                        "last_accepted_dt_days": abs(float(sim.dt_last_done)) / DAY_S,
                    }
                )
            progress_handle.flush()
            state_handle.flush()
            os.fsync(progress_handle.fileno())
            os.fsync(state_handle.fileno())
            callback = backend.stats(sim)
            atomic_write_json(
                paths["status"],
                {
                    "state": "RUNNING",
                    "manifest_sha256": sha256_file(manifest_path),
                    "configuration_fingerprint": lane["configuration_fingerprint"],
                    "sample_index": sample_index,
                    "time_years": float(target_years),
                    "accepted_steps": int(sim.steps_done),
                    "callback_invocations": int(callback["callback_invocations"]),
                    "nonfinite_result_count": int(callback["nonfinite_result_count"]),
                },
            )
            print(f"[m0-ias15-phase] {sample_index + 1}/{len(targets)}", flush=True)
    os.replace(progress_temp, paths["progress"])
    os.replace(state_temp, paths["state"])
    elapsed = time.perf_counter() - start
    callback = backend.stats(sim)
    _require(int(callback["nonfinite_result_count"]) == 0, "Nonfinite callback result.")
    archive = rebound.Simulationarchive(str(paths["archive"]))
    _require(len(archive) == config["expected_archive_snapshots"], "Archive count mismatch.")
    state_audit = _audit_state_file(
        paths["state"],
        fields=PHYSICAL_STATE_FIELDS,
        body_names=bodies,
        fingerprint=lane["configuration_fingerprint"],
        expected_samples=config["expected_samples"],
        cadence_years=config["record_every_years"],
    )
    with events.open("a") as handle:
        handle.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} COMPLETE runtime_seconds={elapsed:.9f}\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_write_json(
        paths["status"],
        {
            "state": "COMPLETED",
            "manifest_sha256": sha256_file(manifest_path),
            "configuration_fingerprint": lane["configuration_fingerprint"],
            "samples": state_audit["samples"],
            "state_rows": state_audit["rows"],
            "accepted_steps": int(sim.steps_done),
            "callback_invocations": int(callback["callback_invocations"]),
            "nonfinite_result_count": int(callback["nonfinite_result_count"]),
        },
    )
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "lane_id": lane["id"],
        "manifest_sha256": sha256_file(manifest_path),
        "configuration": config,
        "configuration_fingerprint": lane["configuration_fingerprint"],
        "command": sys.argv,
        "audit": audit_payload,
        "settings": _settings(sim),
        "runtime_seconds": elapsed,
        "throughput_years_per_wall_second": config["duration_years"] / elapsed,
        "samples": state_audit["samples"],
        "state_rows": state_audit["rows"],
        "archive_snapshots": len(archive),
        "accepted_steps": int(sim.steps_done),
        "rejected_steps": None,
        "rejected_steps_reason": manifest["step_accounting"]["rejected_steps_reason"],
        "force_evaluations": int(callback["callback_invocations"]),
        "callback_stats": callback,
        "scientific_endpoint_timestep_samples": dt_samples,
        "hot_path_proof": backend.hot_path_proof(sim),
        "provenance": {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head": _git("rev-parse", "HEAD"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "runtime": _runtime_identity(rebound),
        },
    }
    summary["artifact_inventory"] = {
        key: _artifact(path)
        for key, path in paths.items()
        if key not in {"directory", "summary"}
    }
    atomic_write_json(paths["summary"], summary)


def _wrap_difference(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _growth(times: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    times = np.asarray(times[1:], dtype=np.float64)
    values = np.abs(np.asarray(values[1:], dtype=np.float64))
    cumulative = np.sqrt(np.cumsum(values**2) / np.arange(1, len(values) + 1))
    positive = cumulative > 0.0
    exponent = float(np.polyfit(np.log(times[positive]), np.log(cumulative[positive]), 1)[0]) if np.count_nonzero(positive) >= 3 else None
    model_errors = {}
    for power in (0.5, 1.5):
        basis = times**power
        amplitude = float(np.dot(values, basis) / np.dot(basis, basis))
        residual = values - amplitude * basis
        model_errors[f"t^{power}"] = {
            "least_squares_amplitude": amplitude,
            "normalized_rms_residual": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(values**2)), 1e-300)),
        }
    return {
        "cumulative_rms_loglog_exponent": exponent,
        "fixed_model_comparisons": model_errors,
        "preferred_fixed_model_descriptive_only": min(model_errors, key=lambda key: model_errors[key]["normalized_rms_residual"]),
    }


def pair_diagnostics(
    times: np.ndarray,
    left: Sequence[NBodyState],
    right: Sequence[NBodyState],
    body_names: Sequence[str],
) -> dict[str, Any]:
    _require(len(times) == len(left) == len(right), "Pair histories differ.")
    body_payload = {}
    all_position = []
    for body_index, body_name in enumerate(body_names[1:], start=1):
        rtn_position = []
        rtn_velocity = []
        elements = {name: [] for name in ("a_relative", "e_abs", *ANGLE_FIELDS)}
        for a, b in zip(left, right):
            reference_r = b.positions[body_index] - b.positions[0]
            reference_v = b.velocities[body_index] - b.velocities[0]
            radial = reference_r / np.linalg.norm(reference_r)
            normal = np.cross(reference_r, reference_v)
            normal /= np.linalg.norm(normal)
            transverse = np.cross(normal, radial)
            basis = np.vstack((radial, transverse, normal))
            delta_r = (a.positions[body_index] - a.positions[0]) - reference_r
            delta_v = (a.velocities[body_index] - a.velocities[0]) - reference_v
            rtn_position.append(basis @ delta_r)
            rtn_velocity.append(basis @ delta_v)
            ea = heliocentric_elements_for_state(a, body_names)[body_index - 1]
            eb = heliocentric_elements_for_state(b, body_names)[body_index - 1]
            elements["a_relative"].append(abs(ea.semi_major_axis_m - eb.semi_major_axis_m) / max(abs((ea.semi_major_axis_m + eb.semi_major_axis_m) / 2.0), 1.0))
            elements["e_abs"].append(abs(ea.eccentricity - eb.eccentricity))
            for name in ANGLE_FIELDS:
                elements[name].append(abs(_wrap_difference(getattr(ea, name), getattr(eb, name))))
        position = np.asarray(rtn_position)
        velocity = np.asarray(rtn_velocity)
        all_position.append(position)
        position_energy = np.sum(position**2)
        transverse_fraction = float(np.sum(position[:, 1] ** 2) / max(position_energy, 1e-300))
        components = {}
        for prefix, array in (("position_m", position), ("velocity_m_per_s", velocity)):
            components[prefix] = {}
            for index, axis in enumerate(("radial", "transverse", "normal")):
                absolute = np.abs(array[:, index])
                worst = int(np.argmax(absolute))
                components[prefix][axis] = {
                    "rms": float(np.sqrt(np.mean(array[:, index] ** 2))),
                    "maximum_abs": float(absolute[worst]),
                    "worst_epoch_years": float(times[worst]),
                    "growth": _growth(times, array[:, index]),
                }
        element_payload = {}
        for name, values in elements.items():
            array = np.asarray(values)
            worst = int(np.argmax(array))
            element_payload[name] = {
                "rms": float(np.sqrt(np.mean(array**2))),
                "maximum_abs": float(array[worst]),
                "worst_epoch_years": float(times[worst]),
                "growth": _growth(times, array),
            }
        phase_angle = max(
            element_payload["mean_anomaly_rad"]["maximum_abs"],
            element_payload["mean_longitude_rad"]["maximum_abs"],
        )
        orientation_angle = max(
            element_payload["inclination_rad"]["maximum_abs"],
            element_payload["longitude_ascending_node_rad"]["maximum_abs"],
            element_payload["argument_perihelion_rad"]["maximum_abs"],
        )
        body_payload[body_name] = {
            "rtn": components,
            "orbital_elements": element_payload,
            "transverse_position_variance_fraction": transverse_fraction,
            "phase_angle_over_orientation_angle": phase_angle / max(orientation_angle, 1e-300),
        }
    all_position_array = np.concatenate(all_position, axis=0)
    global_transverse_fraction = float(
        np.sum(all_position_array[:, 1] ** 2) / max(np.sum(all_position_array**2), 1e-300)
    )
    scaled = _matched_state_comparison(times, left, times, right, body_names)
    worst_body = scaled["worst_body"]
    return {
        "scaled_state": scaled,
        "global_transverse_position_variance_fraction": global_transverse_fraction,
        "worst_body_transverse_position_variance_fraction": body_payload.get(worst_body, {}).get("transverse_position_variance_fraction"),
        "bodies": body_payload,
    }


def _load_energy_and_angular(
    state_path: Path, progress_path: Path, body_names: Sequence[str]
) -> dict[str, Any]:
    times, states, groups = _read_physical_groups(state_path, body_names)
    with progress_path.open(newline="") as handle:
        progress = list(csv.DictReader(handle))
    _require(len(progress) == len(states), f"Progress/state count mismatch: {state_path}")
    decimal_values = []
    compensated_values = []
    reference_decimal = None
    reference_compensated = None
    reference_float = _state_energy(states[0])["corrected"]
    telemetry_max = 0.0
    with localcontext() as context:
        context.prec = 60
        for state, group, row in zip(states, groups, progress):
            decimal_value = decimal_energy(
                group,
                gravitational_constant=Decimal("6.67430e-11"),
                speed_of_light=Decimal("299792458"),
                coefficient_scale=Decimal(1),
            )["corrected"]
            compensated_value = compensated_energy(
                state.masses,
                state.positions,
                state.velocities,
                gravitational_constant=G_SI,
                speed_of_light=299_792_458.0,
                coefficient_scale=1.0,
            )["corrected"]
            if reference_decimal is None:
                reference_decimal = decimal_value
                reference_compensated = compensated_value
            decimal_drift = float((decimal_value - reference_decimal) / abs(reference_decimal))
            compensated_drift = (compensated_value - reference_compensated) / abs(reference_compensated)
            decimal_values.append(decimal_drift)
            compensated_values.append(compensated_drift)
            float_value = _state_energy(state)["corrected"]
            float_drift = (float_value - reference_float) / abs(reference_float)
            telemetry_max = max(
                telemetry_max,
                abs(float(row["corrected_energy_rel_change"]) - float_drift),
            )
    decimal_array = np.asarray(decimal_values)
    angular = np.asarray([float(row["angular_momentum_rel_change"]) for row in progress])
    energy_stats = _energy_statistics(times, decimal_array)
    blocks = _block_metrics(times, decimal_array, float(times[-1]))
    energy_stats["blocks"] = blocks
    energy_stats["same_sign_block_count"] = sum(
        math.copysign(1.0, block["fitted_slope_per_year"])
        == math.copysign(1.0, energy_stats["fitted_slope_per_year"])
        for block in blocks
    )
    return {
        "times": times,
        "states": states,
        "decimal_energy": decimal_array,
        "compensated_energy": np.asarray(compensated_values),
        "energy_statistics": energy_stats,
        "angular_history": angular,
        "angular_statistics": _energy_statistics(times, angular),
        "telemetry_reproduction_max_abs": telemetry_max,
    }


def _lane_data(
    manifest: dict[str, Any], manifest15: dict[str, Any], lane_id: str
) -> dict[str, Any]:
    body_names = manifest["common_configuration"]["body_names"]
    if lane_id == manifest["new_lane"]["id"]:
        root = Path(manifest["paths"]["output_root"])
        lane_definition = manifest["new_lane"]
    else:
        root = Path(manifest15["paths"]["output_root"])
        lane_definition = next(item for item in manifest15["lanes"] if item["id"] == lane_id)
    paths = {
        "state": root / lane_id / "physical_state.csv",
        "progress": root / lane_id / "progress.csv",
        "archive": root / lane_id / "simulationarchive.bin",
        "summary": root / lane_id / "summary.json",
    }
    summary = _load_json(paths["summary"], f"summary {lane_id}")
    config = {**manifest["common_configuration"], **lane_definition["configuration"]}
    _require(
        summary.get("configuration_fingerprint")
        == lane_definition["configuration_fingerprint"],
        f"Summary fingerprint mismatch: {lane_id}",
    )
    _verify_inventory(summary["artifact_inventory"])
    _audit_state_file(
        paths["state"],
        fields=PHYSICAL_STATE_FIELDS,
        body_names=body_names,
        fingerprint=lane_definition["configuration_fingerprint"],
        expected_samples=config["expected_samples"],
        cadence_years=config["record_every_years"],
    )
    with paths["progress"].open(newline="") as handle:
        progress_rows = list(csv.DictReader(handle))
    _require(len(progress_rows) == config["expected_samples"], f"Progress count mismatch: {lane_id}")
    _require(
        all(
            int(row["sample_index"]) == index
            and float(row["time_years"]) == index * config["record_every_years"]
            and row["configuration_fingerprint"] == lane_definition["configuration_fingerprint"]
            and int(row["nonfinite_result_count"]) == 0
            for index, row in enumerate(progress_rows)
        ),
        f"Progress schema/time/fingerprint/counter mismatch: {lane_id}",
    )
    energy = _load_energy_and_angular(paths["state"], paths["progress"], body_names)
    rebound = optional_import_module("rebound")
    archive = rebound.Simulationarchive(str(paths["archive"]))
    archive_dt = [abs(float(snapshot.dt_last_done)) / DAY_S for snapshot in archive if float(snapshot.t) > 0.0]
    archive_proposed = [abs(float(snapshot.dt)) / DAY_S for snapshot in archive if float(snapshot.t) > 0.0]
    final = archive[-1]
    _require(len(archive) == 11, f"Archive count mismatch: {lane_id}")
    _require(float(final.t) == config["duration_years"] * JULIAN_YEAR_S, f"Archive endpoint mismatch: {lane_id}")
    callback_count = int(summary["callback_stats"]["callback_invocations"])
    return {
        **energy,
        "lane_id": lane_id,
        "epsilon": float(summary["configuration"]["epsilon"]),
        "configuration_fingerprint": lane_definition["configuration_fingerprint"],
        "accepted_steps": int(final.steps_done),
        "rejected_steps": None,
        "rejected_steps_reason": manifest["step_accounting"]["rejected_steps_reason"],
        "force_evaluations": callback_count,
        "callback_invocations": callback_count,
        "nonfinite_result_count": int(summary["callback_stats"]["nonfinite_result_count"]),
        "iterations_max_exceeded": int(getattr(final.ri_ias15, "iterations_max_exceeded", getattr(final.ri_ias15, "_iterations_max_exceeded", 0))),
        "archive_snapshots": len(archive),
        "archive_timestep_distribution_days": _distribution(archive_dt),
        "archive_proposed_timestep_distribution_days": _distribution(archive_proposed),
        "runtime_seconds": summary["runtime_seconds"],
        "throughput_years_per_wall_second": 10000.0 / summary["runtime_seconds"],
        "artifact_inventory": summary["artifact_inventory"],
    }


def _envelope(lanes: Sequence[dict[str, Any]], body_names: Sequence[str]) -> dict[str, Any]:
    pair_arrays = []
    pair_names = []
    for left_index in range(len(lanes)):
        for right_index in range(left_index + 1, len(lanes)):
            left = lanes[left_index]
            right = lanes[right_index]
            position = np.stack([a.positions - b.positions for a, b in zip(left["states"], right["states"])]) / AU_M
            velocity = np.stack([a.velocities - b.velocities for a, b in zip(left["states"], right["states"])]) / (AU_M / JULIAN_YEAR_S)
            pair_arrays.append(np.concatenate((position, velocity), axis=2))
            pair_names.append(f"{left['lane_id']}_vs_{right['lane_id']}")
    envelope = np.max(np.abs(np.stack(pair_arrays)), axis=0)
    per_body = {
        name: float(np.sqrt(np.mean(envelope[:, index, :] ** 2)))
        for index, name in enumerate(body_names)
    }
    worst_body = max(per_body, key=per_body.get)
    worst_flat = int(np.argmax(envelope))
    sample, body, component = np.unravel_index(worst_flat, envelope.shape)
    return {
        "definition": "Componentwise maximum absolute difference across all three pairwise IAS15 comparisons, followed by the frozen scaled-state RMS.",
        "pair_names": pair_names,
        "global_scaled_rms": float(np.sqrt(np.mean(envelope**2))),
        "per_body_scaled_rms": per_body,
        "worst_body": worst_body,
        "worst_body_scaled_rms": per_body[worst_body],
        "worst_component": {
            "epoch_years": float(lanes[0]["times"][sample]),
            "body": body_names[body],
            "component": ("x", "y", "z", "vx", "vy", "vz")[component],
            "scaled_abs": float(envelope[sample, body, component]),
        },
    }


def _pair_energy_gate(left: dict[str, Any], right: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    difference = np.abs(left["decimal_energy"] - right["decimal_energy"])
    energy_stats_agree = all(
        abs(left["energy_statistics"][key] - right["energy_statistics"][key])
        <= max(
            thresholds["energy_stat_relative_difference_max"]
            * max(abs(left["energy_statistics"][key]), abs(right["energy_statistics"][key])),
            thresholds["energy_stat_absolute_floor"],
        )
        for key in ("max_abs", "rms", "p99_abs")
    )
    return {
        "energy_history_max_abs_difference": float(np.max(difference)),
        "energy_fitted_change_per_myr_abs_difference": abs(left["energy_statistics"]["fitted_change_per_myr"] - right["energy_statistics"]["fitted_change_per_myr"]),
        "energy_statistics_agree": energy_stats_agree,
        "angular_history_max_abs_difference": float(np.max(np.abs(left["angular_history"] - right["angular_history"]))),
    }


def _figures(
    manifest: dict[str, Any], lanes: Sequence[dict[str, Any]], pairs: dict[str, Any], envelope: dict[str, Any], whfast: dict[str, Any], causal: dict[str, Any]
) -> list[dict[str, Any]]:
    directory = Path(manifest["paths"]["figure_directory"])
    _require(not directory.exists(), f"Figure directory exists: {directory}")
    directory.mkdir(parents=True)
    output = []
    colors = ["#226F54", "#DA7C30", "#4169A1"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    pair_index = 0
    for left_index in range(len(lanes)):
        for right_index in range(left_index + 1, len(lanes)):
            left, right = lanes[left_index], lanes[right_index]
            pair_name = f"{left['lane_id']}_vs_{right['lane_id']}"
            body = pairs[pair_name]["scaled_state"]["worst_body"]
            body_index = manifest["common_configuration"]["body_names"].index(body)
            series = []
            for left_state, right_state in zip(left["states"], right["states"]):
                reference_r = right_state.positions[body_index] - right_state.positions[0]
                reference_v = right_state.velocities[body_index] - right_state.velocities[0]
                radial = reference_r / np.linalg.norm(reference_r)
                normal = np.cross(reference_r, reference_v)
                normal /= np.linalg.norm(normal)
                transverse = np.cross(normal, radial)
                delta = (
                    left_state.positions[body_index]
                    - left_state.positions[0]
                    - reference_r
                )
                series.append(np.vstack((radial, transverse, normal)) @ delta)
            series = np.abs(np.asarray(series))
            for axis_index, axis in enumerate(axes):
                axis.plot(
                    left["times"] / 1000.0,
                    series[:, axis_index],
                    color=colors[pair_index],
                    label=pair_name if axis_index == 0 else None,
                )
            pair_index += 1
    for axis, name in zip(axes, ("Radial", "Transverse", "Normal")):
        axis.set_ylabel(f"|{name}| (m)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (kyr)")
    axes[0].legend(fontsize=7)
    path = directory / "ias15_rtn_differences.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)
    output.append(_artifact(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = list(pairs)
    short_labels = ["default vs 1e-12", "default vs 1e-13", "1e-12 vs 1e-13"]
    phase_fraction = [pairs[name]["global_transverse_position_variance_fraction"] for name in labels]
    axes[0].bar(np.arange(len(labels)), phase_fraction, color=colors)
    axes[0].axhline(
        manifest["qualification_rules"]["phase"]["transverse_position_variance_fraction_min"],
        color="#222222",
        ls="--",
    )
    axes[0].set_xticks(np.arange(len(labels)), short_labels, rotation=15, ha="right")
    axes[0].set(
        ylabel="Transverse position variance fraction",
        title="RTN phase dominance",
        ylim=(0, 1.05),
    )
    axes[0].grid(axis="y", alpha=0.25)
    shape = []
    orientation = []
    phase_angle = []
    for name in labels:
        pair = pairs[name]
        body = pair["bodies"][pair["scaled_state"]["worst_body"]]["orbital_elements"]
        shape.append(max(body["a_relative"]["maximum_abs"], body["e_abs"]["maximum_abs"]))
        orientation.append(
            max(
                body["inclination_rad"]["maximum_abs"],
                body["longitude_ascending_node_rad"]["maximum_abs"],
                body["argument_perihelion_rad"]["maximum_abs"],
            )
        )
        phase_angle.append(
            max(
                body["mean_anomaly_rad"]["maximum_abs"],
                body["mean_longitude_rad"]["maximum_abs"],
            )
        )
    x = np.arange(len(labels))
    axes[1].plot(x, shape, marker="o", label="shape: max(a rel, e abs)", color="#226F54")
    axes[1].plot(x, orientation, marker="o", label="orientation angle", color="#DA7C30")
    axes[1].plot(x, phase_angle, marker="o", label="phase angle", color="#4169A1")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, short_labels, rotation=15, ha="right")
    axes[1].set(title="Worst-body orbital elements", ylabel="Maximum absolute difference")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)
    path = directory / "ias15_phase_vs_nonphase.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)
    output.append(_artifact(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    eps = np.asarray([lane["epsilon"] for lane in lanes])
    steps = np.asarray([lane["accepted_steps"] for lane in lanes])
    order = np.argsort(eps)[::-1]
    ax.plot(eps[order], steps[order], marker="o", color="#4169A1")
    ax.set_xscale("log")
    ax.set(xlabel="IAS15 epsilon", ylabel="Accepted steps", title="Tolerance and successful-step count")
    ax.grid(alpha=0.25)
    path = directory / "ias15_epsilon_step_counts.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)
    output.append(_artifact(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [envelope["global_scaled_rms"], whfast["global_scaled_rms"], 0.1 * whfast["global_scaled_rms"]]
    ax.bar(np.arange(3), values, color=["#226F54", "#DA7C30", "#777777"])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(3), ["IAS15 envelope", "WHFast 0.5 vs 0.25", "10% gate"])
    ax.set(ylabel="Global scaled-state RMS", title="Reference uncertainty versus timestep effect")
    ax.grid(axis="y", alpha=0.25)
    path = directory / "ias15_envelope_vs_whfast.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)
    output.append(_artifact(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for lane, color in zip(lanes, colors):
        axes[0].plot(lane["times"] / 1000.0, lane["decimal_energy"], color=color, label=f"eps={lane['epsilon']:.0e}")
    axes[0].set(xlabel="Time (kyr)", ylabel="Corrected-energy relative change", title="IAS15 corrected energy")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    labels = ["full signature", "current signature", "sync reduction"]
    values = [causal["full_systematic_signature"], causal["current_systematic_signature"], causal["any_sync_material_reduction"]]
    axes[1].bar(np.arange(3), [int(value) for value in values], color=["#226F54" if value else "#B44C43" for value in values])
    axes[1].set_xticks(np.arange(3), labels, rotation=15, ha="right")
    axes[1].set(ylim=(0, 1.15), ylabel="Frozen gate result", title="Manifest-13 causal evidence")
    axes[1].grid(axis="y", alpha=0.25)
    path = directory / "ias15_energy_and_causal_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)
    output.append(_artifact(path))
    return output


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['ias15_reference_status']}",
        "",
        f"Manifest-13 primary mechanism: **{payload['primary_mechanism']}**",
        "",
        f"Step 3 diagnosis: **{payload['step3_diagnosis_status']}**",
        "",
        "## Benchmark",
        "",
        f"- Observed 100-year runtime: {payload['benchmark']['runtime_seconds']:.9g} s.",
        f"- Projected 10-kyr runtime: {payload['benchmark']['projected_10000y_runtime_seconds']:.9g} s "
        f"(limit {payload['benchmark']['projected_runtime_limit_seconds']:.9g} s).",
        f"- Accepted steps: {payload['benchmark']['accepted_steps']}; nonfinite callbacks: "
        f"{payload['benchmark']['callback_stats']['nonfinite_result_count']}.",
        "",
        "## Qualification",
        "",
        f"- Integrity: `{payload['qualification_gates']['integrity']}`.",
        f"- Frozen corrected-energy gates: `{payload['qualification_gates']['energy']}`.",
        f"- Frozen nonphase element gates: `{payload['qualification_gates']['nonphase_elements']}`.",
        f"- Phase signature: `{payload['qualification_gates']['phase']}`.",
        f"- Angular-momentum conclusion unchanged: `{payload['qualification_gates']['angular_momentum']}`.",
        f"- IAS envelope / WHFast global discrepancy: `{payload['envelope_gate']['global_ratio']:.12g}` (limit `0.1`).",
        f"- IAS worst body: `{payload['ias15_uncertainty_envelope']['worst_body']}`; body ratio `{payload['envelope_gate']['worst_body_ratio']:.12g}` (limit `0.1`).",
        "",
        "## IAS15 Lanes",
        "",
        "| epsilon | accepted steps | rejected | callbacks | runtime s | median proposed dt d |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for lane in payload["ias15_lanes"].values():
        lines.append(
            f"| {lane['epsilon']:.0e} | {lane['accepted_steps']} | unavailable | "
            f"{lane['callback_invocations']} | {lane['runtime_seconds']:.6g} | "
            f"{lane['archive_proposed_timestep_distribution_days']['median']:.9g} |"
        )
    lines.extend(
        [
            "",
            "Rejected IAS15 attempt counts are unavailable: the exact installed REBOUND build retries internally, exposes only successful `steps_done`, and serializes no rejected-attempt counter. No REBOUND or callback instrumentation was added.",
            "",
            "| epsilon | endpoint energy | fitted slope / yr (95% CI) | max | RMS | p99 | same-sign blocks | angular max |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lane in payload["ias15_lanes"].values():
        energy = lane["energy_statistics"]
        lines.append(
            f"| {lane['epsilon']:.0e} | {energy['signed_endpoint_change']:.12g} | "
            f"{energy['fitted_slope_per_year']:.12g} "
            f"[{energy['ci95_low_per_year']:.12g}, {energy['ci95_high_per_year']:.12g}] | "
            f"{energy['max_abs']:.12g} | {energy['rms']:.12g} | {energy['p99_abs']:.12g} | "
            f"{energy['same_sign_block_count']}/10 | {lane['angular_statistics']['max_abs']:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Phase Evidence",
            "",
        ]
    )
    for name, pair in payload["pairwise_diagnostics"].items():
        lines.append(
            f"- `{name}`: global RMS `{pair['scaled_state']['global_scaled_rms']:.12g}`; "
            f"transverse variance fraction `{pair['global_transverse_position_variance_fraction']:.9g}`; "
            f"worst body `{pair['scaled_state']['worst_body']}` at `{pair['scaled_state']['worst_body_scaled_rms']:.12g}`."
        )
    lines.extend(
        [
            "",
            "The reported `t^0.5` and `t^1.5` residual comparisons are descriptive; neither model is forced as a gate.",
            "",
            "## Frozen Classification",
            "",
            "Only the manifest-13 raw pointwise IAS15 state-convergence condition is superseded. Every WHFast threshold and all other causal evidence are unchanged.",
            "",
            f"- All non-IAS causal evidence unchanged: {payload['causal_invariance']['all_other_evidence_unchanged']}.",
            f"- Full MEGNO/tangent and physical-only current-sync states exactly match over 10 kyr at both timesteps: {payload['causal_invariance']['full_vs_current_states_exactly_equal_both_timesteps']}.",
            f"- Synchronization material-reduction gate remains false independently of phase uncertainty: {payload['causal_invariance']['synchronization_material_reduction_still_fails_independently_of_reference_phase']}.",
            "",
            f"Smallest next action: {payload['smallest_next_action']}",
            "",
            "## Bounded Step 3e Prompt",
            "",
            payload.get("step3e_prompt") or "Not provided because the qualification/status condition did not pass.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 16")
    audit_payload = audit(manifest_path)
    benchmark_summary = _load_json(
        Path(manifest["paths"]["output_root"])
        / manifest["benchmark"]["id"]
        / "summary.json",
        "default IAS15 benchmark",
    )
    _require(benchmark_summary.get("status") == "PASSED", "Benchmark status changed.")
    _require(
        benchmark_summary.get("manifest_sha256") == sha256_file(manifest_path),
        "Benchmark manifest changed.",
    )
    manifest15 = _load_json(Path(manifest["source_artifacts"]["manifest_15"]["path"]), "manifest 15")
    summary15 = _load_json(Path(manifest["source_artifacts"]["manifest_15_summary"]["path"]), "manifest 15 summary")
    body_names = manifest["common_configuration"]["body_names"]
    lane_ids = [
        "m0_diag_phys_ias15_default_10k",
        "m0_diag_phys_ias15_eps1e12_10k",
        "m0_diag_phys_ias15_eps1e13_10k",
    ]
    lanes = [_lane_data(manifest, manifest15, lane_id) for lane_id in lane_ids]
    pairwise = {}
    pair_energy = {}
    for left_index in range(len(lanes)):
        for right_index in range(left_index + 1, len(lanes)):
            left, right = lanes[left_index], lanes[right_index]
            name = f"{left['lane_id']}_vs_{right['lane_id']}"
            pairwise[name] = pair_diagnostics(left["times"], left["states"], right["states"], body_names)
            pair_energy[name] = _pair_energy_gate(left, right, manifest["qualification_rules"]["frozen_ias15"])
    envelope = _envelope(lanes, body_names)
    full05 = _existing_states(manifest15, "0p5d", body_names, 10000.0)
    full025 = _existing_states(manifest15, "0p25d", body_names, 10000.0)
    whfast = _matched_state_comparison(full05[0], full05[1], full025[0], full025[1], body_names)
    preregistered_denominator = manifest["qualification_rules"]["envelope"]
    _require(
        whfast["global_scaled_rms"]
        == preregistered_denominator["whfast_denominator_global_scaled_rms_preregistered"],
        "Preregistered WHFast global denominator changed.",
    )
    _require(
        whfast["worst_body"] == preregistered_denominator["whfast_denominator_worst_body"]
        and whfast["worst_body_scaled_rms"]
        == preregistered_denominator["whfast_denominator_worst_body_scaled_rms_preregistered"],
        "Preregistered WHFast worst-body denominator changed.",
    )
    control_root = Path(manifest15["paths"]["output_root"])
    controls = {}
    for token in ("0p5d", "0p25d"):
        for mode in ("current_sync", "min_sync"):
            lane_id = f"m0_diag_phys_{mode}_{token}_100k"
            control_times, control_states, _ = _read_physical_groups(
                control_root / lane_id / "physical_state.csv", body_names
            )
            controls[(mode, token)] = (control_times[:101], control_states[:101])
    causal_state_comparisons = {
        "full_0p5_vs_full_0p25": whfast,
        "full_megno_vs_physical_current_0p5": _matched_state_comparison(
            full05[0], full05[1], controls[("current_sync", "0p5d")][0],
            controls[("current_sync", "0p5d")][1], body_names,
        ),
        "full_megno_vs_physical_current_0p25": _matched_state_comparison(
            full025[0], full025[1], controls[("current_sync", "0p25d")][0],
            controls[("current_sync", "0p25d")][1], body_names,
        ),
        "physical_current_vs_min_0p5": _matched_state_comparison(
            controls[("current_sync", "0p5d")][0],
            controls[("current_sync", "0p5d")][1],
            controls[("min_sync", "0p5d")][0],
            controls[("min_sync", "0p5d")][1],
            body_names,
        ),
        "physical_current_vs_min_0p25": _matched_state_comparison(
            controls[("current_sync", "0p25d")][0],
            controls[("current_sync", "0p25d")][1],
            controls[("min_sync", "0p25d")][0],
            controls[("min_sync", "0p25d")][1],
            body_names,
        ),
    }
    worst_body = envelope["worst_body"]
    envelope_gate = {
        "global_ratio": envelope["global_scaled_rms"] / whfast["global_scaled_rms"],
        "worst_body_ratio": envelope["per_body_scaled_rms"][worst_body] / whfast["per_body_scaled_rms"][worst_body],
        "denominator": "matched first 10 kyr of existing full 0.5-day versus 0.25-day WHFast lanes",
    }
    frozen = manifest["qualification_rules"]["frozen_ias15"]
    energy_pass = all(
        item["energy_history_max_abs_difference"] <= frozen["energy_history_max_abs_difference"]
        and item["energy_fitted_change_per_myr_abs_difference"] <= frozen["energy_fitted_change_per_myr_abs_difference"]
        and item["energy_statistics_agree"]
        for item in pair_energy.values()
    )
    nonphase_pass = all(
        body["orbital_elements"]["a_relative"]["maximum_abs"] <= frozen["semimajor_axis_history_relative_max"]
        and body["orbital_elements"]["e_abs"]["maximum_abs"] <= frozen["eccentricity_history_abs_max"]
        and body["orbital_elements"]["inclination_rad"]["maximum_abs"] <= manifest["qualification_rules"]["phase"]["orientation_angle_abs_rad_max"]
        and body["orbital_elements"]["longitude_ascending_node_rad"]["maximum_abs"] <= manifest["qualification_rules"]["phase"]["orientation_angle_abs_rad_max"]
        and body["orbital_elements"]["argument_perihelion_rad"]["maximum_abs"] <= manifest["qualification_rules"]["phase"]["orientation_angle_abs_rad_max"]
        for pair in pairwise.values()
        for body in pair["bodies"].values()
    )
    phase_pass = all(
        pair["global_transverse_position_variance_fraction"] >= manifest["qualification_rules"]["phase"]["transverse_position_variance_fraction_min"]
        and pair["worst_body_transverse_position_variance_fraction"] >= manifest["qualification_rules"]["phase"]["worst_body_transverse_position_variance_fraction_min"]
        and pair["bodies"][pair["scaled_state"]["worst_body"]]["phase_angle_over_orientation_angle"] >= manifest["qualification_rules"]["phase"]["phase_angle_over_orientation_min"]
        for pair in pairwise.values()
    )
    angular_pass = (
        all(lane["angular_statistics"]["max_abs"] <= manifest["qualification_rules"]["angular_momentum_rel_drift_max_per_run"] for lane in lanes)
        and all(item["angular_history_max_abs_difference"] <= manifest["qualification_rules"]["angular_history_pair_difference_max"] for item in pair_energy.values())
    )
    integrity_pass = all(
        lane["nonfinite_result_count"] == 0
        and lane["archive_snapshots"] == 11
        and lane["telemetry_reproduction_max_abs"] <= manifest["qualification_rules"]["telemetry_reproduction_max_abs"]
        and lane["iterations_max_exceeded"] == 0
        for lane in lanes
    )
    envelope_pass = (
        envelope_gate["global_ratio"]
        <= preregistered_denominator["global_over_whfast_0p5_vs_0p25_max"]
        and envelope_gate["worst_body_ratio"]
        <= preregistered_denominator["worst_ias_body_over_same_body_whfast_0p5_vs_0p25_max"]
    )
    qualified = integrity_pass and energy_pass and nonphase_pass and phase_pass and angular_pass and envelope_pass
    status = "IAS15_REFERENCE_QUALIFIED_AT_ROUNDOFF_PHASE_FLOOR" if qualified else "IAS15_REFERENCE_NOT_QUALIFIED"
    _require(status in REFERENCE_STATUSES, "Invalid IAS15 reference status.")
    evidence = dict(summary15["classification_evidence"])
    frozen_evidence = dict(evidence)
    evidence["ias15_tolerance_converged"] = qualified
    primary, step3_status = classify_mechanism(evidence)
    if not qualified:
        primary, step3_status = "BLOCKED", "BLOCKED"
    figures = _figures(manifest, lanes, pairwise, envelope, whfast, evidence)
    lane_output = {}
    for lane in lanes:
        lane_output[lane["lane_id"]] = {
            key: lane[key]
            for key in (
                "epsilon", "accepted_steps", "rejected_steps", "rejected_steps_reason",
                "force_evaluations", "callback_invocations", "nonfinite_result_count",
                "iterations_max_exceeded", "archive_snapshots", "archive_timestep_distribution_days",
                "archive_proposed_timestep_distribution_days", "runtime_seconds",
                "throughput_years_per_wall_second",
                "energy_statistics", "angular_statistics", "telemetry_reproduction_max_abs",
                "artifact_inventory",
            )
        }
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "ias15_reference_status": status,
        "primary_mechanism": primary,
        "step3_diagnosis_status": step3_status,
        "historical_results_unchanged": {
            "manifest_13": {"primary_mechanism": "BLOCKED", "step3_diagnosis_status": "BLOCKED"},
            "manifest_15": {"primary_mechanism": "BLOCKED", "step3_diagnosis_status": "BLOCKED"},
        },
        "audit": audit_payload,
        "source_audit": manifest["installed_runtime"]["source_audit"],
        "benchmark": {
            key: benchmark_summary[key]
            for key in (
                "status",
                "runtime_seconds",
                "projected_10000y_runtime_seconds",
                "projected_runtime_limit_seconds",
                "accepted_steps",
                "rejected_steps",
                "rejected_steps_reason",
                "callback_stats",
            )
        },
        "qualification_gates": {
            "integrity": integrity_pass,
            "energy": energy_pass,
            "nonphase_elements": nonphase_pass,
            "phase": phase_pass,
            "angular_momentum": angular_pass,
            "envelope": envelope_pass,
        },
        "envelope_gate": envelope_gate,
        "ias15_uncertainty_envelope": envelope,
        "whfast_0p5_vs_0p25_10k": whfast,
        "whfast_causal_state_comparisons_10k": causal_state_comparisons,
        "ias15_lanes": lane_output,
        "pairwise_energy_and_angular": pair_energy,
        "pairwise_diagnostics": pairwise,
        "historical_raw_pointwise_gate_diagnostic_only": summary15["ias15"],
        "classification_evidence": evidence,
        "causal_invariance": {
            "only_changed_evidence_key": "ias15_tolerance_converged",
            "all_other_evidence_unchanged": all(
                evidence[key] == value
                for key, value in frozen_evidence.items()
                if key != "ias15_tolerance_converged"
            ),
            "full_vs_current_states_exactly_equal_both_timesteps": (
                causal_state_comparisons["full_megno_vs_physical_current_0p5"]["global_scaled_rms"] == 0.0
                and causal_state_comparisons["full_megno_vs_physical_current_0p25"]["global_scaled_rms"] == 0.0
            ),
            "synchronization_material_reduction_still_fails_independently_of_reference_phase": not summary15["classification_evidence"]["any_sync_material_reduction"],
            "reason": "The 0.5-day min-sync energy max/RMS/p99 ratios exceed the unchanged 0.25 material-reduction limit, so IAS phase uncertainty cannot flip that gate.",
        },
        "classification_supersession": "Only manifest-13 IAS15 raw pointwise state convergence is replaced by manifest-16 phase/roundoff qualification; every WHFast threshold is unchanged.",
        "figures": figures,
        "smallest_next_action": (
            "Proceed only through a separately preregistered Step 3e: one 0.125-day 1 Myr lane evaluated against the manifest-16 IAS15 roundoff/phase envelope."
            if qualified and step3_status == "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
            else "Remain blocked and isolate the first failed IAS15 qualification gate before any new WHFast trajectory."
        ),
        "step3e_prompt_may_be_provided": qualified and step3_status == "STEP3_NUMERICAL_FLOOR_CHARACTERIZED",
        "step3e_prompt": (
            "Proceed with Step 3e only. Preregister a versioned manifest before integration and run exactly one fresh full-M0 compiled-C tangent lane: 0.125-day WHFast, 1,000,000 years, 100-year scientific cadence, 100,000-year archive cadence, MEGNO seed 12345, identical DE431 state and unchanged validated equations. Compare the existing 0.25-day lane with the new 0.125-day lane using every frozen Step-3 physical, orbital, tangent, MEGNO, LCN, corrected-energy, angular-momentum, schema, fingerprint, callback, and artifact criterion. Treat manifest 16's three-lane IAS15 phase/roundoff envelope as a preregistered uncertainty floor; do not weaken a WHFast threshold or require monotonic IAS15 raw phase convergence. Run no other WHFast trajectory, no Stage 4, and no 10 Myr integration. Emit one convergence status, compact reports, focused checks, and the smallest evidence-based next action."
            if qualified and step3_status == "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
            else None
        ),
        "no_new_whfast_run": True,
        "no_stage4_or_10myr": True,
    }
    report_json = Path(manifest["paths"]["report_json"])
    atomic_write_json(report_json, payload)
    _atomic_text(Path(manifest["paths"]["report_markdown"]), _markdown(payload))
    print(f"[m0-ias15-phase] reference_status={status}")
    print(f"[m0-ias15-phase] primary_mechanism={primary}")
    print(f"[m0-ias15-phase] step3_status={step3_status}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Qualify the M0 IAS15 phase/roundoff reference.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("benchmark")
    subparsers.add_parser("run-lane")
    subparsers.add_parser("analyze")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit(args.manifest), indent=2))
    elif args.command == "benchmark":
        benchmark(args.manifest)
    elif args.command == "run-lane":
        run_lane(args.manifest)
    else:
        analyze(args.manifest)


if __name__ == "__main__":
    main()
