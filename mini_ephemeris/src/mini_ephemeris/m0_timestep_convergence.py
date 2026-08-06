from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import load_c_backend
from .long_term_stability_cli import (
    load_rebound_archive_snapshot,
    optional_import_module,
    parse_start_datetime,
    stability_body_list,
)
from .m0_telemetry import STATE_SAMPLE_FIELDS, read_state_samples
from .nbody import NBodyState
from .orbital_elements import (
    ARCSEC_PER_RAD,
    AU_M,
    DAY_S,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
)
from .rebound_gr_tangent_backend_cli import (
    PROGRESS_FIELDS,
    _config_payload,
    _record_targets,
    canonical_hash,
    initial_condition_hash,
    output_paths,
    sha256_file,
)


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json"
)
INNER_BODIES = (
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
)
FINAL_STATUSES = {"M0_1DAY_CONVERGED", "M0_1DAY_NOT_CONVERGED", "BLOCKED"}


class ConvergenceError(RuntimeError):
    pass


@dataclass
class RunData:
    run_id: str
    step_days: float
    body_names: tuple[str, ...]
    times: np.ndarray
    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    variation_positions: np.ndarray
    variation_velocities: np.ndarray
    progress: dict[str, np.ndarray]
    summary: dict[str, Any]
    integrity: dict[str, Any]
    inventory: list[dict[str, Any]]
    elements: dict[str, dict[str, np.ndarray]] | None = None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise ConvergenceError(f"Unreadable {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConvergenceError(f"Invalid {label} {path}: expected a JSON object.")
    return payload


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ConvergenceError(f"Git command failed: git {' '.join(args)}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConvergenceError(message)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConvergenceError(f"Malformed numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ConvergenceError(f"Nonfinite numeric value for {label}: {value!r}")
    return result


def _manifest_configuration(manifest: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    shared = manifest["shared_configuration"]
    return {
        "runner_schema_version": shared["runner_schema_version"],
        "state_sample_schema_version": shared["state_sample_schema_version"],
        "model_id": manifest["model_id"],
        "backend": shared["backend"],
        "kernel_sha256": shared["kernel_sha256"],
        "initial_conditions_sha256": shared["initial_conditions_sha256"],
        "artifact_sha256": shared["c_artifact_sha256"],
        "c_source_sha256": shared["c_source_sha256"],
        "start_date": shared["start_date"],
        "model_scope": shared["model_scope"],
        "body_names": shared["body_names"],
        "duration_years": float(shared["duration_years"]),
        "step_days": float(run["step_days"]),
        "record_every_years": float(shared["record_every_years"]),
        "archive_interval_years": float(shared["archive_interval_years"]),
        "megno_seed": int(shared["megno_seed"]),
        "gr_scale": float(shared["gr_scale"]),
        "include_central_response": bool(shared["include_central_response"]),
    }


def _inventory_entry(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing decisive artifact: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _preflight_payload(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(manifest["paths"]["project_root"])
    output_root = Path(manifest["paths"]["output_root"])
    _require(Path.cwd().resolve() == project_root.resolve(), "Preflight must run at project root.")
    _require(manifest.get("frozen_before_decisive_runs") is True, "Manifest is not frozen.")

    head = _git("rev-parse", "HEAD")
    tag_kind = _git("cat-file", "-t", manifest["provenance"]["validated_c_annotated_tag"])
    tag_commit = _git(
        "rev-parse", manifest["provenance"]["validated_c_annotated_tag"] + "^{commit}"
    )
    _require(tag_kind == "tag", "Validated C milestone is not an annotated tag.")
    _require(
        tag_commit == manifest["provenance"]["validated_c_baseline_commit"],
        "Validated C milestone resolves to the wrong commit.",
    )
    _require(
        head == manifest["provenance"]["step2_commit"],
        "Step 3 must launch from the closed Step 2 commit.",
    )

    protected = []
    for relative, expected_hash in manifest["protected_files"].items():
        path = project_root / relative
        actual_hash = sha256_file(path)
        _require(actual_hash == expected_hash, f"Protected file hash mismatch: {path}")
        protected.append(
            {"path": str(path), "expected_sha256": expected_hash, "actual_sha256": actual_hash}
        )

    endpoint = manifest["endpoint_semantics"]
    targets = _record_targets(
        float(endpoint["duration_years"]), float(endpoint["scientific_cadence_years"])
    )
    _require(
        len(targets) == endpoint["expected_scientific_samples_per_run"],
        "Runner endpoint semantics do not match preregistration.",
    )
    _require(targets[0] == 0.0 and targets[-1] == float(endpoint["duration_years"]), "Endpoint mismatch.")
    _require(
        len(targets) * endpoint["real_body_count"] == endpoint["expected_state_rows_per_run"],
        "Expected state-row count mismatch.",
    )

    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        _require(
            os.path.commonpath((str(output_root.resolve()), str(run_dir.resolve())))
            == str(output_root.resolve()),
            f"Run output escapes convergence root: {run_dir}",
        )
        _require(not run_dir.exists(), f"Decisive output directory already exists: {run_dir}")

    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    shared = manifest["shared_configuration"]
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    start_date = parse_start_datetime(shared["start_date"])
    kernel = Path(manifest["paths"]["kernel"])
    state0 = initial_state_solar_system_barycentric(
        start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=str(kernel)),
    )
    backend = load_c_backend()
    actual_hashes = {
        "kernel_sha256": sha256_file(kernel),
        "initial_conditions_sha256": initial_condition_hash(state0, bodies),
        "artifact_sha256": backend.build_metadata["artifact_sha256"],
        "c_source_sha256": backend.build_metadata["source_sha256"],
    }
    _require(actual_hashes["kernel_sha256"] == shared["kernel_sha256"], "Kernel hash mismatch.")
    _require(
        actual_hashes["initial_conditions_sha256"] == shared["initial_conditions_sha256"],
        "Initial-condition hash mismatch.",
    )
    _require(actual_hashes["artifact_sha256"] == shared["c_artifact_sha256"], "C artifact hash mismatch.")
    _require(actual_hashes["c_source_sha256"] == shared["c_source_sha256"], "C source hash mismatch.")

    fingerprints = []
    for run in manifest["runs"]:
        expected_config = _manifest_configuration(manifest, run)
        runner_args = argparse.Namespace(
            model_id=manifest["model_id"],
            gr_tangent_backend=shared["backend"],
            start_date=start_date,
            model_scope=shared["model_scope"],
            duration_years=float(shared["duration_years"]),
            step_days=float(run["step_days"]),
            record_every_years=float(shared["record_every_years"]),
            archive_interval_years=float(shared["archive_interval_years"]),
            megno_seed=int(shared["megno_seed"]),
            gr_scale=float(shared["gr_scale"]),
            no_central_response=not bool(shared["include_central_response"]),
        )
        actual_config = _config_payload(runner_args, list(bodies), actual_hashes)
        _require(actual_config == expected_config, f"Runner configuration mismatch for {run['id']}.")
        actual_fingerprint = canonical_hash(actual_config)
        _require(
            actual_fingerprint == run["configuration_fingerprint"],
            f"Configuration fingerprint mismatch for {run['id']}.",
        )
        fingerprints.append(
            {"run_id": run["id"], "configuration_fingerprint": actual_fingerprint}
        )

    step2_root = project_root / "output/stability/m0_production_readiness_v1/restart_fresh_process"
    step2_summary_path = step2_root / "gr_tangent_summary_m0_restart_smoke.json"
    step2_summary = _load_json(step2_summary_path, "Step 2 restart summary")
    _require(step2_summary.get("complete") is True, "Step 2 restart summary is incomplete.")
    _require(step2_summary.get("restart", {}).get("resumed") is True, "Step 2 restart was not fresh-process resumed.")
    _require(
        step2_summary.get("restart", {}).get("callbacks_increased_after_reattachment") is True,
        "Step 2 restart did not prove callback reattachment.",
    )
    step2_rows = read_state_samples(
        step2_root / "gr_tangent_state_m0_restart_smoke.csv",
        body_names=step2_summary["configuration"]["body_names"],
        configuration_fingerprint=step2_summary["configuration_fingerprint"],
    )
    _require(len(step2_rows) == 50, "Step 2 restart state evidence has the wrong row count.")
    loaded = load_rebound_archive_snapshot(
        rebound, step2_root / "gr_tangent_archive_m0_restart_smoke.bin"
    )
    _require(int(loaded.N_real) == 10 and int(loaded.N_var) > 0, "Step 2 archive layout is incompatible.")
    float(loaded.megno())
    float(loaded.lyapunov())
    backend.attach(loaded, coefficient_scale=1.0, include_central_response=True)
    hot_path = backend.hot_path_proof(loaded)
    _require(hot_path.get("addresses_match") is True, "Compiled-C callback pointer is not directly attached.")
    before = int(backend.stats(loaded)["callback_invocations"])
    loaded.integrate(float(loaded.t) + DAY_S, exact_finish_time=1)
    after_stats = backend.stats(loaded)
    _require(int(after_stats["callback_invocations"]) > before, "Compiled-C callback did not execute.")
    _require(int(after_stats["nonfinite_result_count"]) == 0, "Compiled-C smoke produced nonfinite results.")

    disk = shutil.disk_usage(project_root)
    _require(
        disk.free >= manifest["estimates"]["disk_bytes"]["required_with_safety_factor"],
        "Insufficient disk space for decisive runs.",
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_head": head,
        "git_dirty": bool(_git("status", "--porcelain")),
        "tag_kind": tag_kind,
        "tag_resolved_commit": tag_commit,
        "protected_files": protected,
        "runner_endpoint_semantics": {
            "sample_count": len(targets),
            "first_time_years": targets[0],
            "last_time_years": targets[-1],
            "state_row_count": len(targets) * len(bodies),
        },
        "configuration_fingerprints": fingerprints,
        "actual_hashes": actual_hashes,
        "step2_restart_evidence": {
            "summary": str(step2_summary_path),
            "state_rows": len(step2_rows),
            "fresh_process_resumed": True,
            "callbacks_increased_after_reattachment": True,
        },
        "compiled_c_smoke": {"hot_path_proof": hot_path, "callback_stats": after_stats},
        "disk": {
            "available_bytes": disk.free,
            "estimated_experiment_bytes": manifest["estimates"]["disk_bytes"]["complete_experiment"],
            "required_with_safety_factor_bytes": manifest["estimates"]["disk_bytes"]["required_with_safety_factor"],
        },
    }


def preflight(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "convergence manifest")
    payload = _preflight_payload(manifest_path, manifest)
    output = Path(manifest["paths"]["output_root"]) / "preflight/m0_timestep_convergence_preflight.json"
    _atomic_write_json(output, payload)
    print(f"[m0-convergence] preflight PASS: {output}")
    return 0


def _load_progress(
    path: Path,
    *,
    expected_times: np.ndarray,
    fingerprint: str,
    model_id: str,
) -> dict[str, np.ndarray]:
    fields = (
        "megno",
        "lcn_1_per_year",
        "newtonian_energy_component_rel_change",
        "angular_momentum_rel_drift",
        "corrected_energy_rel_change",
        "angular_momentum_x_kg_m2_s",
        "angular_momentum_y_kg_m2_s",
        "angular_momentum_z_kg_m2_s",
        "angular_momentum_norm_kg_m2_s",
    )
    values = {name: np.empty(len(expected_times), dtype=np.float64) for name in fields}
    callbacks = np.empty(len(expected_times), dtype=np.int64)
    nonfinite = np.empty(len(expected_times), dtype=np.int64)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == PROGRESS_FIELDS, f"Progress schema mismatch: {path}")
        count = 0
        for index, row in enumerate(reader):
            _require(index < len(expected_times), f"Too many progress rows: {path}")
            expected_years = float(expected_times[index])
            expected_seconds = expected_years * JULIAN_YEAR_S
            _require(float(row["target_time_years"]) == expected_years, f"Target-time mismatch at progress row {index}.")
            _require(float(row["time_years"]) == expected_years, f"Exact year mismatch at progress row {index}.")
            _require(float(row["time_seconds"]) == expected_seconds, f"Exact seconds mismatch at progress row {index}.")
            _require(row["configuration_fingerprint"] == fingerprint, f"Progress fingerprint mismatch at row {index}.")
            _require(row["model_id"] == model_id, f"Progress model mismatch at row {index}.")
            _require(row["runner_schema_version"] == "2", f"Progress runner schema mismatch at row {index}.")
            _require(row["state_sample_schema_version"] == "1", f"Progress state schema mismatch at row {index}.")
            for name in fields:
                values[name][index] = _finite_float(row[name], f"{path}:{index}:{name}")
            callbacks[index] = int(row["callback_invocations"])
            nonfinite[index] = int(row["nonfinite_result_count"])
            count += 1
    _require(count == len(expected_times), f"Progress row count mismatch: {path}")
    _require(np.all(np.diff(callbacks) >= 0), f"Callback counter is not monotonic: {path}")
    _require(np.all(nonfinite == 0), f"Nonfinite callback result recorded: {path}")
    values["callback_invocations"] = callbacks
    values["nonfinite_result_count"] = nonfinite
    return values


def _load_state(
    path: Path,
    *,
    expected_times: np.ndarray,
    body_names: Sequence[str],
    fingerprint: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = len(expected_times)
    body_count = len(body_names)
    masses = np.empty(body_count, dtype=np.float64)
    positions = np.empty((sample_count, body_count, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    variation_positions = np.empty_like(positions)
    variation_velocities = np.empty_like(positions)
    state_names = ("x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")
    variation_names = (
        "variation_x_m",
        "variation_y_m",
        "variation_z_m",
        "variation_vx_m_per_s",
        "variation_vy_m_per_s",
        "variation_vz_m_per_s",
    )
    expected_rows = sample_count * body_count
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == STATE_SAMPLE_FIELDS, f"State schema mismatch: {path}")
        count = 0
        for linear_index, row in enumerate(reader):
            _require(linear_index < expected_rows, f"Too many state rows: {path}")
            sample_index, body_index = divmod(linear_index, body_count)
            expected_years = float(expected_times[sample_index])
            _require(int(row["sample_index"]) == sample_index, f"State sample index mismatch at row {linear_index}.")
            _require(int(row["body_index"]) == body_index, f"State body index mismatch at row {linear_index}.")
            _require(row["body_name"] == body_names[body_index], f"State body name mismatch at row {linear_index}.")
            _require(float(row["time_years"]) == expected_years, f"State exact year mismatch at row {linear_index}.")
            _require(float(row["time_seconds"]) == expected_years * JULIAN_YEAR_S, f"State exact seconds mismatch at row {linear_index}.")
            _require(row["configuration_fingerprint"] == fingerprint, f"State fingerprint mismatch at row {linear_index}.")
            _require(row["state_sample_schema_version"] == "1", f"State schema version mismatch at row {linear_index}.")
            _require(row["variation_config_index"] == "0", f"Variation config mismatch at row {linear_index}.")
            mass = _finite_float(row["mass_kg"], f"{path}:{linear_index}:mass")
            _require(mass > 0.0, f"Nonpositive mass at state row {linear_index}.")
            if sample_index == 0:
                masses[body_index] = mass
            else:
                _require(mass == masses[body_index], f"Mass changed at state row {linear_index}.")
            state_values = np.array(
                [_finite_float(row[name], f"{path}:{linear_index}:{name}") for name in state_names]
            )
            variation_values = np.array(
                [_finite_float(row[name], f"{path}:{linear_index}:{name}") for name in variation_names]
            )
            positions[sample_index, body_index] = state_values[:3]
            velocities[sample_index, body_index] = state_values[3:]
            variation_positions[sample_index, body_index] = variation_values[:3]
            variation_velocities[sample_index, body_index] = variation_values[3:]
            count += 1
    _require(count == expected_rows, f"State row count mismatch: {path}")
    return masses, positions, velocities, variation_positions, variation_velocities


def _load_run(manifest_path: Path, manifest: dict[str, Any], run: dict[str, Any]) -> RunData:
    run_dir = Path(run["output_dir"])
    paths = output_paths(run_dir, run["id"], None)
    required = [paths[name] for name in ("progress", "state", "status", "summary", "restart", "archive")]
    required.append(Path(run["log_path"]))
    inventory = [_inventory_entry(path) for path in required]
    summary = _load_json(paths["summary"], "run summary")
    status = _load_json(paths["status"], "run status")
    restart = _load_json(paths["restart"], "run restart sidecar")
    fingerprint = run["configuration_fingerprint"]
    expected_config = _manifest_configuration(manifest, run)
    _require(summary.get("status") == "COMPLETED" and summary.get("complete") is True, f"Run is incomplete: {run['id']}")
    _require(summary.get("schema_version") == 2, f"Runner schema mismatch: {run['id']}")
    _require(summary.get("state_sample_schema_version") == 1, f"State schema mismatch: {run['id']}")
    _require(summary.get("configuration_fingerprint") == fingerprint, f"Summary fingerprint mismatch: {run['id']}")
    _require(summary.get("configuration") == expected_config, f"Summary configuration mismatch: {run['id']}")
    _require(status.get("state") == "COMPLETED" and float(status.get("time_years")) == 1_000_000.0, f"Status sidecar mismatch: {run['id']}")
    _require(restart.get("state") == "complete", f"Restart sidecar is not complete: {run['id']}")
    _require(restart.get("configuration_fingerprint") == fingerprint, f"Restart fingerprint mismatch: {run['id']}")
    _require(restart.get("checkpoint_state_row_count") == 100010, f"Restart state count mismatch: {run['id']}")
    _require(summary.get("provenance", {}).get("manifest_sha256") == sha256_file(manifest_path), f"Manifest hash mismatch: {run['id']}")
    hot_path = summary.get("hot_path_proof", {})
    _require(hot_path.get("addresses_match") is True, f"C hot path not directly attached: {run['id']}")
    _require(hot_path.get("python_callback_in_force_path") is False, f"Python callback entered force path: {run['id']}")

    endpoint = manifest["endpoint_semantics"]
    expected_times = np.asarray(
        _record_targets(float(endpoint["duration_years"]), float(endpoint["scientific_cadence_years"])),
        dtype=np.float64,
    )
    progress = _load_progress(
        paths["progress"],
        expected_times=expected_times,
        fingerprint=fingerprint,
        model_id=manifest["model_id"],
    )
    masses, positions, velocities, variation_positions, variation_velocities = _load_state(
        paths["state"],
        expected_times=expected_times,
        body_names=manifest["shared_configuration"]["body_names"],
        fingerprint=fingerprint,
    )
    final_stats = summary["diagnostics"]["callback_stats"]
    _require(int(final_stats["nonfinite_result_count"]) == 0, f"Final nonfinite counter is nonzero: {run['id']}")
    _require(int(final_stats["callback_invocations"]) == int(progress["callback_invocations"][-1]), f"Final callback count mismatch: {run['id']}")
    _require(summary["diagnostics"]["rows_written_total"] == len(expected_times), f"Summary sample count mismatch: {run['id']}")
    _require(summary["diagnostics"]["state_rows_written_total"] == len(expected_times) * len(masses), f"Summary state count mismatch: {run['id']}")

    rebound = optional_import_module("rebound")
    archive_sim = load_rebound_archive_snapshot(rebound, paths["archive"])
    _require(float(archive_sim.t) == 1_000_000.0 * JULIAN_YEAR_S, f"Archive endpoint mismatch: {run['id']}")
    _require(int(archive_sim.N_real) == len(masses) and int(archive_sim.N_var) > 0, f"Archive layout mismatch: {run['id']}")
    float(archive_sim.megno())
    float(archive_sim.lyapunov())
    integrity = {
        "passed": True,
        "step_days": float(run["step_days"]),
        "scientific_samples": len(expected_times),
        "state_rows": len(expected_times) * len(masses),
        "exact_times": True,
        "finite_physical_state": True,
        "finite_tangent_state": True,
        "configuration_fingerprint": fingerprint,
        "callback_invocations": int(final_stats["callback_invocations"]),
        "nonfinite_result_count": int(final_stats["nonfinite_result_count"]),
        "archive_final_time_years": float(archive_sim.t) / JULIAN_YEAR_S,
        "runtime_seconds": float(summary["diagnostics"]["runtime_seconds_this_process"]),
        "throughput_years_per_second": 1_000_000.0 / float(summary["diagnostics"]["runtime_seconds_this_process"]),
    }
    return RunData(
        run_id=run["id"],
        step_days=float(run["step_days"]),
        body_names=tuple(manifest["shared_configuration"]["body_names"]),
        times=expected_times,
        masses=masses,
        positions=positions,
        velocities=velocities,
        variation_positions=variation_positions,
        variation_velocities=variation_velocities,
        progress=progress,
        summary=summary,
        integrity=integrity,
        inventory=inventory,
    )


def _pair_physical(left: RunData, right: RunData) -> dict[str, Any]:
    velocity_scale = AU_M / JULIAN_YEAR_S
    position_delta = (left.positions - right.positions) / AU_M
    velocity_delta = (left.velocities - right.velocities) / velocity_scale
    squared = np.sum(position_delta**2, axis=2) + np.sum(velocity_delta**2, axis=2)
    sample_body_rms = np.sqrt(squared / 6.0)
    global_sample_rms = np.sqrt(np.mean(squared, axis=1) / 6.0)
    per_body = {}
    for body_index, body_name in enumerate(left.body_names):
        worst = int(np.argmax(sample_body_rms[:, body_index]))
        per_body[body_name] = {
            "rms": float(np.sqrt(np.mean(squared[:, body_index]) / 6.0)),
            "max_sample_rms": float(sample_body_rms[worst, body_index]),
            "worst_epoch_years": float(left.times[worst]),
        }
    worst_sample = int(np.argmax(global_sample_rms))
    return {
        "global_scaled_rms": float(np.sqrt(np.mean(squared) / 6.0)),
        "max_global_sample_rms": float(global_sample_rms[worst_sample]),
        "worst_epoch_years": float(left.times[worst_sample]),
        "per_body": per_body,
    }


def _scaled_tangent(data: RunData) -> np.ndarray:
    velocity_scale = AU_M / JULIAN_YEAR_S
    return np.concatenate(
        (data.variation_positions / AU_M, data.variation_velocities / velocity_scale),
        axis=2,
    ).reshape(len(data.times), -1)


def _pair_tangent(left: RunData, right: RunData) -> dict[str, Any]:
    left_vector = _scaled_tangent(left)
    right_vector = _scaled_tangent(right)
    left_norm = np.linalg.norm(left_vector, axis=1)
    right_norm = np.linalg.norm(right_vector, axis=1)
    denominator = left_norm * right_norm
    _require(np.all(denominator > 1.0e-300), "Tangent direction is undefined at a matched sample.")
    cosine = np.sum(left_vector * right_vector, axis=1) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    direction_discrepancy = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))
    norm_relative = np.abs(left_norm - right_norm) / np.maximum.reduce(
        (left_norm, right_norm, np.full_like(left_norm, 1.0e-300))
    )
    worst_direction = int(np.argmin(cosine))
    worst_norm = int(np.argmax(norm_relative))
    return {
        "final_direction_cosine": float(cosine[-1]),
        "minimum_direction_cosine": float(cosine[worst_direction]),
        "minimum_direction_cosine_epoch_years": float(left.times[worst_direction]),
        "direction_discrepancy_rms": float(np.sqrt(np.mean(direction_discrepancy**2))),
        "direction_discrepancy_max": float(np.max(direction_discrepancy)),
        "norm_relative_difference_rms": float(np.sqrt(np.mean(norm_relative**2))),
        "norm_relative_difference_max": float(norm_relative[worst_norm]),
        "norm_relative_difference_worst_epoch_years": float(left.times[worst_norm]),
        "final_left_norm": float(left_norm[-1]),
        "final_right_norm": float(right_norm[-1]),
    }


def _compute_elements(data: RunData) -> dict[str, dict[str, np.ndarray]]:
    if data.elements is not None:
        return data.elements
    names = data.body_names[1:]
    output = {
        name: {
            field: np.empty(len(data.times), dtype=np.float64)
            for field in ("a_m", "e", "i_rad", "varpi_rad", "mean_longitude_rad")
        }
        for name in names
    }
    for sample_index in range(len(data.times)):
        state = NBodyState(
            positions=data.positions[sample_index],
            velocities=data.velocities[sample_index],
            masses=data.masses,
        )
        for element in heliocentric_elements_for_state(state, data.body_names, sun_index=0):
            row = output[element.body_name]
            row["a_m"][sample_index] = element.semi_major_axis_m
            row["e"][sample_index] = element.eccentricity
            row["i_rad"][sample_index] = element.inclination_rad
            row["varpi_rad"][sample_index] = element.longitude_perihelion_rad
            row["mean_longitude_rad"][sample_index] = element.mean_longitude_rad
    data.elements = output
    return output


def _wrapped_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = left - right
    return np.arctan2(np.sin(delta), np.cos(delta))


def _pair_elements(left: RunData, right: RunData) -> dict[str, Any]:
    left_elements = _compute_elements(left)
    right_elements = _compute_elements(right)
    per_body = {}
    for body_name in left_elements:
        left_row = left_elements[body_name]
        right_row = right_elements[body_name]
        a_reference = max(
            abs(0.5 * (left_row["a_m"][0] + right_row["a_m"][0])),
            1.0e-12 * AU_M,
        )
        a_relative = np.abs(left_row["a_m"] - right_row["a_m"]) / a_reference
        e_absolute = np.abs(left_row["e"] - right_row["e"])
        i_absolute = np.abs(left_row["i_rad"] - right_row["i_rad"]) * ARCSEC_PER_RAD
        varpi_absolute = np.abs(_wrapped_delta(left_row["varpi_rad"], right_row["varpi_rad"])) * ARCSEC_PER_RAD
        longitude_absolute = np.abs(
            _wrapped_delta(left_row["mean_longitude_rad"], right_row["mean_longitude_rad"])
        ) * ARCSEC_PER_RAD
        a_worst = int(np.argmax(a_relative))
        e_worst = int(np.argmax(e_absolute))
        i_worst = int(np.argmax(i_absolute))
        varpi_worst = int(np.argmax(varpi_absolute))
        longitude_worst = int(np.argmax(longitude_absolute))
        per_body[body_name] = {
            "semimajor_axis_reference_m": float(a_reference),
            "semimajor_axis_max_relative_difference": float(a_relative[a_worst]),
            "semimajor_axis_worst_epoch_years": float(left.times[a_worst]),
            "eccentricity_max_abs_difference": float(e_absolute[e_worst]),
            "eccentricity_worst_epoch_years": float(left.times[e_worst]),
            "inclination_max_abs_difference_arcsec": float(i_absolute[i_worst]),
            "inclination_worst_epoch_years": float(left.times[i_worst]),
            "longitude_perihelion_max_abs_difference_arcsec": float(varpi_absolute[varpi_worst]),
            "longitude_perihelion_worst_epoch_years": float(left.times[varpi_worst]),
            "mean_longitude_max_abs_difference_arcsec": float(longitude_absolute[longitude_worst]),
            "mean_longitude_worst_epoch_years": float(left.times[longitude_worst]),
        }
    return {"per_body": per_body}


def _linear_fit(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    centered_times = times - float(np.mean(times))
    centered_values = values - float(np.mean(values))
    denominator = float(np.dot(centered_times, centered_times))
    _require(denominator > 0.0, "Linear fit has a zero time denominator.")
    slope = float(np.dot(centered_times, centered_values) / denominator)
    intercept = float(np.mean(values) - slope * np.mean(times))
    return slope, intercept


def _perihelion_rate(data: RunData) -> dict[str, float]:
    mercury = _compute_elements(data)["mercury barycenter"]["varpi_rad"]
    unwrapped = np.unwrap(mercury)
    slope, intercept = _linear_fit(data.times, unwrapped)
    residual = unwrapped - (slope * data.times + intercept)
    return {
        "mean_rate_arcsec_per_century": slope * ARCSEC_PER_RAD * 100.0,
        "fit_interval_start_years": float(data.times[0]),
        "fit_interval_end_years": float(data.times[-1]),
        "fit_residual_rms_arcsec": float(np.sqrt(np.mean(residual**2)) * ARCSEC_PER_RAD),
    }


def _series_metrics(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    worst = int(np.argmax(absolute))
    slope, _ = _linear_fit(times, values)
    return {
        "max_abs": float(absolute[worst]),
        "max_abs_worst_epoch_years": float(times[worst]),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p99_abs": float(np.percentile(absolute, 99.0)),
        "fitted_trend_per_year": slope,
        "fitted_change_over_1myr": slope * 1_000_000.0,
        "final": float(values[-1]),
    }


def _pair_scalar(left: RunData, right: RunData, field: str) -> dict[str, float]:
    difference = left.progress[field] - right.progress[field]
    worst = int(np.argmax(np.abs(difference)))
    return {
        "final_abs_difference": float(abs(difference[-1])),
        "history_rms_difference": float(np.sqrt(np.mean(difference**2))),
        "history_max_abs_difference": float(abs(difference[worst])),
        "worst_epoch_years": float(left.times[worst]),
    }


def _pair_lcn(left: RunData, right: RunData) -> dict[str, float]:
    difference = (left.progress["lcn_1_per_year"] - right.progress["lcn_1_per_year"]) * 1_000_000.0
    worst = int(np.argmax(np.abs(difference)))
    return {
        "final_accumulated_abs_difference": float(abs(difference[-1])),
        "history_accumulated_rms_difference": float(np.sqrt(np.mean(difference**2))),
        "history_accumulated_max_abs_difference": float(abs(difference[worst])),
        "worst_epoch_years": float(left.times[worst]),
    }


def _pair_angular(left: RunData, right: RunData) -> dict[str, float]:
    fields = (
        "angular_momentum_x_kg_m2_s",
        "angular_momentum_y_kg_m2_s",
        "angular_momentum_z_kg_m2_s",
    )
    left_vector = np.column_stack([left.progress[name] for name in fields])
    right_vector = np.column_stack([right.progress[name] for name in fields])
    scale = np.maximum(np.linalg.norm(left_vector, axis=1), np.linalg.norm(right_vector, axis=1))
    relative = np.linalg.norm(left_vector - right_vector, axis=1) / np.maximum(scale, 1.0e-300)
    worst = int(np.argmax(relative))
    return {
        "relative_vector_difference_rms": float(np.sqrt(np.mean(relative**2))),
        "relative_vector_difference_max": float(relative[worst]),
        "worst_epoch_years": float(left.times[worst]),
    }



def _threshold_worst_cases(
    manifest: dict[str, Any],
    runs: dict[str, RunData],
    coarse: dict[str, Any],
    fine: dict[str, Any],
    perihelion: dict[str, dict[str, float]],
    energy: dict[str, dict[str, float]],
    angular: dict[str, dict[str, float]],
) -> dict[str, Any]:
    thresholds = manifest["thresholds"]
    physical_ratios = {
        body: fine["physical"]["per_body"][body]["rms"]
        / max(coarse["physical"]["per_body"][body]["rms"], 1.0e-300)
        for body in INNER_BODIES
    }
    physical_body = max(physical_ratios, key=physical_ratios.get)
    global_ratio = fine["physical"]["global_scaled_rms"] / max(
        coarse["physical"]["global_scaled_rms"], 1.0e-300
    )

    fine_elements = fine["orbital_elements"]["per_body"]
    eccentricity_limits = {
        body: (
            thresholds["mercury_eccentricity_history_max_abs"]
            if body == "mercury barycenter"
            else thresholds["inner_planet_eccentricity_history_max_abs"]
        )
        for body in INNER_BODIES
    }
    eccentricity_body = max(
        INNER_BODIES,
        key=lambda body: fine_elements[body]["eccentricity_max_abs_difference"]
        / eccentricity_limits[body],
    )
    semimajor_body = max(
        INNER_BODIES,
        key=lambda body: fine_elements[body]["semimajor_axis_max_relative_difference"],
    )
    peri_difference = abs(
        perihelion["m0_conv_1d_1myr_s12345"]["mean_rate_arcsec_per_century"]
        - perihelion["m0_conv_0p5d_1myr_s12345"]["mean_rate_arcsec_per_century"]
    )
    energy_run = max(energy, key=lambda name: energy[name]["max_abs"])
    trend_run = max(
        energy,
        key=lambda name: abs(energy[name]["fitted_change_over_1myr"])
        / max(0.25 * energy[name]["max_abs"], 1.0e-10),
    )
    angular_run = max(angular, key=lambda name: angular[name]["max_abs"])
    max_nonfinite = max(
        int(run.integrity["nonfinite_result_count"]) for run in runs.values()
    )
    ordered = [
        "m0_conv_2d_1myr_s12345",
        "m0_conv_1d_1myr_s12345",
        "m0_conv_0p5d_1myr_s12345",
    ]
    energy_reductions = {}
    for metric in thresholds["corrected_energy_metrics_nonincreasing_with_step_reduction"]:
        transitions = []
        for left, right in zip(ordered, ordered[1:]):
            transitions.append(
                {
                    "coarse_run": left,
                    "fine_run": right,
                    "coarse_value": energy[left][metric],
                    "fine_value": energy[right][metric],
                    "fine_over_coarse_ratio": energy[right][metric]
                    / max(energy[left][metric], 1.0e-300),
                    "passed": energy[right][metric] <= energy[left][metric],
                    "worst_epoch_years": (
                        energy[right]["max_abs_worst_epoch_years"]
                        if metric == "max_abs"
                        else None
                    ),
                }
            )
        energy_reductions[metric] = {
            "passed": all(item["passed"] for item in transitions),
            "worst_transition": max(
                transitions, key=lambda item: item["fine_over_coarse_ratio"]
            ),
            "transitions": transitions,
        }
    return {
        "physical_state": {
            "global_fine_over_coarse_ratio": global_ratio,
            "limit": thresholds["global_scaled_physical_rms_fine_over_coarse_max"],
            "worst_inner_body": physical_body,
            "worst_inner_fine_over_coarse_ratio": physical_ratios[physical_body],
            "worst_epoch_years": fine["physical"]["per_body"][physical_body][
                "worst_epoch_years"
            ],
        },
        "mercury_perihelion_rate": {
            "body": "mercury barycenter",
            "value_arcsec_per_century": peri_difference,
            "limit_arcsec_per_century": thresholds[
                "mercury_perihelion_rate_difference_arcsec_per_century_max"
            ],
            "fit_interval_years": [0.0, 1_000_000.0],
            "worst_epoch_years": None,
        },
        "eccentricity_history": {
            "worst_body": eccentricity_body,
            "value": fine_elements[eccentricity_body]["eccentricity_max_abs_difference"],
            "limit": eccentricity_limits[eccentricity_body],
            "worst_epoch_years": fine_elements[eccentricity_body][
                "eccentricity_worst_epoch_years"
            ],
        },
        "mercury_eccentricity_history": {
            "worst_body": "mercury barycenter",
            "value": fine_elements["mercury barycenter"][
                "eccentricity_max_abs_difference"
            ],
            "limit": thresholds["mercury_eccentricity_history_max_abs"],
            "worst_epoch_years": fine_elements["mercury barycenter"][
                "eccentricity_worst_epoch_years"
            ],
        },
        "semimajor_axis_history": {
            "worst_body": semimajor_body,
            "value": fine_elements[semimajor_body][
                "semimajor_axis_max_relative_difference"
            ],
            "limit": thresholds["inner_planet_semimajor_axis_history_max_relative"],
            "worst_epoch_years": fine_elements[semimajor_body][
                "semimajor_axis_worst_epoch_years"
            ],
        },
        "tangent": {
            "body": "full-system first variation",
            "final_direction_cosine": fine["tangent"]["final_direction_cosine"],
            "minimum": thresholds["final_tangent_direction_cosine_min"],
            "fine_direction_rms": fine["tangent"]["direction_discrepancy_rms"],
            "coarse_direction_rms": coarse["tangent"]["direction_discrepancy_rms"],
            "worst_epoch_years": fine["tangent"][
                "minimum_direction_cosine_epoch_years"
            ],
        },
        "megno": {
            "body": "full-system diagnostic",
            "final_abs_difference": fine["megno"]["final_abs_difference"],
            "final_limit": thresholds["final_megno_difference_max"],
            "history_rms_difference": fine["megno"]["history_rms_difference"],
            "history_rms_limit": thresholds["megno_history_rms_difference_max"],
            "worst_epoch_years": fine["megno"]["worst_epoch_years"],
        },
        "lcn": {
            "body": "full-system diagnostic",
            "final_accumulated_abs_difference": fine["lcn"][
                "final_accumulated_abs_difference"
            ],
            "limit": thresholds["final_lcn_accumulated_difference_max"],
            "worst_epoch_years": fine["lcn"]["worst_epoch_years"],
        },
        "corrected_energy": {
            "worst_run_by_max_abs": energy_run,
            "max_abs": energy[energy_run]["max_abs"],
            "max_abs_limit": thresholds["corrected_energy_max_abs_per_run"],
            "max_abs_worst_epoch_years": energy[energy_run][
                "max_abs_worst_epoch_years"
            ],
            "worst_run_by_trend_ratio": trend_run,
            "fitted_change_over_1myr": energy[trend_run][
                "fitted_change_over_1myr"
            ],
            "trend_limit": max(0.25 * energy[trend_run]["max_abs"], 1.0e-10),
            "max_abs_by_step_days": {
                str(runs[name].step_days): energy[name]["max_abs"] for name in runs
            },
            "rms_by_step_days": {
                str(runs[name].step_days): energy[name]["rms"] for name in runs
            },
            "p99_abs_by_step_days": {
                str(runs[name].step_days): energy[name]["p99_abs"] for name in runs
            },
            "roundoff_floor": thresholds["corrected_energy_roundoff_floor"],
            "all_maxima_at_roundoff_floor": all(
                energy[name]["max_abs"] <= thresholds["corrected_energy_roundoff_floor"]
                for name in ordered
            ),
            "nonincreasing_with_step_reduction": energy_reductions,
            "trends_by_run": {
                name: {
                    "value_abs_fitted_change_over_1myr": abs(
                        energy[name]["fitted_change_over_1myr"]
                    ),
                    "limit": max(0.25 * energy[name]["max_abs"], 1.0e-10),
                    "fit_interval_years": [0.0, 1_000_000.0],
                    "worst_epoch_years": None,
                }
                for name in ordered
            },
        },
        "angular_momentum": {
            "worst_run": angular_run,
            "max_abs_relative_drift": angular[angular_run]["max_abs"],
            "limit": thresholds["angular_momentum_rel_drift_max_per_run"],
            "worst_epoch_years": angular[angular_run]["max_abs_worst_epoch_years"],
            "accepted_100k_1d_scale": thresholds[
                "accepted_100k_1d_angular_momentum_rel_drift"
            ],
        },
        "callback_nonfinite": {
            "worst_value": max_nonfinite,
            "limit": thresholds["nonfinite_callback_results"],
            "body": "all real and first-variation particles",
            "worst_epoch_years": None,
        },
    }

def evaluate_criteria(
    manifest: dict[str, Any],
    runs: dict[str, RunData],
    coarse: dict[str, Any],
    fine: dict[str, Any],
    perihelion: dict[str, dict[str, float]],
    energy: dict[str, dict[str, float]],
    angular: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    thresholds = manifest["thresholds"]
    coarse_physical = coarse["physical"]
    fine_physical = fine["physical"]
    global_ratio = fine_physical["global_scaled_rms"] / max(
        coarse_physical["global_scaled_rms"], 1.0e-300
    )
    physical_inner = {
        body: fine_physical["per_body"][body]["rms"]
        < coarse_physical["per_body"][body]["rms"]
        for body in INNER_BODIES
    }
    physical_pass = (
        fine_physical["global_scaled_rms"] < coarse_physical["global_scaled_rms"]
        and all(physical_inner.values())
        and global_ratio
        <= thresholds["global_scaled_physical_rms_fine_over_coarse_max"]
    )

    fine_elements = fine["orbital_elements"]["per_body"]
    eccentricity_limits = {
        body: (
            thresholds["mercury_eccentricity_history_max_abs"]
            if body == "mercury barycenter"
            else thresholds["inner_planet_eccentricity_history_max_abs"]
        )
        for body in INNER_BODIES
    }
    eccentricity_checks = {
        body: fine_elements[body]["eccentricity_max_abs_difference"] <= limit
        for body, limit in eccentricity_limits.items()
    }
    semimajor_checks = {
        body: fine_elements[body]["semimajor_axis_max_relative_difference"]
        <= thresholds["inner_planet_semimajor_axis_history_max_relative"]
        for body in INNER_BODIES
    }
    peri_difference = abs(
        perihelion["m0_conv_1d_1myr_s12345"]["mean_rate_arcsec_per_century"]
        - perihelion["m0_conv_0p5d_1myr_s12345"]["mean_rate_arcsec_per_century"]
    )
    tangent_pass = (
        fine["tangent"]["final_direction_cosine"]
        >= thresholds["final_tangent_direction_cosine_min"]
        and fine["tangent"]["direction_discrepancy_rms"]
        < coarse["tangent"]["direction_discrepancy_rms"]
    )
    megno_pass = (
        fine["megno"]["final_abs_difference"] <= thresholds["final_megno_difference_max"]
        and fine["megno"]["history_rms_difference"]
        <= thresholds["megno_history_rms_difference_max"]
    )
    lcn_pass = (
        fine["lcn"]["final_accumulated_abs_difference"]
        <= thresholds["final_lcn_accumulated_difference_max"]
    )

    ordered = [
        "m0_conv_2d_1myr_s12345",
        "m0_conv_1d_1myr_s12345",
        "m0_conv_0p5d_1myr_s12345",
    ]
    roundoff_floor = thresholds["corrected_energy_roundoff_floor"]
    all_max_at_roundoff = all(energy[name]["max_abs"] <= roundoff_floor for name in ordered)
    max_monotonic = all(
        energy[right]["max_abs"] <= energy[left]["max_abs"]
        for left, right in zip(ordered, ordered[1:])
    ) or all_max_at_roundoff
    rms_monotonic = all(
        energy[right]["rms"] <= energy[left]["rms"]
        for left, right in zip(ordered, ordered[1:])
    )
    p99_monotonic = all(
        energy[right]["p99_abs"] <= energy[left]["p99_abs"]
        for left, right in zip(ordered, ordered[1:])
    )
    energy_bounds = {
        name: energy[name]["max_abs"] <= thresholds["corrected_energy_max_abs_per_run"]
        for name in ordered
    }
    energy_trends = {
        name: abs(energy[name]["fitted_change_over_1myr"])
        <= max(0.25 * energy[name]["max_abs"], 1.0e-10)
        for name in ordered
    }
    energy_pass = (
        all(energy_bounds.values())
        and all(energy_trends.values())
        and max_monotonic
        and rms_monotonic
        and p99_monotonic
    )
    angular_checks = {
        name: angular[name]["max_abs"] <= thresholds["angular_momentum_rel_drift_max_per_run"]
        for name in ordered
    }
    return {
        "physical_state": {
            "passed": physical_pass,
            "fine_over_coarse_global_rms_ratio": global_ratio,
            "inner_planet_fine_strictly_smaller": physical_inner,
        },
        "mercury_perihelion_rate": {
            "passed": peri_difference
            <= thresholds["mercury_perihelion_rate_difference_arcsec_per_century_max"],
            "fine_pair_abs_difference_arcsec_per_century": peri_difference,
        },
        "eccentricity_history": {
            "passed": all(eccentricity_checks.values()),
            "per_body": eccentricity_checks,
            "limits": eccentricity_limits,
        },
        "semimajor_axis_history": {
            "passed": all(semimajor_checks.values()),
            "per_body": semimajor_checks,
        },
        "tangent": {"passed": tangent_pass},
        "megno": {"passed": megno_pass},
        "lcn": {"passed": lcn_pass},
        "corrected_energy": {
            "passed": energy_pass,
            "per_run_bounded": energy_bounds,
            "per_run_trend_bounded": energy_trends,
            "max_nonincreasing_or_roundoff_floor": max_monotonic,
            "rms_nonincreasing": rms_monotonic,
            "p99_nonincreasing": p99_monotonic,
            "all_maxima_at_roundoff_floor": all_max_at_roundoff,
        },
        "angular_momentum": {"passed": all(angular_checks.values()), "per_run": angular_checks},
        "run_integrity": {
            "passed": all(run.integrity["passed"] for run in runs.values())
        },
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['final_status']}",
        "",
        "Three serial 1 Myr compiled-C M0 integrations were compared at matched 100-year samples.",
        "",
        "## Runs",
        "",
        "| Run | Step | Runtime | Throughput | Samples | State rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id, run in payload.get("runs", {}).items():
        lines.append(
            f"| {run_id} | {run['step_days']:g} d | {run['runtime_seconds']:.3f} s | "
            f"{run['throughput_years_per_second']:.3f} yr/s | {run['scientific_samples']} | {run['state_rows']} |"
        )
    lines.extend(["", "## Criteria", "", "| Criterion | Result |", "| --- | ---: |"])
    for name, result in payload.get("criteria", {}).items():
        lines.append(f"| {name} | {'PASS' if result.get('passed') else 'FAIL'} |")
    if payload.get("comparisons"):
        coarse = payload["comparisons"]["coarse_2d_vs_1d"]
        fine = payload["comparisons"]["fine_1d_vs_0p5d"]
        ratio = payload["criteria"]["physical_state"]["fine_over_coarse_global_rms_ratio"]
        peri = payload["criteria"]["mercury_perihelion_rate"][
            "fine_pair_abs_difference_arcsec_per_century"
        ]
        angular_worst = payload["threshold_worst_cases"]["angular_momentum"]
        angular_scale_ratio = (
            angular_worst["max_abs_relative_drift"]
            / angular_worst["accepted_100k_1d_scale"]
        )
        lines.extend(
            [
                "",
                "## Key Metrics",
                "",
                f"- Global scaled physical RMS: coarse `{coarse['physical']['global_scaled_rms']:.12g}`, "
                f"fine `{fine['physical']['global_scaled_rms']:.12g}`, ratio `{ratio:.12g}`.",
                f"- Mercury mean-perihelion-rate fine-pair difference: `{peri:.12g}` arcsec/century.",
                f"- Final tangent cosine: coarse `{coarse['tangent']['final_direction_cosine']:.12g}`, "
                f"fine `{fine['tangent']['final_direction_cosine']:.12g}`.",
                f"- Final MEGNO difference: `{fine['megno']['final_abs_difference']:.12g}`.",
                f"- Final accumulated LCN difference: `{fine['lcn']['final_accumulated_abs_difference']:.12g}`.",
                f"- Worst 1 Myr angular-momentum drift: `{angular_worst['max_abs_relative_drift']:.12g}` "
                f"for {angular_worst['worst_run']} at `{angular_worst['worst_epoch_years']:.12g}` years, "
                f"`{angular_scale_ratio:.3f}x` the accepted 100 kyr 1-day scale but below the "
                f"`{angular_worst['limit']:.12g}` bound; the horizons differ by a factor of ten.",
            ]
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Manifest SHA-256: `{payload.get('manifest_sha256')}`.",
            f"- Protected files unchanged: `{payload.get('protected_files_unchanged')}`.",
            "- No 10 Myr integration was launched.",
            "",
        ]
    )
    if payload.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in payload["failures"])
        worst = payload.get("threshold_worst_cases", {})
        semimajor = worst.get("semimajor_axis_history", {})
        energy = worst.get("corrected_energy", {})
        if semimajor:
            lines.append(
                f"- Worst fine-pair semimajor-axis difference: {semimajor['worst_body']} "
                f"`{semimajor['value']:.12g}` at `{semimajor['worst_epoch_years']:.12g}` years "
                f"(limit `{semimajor['limit']:.12g}`)."
            )
        if energy:
            lines.append(
                f"- Worst corrected-energy maximum: {energy['worst_run_by_max_abs']} "
                f"`{energy['max_abs']:.12g}` at `{energy['max_abs_worst_epoch_years']:.12g}` years; "
                "all runs remain below the absolute bound, but reduction/trend rules fail."
            )
        lines.append("")
    if payload.get("next_action"):
        lines.extend(["## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def _write_final_reports(manifest: dict[str, Any], payload: dict[str, Any]) -> None:
    _require(payload["final_status"] in FINAL_STATUSES, "Invalid final status.")
    json_path = Path(manifest["paths"]["report_json"])
    markdown_path = Path(manifest["paths"]["report_markdown"])
    _atomic_write_json(json_path, payload)
    _atomic_write_text(markdown_path, _markdown_report(payload))
    print(f"[m0-convergence] wrote {json_path}")
    print(f"[m0-convergence] wrote {markdown_path}")


def analyze(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "convergence manifest")
    manifest_hash = sha256_file(manifest_path)
    protected_unchanged = all(
        sha256_file(Path(manifest["paths"]["project_root"]) / relative) == expected
        for relative, expected in manifest["protected_files"].items()
    )
    base_payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "model_id": manifest["model_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protected_files_unchanged": protected_unchanged,
        "thresholds": manifest["thresholds"],
        "comparison_definitions": manifest["comparison_definitions"],
        "failures": [],
    }
    try:
        _require(protected_unchanged, "Protected file hash changed during Step 3.")
        loaded = {
            run["id"]: _load_run(manifest_path, manifest, run) for run in manifest["runs"]
        }
        coarse_left = loaded["m0_conv_2d_1myr_s12345"]
        middle = loaded["m0_conv_1d_1myr_s12345"]
        fine_right = loaded["m0_conv_0p5d_1myr_s12345"]
        coarse = {
            "physical": _pair_physical(coarse_left, middle),
            "tangent": _pair_tangent(coarse_left, middle),
            "orbital_elements": _pair_elements(coarse_left, middle),
            "megno": _pair_scalar(coarse_left, middle, "megno"),
            "lcn": _pair_lcn(coarse_left, middle),
            "angular_momentum": _pair_angular(coarse_left, middle),
        }
        fine = {
            "physical": _pair_physical(middle, fine_right),
            "tangent": _pair_tangent(middle, fine_right),
            "orbital_elements": _pair_elements(middle, fine_right),
            "megno": _pair_scalar(middle, fine_right, "megno"),
            "lcn": _pair_lcn(middle, fine_right),
            "angular_momentum": _pair_angular(middle, fine_right),
        }
        perihelion = {name: _perihelion_rate(run) for name, run in loaded.items()}
        energy = {
            name: _series_metrics(run.times, run.progress["corrected_energy_rel_change"])
            for name, run in loaded.items()
        }
        angular = {
            name: _series_metrics(run.times, run.progress["angular_momentum_rel_drift"])
            for name, run in loaded.items()
        }
        criteria = evaluate_criteria(
            manifest, loaded, coarse, fine, perihelion, energy, angular
        )
        threshold_worst_cases = _threshold_worst_cases(
            manifest, loaded, coarse, fine, perihelion, energy, angular
        )
        all_passed = all(result.get("passed") is True for result in criteria.values())
        final_status = "M0_1DAY_CONVERGED" if all_passed else "M0_1DAY_NOT_CONVERGED"
        failures = [name for name, result in criteria.items() if result.get("passed") is not True]
        base_payload.update(
            final_status=final_status,
            failures=failures,
            runs={name: run.integrity for name, run in loaded.items()},
            comparisons={"coarse_2d_vs_1d": coarse, "fine_1d_vs_0p5d": fine},
            mercury_perihelion=perihelion,
            corrected_energy=energy,
            angular_momentum=angular,
            criteria=criteria,
            threshold_worst_cases=threshold_worst_cases,
            next_action=(
                "Do not launch 10 Myr. Preregister one 0.25-day, 1 Myr lane and compare it "
                "with the existing 0.5-day lane, retaining the frozen semimajor-axis and "
                "corrected-energy criteria."
            ),
            artifact_inventory=[entry for run in loaded.values() for entry in run.inventory],
        )
    except Exception as exc:
        base_payload.update(final_status="BLOCKED", failures=[str(exc)], criteria={})
    _write_final_reports(manifest, base_payload)
    return 0 if base_payload["final_status"] == "M0_1DAY_CONVERGED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and analyze M0 timestep convergence.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("analyze")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    if args.command == "preflight":
        raise SystemExit(preflight(manifest_path))
    raise SystemExit(analyze(manifest_path))


if __name__ == "__main__":
    main()
