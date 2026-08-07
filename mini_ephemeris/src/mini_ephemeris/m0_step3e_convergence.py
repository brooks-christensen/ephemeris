from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import load_c_backend
from .long_term_stability_cli import (
    optional_import_module,
    parse_start_datetime,
    stability_body_list,
)
from .m0_energy_precision_diagnosis import decimal_energy
from .m0_ias15_phase_reference import pair_diagnostics
from .m0_integrator_roundoff_diagnosis import (
    _block_metrics,
    _energy_statistics,
    _read_physical_groups,
    _runtime_identity,
)
from .m0_telemetry import gr_potential_energy
from .m0_timestep_convergence import (
    INNER_BODIES,
    ConvergenceError,
    RunData,
    _compute_elements,
    _git,
    _load_json,
    _load_run,
    _pair_angular,
    _pair_elements,
    _pair_lcn,
    _pair_physical,
    _pair_scalar,
    _pair_tangent,
    _perihelion_rate,
    _require,
    _series_metrics,
)
from .nbody import NBodyState
from .orbital_elements import (
    ARCSEC_PER_RAD,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
)
from .rebound_gr_tangent_backend_cli import (
    _config_payload,
    canonical_hash,
    initial_condition_hash,
    output_paths,
    sha256_file,
)
from .stability_diagnostics import (
    center_of_mass_position_velocity,
    total_angular_momentum_vector,
    total_newtonian_energy,
)


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/17_m0_step3e_whfast_0125d_convergence_v1.json"
)
FINAL_STATUSES = {
    "STEP3E_025_DAY_PRODUCTION_VALIDATED",
    "STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED",
    "BLOCKED",
}
EXPECTED_DIRTY_PATHS = {
    "ephemeris_experiment_runner/manifests/17_m0_step3e_whfast_0125d_convergence_v1.json",
    "mini_ephemeris/src/mini_ephemeris/m0_step3e_convergence.py",
    "mini_ephemeris/tests/test_m0_step3e_convergence.py",
}
ANGLE_FIELDS = (
    "inclination_rad",
    "longitude_ascending_node_rad",
    "argument_perihelion_rad",
    "longitude_perihelion_rad",
    "true_anomaly_rad",
    "mean_anomaly_rad",
    "mean_longitude_rad",
)


def _finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"Nonfinite JSON value at {path}.")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")


def _json_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_native(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_native(child) for child in value]
    if isinstance(value, tuple):
        return [_json_native(child) for child in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    payload = _json_native(payload)
    _finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing artifact: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _inventory_entries(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "sha256", "size_bytes"}.issubset(value):
            yield value
        for child in value.values():
            yield from _inventory_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _inventory_entries(child)


def _verify_inventory(value: Any) -> tuple[int, int]:
    entries = list(_inventory_entries(value))
    unique: dict[str, tuple[str, int]] = {}
    for item in entries:
        path = Path(item["path"])
        _require(path.is_file(), f"Missing historical artifact: {path}")
        _require(path.stat().st_size == item["size_bytes"], f"Historical size mismatch: {path}")
        _require(sha256_file(path) == item["sha256"], f"Historical hash mismatch: {path}")
        previous = unique.setdefault(str(path), (item["sha256"], item["size_bytes"]))
        _require(previous == (item["sha256"], item["size_bytes"]), f"Conflicting inventory: {path}")
    return len(entries), len(unique)


def _source_artifact_audit(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["paths"]["project_root"])
    verified = []
    for name, item in manifest["source_artifacts"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        actual = sha256_file(path)
        _require(actual == item["sha256"], f"Authoritative source changed: {name}")
        verified.append({"name": name, "path": str(path), "sha256": actual})
    return verified


def _protected_audit(manifest: dict[str, Any]) -> list[dict[str, str]]:
    root = Path(manifest["paths"]["project_root"])
    verified = []
    for relative, expected in manifest["protected_files"].items():
        actual = sha256_file(root / relative)
        _require(actual == expected, f"Protected file changed: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return verified


def _status_paths() -> dict[str, Path]:
    return {
        "m13": Path("docs/validation/m0-integrator-roundoff-diagnosis-v1/m0_integrator_roundoff_diagnosis_summary.json"),
        "m14": Path("docs/validation/m0-reversibility-roundoff-gate-v1/m0_reversibility_roundoff_gate_summary.json"),
        "m15": Path("docs/validation/m0-integrator-roundoff-diagnosis-continuation-v1/m0_integrator_roundoff_diagnosis_continuation_summary.json"),
        "m16": Path("docs/validation/m0-ias15-phase-reference-v1/m0_ias15_phase_reference_summary.json"),
    }


def _historical_status_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    summaries = {name: _load_json(path, name) for name, path in _status_paths().items()}
    expected = manifest["historical_results_immutable"]
    _require(summaries["m13"].get("primary_mechanism") == expected["manifest_13_primary_mechanism"], "Manifest-13 mechanism changed.")
    _require(summaries["m13"].get("step3_diagnosis_status") == expected["manifest_13_step3_diagnosis_status"], "Manifest-13 status changed.")
    _require(summaries["m14"].get("final_status") == expected["manifest_14_status"], "Manifest-14 gate changed.")
    _require(summaries["m15"].get("primary_mechanism") == expected["manifest_15_primary_mechanism"], "Manifest-15 mechanism changed.")
    _require(summaries["m15"].get("step3_diagnosis_status") == expected["manifest_15_step3_diagnosis_status"], "Manifest-15 status changed.")
    _require(summaries["m16"].get("ias15_reference_status") == expected["manifest_16_ias15_reference_status"], "Manifest-16 IAS15 status changed.")
    _require(summaries["m16"].get("primary_mechanism") == expected["manifest_16_primary_mechanism"], "Manifest-16 mechanism changed.")
    _require(summaries["m16"].get("step3_diagnosis_status") == expected["manifest_16_step3_diagnosis_status"], "Manifest-16 diagnosis changed.")
    step3b = _load_json(Path(manifest["paths"]["step3b_summary"]), "Step 3b summary")
    _require(step3b.get("final_status") == expected["step3b_status"], "Step 3b status changed.")
    total_entries = 0
    unique_paths: set[str] = set()
    for summary in summaries.values():
        entries = list(_inventory_entries(summary))
        _verify_inventory(entries)
        total_entries += len(entries)
        unique_paths.update(str(item["path"]) for item in entries)
    return {
        "manifest_13": [summaries["m13"].get("primary_mechanism"), summaries["m13"].get("step3_diagnosis_status")],
        "manifest_14": summaries["m14"].get("final_status"),
        "manifest_15": [summaries["m15"].get("primary_mechanism"), summaries["m15"].get("step3_diagnosis_status")],
        "manifest_16": [summaries["m16"].get("ias15_reference_status"), summaries["m16"].get("primary_mechanism"), summaries["m16"].get("step3_diagnosis_status")],
        "inventory_entries": total_entries,
        "unique_inventory_paths": len(unique_paths),
    }


def derive_step_accounting(duration_years: int, cadence_years: int, step_days: float) -> dict[str, Any]:
    total_days = Decimal(duration_years) * Decimal("365.25")
    cadence_days = Decimal(cadence_years) * Decimal("365.25")
    step = Decimal(str(step_days))
    full_steps = total_days / step
    cadence_steps = cadence_days / step
    _require(full_steps == full_steps.to_integral_value(), "Full duration requires a fractional step.")
    _require(cadence_steps == cadence_steps.to_integral_value(), "Cadence requires a fractional step.")
    return {
        "full_steps": int(full_steps),
        "steps_per_scientific_interval": int(cadence_steps),
        "fractional_endpoint_step": False,
    }


def derive_energy_prediction(manifest15_summary: dict[str, Any], ias15_floor: float) -> dict[str, float]:
    histories = manifest15_summary["long_history"]
    steps = {"0p5d": 0.5, "0p25d": 0.25}
    q = {
        name: histories[name]["fitted_slope_per_year"] * step / 365.25
        for name, step in steps.items()
    }
    block_q = [
        block["fitted_slope_per_year"] * steps[name] / 365.25
        for name in steps
        for block in histories[name]["blocks"]
    ]
    stochastic_q = max(
        (histories[name]["ci95_high_per_year"] - histories[name]["ci95_low_per_year"])
        * steps[name]
        / (2.0 * 365.25)
        for name in steps
    )
    truncation_q = abs(q["0p5d"] - q["0p25d"])
    expected_steps = 2_922_000_000
    ias_q = ias15_floor / expected_steps
    q_low = min(block_q) - truncation_q - stochastic_q - ias_q
    q_high = max(block_q) + truncation_q + stochastic_q + ias_q
    q_center = 0.5 * (q["0p5d"] + q["0p25d"])
    multiplier = 365.25 / 0.125
    oscillatory = histories["0p25d"]["detrended_peak_to_peak"]
    stochastic_history = 3.0 * histories["0p25d"]["detrended_rms"] * math.sqrt(2.0)
    return {
        "q_0p5": q["0p5d"],
        "q_0p25": q["0p25d"],
        "q_center": q_center,
        "q_low": q_low,
        "q_high": q_high,
        "block_q_min": min(block_q),
        "block_q_max": max(block_q),
        "truncation_q": truncation_q,
        "stochastic_q": stochastic_q,
        "ias15_q": ias_q,
        "slope_center": q_center * multiplier,
        "slope_low": q_low * multiplier,
        "slope_high": q_high * multiplier,
        "oscillatory_allowance": oscillatory,
        "stochastic_history_allowance": stochastic_history,
        "endpoint_low": q_low * multiplier * 1_000_000.0 - oscillatory - stochastic_history - ias15_floor,
        "endpoint_high": q_high * multiplier * 1_000_000.0 + oscillatory + stochastic_history + ias15_floor,
    }


def _assert_close(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=2.0e-15, abs_tol=0.0), f"Preregistered value changed: {label}")


def _prediction_audit(manifest: dict[str, Any]) -> dict[str, float]:
    summary15 = _load_json(_status_paths()["m15"], "manifest-15 summary")
    floor = manifest["energy_prediction"]["components"]["ias15_energy_floor"]
    actual = derive_energy_prediction(summary15, floor)
    expected = manifest["energy_prediction"]
    comparisons = {
        "q_0p5": expected["evidence"]["q_0p5"],
        "q_0p25": expected["evidence"]["q_0p25"],
        "q_center": expected["components"]["systematic_q_center"],
        "q_low": expected["predicted_q_interval"][0],
        "q_high": expected["predicted_q_interval"][1],
        "block_q_min": expected["components"]["empirical_block_q_min"],
        "block_q_max": expected["components"]["empirical_block_q_max"],
        "truncation_q": expected["components"]["truncation_q_half_width"],
        "stochastic_q": expected["components"]["stochastic_q_half_width"],
        "ias15_q": expected["components"]["ias15_q_equivalent"],
        "slope_center": expected["predicted_slope_per_year"]["center"],
        "slope_low": expected["predicted_slope_per_year"]["low"],
        "slope_high": expected["predicted_slope_per_year"]["high"],
        "oscillatory_allowance": expected["components"]["oscillatory_absolute_allowance"],
        "stochastic_history_allowance": expected["components"]["stochastic_1myr_absolute_allowance"],
        "endpoint_low": expected["predicted_1myr_endpoint_envelope"][0],
        "endpoint_high": expected["predicted_1myr_endpoint_envelope"][1],
    }
    for name, value in comparisons.items():
        _assert_close(actual[name], value, name)
    return actual


def _runner_configuration(manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    shared = manifest["shared_configuration"]
    bodies = stability_body_list(shared["model_scope"], include_pluto=True)
    start = parse_start_datetime(shared["start_date"])
    kernel = Path(manifest["paths"]["kernel"])
    initial = initial_state_solar_system_barycentric(
        start,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=str(kernel)),
    )
    backend = load_c_backend()
    hashes = {
        "kernel_sha256": sha256_file(kernel),
        "initial_conditions_sha256": initial_condition_hash(initial, bodies),
        "artifact_sha256": backend.build_metadata["artifact_sha256"],
        "c_source_sha256": backend.build_metadata["source_sha256"],
    }
    args = argparse.Namespace(
        model_id=manifest["model_id"],
        gr_tangent_backend=shared["backend"],
        start_date=start,
        model_scope=shared["model_scope"],
        duration_years=float(shared["duration_years"]),
        step_days=float(manifest["new_lane"]["step_days"]),
        record_every_years=float(shared["record_every_years"]),
        archive_interval_years=float(shared["archive_interval_years"]),
        megno_seed=int(shared["megno_seed"]),
        gr_scale=float(shared["gr_scale"]),
        no_central_response=not bool(shared["include_central_response"]),
    )
    configuration = _config_payload(args, list(bodies), hashes)
    return canonical_hash(configuration), configuration


def _runtime_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    rebound = optional_import_module("rebound")
    reboundx = optional_import_module("reboundx")
    actual = _runtime_identity(rebound)
    expected = manifest["runtime_identity"]
    checks = {
        "rebound_version": actual["rebound_version"],
        "rebound_build": actual["rebound_build"],
        "rebound_githash": actual["rebound_githash"],
        "rebound_shared_library_sha256": actual["shared_library_sha256"],
        "rebound_header_sha256": actual["header_sha256"],
    }
    for name, value in checks.items():
        _require(value == expected[name], f"Runtime identity changed: {name}")
    _require(reboundx.__version__ == expected["reboundx_version"], "REBOUNDx version changed.")
    compiler = subprocess.run(["cc", "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    _require(compiler == expected["compiler"], "Compiler identity changed.")
    floating = {
        "fegetround": int(ctypes.CDLL(None).fegetround()),
        "python_float_mant_dig": sys.float_info.mant_dig,
        "python_float_radix": sys.float_info.radix,
        "python_float_rounds": sys.float_info.rounds,
    }
    _require(floating == expected["floating_point"], "Floating-point environment changed.")
    _require(np.__version__ == expected["numpy_version"], "NumPy version changed.")
    return {**actual, "reboundx_version": reboundx.__version__, "compiler": compiler, "floating_point": floating, "numpy_version": np.__version__, "platform": platform.platform()}


def audit(manifest_path: Path, *, write: bool = True) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "Step 3e manifest")
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Step 3e must run at project root.")
    _require(manifest.get("frozen_before_authorized_trajectory") is True, "Manifest is not frozen.")
    head = _git("rev-parse", "HEAD")
    _require(head == manifest["provenance"]["starting_commit"], "Wrong Step 3e starting commit.")
    _git("merge-base", "--is-ancestor", manifest["provenance"]["validated_c_baseline_commit"], head)
    _git("merge-base", "--is-ancestor", manifest["provenance"]["step2_commit"], head)
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C tag is not annotated.")
    _require(_git("rev-parse", tag + "^{commit}") == manifest["provenance"]["validated_c_baseline_commit"], "Compiled-C tag target changed.")
    protected = _protected_audit(manifest)
    sources = _source_artifact_audit(manifest)
    historical = _historical_status_audit(manifest)
    prediction = _prediction_audit(manifest)

    accounting = derive_step_accounting(1_000_000, 100, manifest["new_lane"]["step_days"])
    endpoint = manifest["endpoint_semantics"]
    _require(accounting["full_steps"] == endpoint["expected_full_steps"], "Full-step count changed.")
    _require(accounting["steps_per_scientific_interval"] == endpoint["steps_per_scientific_interval"], "Cadence step count changed.")

    manifest11_path = Path(manifest["paths"]["manifest_11"])
    manifest11 = _load_json(manifest11_path, "manifest 11")
    existing = _load_run(manifest11_path, manifest11, manifest11["decisive_run"])
    _require(existing.integrity["callback_invocations"] == 1_461_000_000, "Existing 0.25-day callback count changed.")
    step3b_summary = _load_json(Path(manifest["paths"]["step3b_summary"]), "Step 3b summary")
    _verify_inventory(step3b_summary["artifact_inventory"]["new_0p25d"])

    rebound = optional_import_module("rebound")
    existing_paths = output_paths(Path(manifest11["decisive_run"]["output_dir"]), manifest11["decisive_run"]["id"], None)
    archive = rebound.Simulationarchive(str(existing_paths["archive"]))
    _require(len(archive) == 11, "Existing 0.25-day archive count changed.")
    _require(float(archive[-1].t) == 1_000_000.0 * JULIAN_YEAR_S, "Existing 0.25-day archive endpoint changed.")

    fingerprint, configuration = _runner_configuration(manifest)
    _require(fingerprint == manifest["new_lane"]["configuration_fingerprint"], "New-lane fingerprint changed.")
    runtime = _runtime_audit(manifest)
    disk = shutil.disk_usage(root)
    _require(disk.free >= manifest["operational_gate"]["disk_required_with_atomic_and_safety_allowance_bytes"], "Insufficient Step 3e disk space.")

    run_dir = Path(manifest["new_lane"]["output_dir"])
    report_json = Path(manifest["paths"]["report_json"])
    if not run_dir.exists():
        _require(not report_json.exists(), "Step 3e report exists before the lane.")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": head,
        "git_dirty_paths": _git("status", "--porcelain").splitlines(),
        "tag": {"name": tag, "object_type": "tag", "target": manifest["provenance"]["validated_c_baseline_commit"]},
        "protected_files": protected,
        "source_artifacts": sources,
        "historical": historical,
        "existing_0p25_lane": existing.integrity,
        "existing_0p25_archive_snapshots": len(archive),
        "runtime": runtime,
        "step_accounting": accounting,
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "energy_prediction_rederived": prediction,
        "available_disk_bytes": disk.free,
        "required_disk_bytes": manifest["operational_gate"]["disk_required_with_atomic_and_safety_allowance_bytes"],
        "manifest_sha256": sha256_file(manifest_path),
    }
    if write:
        output = Path(manifest["paths"]["output_root"]) / "operations/audit.json"
        _atomic_json(output, payload)
        print(f"[step3e] audit PASS: {output}")
    return payload


def _read_prefix(progress_path: Path, state_path: Path) -> dict[str, Any] | None:
    if not progress_path.is_file() or not state_path.is_file():
        return None
    try:
        with progress_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 101:
            return None
        with state_path.open() as handle:
            state_rows = max(sum(1 for _ in handle) - 1, 0)
        row = rows[100]
        return {
            "progress_rows": len(rows),
            "state_rows": state_rows,
            "sample_index": 100,
            "time_years": float(row["time_years"]),
            "callback_invocations": int(row["callback_invocations"]),
            "nonfinite_result_count": int(row["nonfinite_result_count"]),
            "configuration_fingerprint": row["configuration_fingerprint"],
        }
    except (OSError, ValueError, KeyError):
        return None


def run_lane(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Step 3e manifest")
    audit_payload = audit(manifest_path, write=True)
    root = Path(manifest["paths"]["project_root"])
    lane = manifest["new_lane"]
    run_dir = Path(lane["output_dir"])
    log_path = Path(lane["log_path"])
    paths = output_paths(run_dir, lane["id"], None)
    resume = run_dir.exists()
    if resume:
        summary_path = paths["summary"]
        if summary_path.is_file() and _load_json(summary_path, "lane summary").get("complete") is True:
            print("[step3e] authenticated lane is already complete; no trajectory launched.")
            return 0
        restart = _load_json(paths["restart"], "restart sidecar")
        _require(restart.get("configuration_fingerprint") == lane["configuration_fingerprint"], "Partial-lane fingerprint mismatch.")
    else:
        _require(not log_path.exists(), "Collision: Step 3e log already exists.")

    command = list(lane["command"])
    if resume:
        command.append("--resume")
    operation_root = Path(manifest["paths"]["output_root"]) / "operations"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    launch = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "resume": resume,
        "manifest_sha256": sha256_file(manifest_path),
        "configuration_fingerprint": lane["configuration_fingerprint"],
        "expected_runtime_and_disk": manifest["operational_gate"],
        "available_disk_bytes": shutil.disk_usage(root).free,
        "protected_files": audit_payload["protected_files"],
        "expected_artifacts": [str(paths[name]) for name in ("progress", "state", "status", "summary", "restart", "archive")] + [str(log_path)],
    }
    _atomic_json(operation_root / "launch_record.json", launch)
    started = time.monotonic()
    mode = "a" if resume else "x"
    with log_path.open(mode) as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        prefix_payload: dict[str, Any] | None = None
        prefix_previously_passed = (operation_root / "prefix_gate.json").is_file() and _load_json(operation_root / "prefix_gate.json", "prefix gate").get("passed") is True
        while process.poll() is None and not prefix_previously_passed:
            prefix = _read_prefix(paths["progress"], paths["state"])
            if prefix is not None and prefix["time_years"] >= manifest["operational_gate"]["prefix"]["minimum_completed_years"]:
                elapsed = time.monotonic() - started
                projected = elapsed * 100.0
                required = manifest["operational_gate"]["prefix"]
                passed = (
                    prefix["sample_index"] == 100
                    and prefix["time_years"] == 10_000.0
                    and prefix["callback_invocations"] == required["minimum_completed_steps"]
                    and prefix["nonfinite_result_count"] == 0
                    and prefix["configuration_fingerprint"] == lane["configuration_fingerprint"]
                    and prefix["progress_rows"] >= required["expected_progress_rows_including_t0"]
                    and prefix["state_rows"] >= required["expected_state_rows_including_t0"]
                    and elapsed <= required["maximum_elapsed_seconds"]
                    and shutil.disk_usage(root).free >= manifest["operational_gate"]["disk_required_with_atomic_and_safety_allowance_bytes"]
                )
                prefix_payload = {
                    **prefix,
                    "schema_version": 1,
                    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "elapsed_seconds": elapsed,
                    "projected_runtime_seconds": projected,
                    "projected_throughput_years_per_second": 1_000_000.0 / projected,
                    "maximum_elapsed_seconds": required["maximum_elapsed_seconds"],
                    "available_disk_bytes": shutil.disk_usage(root).free,
                    "passed": passed,
                }
                _atomic_json(operation_root / "prefix_gate.json", prefix_payload)
                if not passed:
                    process.terminate()
                    process.wait(timeout=60)
                    print("[step3e] prefix gate BLOCKED; partial authenticated lane retained.")
                    return 2
                print(f"[step3e] prefix gate PASS in {elapsed:.3f} s; continuing the same trajectory.", flush=True)
                prefix_previously_passed = True
            time.sleep(0.5)
        return_code = process.wait()
    completion = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "return_code": return_code,
        "elapsed_seconds_this_process": time.monotonic() - started,
        "prefix_gate": prefix_payload or _load_json(operation_root / "prefix_gate.json", "prefix gate"),
    }
    _atomic_json(operation_root / "process_completion.json", completion)
    _require(return_code == 0, f"Authorized lane exited with status {return_code}.")
    print("[step3e] authorized 0.125-day lane completed.")
    return 0


def _run_state(run: RunData, index: int) -> NBodyState:
    return NBodyState(run.positions[index], run.velocities[index], run.masses)


def _energy_reconstruction(run: RunData, state_path: Path, progress_path: Path) -> dict[str, Any]:
    times, _, groups = _read_physical_groups(state_path, run.body_names)
    _require(np.array_equal(times, run.times), f"Energy times differ: {run.run_id}")
    with progress_path.open(newline="") as handle:
        progress_rows = list(csv.DictReader(handle))
    _require(len(progress_rows) == len(run.times), f"Energy progress count differs: {run.run_id}")
    decimal_values = []
    float_values = []
    telemetry_component_max = {"newtonian": 0.0, "gr_potential": 0.0, "corrected": 0.0, "relative": 0.0}
    reference_decimal = None
    reference_float = None
    with localcontext() as context:
        context.prec = 60
        for index, group in enumerate(groups):
            decimal_value = decimal_energy(
                group,
                gravitational_constant=Decimal("6.67430e-11"),
                speed_of_light=Decimal("299792458"),
                coefficient_scale=Decimal(1),
            )["corrected"]
            state = _run_state(run, index)
            newtonian = total_newtonian_energy(state)
            gr_value = gr_potential_energy(state)
            corrected = newtonian + gr_value
            if reference_decimal is None:
                reference_decimal = decimal_value
                reference_float = corrected
            decimal_values.append(float((decimal_value - reference_decimal) / abs(reference_decimal)))
            float_drift = (corrected - reference_float) / abs(reference_float)
            float_values.append(float_drift)
            telemetry_component_max["newtonian"] = max(telemetry_component_max["newtonian"], abs(newtonian - float(progress_rows[index]["newtonian_energy_j"])))
            telemetry_component_max["gr_potential"] = max(telemetry_component_max["gr_potential"], abs(gr_value - float(progress_rows[index]["gr_potential_energy_j"])))
            telemetry_component_max["corrected"] = max(telemetry_component_max["corrected"], abs(corrected - float(progress_rows[index]["corrected_energy_j"])))
            telemetry_component_max["relative"] = max(telemetry_component_max["relative"], abs(float_drift - float(progress_rows[index]["corrected_energy_rel_change"])))
    decimal_array = np.asarray(decimal_values, dtype=np.float64)
    statistics = _energy_statistics(run.times, decimal_array)
    blocks = _block_metrics(run.times, decimal_array, 1_000_000.0)
    statistics["blocks"] = blocks
    statistics["same_sign_block_count"] = sum(block["fitted_slope_per_year"] > 0.0 for block in blocks)
    statistics["energy_change_per_step"] = statistics["fitted_slope_per_year"] * run.step_days / 365.25
    return {
        "history": decimal_array,
        "float_history": np.asarray(float_values),
        "statistics": statistics,
        "telemetry_reproduction_max_abs": telemetry_component_max,
    }


def energy_prediction_gate(manifest: dict[str, Any], times: np.ndarray, history: np.ndarray, statistics: dict[str, Any]) -> dict[str, Any]:
    prediction = manifest["energy_prediction"]
    slope = prediction["predicted_slope_per_year"]
    components = prediction["components"]
    allowance = (
        components["oscillatory_absolute_allowance"]
        + components["stochastic_1myr_absolute_allowance"] * np.sqrt(times / 1_000_000.0)
        + components["ias15_energy_floor"]
    )
    lower = slope["low"] * times - allowance
    upper = slope["high"] * times + allowance
    below = np.maximum(lower - history, 0.0)
    above = np.maximum(history - upper, 0.0)
    excess = np.maximum(below, above)
    worst = int(np.argmax(excess))
    q_low, q_high = prediction["predicted_q_interval"]
    q_value = statistics["energy_change_per_step"]
    block_q = [block["fitted_slope_per_year"] * 0.125 / 365.25 for block in statistics["blocks"]]
    block_interval_pass = [q_low <= value <= q_high for value in block_q]
    checks = {
        "positive_slope": statistics["fitted_slope_per_year"] > 0.0,
        "positive_final": float(history[-1]) > 0.0,
        "slope_interval": slope["low"] <= statistics["fitted_slope_per_year"] <= slope["high"],
        "q_interval": q_low <= q_value <= q_high,
        "history_envelope": float(excess[worst]) == 0.0,
        "block_q_interval": all(block_interval_pass),
        "same_sign_blocks": statistics["same_sign_block_count"] >= prediction["block_prediction"]["minimum_same_sign_blocks"],
        "absolute_bound": statistics["max_abs"] <= manifest["thresholds"]["corrected_energy_max_abs_per_run"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "q": q_value,
        "q_interval": [q_low, q_high],
        "block_q": block_q,
        "block_q_interval_pass": block_interval_pass,
        "same_sign_blocks": statistics["same_sign_block_count"],
        "slope_interval": [slope["low"], slope["high"]],
        "history_max_envelope_excess": float(excess[worst]),
        "history_worst_epoch_years": float(times[worst]),
        "endpoint_envelope": prediction["predicted_1myr_endpoint_envelope"],
        "lower_history": lower,
        "upper_history": upper,
    }


def _rtn_comparison(left: RunData, right: RunData) -> dict[str, Any]:
    bodies = {}
    all_position = []
    for body_index, body_name in enumerate(left.body_names[1:], start=1):
        reference_r = right.positions[:, body_index] - right.positions[:, 0]
        reference_v = right.velocities[:, body_index] - right.velocities[:, 0]
        radial = reference_r / np.linalg.norm(reference_r, axis=1)[:, None]
        normal = np.cross(reference_r, reference_v)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        transverse = np.cross(normal, radial)
        delta_r = (left.positions[:, body_index] - left.positions[:, 0]) - reference_r
        delta_v = (left.velocities[:, body_index] - left.velocities[:, 0]) - reference_v
        position = np.column_stack((np.sum(delta_r * radial, axis=1), np.sum(delta_r * transverse, axis=1), np.sum(delta_r * normal, axis=1)))
        velocity = np.column_stack((np.sum(delta_v * radial, axis=1), np.sum(delta_v * transverse, axis=1), np.sum(delta_v * normal, axis=1)))
        all_position.append(position)
        payload = {}
        for label, values in (("position_m", position), ("velocity_m_per_s", velocity)):
            payload[label] = {}
            for axis, axis_name in enumerate(("radial", "transverse", "normal")):
                absolute = np.abs(values[:, axis])
                worst = int(np.argmax(absolute))
                payload[label][axis_name] = {
                    "rms": float(np.sqrt(np.mean(values[:, axis] ** 2))),
                    "maximum_abs": float(absolute[worst]),
                    "worst_epoch_years": float(left.times[worst]),
                }
        payload["transverse_position_variance_fraction"] = float(np.sum(position[:, 1] ** 2) / max(np.sum(position**2), 1.0e-300))
        bodies[body_name] = payload
    all_values = np.concatenate(all_position)
    return {
        "global_transverse_position_variance_fraction": float(np.sum(all_values[:, 1] ** 2) / max(np.sum(all_values**2), 1.0e-300)),
        "bodies": bodies,
    }


def _phase_diagnostics(left: RunData, right: RunData, sample_count: int = 101) -> dict[str, Any]:
    times = left.times[:sample_count]
    left_states = [_run_state(left, index) for index in range(sample_count)]
    right_states = [_run_state(right, index) for index in range(sample_count)]
    return pair_diagnostics(times, left_states, right_states, left.body_names)


def _phase_gate(manifest: dict[str, Any], previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    thresholds = manifest["thresholds"]["manifest16_phase"]
    orientations = []
    ratios = []
    for body in current["bodies"].values():
        elements = body["orbital_elements"]
        orientations.append(max(
            elements["inclination_rad"]["maximum_abs"],
            elements["longitude_ascending_node_rad"]["maximum_abs"],
            elements["argument_perihelion_rad"]["maximum_abs"],
        ))
        ratios.append(body["phase_angle_over_orientation_angle"])
    checks = {
        "global_transverse": current["global_transverse_position_variance_fraction"] >= thresholds["global_transverse_position_variance_fraction_min"],
        "worst_body_transverse": current["worst_body_transverse_position_variance_fraction"] >= thresholds["worst_body_transverse_position_variance_fraction_min"],
        "orientation": max(orientations) <= thresholds["orientation_angle_abs_rad_max"],
        "phase_over_orientation": min(ratios) >= thresholds["phase_angle_over_orientation_min"],
        "first_10k_state_improves": current["scaled_state"]["global_scaled_rms"] < previous["scaled_state"]["global_scaled_rms"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_orientation_angle_rad": max(orientations),
        "minimum_phase_over_orientation": min(ratios),
        "ias15_global_scaled_rms": thresholds["ias15_global_scaled_rms"],
        "current_over_ias15_global_envelope": current["scaled_state"]["global_scaled_rms"] / thresholds["ias15_global_scaled_rms"],
        "previous_global_scaled_rms": previous["scaled_state"]["global_scaled_rms"],
        "current_global_scaled_rms": current["scaled_state"]["global_scaled_rms"],
    }


def _extended_elements(run: RunData) -> dict[str, dict[str, np.ndarray]]:
    fields = ("semi_major_axis_m", "eccentricity", *ANGLE_FIELDS)
    output = {
        name: {field: np.empty(len(run.times), dtype=np.float64) for field in fields}
        for name in run.body_names[1:]
    }
    for index in range(len(run.times)):
        for element in heliocentric_elements_for_state(_run_state(run, index), run.body_names):
            row = output[element.body_name]
            for field in fields:
                row[field][index] = getattr(element, field)
    return output


def _wrapped(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _all_element_differences(left: dict[str, dict[str, np.ndarray]], right: dict[str, dict[str, np.ndarray]], times: np.ndarray) -> dict[str, Any]:
    output = {}
    for body_name in left:
        body = {}
        for field in left[body_name]:
            if field == "semi_major_axis_m":
                reference = max(abs(0.5 * (left[body_name][field][0] + right[body_name][field][0])), 1.0)
                values = np.abs(left[body_name][field] - right[body_name][field]) / reference
                units = "relative"
            elif field == "eccentricity":
                values = np.abs(left[body_name][field] - right[body_name][field])
                units = "absolute"
            else:
                values = np.abs(_wrapped(left[body_name][field] - right[body_name][field]))
                units = "rad"
            worst = int(np.argmax(values))
            body[field] = {
                "maximum_abs": float(values[worst]),
                "rms": float(np.sqrt(np.mean(values**2))),
                "worst_epoch_years": float(times[worst]),
                "units": units,
            }
        output[body_name] = body
    return output


def _naff_lite(times: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    cadence = float(times[1] - times[0])
    centered = np.asarray(values, dtype=np.complex128) - np.mean(values)
    spectrum = np.fft.fft(centered * np.hanning(len(centered)))
    frequencies = np.fft.fftfreq(len(centered), d=cadence)
    power = np.abs(spectrum) ** 2
    power[0] = 0.0
    index = int(np.argmax(power))
    refined_index = float(index)
    if 0 < index < len(power) - 1:
        left, center, right = np.log(np.maximum(power[index - 1:index + 2], 1.0e-300))
        denominator = left - 2.0 * center + right
        if denominator != 0.0:
            refined_index += 0.5 * (left - right) / denominator
    if refined_index > len(power) / 2.0:
        refined_index -= len(power)
    frequency = refined_index / (len(power) * cadence)
    resolution = 1.0 / (len(power) * cadence)
    nyquist = 0.5 / cadence
    return {
        "frequency_cycles_per_year": float(frequency),
        "frequency_arcsec_per_year": float(frequency * 360.0 * 3600.0),
        "fourier_resolution_cycles_per_year": resolution,
        "nyquist_cycles_per_year": nyquist,
        "alias_risk": abs(frequency) < 2.0 * resolution or abs(frequency) > 0.8 * nyquist,
    }


def _secular_frequencies(times: np.ndarray, elements: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    output = {}
    for body_name, values in elements.items():
        eccentricity_vector = values["eccentricity"] * np.exp(1j * values["longitude_perihelion_rad"])
        inclination_vector = np.sin(0.5 * values["inclination_rad"]) * np.exp(1j * values["longitude_ascending_node_rad"])
        output[body_name] = {"eccentricity_mode": _naff_lite(times, eccentricity_vector), "inclination_mode": _naff_lite(times, inclination_vector)}
    return output


def _frequency_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for body_name in left:
        output[body_name] = {}
        for mode in ("eccentricity_mode", "inclination_mode"):
            difference = abs(left[body_name][mode]["frequency_cycles_per_year"] - right[body_name][mode]["frequency_cycles_per_year"])
            output[body_name][mode] = {
                "absolute_difference_cycles_per_year": difference,
                "absolute_difference_arcsec_per_year": difference * 360.0 * 3600.0,
                "candidate": left[body_name][mode],
                "reference": right[body_name][mode],
                "alias_risk": left[body_name][mode]["alias_risk"] or right[body_name][mode]["alias_risk"],
            }
    return output


def _momentum_com(run: RunData) -> dict[str, Any]:
    momentum = np.sum(run.masses[None, :, None] * run.velocities, axis=1)
    total_mass = float(np.sum(run.masses))
    com_position = np.sum(run.masses[None, :, None] * run.positions, axis=1) / total_mass
    com_velocity = momentum / total_mass
    momentum_delta = np.linalg.norm(momentum - momentum[0], axis=1)
    position_delta = np.linalg.norm(com_position - com_position[0], axis=1)
    velocity_delta = np.linalg.norm(com_velocity - com_velocity[0], axis=1)
    return {
        "linear_momentum_initial_norm_kg_m_per_s": float(np.linalg.norm(momentum[0])),
        "linear_momentum_max_abs_change_kg_m_per_s": float(np.max(momentum_delta)),
        "com_position_initial_norm_m": float(np.linalg.norm(com_position[0])),
        "com_position_max_displacement_m": float(np.max(position_delta)),
        "com_velocity_initial_norm_m_per_s": float(np.linalg.norm(com_velocity[0])),
        "com_velocity_max_abs_change_m_per_s": float(np.max(velocity_delta)),
    }


def _figure_inventory(manifest: dict[str, Any], payload: dict[str, Any], runs: dict[str, RunData], current: dict[str, Any], previous: dict[str, Any], energy: dict[str, Any], all_elements: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(manifest["paths"]["figure_directory"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    created = []
    body_names = list(runs["new"].body_names)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(body_names))
    ax.bar(x - 0.18, [previous["physical"]["per_body"][name]["rms"] for name in body_names], 0.36, label="0.5 d vs 0.25 d")
    ax.bar(x + 0.18, [current["physical"]["per_body"][name]["rms"] for name in body_names], 0.36, label="0.25 d vs 0.125 d")
    ax.set_yscale("log")
    ax.set_xticks(x, [name.replace(" barycenter", "") for name in body_names], rotation=35, ha="right")
    ax.set_ylabel("Scaled Cartesian RMS")
    ax.legend()
    fig.tight_layout()
    created.append(_save_figure(fig, figure_dir / "physical_state_convergence.png"))

    prediction_gate = payload["corrected_energy"]["prediction_gate"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(runs["new"].times, energy["history"], linewidth=0.8, label="0.125 d recomputed")
    ax.fill_between(runs["new"].times, prediction_gate.pop("lower_history"), prediction_gate.pop("upper_history"), alpha=0.25, label="preregistered envelope")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Corrected-energy relative change")
    ax.legend()
    fig.tight_layout()
    created.append(_save_figure(fig, figure_dir / "corrected_energy_prediction.png"))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    inner = list(INNER_BODIES)
    x = np.arange(len(inner))
    axes[0].bar(x, [all_elements[name]["semi_major_axis_m"]["maximum_abs"] for name in inner])
    axes[0].axhline(manifest["thresholds"]["inner_planet_semimajor_axis_history_max_relative"], color="black", linestyle="--")
    axes[0].set_ylabel("max relative a difference")
    axes[1].bar(x, [all_elements[name]["eccentricity"]["maximum_abs"] for name in inner])
    axes[1].set_ylabel("max absolute e difference")
    axes[1].set_xticks(x, [name.replace(" barycenter", "") for name in inner])
    fig.tight_layout()
    created.append(_save_figure(fig, figure_dir / "inner_orbital_elements.png"))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(runs["new"].times, np.abs(runs["candidate"].progress["megno"] - runs["new"].progress["megno"]), label="MEGNO difference")
    ax.plot(runs["new"].times, np.abs((runs["candidate"].progress["lcn_1_per_year"] - runs["new"].progress["lcn_1_per_year"]) * 1_000_000.0), label="accumulated LCN difference")
    ax.set_yscale("log")
    ax.set_xlabel("Time (years)")
    ax.legend()
    fig.tight_layout()
    created.append(_save_figure(fig, figure_dir / "megno_lcn_convergence.png"))

    return created


def _save_figure(fig: Any, path: Path) -> dict[str, Any]:
    fig.savefig(path, dpi=150, metadata={"Software": "mini_ephemeris Step 3e"})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return _artifact(path)


def _markdown(payload: dict[str, Any]) -> str:
    if payload["final_status"] == "BLOCKED":
        lines = ["# BLOCKED", "", "Step 3e could not produce a defensible scientific classification.", "", "## Failures", ""]
        lines.extend("- " + str(failure) for failure in payload["failures"])
        lines.extend(["", "No additional trajectory, Stage 4 command, or 10 Myr integration was launched.", ""])
        return chr(10).join(lines)
    lines = [
        f"# {payload['final_status']}",
        "",
        "Step 3e ran one preregistered 0.125-day, 1 Myr compiled-C tangent/MEGNO WHFast lane and compared it with the unchanged 0.25-day production candidate.",
        "",
        "## Decision",
        "",
        f"- Production candidate: `0.25 day`.",
        f"- Raw physical convergence: `{payload['criteria']['physical_state_raw']['passed']}`; phase-aware state result: `{payload['criteria']['physical_state']['passed']}`.",
        f"- Corrected-energy prediction: `{payload['criteria']['corrected_energy']['passed']}`.",
        f"- Integrity: `{payload['criteria']['integrity']['passed']}`.",
        "- No Stage 4 or 10 Myr command was provided or executed.",
        f"- Smallest follow-up: {payload['smallest_follow_up_diagnostic']}",
        "",
        "## Runtime And Integrity",
        "",
        f"- Runtime: `{payload['runs']['new']['runtime_seconds']:.6f}` s; throughput `{payload['runs']['new']['throughput_years_per_second']:.6f}` yr/s.",
        f"- Samples/state rows/archive snapshots: `{payload['runs']['new']['scientific_samples']}` / `{payload['runs']['new']['state_rows']}` / `{payload['runs']['new']['archive_snapshots']}`.",
        f"- Callback/nonfinite counts: `{payload['runs']['new']['callback_invocations']}` / `{payload['runs']['new']['nonfinite_result_count']}`.",
        f"- Fingerprint: `{payload['runs']['new']['configuration_fingerprint']}`.",
        "- In-lane prefix: {:.6f} s, projected {:.6f} s, PASS; reconstructed conservatively from the persisted pre-Popen launch UTC after correcting the row-order parser.".format(payload["operational_prefix_gate"]["elapsed_seconds"], payload["operational_prefix_gate"]["projected_runtime_seconds"]),
        "",
        "## Physical And Orbital Convergence",
        "",
        f"- Global scaled RMS old/new: `{payload['comparisons']['previous']['physical']['global_scaled_rms']:.12g}` / `{payload['comparisons']['current']['physical']['global_scaled_rms']:.12g}`; ratio `{payload['criteria']['physical_state_raw']['global_ratio']:.12g}`.",
        f"- Worst inner ratio: `{payload['criteria']['physical_state_raw']['worst_inner_body']}` at `{payload['criteria']['physical_state_raw']['worst_inner_ratio']:.12g}`.",
        f"- Worst semimajor-axis relative difference: `{payload['criteria']['semimajor_axis']['worst_body']}` `{payload['criteria']['semimajor_axis']['worst_value']:.12g}`.",
        f"- Worst eccentricity difference: `{payload['criteria']['eccentricity']['worst_body']}` `{payload['criteria']['eccentricity']['worst_value']:.12g}`.",
        f"- Mercury perihelion-rate difference: `{payload['criteria']['mercury_perihelion']['difference_arcsec_per_century']:.12g}` arcsec/century.",
        "",
        "## Failed Thresholds",
        "",
        "- Global RMS ratio {:.12g} exceeds {:.12g} by {:.12g} at {:.12g} years.".format(payload["failed_thresholds"]["global_scaled_rms_ratio"]["value"], payload["failed_thresholds"]["global_scaled_rms_ratio"]["maximum"], payload["failed_thresholds"]["global_scaled_rms_ratio"]["excess"], payload["failed_thresholds"]["global_scaled_rms_ratio"]["worst_epoch_years"]),
        "- Mercury RMS ratio {:.12g} misses strict less-than-1 improvement by {:.12g} at {:.12g} years.".format(payload["failed_thresholds"]["mercury_strict_rms_improvement"]["value"], payload["failed_thresholds"]["mercury_strict_rms_improvement"]["excess"], payload["failed_thresholds"]["mercury_strict_rms_improvement"]["worst_epoch_years"]),
        "- Phase fallback orientation: {} {:.12g} rad versus {:.12g} rad.".format(payload["failed_thresholds"]["phase_orientation"]["body"], payload["failed_thresholds"]["phase_orientation"]["value_rad"], payload["failed_thresholds"]["phase_orientation"]["maximum_rad"]),
        "- Phase/orientation ratio: {} {:.12g} versus minimum {:.12g}.".format(payload["failed_thresholds"]["phase_over_orientation"]["body"], payload["failed_thresholds"]["phase_over_orientation"]["value"], payload["failed_thresholds"]["phase_over_orientation"]["minimum"]),
        "",
        "## Tangent And Chaos",
        "",
        f"- Final tangent cosine: `{payload['comparisons']['current']['tangent']['final_direction_cosine']:.12g}`; direction RMS old/new `{payload['comparisons']['previous']['tangent']['direction_discrepancy_rms']:.12g}` / `{payload['comparisons']['current']['tangent']['direction_discrepancy_rms']:.12g}`.",
        f"- Final MEGNO difference: `{payload['comparisons']['current']['megno']['final_abs_difference']:.12g}`.",
        f"- Final accumulated LCN difference: `{payload['comparisons']['current']['lcn']['final_accumulated_abs_difference']:.12g}`.",
        "",
        "## Corrected Energy",
        "",
        f"- Recomputed 0.125-day fitted change over 1 Myr: `{payload['corrected_energy']['new']['fitted_change_over_history']:.12g}`.",
        f"- Recomputed q: `{payload['corrected_energy']['prediction_gate']['q']:.12g}`; interval `{payload['corrected_energy']['prediction_gate']['q_interval']}`.",
        f"- Same-sign 100-kyr blocks: `{payload['corrected_energy']['prediction_gate']['same_sign_blocks']}` of 10.",
        f"- Maximum prediction-envelope excess: `{payload['corrected_energy']['prediction_gate']['history_max_envelope_excess']:.12g}`.",
        f"- Historical Step 3 trend gate (diagnostic only after manifest 16): `{payload['corrected_energy']['historical_step3_trend_gate_passed']}`.",
        "",
        "## Criteria",
        "",
        "| Criterion | Result |",
        "| --- | ---: |",
    ]
    for name, result in payload["criteria"].items():
        lines.append(f"| {name} | {'PASS' if result['passed'] else 'FAIL'} |")
    lines.extend(["", "## Artifacts", "", f"- Manifest SHA-256: `{payload['manifest_sha256']}`.", f"- New raw artifact count: `{len(payload['artifact_inventory']['new_lane'])}`.", f"- Figure count: `{len(payload['artifact_inventory']['figures'])}`.", ""])
    if payload["failures"]:
        lines.extend(["## Failures", "", *[f"- {failure}" for failure in payload["failures"]], ""])
    return "\n".join(lines)


def analyze(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Step 3e manifest")
    manifest_hash = sha256_file(manifest_path)
    base: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "historical_results_unchanged": manifest["historical_results_immutable"],
        "primary_mechanism": "SYSTEMATIC_WHFAST_STEP_BIAS",
        "step3_diagnosis_status": "STEP3_NUMERICAL_FLOOR_CHARACTERIZED",
        "stage4_command_provided": False,
        "stage4_or_10myr_executed": False,
        "failures": [],
    }
    try:
        audit_payload = audit(manifest_path, write=False)
        manifest10_path = Path(manifest["paths"]["manifest_10"])
        manifest10 = _load_json(manifest10_path, "manifest 10")
        manifest11_path = Path(manifest["paths"]["manifest_11"])
        manifest11 = _load_json(manifest11_path, "manifest 11")
        run05_definition = next(item for item in manifest10["runs"] if item["id"] == "m0_conv_0p5d_1myr_s12345")
        run05 = _load_run(manifest10_path, manifest10, run05_definition)
        run025 = _load_run(manifest11_path, manifest11, manifest11["decisive_run"])
        run0125 = _load_run(manifest_path, manifest, manifest["new_lane"])
        runs = {"coarse": run05, "candidate": run025, "new": run0125}
        rebound = optional_import_module("rebound")
        new_paths = output_paths(Path(manifest["new_lane"]["output_dir"]), manifest["new_lane"]["id"], None)
        archive = rebound.Simulationarchive(str(new_paths["archive"]))
        _require(len(archive) == manifest["endpoint_semantics"]["expected_archive_snapshots"], "New archive snapshot count mismatch.")
        _require(int(archive[-1].steps_done) == manifest["endpoint_semantics"]["expected_full_steps"], "New archive step count mismatch.")
        _require(run0125.integrity["callback_invocations"] == manifest["endpoint_semantics"]["expected_callback_invocations"], "New callback total mismatch.")

        previous = {
            "physical": _pair_physical(run05, run025),
            "tangent": _pair_tangent(run05, run025),
            "orbital_elements": _pair_elements(run05, run025),
            "megno": _pair_scalar(run05, run025, "megno"),
            "lcn": _pair_lcn(run05, run025),
            "angular_momentum": _pair_angular(run05, run025),
        }
        current = {
            "physical": _pair_physical(run025, run0125),
            "tangent": _pair_tangent(run025, run0125),
            "orbital_elements": _pair_elements(run025, run0125),
            "megno": _pair_scalar(run025, run0125, "megno"),
            "lcn": _pair_lcn(run025, run0125),
            "angular_momentum": _pair_angular(run025, run0125),
        }
        current["rtn"] = _rtn_comparison(run025, run0125)
        previous_phase = _phase_diagnostics(run05, run025)
        current_phase = _phase_diagnostics(run025, run0125)
        phase_gate = _phase_gate(manifest, previous_phase, current_phase)

        previous_elements = _extended_elements(run025)
        current_elements = _extended_elements(run0125)
        all_element_differences = _all_element_differences(previous_elements, current_elements, run0125.times)
        secular = _frequency_comparison(
            _secular_frequencies(run025.times, previous_elements),
            _secular_frequencies(run0125.times, current_elements),
        )

        energy025_paths = output_paths(Path(manifest11["decisive_run"]["output_dir"]), manifest11["decisive_run"]["id"], None)
        energy025 = _energy_reconstruction(run025, energy025_paths["state"], energy025_paths["progress"])
        energy0125 = _energy_reconstruction(run0125, new_paths["state"], new_paths["progress"])
        energy_gate = energy_prediction_gate(manifest, run0125.times, energy0125["history"], energy0125["statistics"])
        telemetry_exact = all(value == 0.0 for value in energy0125["telemetry_reproduction_max_abs"].values())
        _require(telemetry_exact, "New energy telemetry does not exactly reproduce from state rows.")

        thresholds = manifest["thresholds"]
        global_ratio = current["physical"]["global_scaled_rms"] / previous["physical"]["global_scaled_rms"]
        inner_ratios = {body: current["physical"]["per_body"][body]["rms"] / previous["physical"]["per_body"][body]["rms"] for body in INNER_BODIES}
        raw_physical_pass = (
            current["physical"]["global_scaled_rms"] < previous["physical"]["global_scaled_rms"]
            and all(current["physical"]["per_body"][body]["rms"] < previous["physical"]["per_body"][body]["rms"] for body in INNER_BODIES)
            and global_ratio <= thresholds["global_scaled_physical_rms_fine_over_coarse_max"]
        )
        element_body = current["orbital_elements"]["per_body"]
        a_checks = {body: element_body[body]["semimajor_axis_max_relative_difference"] <= thresholds["inner_planet_semimajor_axis_history_max_relative"] for body in INNER_BODIES}
        e_checks = {body: element_body[body]["eccentricity_max_abs_difference"] <= (thresholds["mercury_eccentricity_history_max_abs"] if body == "mercury barycenter" else thresholds["inner_planet_eccentricity_history_max_abs"]) for body in INNER_BODIES}
        peri025 = _perihelion_rate(run025)
        peri0125 = _perihelion_rate(run0125)
        peri_difference = abs(peri025["mean_rate_arcsec_per_century"] - peri0125["mean_rate_arcsec_per_century"])
        tangent_pass = current["tangent"]["final_direction_cosine"] >= thresholds["final_tangent_direction_cosine_min"] and current["tangent"]["direction_discrepancy_rms"] < previous["tangent"]["direction_discrepancy_rms"]
        megno_pass = current["megno"]["final_abs_difference"] <= thresholds["final_megno_difference_max"] and current["megno"]["history_rms_difference"] <= thresholds["megno_history_rms_difference_max"]
        lcn_pass = current["lcn"]["final_accumulated_abs_difference"] <= thresholds["final_lcn_accumulated_difference_max"]
        angular025 = _series_metrics(run025.times, run025.progress["angular_momentum_rel_drift"])
        angular0125 = _series_metrics(run0125.times, run0125.progress["angular_momentum_rel_drift"])
        angular_pass = max(angular025["max_abs"], angular0125["max_abs"]) <= thresholds["angular_momentum_rel_drift_max_per_run"]
        integrity_pass = run025.integrity["passed"] and run0125.integrity["passed"] and len(archive) == 11 and telemetry_exact
        nonphase_pass = all(a_checks.values()) and all(e_checks.values()) and peri_difference <= thresholds["mercury_perihelion_rate_difference_arcsec_per_century_max"]
        physical_pass = raw_physical_pass or (phase_gate["passed"] and nonphase_pass)
        worst_a = max(INNER_BODIES, key=lambda body: element_body[body]["semimajor_axis_max_relative_difference"])
        worst_e = max(INNER_BODIES, key=lambda body: element_body[body]["eccentricity_max_abs_difference"] / (thresholds["mercury_eccentricity_history_max_abs"] if body == "mercury barycenter" else thresholds["inner_planet_eccentricity_history_max_abs"]))
        worst_ratio_body = max(INNER_BODIES, key=inner_ratios.get)
        historical_trend_limit = max(0.25 * energy0125["statistics"]["max_abs"], 1.0e-10)
        historical_trend_pass = abs(energy0125["statistics"]["fitted_change_over_history"]) <= historical_trend_limit
        phase_orientation_by_body = {name: max(body["orbital_elements"][field]["maximum_abs"] for field in ("inclination_rad", "longitude_ascending_node_rad", "argument_perihelion_rad")) for name, body in current_phase["bodies"].items()}
        phase_ratio_by_body = {name: body["phase_angle_over_orientation_angle"] for name, body in current_phase["bodies"].items()}
        orientation_body = max(phase_orientation_by_body, key=phase_orientation_by_body.get)
        phase_ratio_body = min(phase_ratio_by_body, key=phase_ratio_by_body.get)
        failed_thresholds = {
            "global_scaled_rms_ratio": {"value": global_ratio, "maximum": thresholds["global_scaled_physical_rms_fine_over_coarse_max"], "excess": global_ratio - thresholds["global_scaled_physical_rms_fine_over_coarse_max"], "worst_epoch_years": current["physical"]["worst_epoch_years"]},
            "mercury_strict_rms_improvement": {"value": inner_ratios["mercury barycenter"], "strict_maximum": 1.0, "excess": inner_ratios["mercury barycenter"] - 1.0, "worst_epoch_years": current["physical"]["per_body"]["mercury barycenter"]["worst_epoch_years"]},
            "phase_orientation": {"body": orientation_body, "value_rad": phase_orientation_by_body[orientation_body], "maximum_rad": thresholds["manifest16_phase"]["orientation_angle_abs_rad_max"], "excess_rad": phase_orientation_by_body[orientation_body] - thresholds["manifest16_phase"]["orientation_angle_abs_rad_max"]},
            "phase_over_orientation": {"body": phase_ratio_body, "value": phase_ratio_by_body[phase_ratio_body], "minimum": thresholds["manifest16_phase"]["phase_angle_over_orientation_min"], "shortfall": thresholds["manifest16_phase"]["phase_angle_over_orientation_min"] - phase_ratio_by_body[phase_ratio_body]},
        }
        criteria = {
            "integrity": {"passed": integrity_pass},
            "physical_state_raw": {"passed": raw_physical_pass, "global_ratio": global_ratio, "limit": thresholds["global_scaled_physical_rms_fine_over_coarse_max"], "inner_ratios": inner_ratios, "worst_inner_body": worst_ratio_body, "worst_inner_ratio": inner_ratios[worst_ratio_body]},
            "phase_aware_interpretation": phase_gate,
            "physical_state": {"passed": physical_pass, "used_phase_supersession": not raw_physical_pass and physical_pass},
            "semimajor_axis": {"passed": all(a_checks.values()), "per_body": a_checks, "worst_body": worst_a, "worst_value": element_body[worst_a]["semimajor_axis_max_relative_difference"]},
            "eccentricity": {"passed": all(e_checks.values()), "per_body": e_checks, "worst_body": worst_e, "worst_value": element_body[worst_e]["eccentricity_max_abs_difference"]},
            "mercury_perihelion": {"passed": peri_difference <= thresholds["mercury_perihelion_rate_difference_arcsec_per_century_max"], "difference_arcsec_per_century": peri_difference, "limit": thresholds["mercury_perihelion_rate_difference_arcsec_per_century_max"]},
            "tangent": {"passed": tangent_pass},
            "megno": {"passed": megno_pass},
            "lcn": {"passed": lcn_pass},
            "corrected_energy": {"passed": energy_gate["passed"]},
            "angular_momentum": {"passed": angular_pass},
        }
        failures = [name for name, result in criteria.items() if name not in {"physical_state_raw", "phase_aware_interpretation"} and not result["passed"]]
        final_status = "STEP3E_025_DAY_PRODUCTION_VALIDATED" if not failures else "STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED"
        prefix_path = Path(manifest["paths"]["output_root"]) / "operations/prefix_gate.json"
        prefix_gate_payload = _load_json(prefix_path, "prefix gate")
        _require(prefix_gate_payload.get("passed") is True, "Operational prefix gate did not pass.")
        new_inventory = [_artifact(path) for path in [new_paths[name] for name in ("progress", "state", "status", "summary", "restart", "archive")] + [Path(manifest["new_lane"]["log_path"])] ]
        run_summary = {**run0125.integrity, "archive_snapshots": len(archive), "steps_done": int(archive[-1].steps_done)}
        base.update(
            final_status=final_status,
            failures=failures,
            failed_thresholds=failed_thresholds,
            audit=audit_payload,
            operational_prefix_gate=prefix_gate_payload,
            runs={"candidate": run025.integrity, "new": run_summary},
            comparisons={"previous": previous, "current": current},
            phase_diagnostics_10k={"previous": previous_phase, "current": current_phase, "gate": phase_gate},
            all_orbital_elements=all_element_differences,
            secular_frequency_diagnostics=secular,
            mercury_perihelion={"candidate": peri025, "reference": peri0125},
            corrected_energy={
                "candidate": energy025["statistics"],
                "new": energy0125["statistics"],
                "telemetry_reproduction_max_abs": energy0125["telemetry_reproduction_max_abs"],
                "prediction_gate": energy_gate,
                "historical_step3_trend_gate_passed": historical_trend_pass,
                "historical_step3_trend_limit": historical_trend_limit,
                "historical_step3_trend_role": "diagnostic_only_after_manifest_16",
            },
            angular_momentum={"candidate": angular025, "new": angular0125, "pair": current["angular_momentum"]},
            linear_momentum_and_com={"candidate": _momentum_com(run025), "new": _momentum_com(run0125)},
            criteria=criteria,
            artifact_inventory={"new_lane": new_inventory, "figures": []},
            step3_convergence_problem_closed=final_status == "STEP3E_025_DAY_PRODUCTION_VALIDATED",
            validated_production_timestep_days=0.25 if final_status == "STEP3E_025_DAY_PRODUCTION_VALIDATED" else None,
            smallest_follow_up_diagnostic=("No further Step 3 convergence diagnostic is required." if final_status == "STEP3E_025_DAY_PRODUCTION_VALIDATED" else "Perform an offline windowed Mercury RTN and osculating-angle decomposition of the existing 0.5-day, 0.25-day, and 0.125-day stored rows to localize the nonmonotonic Mercury state error and test whether it is secular-orientation leakage or bounded phase beating; do not start another integration."),
        )
        figures = _figure_inventory(manifest, base, runs, current, previous, energy0125, all_element_differences)
        base["artifact_inventory"]["figures"] = figures
    except Exception as exc:
        base = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "final_status": "BLOCKED",
            "failures": [str(exc)],
            "criteria": {"integrity": {"passed": False}},
            "artifact_inventory": {"new_lane": [], "figures": []},
            "step3_convergence_problem_closed": False,
            "validated_production_timestep_days": None,
            "historical_results_unchanged": manifest["historical_results_immutable"],
            "stage4_command_provided": False,
            "stage4_or_10myr_executed": False,
        }
    _require(base["final_status"] in FINAL_STATUSES, "Invalid Step 3e final status.")
    _finite_json(base)
    _atomic_json(Path(manifest["paths"]["report_json"]), base)
    _atomic_text(Path(manifest["paths"]["report_markdown"]), _markdown(base))
    print(f"[step3e] {base['final_status']}")
    return 0 if base["final_status"] == "STEP3E_025_DAY_PRODUCTION_VALIDATED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and analyze the preregistered M0 Step 3e lane.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("command", choices=("audit", "run", "analyze"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            audit(args.manifest, write=True)
            raise SystemExit(0)
        if args.command == "run":
            raise SystemExit(run_lane(args.manifest))
        raise SystemExit(analyze(args.manifest))
    except ConvergenceError as exc:
        raise SystemExit(f"Step 3e error: {exc}") from exc


if __name__ == "__main__":
    main()
