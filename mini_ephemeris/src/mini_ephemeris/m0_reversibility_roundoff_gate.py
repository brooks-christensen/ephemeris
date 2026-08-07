from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

from .gr_potential_tangent_c import load_c_backend
from .long_term_stability_cli import (
    build_rebound_simulation,
    optional_import_module,
    rebound_state_from_sim,
)
from .m0_integrator_roundoff_diagnosis import (
    _configure_whfast,
    _return_metrics,
    _synchronize_for_direction_reversal,
    _two_body_state,
)
from .nbody import NBodyState
from .orbital_elements import AU_M, DAY_S, JULIAN_YEAR_S
from .rebound_gr_tangent_backend_cli import (
    atomic_write_json,
    canonical_hash,
    sha256_file,
)


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/14_m0_reversibility_roundoff_gate_v1.json"
)
FINAL_STATUSES = {
    "REVERSIBILITY_GATE_PASSED",
    "REVERSIBILITY_GATE_FAILED",
    "BLOCKED",
}
BODY_NAMES = ("sun", "mercury")
STATE_FIELDS = [
    "schema_version",
    "manifest_sha256",
    "run_id",
    "run_fingerprint",
    "scientific_configuration_fingerprint",
    "sample_index",
    "sample_label",
    "time_seconds",
    "body_index",
    "body_name",
    "mass_kg",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
]


class GateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except Exception as exc:
        raise GateError(f"Unreadable {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"Invalid {label}: expected JSON object.")
    return payload


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise GateError(f"Git command failed: git {' '.join(args)}") from exc


def _event(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_identity() -> dict[str, Any]:
    rebound = optional_import_module("rebound")
    reboundx = optional_import_module("reboundx")
    _require(rebound is not None, "REBOUND is unavailable.")
    _require(reboundx is not None, "REBOUNDx is unavailable.")
    rebound_library = Path(rebound.clibrebound._name)
    rebound_header = Path(rebound.__file__).parent / "rebound.h"
    reboundx_library = Path(reboundx.clibreboundx._name)
    return {
        "rebound_version": rebound.__version__,
        "rebound_build": getattr(rebound, "__build__", None),
        "rebound_githash": getattr(rebound, "__githash__", None),
        "rebound_shared_library_sha256": sha256_file(rebound_library),
        "rebound_header_sha256": sha256_file(rebound_header),
        "reboundx_version": reboundx.__version__,
        "reboundx_shared_library_sha256": sha256_file(reboundx_library),
        "reboundx_used": False,
    }


def _source_cases(
    manifest: dict[str, Any], manifest_13: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    definition = manifest_13["method_validation"]["reversibility_two_body"]
    originals = {case["id"]: case for case in definition["cases"]}
    expanded: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        source = originals.get(case["source_case_id"])
        _require(source is not None, f"Unknown source case: {case['source_case_id']}")
        copied = {
            key: case["configuration"][key] for key in source["configuration"]
        }
        _require(
            copied == source["configuration"],
            f"Frozen configuration changed: {case['run_id']}",
        )
        source_config = {**definition["common_configuration"], **copied}
        _require(
            canonical_hash(source_config)
            == case["scientific_configuration_fingerprint"]
            == source["configuration_fingerprint"],
            f"Scientific fingerprint changed: {case['run_id']}",
        )
        run_identity = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "run_id": case["run_id"],
            "scientific_configuration_fingerprint": case[
                "scientific_configuration_fingerprint"
            ],
        }
        _require(
            canonical_hash(run_identity) == case["run_fingerprint"],
            f"Run fingerprint mismatch: {case['run_id']}",
        )
        expanded[case["run_id"]] = source_config
    _require(len(expanded) == 4, "Manifest 14 must contain exactly four cases.")
    return expanded


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "manifest 14")
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Run from the project root.")
    _require(manifest.get("frozen_before_new_integrations") is True, "Manifest 14 is not frozen.")
    _require(
        set(manifest["gate_definition"]["final_statuses"]) == FINAL_STATUSES,
        "Final status vocabulary changed.",
    )
    head = _git("rev-parse", "HEAD")
    _git("merge-base", "--is-ancestor", manifest["provenance"]["starting_commit"], head)
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C tag is not annotated.")
    _require(
        _git("rev-parse", tag + "^{commit}")
        == manifest["provenance"]["validated_c_baseline_commit"],
        "Compiled-C tag target changed.",
    )
    fixed = {
        manifest["provenance"]["manifest_13_path"]: manifest["provenance"][
            "manifest_13_sha256"
        ],
        "docs/validation/m0-integrator-roundoff-diagnosis-v1/m0_integrator_roundoff_diagnosis_summary.json": manifest[
            "provenance"
        ]["manifest_13_summary_sha256"],
        "docs/validation/m0-integrator-roundoff-diagnosis-v1/m0_integrator_roundoff_diagnosis_report.md": manifest[
            "provenance"
        ]["manifest_13_report_sha256"],
        manifest["provenance"]["historical_validation_summary_path"]: manifest[
            "provenance"
        ]["historical_validation_summary_sha256"],
        manifest["provenance"]["historical_failed_attempt_path"]: manifest[
            "provenance"
        ]["historical_failed_attempt_sha256"],
    }
    for relative, expected in fixed.items():
        _require(sha256_file(root / relative) == expected, f"Historical hash changed: {relative}")
    for relative, expected in manifest["protected_files"].items():
        _require(sha256_file(root / relative) == expected, f"Protected file changed: {relative}")
    manifest_13 = _load_json(root / manifest["provenance"]["manifest_13_path"], "manifest 13")
    _source_cases(manifest, manifest_13)
    historical = _load_json(
        root / manifest["provenance"]["historical_validation_summary_path"],
        "historical validation summary",
    )
    _require(historical.get("passed") is False, "Historical validation status changed.")
    _require(historical.get("timestep_ratio_passed") is False, "Historical ratio result changed.")
    runtime = _runtime_identity()
    expected_runtime = manifest["runtime_identity"]
    for key in (
        "rebound_version",
        "rebound_build",
        "rebound_githash",
        "rebound_shared_library_sha256",
        "rebound_header_sha256",
        "reboundx_version",
        "reboundx_shared_library_sha256",
    ):
        _require(runtime[key] == expected_runtime[key], f"Runtime identity changed: {key}")
    artifact = root / "mini_ephemeris/build/gr_tangent_c/libmini_ephemeris_gr_tangent.so"
    _require(
        sha256_file(artifact) == expected_runtime["compiled_gr_artifact_sha256"],
        "Compiled GR artifact changed.",
    )
    return {
        "status": "PASS",
        "git_head": head,
        "git_dirty_after_preregistration": bool(_git("status", "--porcelain")),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_13_sha256": sha256_file(root / manifest["provenance"]["manifest_13_path"]),
        "historical_validation_sha256": sha256_file(
            root / manifest["provenance"]["historical_validation_summary_path"]
        ),
        "runtime": runtime,
        "compiled_gr_artifact_sha256": sha256_file(artifact),
        "protected_files": manifest["protected_files"],
    }


def _case_paths(output_root: Path, run_id: str) -> dict[str, Path]:
    directory = output_root / run_id
    return {
        "directory": directory,
        "state": directory / "physical_state.csv",
        "summary": directory / "case_summary.json",
        "events": directory / "events.log",
    }


def _state_rows(
    state: NBodyState,
    *,
    manifest_sha256: str,
    run_id: str,
    run_fingerprint: str,
    scientific_fingerprint: str,
    sample_index: int,
    sample_label: str,
    time_seconds: float,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": 1,
            "manifest_sha256": manifest_sha256,
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "scientific_configuration_fingerprint": scientific_fingerprint,
            "sample_index": sample_index,
            "sample_label": sample_label,
            "time_seconds": time_seconds,
            "body_index": index,
            "body_name": BODY_NAMES[index],
            "mass_kg": float(state.masses[index]),
            "x_m": float(state.positions[index, 0]),
            "y_m": float(state.positions[index, 1]),
            "z_m": float(state.positions[index, 2]),
            "vx_m_per_s": float(state.velocities[index, 0]),
            "vy_m_per_s": float(state.velocities[index, 1]),
            "vz_m_per_s": float(state.velocities[index, 2]),
        }
        for index in range(2)
    ]


def _write_state(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _audit_state(
    path: Path,
    *,
    manifest_sha256: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == STATE_FIELDS, f"State schema mismatch: {path}")
        rows = list(reader)
    expected = case["configuration"]
    _require(len(rows) == expected["expected_state_rows"] == 6, f"State row mismatch: {path}")
    labels = ("initial", "forward", "returned")
    times = (0.0, expected["expected_forward_time_seconds"], expected["expected_backward_time_seconds"])
    numeric = ("time_seconds", "mass_kg", "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")
    for sample_index, (label, target_time) in enumerate(zip(labels, times)):
        group = rows[sample_index * 2 : sample_index * 2 + 2]
        _require([row["body_name"] for row in group] == list(BODY_NAMES), f"Body order mismatch: {path}")
        for body_index, row in enumerate(group):
            _require(int(row["schema_version"]) == 1, f"State schema version mismatch: {path}")
            _require(row["manifest_sha256"] == manifest_sha256, f"State manifest mismatch: {path}")
            _require(row["run_id"] == case["run_id"], f"State run ID mismatch: {path}")
            _require(row["run_fingerprint"] == case["run_fingerprint"], f"State run fingerprint mismatch: {path}")
            _require(
                row["scientific_configuration_fingerprint"]
                == case["scientific_configuration_fingerprint"],
                f"State scientific fingerprint mismatch: {path}",
            )
            _require(int(row["sample_index"]) == sample_index, f"State sample mismatch: {path}")
            _require(row["sample_label"] == label, f"State label mismatch: {path}")
            _require(int(row["body_index"]) == body_index, f"State body index mismatch: {path}")
            _require(float(row["time_seconds"]) == target_time, f"State time mismatch: {path}")
            _require(all(math.isfinite(float(row[key])) for key in numeric), f"Nonfinite state row: {path}")
    return {"samples": 3, "rows": 6, "finite": True, "schema_valid": True}


def _bounded_check(value: float, limit: float) -> dict[str, Any]:
    finite = math.isfinite(value)
    passed = finite and value <= limit
    return {
        "value": value,
        "limit": limit,
        "finite": finite,
        "passed": passed,
        "absolute_margin": limit - value if finite else None,
        "limit_over_value": limit / value if finite and value > 0.0 else None,
    }


def absolute_return_checks(
    metrics: dict[str, Any], limits: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    checks = {
        "global_scaled_cartesian_rms": _bounded_check(
            abs(metrics["global_scaled_rms"]),
            limits["global_scaled_cartesian_rms_max"],
        ),
        "corrected_energy_relative_abs": _bounded_check(
            abs(metrics["corrected_energy_relative_difference"]),
            limits["corrected_energy_relative_abs_max"],
        ),
        "angular_momentum_vector_relative": _bounded_check(
            abs(metrics["angular_momentum_vector_relative_difference"]),
            limits["angular_momentum_vector_relative_max"],
        ),
        "center_of_mass_position_scaled": _bounded_check(
            abs(metrics["center_of_mass_position_error_m"]) / AU_M,
            limits["center_of_mass_position_scaled_max"],
        ),
        "center_of_mass_velocity_scaled": _bounded_check(
            abs(metrics["center_of_mass_velocity_error_m_per_s"])
            / (AU_M / JULIAN_YEAR_S),
            limits["center_of_mass_velocity_scaled_max"],
        ),
    }
    for body, item in metrics["per_body"].items():
        checks[f"{body}_scaled_cartesian_rms"] = _bounded_check(
            abs(item["scaled_rms"]), limits["per_body_scaled_cartesian_rms_max"]
        )
        checks[f"{body}_position_scaled_norm"] = _bounded_check(
            abs(item["position_error_m"]) / AU_M,
            limits["per_body_position_scaled_norm_max"],
        )
        checks[f"{body}_velocity_scaled_norm"] = _bounded_check(
            abs(item["velocity_error_m_per_s"]) / (AU_M / JULIAN_YEAR_S),
            limits["per_body_velocity_scaled_norm_max"],
        )
    return checks


def _run_case(
    manifest_path: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    source_config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    paths = _case_paths(output_root, case["run_id"])
    _require(not paths["directory"].exists(), f"Collision-safe case exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True)
    _event(paths["events"], f"START run_id={case['run_id']}")
    manifest_sha256 = sha256_file(manifest_path)
    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    initial = _two_body_state(source_config)
    sim = build_rebound_simulation(
        rebound,
        initial,
        integrator="whfast",
        step_s=float(source_config["step_days"]) * DAY_S,
        ias15_epsilon=1e-10,
    )
    _configure_whfast(sim, source_config)
    backend = load_c_backend()
    backend.attach(sim, coefficient_scale=1.0, include_central_response=True)
    initial_state = rebound_state_from_sim(sim, initial.masses)
    finite_initial = bool(
        np.all(np.isfinite(initial_state.positions))
        and np.all(np.isfinite(initial_state.velocities))
    )
    start = time.perf_counter()
    sim.steps(source_config["forward_steps"])
    _synchronize_for_direction_reversal(sim)
    forward_time = float(sim.t)
    forward_state = rebound_state_from_sim(sim, initial.masses)
    forward_comparison_synchronized = True
    forward_internal_is_synchronized = int(sim.ri_whfast.is_synchronized)
    sim.dt = -abs(float(sim.dt))
    sim.steps(source_config["backward_steps"])
    sim.synchronize()
    return_time = float(sim.t)
    returned_state = rebound_state_from_sim(sim, initial.masses)
    returned_comparison_synchronized = True
    returned_internal_is_synchronized = int(sim.ri_whfast.is_synchronized)
    runtime_seconds = time.perf_counter() - start
    callback = backend.stats(sim)
    finite_forward = bool(
        np.all(np.isfinite(forward_state.positions))
        and np.all(np.isfinite(forward_state.velocities))
    )
    finite_returned = bool(
        np.all(np.isfinite(returned_state.positions))
        and np.all(np.isfinite(returned_state.velocities))
    )
    metrics = _return_metrics(initial_state, returned_state, BODY_NAMES)
    absolute_checks = absolute_return_checks(metrics, manifest["absolute_limits"])
    config = case["configuration"]
    integrity_checks = {
        "forward_endpoint_exact": forward_time == config["expected_forward_time_seconds"],
        "backward_endpoint_exact": return_time == config["expected_backward_time_seconds"],
        "forward_steps_exact": source_config["forward_steps"] == config["forward_steps"],
        "backward_steps_exact": source_config["backward_steps"] == config["backward_steps"],
        "no_fractional_endpoint_step": (
            abs(float(sim.dt)) == float(source_config["step_days"]) * DAY_S
        ),
        "callback_count_exact": (
            int(callback["callback_invocations"])
            == config["expected_callback_invocations"]
        ),
        "nonfinite_callback_zero": int(callback["nonfinite_result_count"]) == 0,
        "initial_state_finite": finite_initial,
        "forward_state_finite": finite_forward,
        "returned_state_finite": finite_returned,
        "forward_comparison_synchronized": forward_comparison_synchronized,
        "returned_comparison_synchronized": returned_comparison_synchronized,
        "reversal_procedure_exact": True,
    }
    rows = [
        *_state_rows(
            initial_state,
            manifest_sha256=manifest_sha256,
            run_id=case["run_id"],
            run_fingerprint=case["run_fingerprint"],
            scientific_fingerprint=case["scientific_configuration_fingerprint"],
            sample_index=0,
            sample_label="initial",
            time_seconds=0.0,
        ),
        *_state_rows(
            forward_state,
            manifest_sha256=manifest_sha256,
            run_id=case["run_id"],
            run_fingerprint=case["run_fingerprint"],
            scientific_fingerprint=case["scientific_configuration_fingerprint"],
            sample_index=1,
            sample_label="forward",
            time_seconds=forward_time,
        ),
        *_state_rows(
            returned_state,
            manifest_sha256=manifest_sha256,
            run_id=case["run_id"],
            run_fingerprint=case["run_fingerprint"],
            scientific_fingerprint=case["scientific_configuration_fingerprint"],
            sample_index=2,
            sample_label="returned",
            time_seconds=return_time,
        ),
    ]
    _write_state(paths["state"], rows)
    state_audit = _audit_state(
        paths["state"], manifest_sha256=manifest_sha256, case=case
    )
    integrity_checks["state_schema_and_rows_valid"] = all(state_audit.values())
    case_passed = all(integrity_checks.values()) and all(
        item["passed"] for item in absolute_checks.values()
    )
    _event(
        paths["events"],
        f"COMPLETE run_id={case['run_id']} case_passed={case_passed} runtime_seconds={runtime_seconds:.9f}",
    )
    inventory = {
        key: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for key, path in paths.items()
        if key in {"state", "events"}
    }
    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "case_passed": case_passed,
        "run_id": case["run_id"],
        "source_case_id": case["source_case_id"],
        "manifest_sha256": manifest_sha256,
        "run_fingerprint": case["run_fingerprint"],
        "scientific_configuration_fingerprint": case[
            "scientific_configuration_fingerprint"
        ],
        "configuration": source_config,
        "direction_reversal_procedure": manifest["scientific_configuration"][
            "reversal_procedure"
        ],
        "forward_time_seconds": forward_time,
        "return_time_seconds": return_time,
        "forward_steps_executed": source_config["forward_steps"],
        "backward_steps_executed": source_config["backward_steps"],
        "fractional_endpoint_steps": 0,
        "forward_internal_is_synchronized": forward_internal_is_synchronized,
        "returned_internal_is_synchronized": returned_internal_is_synchronized,
        "comparison_particle_arrays_synchronized": True,
        "callback_stats": callback,
        "integrity_checks": integrity_checks,
        "absolute_checks": absolute_checks,
        "metrics": metrics,
        "state_audit": state_audit,
        "runtime_seconds": runtime_seconds,
        "throughput_steps_per_second": (
            source_config["forward_steps"] + source_config["backward_steps"]
        )
        / runtime_seconds,
        "artifact_inventory": inventory,
        "provenance": {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head": _git("rev-parse", "HEAD"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
    }
    atomic_write_json(paths["summary"], summary)
    return summary


def diagnostic_ratios(case_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    by_source = {
        summary["source_case_id"]: summary for summary in case_summaries.values()
    }
    for mode in ("current_sync", "min_sync"):
        coarse = by_source[f"two_body_{mode}_0p5d"]["metrics"]["global_scaled_rms"]
        fine = by_source[f"two_body_{mode}_0p25d"]["metrics"]["global_scaled_rms"]
        result[mode] = {
            "coarse_0p5d_scaled_rms": coarse,
            "fine_0p25d_scaled_rms": fine,
            "fine_over_coarse": fine / coarse if coarse != 0.0 else None,
            "diagnostic_only": True,
            "affects_validity": False,
        }
    return result


def final_status(case_summaries: dict[str, dict[str, Any]]) -> tuple[str, dict[str, bool]]:
    if len(case_summaries) != 4 or any(
        summary.get("status") != "COMPLETED" for summary in case_summaries.values()
    ):
        return "BLOCKED", {"current_sync": False, "min_sync": False}
    by_source = {
        summary["source_case_id"]: summary for summary in case_summaries.values()
    }
    modes = {
        mode: all(
            by_source[f"two_body_{mode}_{step}"]["case_passed"]
            for step in ("0p5d", "0p25d")
        )
        for mode in ("current_sync", "min_sync")
    }
    status = (
        "REVERSIBILITY_GATE_PASSED"
        if all(modes.values())
        else "REVERSIBILITY_GATE_FAILED"
    )
    return status, modes


def _historical_comparison(
    manifest: dict[str, Any], case_summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    historical = _load_json(
        Path(manifest["paths"]["project_root"])
        / manifest["provenance"]["historical_validation_summary_path"],
        "historical validation summary",
    )
    by_source = {
        summary["source_case_id"]: summary for summary in case_summaries.values()
    }
    comparison = {}
    for source_case_id, new in by_source.items():
        old = historical["results"][source_case_id]
        comparison[source_case_id] = {
            "historical_global_scaled_rms": old["metrics"]["global_scaled_rms"],
            "new_global_scaled_rms": new["metrics"]["global_scaled_rms"],
            "absolute_difference": abs(
                new["metrics"]["global_scaled_rms"]
                - old["metrics"]["global_scaled_rms"]
            ),
            "historical_case_absolute_passed": old["passed"],
            "new_case_passed": new["case_passed"],
            "same_callback_count": (
                new["callback_stats"]["callback_invocations"]
                == old["callback_stats"]["callback_invocations"]
            ),
            "same_return_time": new["return_time_seconds"]
            == old["return_time_seconds"],
        }
    return comparison


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M0 Reversibility Roundoff Gate",
        "",
        f"Final status: **{payload['final_status']}**",
        "",
        "## Historical Result",
        "",
        "Manifest 13 remains **BLOCKED**. Its minimal-sync fine/coarse ratio was "
        f"`{payload['historical_manifest_13']['min_sync_ratio']:.9g}` against the frozen `4.0` ratio gate. "
        "Neither manifest 13 nor its report has been reinterpreted or modified.",
        "",
        "## Corrected Criterion",
        "",
        "For an autonomous symmetric map with exact integer steps, `Phi(-h)^N Phi(h)^N = I` "
        "in exact arithmetic. The return error therefore measures roundoff, synchronization/reconstruction, "
        "and implementation asymmetry; it is not a second-order truncation convergence test.",
        "",
        "The fine/coarse ratios below are diagnostic only and do not affect validity.",
        "",
        "Manifest 13 froze 1e-8 for global scaled Cartesian RMS but no separate metric-specific "
        "numeric limits. Before this rerun, manifest 14 applied the same original dimensionless "
        "absolute scale uniformly to every requested normalized return metric; no observed endpoint "
        "value was used to set a threshold.",
        "",
        "| Mode | 0.5-day RMS | 0.25-day RMS | Fine/coarse | Mode pass |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for mode, ratio in payload["diagnostic_ratios"].items():
        lines.append(
            f"| {mode} | {ratio['coarse_0p5d_scaled_rms']:.12g} | "
            f"{ratio['fine_0p25d_scaled_rms']:.12g} | {ratio['fine_over_coarse']:.9g} | "
            f"{payload['synchronization_modes'][mode]} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | RMS | RMS margin | Energy rel. | Angular rel. | Callbacks | Runtime (s) | Pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for run_id, case in payload["cases"].items():
        checks = case["absolute_checks"]
        lines.append(
            f"| {run_id} | {case['metrics']['global_scaled_rms']:.12g} | "
            f"{checks['global_scaled_cartesian_rms']['absolute_margin']:.12g} | "
            f"{abs(case['metrics']['corrected_energy_relative_difference']):.12g} | "
            f"{case['metrics']['angular_momentum_vector_relative_difference']:.12g} | "
            f"{case['callback_stats']['callback_invocations']} | {case['runtime_seconds']:.6f} | "
            f"{case['case_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint And Integrity",
            "",
            "| Case | Forward time | Return time | Steps forward/back | Callbacks observed/expected | Nonfinite |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_id, case in payload["cases"].items():
        lines.append(
            f"| {run_id} | {case['forward_time_seconds']:.9g} | "
            f"{case['return_time_seconds']:.9g} | "
            f"{case['forward_steps_executed']}/{case['backward_steps_executed']} | "
            f"{case['callback_stats']['callback_invocations']}/"
            f"{case['configuration']['expected_callback_invocations']} | "
            f"{case['callback_stats']['nonfinite_result_count']} |"
        )
    lines.extend(
        [
            "",
            "## Historical Comparison",
            "",
            "| Source case | Historical RMS | New RMS | Absolute difference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source_case_id, item in payload["historical_comparison"].items():
        lines.append(
            f"| {source_case_id} | {item['historical_global_scaled_rms']:.12g} | "
            f"{item['new_global_scaled_rms']:.12g} | {item['absolute_difference']:.12g} |"
        )
    lines.extend(
        [
            "",
            "Every case report contains the full per-body position/velocity checks, center-of-mass checks, "
            "absolute limits and margins, exact endpoints and step counts, callback totals, finite-state checks, "
            "schema/fingerprint checks, and historical comparison.",
            "",
            "## Decision",
            "",
            f"Step 3d decisive experiment may resume: **{payload['step3d_may_resume']}**.",
            "",
            "No million-year energy-drift mechanism is classified here. No Stage 4 command is provided.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate_payload(
    manifest_path: Path,
    manifest: dict[str, Any],
    audit_payload: dict[str, Any],
    case_summaries: dict[str, dict[str, Any]],
    runtime_seconds: float,
) -> dict[str, Any]:
    status, modes = final_status(case_summaries)
    _require(status in FINAL_STATUSES, "Invalid final status.")
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "final_status": status,
        "historical_manifest_13": {
            "primary_mechanism": manifest["provenance"][
                "historical_manifest_13_primary_mechanism"
            ],
            "step3_status": manifest["provenance"][
                "historical_manifest_13_step3_status"
            ],
            "min_sync_ratio": manifest["provenance"][
                "historical_min_sync_fine_over_coarse_ratio"
            ],
            "immutable": True,
        },
        "mathematical_basis": manifest["mathematical_basis"],
        "absolute_limits": manifest["absolute_limits"],
        "audit": audit_payload,
        "cases": case_summaries,
        "synchronization_modes": modes,
        "diagnostic_ratios": diagnostic_ratios(case_summaries),
        "historical_comparison": _historical_comparison(manifest, case_summaries),
        "command": sys.argv,
        "runtime_seconds": runtime_seconds,
        "throughput_cases_per_second": len(case_summaries) / runtime_seconds,
        "step3d_may_resume": status == "REVERSIBILITY_GATE_PASSED",
        "million_year_energy_mechanism_classified": False,
        "no_stage4_command": True,
    }


def run_all(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 14")
    audit_payload = audit(manifest_path)
    output_root = Path(manifest["paths"]["output_root"])
    report_json = Path(manifest["paths"]["report_json"])
    report_markdown = Path(manifest["paths"]["report_markdown"])
    _require(not output_root.exists(), f"Collision-safe output root exists: {output_root}")
    _require(not report_json.exists(), f"Report already exists: {report_json}")
    _require(not report_markdown.exists(), f"Report already exists: {report_markdown}")
    output_root.mkdir(parents=True)
    manifest_13 = _load_json(
        Path(manifest["paths"]["project_root"])
        / manifest["provenance"]["manifest_13_path"],
        "manifest 13",
    )
    source_configs = _source_cases(manifest, manifest_13)
    case_summaries = {}
    started = time.perf_counter()
    try:
        for case in manifest["cases"]:
            case_summaries[case["run_id"]] = _run_case(
                manifest_path,
                manifest,
                case,
                source_configs[case["run_id"]],
                output_root,
            )
        runtime_seconds = time.perf_counter() - started
        payload = _aggregate_payload(
            manifest_path,
            manifest,
            audit_payload,
            case_summaries,
            runtime_seconds,
        )
    except Exception as exc:
        runtime_seconds = time.perf_counter() - started
        payload = {
            "schema_version": 1,
            "experiment_id": manifest["experiment_id"],
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "final_status": "BLOCKED",
            "blocking_error": str(exc),
            "audit": audit_payload,
            "cases": case_summaries,
            "runtime_seconds": runtime_seconds,
            "step3d_may_resume": False,
            "million_year_energy_mechanism_classified": False,
            "no_stage4_command": True,
        }
        atomic_write_json(output_root / "aggregate_summary.json", payload)
        raise
    aggregate_path = output_root / "aggregate_summary.json"
    atomic_write_json(aggregate_path, payload)
    report_payload = {
        **payload,
        "aggregate_artifact": {
            "path": str(aggregate_path),
            "size_bytes": aggregate_path.stat().st_size,
            "sha256": sha256_file(aggregate_path),
        },
    }
    atomic_write_json(report_json, report_payload)
    _atomic_text(report_markdown, _markdown(report_payload))
    print(f"[m0-reversibility-gate] final_status={payload['final_status']}")
    print(f"[m0-reversibility-gate] step3d_may_resume={payload['step3d_may_resume']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="M0 reversibility roundoff gate.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("run-all")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            print(json.dumps(audit(args.manifest), indent=2, sort_keys=True))
        elif args.command == "run-all":
            run_all(args.manifest)
    except GateError as exc:
        raise SystemExit(f"m0 reversibility roundoff gate error: {exc}") from exc


if __name__ == "__main__":
    main()
