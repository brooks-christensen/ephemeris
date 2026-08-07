from __future__ import annotations

import argparse
import csv
import datetime as dt
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import CBackend, load_c_backend
from .long_term_stability_cli import (
    build_rebound_simulation,
    configure_rebound_simulationarchive,
    optional_import_module,
    rebound_state_from_sim,
    stability_body_list,
)
from .m0_energy_precision_diagnosis import (
    compensated_energy,
    decimal_energy,
    float64_energy,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import (
    AU_M,
    DAY_S,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
)
from .rebound_gr_tangent_backend_cli import (
    atomic_write_json,
    canonical_hash,
    initial_condition_hash,
    sha256_file,
)
from .stability_diagnostics import (
    center_of_mass_position_velocity,
    total_angular_momentum_vector,
)


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/13_m0_integrator_roundoff_diagnosis_v1.json"
)
PHYSICAL_STATE_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
PHYSICAL_STATE_FIELDS = [
    "schema_version",
    "configuration_fingerprint",
    "sample_index",
    "time_seconds",
    "time_years",
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
PROGRESS_FIELDS = [
    "schema_version",
    "configuration_fingerprint",
    "sample_index",
    "target_time_years",
    "time_seconds",
    "time_years",
    "integrator",
    "synchronization_mode",
    "newtonian_energy_j",
    "gr_potential_energy_j",
    "corrected_energy_j",
    "corrected_energy_rel_change",
    "angular_momentum_x_kg_m2_s",
    "angular_momentum_y_kg_m2_s",
    "angular_momentum_z_kg_m2_s",
    "angular_momentum_norm_kg_m2_s",
    "angular_momentum_rel_change",
    "center_of_mass_x_m",
    "center_of_mass_y_m",
    "center_of_mass_z_m",
    "center_of_mass_vx_m_per_s",
    "center_of_mass_vy_m_per_s",
    "center_of_mass_vz_m_per_s",
    "callback_invocations",
    "nonfinite_result_count",
]
ENERGY_TIMESERIES_FIELDS = [
    "schema_version",
    "manifest_sha256",
    "lane_id",
    "sample_index",
    "time_years",
    "recorded_corrected_rel_change",
    "float64_corrected_rel_change",
    "compensated_corrected_rel_change",
    "decimal_corrected_rel_change",
    "float64_minus_recorded",
    "compensated_minus_decimal",
]
PRIMARY_MECHANISMS = {
    "BOUNDED_ENERGY_OSCILLATION",
    "RANDOM_WALK_ROUNDOFF",
    "SYSTEMATIC_WHFAST_STEP_BIAS",
    "SYNCHRONIZATION_RECALCULATION_BIAS",
    "VARIATION_MEGNO_COUPLING",
    "CORRECTED_INVARIANT_OR_FORCE_PROBLEM",
    "MIXED_OR_INCONCLUSIVE",
    "BLOCKED",
}
STEP3_STATUSES = {
    "STEP3_NUMERICAL_FLOOR_CHARACTERIZED",
    "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED",
    "STEP3_FORCE_INVARIANT_PROBLEM",
    "STEP3_DIAGNOSIS_INCONCLUSIVE",
    "BLOCKED",
}
INNER_BODIES = (
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
)


class DiagnosisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except Exception as exc:
        raise DiagnosisError(f"Unreadable {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"Invalid {label}: expected JSON object.")
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise DiagnosisError(f"Git command failed: git {' '.join(args)}") from exc


def _canonical_float_token(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    return math.isfinite(value) and str(value) == token


def _artifact_entries(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    inventory = payload.get("artifact_inventory")
    if isinstance(inventory, list):
        entries = inventory
    elif isinstance(inventory, dict):
        entries = [
            entry
            for group in inventory.values()
            if isinstance(group, list)
            for entry in group
        ]
    else:
        raise DiagnosisError(f"Missing artifact inventory in {label}.")
    _require(all(isinstance(entry, dict) for entry in entries), f"Bad {label} inventory.")
    return entries


def _expanded_lane(manifest: dict[str, Any], lane_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lane = next((item for item in manifest["lanes"] if item["id"] == lane_id), None)
    _require(lane is not None, f"Unknown lane: {lane_id}")
    config = {**manifest["common_configuration"], **lane["configuration"]}
    _require(canonical_hash(config) == lane["configuration_fingerprint"], f"Fingerprint mismatch: {lane_id}")
    return lane, config


def _runtime_identity(rebound: Any) -> dict[str, Any]:
    library = Path(rebound.clibrebound._name)
    header = Path(rebound.__file__).parent / "rebound.h"
    return {
        "rebound_version": rebound.__version__,
        "rebound_build": getattr(rebound, "__build__", None),
        "rebound_githash": getattr(rebound, "__githash__", None),
        "shared_library_path": str(library),
        "shared_library_sha256": sha256_file(library),
        "header_path": str(header),
        "header_sha256": sha256_file(header),
    }



def _audit_continuation_provenance(
    manifest: dict[str, Any], root: Path
) -> dict[str, Any] | None:
    provenance = manifest.get("continuation_provenance")
    if provenance is None:
        return None
    locked_sections = provenance["locked_sections_imported_unchanged"]
    source_manifest_path = Path(provenance["source_manifest_13_path"])
    source_manifest = _load_json(source_manifest_path, "source manifest 13")
    _require(
        sha256_file(source_manifest_path) == provenance["source_manifest_13_sha256"],
        "Source manifest 13 hash changed.",
    )
    for section in locked_sections:
        _require(
            manifest[section] == source_manifest[section],
            f"Manifest 13 locked section changed: {section}",
        )
    fixed = (
        ("source_manifest_13_summary_path", "source_manifest_13_summary_sha256"),
        ("source_manifest_13_report_path", "source_manifest_13_report_sha256"),
        ("source_manifest_14_path", "source_manifest_14_sha256"),
        ("source_manifest_14_summary_path", "source_manifest_14_summary_sha256"),
        ("source_manifest_14_report_path", "source_manifest_14_report_sha256"),
    )
    for path_key, hash_key in fixed:
        path = Path(provenance[path_key])
        _require(
            sha256_file(path) == provenance[hash_key],
            f"Continuation source hash changed: {path}",
        )
    manifest_13_summary = _load_json(
        Path(provenance["source_manifest_13_summary_path"]), "manifest 13 summary"
    )
    _require(
        manifest_13_summary.get("primary_mechanism")
        == provenance["historical_manifest_13_primary_mechanism"]
        == "BLOCKED",
        "Historical manifest 13 mechanism changed.",
    )
    _require(
        manifest_13_summary.get("step3_diagnosis_status")
        == provenance["historical_manifest_13_step3_status"]
        == "BLOCKED",
        "Historical manifest 13 status changed.",
    )
    gate_summary = _load_json(
        Path(provenance["source_manifest_14_summary_path"]), "manifest 14 summary"
    )
    gate = manifest["continuation_method_gate"]
    _require(
        gate_summary.get("final_status")
        == gate["required_status"]
        == "REVERSIBILITY_GATE_PASSED",
        "Manifest 14 absolute reversibility gate did not pass.",
    )
    _require(
        gate_summary.get("step3d_may_resume") is True,
        "Manifest 14 does not permit continuation.",
    )
    _require(
        all(
            item.get("diagnostic_only") is True
            and item.get("affects_validity") is False
            for item in gate_summary["diagnostic_ratios"].values()
        ),
        "Manifest 14 ratio policy changed.",
    )
    _require(
        gate["fine_coarse_return_error_ratios_are_diagnostic_only"] is True
        and gate["fine_coarse_return_error_ratios_affect_validity"] is False
        and gate["does_not_change_scientific_or_causal_thresholds"] is True,
        "Continuation method-gate policy changed.",
    )
    return {
        "source_manifest_13_sha256": sha256_file(source_manifest_path),
        "source_manifest_14_sha256": sha256_file(
            Path(provenance["source_manifest_14_path"])
        ),
        "source_manifest_14_summary_sha256": sha256_file(
            Path(provenance["source_manifest_14_summary_path"])
        ),
        "source_manifest_14_status": gate_summary["final_status"],
        "locked_sections_verified": locked_sections,
        "fine_coarse_ratios_diagnostic_only": True,
    }


def _audit_state_file(
    path: Path,
    *,
    fields: Sequence[str],
    body_names: Sequence[str],
    fingerprint: str,
    expected_samples: int,
    cadence_years: float,
) -> dict[str, Any]:
    numeric = (
        "mass_kg",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_per_s",
        "vy_m_per_s",
        "vz_m_per_s",
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == list(fields), f"State schema mismatch: {path}")
        count = 0
        for row in reader:
            sample, body = divmod(count, len(body_names))
            _require(int(row["sample_index"]) == sample, f"State sample order mismatch: {path}")
            _require(int(row["body_index"]) == body, f"State body order mismatch: {path}")
            _require(row["body_name"] == body_names[body], f"State body name mismatch: {path}")
            _require(row["configuration_fingerprint"] == fingerprint, f"State fingerprint mismatch: {path}")
            _require(float(row["time_years"]) == sample * cadence_years, f"State time mismatch: {path}")
            for field in numeric:
                _require(_canonical_float_token(row[field]), f"Noncanonical/nonfinite {field}: {path}")
            count += 1
    expected_rows = expected_samples * len(body_names)
    _require(count == expected_rows, f"State row count {count} != {expected_rows}: {path}")
    return {"samples": expected_samples, "rows": count}


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "manifest 13")
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Run from the project root.")
    _require(manifest.get("frozen_before_new_integrations") is True, "Manifest 13 is not frozen.")
    head = _git("rev-parse", "HEAD")
    start = manifest["provenance"]["starting_commit"]
    _git("merge-base", "--is-ancestor", start, head)
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C tag is not annotated.")
    _require(
        _git("rev-parse", tag + "^{commit}")
        == manifest["provenance"]["validated_c_baseline_commit"],
        "Compiled-C tag target mismatch.",
    )
    fixed = {
        "ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json": manifest["provenance"]["manifest_10_sha256"],
        "ephemeris_experiment_runner/manifests/11_m0_timestep_convergence_0p25_v1.json": manifest["provenance"]["manifest_11_sha256"],
        "ephemeris_experiment_runner/manifests/12_m0_energy_precision_diagnosis_v1.json": manifest["provenance"]["manifest_12_sha256"],
        str(Path(manifest["paths"]["step3_summary"]).relative_to(root)): manifest["provenance"]["step3_summary_sha256"],
        str(Path(manifest["paths"]["step3b_summary"]).relative_to(root)): manifest["provenance"]["step3b_summary_sha256"],
        str(Path(manifest["paths"]["step3c_summary"]).relative_to(root)): manifest["provenance"]["step3c_summary_sha256"],
    }
    for relative, expected in fixed.items():
        _require(sha256_file(root / relative) == expected, f"Historical hash mismatch: {relative}")
    protected = []
    for relative, expected in manifest["protected_files"].items():
        actual = sha256_file(root / relative)
        _require(actual == expected, f"Protected file mismatch: {relative}")
        protected.append({"path": relative, "sha256": actual})
    inventory_counts: dict[str, int] = {}
    unique_paths: set[str] = set()
    for key in ("step3_summary", "step3b_summary"):
        payload = _load_json(Path(manifest["paths"][key]), key)
        entries = _artifact_entries(payload, key)
        for item in entries:
            path = Path(item["path"])
            _require(path.stat().st_size == item["size_bytes"], f"Artifact size mismatch: {path}")
            _require(sha256_file(path) == item["sha256"], f"Artifact hash mismatch: {path}")
            unique_paths.add(str(path))
        inventory_counts[key] = len(entries)
    step3c = _load_json(Path(manifest["paths"]["step3c_summary"]), "Step 3c summary")
    _require(step3c.get("diagnosis") == "ENERGY_DRIFT_CONFIRMED", "Step 3c status changed.")
    for item in step3c["timeseries_inventory"]:
        path = Path(item["path"])
        _require(path.stat().st_size == item["size_bytes"], f"Step 3c size mismatch: {path}")
        _require(sha256_file(path) == item["sha256"], f"Step 3c hash mismatch: {path}")
    body_names = manifest["common_configuration"]["body_names"]
    old_lanes = {
        "m0_conv_0p5d_1myr_s12345": (
            "68fb88378722f7408b60bb1251c6755611da521fa7f26fcafc10736c7b2da59e",
            "0p5d",
        ),
        "m0_conv_0p25d_1myr_s12345": (
            "3e79729659677339dd5a4cd64c9f8d217af3d97a863786b9388b6a5b8b42533c",
            "0p25d",
        ),
    }
    existing_integrity = {}
    for lane_id, (fingerprint, _) in old_lanes.items():
        directory = Path(manifest["paths"]["existing_convergence_root"]) / lane_id
        state_path = directory / f"gr_tangent_state_{lane_id}.csv"
        progress_path = directory / f"gr_tangent_progress_{lane_id}.csv"
        existing_integrity[lane_id] = _audit_state_file(
            state_path,
            fields=_existing_state_fields(),
            body_names=body_names,
            fingerprint=fingerprint,
            expected_samples=10001,
            cadence_years=100.0,
        )
        with progress_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        _require(len(rows) == 10001, f"Existing progress count mismatch: {lane_id}")
        _require(all(float(row["time_years"]) == i * 100.0 for i, row in enumerate(rows)), f"Existing progress times mismatch: {lane_id}")
        _require(all(int(row["nonfinite_result_count"]) == 0 for row in rows), f"Existing nonfinite result: {lane_id}")
    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    runtime = _runtime_identity(rebound)
    expected_runtime = manifest["installed_runtime"]
    for key in ("rebound_version", "rebound_build", "rebound_githash"):
        _require(runtime[key] == expected_runtime[key], f"REBOUND identity mismatch: {key}")
    _require(runtime["shared_library_sha256"] == expected_runtime["rebound_shared_library_sha256"], "REBOUND library hash mismatch.")
    _require(runtime["header_sha256"] == expected_runtime["rebound_header_sha256"], "REBOUND header hash mismatch.")
    c_artifact = root / "mini_ephemeris/build/gr_tangent_c/libmini_ephemeris_gr_tangent.so"
    _require(sha256_file(c_artifact) == expected_runtime["compiled_gr_artifact_sha256"], "Compiled GR artifact mismatch.")
    continuation = _audit_continuation_provenance(manifest, root)
    return {
        "status": "PASS",
        "git_head": head,
        "git_dirty_after_preregistration": bool(_git("status", "--porcelain")),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tag_object_type": "tag",
        "protected_files": protected,
        "historical_inventory_entries": inventory_counts,
        "unique_historical_artifacts": len(unique_paths),
        "step3c_timeseries_verified": len(step3c["timeseries_inventory"]),
        "existing_lane_integrity": existing_integrity,
        "runtime": runtime,
        "compiled_gr_artifact_sha256": sha256_file(c_artifact),
        "continuation_provenance": continuation,
    }


def _existing_state_fields() -> list[str]:
    from .m0_telemetry import STATE_SAMPLE_FIELDS

    return STATE_SAMPLE_FIELDS


def _lane_paths(manifest: dict[str, Any], lane_id: str) -> dict[str, Path]:
    directory = Path(manifest["paths"]["output_root"]) / lane_id
    return {
        "directory": directory,
        "progress": directory / "progress.csv",
        "state": directory / "physical_state.csv",
        "archive": directory / "simulationarchive.bin",
        "status": directory / "status.json",
        "summary": directory / "summary.json",
        "events": directory / "events.log",
    }


def _event(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _state_rows(
    state: NBodyState,
    body_names: Sequence[str],
    *,
    sample_index: int,
    time_seconds: float,
    fingerprint: str,
) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": PHYSICAL_STATE_SCHEMA_VERSION,
            "configuration_fingerprint": fingerprint,
            "sample_index": sample_index,
            "time_seconds": time_seconds,
            "time_years": time_seconds / JULIAN_YEAR_S,
            "body_index": index,
            "body_name": body_names[index],
            "mass_kg": float(state.masses[index]),
            "x_m": float(state.positions[index, 0]),
            "y_m": float(state.positions[index, 1]),
            "z_m": float(state.positions[index, 2]),
            "vx_m_per_s": float(state.velocities[index, 0]),
            "vy_m_per_s": float(state.velocities[index, 1]),
            "vz_m_per_s": float(state.velocities[index, 2]),
        }
        for index in range(len(body_names))
    ]


def _state_energy(state: NBodyState) -> dict[str, float]:
    return float64_energy(
        state.masses,
        state.positions,
        state.velocities,
        gravitational_constant=G_SI,
        speed_of_light=299_792_458.0,
        coefficient_scale=1.0,
    )


def _settings(sim: Any) -> dict[str, Any]:
    payload = {
        "integrator": sim.integrator,
        "dt_seconds": float(sim.dt),
        "gravity": sim.gravity,
        "boundary": sim.boundary,
        "N": int(sim.N),
        "N_real": int(sim.N_real),
        "N_var": int(sim.N_var),
        "N_active": int(sim.N_active),
        "force_is_velocity_dependent": int(sim.force_is_velocity_dependent),
    }
    if sim.integrator == "whfast":
        payload["whfast"] = {
            key: getattr(sim.ri_whfast, key)
            for key in (
                "coordinates",
                "kernel",
                "corrector",
                "corrector2",
                "safe_mode",
                "keep_unsynchronized",
                "is_synchronized",
                "recalculate_coordinates_this_timestep",
            )
        }
    if sim.integrator == "ias15":
        payload["ias15"] = {
            "epsilon": float(sim.ri_ias15.epsilon),
            "min_dt": float(sim.ri_ias15.min_dt),
            "adaptive_mode": sim.ri_ias15.adaptive_mode,
        }
    return payload


def _initial_state(manifest: dict[str, Any]) -> tuple[list[str], NBodyState]:
    config = manifest["common_configuration"]
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    _require(
        list(bodies) == config["body_names"],
        "Body order differs from the frozen manifest configuration.",
    )
    state = initial_state_solar_system_barycentric(
        dt.datetime.fromisoformat(config["start_date"]),
        bodies=bodies,
        config=EphemerisConfig(kernel_path=manifest["paths"]["kernel"]),
    )
    _require(initial_condition_hash(state, bodies) == config["initial_conditions_sha256"], "Initial condition hash mismatch.")
    return bodies, state


def _build_simulation(
    manifest: dict[str, Any], config: dict[str, Any]
) -> tuple[Any, Any, CBackend, list[str], NBodyState]:
    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    bodies, state0 = _initial_state(manifest)
    if config["integrator"] == "whfast":
        sim = build_rebound_simulation(
            rebound,
            state0,
            integrator="whfast",
            step_s=float(config["step_days"]) * DAY_S,
            ias15_epsilon=1e-10,
        )
        sim.ri_whfast.coordinates = config["coordinates"]
        sim.ri_whfast.kernel = config["kernel"]
        sim.ri_whfast.corrector = config["corrector"]
        sim.ri_whfast.corrector2 = config["corrector2"]
        sim.ri_whfast.safe_mode = config["safe_mode"]
        sim.ri_whfast.keep_unsynchronized = config["keep_unsynchronized"]
        sim.ri_whfast.recalculate_coordinates_this_timestep = config[
            "recalculate_coordinates_this_timestep"
        ]
    elif config["integrator"] == "ias15":
        sim = build_rebound_simulation(
            rebound,
            state0,
            integrator="ias15",
            step_s=DAY_S,
            ias15_epsilon=float(config["epsilon"]),
        )
        sim.dt = float(config["initial_dt_days"]) * DAY_S
        sim.ri_ias15.epsilon = float(config["epsilon"])
        sim.ri_ias15.min_dt = float(config["min_dt_seconds"])
        sim.ri_ias15.adaptive_mode = config["adaptive_mode"]
    else:
        raise DiagnosisError(f"Unsupported integrator: {config['integrator']}")
    backend = load_c_backend()
    backend.attach(sim, coefficient_scale=1.0, include_central_response=True)
    _require(int(sim.N) == 10 and int(sim.N_var) == 0, "Physical-only layout is invalid.")
    _require(int(sim.force_is_velocity_dependent) == 0, "GR callback became velocity dependent.")
    return rebound, sim, backend, bodies, state0


def _write_csv_atomic(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _progress_row(
    state: NBodyState,
    *,
    sim: Any,
    backend: CBackend,
    fingerprint: str,
    sample_index: int,
    target_years: float,
    energy_reference: float,
    angular_reference: float,
    synchronization_mode: str,
) -> dict[str, Any]:
    energy = _state_energy(state)
    angular = total_angular_momentum_vector(state)
    angular_norm = float(np.linalg.norm(angular))
    center_position, center_velocity = center_of_mass_position_velocity(state)
    callback = backend.stats(sim)
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "sample_index": sample_index,
        "target_time_years": target_years,
        "time_seconds": float(sim.t),
        "time_years": float(sim.t) / JULIAN_YEAR_S,
        "integrator": sim.integrator,
        "synchronization_mode": synchronization_mode,
        "newtonian_energy_j": energy["newtonian"],
        "gr_potential_energy_j": energy["gr_potential"],
        "corrected_energy_j": energy["corrected"],
        "corrected_energy_rel_change": (energy["corrected"] - energy_reference)
        / (abs(energy_reference) if energy_reference != 0.0 else 1.0),
        "angular_momentum_x_kg_m2_s": float(angular[0]),
        "angular_momentum_y_kg_m2_s": float(angular[1]),
        "angular_momentum_z_kg_m2_s": float(angular[2]),
        "angular_momentum_norm_kg_m2_s": angular_norm,
        "angular_momentum_rel_change": (angular_norm - angular_reference)
        / (abs(angular_reference) if angular_reference != 0.0 else 1.0),
        "center_of_mass_x_m": float(center_position[0]),
        "center_of_mass_y_m": float(center_position[1]),
        "center_of_mass_z_m": float(center_position[2]),
        "center_of_mass_vx_m_per_s": float(center_velocity[0]),
        "center_of_mass_vy_m_per_s": float(center_velocity[1]),
        "center_of_mass_vz_m_per_s": float(center_velocity[2]),
        "callback_invocations": int(callback["callback_invocations"]),
        "nonfinite_result_count": int(callback["nonfinite_result_count"]),
    }


def run_lane(manifest_path: Path, lane_id: str) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit_payload = audit(manifest_path)
    lane, config = _expanded_lane(manifest, lane_id)
    _require(config["kind"] in {"whfast_physical_control", "ias15_physical_reference"}, "Not a scientific lane.")
    if config["kind"] == "ias15_physical_reference":
        benchmark_path = Path(manifest["paths"]["output_root"]) / "ias15_benchmark_summary.json"
        benchmark = _load_json(benchmark_path, "IAS15 benchmark")
        _require(benchmark.get("passed") is True, "IAS15 benchmark gate did not pass.")
        _require(benchmark.get("manifest_sha256") == sha256_file(manifest_path), "IAS15 benchmark manifest mismatch.")
    paths = _lane_paths(manifest, lane_id)
    _require(not paths["directory"].exists(), f"Collision-safe lane already exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True)
    _event(paths["events"], f"START lane={lane_id} command={' '.join(sys.argv)}")
    progress_temp = paths["progress"].with_name(paths["progress"].name + ".tmp")
    state_temp = paths["state"].with_name(paths["state"].name + ".tmp")
    try:
        rebound, sim, backend, bodies, state0 = _build_simulation(manifest, config)
        configure_rebound_simulationarchive(
            sim,
            paths["archive"],
            interval_s=float(config["archive_interval_years"]) * JULIAN_YEAR_S,
            delete_existing=True,
        )
        reference_energy = _state_energy(state0)["corrected"]
        reference_angular = float(np.linalg.norm(total_angular_momentum_vector(state0)))
        targets = np.arange(config["expected_samples"], dtype=np.float64) * float(
            config["record_every_years"]
        )
        start = time.perf_counter()
        with (
            progress_temp.open("w", newline="") as progress_handle,
            state_temp.open("w", newline="") as state_handle,
        ):
            progress_writer = csv.DictWriter(progress_handle, fieldnames=PROGRESS_FIELDS)
            state_writer = csv.DictWriter(state_handle, fieldnames=PHYSICAL_STATE_FIELDS)
            progress_writer.writeheader()
            state_writer.writeheader()
            for sample_index, target_years in enumerate(targets):
                sim.integrate(float(target_years) * JULIAN_YEAR_S, exact_finish_time=1)
                _require(float(sim.t) == float(target_years) * JULIAN_YEAR_S, "Lane missed exact target time.")
                state = rebound_state_from_sim(sim, state0.masses)
                _require(np.all(np.isfinite(state.positions)) and np.all(np.isfinite(state.velocities)), "Nonfinite lane state.")
                progress_writer.writerow(
                    _progress_row(
                        state,
                        sim=sim,
                        backend=backend,
                        fingerprint=lane["configuration_fingerprint"],
                        sample_index=sample_index,
                        target_years=float(target_years),
                        energy_reference=reference_energy,
                        angular_reference=reference_angular,
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
                progress_handle.flush()
                state_handle.flush()
                os.fsync(progress_handle.fileno())
                os.fsync(state_handle.fileno())
                callback = backend.stats(sim)
                atomic_write_json(
                    paths["status"],
                    {
                        "state": "RUNNING",
                        "lane_id": lane_id,
                        "manifest_sha256": sha256_file(manifest_path),
                        "configuration_fingerprint": lane["configuration_fingerprint"],
                        "sample_index": sample_index,
                        "time_years": float(target_years),
                        "callback_invocations": int(callback["callback_invocations"]),
                        "nonfinite_result_count": int(callback["nonfinite_result_count"]),
                    },
                )
                print(f"[m0-roundoff] {lane_id} {sample_index + 1}/{len(targets)}", flush=True)
        os.replace(progress_temp, paths["progress"])
        os.replace(state_temp, paths["state"])
        elapsed = time.perf_counter() - start
        callback = backend.stats(sim)
        expected_callbacks = config["expected_callback_invocations"]
        if isinstance(expected_callbacks, int):
            _require(int(callback["callback_invocations"]) == expected_callbacks, "WHFast callback total mismatch.")
        _require(int(callback["nonfinite_result_count"]) == 0, "Nonfinite callback result.")
        archive = rebound.Simulationarchive(str(paths["archive"]))
        expected_snapshots = int(config["duration_years"] / config["archive_interval_years"]) + 1
        _require(len(archive) == expected_snapshots, "SimulationArchive snapshot count mismatch.")
        state_audit = _audit_state_file(
            paths["state"],
            fields=PHYSICAL_STATE_FIELDS,
            body_names=bodies,
            fingerprint=lane["configuration_fingerprint"],
            expected_samples=config["expected_samples"],
            cadence_years=config["record_every_years"],
        )
        summary = {
            "schema_version": 1,
            "status": "COMPLETED",
            "lane_id": lane_id,
            "manifest_path": str(manifest_path),
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
            "callback_stats": callback,
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
        atomic_write_json(
            paths["status"],
            {
                "state": "COMPLETED",
                "lane_id": lane_id,
                "manifest_sha256": sha256_file(manifest_path),
                "configuration_fingerprint": lane["configuration_fingerprint"],
                "samples": state_audit["samples"],
                "state_rows": state_audit["rows"],
                "callback_invocations": int(callback["callback_invocations"]),
                "nonfinite_result_count": int(callback["nonfinite_result_count"]),
            },
        )
        _event(paths["events"], f"COMPLETE lane={lane_id} runtime_seconds={elapsed:.9f}")
        inventory = {
            key: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in paths.items()
            if key not in {"directory", "summary"}
        }
        summary["artifact_inventory"] = inventory
        atomic_write_json(paths["summary"], summary)
    except Exception as exc:
        atomic_write_json(
            paths["status"],
            {
                "state": "FAILED",
                "lane_id": lane_id,
                "manifest_sha256": sha256_file(manifest_path),
                "failure": str(exc),
            },
        )
        _event(paths["events"], f"FAILED lane={lane_id} error={exc}")
        raise


def _scaled_state_metrics(initial: NBodyState, final: NBodyState, body_names: Sequence[str]) -> dict[str, Any]:
    scaled = np.column_stack(
        (
            (final.positions - initial.positions) / AU_M,
            (final.velocities - initial.velocities) / (AU_M / JULIAN_YEAR_S),
        )
    )
    per_body = {
        name: {
            "scaled_rms": float(np.sqrt(np.mean(scaled[index] ** 2))),
            "position_error_m": float(np.linalg.norm(final.positions[index] - initial.positions[index])),
            "velocity_error_m_per_s": float(np.linalg.norm(final.velocities[index] - initial.velocities[index])),
        }
        for index, name in enumerate(body_names)
    }
    worst = max(per_body, key=lambda name: per_body[name]["scaled_rms"])
    return {
        "global_scaled_rms": float(np.sqrt(np.mean(scaled**2))),
        "per_body": per_body,
        "worst_body": worst,
    }


def _return_metrics(initial: NBodyState, final: NBodyState, body_names: Sequence[str]) -> dict[str, Any]:
    physical = _scaled_state_metrics(initial, final, body_names)
    initial_energy = _state_energy(initial)["corrected"]
    final_energy = _state_energy(final)["corrected"]
    initial_angular = total_angular_momentum_vector(initial)
    final_angular = total_angular_momentum_vector(final)
    initial_cp, initial_cv = center_of_mass_position_velocity(initial)
    final_cp, final_cv = center_of_mass_position_velocity(final)
    element_errors: dict[str, Any] = {}
    initial_elements = heliocentric_elements_for_state(initial, body_names, sun_index=0)
    final_elements = heliocentric_elements_for_state(final, body_names, sun_index=0)
    for left, right in zip(initial_elements, final_elements):
        element_errors[left.body_name] = {
            "semimajor_axis_relative": abs(right.semi_major_axis_m - left.semi_major_axis_m)
            / max(abs(left.semi_major_axis_m), 1.0),
            "eccentricity_abs": abs(right.eccentricity - left.eccentricity),
            "mean_longitude_abs_rad": abs(
                math.atan2(
                    math.sin(right.mean_longitude_rad - left.mean_longitude_rad),
                    math.cos(right.mean_longitude_rad - left.mean_longitude_rad),
                )
            ),
        }
    physical.update(
        corrected_energy_relative_difference=(final_energy - initial_energy)
        / (abs(initial_energy) if initial_energy != 0.0 else 1.0),
        angular_momentum_vector_relative_difference=float(
            np.linalg.norm(final_angular - initial_angular)
            / max(np.linalg.norm(initial_angular), 1.0)
        ),
        center_of_mass_position_error_m=float(np.linalg.norm(final_cp - initial_cp)),
        center_of_mass_velocity_error_m_per_s=float(np.linalg.norm(final_cv - initial_cv)),
        orbital_element_errors=element_errors,
    )
    return physical


def _configure_whfast(sim: Any, config: dict[str, Any]) -> None:
    sim.ri_whfast.coordinates = config["coordinates"]
    sim.ri_whfast.kernel = config["kernel"]
    sim.ri_whfast.corrector = config["corrector"]
    sim.ri_whfast.corrector2 = config["corrector2"]
    sim.ri_whfast.safe_mode = config["safe_mode"]
    sim.ri_whfast.keep_unsynchronized = config["keep_unsynchronized"]
    sim.ri_whfast.recalculate_coordinates_this_timestep = config.get(
        "recalculate_coordinates_this_timestep", 0
    )


def _synchronize_for_direction_reversal(sim: Any) -> None:
    keep_unsynchronized = int(sim.ri_whfast.keep_unsynchronized)
    if keep_unsynchronized:
        sim.ri_whfast.keep_unsynchronized = 0
    sim.synchronize()
    sim.ri_whfast.keep_unsynchronized = keep_unsynchronized


def _two_body_state(config: dict[str, Any]) -> NBodyState:
    masses = np.asarray([float(value) for value in config["masses_kg"]], dtype=np.float64)
    separation = float(config["separation_m"])
    total = float(np.sum(masses))
    positions = np.zeros((2, 3), dtype=np.float64)
    positions[0, 0] = -masses[1] / total * separation
    positions[1, 0] = masses[0] / total * separation
    relative_accel = G_SI * total / separation**2 + (
        6.0
        * (G_SI * masses[0]) ** 2
        * (1.0 + masses[1] / masses[0])
        / (299_792_458.0**2 * separation**3)
    )
    relative_speed = math.sqrt(relative_accel * separation)
    velocities = np.zeros((2, 3), dtype=np.float64)
    velocities[0, 1] = -masses[1] / total * relative_speed
    velocities[1, 1] = masses[0] / total * relative_speed
    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def validate_reversibility(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit(manifest_path)
    definition = manifest["method_validation"]["reversibility_two_body"]
    output = Path(manifest["paths"]["output_root"]) / "reversibility_method_validation"
    _require(not output.exists(), f"Validation output already exists: {output}")
    output.mkdir(parents=True)
    rebound = optional_import_module("rebound")
    _require(rebound is not None, "REBOUND is unavailable.")
    results = {}
    for case in definition["cases"]:
        config = {**definition["common_configuration"], **case["configuration"]}
        _require(canonical_hash(config) == case["configuration_fingerprint"], "Two-body fingerprint mismatch.")
        state0 = _two_body_state(config)
        sim = build_rebound_simulation(
            rebound,
            state0,
            integrator="whfast",
            step_s=float(config["step_days"]) * DAY_S,
            ias15_epsilon=1e-10,
        )
        _configure_whfast(sim, config)
        backend = load_c_backend()
        backend.attach(sim, coefficient_scale=1.0, include_central_response=True)
        sim.steps(config["forward_steps"])
        _synchronize_for_direction_reversal(sim)
        forward_time = float(sim.t)
        sim.dt = -abs(float(sim.dt))
        sim.steps(config["backward_steps"])
        sim.synchronize()
        final = rebound_state_from_sim(sim, state0.masses)
        callback = backend.stats(sim)
        metrics = _return_metrics(state0, final, ["sun", "mercury"])
        passed = (
            abs(float(sim.t)) <= definition["required_final_time_abs_seconds_max"]
            and metrics["global_scaled_rms"] <= definition["required_scaled_state_rms_max"]
            and int(callback["callback_invocations"]) == config["expected_callback_invocations"]
            and int(callback["nonfinite_result_count"]) == 0
        )
        results[case["id"]] = {
            "configuration": config,
            "configuration_fingerprint": case["configuration_fingerprint"],
            "forward_time_seconds": forward_time,
            "return_time_seconds": float(sim.t),
            "callback_stats": callback,
            "metrics": metrics,
            "passed": passed,
        }
    ratio_pass = True
    for mode in ("current_sync", "min_sync"):
        coarse = results[f"two_body_{mode}_0p5d"]["metrics"]["global_scaled_rms"]
        fine = results[f"two_body_{mode}_0p25d"]["metrics"]["global_scaled_rms"]
        ratio_pass = ratio_pass and fine <= definition["required_0p25_error_over_0p5_error_max"] * coarse + 1e-30
    payload = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "passed": all(item["passed"] for item in results.values()) and ratio_pass,
        "timestep_ratio_passed": ratio_pass,
        "direction_reversal_procedure": (
            "At the forward comparison point, temporarily set keep_unsynchronized=0, "
            "synchronize with the positive timestep so internal Jacobi coordinates are "
            "at the physical endpoint, restore the configured mode, then negate dt."
        ),
        "results": results,
    }
    atomic_write_json(output / "summary.json", payload)
    _require(payload["passed"], "Two-body reversibility method validation failed.")


def benchmark_ias15(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit(manifest_path)
    definition = manifest["method_validation"]["ias15_benchmark"]
    root = Path(manifest["paths"]["output_root"])
    summary_path = root / "ias15_benchmark_summary.json"
    _require(not summary_path.exists(), f"Benchmark output already exists: {summary_path}")
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    projections = []
    for case in definition["cases"]:
        config = {**manifest["common_configuration"], **case["configuration"]}
        _require(canonical_hash(config) == case["configuration_fingerprint"], "Benchmark fingerprint mismatch.")
        _, sim, backend, bodies, state0 = _build_simulation(manifest, config)
        case_dir = root / case["id"]
        _require(not case_dir.exists(), f"Benchmark case exists: {case_dir}")
        case_dir.mkdir()
        start_state = rebound_state_from_sim(sim, state0.masses)
        start = time.perf_counter()
        sim.integrate(config["duration_years"] * JULIAN_YEAR_S, exact_finish_time=1)
        elapsed = time.perf_counter() - start
        end_state = rebound_state_from_sim(sim, state0.masses)
        callback = backend.stats(sim)
        _require(
            float(sim.t) == config["duration_years"] * JULIAN_YEAR_S,
            "IAS15 benchmark missed its exact endpoint.",
        )
        _require(
            np.all(np.isfinite(end_state.positions)) and np.all(np.isfinite(end_state.velocities)),
            "IAS15 benchmark produced a nonfinite state.",
        )
        _require(int(callback["callback_invocations"]) > 0, "IAS15 benchmark did not exercise the callback.")
        _require(int(callback["nonfinite_result_count"]) == 0, "IAS15 benchmark callback was nonfinite.")
        rows = [
            *_state_rows(start_state, bodies, sample_index=0, time_seconds=0.0, fingerprint=case["configuration_fingerprint"]),
            *_state_rows(end_state, bodies, sample_index=1, time_seconds=float(sim.t), fingerprint=case["configuration_fingerprint"]),
        ]
        _write_csv_atomic(case_dir / "physical_state.csv", PHYSICAL_STATE_FIELDS, rows)
        projected = elapsed * 100.0
        projections.append(projected)
        results[case["id"]] = {
            "configuration": config,
            "configuration_fingerprint": case["configuration_fingerprint"],
            "runtime_seconds": elapsed,
            "projected_10000y_runtime_seconds": projected,
            "callback_stats": callback,
            "samples": 2,
            "state_rows": 20,
            "state_sha256": sha256_file(case_dir / "physical_state.csv"),
        }
    combined = float(sum(projections))
    passed = combined <= definition["combined_runtime_limit_seconds"]
    payload = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(manifest_path),
        "combined_projected_runtime_seconds": combined,
        "limit_seconds": definition["combined_runtime_limit_seconds"],
        "passed": passed,
        "results": results,
    }
    atomic_write_json(summary_path, payload)
    _require(passed, "Projected IAS15 runtime exceeds preregistered limit.")



def _require_reversibility_gate(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    if manifest.get("continuation_method_gate") is not None:
        evidence = _audit_continuation_provenance(
            manifest, Path(manifest["paths"]["project_root"])
        )
        _require(evidence is not None, "Continuation method-gate evidence is missing.")
        return {
            "source": "manifest 14",
            "status": evidence["source_manifest_14_status"],
            "summary_sha256": evidence["source_manifest_14_summary_sha256"],
            "fine_coarse_ratios_diagnostic_only": evidence[
                "fine_coarse_ratios_diagnostic_only"
            ],
        }
    validation_path = (
        Path(manifest["paths"]["output_root"])
        / "reversibility_method_validation/summary.json"
    )
    validation = _load_json(validation_path, "reversibility validation")
    _require(
        validation.get("passed") is True,
        "Reversibility method validation did not pass.",
    )
    _require(
        validation.get("manifest_sha256") == sha256_file(manifest_path),
        "Validation manifest mismatch.",
    )
    return {
        "source": "local manifest-13 method validation",
        "status": "PASSED",
        "summary_sha256": sha256_file(validation_path),
        "fine_coarse_ratios_diagnostic_only": False,
    }


def run_reversibility(manifest_path: Path, lane_id: str) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit_payload = audit(manifest_path)
    method_gate = _require_reversibility_gate(manifest_path, manifest)
    lane, config = _expanded_lane(manifest, lane_id)
    _require(config["kind"] == "whfast_reversibility", "Not a reversibility lane.")
    paths = _lane_paths(manifest, lane_id)
    _require(not paths["directory"].exists(), f"Collision-safe lane already exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True)
    _event(paths["events"], f"START lane={lane_id} command={' '.join(sys.argv)}")
    try:
        _, sim, backend, bodies, state0 = _build_simulation(manifest, config)
        initial = rebound_state_from_sim(sim, state0.masses)
        start = time.perf_counter()
        sim.steps(config["forward_steps"])
        _synchronize_for_direction_reversal(sim)
        forward = rebound_state_from_sim(sim, state0.masses)
        forward_time = float(sim.t)
        sim.dt = -abs(float(sim.dt))
        sim.steps(config["backward_steps"])
        sim.synchronize()
        returned = rebound_state_from_sim(sim, state0.masses)
        elapsed = time.perf_counter() - start
        callback = backend.stats(sim)
        _require(int(callback["callback_invocations"]) == config["expected_callback_invocations"], "Reversibility callback total mismatch.")
        _require(int(callback["nonfinite_result_count"]) == 0, "Reversibility nonfinite callback result.")
        _require(abs(float(sim.t)) <= 1e-6, "Reversibility return time mismatch.")
        rows = [
            *_state_rows(initial, bodies, sample_index=0, time_seconds=0.0, fingerprint=lane["configuration_fingerprint"]),
            *_state_rows(forward, bodies, sample_index=1, time_seconds=forward_time, fingerprint=lane["configuration_fingerprint"]),
            *_state_rows(returned, bodies, sample_index=2, time_seconds=float(sim.t), fingerprint=lane["configuration_fingerprint"]),
        ]
        _write_csv_atomic(paths["state"], PHYSICAL_STATE_FIELDS, rows)
        _event(paths["events"], f"COMPLETE lane={lane_id} runtime_seconds={elapsed:.9f}")
        summary = {
            "schema_version": 1,
            "status": "COMPLETED",
            "lane_id": lane_id,
            "manifest_sha256": sha256_file(manifest_path),
            "configuration": config,
            "configuration_fingerprint": lane["configuration_fingerprint"],
            "command": sys.argv,
            "audit": audit_payload,
            "method_gate": method_gate,
            "settings": _settings(sim),
            "direction_reversal_procedure": (
                "At the forward comparison point, temporarily set keep_unsynchronized=0, "
                "synchronize with the positive timestep so internal Jacobi coordinates are "
                "at the physical endpoint, restore the configured mode, then negate dt."
            ),
            "forward_time_seconds": forward_time,
            "return_time_seconds": float(sim.t),
            "runtime_seconds": elapsed,
            "callback_stats": callback,
            "metrics": _return_metrics(initial, returned, bodies),
            "samples": 3,
            "state_rows": 30,
            "artifact_inventory": {
                "state": {"path": str(paths["state"]), "size_bytes": paths["state"].stat().st_size, "sha256": sha256_file(paths["state"])},
                "events": {"path": str(paths["events"]), "size_bytes": paths["events"].stat().st_size, "sha256": sha256_file(paths["events"])},
            },
        }
        atomic_write_json(paths["summary"], summary)
    except Exception as exc:
        atomic_write_json(paths["status"], {"state": "FAILED", "lane_id": lane_id, "failure": str(exc)})
        _event(paths["events"], f"FAILED lane={lane_id} error={exc}")
        raise


def _ols(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    _require(len(times) == len(values) and len(values) >= 3, "OLS history is incomplete.")
    centered_t = times - np.mean(times)
    centered_v = values - np.mean(values)
    denominator = float(np.dot(centered_t, centered_t))
    slope = float(np.dot(centered_t, centered_v) / denominator)
    intercept = float(np.mean(values) - slope * np.mean(times))
    residual = values - (intercept + slope * times)
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(centered_v, centered_v))
    standard_error = math.sqrt((sse / (len(times) - 2)) / denominator)
    critical = float(stats.t.ppf(0.975, len(times) - 2))
    return {
        "slope_per_year": slope,
        "intercept": intercept,
        "ci95_low": slope - critical * standard_error,
        "ci95_high": slope + critical * standard_error,
        "r_squared": 1.0 - sse / sst if sst > 0.0 else 1.0,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_peak_to_peak": float(np.ptp(residual)),
    }


def _energy_statistics(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    fit = _ols(times, values)
    absolute = np.abs(values)
    worst = int(np.argmax(absolute))
    return {
        "signed_endpoint_change": float(values[-1] - values[0]),
        "max_abs": float(absolute[worst]),
        "max_abs_worst_epoch_years": float(times[worst]),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p99_abs": float(np.percentile(absolute, 99.0)),
        "fitted_slope_per_year": fit["slope_per_year"],
        "fitted_intercept": fit["intercept"],
        "fitted_change_over_history": fit["slope_per_year"] * float(times[-1] - times[0]),
        "fitted_change_per_myr": fit["slope_per_year"] * 1_000_000.0,
        "ci95_low_per_year": fit["ci95_low"],
        "ci95_high_per_year": fit["ci95_high"],
        "r_squared": fit["r_squared"],
        "detrended_rms": fit["residual_rms"],
        "detrended_peak_to_peak": fit["residual_peak_to_peak"],
    }


def long_history_analysis(times: np.ndarray, values: np.ndarray, step_days: float) -> dict[str, Any]:
    result: dict[str, Any] = _energy_statistics(times, values)
    blocks = []
    for index in range(10):
        mask = (times >= index * 100000.0) & (times <= (index + 1) * 100000.0)
        block_stats = _energy_statistics(times[mask], values[mask])
        blocks.append({"index": index, **block_stats})
    prefixes = {}
    for kyr in (10, 30, 100, 300, 1000):
        mask = times <= kyr * 1000.0
        stats_payload = _energy_statistics(times[mask], values[mask])
        stats_payload["range"] = float(np.ptp(values[mask]))
        prefixes[str(kyr)] = stats_payload
    fit = _ols(times, values)
    residual = values - (fit["intercept"] + fit["slope_per_year"] * times)
    autocorrelation = {
        str(lag): float(np.corrcoef(residual[:-lag], residual[lag:])[0, 1])
        for lag in (1, 10, 100, 1000)
    }
    frequency, power = signal.periodogram(
        values, fs=0.01, detrend="linear", scaling="density", window="boxcar"
    )
    order = np.argsort(power[1:])[-5:][::-1] + 1
    periodogram = [
        {"period_years": float(1.0 / frequency[index]), "power": float(power[index])}
        for index in order
    ]
    allan = {}
    for length in (1, 3, 10, 30, 100, 300, 1000):
        averages = np.convolve(values, np.ones(length) / length, mode="valid")
        differences = averages[length:] - averages[:-length]
        allan[str(length * 100)] = (
            float(np.sqrt(0.5 * np.mean(differences**2))) if len(differences) else None
        )
    increments = np.diff(values)
    increment_payload = {
        "mean": float(np.mean(increments)),
        "sample_std": float(np.std(increments, ddof=1)),
        "rms": float(np.sqrt(np.mean(increments**2))),
        "median": float(np.median(increments)),
        "mad": float(np.median(np.abs(increments - np.median(increments)))),
        "p01": float(np.percentile(increments, 1.0)),
        "p99": float(np.percentile(increments, 99.0)),
        "skew": float(stats.skew(increments, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(increments, fisher=True, bias=False)),
        "positive_fraction": float(np.mean(increments > 0.0)),
        "lag1_autocorrelation": float(np.corrcoef(increments[:-1], increments[1:])[0, 1]),
    }
    durations = np.asarray([10.0, 30.0, 100.0, 300.0, 1000.0])
    ranges = np.asarray([prefixes[str(int(value))]["range"] for value in durations])
    endpoints = np.abs(
        np.asarray([prefixes[str(int(value))]["signed_endpoint_change"] for value in durations])
    )
    result.update(
        blocks=blocks,
        same_sign_block_count=sum(
            math.copysign(1.0, block["fitted_slope_per_year"])
            == math.copysign(1.0, result["fitted_slope_per_year"])
            for block in blocks
        ),
        prefixes=prefixes,
        residual_autocorrelation=autocorrelation,
        periodogram_top_peaks=periodogram,
        allan_deviation_by_tau_years=allan,
        increment_distribution=increment_payload,
        energy_change_per_step=result["fitted_slope_per_year"] * step_days / 365.25,
        range_elapsed_exponent=float(np.polyfit(np.log(durations), np.log(ranges), 1)[0]),
        endpoint_elapsed_exponent=float(np.polyfit(np.log(durations), np.log(endpoints), 1)[0]),
    )
    return result


def _load_step3c_history(manifest: dict[str, Any], token: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(manifest["paths"]["step3c_timeseries_root"]) / f"m0_conv_{token}_1myr_s12345_energy_methods.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["time_years"]) for row in rows], dtype=np.float64),
        np.asarray([float(row["decimal_rel_change"]) for row in rows], dtype=np.float64),
    )


def _read_physical_groups(path: Path, body_names: Sequence[str]) -> tuple[np.ndarray, list[NBodyState], list[list[dict[str, str]]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) % len(body_names) == 0, f"Incomplete state groups: {path}")
    groups = [rows[index : index + len(body_names)] for index in range(0, len(rows), len(body_names))]
    times = []
    states = []
    for sample_index, group in enumerate(groups):
        _require([row["body_name"] for row in group] == list(body_names), f"Body order mismatch: {path}")
        _require(all(int(row["sample_index"]) == sample_index for row in group), f"Sample order mismatch: {path}")
        times.append(float(group[0]["time_years"]))
        states.append(
            NBodyState(
                masses=np.asarray([float(row["mass_kg"]) for row in group]),
                positions=np.asarray([[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in group]),
                velocities=np.asarray([[float(row["vx_m_per_s"]), float(row["vy_m_per_s"]), float(row["vz_m_per_s"])] for row in group]),
            )
        )
    return np.asarray(times), states, groups


def _recompute_lane_energy(
    manifest_path: Path,
    manifest: dict[str, Any],
    lane_id: str,
    body_names: Sequence[str],
) -> dict[str, Any]:
    paths = _lane_paths(manifest, lane_id)
    times, states, string_groups = _read_physical_groups(paths["state"], body_names)
    with paths["progress"].open(newline="") as handle:
        progress = list(csv.DictReader(handle))
    _require(len(progress) == len(states), f"Progress/state mismatch: {lane_id}")
    histories = {"recorded": [], "float64": [], "compensated": []}
    decimal_history: list[Decimal] = []
    references: dict[str, float] = {}
    decimal_reference: Decimal | None = None
    output_dir = Path(manifest["paths"]["output_root"]) / "recomputed_energy"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"{lane_id}.csv"
    _require(not output.exists(), f"Recomputed timeseries already exists: {output}")
    temporary = output.with_name(output.name + ".tmp")
    telemetry_max = 0.0
    agreement_max = 0.0
    with localcontext() as context:
        context.prec = 60
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ENERGY_TIMESERIES_FIELDS)
            writer.writeheader()
            for index, (state, rows, progress_row) in enumerate(zip(states, string_groups, progress)):
                float_energy = _state_energy(state)
                compensated = compensated_energy(
                    state.masses,
                    state.positions,
                    state.velocities,
                    gravitational_constant=G_SI,
                    speed_of_light=299_792_458.0,
                    coefficient_scale=1.0,
                )
                decimal_value = decimal_energy(
                    rows,
                    gravitational_constant=Decimal("6.67430e-11"),
                    speed_of_light=Decimal("299792458"),
                    coefficient_scale=Decimal(1),
                )
                if index == 0:
                    references = {
                        "float64": float_energy["corrected"],
                        "compensated": compensated["corrected"],
                    }
                    decimal_reference = decimal_value["corrected"]
                _require(decimal_reference is not None, "Decimal reference missing.")
                recorded = float(progress_row["corrected_energy_rel_change"])
                float_drift = (float_energy["corrected"] - references["float64"]) / abs(references["float64"])
                compensated_drift = (compensated["corrected"] - references["compensated"]) / abs(references["compensated"])
                decimal_drift = (decimal_value["corrected"] - decimal_reference) / abs(decimal_reference)
                telemetry_max = max(telemetry_max, abs(float_drift - recorded))
                agreement_max = max(
                    agreement_max,
                    abs(float(Decimal.from_float(compensated_drift) - decimal_drift)),
                )
                histories["recorded"].append(recorded)
                histories["float64"].append(float_drift)
                histories["compensated"].append(compensated_drift)
                decimal_history.append(decimal_drift)
                writer.writerow(
                    {
                        "schema_version": 1,
                        "manifest_sha256": sha256_file(manifest_path),
                        "lane_id": lane_id,
                        "sample_index": index,
                        "time_years": rows[0]["time_years"],
                        "recorded_corrected_rel_change": progress_row["corrected_energy_rel_change"],
                        "float64_corrected_rel_change": float_drift,
                        "compensated_corrected_rel_change": compensated_drift,
                        "decimal_corrected_rel_change": str(decimal_drift),
                        "float64_minus_recorded": float_drift - recorded,
                        "compensated_minus_decimal": str(Decimal.from_float(compensated_drift) - decimal_drift),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    statistics_payload = {
        method: _energy_statistics(times, np.asarray(values))
        for method, values in histories.items()
    }
    statistics_payload["decimal"] = _energy_statistics(
        times, np.asarray([float(value) for value in decimal_history])
    )
    return {
        "times": times,
        "states": states,
        "decimal_values": np.asarray([float(value) for value in decimal_history]),
        "statistics": statistics_payload,
        "telemetry_reproduction_max_abs": telemetry_max,
        "compensated_decimal_max_abs": agreement_max,
        "timeseries": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def _existing_states(manifest: dict[str, Any], token: str, body_names: Sequence[str], max_years: float) -> tuple[np.ndarray, list[NBodyState]]:
    lane_id = f"m0_conv_{token}_1myr_s12345"
    path = Path(manifest["paths"]["existing_convergence_root"]) / lane_id / f"gr_tangent_state_{lane_id}.csv"
    times, states, _ = _read_physical_groups(path, body_names)
    count = int(max_years / 100.0) + 1
    return times[:count], states[:count]


def _matched_state_comparison(
    times_left: np.ndarray,
    left: Sequence[NBodyState],
    times_right: np.ndarray,
    right: Sequence[NBodyState],
    body_names: Sequence[str],
) -> dict[str, Any]:
    _require(np.array_equal(times_left, times_right), "Matched state times differ.")
    position = np.stack([a.positions - b.positions for a, b in zip(left, right)]) / AU_M
    velocity = np.stack([a.velocities - b.velocities for a, b in zip(left, right)]) / (AU_M / JULIAN_YEAR_S)
    scaled = np.concatenate((position, velocity), axis=2)
    per_body = {
        name: float(np.sqrt(np.mean(scaled[:, index, :] ** 2)))
        for index, name in enumerate(body_names)
    }
    worst_body = max(per_body, key=per_body.get)
    orbital: dict[str, dict[str, float]] = {
        name: {"semimajor_axis_relative_max": 0.0, "eccentricity_abs_max": 0.0, "mean_longitude_abs_rad_max": 0.0}
        for name in body_names[1:]
    }
    for state_left, state_right in zip(left, right):
        elements_left = heliocentric_elements_for_state(state_left, body_names, sun_index=0)
        elements_right = heliocentric_elements_for_state(state_right, body_names, sun_index=0)
        for a, b in zip(elements_left, elements_right):
            target = orbital[a.body_name]
            target["semimajor_axis_relative_max"] = max(
                target["semimajor_axis_relative_max"],
                abs(a.semi_major_axis_m - b.semi_major_axis_m)
                / max(abs((a.semi_major_axis_m + b.semi_major_axis_m) / 2.0), 1.0),
            )
            target["eccentricity_abs_max"] = max(
                target["eccentricity_abs_max"], abs(a.eccentricity - b.eccentricity)
            )
            angle = abs(
                math.atan2(
                    math.sin(a.mean_longitude_rad - b.mean_longitude_rad),
                    math.cos(a.mean_longitude_rad - b.mean_longitude_rad),
                )
            )
            target["mean_longitude_abs_rad_max"] = max(target["mean_longitude_abs_rad_max"], angle)
    return {
        "samples": len(left),
        "global_scaled_rms": float(np.sqrt(np.mean(scaled**2))),
        "per_body_scaled_rms": per_body,
        "worst_body": worst_body,
        "worst_body_scaled_rms": per_body[worst_body],
        "orbital_elements": orbital,
    }


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def _block_metrics(times: np.ndarray, values: np.ndarray, duration_years: float) -> list[dict[str, float]]:
    width = duration_years / 10.0
    return [
        _energy_statistics(
            times[(times >= index * width) & (times <= (index + 1) * width)],
            values[(times >= index * width) & (times <= (index + 1) * width)],
        )
        for index in range(10)
    ]


def control_history_analysis(
    times: np.ndarray, values: np.ndarray, step_days: float
) -> dict[str, Any]:
    result: dict[str, Any] = _energy_statistics(times, values)
    blocks = _block_metrics(times, values, float(times[-1] - times[0]))
    result.update(
        blocks=blocks,
        same_sign_block_count=sum(
            math.copysign(1.0, block["fitted_slope_per_year"])
            == math.copysign(1.0, result["fitted_slope_per_year"])
            for block in blocks
        ),
        energy_change_per_step=result["fitted_slope_per_year"] * step_days / 365.25,
    )
    return result


def _reproduction(
    left_times: np.ndarray,
    left_values: np.ndarray,
    right_times: np.ndarray,
    right_values: np.ndarray,
    threshold: dict[str, Any],
) -> dict[str, Any]:
    _require(np.array_equal(left_times, right_times), "Reproduction times differ.")
    left = _energy_statistics(left_times, left_values)
    right = _energy_statistics(right_times, right_values)
    metric_differences = {
        key: _relative_difference(left[key], right[key])
        for key in ("max_abs", "rms", "p99_abs")
    }
    left_blocks = _block_metrics(left_times, left_values, float(left_times[-1]))
    right_blocks = _block_metrics(right_times, right_values, float(right_times[-1]))
    paired_sign = sum(
        math.copysign(1.0, a["fitted_slope_per_year"])
        == math.copysign(1.0, b["fitted_slope_per_year"])
        for a, b in zip(left_blocks, right_blocks)
    )
    left_vector = np.asarray([item["fitted_slope_per_year"] for item in left_blocks])
    right_vector = np.asarray([item["fitted_slope_per_year"] for item in right_blocks])
    block_rms = float(
        np.sqrt(np.mean((left_vector - right_vector) ** 2))
        / max(np.sqrt(np.mean(left_vector**2)), np.sqrt(np.mean(right_vector**2)), 1e-300)
    )
    fitted_difference = _relative_difference(
        left["fitted_change_over_history"], right["fitted_change_over_history"]
    )
    passed = (
        math.copysign(1.0, left["fitted_slope_per_year"])
        == math.copysign(1.0, right["fitted_slope_per_year"])
        and fitted_difference <= threshold["normalized_fitted_change_relative_difference_max"]
        and all(value <= threshold["energy_max_rms_p99_relative_difference_max"] for value in metric_differences.values())
        and paired_sign >= threshold["paired_block_common_sign_count_min"]
        and block_rms <= threshold["block_slope_vector_normalized_rms_difference_max"]
    )
    return {
        "passed": passed,
        "left": left,
        "right": right,
        "fitted_change_relative_difference": fitted_difference,
        "metric_relative_differences": metric_differences,
        "paired_block_common_sign_count": paired_sign,
        "block_slope_normalized_rms_difference": block_rms,
    }


def _material_reduction(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    physical_ok: bool,
) -> dict[str, Any]:
    ratios = {
        "absolute_normalized_fitted_slope": abs(candidate["fitted_slope_per_year"])
        / max(abs(baseline["fitted_slope_per_year"]), 1e-300),
        "max_abs": candidate["max_abs"] / max(baseline["max_abs"], 1e-300),
        "rms": candidate["rms"] / max(baseline["rms"], 1e-300),
        "p99_abs": candidate["p99_abs"] / max(baseline["p99_abs"], 1e-300),
    }
    return {
        "passed": all(value <= 0.25 for value in ratios.values()) and physical_ok,
        "ratios": ratios,
        "physical_accuracy_not_worse": physical_ok,
    }


def _systematic_signature(
    coarse: dict[str, Any], fine: dict[str, Any], threshold: dict[str, Any]
) -> dict[str, Any]:
    slope_ratio = fine["fitted_slope_per_year"] / coarse["fitted_slope_per_year"]
    q_coarse = coarse["fitted_slope_per_year"] * 0.5 / 365.25
    q_fine = fine["fitted_slope_per_year"] * 0.25 / 365.25
    q_relative = _relative_difference(q_coarse, q_fine)
    passed = (
        threshold["slope_ratio_0p25_over_0p5_min"]
        <= slope_ratio
        <= threshold["slope_ratio_0p25_over_0p5_max"]
        and math.copysign(1.0, q_coarse) == math.copysign(1.0, q_fine)
        and q_relative <= threshold["q_relative_difference_max"]
        and coarse["same_sign_block_count"] >= threshold["same_sign_blocks_min"]
        and fine["same_sign_block_count"] >= threshold["same_sign_blocks_min"]
    )
    return {
        "passed": passed,
        "slope_ratio_0p25_over_0p5": slope_ratio,
        "q_0p5": q_coarse,
        "q_0p25": q_fine,
        "q_relative_difference": q_relative,
        "coarse_same_sign_blocks": coarse["same_sign_block_count"],
        "fine_same_sign_blocks": fine["same_sign_block_count"],
    }


def classify_mechanism(evidence: dict[str, Any]) -> tuple[str, str]:
    if not evidence["integrity_passed"] or not evidence["ias15_tolerance_converged"] or not evidence["reversibility_valid"]:
        return "BLOCKED", "BLOCKED"
    if evidence["ias15_force_problem"]:
        return "CORRECTED_INVARIANT_OR_FORCE_PROBLEM", "STEP3_FORCE_INVARIANT_PROBLEM"
    if evidence["current_reproduces_full_both"] and evidence["min_material_reduction_both"] and not evidence["ias15_reproduces"]:
        return "SYNCHRONIZATION_RECALCULATION_BIAS", "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED"
    if evidence["current_material_reduction_both"] and evidence["current_min_compatible_both"] and not evidence["ias15_reproduces"]:
        return "VARIATION_MEGNO_COUPLING", "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED"
    if evidence["full_systematic_signature"] and evidence["current_systematic_signature"] and not evidence["any_sync_material_reduction"] and not evidence["ias15_reproduces"]:
        return "SYSTEMATIC_WHFAST_STEP_BIAS", "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
    if evidence["random_walk_rules_passed"]:
        return "RANDOM_WALK_ROUNDOFF", "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
    if evidence["bounded_rules_passed"]:
        return "BOUNDED_ENERGY_OSCILLATION", "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
    return "MIXED_OR_INCONCLUSIVE", "STEP3_DIAGNOSIS_INCONCLUSIVE"


def _reversibility_timestep_diagnostics(
    reversibility: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "global_scaled_rms",
        "corrected_energy_relative_difference",
        "angular_momentum_vector_relative_difference",
        "center_of_mass_position_error_m",
        "center_of_mass_velocity_error_m_per_s",
    )
    diagnostics = {}
    for mode in ("current_sync", "min_sync"):
        coarse = reversibility[
            f"m0_diag_reversibility_{mode}_0p5d_10k"
        ]["metrics"]
        fine = reversibility[
            f"m0_diag_reversibility_{mode}_0p25d_10k"
        ]["metrics"]
        metrics = {}
        for name in metric_names:
            coarse_error = abs(float(coarse[name]))
            fine_error = abs(float(fine[name]))
            ratio = fine_error / max(coarse_error, 1e-300)
            metrics[name] = {
                "coarse_0p5d_abs": coarse_error,
                "fine_0p25d_abs": fine_error,
                "fine_over_coarse_ratio": ratio,
                "apparent_order": math.log(ratio) / math.log(0.5),
            }
        diagnostics[mode] = {
            "diagnostic_only": True,
            "affects_validity": False,
            "metrics": metrics,
        }
    return diagnostics


def _figure_save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": "mini_ephemeris"})
    plt.close(fig)


def _make_figures(
    manifest: dict[str, Any],
    long_histories: dict[str, dict[str, Any]],
    recomputed: dict[str, dict[str, Any]],
    reversibility: dict[str, Any],
) -> list[dict[str, Any]]:
    directory = Path(manifest["paths"]["figure_directory"])
    _require(not directory.exists(), f"Figure directory already exists: {directory}")
    directory.mkdir(parents=True)
    figures = []
    colors = {"0p5d": "#1f77b4", "0p25d": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for token, payload in long_histories.items():
        ax.plot(payload["times"] / 1000.0, payload["values"], color=colors[token], lw=0.9, label=token)
        fit = payload["metrics"]
        ax.plot(
            payload["times"] / 1000.0,
            fit["fitted_intercept"] + fit["fitted_slope_per_year"] * payload["times"],
            color=colors[token],
            ls="--",
            lw=1.2,
        )
    ax.set(xlabel="Time (kyr)", ylabel="Corrected-energy relative change", title="Existing 1 Myr histories and global fits")
    ax.legend()
    ax.grid(alpha=0.25)
    path = directory / "long_history_energy_and_fits.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(10)
    ax.bar(x - 0.18, [item["fitted_slope_per_year"] * 1e16 for item in long_histories["0p5d"]["metrics"]["blocks"]], 0.36, label="0.5 day", color=colors["0p5d"])
    ax.bar(x + 0.18, [item["fitted_slope_per_year"] * 1e16 for item in long_histories["0p25d"]["metrics"]["blocks"]], 0.36, label="0.25 day", color=colors["0p25d"])
    ax.set(xlabel="100 kyr block", ylabel="Slope (1e-16 / year)", title="Blockwise corrected-energy slopes")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = directory / "blockwise_slopes.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for axis, token in zip(axes, ("0p5d", "0p25d")):
        full = long_histories[token]
        mask = full["times"] <= 100000.0
        axis.plot(full["times"][mask] / 1000.0, full["values"][mask], label="full MEGNO", color="#444444")
        for mode, color in (("current_sync", "#2ca02c"), ("min_sync", "#9467bd")):
            lane_id = f"m0_diag_phys_{mode}_{token}_100k"
            axis.plot(recomputed[lane_id]["times"] / 1000.0, recomputed[lane_id]["decimal_values"], label=mode, color=color)
        axis.set(title=token, xlabel="Time (kyr)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Corrected-energy relative change")
    axes[1].legend(fontsize=8)
    path = directory / "full_current_min_sync_comparison.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lane_id, color in (("m0_diag_phys_ias15_eps1e12_10k", "#ff7f0e"), ("m0_diag_phys_ias15_eps1e13_10k", "#17becf")):
        ax.plot(recomputed[lane_id]["times"] / 1000.0, recomputed[lane_id]["decimal_values"], label=lane_id.split("_")[-2], color=color)
    for token, color in colors.items():
        full = long_histories[token]
        mask = full["times"] <= 10000.0
        ax.plot(full["times"][mask] / 1000.0, full["values"][mask], ls="--", color=color, label=f"WHFast {token}")
    ax.set(xlabel="Time (kyr)", ylabel="Corrected-energy relative change", title="IAS15 tolerance comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    path = directory / "ias15_comparison.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(9, 4.5))
    lane_ids = list(reversibility)
    values = [reversibility[lane]["metrics"]["global_scaled_rms"] for lane in lane_ids]
    ax.bar(np.arange(len(lane_ids)), values, color=["#2ca02c", "#1f77b4", "#9467bd", "#d62728"])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(lane_ids)), [lane.replace("m0_diag_reversibility_", "").replace("_10k", "") for lane in lane_ids], rotation=20, ha="right")
    ax.set(ylabel="Return scaled-state RMS", title="10 kyr forward-backward reversibility")
    ax.grid(axis="y", alpha=0.25)
    path = directory / "reversibility_errors.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return figures


def _markdown_report(payload: dict[str, Any]) -> str:
    ias15 = payload["ias15"]
    ias15_state = ias15["state_agreement"]
    ias15_thresholds = ias15["thresholds"]
    evidence = payload["classification_evidence"]
    lines = [
        f"# {payload['primary_mechanism']}",
        "",
        f"Step 3 diagnosis status: **{payload['step3_diagnosis_status']}**",
        "",
        "## Classification Gates",
        "",
        f"- Artifact and telemetry integrity: {evidence['integrity_passed']}.",
        f"- Manifest-14 reversibility validity: {evidence['reversibility_valid']}.",
        f"- IAS15 tolerance convergence: {ias15['tolerance_converged']}.",
        f"- IAS15 global scaled-state RMS: {ias15_state['global_scaled_rms']:.12g} "
        f"(limit {ias15_thresholds['global_scaled_state_rms_max']:.12g}).",
        f"- IAS15 worst body: {ias15_state['worst_body']} at "
        f"{ias15_state['worst_body_scaled_rms']:.12g} "
        f"(per-body limit {ias15_thresholds['per_body_scaled_state_rms_max']:.12g}).",
        f"- IAS15 corrected-energy history difference: "
        f"{ias15['energy_history_max_abs_difference']:.12g} "
        f"(limit {ias15_thresholds['energy_history_max_abs_difference']:.12g}); "
        f"orbital elements agree: {ias15['orbital_elements_agree']}.",
        "",
        "## Long History",
        "",
        "| Lane | Fitted change / Myr | R2 | q per step | Same-sign blocks | Range exponent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for token, item in payload["long_history"].items():
        lines.append(
            f"| {token} | {item['fitted_change_per_myr']:.12g} | {item['r_squared']:.9g} | "
            f"{item['energy_change_per_step']:.12g} | {item['same_sign_block_count']} | "
            f"{item['range_elapsed_exponent']:.6g} |"
        )
    lines.extend(["", "## Controls", ""])
    for lane_id, item in payload["new_lanes"].items():
        if "energy" in item:
            energy = item["energy"]["statistics"]["decimal"]
            lines.append(
                f"- {lane_id}: fitted change/Myr `{energy['fitted_change_per_myr']:.12g}`, "
                f"max `{energy['max_abs']:.12g}`, runtime `{item['runtime_seconds']:.3f}` s."
            )
    lines.extend(["", "## Reversibility Diagnostics", ""])
    lines.extend(
        [
            "Fine/coarse ratios and apparent orders are diagnostic only and do not affect validity.",
            "",
            "| Mode | Global RMS 0.5 d | Global RMS 0.25 d | Fine/coarse | Apparent order |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, item in payload["reversibility_diagnostics"].items():
        metric = item["metrics"]["global_scaled_rms"]
        lines.append(
            f"| {mode} | {metric['coarse_0p5d_abs']:.12g} | "
            f"{metric['fine_0p25d_abs']:.12g} | "
            f"{metric['fine_over_coarse_ratio']:.12g} | "
            f"{metric['apparent_order']:.9g} |"
        )
    lines.extend(["", "## Candidate Mechanisms", ""])
    for name, item in payload["candidate_mechanisms"].items():
        lines.append(f"- **{name}**: {item['assessment']}. Evidence for: {item['for']}. Evidence against: {item['against']}.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Present production configuration must change: `{payload['production_configuration_must_change']}`.",
            f"- A final 0.125-day lane is justified now: `{payload['final_0p125_lane_justified']}`.",
            "- Historical Step 3, Step 3b, and Step 3c statuses remain unchanged.",
            "- No production timestep is promoted, and no Stage 4 command is provided.",
            "",
            "## Next Action",
            "",
            payload["smallest_next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def analyze(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit_payload = audit(manifest_path)
    body_names = manifest["common_configuration"]["body_names"]
    scientific_ids = [lane["id"] for lane in manifest["lanes"] if lane["configuration"]["kind"] in {"whfast_physical_control", "ias15_physical_reference"}]
    reversibility_ids = [lane["id"] for lane in manifest["lanes"] if lane["configuration"]["kind"] == "whfast_reversibility"]
    summaries = {}
    for lane_id in scientific_ids + reversibility_ids:
        summary = _load_json(_lane_paths(manifest, lane_id)["summary"], f"lane summary {lane_id}")
        _require(summary.get("status") == "COMPLETED", f"Lane is incomplete: {lane_id}")
        _require(summary.get("manifest_sha256") == sha256_file(manifest_path), f"Lane manifest mismatch: {lane_id}")
        summaries[lane_id] = summary
    recomputed = {
        lane_id: _recompute_lane_energy(manifest_path, manifest, lane_id, body_names)
        for lane_id in scientific_ids
    }
    integrity_passed = all(
        item["telemetry_reproduction_max_abs"] == 0.0
        and item["compensated_decimal_max_abs"] <= 1.8189894035460633e-12
        for item in recomputed.values()
    )
    long_histories = {}
    for token, step in (("0p5d", 0.5), ("0p25d", 0.25)):
        times, values = _load_step3c_history(manifest, token)
        long_histories[token] = {
            "times": times,
            "values": values,
            "metrics": long_history_analysis(times, values, step),
        }
    full_signature = _systematic_signature(
        long_histories["0p5d"]["metrics"],
        long_histories["0p25d"]["metrics"],
        manifest["comparison_definitions"]["systematic_per_step_signature"],
    )
    existing_prefixes = {}
    controls_metrics = {}
    state_comparisons = {}
    for token in ("0p5d", "0p25d"):
        full = long_histories[token]
        mask100 = full["times"] <= 100000.0
        existing_prefixes[token] = {
            "times": full["times"][mask100],
            "values": full["values"][mask100],
            "statistics": _energy_statistics(full["times"][mask100], full["values"][mask100]),
        }
        old_times, old_states = _existing_states(manifest, token, body_names, 100000.0)
        for mode in ("current_sync", "min_sync"):
            lane_id = f"m0_diag_phys_{mode}_{token}_100k"
            data = recomputed[lane_id]
            controls_metrics[lane_id] = control_history_analysis(
                data["times"], data["decimal_values"], 0.5 if token == "0p5d" else 0.25
            )
            state_comparisons[f"{lane_id}_vs_full"] = _matched_state_comparison(
                data["times"], data["states"], old_times, old_states, body_names
            )
    reproduction_threshold = manifest["comparison_definitions"]["reproduction"]
    current_reproduction = {}
    min_current_reproduction = {}
    for token in ("0p5d", "0p25d"):
        current_id = f"m0_diag_phys_current_sync_{token}_100k"
        min_id = f"m0_diag_phys_min_sync_{token}_100k"
        current_reproduction[token] = _reproduction(
            existing_prefixes[token]["times"], existing_prefixes[token]["values"],
            recomputed[current_id]["times"], recomputed[current_id]["decimal_values"], reproduction_threshold
        )
        min_current_reproduction[token] = _reproduction(
            recomputed[current_id]["times"], recomputed[current_id]["decimal_values"],
            recomputed[min_id]["times"], recomputed[min_id]["decimal_values"], reproduction_threshold
        )
    ias_coarse = recomputed["m0_diag_phys_ias15_eps1e12_10k"]
    ias_fine = recomputed["m0_diag_phys_ias15_eps1e13_10k"]
    ias_state = _matched_state_comparison(
        ias_coarse["times"], ias_coarse["states"], ias_fine["times"], ias_fine["states"], body_names
    )
    ias_threshold = manifest["comparison_definitions"]["ias15_tolerance_convergence"]
    ias_energy_difference = np.max(np.abs(ias_coarse["decimal_values"] - ias_fine["decimal_values"]))
    ias_stats_coarse = ias_coarse["statistics"]["decimal"]
    ias_stats_fine = ias_fine["statistics"]["decimal"]
    ias_stat_agreement = all(
        abs(ias_stats_coarse[key] - ias_stats_fine[key])
        <= max(ias_threshold["energy_stat_relative_difference_max"] * max(abs(ias_stats_coarse[key]), abs(ias_stats_fine[key])), ias_threshold["energy_stat_absolute_floor"])
        for key in ("max_abs", "rms", "p99_abs")
    )
    ias_orbital_pass = all(
        item["semimajor_axis_relative_max"] <= ias_threshold["semimajor_axis_history_relative_max"]
        and item["eccentricity_abs_max"] <= ias_threshold["eccentricity_history_abs_max"]
        for item in ias_state["orbital_elements"].values()
    )
    ias_converged = (
        ias_state["global_scaled_rms"] <= ias_threshold["global_scaled_state_rms_max"]
        and all(value <= ias_threshold["per_body_scaled_state_rms_max"] for value in ias_state["per_body_scaled_rms"].values())
        and ias_orbital_pass
        and ias_energy_difference <= ias_threshold["energy_history_max_abs_difference"]
        and abs(ias_stats_coarse["fitted_change_per_myr"] - ias_stats_fine["fitted_change_per_myr"])
        <= ias_threshold["energy_fitted_change_per_myr_abs_difference"]
        and ias_stat_agreement
    )
    ias_force_threshold = manifest["comparison_definitions"]["ias15_force_problem"]
    ias_force_problem = all(
        item["statistics"]["decimal"]["fitted_change_per_myr"]
        >= ias_force_threshold["minimum_abs_fitted_change_per_myr"]
        and item["statistics"]["decimal"]["ci95_low_per_year"] > 0.0
        for item in (ias_coarse, ias_fine)
    )
    ias_reproduces = ias_force_problem
    ias_reference = ias_fine
    physical_accuracy = {}
    for token in ("0p5d", "0p25d"):
        for mode in ("current_sync", "min_sync"):
            lane_id = f"m0_diag_phys_{mode}_{token}_100k"
            data = recomputed[lane_id]
            physical_accuracy[lane_id] = _matched_state_comparison(
                data["times"][:101], data["states"][:101],
                ias_reference["times"], ias_reference["states"], body_names
            )
    material = {}
    current_material = {}
    for token in ("0p5d", "0p25d"):
        current_id = f"m0_diag_phys_current_sync_{token}_100k"
        min_id = f"m0_diag_phys_min_sync_{token}_100k"
        current_accuracy = physical_accuracy[current_id]
        min_accuracy = physical_accuracy[min_id]
        physical_ok = (
            min_accuracy["global_scaled_rms"] <= 1.25 * current_accuracy["global_scaled_rms"] + 1e-14
            and all(
                min_accuracy["per_body_scaled_rms"][body]
                <= 1.25 * current_accuracy["per_body_scaled_rms"][body] + 1e-14
                for body in INNER_BODIES
            )
        )
        material[token] = _material_reduction(
            controls_metrics[current_id], controls_metrics[min_id], physical_ok
        )
        current_material[token] = _material_reduction(
            existing_prefixes[token]["statistics"], controls_metrics[current_id], True
        )
    current_signature = _systematic_signature(
        controls_metrics["m0_diag_phys_current_sync_0p5d_100k"],
        controls_metrics["m0_diag_phys_current_sync_0p25d_100k"],
        manifest["comparison_definitions"]["systematic_per_step_signature"],
    )
    min_signature = _systematic_signature(
        controls_metrics["m0_diag_phys_min_sync_0p5d_100k"],
        controls_metrics["m0_diag_phys_min_sync_0p25d_100k"],
        manifest["comparison_definitions"]["systematic_per_step_signature"],
    )
    reversibility = {lane_id: summaries[lane_id] for lane_id in reversibility_ids}
    reversibility_diagnostics = _reversibility_timestep_diagnostics(reversibility)
    reversibility_valid = all(
        abs(item["return_time_seconds"]) <= 1e-6
        and item["callback_stats"]["callback_invocations"] == item["configuration"]["expected_callback_invocations"]
        and item["callback_stats"]["nonfinite_result_count"] == 0
        for item in reversibility.values()
    )
    bounded_threshold = manifest["comparison_definitions"]["bounded_history"]
    bounded_rules = all(
        item["metrics"]["range_elapsed_exponent"] <= bounded_threshold["range_elapsed_exponent_max"]
        and item["metrics"]["prefixes"]["1000"]["range"] / item["metrics"]["prefixes"]["100"]["range"]
        <= bounded_threshold["range_1000k_over_100k_max"]
        and abs(item["metrics"]["prefixes"]["1000"]["fitted_slope_per_year"])
        / max(abs(item["metrics"]["prefixes"]["100"]["fitted_slope_per_year"]), 1e-300)
        <= bounded_threshold["slope_1000k_over_100k_max"]
        and item["metrics"]["same_sign_block_count"] <= bounded_threshold["same_sign_100k_blocks_max"]
        for item in long_histories.values()
    )
    random_threshold = manifest["comparison_definitions"]["random_walk"]
    random_rules = (
        not full_signature["passed"]
        and all(
            random_threshold["elapsed_growth_exponent_min"]
            <= item["metrics"]["range_elapsed_exponent"]
            <= random_threshold["elapsed_growth_exponent_max"]
            for item in long_histories.values()
        )
        and not ias_reproduces
    )
    evidence = {
        "integrity_passed": integrity_passed,
        "ias15_tolerance_converged": ias_converged,
        "reversibility_valid": reversibility_valid,
        "ias15_force_problem": ias_force_problem,
        "ias15_reproduces": ias_reproduces,
        "current_reproduces_full_both": all(item["passed"] for item in current_reproduction.values()),
        "min_material_reduction_both": all(item["passed"] for item in material.values()),
        "current_material_reduction_both": all(item["passed"] for item in current_material.values()),
        "current_min_compatible_both": all(item["passed"] for item in min_current_reproduction.values()),
        "full_systematic_signature": full_signature["passed"],
        "current_systematic_signature": current_signature["passed"],
        "min_systematic_signature": min_signature["passed"],
        "any_sync_material_reduction": all(item["passed"] for item in material.values()),
        "random_walk_rules_passed": random_rules,
        "bounded_rules_passed": bounded_rules,
    }
    primary, step3_status = classify_mechanism(evidence)
    _require(primary in PRIMARY_MECHANISMS and step3_status in STEP3_STATUSES, "Invalid classification.")
    candidate = {
        "BOUNDED_ENERGY_OSCILLATION": {"assessment": "passes" if bounded_rules else "fails", "for": "periodic residual peaks are present", "against": "range expansion and block/prefix gates"},
        "RANDOM_WALK_ROUNDOFF": {"assessment": "passes" if random_rules else "fails", "for": "increment distributions are near centered", "against": "systematic q_h signature and sign-consistent blocks"},
        "SYSTEMATIC_WHFAST_STEP_BIAS": {"assessment": "passes" if primary == "SYSTEMATIC_WHFAST_STEP_BIAS" else "fails or is superseded", "for": f"full signature={full_signature['passed']}, current signature={current_signature['passed']}", "against": f"sync material reduction={evidence['any_sync_material_reduction']}"},
        "SYNCHRONIZATION_RECALCULATION_BIAS": {"assessment": "passes" if primary == "SYNCHRONIZATION_RECALCULATION_BIAS" else "fails", "for": f"current reproduces={evidence['current_reproduces_full_both']}; min reduction={evidence['min_material_reduction_both']}", "against": f"IAS15 reproduces={ias_reproduces}"},
        "VARIATION_MEGNO_COUPLING": {"assessment": "passes" if primary == "VARIATION_MEGNO_COUPLING" else "fails", "for": f"current reduction={evidence['current_material_reduction_both']}", "against": f"current reproduces full={evidence['current_reproduces_full_both']}"},
        "CORRECTED_INVARIANT_OR_FORCE_PROBLEM": {"assessment": "passes" if ias_force_problem else "fails", "for": f"IAS15 converged={ias_converged}", "against": f"significant IAS15 drift={ias_force_problem}"},
        "MIXED_OR_INCONCLUSIVE": {"assessment": "passes" if primary == "MIXED_OR_INCONCLUSIVE" else "not needed", "for": "reserved for unresolved mixtures", "against": f"selected mechanism={primary}"},
    }
    figures = _make_figures(manifest, long_histories, recomputed, reversibility)
    new_lanes = {}
    for lane_id in scientific_ids:
        new_lanes[lane_id] = {
            "runtime_seconds": summaries[lane_id]["runtime_seconds"],
            "throughput_years_per_wall_second": summaries[lane_id]["throughput_years_per_wall_second"],
            "callback_stats": summaries[lane_id]["callback_stats"],
            "artifact_inventory": summaries[lane_id]["artifact_inventory"],
            "energy": {
                "statistics": recomputed[lane_id]["statistics"],
                "telemetry_reproduction_max_abs": recomputed[lane_id]["telemetry_reproduction_max_abs"],
                "compensated_decimal_max_abs": recomputed[lane_id]["compensated_decimal_max_abs"],
                "timeseries": recomputed[lane_id]["timeseries"],
            },
        }
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "primary_mechanism": primary,
        "step3_diagnosis_status": step3_status,
        "historical_statuses_unchanged": manifest["provenance"]["historical_statuses_immutable"],
        "audit": audit_payload,
        "long_history": {token: item["metrics"] for token, item in long_histories.items()},
        "systematic_signatures": {"full": full_signature, "current_sync": current_signature, "min_sync": min_signature},
        "new_lanes": new_lanes,
        "reproduction": {"current_vs_full": current_reproduction, "min_vs_current": min_current_reproduction},
        "material_reduction": {"min_vs_current": material, "current_vs_full": current_material},
        "state_comparisons": state_comparisons,
        "physical_accuracy_against_ias15": physical_accuracy,
        "ias15": {
            "tolerance_converged": ias_converged,
            "thresholds": ias_threshold,
            "state_agreement": ias_state,
            "energy_history_max_abs_difference": float(ias_energy_difference),
            "energy_statistics_agree": ias_stat_agreement,
            "orbital_elements_agree": ias_orbital_pass,
            "force_problem_rule_passed": ias_force_problem,
        },
        "reversibility": reversibility,
        "reversibility_diagnostics": reversibility_diagnostics,
        "classification_evidence": evidence,
        "candidate_mechanisms": candidate,
        "production_configuration_must_change": step3_status == "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED",
        "final_0p125_lane_justified": step3_status == "STEP3_NUMERICAL_FLOOR_CHARACTERIZED",
        "smallest_next_action": (
            "Revalidate the supported WHFast safe_mode=0, keep_unsynchronized=1 configuration over bounded force, archive/restart, and convergence checks before any new 1 Myr lane."
            if step3_status == "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED"
            else "Do not modify validated physics; isolate the corrected invariant/force inconsistency."
            if step3_status == "STEP3_FORCE_INVARIANT_PROBLEM"
            else "Use the separately preregistered bounded next step described by the status policy."
            if step3_status == "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"
            else "Preregister one bounded 10 kyr IAS15 epsilon=1e-14 tolerance lane, after a 100-year runtime benchmark, and compare epsilon=1e-13 versus 1e-14 against the unchanged convergence gates before another WHFast trajectory."
        ),
        "figures": figures,
        "no_stage4_command": True,
    }
    atomic_write_json(Path(manifest["paths"]["report_json"]), payload)
    _atomic_text(Path(manifest["paths"]["report_markdown"]), _markdown_report(payload))
    print(f"[m0-roundoff] primary_mechanism={primary}")
    print(f"[m0-roundoff] step3_status={step3_status}")



def _make_blocked_figures(
    manifest: dict[str, Any],
    long_histories: dict[str, dict[str, Any]],
    validation: dict[str, Any],
) -> list[dict[str, Any]]:
    directory = Path(manifest["paths"]["figure_directory"])
    _require(not directory.exists(), f"Figure directory already exists: {directory}")
    directory.mkdir(parents=True)
    figures = []
    colors = {"0p5d": "#1f77b4", "0p25d": "#d62728"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for token, item in long_histories.items():
        times = item["times"]
        values = item["values"]
        metrics = item["metrics"]
        ax.plot(times / 1000.0, values, color=colors[token], lw=0.9, label=token)
        ax.plot(
            times / 1000.0,
            metrics["fitted_intercept"] + metrics["fitted_slope_per_year"] * times,
            color=colors[token],
            ls="--",
            lw=1.2,
        )
    ax.set(
        xlabel="Time (kyr)",
        ylabel="Corrected-energy relative change",
        title="Existing 1 Myr histories and global fits",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    path = directory / "long_history_energy_and_fits.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(10)
    for offset, token, label in ((-0.18, "0p5d", "0.5 day"), (0.18, "0p25d", "0.25 day")):
        values = [
            item["fitted_slope_per_year"] * 1e16
            for item in long_histories[token]["metrics"]["blocks"]
        ]
        ax.bar(x + offset, values, 0.36, label=label, color=colors[token])
    ax.set(
        xlabel="100 kyr block",
        ylabel="Slope (1e-16 / year)",
        title="Blockwise corrected-energy slopes",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = directory / "blockwise_slopes.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})

    lane_ids = sorted(validation["results"])
    values = [
        validation["results"][lane]["metrics"]["global_scaled_rms"] for lane in lane_ids
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(
        np.arange(len(lane_ids)),
        values,
        color=["#1f77b4", "#2ca02c", "#d62728", "#9467bd"],
    )
    ax.axhline(
        manifest["method_validation"]["reversibility_two_body"][
            "required_scaled_state_rms_max"
        ],
        color="#222222",
        ls="--",
        lw=1.0,
        label="absolute gate",
    )
    labels = [
        lane.replace("two_body_", "").replace("_sync_", "\nsync ")
        for lane in lane_ids
    ]
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(lane_ids)), labels)
    ax.set(
        ylabel="Return scaled-state RMS",
        title="Corrected two-body reversibility method validation",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = directory / "reversibility_method_validation_errors.png"
    _figure_save(fig, path)
    figures.append({"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return figures


def _blocked_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# M0 Integrator/Roundoff Diagnosis: BLOCKED",
        "",
        "Primary mechanism: **BLOCKED**",
        "",
        "Step 3 diagnosis status: **BLOCKED**",
        "",
        "## Blocking Gate",
        "",
        payload["blocking_reason"],
        "",
        "All four corrected-protocol two-body returns pass the absolute 1e-8 state-error gate, "
        "exact-time gate, callback-total gate, and nonfinite gate. The frozen timestep-ratio gate still fails:",
        "",
        "| Mode | 0.5-day RMS | 0.25-day RMS | Fine/coarse | Limit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in payload["reversibility_method_validation"]["ratios"].items():
        lines.append(
            f"| {mode} | {item['coarse_scaled_rms']:.12g} | {item['fine_scaled_rms']:.12g} | "
            f"{item['fine_over_coarse']:.9g} | {item['limit']:.9g} |"
        )
    lines.extend(
        [
            "",
            "The first failed attempt is preserved separately. It demonstrated that flipping dt while "
            "retaining the unsynchronized internal Jacobi state starts the reverse leg from the wrong leapfrog phase. "
            "The corrected attempt uses the installed REBOUND 4.6.0 source-defined endpoint synchronization transition.",
            "",
            "## Existing Histories",
            "",
            "| Timestep | Fitted change / Myr | R2 | q per step | Same-sign blocks | Range exponent |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for token, item in payload["long_history"].items():
        lines.append(
            f"| {token} | {item['fitted_change_per_myr']:.12g} | {item['r_squared']:.9g} | "
            f"{item['energy_change_per_step']:.12g} | {item['same_sign_block_count']} | "
            f"{item['range_elapsed_exponent']:.6g} |"
        )
    lines.extend(["", "## Candidate Mechanisms", ""])
    for name, item in payload["candidate_mechanisms"].items():
        lines.append(
            f"- **{name}**: {item['assessment']}. For: {item['for']}. Against: {item['against']}."
        )
    lines.extend(
        [
            "",
            "## Configuration Matrix",
            "",
            "No decisive M0 lane was launched. These frozen configurations remain preregistered but unobserved:",
            "",
            "| Lane | Integrator | Purpose | Step / epsilon | Duration |",
            "| --- | --- | --- | --- | ---: |",
        ]
    )
    for item in payload["preregistered_unlaunched_lanes"]:
        lines.append(
            f"| {item['lane_id']} | {item['integrator']} | {item['purpose']} | "
            f"{item['step_or_epsilon']} | {item['duration_years']:.9g} yr |"
        )
    lines.extend(
        [
            "",
            "## Runtime And Scope",
            "",
            "No decisive scientific lane ran, so scientific runtime and throughput are not applicable. "
            "The two bounded four-case method-validation commands each completed in under four wall-clock seconds; "
            "the driver did not instrument per-case throughput.",
            "",
            "The requested full/current/min-sync, IAS15, and 10 kyr M0 reversibility figures are unavailable "
            "because producing them would require crossing the failed frozen gate.",
            "",
            "## Decision",
            "",
            "- Whether the production configuration must change is undetermined.",
            "- A final 0.125-day lane is not justified.",
            "- Historical Step 3, Step 3b, and Step 3c statuses remain unchanged.",
            "- No Stage 4 command is provided.",
            "",
            "## Smallest Next Action",
            "",
            payload["smallest_next_action"],
            "",
        ]
    )
    return "\n".join(lines)



def report_blocked(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "manifest 13")
    audit_payload = audit(manifest_path)
    output_root = Path(manifest["paths"]["output_root"])
    validation_path = output_root / "reversibility_method_validation/summary.json"
    validation = _load_json(validation_path, "reversibility method validation")
    _require(
        validation.get("passed") is False,
        "Blocked report requires a failed method-validation gate.",
    )
    _require(
        validation.get("manifest_sha256") == sha256_file(manifest_path),
        "Validation manifest mismatch.",
    )

    long_histories = {}
    for token, step_days in (("0p5d", 0.5), ("0p25d", 0.25)):
        times, values = _load_step3c_history(manifest, token)
        long_histories[token] = {
            "times": times,
            "values": values,
            "metrics": long_history_analysis(times, values, step_days),
        }
    signature = _systematic_signature(
        long_histories["0p5d"]["metrics"],
        long_histories["0p25d"]["metrics"],
        manifest["comparison_definitions"]["systematic_per_step_signature"],
    )
    ratio_limit = manifest["method_validation"]["reversibility_two_body"][
        "required_0p25_error_over_0p5_error_max"
    ]
    ratios = {}
    for mode in ("current_sync", "min_sync"):
        coarse = validation["results"][f"two_body_{mode}_0p5d"]["metrics"][
            "global_scaled_rms"
        ]
        fine = validation["results"][f"two_body_{mode}_0p25d"]["metrics"][
            "global_scaled_rms"
        ]
        ratios[mode] = {
            "coarse_scaled_rms": coarse,
            "fine_scaled_rms": fine,
            "fine_over_coarse": fine / coarse,
            "limit": ratio_limit,
            "passed": fine <= ratio_limit * coarse + 1e-30,
        }
    figures = _make_blocked_figures(manifest, long_histories, validation)
    attempts = []
    for relative in (
        "reversibility_method_validation_failed_attempt_1/summary.json",
        "reversibility_method_validation/summary.json",
    ):
        path = output_root / relative
        attempts.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unlaunched = []
    for lane in manifest["lanes"]:
        config = {**manifest["common_configuration"], **lane["configuration"]}
        unlaunched.append(
            {
                "lane_id": lane["id"],
                "configuration_fingerprint": lane["configuration_fingerprint"],
                "kind": config["kind"],
                "integrator": config["integrator"],
                "purpose": config["purpose"],
                "step_or_epsilon": config.get("step_days", config.get("epsilon")),
                "duration_years": config.get(
                    "duration_years", config.get("forward_years")
                ),
                "safe_mode": config.get("safe_mode"),
                "keep_unsynchronized": config.get("keep_unsynchronized"),
                "variations": config["variations"],
                "megno": config["megno"],
            }
        )
    candidate = {
        "BOUNDED_ENERGY_OSCILLATION": {
            "assessment": "not established",
            "for": "oscillatory residual power exists",
            "against": "range growth and ten same-sign blocks fail the bounded-history interpretation",
        },
        "RANDOM_WALK_ROUNDOFF": {
            "assessment": "not established",
            "for": "the 0.5-day range exponent overlaps a square-root-like regime",
            "against": "the 0.25-day exponent and preregistered per-step signature are inconsistent with a pure random walk",
        },
        "SYSTEMATIC_WHFAST_STEP_BIAS": {
            "assessment": "plausible but unclassified",
            "for": f"existing full-history systematic signature passed={signature['passed']}",
            "against": "physical-only synchronization controls and IAS15 were not authorized past the failed gate",
        },
        "SYNCHRONIZATION_RECALCULATION_BIAS": {
            "assessment": "unclassified",
            "for": "the failed first method attempt confirms reversal is sensitive to internal leapfrog phase",
            "against": "no 100 kyr current/min synchronization control ran",
        },
        "VARIATION_MEGNO_COUPLING": {
            "assessment": "unclassified",
            "for": "the existing histories include variations and MEGNO",
            "against": "no matched physical-only control ran",
        },
        "CORRECTED_INVARIANT_OR_FORCE_PROBLEM": {
            "assessment": "unclassified",
            "for": "the existing corrected-energy drift is independently reconstructed",
            "against": "no tolerance-converged IAS15 reference ran",
        },
        "MIXED_OR_INCONCLUSIVE": {
            "assessment": "superseded by BLOCKED",
            "for": "several mechanisms remain unresolved",
            "against": "the failed integrity gate requires BLOCKED rather than a causal conclusion",
        },
    }
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "primary_mechanism": "BLOCKED",
        "step3_diagnosis_status": "BLOCKED",
        "blocking_reason": (
            "The corrected two-body reversibility method passes every absolute gate, but the "
            f"minimal-sync 0.25/0.5-day return-error ratio is {ratios['min_sync']['fine_over_coarse']:.9g}, "
            f"above the frozen maximum {ratio_limit:.9g}."
        ),
        "audit": audit_payload,
        "historical_statuses_unchanged": manifest["provenance"][
            "historical_statuses_immutable"
        ],
        "long_history": {
            token: item["metrics"] for token, item in long_histories.items()
        },
        "existing_systematic_signature": signature,
        "reversibility_method_validation": {
            "passed": validation["passed"],
            "absolute_case_results": validation["results"],
            "ratios": ratios,
            "attempt_artifacts": attempts,
        },
        "candidate_mechanisms": candidate,
        "preregistered_unlaunched_lanes": unlaunched,
        "decisive_scientific_lanes_launched": 0,
        "scientific_runtime_and_throughput": None,
        "production_configuration_must_change": None,
        "final_0p125_lane_justified": False,
        "smallest_next_action": (
            "In a separately preregistered follow-up, rerun only the four 365-day two-body method cases "
            "with a roundoff-aware reversibility scaling rule that retains the frozen absolute error, "
            "exact-time, callback, and nonfinite gates; do not launch an M0 trajectory until that gate passes."
        ),
        "figures": figures,
        "unavailable_figures": [
            "full/current_sync/min_sync comparison",
            "IAS15 comparison",
            "10 kyr M0 reversibility comparison",
        ],
        "no_stage4_command": True,
    }
    atomic_write_json(Path(manifest["paths"]["report_json"]), payload)
    _atomic_text(
        Path(manifest["paths"]["report_markdown"]), _blocked_markdown(payload)
    )
    print("[m0-roundoff] primary_mechanism=BLOCKED")
    print("[m0-roundoff] step3_status=BLOCKED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded M0 integrator/roundoff diagnosis.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("validate-reversibility")
    subparsers.add_parser("benchmark-ias15")
    lane = subparsers.add_parser("run-lane")
    lane.add_argument("--lane-id", required=True)
    reverse = subparsers.add_parser("run-reversibility")
    reverse.add_argument("--lane-id", required=True)
    subparsers.add_parser("analyze")
    subparsers.add_parser("report-blocked")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            print(json.dumps(audit(args.manifest), indent=2, sort_keys=True))
        elif args.command == "validate-reversibility":
            validate_reversibility(args.manifest)
        elif args.command == "benchmark-ias15":
            benchmark_ias15(args.manifest)
        elif args.command == "run-lane":
            run_lane(args.manifest, args.lane_id)
        elif args.command == "run-reversibility":
            run_reversibility(args.manifest, args.lane_id)
        elif args.command == "analyze":
            analyze(args.manifest)
        elif args.command == "report-blocked":
            report_blocked(args.manifest)
    except DiagnosisError as exc:
        raise SystemExit(f"m0 roundoff diagnosis error: {exc}") from exc


if __name__ == "__main__":
    main()
