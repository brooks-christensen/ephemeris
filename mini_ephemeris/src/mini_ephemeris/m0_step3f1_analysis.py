from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .m0_energy_precision_diagnosis import float64_energy
from .m0_integrator_roundoff_diagnosis import _read_physical_groups
from .m0_step3e1_offline_diagnosis import (
    _element_pair,
    _element_series,
    _reconstruct,
    _scaled_defect,
    _wrap,
)
from .m0_step3e_convergence import _naff_lite
from .m0_step3f1_contract import (
    AU_M,
    BODY_NAMES,
    JULIAN_YEAR_S,
    PROGRESS_FIELDS,
    STATE_FIELDS,
    VELOCITY_SCALE,
    artifact_identity,
    lane_paths,
    load_json,
    require,
    sha256_file,
    validate_manifest,
)
from .m0_step3f1_runner import audit
from .m0_timestep_convergence import RunData, _compute_elements, _pair_physical, _pair_tangent
from .nbody import G_SI, NBodyState
from .orbital_elements import ARCSEC_PER_RAD
from .rebound_gr_tangent_backend_cli import atomic_write_json
from .stability_diagnostics import total_angular_momentum_vector


METRIC_FIELDS = (
    "comparison",
    "category",
    "metric",
    "body",
    "value",
    "units",
    "threshold",
    "passed",
    "worst_epoch_years",
)


def _finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite result at {path}.")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing artifact: {path}")
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or ()), rows


def _progress_arrays(rows: Sequence[dict[str, str]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for field in rows[0]:
        if all(row[field] != "" for row in rows):
            try:
                output[field] = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
            except ValueError:
                pass
    return output


def _new_lane(manifest: dict[str, Any], lane_key: str) -> RunData:
    lane = manifest["lane_contracts"][lane_key]
    common = manifest["common_physical_contract"]
    paths = lane_paths(manifest, lane_key)
    for name in ("progress", "state", "archive", "status", "summary", "events", "restart_check"):
        require(paths[name].is_file(), f"Missing Lane {lane_key} artifact: {name}")
    summary = load_json(paths["summary"], f"Lane {lane_key} summary")
    status = load_json(paths["status"], f"Lane {lane_key} status")
    restart = load_json(paths["restart_check"], f"Lane {lane_key} restart check")
    recovered_mismatch = (
        lane_key == "P"
        and summary["status"] == "RECOVERED_COMPLETE_WITH_CALLBACK_ACCOUNTING_MISMATCH"
        and status["state"] == "RECOVERED_COMPLETE_WITH_CALLBACK_ACCOUNTING_MISMATCH"
    )
    normal_complete = summary["status"] == "COMPLETED" and status["state"] == "COMPLETED"
    require(normal_complete or recovered_mismatch, f"Lane {lane_key} is incomplete.")
    require(restart["status"] == "PASS" or (recovered_mismatch and restart["status"] == "FAIL" and restart["callback_accounting_passed"] is False), f"Lane {lane_key} restart check has an unrecognized failure.")
    require(summary["configuration_fingerprint"] == lane["configuration_fingerprint"], f"Lane {lane_key} summary fingerprint changed.")
    for name, item in summary["artifact_inventory"].items():
        path = Path(item["path"])
        require(path.stat().st_size == item["size_bytes"], f"Lane {lane_key} {name} size changed.")
        require(sha256_file(path) == item["sha256"], f"Lane {lane_key} {name} hash changed.")

    progress_fields, progress_rows = _read_rows(paths["progress"])
    state_fields, state_rows = _read_rows(paths["state"])
    require(progress_fields == PROGRESS_FIELDS, f"Lane {lane_key} progress schema changed.")
    require(state_fields == STATE_FIELDS, f"Lane {lane_key} state schema changed.")
    require(len(progress_rows) == common["scientific_samples"], f"Lane {lane_key} progress count changed.")
    require(len(state_rows) == common["state_rows"], f"Lane {lane_key} state count changed.")
    times = np.arange(101, dtype=np.float64) * 100.0
    masses = np.empty(10, dtype=np.float64)
    positions = np.empty((101, 10, 3), dtype=np.float64)
    velocities = np.empty((101, 10, 3), dtype=np.float64)
    variation_positions = np.zeros((101, 10, 3), dtype=np.float64)
    variation_velocities = np.zeros((101, 10, 3), dtype=np.float64)
    identity = artifact_identity(manifest, lane_key)
    for linear, row in enumerate(state_rows):
        sample, body = divmod(linear, 10)
        require(int(row["sample_index"]) == sample and int(row["body_index"]) == body, f"Lane {lane_key} state ordering changed.")
        require(row["body_name"] == BODY_NAMES[body], f"Lane {lane_key} body order changed.")
        require(int(row["step_count"]) == sample * 146100, f"Lane {lane_key} state step changed.")
        require(float(row["time_years"]) == times[sample], f"Lane {lane_key} state time changed.")
        require(float(row["time_seconds"]) == sample * 146100 * 21600.0, f"Lane {lane_key} state seconds changed.")
        require(row["schema_version"] == "1" and row["configuration_fingerprint"] == lane["configuration_fingerprint"], f"Lane {lane_key} state identity changed.")
        require(row["lane_id"] == lane["id"] and row["artifact_identity"] == identity, f"Lane {lane_key} artifact identity changed.")
        mass = float(row["mass_kg"])
        if sample == 0:
            masses[body] = mass
        else:
            require(mass == masses[body], f"Lane {lane_key} mass changed.")
        values = np.asarray([float(row[name]) for name in ("x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")])
        require(np.all(np.isfinite(values)), f"Lane {lane_key} physical state is nonfinite.")
        positions[sample, body], velocities[sample, body] = values[:3], values[3:]
        variation_names = (
            "variation_x_m", "variation_y_m", "variation_z_m",
            "variation_vx_m_per_s", "variation_vy_m_per_s", "variation_vz_m_per_s",
        )
        if lane_key == "T":
            require(row["variation_config_index"] == "0", "Lane T variation identity changed.")
            variation = np.asarray([float(row[name]) for name in variation_names])
            require(np.all(np.isfinite(variation)), "Lane T variation is nonfinite.")
            variation_positions[sample, body], variation_velocities[sample, body] = variation[:3], variation[3:]
        else:
            require(row["variation_config_index"] == "" and all(row[name] == "" for name in variation_names), "Lane P contains variation telemetry.")

    for sample, row in enumerate(progress_rows):
        require(int(row["sample_index"]) == sample and int(row["step_count"]) == sample * 146100, f"Lane {lane_key} progress ordering changed.")
        require(float(row["time_years"]) == times[sample], f"Lane {lane_key} progress time changed.")
        require(row["configuration_fingerprint"] == lane["configuration_fingerprint"] and row["artifact_identity"] == identity, f"Lane {lane_key} progress identity changed.")
        require(row["kernel"] == lane["kernel"] and int(row["corrector"]) == lane["corrector"], f"Lane {lane_key} WHFast settings changed.")
        require(int(row["safe_mode"]) == 0 and int(row["keep_unsynchronized"]) == 1, f"Lane {lane_key} synchronization settings changed.")
        expected_sync = 1 if sample == 0 else 0
        require(int(row["live_is_synchronized_before_sample"]) == expected_sync, f"Lane {lane_key} live-map state changed.")
        require(int(row["live_is_synchronized_after_sample"]) == expected_sync, f"Lane {lane_key} sampling mutated the live map.")
        require(int(row["nonfinite_result_count"]) == 0, f"Lane {lane_key} callback returned nonfinite data.")
    callback_accounting_passed = int(progress_rows[-1]["callback_invocations"]) == lane["expected_callback_invocations"]
    require(callback_accounting_passed or recovered_mismatch, f"Lane {lane_key} callback mismatch lacks recovery provenance.")

    import rebound

    archive_hash = sha256_file(paths["archive"])
    archive = rebound.Simulationarchive(str(paths["archive"]))
    require(len(archive) == 11, f"Lane {lane_key} archive count changed.")
    archive_times = [float(archive[index].t) / JULIAN_YEAR_S for index in range(11)]
    require(archive_times == [1000.0 * index for index in range(11)], f"Lane {lane_key} archive times changed.")
    for index in range(11):
        sim = archive[index]
        require(str(sim.integrator) == "whfast" and str(sim.ri_whfast.kernel) == lane["kernel"], f"Lane {lane_key} archive settings changed.")
        require(int(sim.ri_whfast.safe_mode) == 0 and int(sim.ri_whfast.keep_unsynchronized) == 1, f"Lane {lane_key} archive synchronization fields changed.")
        expected_var = 0 if lane_key == "P" else 10
        require(int(sim.N_real) == 10 and int(sim.N_var) == expected_var, f"Lane {lane_key} archive particle layout changed.")
    require(sha256_file(paths["archive"]) == archive_hash, f"Lane {lane_key} archive was mutated by inspection.")
    integrity = {
        "passed": callback_accounting_passed and restart["status"] == "PASS",
        "trajectory_complete": True,
        "preregistered_callback_accounting_passed": callback_accounting_passed,
        "scientific_samples": len(progress_rows),
        "state_rows": len(state_rows),
        "archive_snapshots": len(archive),
        "steps": int(progress_rows[-1]["steps_done"]),
        "callback_invocations": int(progress_rows[-1]["callback_invocations"]),
        "nonfinite_result_count": 0,
        "live_map_preserved": True,
        "restart": restart,
        "artifacts": {name: _artifact(paths[name]) for name in ("progress", "state", "archive", "status", "summary", "events", "restart_check")},
    }
    return RunData(
        run_id=lane["id"], step_days=0.25, body_names=BODY_NAMES, times=times,
        masses=masses, positions=positions, velocities=velocities,
        variation_positions=variation_positions, variation_velocities=variation_velocities,
        progress=_progress_arrays(progress_rows), summary=summary, integrity=integrity,
        inventory=list(integrity["artifacts"].values()),
    )


def _run_data(
    run_id: str,
    times: np.ndarray,
    states: Sequence[NBodyState],
    progress_rows: Sequence[dict[str, str]],
    variation_positions: np.ndarray | None = None,
    variation_velocities: np.ndarray | None = None,
) -> RunData:
    masses = states[0].masses.copy()
    positions = np.stack([state.positions for state in states])
    velocities = np.stack([state.velocities for state in states])
    shape = positions.shape
    return RunData(
        run_id=run_id, step_days=0.0, body_names=BODY_NAMES, times=times,
        masses=masses, positions=positions, velocities=velocities,
        variation_positions=np.zeros(shape) if variation_positions is None else variation_positions,
        variation_velocities=np.zeros(shape) if variation_velocities is None else variation_velocities,
        progress=_progress_arrays(progress_rows), summary={}, integrity={"passed": True}, inventory=[],
    )


def _ias15(manifest: dict[str, Any]) -> RunData:
    manifest18 = load_json(Path(manifest["source_artifacts"]["manifest_18"][0]), "Manifest 18")
    inventory = manifest18["ias15_evidence_artifacts"]["m0_diag_phys_ias15_default_10k"]
    state_path = Path(inventory["state"][0])
    progress_path = Path(inventory["progress"][0])
    times, states, _ = _read_physical_groups(state_path, BODY_NAMES)
    _, progress = _read_rows(progress_path)
    require(np.array_equal(times, np.arange(101) * 100.0), "IAS15 timestamps changed.")
    require(len(states) == 101 and len(progress) == 101, "IAS15 sample count changed.")
    return _run_data("m0_diag_phys_ias15_default_10k", times, states, progress)


def _historical_tangent(manifest: dict[str, Any]) -> RunData:
    manifest18 = load_json(Path(manifest["source_artifacts"]["manifest_18"][0]), "Manifest 18")
    lane = next(item for item in manifest18["stored_lanes"] if item["run_id"] == "m0_conv_0p25d_1myr_s12345")
    root = Path(lane["output_dir"])
    state_path = (root / lane["artifact_inventory"]["state"][0]).resolve()
    progress_path = (root / lane["artifact_inventory"]["progress"][0]).resolve()
    _, all_progress = _read_rows(progress_path)
    progress = all_progress[:101]
    _, all_state = _read_rows(state_path)
    rows = all_state[:1010]
    states: list[NBodyState] = []
    vp = np.empty((101, 10, 3), dtype=np.float64)
    vv = np.empty((101, 10, 3), dtype=np.float64)
    for sample in range(101):
        group = rows[sample * 10 : (sample + 1) * 10]
        require([row["body_name"] for row in group] == list(BODY_NAMES), "Historical tangent body order changed.")
        states.append(NBodyState(
            masses=np.asarray([float(row["mass_kg"]) for row in group]),
            positions=np.asarray([[float(row[name]) for name in ("x_m", "y_m", "z_m")] for row in group]),
            velocities=np.asarray([[float(row[name]) for name in ("vx_m_per_s", "vy_m_per_s", "vz_m_per_s")] for row in group]),
        ))
        vp[sample] = np.asarray([[float(row[name]) for name in ("variation_x_m", "variation_y_m", "variation_z_m")] for row in group])
        vv[sample] = np.asarray([[float(row[name]) for name in ("variation_vx_m_per_s", "variation_vy_m_per_s", "variation_vz_m_per_s")] for row in group])
    times = np.arange(101, dtype=np.float64) * 100.0
    return _run_data(lane["run_id"] + "_first_10k", times, states, progress, vp, vv)


def _raw_detail(left: RunData, right: RunData) -> dict[str, Any]:
    scaled = _scaled_defect(left.positions, left.velocities, right.positions, right.velocities)
    squared = np.sum(scaled**2, axis=2)
    total = float(np.sum(squared))
    position = np.linalg.norm(left.positions - right.positions, axis=2)
    velocity = np.linalg.norm(left.velocities - right.velocities, axis=2)
    body = {}
    for index, name in enumerate(BODY_NAMES):
        p_worst = np.unravel_index(int(np.argmax(np.abs(left.positions[:, index] - right.positions[:, index]))), (101, 3))
        v_worst = np.unravel_index(int(np.argmax(np.abs(left.velocities[:, index] - right.velocities[:, index]))), (101, 3))
        body[name] = {
            "scaled_rms": float(np.sqrt(np.mean(squared[:, index]) / 6.0)),
            "squared_error_contribution": float(np.sum(squared[:, index]) / max(total, 1.0e-300)),
            "position_vector_max_m": float(np.max(position[:, index])),
            "velocity_vector_max_m_per_s": float(np.max(velocity[:, index])),
            "worst_position_component": ("x", "y", "z")[p_worst[1]],
            "worst_position_epoch_years": float(left.times[p_worst[0]]),
            "worst_velocity_component": ("vx", "vy", "vz")[v_worst[1]],
            "worst_velocity_epoch_years": float(left.times[v_worst[0]]),
        }
    sample_rms = np.sqrt(np.mean(scaled**2, axis=(1, 2)))
    return {
        "global_scaled_rms": float(np.sqrt(np.mean(scaled**2))),
        "sample_scaled_rms": sample_rms,
        "per_body": body,
    }


def _rtn(left: RunData, right: RunData) -> dict[str, Any]:
    output = {}
    for body, name in enumerate(BODY_NAMES[1:], start=1):
        r = right.positions[:, body] - right.positions[:, 0]
        v = right.velocities[:, body] - right.velocities[:, 0]
        radial = r / np.linalg.norm(r, axis=1)[:, None]
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        transverse = np.cross(normal, radial)
        basis = np.stack((radial, transverse, normal), axis=2)
        dr = (left.positions[:, body] - left.positions[:, 0]) - r
        dv = (left.velocities[:, body] - left.velocities[:, 0]) - v
        pr = np.einsum("ni,nij->nj", dr, basis)
        pv = np.einsum("ni,nij->nj", dv, basis)
        output[name] = {
            "position_rms_m": {axis: float(np.sqrt(np.mean(pr[:, i] ** 2))) for i, axis in enumerate(("R", "T", "N"))},
            "velocity_rms_m_per_s": {axis: float(np.sqrt(np.mean(pv[:, i] ** 2))) for i, axis in enumerate(("R", "T", "N"))},
            "transverse_position_squared_fraction": float(np.sum(pr[:, 1] ** 2) / max(float(np.sum(pr**2)), 1.0e-300)),
        }
    return output
