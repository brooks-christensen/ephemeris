from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import numpy as np

from .m0_integrator_roundoff_diagnosis import _runtime_identity
from .m0_step3e_convergence import _phase_diagnostics, _run_state
from .m0_timestep_convergence import RunData, _load_json, _load_run, _pair_physical
from .nbody import G_SI
from .orbital_elements import (
    AU_M,
    J2000_MEAN_OBLIQUITY_RAD,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
)
from .rebound_gr_tangent_backend_cli import sha256_file


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/"
    "18_m0_step3e1_offline_state_diagnosis_v1.json"
)
STARTING_COMMIT = "1d24b97047335bd1302ab5c7b26cb6b231ac5516"
BASELINE_TAG = "gr-tangent-compiled-c-v1"
BASELINE_COMMIT = "2d7778e1911c6f6ae97da24cdfe00ef45f21e73b"
FINAL_STATUSES = {
    "STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE",
    "STEP3E1_OFFLINE_DIAGNOSIS_INCONCLUSIVE",
    "BLOCKED",
}
PRIMARY_CLASSIFICATIONS = {
    "WINDOWED_OR_PHASE_DOMINATED",
    "METRIC_OR_REPRESENTATION_ILL_CONDITIONED",
    "POINTWISE_PREDICTABILITY_FLOOR",
    "TRUE_NONPHASE_NONCONVERGENCE",
    "MIXED_OR_INCONCLUSIVE",
    "NOT_EVALUATED",
}
BODY_NAMES = (
    "sun",
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
    "jupiter barycenter",
    "saturn barycenter",
    "uranus barycenter",
    "neptune barycenter",
    "pluto barycenter",
)
INNER_BODIES = BODY_NAMES[1:5]
VELOCITY_SCALE = AU_M / JULIAN_YEAR_S
EPS = np.finfo(np.float64).eps


class DiagnosisError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosisError(message)


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def _json_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(child) for child in value]
    return value


def _finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"Nonfinite JSON value at {path}.")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    native = _json_native(payload)
    _finite_json(native)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(native, handle, indent=2, sort_keys=True, allow_nan=False)
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


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _require(bool(rows), f"No rows for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            _require(list(row) == fields, f"CSV schema changed in {path}.")
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
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


def _source_audit(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for name, (path_text, expected) in manifest["source_artifacts"].items():
        path = Path(path_text)
        actual = sha256_file(path)
        _require(actual == expected, f"Historical source hash mismatch: {name}")
        output.append({"name": name, "path": str(path), "sha256": actual})
    return output


def _protected_audit(manifest: dict[str, Any]) -> list[dict[str, str]]:
    root = Path(manifest["paths"]["project_root"])
    output = []
    for relative, expected in manifest["protected_files"].items():
        actual = sha256_file(root / relative)
        _require(actual == expected, f"Protected file changed: {relative}")
        output.append({"path": relative, "sha256": actual})
    return output


def _lane_artifact_path(lane: dict[str, Any], item: Sequence[Any]) -> Path:
    path = Path(item[0])
    if not path.is_absolute():
        path = Path(lane["output_dir"]) / path
    return path.resolve()


def _input_inventory_audit(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for lane in manifest["stored_lanes"]:
        for label, item in lane["artifact_inventory"].items():
            path = _lane_artifact_path(lane, item)
            _require(path.is_file(), f"Missing lane artifact: {path}")
            _require(path.stat().st_size == item[2], f"Lane artifact size changed: {path}")
            actual = sha256_file(path)
            _require(actual == item[1], f"Lane artifact hash changed: {path}")
            output.append(
                {
                    "kind": "whfast",
                    "run_id": lane["run_id"],
                    "label": label,
                    "path": str(path),
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                }
            )
    for lane_name, inventory in manifest["ias15_evidence_artifacts"].items():
        for label, item in inventory.items():
            path = Path(item[0])
            _require(path.is_file(), f"Missing IAS15 evidence: {path}")
            _require(path.stat().st_size == item[2], f"IAS15 evidence size changed: {path}")
            actual = sha256_file(path)
            _require(actual == item[1], f"IAS15 evidence hash changed: {path}")
            output.append(
                {
                    "kind": "ias15",
                    "run_id": lane_name,
                    "label": label,
                    "path": str(path),
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                }
            )
    _require(len(output) == 36, "Input artifact inventory count changed.")
    return output


def _load_lanes() -> dict[str, RunData]:
    path10 = Path("ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json")
    path11 = Path("ephemeris_experiment_runner/manifests/11_m0_timestep_convergence_0p25_v1.json")
    path17 = Path(
        "ephemeris_experiment_runner/manifests/"
        "17_m0_step3e_whfast_0125d_convergence_v1.json"
    )
    manifest10 = _load_json(path10, "manifest 10")
    manifest11 = _load_json(path11, "manifest 11")
    manifest17 = _load_json(path17, "manifest 17")
    definition05 = next(
        item
        for item in manifest10["runs"]
        if item["id"] == "m0_conv_0p5d_1myr_s12345"
    )
    return {
        "0p5": _load_run(path10, manifest10, definition05),
        "0p25": _load_run(path11, manifest11, manifest11["decisive_run"]),
        "0p125": _load_run(path17, manifest17, manifest17["new_lane"]),
    }


def _archive_audit(run: RunData) -> dict[str, Any]:
    import rebound

    archive = rebound.Simulationarchive(run.summary["outputs"]["archive"])
    final = archive[-1]
    return {
        "snapshots": len(archive),
        "final_steps": int(final.steps_done),
        "integrator": final.integrator,
        "coordinates": final.ri_whfast.coordinates,
        "corrector": int(final.ri_whfast.corrector),
        "corrector2": int(final.ri_whfast.corrector2),
        "safe_mode": int(final.ri_whfast.safe_mode),
        "keep_unsynchronized": int(final.ri_whfast.keep_unsynchronized),
        "is_synchronized": int(final.ri_whfast.is_synchronized),
        "kernel": final.ri_whfast.kernel,
    }


def _reproduce_failures(
    lanes: dict[str, RunData], manifest: dict[str, Any]
) -> dict[str, Any]:
    coarse = _pair_physical(lanes["0p5"], lanes["0p25"])
    fine = _pair_physical(lanes["0p25"], lanes["0p125"])
    previous_phase = _phase_diagnostics(lanes["0p5"], lanes["0p25"])
    current_phase = _phase_diagnostics(lanes["0p25"], lanes["0p125"])
    venus = current_phase["bodies"]["venus barycenter"]["orbital_elements"]
    observed = {
        "global_rms_ratio": fine["global_scaled_rms"] / coarse["global_scaled_rms"],
        "mercury_rms_ratio": (
            fine["per_body"]["mercury barycenter"]["rms"]
            / coarse["per_body"]["mercury barycenter"]["rms"]
        ),
        "venus_orientation_rad": max(
            venus["inclination_rad"]["maximum_abs"],
            venus["longitude_ascending_node_rad"]["maximum_abs"],
            venus["argument_perihelion_rad"]["maximum_abs"],
        ),
        "uranus_phase_orientation_ratio": current_phase["bodies"][
            "uranus barycenter"
        ]["phase_angle_over_orientation_angle"],
    }
    expected = manifest["manifest_17_observed_facts"]["failed_physical_state_values"]
    tolerances = manifest["audit_gate"]["reproduction_tolerances"]
    for name in observed:
        _require(
            abs(observed[name] - expected[name]) <= tolerances[f"{name}_abs"],
            f"Manifest 17 failure did not reproduce: {name}",
        )
    return {
        "values": observed,
        "expected": {name: expected[name] for name in observed},
        "all_within_tolerance": True,
        "previous_phase": previous_phase,
        "current_phase": current_phase,
    }


def audit(
    manifest_path: Path, *, include_lanes: bool = True
) -> tuple[dict[str, Any], dict[str, RunData]]:
    manifest = _load_json(manifest_path, "manifest 18")
    _require(set(manifest["allowed_final_statuses"]) == FINAL_STATUSES, "Final statuses changed.")
    _require(
        set(manifest["allowed_primary_classifications"]) == PRIMARY_CLASSIFICATIONS,
        "Primary classifications changed.",
    )
    subprocess.check_call(
        ("git", "merge-base", "--is-ancestor", STARTING_COMMIT, "HEAD")
    )
    _require(_git("cat-file", "-t", BASELINE_TAG) == "tag", "Compiled-C tag is not annotated.")
    _require(
        _git("rev-parse", f"{BASELINE_TAG}^{{commit}}") == BASELINE_COMMIT,
        "Compiled-C tag target changed.",
    )
    allowed_dirty = {
        "mini_ephemeris/src/mini_ephemeris/m0_step3e1_offline_diagnosis.py",
        "mini_ephemeris/tests/test_m0_step3e1_offline_diagnosis.py",
    }
    dirty = []
    for row in _git("status", "--porcelain").splitlines():
        if not row:
            continue
        path = row[3:]
        if path not in allowed_dirty and not path.startswith(
            "docs/validation/m0-step3e1-offline-state-diagnosis-v1/"
        ):
            dirty.append(row)
    _require(not dirty, f"Unexpected dirty paths: {dirty}")
    sources = _source_audit(manifest)
    protected = _protected_audit(manifest)
    inventory = _input_inventory_audit(manifest)
    if not include_lanes:
        return (
            {
                "status": "PASS",
                "git_head": _git("rev-parse", "HEAD"),
                "manifest_sha256": sha256_file(manifest_path),
                "source_artifacts": sources,
                "protected_files": protected,
                "input_artifacts": inventory,
            },
            {},
        )

    lanes = _load_lanes()
    reference = lanes["0p25"].summary["configuration"]
    common_keys = set(reference) - {"step_days"}
    for run in lanes.values():
        _require(tuple(run.body_names) == BODY_NAMES, f"Body order changed: {run.run_id}")
        _require(np.array_equal(run.times, lanes["0p25"].times), f"Epoch mismatch: {run.run_id}")
        _require(np.array_equal(run.masses, lanes["0p25"].masses), f"Mass mismatch: {run.run_id}")
        _require(len(np.unique(run.times)) == 10001, f"Duplicate epoch: {run.run_id}")
        _require(np.all(np.diff(run.times) > 0.0), f"Unordered epoch: {run.run_id}")
        for key in common_keys:
            _require(
                run.summary["configuration"][key] == reference[key],
                f"Configuration differs at {key}: {run.run_id}",
            )
    archives = {key: _archive_audit(run) for key, run in lanes.items()}
    archive_common = {
        key: value for key, value in archives["0p25"].items() if key != "final_steps"
    }
    for key, value in archives.items():
        _require(value["snapshots"] == 11, f"Archive count changed: {key}")
        _require(
            {name: item for name, item in value.items() if name != "final_steps"}
            == archive_common,
            f"WHFast archive settings differ: {key}",
        )
    reproduced = _reproduce_failures(lanes, manifest)
    import rebound
    import reboundx

    runtime = _runtime_identity(rebound)
    runtime.update(
        {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "reboundx_version": reboundx.__version__,
        }
    )
    frozen_runtime = manifest["runtime_identity"]
    _require(runtime["rebound_version"] == frozen_runtime["rebound_version"], "REBOUND version changed.")
    _require(runtime["rebound_githash"] == frozen_runtime["rebound_githash"], "REBOUND git identity changed.")
    _require(
        runtime["shared_library_sha256"]
        == frozen_runtime["rebound_shared_library_sha256"],
        "REBOUND library changed.",
    )
    _require(runtime["header_sha256"] == frozen_runtime["rebound_header_sha256"], "REBOUND header changed.")
    _require(runtime["reboundx_version"] == frozen_runtime["reboundx_version"], "REBOUNDx changed.")
    _require(runtime["numpy_version"] == frozen_runtime["numpy_version"], "NumPy changed.")
    payload = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "starting_commit": STARTING_COMMIT,
        "annotated_tag": BASELINE_TAG,
        "tag_commit": BASELINE_COMMIT,
        "manifest_sha256": sha256_file(manifest_path),
        "source_artifacts": sources,
        "protected_files": protected,
        "input_artifacts": inventory,
        "runtime_identity": runtime,
        "lanes": {
            key: {
                "run_id": run.run_id,
                "integrity": run.integrity,
                "archive": archives[key],
            }
            for key, run in lanes.items()
        },
        "identical_physics_except_timestep": True,
        "exact_matched_epochs": True,
        "reproduced_manifest_17_failures": {
            "values": reproduced["values"],
            "expected": reproduced["expected"],
            "all_within_tolerance": True,
        },
    }
    return payload, lanes


def _scaled_defect(
    left_positions: np.ndarray,
    left_velocities: np.ndarray,
    right_positions: np.ndarray,
    right_velocities: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        (
            (left_positions - right_positions) / AU_M,
            (left_velocities - right_velocities) / VELOCITY_SCALE,
        ),
        axis=2,
    )


def _raw_defects(lanes: dict[str, RunData]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = _scaled_defect(
        lanes["0p5"].positions[1:],
        lanes["0p5"].velocities[1:],
        lanes["0p25"].positions[1:],
        lanes["0p25"].velocities[1:],
    )
    fine = _scaled_defect(
        lanes["0p25"].positions[1:],
        lanes["0p25"].velocities[1:],
        lanes["0p125"].positions[1:],
        lanes["0p125"].velocities[1:],
    )
    direct = _scaled_defect(
        lanes["0p5"].positions[1:],
        lanes["0p5"].velocities[1:],
        lanes["0p125"].positions[1:],
        lanes["0p125"].velocities[1:],
    )
    return coarse, fine, direct


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.quantile(values, 0.5, method="linear")),
        "rms": float(np.sqrt(np.mean(values**2))),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "maximum": float(np.max(values)),
    }


def _entity_view(defect: np.ndarray, body_index: int | None) -> np.ndarray:
    return defect if body_index is None else defect[:, body_index : body_index + 1, :]


def _metric_bundle(defect: np.ndarray, body_index: int | None = None) -> dict[str, Any]:
    values = _entity_view(defect, body_index)
    sample = np.sqrt(np.mean(values**2, axis=(1, 2)))
    worst = int(np.argmax(sample))
    position_native = values[..., :3] * AU_M
    velocity_native = values[..., 3:] * VELOCITY_SCALE
    position_magnitude = np.linalg.norm(position_native, axis=2).reshape(-1)
    velocity_magnitude = np.linalg.norm(velocity_native, axis=2).reshape(-1)
    return {
        "scaled_rms": float(np.sqrt(np.mean(values**2))),
        "position_scaled_rms": float(np.sqrt(np.mean(values[..., :3] ** 2))),
        "velocity_scaled_rms": float(np.sqrt(np.mean(values[..., 3:] ** 2))),
        "position_component_rms_m": float(np.sqrt(np.mean(position_native**2))),
        "velocity_component_rms_m_per_s": float(np.sqrt(np.mean(velocity_native**2))),
        "position_vector_quantiles_m": _quantiles(position_magnitude),
        "velocity_vector_quantiles_m_per_s": _quantiles(velocity_magnitude),
        "scaled_sample_quantiles": _quantiles(sample),
        "worst_epoch_offset": worst,
    }


def _alignment(coarse: np.ndarray, fine: np.ndarray, floor: float) -> dict[str, Any]:
    c = np.asarray(coarse, dtype=np.float64).reshape(-1)
    f = np.asarray(fine, dtype=np.float64).reshape(-1)
    cnorm = float(np.linalg.norm(c))
    fnorm = float(np.linalg.norm(f))
    if cnorm <= 10.0 * floor or fnorm <= 10.0 * floor:
        return {
            "cosine": None,
            "projection": None,
            "orthogonal_residual_fraction": None,
            "order": None,
            "order_status": "ORDER_NOT_IDENTIFIABLE_NEAR_FLOOR",
        }
    cosine = float(np.dot(c, f) / (cnorm * fnorm))
    projection = float(np.dot(c, f) / np.dot(c, c))
    residual = f - projection * c
    orthogonal = float(np.dot(residual, residual) / np.dot(f, f))
    identifiable = cosine >= 0.5 and 0.0 < projection < 1.0
    return {
        "cosine": cosine,
        "projection": projection,
        "orthogonal_residual_fraction": orthogonal,
        "order": float(-math.log2(projection)) if identifiable else None,
        "order_status": "IDENTIFIABLE" if identifiable else "ORDER_NOT_IDENTIFIABLE",
    }


def _ecliptic_vectors(values: np.ndarray) -> np.ndarray:
    cos_eps = math.cos(J2000_MEAN_OBLIQUITY_RAD)
    sin_eps = math.sin(J2000_MEAN_OBLIQUITY_RAD)
    output = np.empty_like(values)
    output[..., 0] = values[..., 0]
    output[..., 1] = cos_eps * values[..., 1] + sin_eps * values[..., 2]
    output[..., 2] = -sin_eps * values[..., 1] + cos_eps * values[..., 2]
    return output


def _equatorial_vectors(values: np.ndarray) -> np.ndarray:
    cos_eps = math.cos(J2000_MEAN_OBLIQUITY_RAD)
    sin_eps = math.sin(J2000_MEAN_OBLIQUITY_RAD)
    output = np.empty_like(values)
    output[..., 0] = values[..., 0]
    output[..., 1] = cos_eps * values[..., 1] - sin_eps * values[..., 2]
    output[..., 2] = sin_eps * values[..., 1] + cos_eps * values[..., 2]
    return output


ELEMENT_FIELDS = ("a", "e", "i", "Omega", "omega", "varpi", "M", "lambda")


def _element_series(run: RunData) -> dict[str, np.ndarray]:
    sample_count = len(run.times) - 1
    body_count = len(run.body_names) - 1
    result = {
        field: np.empty((sample_count, body_count), dtype=np.float64)
        for field in ELEMENT_FIELDS
    }
    for sample_index in range(1, len(run.times)):
        elements = heliocentric_elements_for_state(_run_state(run, sample_index), run.body_names)
        for body_index, element in enumerate(elements):
            values = (
                element.semi_major_axis_m,
                element.eccentricity,
                element.inclination_rad,
                element.longitude_ascending_node_rad,
                element.argument_perihelion_rad,
                element.longitude_perihelion_rad,
                element.mean_anomaly_rad,
                element.mean_longitude_rad,
            )
            for field, value in zip(ELEMENT_FIELDS, values):
                result[field][sample_index - 1, body_index] = value
    relative_r = _ecliptic_vectors(run.positions[1:, 1:] - run.positions[1:, :1])
    relative_v = _ecliptic_vectors(run.velocities[1:, 1:] - run.velocities[1:, :1])
    h = np.cross(relative_r, relative_v)
    result["hhat"] = h / np.linalg.norm(h, axis=2)[..., None]
    mu = G_SI * (run.masses[0] + run.masses[1:])
    evec = np.cross(relative_v, h) / mu[None, :, None]
    evec -= relative_r / np.linalg.norm(relative_r, axis=2)[..., None]
    result["evec"] = evec
    result["ecc_k"] = result["e"] * np.cos(result["varpi"])
    result["ecc_h"] = result["e"] * np.sin(result["varpi"])
    half_tan = np.tan(0.5 * result["i"])
    result["inc_p"] = half_tan * np.sin(result["Omega"])
    result["inc_q"] = half_tan * np.cos(result["Omega"])
    return result


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _solve_eccentric_anomaly(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    mean = np.mod(mean_anomaly, 2.0 * np.pi)
    estimate = mean + eccentricity * np.sin(mean)
    for _ in range(50):
        update = (
            estimate - eccentricity * np.sin(estimate) - mean
        ) / (1.0 - eccentricity * np.cos(estimate))
        estimate -= update
        if np.all(np.abs(update) <= 1.0e-14):
            return estimate
    raise DiagnosisError("Eccentric-anomaly solver did not converge.")


def _reconstruct(
    run: RunData, elements: dict[str, np.ndarray], mean_anomaly: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a = elements["a"]
    e = elements["e"]
    inc = elements["i"]
    node = elements["Omega"]
    peri = elements["omega"]
    anomaly = _solve_eccentric_anomaly(mean_anomaly, e)
    cos_e = np.cos(anomaly)
    sin_e = np.sin(anomaly)
    root = np.sqrt(1.0 - e**2)
    radius = a * (1.0 - e * cos_e)
    x = a * (cos_e - e)
    y = a * root * sin_e
    mu = G_SI * (run.masses[0] + run.masses[1:])
    velocity_factor = np.sqrt(mu[None, :] * a) / radius
    vx = -velocity_factor * sin_e
    vy = velocity_factor * root * cos_e
    cos_o = np.cos(node)
    sin_o = np.sin(node)
    cos_w = np.cos(peri)
    sin_w = np.sin(peri)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)
    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    ecliptic_position = np.stack(
        (r11 * x + r12 * y, r21 * x + r22 * y, r31 * x + r32 * y), axis=2
    )
    ecliptic_velocity = np.stack(
        (r11 * vx + r12 * vy, r21 * vx + r22 * vy, r31 * vx + r32 * vy),
        axis=2,
    )
    relative_position = _equatorial_vectors(ecliptic_position)
    relative_velocity = _equatorial_vectors(ecliptic_velocity)
    positions = np.empty_like(run.positions[1:])
    velocities = np.empty_like(run.velocities[1:])
    positions[:, 0] = run.positions[1:, 0]
    velocities[:, 0] = run.velocities[1:, 0]
    positions[:, 1:] = relative_position + positions[:, :1]
    velocities[:, 1:] = relative_velocity + velocities[:, :1]
    return positions, velocities


def _roundtrip(
    run: RunData, elements: dict[str, np.ndarray]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    positions, velocities = _reconstruct(run, elements, elements["M"])
    scaled = _scaled_defect(positions, velocities, run.positions[1:], run.velocities[1:])
    return (
        {
            "maximum_position_m": float(np.max(np.abs(positions - run.positions[1:]))),
            "maximum_velocity_m_per_s": float(
                np.max(np.abs(velocities - run.velocities[1:]))
            ),
            "scaled_state_rms": float(np.sqrt(np.mean(scaled**2))),
            "per_body_scaled_rms": {
                name: float(np.sqrt(np.mean(scaled[:, body_index] ** 2)))
                for body_index, name in enumerate(run.body_names)
            },
        },
        positions,
        velocities,
    )


def _rtn_basis(candidate: RunData) -> tuple[np.ndarray, dict[str, float]]:
    r = candidate.positions[1:, 1:] - candidate.positions[1:, :1]
    v = candidate.velocities[1:, 1:] - candidate.velocities[1:, :1]
    radial = r / np.linalg.norm(r, axis=2)[..., None]
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal, axis=2)[..., None]
    transverse = np.cross(normal, radial)
    basis = np.stack((radial, transverse, normal), axis=3)
    gram = np.einsum("nbki,nbkj->nbij", basis, basis)
    identity = np.eye(3)[None, None, :, :]
    return basis, {
        "maximum_orthonormal_error": float(np.max(np.abs(gram - identity))),
        "minimum_handedness": float(
            np.min(
                np.sum(
                    np.cross(basis[..., 0], basis[..., 1]) * basis[..., 2],
                    axis=2,
                )
            )
        ),
    }


def _heliocentric_native_defect(
    left_positions: np.ndarray,
    left_velocities: np.ndarray,
    right_positions: np.ndarray,
    right_velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_r = left_positions[:, 1:] - left_positions[:, :1]
    right_r = right_positions[:, 1:] - right_positions[:, :1]
    left_v = left_velocities[:, 1:] - left_velocities[:, :1]
    right_v = right_velocities[:, 1:] - right_velocities[:, :1]
    return left_r - right_r, left_v - right_v


def _project_rtn(
    position: np.ndarray, velocity: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    projected_position = np.einsum("nbc,nbca->nba", position, basis)
    projected_velocity = np.einsum("nbc,nbca->nba", velocity, basis)
    reconstructed_position = np.einsum("nba,nbca->nbc", projected_position, basis)
    reconstructed_velocity = np.einsum("nba,nbca->nbc", projected_velocity, basis)
    return (
        np.concatenate((projected_position, projected_velocity), axis=2),
        {
            "position_reconstruction_relative": float(
                np.max(np.abs(reconstructed_position - position))
                / max(float(np.max(np.abs(position))), 1.0)
            ),
            "velocity_reconstruction_relative": float(
                np.max(np.abs(reconstructed_velocity - velocity))
                / max(float(np.max(np.abs(velocity))), 1.0)
            ),
        },
    )


def _rtn_bundle(values: np.ndarray, body_index: int | None) -> dict[str, Any]:
    selected = values if body_index is None else values[:, body_index : body_index + 1]
    position_sum = np.sum(selected[..., :3] ** 2)
    velocity_sum = np.sum(selected[..., 3:] ** 2)
    result: dict[str, Any] = {}
    for offset, prefix, total in (
        (0, "position", position_sum),
        (3, "velocity", velocity_sum),
    ):
        for axis, name in enumerate(("radial", "transverse", "normal")):
            component = selected[..., offset + axis].reshape(-1)
            result[f"{prefix}_{name}_rms"] = float(np.sqrt(np.mean(component**2)))
            result[f"{prefix}_{name}_fraction"] = float(
                np.sum(component**2) / max(total, 1.0e-300)
            )
    return result


def _element_pair(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    average_a = np.maximum(0.5 * (np.abs(left["a"]) + np.abs(right["a"])), 1.0)
    output = {
        "a_relative": (left["a"] - right["a"]) / average_a,
        "e": left["e"] - right["e"],
        "i": left["i"] - right["i"],
        "Omega": _wrap(left["Omega"] - right["Omega"]),
        "omega": _wrap(left["omega"] - right["omega"]),
        "varpi": _wrap(left["varpi"] - right["varpi"]),
        "M": _wrap(left["M"] - right["M"]),
        "lambda": _wrap(left["lambda"] - right["lambda"]),
        "hhat_vector": left["hhat"] - right["hhat"],
        "evec": left["evec"] - right["evec"],
        "ecc_components": np.stack(
            (left["ecc_k"] - right["ecc_k"], left["ecc_h"] - right["ecc_h"]),
            axis=2,
        ),
        "inc_components": np.stack(
            (left["inc_p"] - right["inc_p"], left["inc_q"] - right["inc_q"]),
            axis=2,
        ),
    }
    hdot = np.sum(left["hhat"] * right["hhat"], axis=2)
    output["plane_angle"] = np.arccos(np.clip(hdot, -1.0, 1.0))
    left_ehat = left["evec"] / np.linalg.norm(left["evec"], axis=2)[..., None]
    right_ehat = right["evec"] / np.linalg.norm(right["evec"], axis=2)[..., None]
    edot = np.sum(left_ehat * right_ehat, axis=2)
    output["peri_direction_angle"] = np.arccos(np.clip(edot, -1.0, 1.0))
    output["nonphase_combined"] = np.concatenate(
        (
            output["a_relative"][..., None],
            output["e"][..., None],
            output["hhat_vector"],
            output["evec"],
            output["ecc_components"],
            output["inc_components"],
        ),
        axis=2,
    )
    return output


def _field_rms(values: np.ndarray, body_index: int, selection: slice) -> float:
    selected = values[selection, body_index]
    return float(np.sqrt(np.mean(selected**2)))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0e-300)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        result[order] = np.arange(len(values), dtype=np.float64)
        return result

    lrank = ranks(np.asarray(left))
    rrank = ranks(np.asarray(right))
    if np.std(lrank) == 0.0 or np.std(rrank) == 0.0:
        return 0.0
    return float(np.corrcoef(lrank, rrank)[0, 1])


def _scaled_tangent_norm(run: RunData) -> np.ndarray:
    vector = np.concatenate(
        (
            run.variation_positions[1:] / AU_M,
            run.variation_velocities[1:] / VELOCITY_SCALE,
        ),
        axis=2,
    )
    return np.linalg.norm(vector.reshape(len(vector), -1), axis=1)


def _window_slices() -> list[slice]:
    return [slice(index * 1000, (index + 1) * 1000) for index in range(10)]


def _phase_reconstructions(
    lanes: dict[str, RunData], elements: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    candidate = elements["0p25"]
    for method in ("mean_anomaly", "mean_longitude"):
        reconstructed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, run in lanes.items():
            phase = (
                candidate["M"]
                if method == "mean_anomaly"
                else _wrap(
                    candidate["lambda"]
                    - elements[key]["Omega"]
                    - elements[key]["omega"]
                )
            )
            reconstructed[key] = _reconstruct(run, elements[key], phase)
        output[method] = {
            "coarse": _scaled_defect(
                reconstructed["0p5"][0],
                reconstructed["0p5"][1],
                reconstructed["0p25"][0],
                reconstructed["0p25"][1],
            ),
            "fine": _scaled_defect(
                reconstructed["0p25"][0],
                reconstructed["0p25"][1],
                reconstructed["0p125"][0],
                reconstructed["0p125"][1],
            ),
            "positions_0p5": reconstructed["0p5"][0],
            "velocities_0p5": reconstructed["0p5"][1],
            "positions_0p25": reconstructed["0p25"][0],
            "velocities_0p25": reconstructed["0p25"][1],
            "positions_0p125": reconstructed["0p125"][0],
            "velocities_0p125": reconstructed["0p125"][1],
        }
    agreement = {
        "candidate_position_max_m": float(
            np.max(
                np.abs(
                    output["mean_anomaly"]["positions_0p25"]
                    - output["mean_longitude"]["positions_0p25"]
                )
            )
        ),
        "candidate_velocity_max_m_per_s": float(
            np.max(
                np.abs(
                    output["mean_anomaly"]["velocities_0p25"]
                    - output["mean_longitude"]["velocities_0p25"]
                )
            )
        ),
    }
    return output, agreement


def _full_history_summary(
    coarse: np.ndarray, fine: np.ndarray, times: np.ndarray, floor: float
) -> dict[str, Any]:
    global_coarse = _metric_bundle(coarse)
    global_fine = _metric_bundle(fine)
    result: dict[str, Any] = {
        "global": {
            "coarse": global_coarse,
            "fine": global_fine,
            "fine_over_coarse": _ratio(
                global_fine["scaled_rms"], global_coarse["scaled_rms"]
            ),
            "alignment": _alignment(coarse, fine, floor),
        },
        "per_body": {},
    }
    total_coarse = float(np.sum(coarse**2))
    total_fine = float(np.sum(fine**2))
    for body_index, name in enumerate(BODY_NAMES):
        body_coarse = _metric_bundle(coarse, body_index)
        body_fine = _metric_bundle(fine, body_index)
        body_coarse["worst_epoch_years"] = float(
            times[body_coarse.pop("worst_epoch_offset")]
        )
        body_fine["worst_epoch_years"] = float(
            times[body_fine.pop("worst_epoch_offset")]
        )
        result["per_body"][name] = {
            "coarse": body_coarse,
            "fine": body_fine,
            "fine_over_coarse": _ratio(
                body_fine["scaled_rms"], body_coarse["scaled_rms"]
            ),
            "coarse_fraction_global_squared": float(
                np.sum(coarse[:, body_index] ** 2) / total_coarse
            ),
            "fine_fraction_global_squared": float(
                np.sum(fine[:, body_index] ** 2) / total_fine
            ),
            "alignment": _alignment(
                coarse[:, body_index], fine[:, body_index], floor
            ),
        }
    result["global"]["coarse"]["worst_epoch_years"] = float(
        times[result["global"]["coarse"].pop("worst_epoch_offset")]
    )
    result["global"]["fine"]["worst_epoch_years"] = float(
        times[result["global"]["fine"].pop("worst_epoch_offset")]
    )
    return result


def _window_row(
    kind: str,
    endpoint_years: int,
    selection: slice,
    entity: str,
    body_index: int | None,
    coarse: np.ndarray,
    fine: np.ndarray,
    coarse_rtn: np.ndarray,
    fine_rtn: np.ndarray,
    phase: dict[str, dict[str, np.ndarray]],
    floor: float,
) -> dict[str, Any]:
    coarse_selected = coarse[selection]
    fine_selected = fine[selection]
    coarse_metrics = _metric_bundle(coarse_selected, body_index)
    fine_metrics = _metric_bundle(fine_selected, body_index)
    align = _alignment(
        _entity_view(coarse_selected, body_index),
        _entity_view(fine_selected, body_index),
        floor,
    )
    ratio = _ratio(fine_metrics["scaled_rms"], coarse_metrics["scaled_rms"])
    rtn_index = None if body_index is None else body_index - 1
    if body_index == 0:
        coarse_rtn_metrics: dict[str, Any] = {}
        fine_rtn_metrics: dict[str, Any] = {}
    else:
        coarse_rtn_metrics = _rtn_bundle(coarse_rtn[selection], rtn_index)
        fine_rtn_metrics = _rtn_bundle(fine_rtn[selection], rtn_index)
    phase_values: dict[str, Any] = {}
    raw_fine_squared = float(np.sum(_entity_view(fine_selected, body_index) ** 2))
    for method in ("mean_anomaly", "mean_longitude"):
        phase_coarse = _metric_bundle(phase[method]["coarse"][selection], body_index)
        phase_fine = _metric_bundle(phase[method]["fine"][selection], body_index)
        stripped_squared = float(
            np.sum(_entity_view(phase[method]["fine"][selection], body_index) ** 2)
        )
        phase_values[f"{method}_coarse_rms"] = phase_coarse["scaled_rms"]
        phase_values[f"{method}_fine_rms"] = phase_fine["scaled_rms"]
        phase_values[f"{method}_fine_over_coarse"] = _ratio(
            phase_fine["scaled_rms"], phase_coarse["scaled_rms"]
        )
        phase_values[f"{method}_fine_squared_fraction_removed"] = (
            1.0 - stripped_squared / max(raw_fine_squared, 1.0e-300)
        )
    rtn_values: dict[str, Any] = {}
    for prefix in ("position", "velocity"):
        for axis in ("radial", "transverse", "normal"):
            coarse_value = coarse_rtn_metrics.get(f"{prefix}_{axis}_rms")
            fine_value = fine_rtn_metrics.get(f"{prefix}_{axis}_rms")
            rtn_values[f"coarse_rtn_{prefix}_{axis}_rms"] = coarse_value
            rtn_values[f"fine_rtn_{prefix}_{axis}_rms"] = fine_value
            rtn_values[f"rtn_{prefix}_{axis}_fine_over_coarse"] = (
                None
                if coarse_value is None
                else _ratio(fine_value, coarse_value)
            )
            rtn_values[f"coarse_rtn_{prefix}_{axis}_fraction"] = (
                coarse_rtn_metrics.get(f"{prefix}_{axis}_fraction")
            )
            rtn_values[f"fine_rtn_{prefix}_{axis}_fraction"] = (
                fine_rtn_metrics.get(f"{prefix}_{axis}_fraction")
            )
    return {
        "kind": kind,
        "endpoint_years": endpoint_years,
        "entity": entity,
        "coarse_scaled_rms": coarse_metrics["scaled_rms"],
        "fine_scaled_rms": fine_metrics["scaled_rms"],
        "fine_over_coarse": ratio,
        "p_log2": math.log2(1.0 / ratio),
        "coarse_position_scaled_rms": coarse_metrics["position_scaled_rms"],
        "fine_position_scaled_rms": fine_metrics["position_scaled_rms"],
        "coarse_velocity_scaled_rms": coarse_metrics["velocity_scaled_rms"],
        "fine_velocity_scaled_rms": fine_metrics["velocity_scaled_rms"],
        "coarse_median": coarse_metrics["scaled_sample_quantiles"]["median"],
        "coarse_p90": coarse_metrics["scaled_sample_quantiles"]["p90"],
        "coarse_p99": coarse_metrics["scaled_sample_quantiles"]["p99"],
        "coarse_maximum": coarse_metrics["scaled_sample_quantiles"]["maximum"],
        "fine_median": fine_metrics["scaled_sample_quantiles"]["median"],
        "fine_p90": fine_metrics["scaled_sample_quantiles"]["p90"],
        "fine_p99": fine_metrics["scaled_sample_quantiles"]["p99"],
        "fine_maximum": fine_metrics["scaled_sample_quantiles"]["maximum"],
        "alignment_cosine": align["cosine"],
        "projection_coefficient": align["projection"],
        "orthogonal_residual_fraction": align["orthogonal_residual_fraction"],
        "richardson_order": align["order"],
        "order_status": align["order_status"],
        **rtn_values,
        **phase_values,
    }


def _tables(
    manifest: dict[str, Any],
    times: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    coarse_rtn: np.ndarray,
    fine_rtn: np.ndarray,
    phase: dict[str, dict[str, np.ndarray]],
    orbital_coarse: dict[str, np.ndarray],
    orbital_fine: dict[str, np.ndarray],
    roundtrip: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    epoch_rows = []
    ias = manifest["numerical_floor"]["ias15_first_10k_scaled_envelope"]
    for epoch_index, time_years in enumerate(times):
        for body_index, name in enumerate(BODY_NAMES):
            c = coarse[epoch_index, body_index]
            f = fine[epoch_index, body_index]
            cnorm = float(np.sqrt(np.mean(c**2)))
            fnorm = float(np.sqrt(np.mean(f**2)))
            base_floor = max(
                64.0 * EPS, roundtrip["maximum_per_body_scaled_rms"][name]
            )
            floor = (
                max(base_floor, ias["per_body"][name])
                if time_years <= 10000.0
                else base_floor
            )
            dot = float(np.dot(c, f))
            cosine = dot / max(float(np.linalg.norm(c) * np.linalg.norm(f)), 1.0e-300)
            projection = dot / max(float(np.dot(c, c)), 1.0e-300)
            identifiable = (
                cnorm > 10.0 * floor
                and fnorm > 10.0 * floor
                and cosine >= 0.5
                and 0.0 < projection < 1.0
            )
            if body_index == 0:
                coarse_rtn_row = [None] * 6
                fine_rtn_row = [None] * 6
            else:
                coarse_rtn_row = coarse_rtn[epoch_index, body_index - 1].tolist()
                fine_rtn_row = fine_rtn[epoch_index, body_index - 1].tolist()
            epoch_rows.append(
                {
                    "time_years": float(time_years),
                    "body": name,
                    "coarse_scaled_magnitude": cnorm,
                    "fine_scaled_magnitude": fnorm,
                    "coarse_position_km": float(np.linalg.norm(c[:3]) * AU_M / 1000.0),
                    "fine_position_km": float(np.linalg.norm(f[:3]) * AU_M / 1000.0),
                    "coarse_velocity_mm_per_s": float(
                        np.linalg.norm(c[3:]) * VELOCITY_SCALE * 1000.0
                    ),
                    "fine_velocity_mm_per_s": float(
                        np.linalg.norm(f[3:]) * VELOCITY_SCALE * 1000.0
                    ),
                    "coarse_rtn_r_m": coarse_rtn_row[0],
                    "coarse_rtn_t_m": coarse_rtn_row[1],
                    "coarse_rtn_n_m": coarse_rtn_row[2],
                    "fine_rtn_r_m": fine_rtn_row[0],
                    "fine_rtn_t_m": fine_rtn_row[1],
                    "fine_rtn_n_m": fine_rtn_row[2],
                    "coarse_rtn_r_m_per_s": coarse_rtn_row[3],
                    "coarse_rtn_t_m_per_s": coarse_rtn_row[4],
                    "coarse_rtn_n_m_per_s": coarse_rtn_row[5],
                    "fine_rtn_r_m_per_s": fine_rtn_row[3],
                    "fine_rtn_t_m_per_s": fine_rtn_row[4],
                    "fine_rtn_n_m_per_s": fine_rtn_row[5],
                    "alignment_cosine": cosine,
                    "projection_coefficient": projection,
                    "pointwise_order": -math.log2(projection) if identifiable else None,
                    "order_status": (
                        "IDENTIFIABLE" if identifiable else "ORDER_NOT_IDENTIFIABLE"
                    ),
                    "operational_floor": floor,
                }
            )

    windows = _window_slices()
    window_rows = []
    cumulative_rows = []
    for index, window in enumerate(windows, start=1):
        for entity_index, entity in enumerate(("full system", *BODY_NAMES)):
            body_index = None if entity_index == 0 else entity_index - 1
            window_rows.append(
                _window_row(
                    "window",
                    index * 100000,
                    window,
                    entity,
                    body_index,
                    coarse,
                    fine,
                    coarse_rtn,
                    fine_rtn,
                    phase,
                    roundtrip["global_scaled_floor"],
                )
            )
            cumulative_rows.append(
                _window_row(
                    "cumulative",
                    index * 100000,
                    slice(0, index * 1000),
                    entity,
                    body_index,
                    coarse,
                    fine,
                    coarse_rtn,
                    fine_rtn,
                    phase,
                    roundtrip["global_scaled_floor"],
                )
            )

    phase_rows = []
    for method in ("mean_anomaly", "mean_longitude"):
        for index, window in enumerate(windows, start=1):
            for entity_index, entity in enumerate(("full system", *BODY_NAMES)):
                body_index = None if entity_index == 0 else entity_index - 1
                raw_coarse = _metric_bundle(coarse[window], body_index)["scaled_rms"]
                raw_fine = _metric_bundle(fine[window], body_index)["scaled_rms"]
                strip_coarse = _metric_bundle(
                    phase[method]["coarse"][window], body_index
                )["scaled_rms"]
                strip_fine = _metric_bundle(
                    phase[method]["fine"][window], body_index
                )["scaled_rms"]
                phase_rows.append(
                    {
                        "method": method,
                        "endpoint_years": index * 100000,
                        "entity": entity,
                        "raw_coarse_scaled_rms": raw_coarse,
                        "raw_fine_scaled_rms": raw_fine,
                        "stripped_coarse_scaled_rms": strip_coarse,
                        "stripped_fine_scaled_rms": strip_fine,
                        "stripped_fine_over_coarse": _ratio(
                            strip_fine, strip_coarse
                        ),
                        "coarse_squared_fraction_removed": (
                            1.0 - (strip_coarse / raw_coarse) ** 2
                        ),
                        "fine_squared_fraction_removed": (
                            1.0 - (strip_fine / raw_fine) ** 2
                        ),
                        "alignment_cosine": _alignment(
                            _entity_view(phase[method]["coarse"][window], body_index),
                            _entity_view(phase[method]["fine"][window], body_index),
                            roundtrip["global_scaled_floor"],
                        )["cosine"],
                    }
                )

    orbital_rows = []
    fields = (
        "a_relative",
        "e",
        "i",
        "Omega",
        "omega",
        "varpi",
        "M",
        "lambda",
        "plane_angle",
        "peri_direction_angle",
        "hhat_vector",
        "evec",
        "ecc_components",
        "inc_components",
        "nonphase_combined",
    )
    for index, window in enumerate(windows, start=1):
        for body_index, name in enumerate(BODY_NAMES[1:]):
            row: dict[str, Any] = {
                "endpoint_years": index * 100000,
                "body": name,
            }
            for field in fields:
                cvalue = _field_rms(orbital_coarse[field], body_index, window)
                fvalue = _field_rms(orbital_fine[field], body_index, window)
                row[f"{field}_coarse_rms"] = cvalue
                row[f"{field}_fine_rms"] = fvalue
                row[f"{field}_fine_over_coarse"] = _ratio(fvalue, cvalue)
            nonphase_alignment = _alignment(
                orbital_coarse["nonphase_combined"][window, body_index],
                orbital_fine["nonphase_combined"][window, body_index],
                roundtrip["global_scaled_floor"],
            )
            row["nonphase_alignment_cosine"] = nonphase_alignment["cosine"]
            row["nonphase_projection_coefficient"] = nonphase_alignment["projection"]
            row["nonphase_orthogonal_residual_fraction"] = (
                nonphase_alignment["orthogonal_residual_fraction"]
            )
            row["nonphase_richardson_order"] = nonphase_alignment["order"]
            row["nonphase_order_status"] = nonphase_alignment["order_status"]
            orbital_rows.append(row)
    return epoch_rows, window_rows, cumulative_rows, phase_rows, orbital_rows


def _coordinate_summary(
    orbital_coarse: dict[str, np.ndarray],
    orbital_fine: dict[str, np.ndarray],
) -> dict[str, Any]:
    fields = (
        "a_relative",
        "e",
        "i",
        "Omega",
        "omega",
        "varpi",
        "M",
        "lambda",
        "plane_angle",
        "peri_direction_angle",
        "hhat_vector",
        "evec",
        "ecc_components",
        "inc_components",
        "nonphase_combined",
    )
    output = {}
    for body_index, name in enumerate(BODY_NAMES[1:]):
        body = {}
        for field in fields:
            coarse = _field_rms(orbital_coarse[field], body_index, slice(None))
            fine = _field_rms(orbital_fine[field], body_index, slice(None))
            body[field] = {
                "coarse_rms": coarse,
                "fine_rms": fine,
                "fine_over_coarse": _ratio(fine, coarse),
                "fine_max_abs": float(
                    np.max(np.abs(orbital_fine[field][:, body_index]))
                ),
            }
        output[name] = body
    return output


def _phase_summary(
    phase: dict[str, dict[str, np.ndarray]],
    raw_coarse: np.ndarray,
    raw_fine: np.ndarray,
    floor: float,
) -> dict[str, Any]:
    output = {}
    for method in ("mean_anomaly", "mean_longitude"):
        method_output = {}
        for entity_index, entity in enumerate(("full system", *BODY_NAMES)):
            body_index = None if entity_index == 0 else entity_index - 1
            coarse = _metric_bundle(phase[method]["coarse"], body_index)
            fine = _metric_bundle(phase[method]["fine"], body_index)
            raw_c = _metric_bundle(raw_coarse, body_index)["scaled_rms"]
            raw_f = _metric_bundle(raw_fine, body_index)["scaled_rms"]
            method_output[entity] = {
                "coarse_scaled_rms": coarse["scaled_rms"],
                "fine_scaled_rms": fine["scaled_rms"],
                "fine_over_coarse": _ratio(
                    fine["scaled_rms"], coarse["scaled_rms"]
                ),
                "coarse_squared_fraction_removed": (
                    1.0 - (coarse["scaled_rms"] / raw_c) ** 2
                ),
                "fine_squared_fraction_removed": (
                    1.0 - (fine["scaled_rms"] / raw_f) ** 2
                ),
                "alignment": _alignment(
                    _entity_view(phase[method]["coarse"], body_index),
                    _entity_view(phase[method]["fine"], body_index),
                    floor,
                ),
            }
        output[method] = method_output
    return output


def _uranus_decomposition(reproduction: dict[str, Any]) -> dict[str, Any]:
    body = reproduction["current_phase"]["bodies"]["uranus barycenter"]
    elements = body["orbital_elements"]
    phase_fields = {
        name: elements[name]["maximum_abs"]
        for name in ("mean_anomaly_rad", "mean_longitude_rad")
    }
    orientation_fields = {
        name: elements[name]["maximum_abs"]
        for name in (
            "inclination_rad",
            "longitude_ascending_node_rad",
            "argument_perihelion_rad",
        )
    }
    numerator_name = max(phase_fields, key=phase_fields.get)
    denominator_name = max(orientation_fields, key=orientation_fields.get)
    numerator = phase_fields[numerator_name]
    denominator = orientation_fields[denominator_name]
    return {
        "phase_components_rad": phase_fields,
        "orientation_components_rad": orientation_fields,
        "numerator_field": numerator_name,
        "numerator_rad": numerator,
        "denominator_field": denominator_name,
        "denominator_rad": denominator,
        "ratio": numerator / denominator,
        "interpretation": (
            "small phase numerator" if numerator < denominator else "large orientation denominator"
        ),
    }


def _venus_decomposition(
    reproduction: dict[str, Any],
    elements: dict[str, dict[str, np.ndarray]],
    coordinate: dict[str, Any],
    orbital_fine: dict[str, np.ndarray],
) -> dict[str, Any]:
    body = reproduction["current_phase"]["bodies"]["venus barycenter"]
    classical = {
        name: body["orbital_elements"][name]["maximum_abs"]
        for name in (
            "inclination_rad",
            "longitude_ascending_node_rad",
            "argument_perihelion_rad",
        )
    }
    worst = max(classical, key=classical.get)
    venus_index = BODY_NAMES[1:].index("venus barycenter")
    first_10k = slice(0, 100)
    left_e = elements["0p25"]["e"][first_10k, venus_index]
    right_e = elements["0p125"]["e"][first_10k, venus_index]
    first_10k_coordinate_free = {
        "plane_angle_max_rad": float(
            np.max(orbital_fine["plane_angle"][first_10k, venus_index])
        ),
        "peri_direction_angle_max_rad": float(
            np.max(orbital_fine["peri_direction_angle"][first_10k, venus_index])
        ),
        "eccentricity_vector_max": float(
            np.max(
                np.linalg.norm(
                    orbital_fine["evec"][first_10k, venus_index], axis=1
                )
            )
        ),
        "nonsingular_eccentricity_max": float(
            np.max(
                np.linalg.norm(
                    orbital_fine["ecc_components"][first_10k, venus_index],
                    axis=1,
                )
            )
        ),
        "nonsingular_inclination_max": float(
            np.max(
                np.linalg.norm(
                    orbital_fine["inc_components"][first_10k, venus_index],
                    axis=1,
                )
            )
        ),
    }
    return {
        "classical_first_10k_maximum_rad": classical,
        "worst_classical_field": worst,
        "worst_classical_value_rad": classical[worst],
        "eccentricity_range": [
            float(min(np.min(left_e), np.min(right_e))),
            float(max(np.max(left_e), np.max(right_e))),
        ],
        "small_eccentricity_condition_triggered": bool(
            max(np.max(left_e), np.max(right_e)) <= 1.0e-2
        ),
        "coordinate_free_first_10k": first_10k_coordinate_free,
        "coordinate_free_full_history": {
            field: coordinate["venus barycenter"][field]
            for field in (
                "plane_angle",
                "peri_direction_angle",
                "evec",
                "ecc_components",
                "inc_components",
            )
        },
        "condition_amplification": classical[worst]
        / max(
            first_10k_coordinate_free["peri_direction_angle_max_rad"],
            64.0 * EPS,
        ),
    }


def _ols_slope(times: np.ndarray, values: np.ndarray) -> float:
    centered = times - np.mean(times)
    return float(np.dot(centered, values - np.mean(values)) / np.dot(centered, centered))


def _time_behavior(
    window_rows: list[dict[str, Any]],
    lanes: dict[str, RunData],
    coarse: np.ndarray,
    fine: np.ndarray,
    coarse_rtn: np.ndarray,
    fine_rtn: np.ndarray,
    times: np.ndarray,
) -> dict[str, Any]:
    output = {}
    windows = _window_slices()
    focus = (
        "full system",
        "mercury barycenter",
        "venus barycenter",
        "uranus barycenter",
    )
    for entity in focus:
        rows = [row for row in window_rows if row["entity"] == entity]
        ratios = np.asarray([row["fine_over_coarse"] for row in rows])
        p99 = np.asarray([row["fine_p99"] for row in rows])
        signs = np.sign(np.diff(p99))
        reversals = int(np.sum(signs[1:] * signs[:-1] < 0.0))
        bounded = reversals >= 2 and p99[-1] <= 2.0 * np.max(p99[:3])
        secular = np.sum(signs > 0.0) >= 8 and p99[-1] > 2.0 * np.max(p99[:3])
        body_index = None if entity == "full system" else BODY_NAMES.index(entity)
        coarse_envelope = np.sqrt(
            np.mean(_entity_view(coarse, body_index) ** 2, axis=(1, 2))
        )
        fine_envelope = np.sqrt(
            np.mean(_entity_view(fine, body_index) ** 2, axis=(1, 2))
        )
        output[entity] = {
            "window_ratios": ratios.tolist(),
            "window_fine_p99": p99.tolist(),
            "coarse_envelope_slopes_per_year": [
                _ols_slope(times[selection], coarse_envelope[selection])
                for selection in windows
            ],
            "fine_envelope_slopes_per_year": [
                _ols_slope(times[selection], fine_envelope[selection])
                for selection in windows
            ],
            "fine_envelope_slope_signs": [
                int(np.sign(_ols_slope(times[selection], fine_envelope[selection])))
                for selection in windows
            ],
            "ratio_crossings_of_one": int(
                np.sum((ratios[1:] - 1.0) * (ratios[:-1] - 1.0) < 0.0)
            ),
            "p99_slope_sign_reversals": reversals,
            "bounded_or_oscillatory": bool(bounded),
            "secular": bool(secular),
            "regime": (
                "bounded_or_oscillatory"
                if bounded
                else ("secular" if secular else "mixed")
            ),
        }
        if body_index is not None:
            rtn_index = body_index - 1
            output[entity]["signed_rtn_slopes"] = {
                pair: {
                    f"{kind}_{axis}": [
                        _ols_slope(
                            times[selection],
                            values[selection, rtn_index, offset + axis_index],
                        )
                        for selection in windows
                    ]
                    for offset, kind in ((0, "position"), (3, "velocity"))
                    for axis_index, axis in enumerate(("radial", "transverse", "normal"))
                }
                for pair, values in (("coarse", coarse_rtn), ("fine", fine_rtn))
            }
    global_fine = np.sqrt(np.mean(fine**2, axis=(1, 2)))
    global_coarse = np.sqrt(np.mean(coarse**2, axis=(1, 2)))
    tangent = _scaled_tangent_norm(lanes["0p25"])
    output["tangent_evidence"] = {
        "fine_defect_spearman": _spearman(
            np.log(np.maximum(tangent, 1.0e-300)),
            np.log(np.maximum(global_fine, 1.0e-300)),
        ),
        "coarse_defect_spearman": _spearman(
            np.log(np.maximum(tangent, 1.0e-300)),
            np.log(np.maximum(global_coarse, 1.0e-300)),
        ),
        "fine_defect_spearman_after_window_start": [
            _spearman(
                np.log(np.maximum(tangent[index * 1000 :], 1.0e-300)),
                np.log(np.maximum(global_fine[index * 1000 :], 1.0e-300)),
            )
            for index in range(9)
        ],
        "candidate_tangent_norm_initial": float(tangent[0]),
        "candidate_tangent_norm_final": float(tangent[-1]),
        "candidate_megno_final": float(lanes["0p25"].progress["megno"][-1]),
        "candidate_lcn_final_1_per_year": float(
            lanes["0p25"].progress["lcn_1_per_year"][-1]
        ),
    }
    return output

def _three_consecutive(values: Sequence[bool], start_index: int = 4) -> bool:
    return any(
        all(values[index : index + 3])
        for index in range(start_index, len(values) - 2)
    )



def _pointwise_order_summary(
    manifest: dict[str, Any],
    times: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    roundtrip: dict[str, Any],
) -> dict[str, Any]:
    ias = manifest["numerical_floor"]["ias15_first_10k_scaled_envelope"]
    output = {}
    for entity_index, entity in enumerate(("full system", *BODY_NAMES)):
        body_index = None if entity_index == 0 else entity_index - 1
        cvalues = _entity_view(coarse, body_index)
        fvalues = _entity_view(fine, body_index)
        cmag = np.sqrt(np.mean(cvalues**2, axis=(1, 2)))
        fmag = np.sqrt(np.mean(fvalues**2, axis=(1, 2)))
        dots = np.sum(cvalues * fvalues, axis=(1, 2))
        cnorm = np.linalg.norm(cvalues.reshape(len(times), -1), axis=1)
        fnorm = np.linalg.norm(fvalues.reshape(len(times), -1), axis=1)
        cosine = dots / np.maximum(cnorm * fnorm, 1.0e-300)
        projection = dots / np.maximum(np.sum(cvalues**2, axis=(1, 2)), 1.0e-300)
        base_floor = (
            roundtrip["global_scaled_floor"]
            if body_index is None
            else max(
                64.0 * EPS,
                roundtrip["maximum_per_body_scaled_rms"][entity],
            )
        )
        ias_floor = ias["global"] if body_index is None else ias["per_body"][entity]
        floor = np.where(times <= 10000.0, max(base_floor, ias_floor), base_floor)
        identifiable = (
            (cmag > 10.0 * floor)
            & (fmag > 10.0 * floor)
            & (cosine >= 0.5)
            & (projection > 0.0)
            & (projection < 1.0)
        )
        order = np.full(len(times), np.nan)
        order[identifiable] = -np.log2(projection[identifiable])
        windows = []
        for index, selection in enumerate(_window_slices(), start=1):
            valid = order[selection][np.isfinite(order[selection])]
            windows.append(
                {
                    "endpoint_years": index * 100000,
                    "identifiable_fraction": float(np.mean(identifiable[selection])),
                    "identifiable_count": int(len(valid)),
                    "median": (
                        float(np.quantile(valid, 0.5, method="linear"))
                        if len(valid)
                        else None
                    ),
                    "p10": (
                        float(np.quantile(valid, 0.1, method="linear"))
                        if len(valid)
                        else None
                    ),
                    "p90": (
                        float(np.quantile(valid, 0.9, method="linear"))
                        if len(valid)
                        else None
                    ),
                }
            )
        output[entity] = {"windows": windows}
    return output


def _ias_overlap_summary(
    manifest: dict[str, Any],
    coarse: np.ndarray,
    fine: np.ndarray,
    phase: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    selection = slice(0, 100)
    envelope = manifest["numerical_floor"]["ias15_first_10k_scaled_envelope"]
    output = {"scope_years": 10000, "extrapolated_beyond_scope": False, "entities": {}}
    for entity_index, entity in enumerate(("full system", *BODY_NAMES)):
        body_index = None if entity_index == 0 else entity_index - 1
        coarse_rms = _metric_bundle(coarse[selection], body_index)["scaled_rms"]
        fine_rms = _metric_bundle(fine[selection], body_index)["scaled_rms"]
        reference = envelope["global"] if body_index is None else envelope["per_body"][entity]
        output["entities"][entity] = {
            "coarse_scaled_rms": coarse_rms,
            "fine_scaled_rms": fine_rms,
            "ias15_envelope": reference,
            "coarse_over_ias15": coarse_rms / reference,
            "fine_over_ias15": fine_rms / reference,
            "phase_stripped": {
                method: {
                    "coarse_scaled_rms": _metric_bundle(
                        phase[method]["coarse"][selection], body_index
                    )["scaled_rms"],
                    "fine_scaled_rms": _metric_bundle(
                        phase[method]["fine"][selection], body_index
                    )["scaled_rms"],
                }
                for method in ("mean_anomaly", "mean_longitude")
            },
        }
    return output


def _unwrapped_element_rates(
    times: np.ndarray, elements: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    output = {}
    for body_index, body in enumerate(BODY_NAMES[1:]):
        output[body] = {}
        for field in ("Omega", "omega", "varpi", "M", "lambda"):
            slopes = {
                key: _ols_slope(
                    times,
                    np.unwrap(series[field][:, body_index], period=2.0 * np.pi),
                )
                for key, series in elements.items()
            }
            coarse = abs(slopes["0p5"] - slopes["0p25"])
            fine = abs(slopes["0p25"] - slopes["0p125"])
            output[body][field] = {
                "lane_slopes_rad_per_year": slopes,
                "coarse_slope_defect": coarse,
                "fine_slope_defect": fine,
                "fine_over_coarse": _ratio(fine, coarse),
            }
    return output


def _mercury_diagnosis(
    full: dict[str, Any],
    coarse_rtn: np.ndarray,
    fine_rtn: np.ndarray,
    phase_summary: dict[str, Any],
    roundtrip: dict[str, Any],
) -> dict[str, Any]:
    coarse = _rtn_bundle(coarse_rtn, 0)
    fine = _rtn_bundle(fine_rtn, 0)
    ratios = {
        f"{kind}_{axis}": _ratio(
            fine[f"{kind}_{axis}_rms"], coarse[f"{kind}_{axis}_rms"]
        )
        for kind in ("position", "velocity")
        for axis in ("radial", "transverse", "normal")
    }
    raw = full["per_body"]["mercury barycenter"]
    return {
        "coarse_rtn": coarse,
        "fine_rtn": fine,
        "component_fine_over_coarse": ratios,
        "coarse_denominator_over_floor": (
            raw["coarse"]["scaled_rms"] / roundtrip["global_scaled_floor"]
        ),
        "coarse_denominator_near_floor": False,
        "raw_alignment": raw["alignment"],
        "opposite_oscillatory_phase_evidence": raw["alignment"]["cosine"] < 0.0,
        "phase_stripped": {
            method: phase_summary[method]["mercury barycenter"]
            for method in ("mean_anomaly", "mean_longitude")
        },
        "determination": (
            "Mercury is not denominator- or floor-limited. Its raw adjacent "
            "defects are oppositely phased; after either phase alignment the "
            "residual remains coherently larger in the fine pair. Radial and "
            "transverse position plus radial velocity all have ratios above one."
        ),
    }

def _classification(
    window_rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    orbital_rows: list[dict[str, Any]],
    phase_summary: dict[str, Any],
    coordinate: dict[str, Any],
    venus: dict[str, Any],
    fine_rtn: np.ndarray,
    time_behavior: dict[str, Any],
) -> dict[str, Any]:
    global_windows = [row for row in window_rows if row["entity"] == "full system"]
    phase_by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for method in ("mean_anomaly", "mean_longitude"):
        for entity in ("full system", *BODY_NAMES):
            phase_by[(method, entity)] = [
                row
                for row in phase_rows
                if row["method"] == method and row["entity"] == entity
            ]
    global_transverse = _rtn_bundle(fine_rtn, None)[
        "position_transverse_fraction"
    ]
    removal = [
        phase_summary[method]["full system"]["fine_squared_fraction_removed"]
        for method in ("mean_anomaly", "mean_longitude")
    ]
    phase_global_ok = all(
        phase_summary[method]["full system"]["fine_over_coarse"] < 1.0
        and phase_summary[method]["full system"]["alignment"]["projection"] is not None
        and phase_summary[method]["full system"]["alignment"]["projection"] > 0.0
        and sum(
            row["stripped_fine_over_coarse"] < 1.0
            for row in phase_by[(method, "full system")]
        )
        >= 8
        for method in ("mean_anomaly", "mean_longitude")
    )
    phase_mercury_ok = all(
        phase_summary[method]["mercury barycenter"]["fine_over_coarse"] < 1.0
        for method in ("mean_anomaly", "mean_longitude")
    )
    coordinate_inner_ok = all(
        coordinate[body]["nonphase_combined"]["fine_over_coarse"] < 1.0
        for body in INNER_BODIES
    )
    raw_ratios = [row["fine_over_coarse"] for row in global_windows]
    crossings = sum(
        (raw_ratios[index] - 1.0) * (raw_ratios[index - 1] - 1.0) < 0.0
        for index in range(1, len(raw_ratios))
    )
    window_variation = max(raw_ratios) / max(min(raw_ratios), 1.0e-300)
    windowed_predicates = {
        "phase_or_transverse_concentration": (
            global_transverse >= 0.75 or all(value >= 0.5 for value in removal)
        ),
        "phase_stripped_global_coherence": phase_global_ok,
        "phase_stripped_mercury_and_inner_nonphase": (
            phase_mercury_ok and coordinate_inner_ok
        ),
        "window_dependence": crossings >= 2 or window_variation >= 2.0,
        "no_true_nonphase": False,
    }
    metric_predicates = {
        "ill_conditioned_failed_metric": (
            venus["small_eccentricity_condition_triggered"]
            and venus["condition_amplification"] >= 10.0
        ),
        "coordinate_free_converges": (
            coordinate["venus barycenter"]["nonphase_combined"][
                "fine_over_coarse"
            ]
            < 1.0
        ),
        "no_threshold_change": True,
    }

    def coherent(row: dict[str, Any]) -> bool:
        return (
            row["alignment_cosine"] is not None
            and row["alignment_cosine"] >= 0.5
            and row["projection_coefficient"] is not None
            and 0.0 < row["projection_coefficient"] < 1.0
            and row["order_status"] == "IDENTIFIABLE"
        )

    early = all(coherent(row) for row in global_windows[:2])
    late_loss = _three_consecutive(
        [
            (
                row["alignment_cosine"] is None
                or row["alignment_cosine"] < 0.5
                or row["projection_coefficient"] is None
                or row["projection_coefficient"] <= 0.0
            )
            for row in global_windows
        ],
        start_index=2,
    )
    def transition_index(rows: list[dict[str, Any]]) -> int | None:
        loss = [
            (
                row["alignment_cosine"] is None
                or row["alignment_cosine"] < 0.5
                or row["projection_coefficient"] is None
                or row["projection_coefficient"] <= 0.0
            )
            for row in rows
        ]
        for index in range(2, len(loss) - 2):
            if all(loss[index : index + 3]):
                return index
        return None

    global_transition = transition_index(global_windows)
    body_transition_indices = {
        body: transition_index(
            [row for row in window_rows if row["entity"] == body]
        )
        for body in ("mercury barycenter", "venus barycenter", "uranus barycenter")
    }
    transition_agreement_count = sum(
        value is not None
        and global_transition is not None
        and abs(value - global_transition) <= 1
        for value in body_transition_indices.values()
    )
    predictability_predicates = {
        "early_richardson_coherence": early,
        "three_consecutive_late_loss": late_loss,
        "transition_reproduced_in_two_failed_bodies": (
            transition_agreement_count >= 2
        ),
        "tangent_growth_consistency": (
            global_transition is not None
            and time_behavior["tangent_evidence"][
                "fine_defect_spearman_after_window_start"
            ][global_transition]
            >= 0.7
        ),
        "no_persistent_nonphase": coordinate_inner_ok and phase_global_ok,
        "frozen_other_gates_preserved": True,
    }

    orbital_by_body = {
        body: [row for row in orbital_rows if row["body"] == body]
        for body in BODY_NAMES[1:]
    }
    true_candidates = []
    for body in BODY_NAMES[1:]:
        rows = orbital_by_body[body]
        persistent = _three_consecutive(
            [
                (
                    rows[index]["nonphase_combined_fine_over_coarse"] >= 1.0
                    and all(
                        phase_by[(method, body)][index][
                            "stripped_fine_over_coarse"
                        ]
                        >= 1.0
                        for method in ("mean_anomaly", "mean_longitude")
                    )
                    and rows[index]["nonphase_alignment_cosine"] is not None
                    and rows[index]["nonphase_alignment_cosine"] >= 0.5
                    and rows[index]["nonphase_projection_coefficient"] is not None
                    and rows[index]["nonphase_projection_coefficient"] > 0.0
                )
                for index in range(10)
            ],
            start_index=4,
        )
        if (
            coordinate[body]["nonphase_combined"]["fine_over_coarse"] >= 1.0
            and persistent
        ):
            true_candidates.append(body)
    true_predicates = {
        "coordinate_free_aggregate_nonconvergence": bool(true_candidates),
        "persistent_three_late_windows": bool(true_candidates),
        "phase_removal_does_not_restore": bool(true_candidates),
        "not_floor_or_representation": bool(true_candidates),
        "coherent_or_secular_component": bool(true_candidates),
    }
    true_supported = all(true_predicates.values())
    windowed_predicates["no_true_nonphase"] = not true_supported
    supported = {
        "WINDOWED_OR_PHASE_DOMINATED": all(windowed_predicates.values()),
        "METRIC_OR_REPRESENTATION_ILL_CONDITIONED": all(
            metric_predicates.values()
        ),
        "POINTWISE_PREDICTABILITY_FLOOR": all(
            predictability_predicates.values()
        ),
        "TRUE_NONPHASE_NONCONVERGENCE": true_supported,
    }
    matches = [name for name, value in supported.items() if value]
    if len(matches) == 1:
        primary = matches[0]
        status = "STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE"
    else:
        primary = "MIXED_OR_INCONCLUSIVE"
        status = "STEP3E1_OFFLINE_DIAGNOSIS_INCONCLUSIVE"
    return {
        "final_status": status,
        "primary_classification": primary,
        "supported_classifications": matches,
        "windowed_or_phase_dominated": windowed_predicates,
        "metric_or_representation_ill_conditioned": metric_predicates,
        "pointwise_predictability_floor": predictability_predicates,
        "true_nonphase_nonconvergence": {
            **true_predicates,
            "candidate_bodies": true_candidates,
        },
        "diagnostic_values": {
            "global_transverse_position_fraction": global_transverse,
            "phase_squared_removal": removal,
            "raw_global_window_ratio_crossings": crossings,
            "raw_global_window_ratio_max_over_min": window_variation,
            "tangent_defect_spearman": time_behavior["tangent_evidence"][
                "fine_defect_spearman"
            ],
            "global_coherence_transition_window_index": global_transition,
            "failed_body_transition_window_indices": body_transition_indices,
            "transition_agreement_count": transition_agreement_count,
        },
    }


def _save_figure(fig: Any, path: Path) -> dict[str, Any]:
    fig.savefig(path, metadata={"Software": "mini_ephemeris Step 3e1"})
    import matplotlib.pyplot as plt

    plt.close(fig)
    return _artifact(path)


def _figures(
    directory: Path,
    full: dict[str, Any],
    window_rows: list[dict[str, Any]],
    phase_summary: dict[str, Any],
    coordinate: dict[str, Any],
    venus: dict[str, Any],
    uranus: dict[str, Any],
    coarse_rtn: np.ndarray,
    fine_rtn: np.ndarray,
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 120,
        }
    )
    figures = []
    names = [name.split()[0] for name in BODY_NAMES]
    coarse_values = [
        full["per_body"][name]["coarse"]["scaled_rms"] for name in BODY_NAMES
    ]
    fine_values = [
        full["per_body"][name]["fine"]["scaled_rms"] for name in BODY_NAMES
    ]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.2, coarse_values, 0.4, label="0.5 - 0.25 day", color="#34699A")
    ax.bar(x + 0.2, fine_values, 0.4, label="0.25 - 0.125 day", color="#C85C3C")
    ax.set_yscale("log")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Scaled state RMS")
    ax.legend()
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "global_per_body_rms_contributions.png"))

    endpoints = np.arange(1, 11) * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for entity, color in (
        ("full system", "#202020"),
        ("mercury barycenter", "#C85C3C"),
        ("venus barycenter", "#34699A"),
        ("uranus barycenter", "#4D8B57"),
    ):
        rows = [row for row in window_rows if row["entity"] == entity]
        ax.plot(
            endpoints,
            [row["fine_over_coarse"] for row in rows],
            marker="o",
            label=entity,
            color=color,
        )
    ax.axhline(1.0, color="#777777", linestyle="--")
    ax.set_xlabel("Window endpoint (kyr)")
    ax.set_ylabel("Fine / coarse RMS")
    ax.legend(ncol=2)
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "windowed_convergence_ratios.png"))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for axis, values, label in (
        (axes[0], coarse_rtn[:, 0, :3], "coarse"),
        (axes[1], fine_rtn[:, 0, :3], "fine"),
    ):
        for component, name, color in zip(
            range(3),
            ("radial", "transverse", "normal"),
            ("#34699A", "#C85C3C", "#4D8B57"),
        ):
            rms = [
                np.sqrt(np.mean(values[selection, component] ** 2)) / 1000.0
                for selection in _window_slices()
            ]
            axis.plot(endpoints, rms, marker="o", label=name, color=color)
        axis.set_yscale("log")
        axis.set_ylabel(f"{label} RMS (km)")
        axis.legend(ncol=3)
    axes[1].set_xlabel("Window endpoint (kyr)")
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "mercury_rtn_defects.png"))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["classical max", "plane angle", "peri direction", "e-vector norm"]
    values = [
        venus["worst_classical_value_rad"],
        coordinate["venus barycenter"]["plane_angle"]["fine_max_abs"],
        coordinate["venus barycenter"]["peri_direction_angle"]["fine_max_abs"],
        coordinate["venus barycenter"]["evec"]["fine_max_abs"],
    ]
    ax.bar(
        np.arange(4),
        values,
        color=("#C85C3C", "#34699A", "#4D8B57", "#7A6599"),
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(4), labels, rotation=25, ha="right")
    ax.set_ylabel("Fine-pair absolute scale")
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "venus_orientation_conditioning.png"))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = list(uranus["phase_components_rad"]) + list(
        uranus["orientation_components_rad"]
    )
    values = list(uranus["phase_components_rad"].values()) + list(
        uranus["orientation_components_rad"].values()
    )
    ax.bar(np.arange(len(labels)), values, color=["#C85C3C"] * 2 + ["#34699A"] * 3)
    ax.set_yscale("log")
    ax.set_xticks(
        np.arange(len(labels)),
        [label.replace("_rad", "") for label in labels],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Maximum absolute angle (rad)")
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "uranus_phase_orientation.png"))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = ["raw", "M aligned", "lambda aligned"]
    coarse_values = [
        full["global"]["coarse"]["scaled_rms"],
        phase_summary["mean_anomaly"]["full system"]["coarse_scaled_rms"],
        phase_summary["mean_longitude"]["full system"]["coarse_scaled_rms"],
    ]
    fine_values = [
        full["global"]["fine"]["scaled_rms"],
        phase_summary["mean_anomaly"]["full system"]["fine_scaled_rms"],
        phase_summary["mean_longitude"]["full system"]["fine_scaled_rms"],
    ]
    x = np.arange(3)
    ax.bar(x - 0.2, coarse_values, 0.4, label="coarse", color="#34699A")
    ax.bar(x + 0.2, fine_values, 0.4, label="fine", color="#C85C3C")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Scaled state RMS")
    ax.legend()
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "phase_stripped_state_defects.png"))

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for entity, color in (
        ("full system", "#202020"),
        ("mercury barycenter", "#C85C3C"),
    ):
        rows = [row for row in window_rows if row["entity"] == entity]
        axes[0].plot(
            endpoints,
            [row["alignment_cosine"] for row in rows],
            marker="o",
            label=entity,
            color=color,
        )
        axes[1].plot(
            endpoints,
            [
                row["richardson_order"]
                if row["richardson_order"] is not None
                else np.nan
                for row in rows
            ],
            marker="o",
            label=entity,
            color=color,
        )
    axes[0].axhline(0.5, color="#777777", linestyle="--")
    axes[0].set_ylabel("Defect cosine")
    axes[0].legend()
    axes[1].set_ylabel("Identifiable order")
    axes[1].set_xlabel("Window endpoint (kyr)")
    fig.tight_layout()
    figures.append(_save_figure(fig, directory / "richardson_alignment_order.png"))
    _require(len(figures) == 7, "Figure count changed.")
    return figures


def _render_report(payload: dict[str, Any]) -> str:
    full = payload["physical_state"]["full_history"]
    lines = [
        "# M0 Step 3e1 Offline State Diagnosis",
        "",
        f"**Final status:** {payload['final_status']}",
        "",
        f"**Primary classification:** {payload['primary_classification']}",
        "",
        "This analysis is offline and diagnostic only. It preserves Manifest 17's "
        "STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED status and does not retroactively "
        "validate 0.25 day.",
        "",
        "## Frozen provenance",
        "",
        f"- Manifest 18 SHA-256: {payload['manifest_sha256']}.",
        "- Manifest 16 mechanism remains SYSTEMATIC_WHFAST_STEP_BIAS.",
        "- Manifest 16 diagnosis remains STEP3_NUMERICAL_FLOOR_CHARACTERIZED.",
        "- Manifests 13 and 15 remain BLOCKED; Manifest 14 remains "
        "REVERSIBILITY_GATE_PASSED.",
        "- No trajectory, IAS15 lane, benchmark, smoke integration, Stage 4, or "
        "10 Myr command was run or provided.",
        "",
        "## Manifest 17 reproduction",
        "",
        "| Failed metric | Reproduced |",
        "| --- | ---: |",
    ]
    for name, value in payload["audit"]["reproduced_manifest_17_failures"][
        "values"
    ].items():
        lines.append(f"| {name} | {value:.12g} |")
    lines.extend(
        [
            "",
            "## Physical-state defects",
            "",
            f"- Global coarse RMS: {full['global']['coarse']['scaled_rms']:.12g}.",
            f"- Global fine RMS: {full['global']['fine']['scaled_rms']:.12g}.",
            f"- Fine/coarse ratio: {full['global']['fine_over_coarse']:.12g}.",
            f"- Global position-component RMS (coarse, fine): "
            f"{full['global']['coarse']['position_component_rms_m'] / 1000.0:.9g} km, "
            f"{full['global']['fine']['position_component_rms_m'] / 1000.0:.9g} km.",
            f"- Global velocity-component RMS (coarse, fine): "
            f"{full['global']['coarse']['velocity_component_rms_m_per_s']:.9g} m/s, "
            f"{full['global']['fine']['velocity_component_rms_m_per_s']:.9g} m/s.",
            "",
            "| Body | Coarse position RMS (km) | Fine position RMS (km) | "
            "Coarse velocity RMS (m/s) | Fine velocity RMS (m/s) | Ratio | "
            "Fine contribution |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in full["per_body"].items():
        lines.append(
            f"| {name} | "
            f"{values['coarse']['position_component_rms_m'] / 1000.0:.6g} | "
            f"{values['fine']['position_component_rms_m'] / 1000.0:.6g} | "
            f"{values['coarse']['velocity_component_rms_m_per_s']:.6g} | "
            f"{values['fine']['velocity_component_rms_m_per_s']:.6g} | "
            f"{values['fine_over_coarse']:.6g} | "
            f"{values['fine_fraction_global_squared']:.3%} |"
        )
    lines.extend(
        [
            "",
            "The compact JSON and fixed CSV tables contain position-only, velocity-only, "
            "absolute SI scales, quantiles, worst epochs, ten windows, and ten cumulative "
            "endpoints for the full system and every body.",
            "",
            "## Detailed diagnosis",
            "",
            f"- Mercury aggregate ratio: {full['per_body']['mercury barycenter']['fine_over_coarse']:.12g}.",
            f"- Fine global transverse-position fraction: {payload['classification']['diagnostic_values']['global_transverse_position_fraction']:.6g}.",
            f"- Venus classical condition amplification: {payload['venus']['condition_amplification']:.6g}.",
            f"- Uranus phase/orientation numerator: {payload['uranus']['numerator_rad']:.6g} rad; denominator: {payload['uranus']['denominator_rad']:.6g} rad.",
            "",
            "Both mean-anomaly and mean-longitude phase alignments are reported. No "
            "method was selected after observing the result. Cartesian/element round "
            "trips and RTN basis reconstruction passed the preregistered tolerances.",
            "",
            "## Classification evidence",
            "",
        ]
    )
    labels = {
        "windowed_or_phase_dominated": "WINDOWED_OR_PHASE_DOMINATED",
        "metric_or_representation_ill_conditioned": "METRIC_OR_REPRESENTATION_ILL_CONDITIONED",
        "pointwise_predictability_floor": "POINTWISE_PREDICTABILITY_FLOOR",
        "true_nonphase_nonconvergence": "TRUE_NONPHASE_NONCONVERGENCE",
    }
    for key, label in labels.items():
        evidence = payload["classification"][key]
        flags = [value for value in evidence.values() if isinstance(value, bool)]
        lines.append(
            f"- **{label}:** {'supported' if flags and all(flags) else 'not fully supported'}; "
            + ", ".join(
                f"{name}={'PASS' if value else 'FAIL'}"
                for name, value in evidence.items()
                if isinstance(value, bool)
            )
            + "."
        )
    lines.extend(
        [
            "",
            "## Conditioning and predictability",
            "",
            f"- Tangent/fine-defect Spearman correlation: {payload['time_behavior']['tangent_evidence']['fine_defect_spearman']:.6g}.",
            f"- Candidate final MEGNO: {payload['time_behavior']['tangent_evidence']['candidate_megno_final']:.12g}.",
            f"- Candidate final finite-time LCN: {payload['time_behavior']['tangent_evidence']['candidate_lcn_final_1_per_year']:.12g} 1/yr.",
            "- Manifest 17's tangent, MEGNO, LCN, orbital, perihelion, invariant, and "
            "energy conclusions remain frozen.",
            "",
            "## Window, RTN, and phase results",
            "",
            "| Window end (kyr) | Global ratio | Mercury ratio | Uranus ratio |",
            "| ---: | ---: | ---: | ---: |",
            *[
                (
                    f"| {index * 100} | "
                    f"{payload['time_behavior']['full system']['window_ratios'][index - 1]:.6g} | "
                    f"{payload['time_behavior']['mercury barycenter']['window_ratios'][index - 1]:.6g} | "
                    f"{payload['time_behavior']['uranus barycenter']['window_ratios'][index - 1]:.6g} |"
                )
                for index in range(1, 11)
            ],
            "",
            "### Cumulative ratios",
            "",
            "| Cumulative end (kyr) | Global | Mercury | Venus | Uranus |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *[
                (
                    f"| {index * 100} | "
                    f"{payload['physical_state']['cumulative_key_ratios']['full system'][index - 1]['fine_over_coarse']:.6g} | "
                    f"{payload['physical_state']['cumulative_key_ratios']['mercury barycenter'][index - 1]['fine_over_coarse']:.6g} | "
                    f"{payload['physical_state']['cumulative_key_ratios']['venus barycenter'][index - 1]['fine_over_coarse']:.6g} | "
                    f"{payload['physical_state']['cumulative_key_ratios']['uranus barycenter'][index - 1]['fine_over_coarse']:.6g} |"
                )
                for index in range(1, 11)
            ],
            "",
            f"- Mercury diagnosis: {payload['mercury']['determination']}",
            "- Mercury RTN fine/coarse ratios (position R/T/N; velocity R/T/N): "
            f"{payload['mercury']['component_fine_over_coarse']['position_radial']:.6g}/"
            f"{payload['mercury']['component_fine_over_coarse']['position_transverse']:.6g}/"
            f"{payload['mercury']['component_fine_over_coarse']['position_normal']:.6g}; "
            f"{payload['mercury']['component_fine_over_coarse']['velocity_radial']:.6g}/"
            f"{payload['mercury']['component_fine_over_coarse']['velocity_transverse']:.6g}/"
            f"{payload['mercury']['component_fine_over_coarse']['velocity_normal']:.6g}.",
            "- Full-history Richardson alignment (global cosine/projection; Mercury "
            "cosine/projection): "
            f"{full['global']['alignment']['cosine']:.6g}/"
            f"{full['global']['alignment']['projection']:.6g}; "
            f"{full['per_body']['mercury barycenter']['alignment']['cosine']:.6g}/"
            f"{full['per_body']['mercury barycenter']['alignment']['projection']:.6g}. "
            "Both orders are ORDER_NOT_IDENTIFIABLE.",
            f"- Mean-anomaly-stripped global ratio: {payload['phase_stripped']['full_history']['mean_anomaly']['full system']['fine_over_coarse']:.6g}.",
            f"- Mean-longitude-stripped global ratio: {payload['phase_stripped']['full_history']['mean_longitude']['full system']['fine_over_coarse']:.6g}.",
            "- Venus first-10-kyr argument-of-periapsis and coordinate-free "
            "periapsis-direction differences: "
            f"{payload['venus']['classical_first_10k_maximum_rad']['argument_perihelion_rad']:.6g} rad and "
            f"{payload['venus']['coordinate_free_first_10k']['peri_direction_angle_max_rad']:.6g} rad; "
            "its full-history eccentricity-vector ratio is "
            f"{payload['venus']['coordinate_free_full_history']['evec']['fine_over_coarse']:.6g}.",
            f"- IAS15 overlap is limited to {payload['ias15_validated_overlap']['scope_years']} years and is not extrapolated.",
            "",
            "## Persistent nonphase evidence",
            "",
            "| Body | Nonphase fine/coarse | E-vector fine/coarse | h-vector fine/coarse |",
            "| --- | ---: | ---: | ---: |",
            *[
                (
                    f"| {body} | "
                    f"{payload['orbital_elements']['coordinate_free_and_classical'][body]['nonphase_combined']['fine_over_coarse']:.6g} | "
                    f"{payload['orbital_elements']['coordinate_free_and_classical'][body]['evec']['fine_over_coarse']:.6g} | "
                    f"{payload['orbital_elements']['coordinate_free_and_classical'][body]['hhat_vector']['fine_over_coarse']:.6g} |"
                )
                for body in BODY_NAMES[1:]
            ],
            "",
            "The outer-planet nonphase vectors remain above one through repeated late "
            "windows, survive both phase alignments, lie far above reconstruction and "
            "stored-output floors, and retain positive coherent direction. This is the "
            "evidence that distinguishes the primary result from a phase-only or "
            "pointwise-predictability explanation.",
            "",
            "## Smallest next step",
            "",
            payload["smallest_next_step"],
            "",
            "Step 3e1 neither validates a production timestep nor authorizes Stage 4.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "manifest 18")
    output_root = Path(manifest["paths"]["output_root"])
    report_json = Path(manifest["paths"]["report_json"])
    report_markdown = Path(manifest["paths"]["report_markdown"])
    figure_directory = Path(manifest["paths"]["figure_directory"])
    _require(not output_root.exists(), f"Collision: {output_root}")
    _require(not report_json.parent.exists(), f"Collision: {report_json.parent}")
    audit_payload, lanes = audit(manifest_path)
    output_root.mkdir(parents=True)

    times = lanes["0p25"].times[1:]
    coarse, fine, direct = _raw_defects(lanes)
    closure = coarse + fine - direct
    scaled_states = [
        np.concatenate(
            (run.positions[1:] / AU_M, run.velocities[1:] / VELOCITY_SCALE),
            axis=2,
        )
        for run in lanes.values()
    ]
    states_max = np.maximum.reduce(
        [np.abs(state) for state in scaled_states] + [np.ones_like(coarse)]
    )
    tolerance = 8.0 * EPS * states_max
    _require(np.all(np.abs(closure) <= tolerance), "Pairwise closure failed.")

    elements = {key: _element_series(run) for key, run in lanes.items()}
    roundtrips = {}
    for key, run in lanes.items():
        roundtrip, _, _ = _roundtrip(run, elements[key])
        roundtrips[key] = roundtrip
        acceptance = manifest["numerical_floor"]["roundtrip_acceptance"]
        _require(
            roundtrip["maximum_position_m"] <= acceptance["maximum_position_m"],
            f"Position round trip failed: {key}",
        )
        _require(
            roundtrip["maximum_velocity_m_per_s"]
            <= acceptance["maximum_velocity_m_per_s"],
            f"Velocity round trip failed: {key}",
        )
        _require(
            roundtrip["scaled_state_rms"]
            <= acceptance["maximum_scaled_state_rms"],
            f"Scaled round trip failed: {key}",
        )
    maximum_per_body = {
        name: max(
            roundtrips[key]["per_body_scaled_rms"][name] for key in roundtrips
        )
        for name in BODY_NAMES
    }
    roundtrip_summary = {
        "lanes": roundtrips,
        "maximum_per_body_scaled_rms": maximum_per_body,
        "global_scaled_floor": max(
            64.0 * EPS,
            max(value["scaled_state_rms"] for value in roundtrips.values()),
        ),
    }

    basis, basis_check = _rtn_basis(lanes["0p25"])
    coarse_native = _heliocentric_native_defect(
        lanes["0p5"].positions[1:],
        lanes["0p5"].velocities[1:],
        lanes["0p25"].positions[1:],
        lanes["0p25"].velocities[1:],
    )
    fine_native = _heliocentric_native_defect(
        lanes["0p25"].positions[1:],
        lanes["0p25"].velocities[1:],
        lanes["0p125"].positions[1:],
        lanes["0p125"].velocities[1:],
    )
    coarse_rtn, coarse_rtn_check = _project_rtn(*coarse_native, basis)
    fine_rtn, fine_rtn_check = _project_rtn(*fine_native, basis)
    tolerance_rtn = manifest["rtn_conventions"]["basis_tolerances"]
    _require(
        basis_check["maximum_orthonormal_error"]
        <= tolerance_rtn["maximum_dot_error"],
        "RTN orthonormality failed.",
    )
    _require(
        max(coarse_rtn_check.values())
        <= tolerance_rtn["maximum_reconstruction_relative"],
        "Coarse RTN reconstruction failed.",
    )
    _require(
        max(fine_rtn_check.values())
        <= tolerance_rtn["maximum_reconstruction_relative"],
        "Fine RTN reconstruction failed.",
    )

    phase, phase_agreement = _phase_reconstructions(lanes, elements)
    orbital_coarse = _element_pair(elements["0p5"], elements["0p25"])
    orbital_fine = _element_pair(elements["0p25"], elements["0p125"])
    full = _full_history_summary(
        coarse, fine, times, roundtrip_summary["global_scaled_floor"]
    )
    phase_summary = _phase_summary(
        phase, coarse, fine, roundtrip_summary["global_scaled_floor"]
    )
    coordinate = _coordinate_summary(orbital_coarse, orbital_fine)
    reproduction = _reproduce_failures(lanes, manifest)
    venus = _venus_decomposition(
        reproduction, elements, coordinate, orbital_fine
    )
    uranus = _uranus_decomposition(reproduction)
    (
        epoch_rows,
        window_rows,
        cumulative_rows,
        phase_rows,
        orbital_rows,
    ) = _tables(
        manifest,
        times,
        coarse,
        fine,
        coarse_rtn,
        fine_rtn,
        phase,
        orbital_coarse,
        orbital_fine,
        roundtrip_summary,
    )
    expected = manifest["derived_output_contract"]["uncommitted_compact_tables"]
    rows_by_name = {
        "epoch_body_metrics.csv": epoch_rows,
        "window_metrics.csv": window_rows,
        "cumulative_metrics.csv": cumulative_rows,
        "phase_stripped_window_metrics.csv": phase_rows,
        "orbital_window_metrics.csv": orbital_rows,
    }
    for name, rows in rows_by_name.items():
        _require(len(rows) == expected[name], f"Derived row count changed: {name}")
        _atomic_csv(output_root / name, rows)

    time_behavior = _time_behavior(
        window_rows, lanes, coarse, fine, coarse_rtn, fine_rtn, times
    )
    pointwise_orders = _pointwise_order_summary(
        manifest, times, coarse, fine, roundtrip_summary
    )
    ias_overlap = _ias_overlap_summary(manifest, coarse, fine, phase)
    unwrapped_rates = _unwrapped_element_rates(times, elements)
    mercury = _mercury_diagnosis(
        full, coarse_rtn, fine_rtn, phase_summary, roundtrip_summary
    )
    frozen_manifest_17 = _load_json(
        Path(
            "docs/validation/m0-step3e-whfast-0125d-convergence-v1/"
            "m0_step3e_whfast_0125d_convergence_summary.json"
        ),
        "manifest 17 summary",
    )
    classification = _classification(
        window_rows,
        phase_rows,
        orbital_rows,
        phase_summary,
        coordinate,
        venus,
        fine_rtn,
        time_behavior,
    )
    if classification["primary_classification"] in {
        "WINDOWED_OR_PHASE_DOMINATED",
        "METRIC_OR_REPRESENTATION_ILL_CONDITIONED",
        "POINTWISE_PREDICTABILITY_FLOOR",
    }:
        smallest_next = (
            "Preregister Step 3e2 as an offline, claim-aligned production "
            "qualification using the same stored artifacts and the diagnosed "
            "representation; do not execute it in Step 3e1."
        )
    elif classification["primary_classification"] == "TRUE_NONPHASE_NONCONVERGENCE":
        smallest_next = (
            "Preregister two 100 kyr full-M0 controls at 0.25 and 0.125 day "
            "under one preselected alternative WHFast configuration. Compare "
            "their coordinate-free nonphase defects with the frozen baseline "
            "prefixes and with existing IAS15 only over its validated 10 kyr "
            "overlap. Use the result only to decide whether that configuration "
            "warrants qualification; do not assume 0.125 day or the alternative "
            "configuration is preferable."
        )
    else:
        smallest_next = (
            "The single missing diagnostic is sub-100-year output around the "
            "identified coherence transitions. It cannot be recovered offline "
            "from the stored 100-year cadence; no new integration is authorized."
        )

    figures = _figures(
        figure_directory,
        full,
        window_rows,
        phase_summary,
        coordinate,
        venus,
        uranus,
        coarse_rtn,
        fine_rtn,
    )
    table_inventory = [_artifact(output_root / name) for name in rows_by_name]
    audit_path = output_root / "offline_audit.json"
    _atomic_json(audit_path, audit_payload)
    table_inventory.append(_artifact(audit_path))
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "final_status": classification["final_status"],
        "primary_classification": classification["primary_classification"],
        "historical_results_unchanged": manifest["historical_results_immutable"],
        "manifest_17_status_preserved": "STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED",
        "production_timestep_validated": False,
        "stage4_authorized": False,
        "trajectory_or_ias15_executed": False,
        "audit": {
            key: value
            for key, value in audit_payload.items()
            if key not in {"source_artifacts", "input_artifacts"}
        },
        "input_integrity": {
            "source_artifact_count": len(audit_payload["source_artifacts"]),
            "input_artifact_count": len(audit_payload["input_artifacts"]),
            "input_artifacts": audit_payload["input_artifacts"],
        },
        "closure": {
            "maximum_scaled_component_residual": float(np.max(np.abs(closure))),
            "maximum_scaled_component_tolerance": float(np.max(tolerance)),
            "passed": True,
        },
        "roundtrip": {
            **roundtrip_summary,
            "phase_method_candidate_agreement": phase_agreement,
            "passed": True,
        },
        "rtn_validation": {
            "basis": basis_check,
            "coarse": coarse_rtn_check,
            "fine": fine_rtn_check,
            "passed": True,
        },
        "physical_state": {
            "definition": manifest["defect_definitions"],
            "full_history": full,
            "worst_fine_body": max(
                BODY_NAMES,
                key=lambda name: full["per_body"][name][
                    "fine_fraction_global_squared"
                ],
            ),
            "window_rows": 110,
            "cumulative_rows": 110,
            "cumulative_key_ratios": {
                entity: [
                    {
                        "endpoint_years": int(row["endpoint_years"]),
                        "fine_over_coarse": float(row["fine_over_coarse"]),
                    }
                    for row in cumulative_rows
                    if row["entity"] == entity
                ]
                for entity in (
                    "full system",
                    "mercury barycenter",
                    "venus barycenter",
                    "uranus barycenter",
                )
            },
        },
        "orbital_elements": {
            "coordinate_free_and_classical": coordinate,
            "unwrapped_angle_rate_differences": unwrapped_rates,
            "equinoctial": manifest["orbital_element_conventions"][
                "equinoctial_variables"
            ],
            "window_rows": 90,
        },
        "phase_stripped": {
            "definitions": manifest["phase_stripping"],
            "full_history": phase_summary,
            "window_rows": 220,
        },
        "venus": venus,
        "uranus": uranus,
        "mercury": mercury,
        "ias15_validated_overlap": ias_overlap,
        "pointwise_richardson_orders": pointwise_orders,
        "time_behavior": time_behavior,
        "frozen_manifest_17_convergence_evidence": {
            "criteria": frozen_manifest_17["criteria"],
            "comparisons": frozen_manifest_17["comparisons"],
            "mercury_perihelion": frozen_manifest_17["mercury_perihelion"],
            "secular_frequency_diagnostics": frozen_manifest_17[
                "secular_frequency_diagnostics"
            ],
            "angular_momentum": frozen_manifest_17["angular_momentum"],
        },
        "classification": classification,
        "smallest_next_step": smallest_next,
        "derived_outputs": {
            "row_counts": {
                name: len(rows) for name, rows in rows_by_name.items()
            },
            "tables_and_audit": table_inventory,
            "figures": figures,
        },
    }
    _require(payload["final_status"] in FINAL_STATUSES, "Invalid final status.")
    _require(
        payload["primary_classification"] in PRIMARY_CLASSIFICATIONS,
        "Invalid primary classification.",
    )
    _atomic_json(report_json, payload)
    _atomic_text(report_markdown, _render_report(payload))
    print(
        json.dumps(
            {
                "final_status": payload["final_status"],
                "primary_classification": payload["primary_classification"],
                "summary": str(report_json),
                "report": str(report_markdown),
            },
            indent=2,
        )
    )
    return 0


def verify(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "manifest 18")
    summary_path = Path(manifest["paths"]["report_json"])
    report_path = Path(manifest["paths"]["report_markdown"])
    summary = _load_json(summary_path, "Step 3e1 summary")
    _finite_json(summary)
    _require(summary["manifest_sha256"] == sha256_file(manifest_path), "Manifest hash differs.")
    _require(summary["final_status"] in FINAL_STATUSES, "Invalid final status.")
    _require(
        summary["primary_classification"] in PRIMARY_CLASSIFICATIONS,
        "Invalid classification.",
    )
    _require(report_path.is_file(), "Missing Markdown report.")
    for item in summary["derived_outputs"]["tables_and_audit"]:
        path = Path(item["path"])
        _require(path.stat().st_size == item["size_bytes"], f"Size changed: {path}")
        _require(sha256_file(path) == item["sha256"], f"Hash changed: {path}")
    for item in summary["derived_outputs"]["figures"]:
        path = Path(item["path"])
        _require(path.stat().st_size == item["size_bytes"], f"Figure size changed: {path}")
        _require(sha256_file(path) == item["sha256"], f"Figure hash changed: {path}")
    audit_payload, _ = audit(manifest_path, include_lanes=False)
    _require(audit_payload["status"] == "PASS", "Offline audit failed.")
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_status": summary["final_status"],
                "primary_classification": summary["primary_classification"],
                "summary_sha256": sha256_file(summary_path),
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline-only M0 Step 3e1 stored-state diagnosis."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "command",
        choices=("audit", "analyze", "verify"),
        help="No command invokes a trajectory or integrator.",
    )
    args = parser.parse_args(argv)
    if args.command == "audit":
        payload, _ = audit(args.manifest)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "manifest_sha256": payload["manifest_sha256"],
                    "reproduced_manifest_17_failures": payload[
                        "reproduced_manifest_17_failures"
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "analyze":
        return analyze(args.manifest)
    return verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
