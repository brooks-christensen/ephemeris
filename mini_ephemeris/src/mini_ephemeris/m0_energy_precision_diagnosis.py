from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Iterator, Sequence

import numpy as np

from .m0_telemetry import STATE_SAMPLE_FIELDS
from .m0_timestep_convergence import _atomic_write_json, _atomic_write_text, _series_metrics
from .rebound_gr_tangent_backend_cli import PROGRESS_FIELDS, sha256_file


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/12_m0_energy_precision_diagnosis_v1.json"
)
FINAL_DIAGNOSES = {
    "ENERGY_TELEMETRY_PRECISION_LIMITED",
    "ENERGY_DRIFT_CONFIRMED",
    "INCONCLUSIVE",
    "BLOCKED",
}
ENERGY_FIELDS = (
    "newtonian_energy_j",
    "gr_potential_energy_j",
    "corrected_energy_j",
    "corrected_energy_rel_change",
)
STATE_NUMERIC_FIELDS = (
    "mass_kg",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
)
TIMESERIES_FIELDS = (
    "schema_version",
    "manifest_sha256",
    "lane_id",
    "sample_index",
    "time_years",
    "recorded_newtonian_j",
    "recorded_gr_potential_j",
    "recorded_corrected_j",
    "recorded_rel_change",
    "float64_kinetic_j",
    "float64_pair_potential_j",
    "float64_gr_potential_j",
    "float64_newtonian_j",
    "float64_corrected_j",
    "float64_rel_change",
    "float64_rel_change_minus_recorded",
    "float64_corrected_minus_decimal_j",
    "compensated_kinetic_j",
    "compensated_pair_potential_j",
    "compensated_gr_potential_j",
    "compensated_newtonian_j",
    "compensated_corrected_j",
    "compensated_rel_change",
    "compensated_rel_change_minus_recorded",
    "compensated_corrected_minus_decimal_j",
    "decimal_kinetic_j",
    "decimal_pair_potential_j",
    "decimal_gr_potential_j",
    "decimal_newtonian_j",
    "decimal_corrected_j",
    "decimal_rel_change",
    "decimal_rel_change_minus_recorded",
    "float64_rel_change_minus_decimal",
    "compensated_rel_change_minus_decimal",
    "float64_corrected_cancellation_factor",
    "float64_drift_subtraction_cancellation_factor",
    "compensated_corrected_cancellation_factor",
    "compensated_drift_subtraction_cancellation_factor",
    "decimal_corrected_cancellation_factor",
    "decimal_drift_subtraction_cancellation_factor",
    "float64_newtonian_abs_tolerance_j",
    "float64_gr_abs_tolerance_j",
    "float64_corrected_abs_tolerance_j",
    "float64_drift_abs_tolerance",
)


class EnergyDiagnosisError(RuntimeError):
    pass


@dataclass
class LaneResult:
    lane_id: str
    summary: dict[str, Any]
    statistics_internal: dict[str, dict[str, Any]]
    timeseries_inventory: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnergyDiagnosisError(message)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise EnergyDiagnosisError(f"Git command failed: git {' '.join(args)}") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise EnergyDiagnosisError(f"Unreadable {label} {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"Invalid {label}: expected an object.")
    return payload


def _canonical_float_token(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    return math.isfinite(value) and str(value) == token


def _artifact_inventory(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
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
        raise EnergyDiagnosisError(f"Missing artifact inventory in {label}.")
    _require(
        all(isinstance(entry, dict) for entry in entries),
        f"Invalid artifact inventory entries in {label}.",
    )
    return entries


def _verify_fixed_inputs(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Run the diagnosis at project root.")
    _require(
        manifest.get("frozen_before_diagnostic_calculation") is True,
        "Diagnostic manifest is not frozen.",
    )
    _require(
        _git("rev-parse", "HEAD") == manifest["provenance"]["starting_commit"],
        "Diagnostic must start from the closed Step 3b commit.",
    )
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C tag is not annotated.")
    _require(
        _git("rev-parse", tag + "^{commit}")
        == manifest["provenance"]["validated_c_baseline_commit"],
        "Compiled-C tag resolves to the wrong commit.",
    )
    fixed_hashes = {
        "manifest_10": manifest["provenance"]["manifest_10_sha256"],
        "manifest_11": manifest["provenance"]["manifest_11_sha256"],
        "step3_summary": manifest["provenance"]["step3_summary_sha256"],
        "step3b_summary": manifest["provenance"]["step3b_summary_sha256"],
    }
    for key, expected in fixed_hashes.items():
        path = Path(manifest["paths"][key])
        _require(sha256_file(path) == expected, f"Fixed input hash mismatch: {path}")
    historical_artifacts: dict[str, int] = {}
    unique_artifact_paths: set[str] = set()
    for key in ("step3_summary", "step3b_summary"):
        payload = _load_json(Path(manifest["paths"][key]), key)
        entries = _artifact_inventory(payload, key)
        for entry in entries:
            artifact_path = Path(entry["path"])
            _require(
                artifact_path.stat().st_size == entry["size_bytes"],
                f"Historical artifact size mismatch: {artifact_path}",
            )
            _require(
                sha256_file(artifact_path) == entry["sha256"],
                f"Historical artifact hash mismatch: {artifact_path}",
            )
            unique_artifact_paths.add(str(artifact_path))
        historical_artifacts[key] = len(entries)
    protected = []
    for relative, expected in manifest["protected_files"].items():
        path = root / relative
        actual = sha256_file(path)
        _require(actual == expected, f"Protected file hash mismatch: {relative}")
        protected.append({"path": str(path), "sha256": actual})
    lane_inputs = []
    for lane in manifest["input_lanes"]:
        state_path = Path(lane["state_path"])
        progress_path = Path(lane["progress_path"])
        _require(state_path.stat().st_size == lane["state_size_bytes"], "State size changed.")
        _require(
            progress_path.stat().st_size == lane["progress_size_bytes"],
            "Progress size changed.",
        )
        _require(sha256_file(state_path) == lane["state_sha256"], "State hash changed.")
        _require(
            sha256_file(progress_path) == lane["progress_sha256"],
            "Progress hash changed.",
        )
        lane_inputs.append(
            {
                "id": lane["id"],
                "state_path": str(state_path),
                "state_sha256": lane["state_sha256"],
                "progress_path": str(progress_path),
                "progress_sha256": lane["progress_sha256"],
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protected_files": protected,
        "historical_artifact_inventory": {
            "entries_verified": historical_artifacts,
            "unique_artifacts_verified": len(unique_artifact_paths),
        },
        "lane_inputs": lane_inputs,
    }


def _audit_state_csv(path: Path, manifest: dict[str, Any], lane: dict[str, Any]) -> dict[str, Any]:
    expected_names = manifest["state_schema"]["body_order"]
    expected_samples = manifest["state_schema"]["expected_samples_per_lane"]
    expected_rows = manifest["state_schema"]["expected_state_rows_per_lane"]
    masses: dict[int, str] = {}
    canonical_tokens = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == STATE_SAMPLE_FIELDS, f"State schema mismatch: {path}")
        count = 0
        for row in reader:
            sample_index, body_index = divmod(count, len(expected_names))
            _require(int(row["sample_index"]) == sample_index, "State sample sequence mismatch.")
            _require(int(row["body_index"]) == body_index, "State body sequence mismatch.")
            _require(row["body_name"] == expected_names[body_index], "State body name mismatch.")
            _require(
                row["configuration_fingerprint"] == lane["configuration_fingerprint"],
                "State fingerprint mismatch.",
            )
            _require(float(row["time_years"]) == sample_index * 100.0, "State year mismatch.")
            _require(
                float(row["time_seconds"]) == sample_index * 100.0 * 31_557_600.0,
                "State seconds mismatch.",
            )
            for field in STATE_NUMERIC_FIELDS:
                _require(
                    _canonical_float_token(row[field]),
                    f"State token is not canonical round-trip float64: {field}={row[field]}",
                )
                canonical_tokens += 1
            masses.setdefault(body_index, row["mass_kg"])
            _require(masses[body_index] == row["mass_kg"], "Particle mass token changed.")
            count += 1
    _require(count == expected_rows, f"State row count mismatch: {count}")
    _require(count // len(expected_names) == expected_samples, "State sample count mismatch.")
    return {
        "rows": count,
        "samples": count // len(expected_names),
        "canonical_roundtrip_tokens": canonical_tokens,
        "body_order": expected_names,
        "mass_tokens": [masses[index] for index in range(len(expected_names))],
    }


def _load_progress(path: Path, manifest: dict[str, Any], lane: dict[str, Any]) -> list[dict[str, str]]:
    expected = manifest["state_schema"]["expected_samples_per_lane"]
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == PROGRESS_FIELDS, f"Progress schema mismatch: {path}")
        for index, row in enumerate(reader):
            _require(index < expected, "Too many progress rows.")
            _require(float(row["time_years"]) == index * 100.0, "Progress year mismatch.")
            _require(
                float(row["time_seconds"]) == index * 100.0 * 31_557_600.0,
                "Progress seconds mismatch.",
            )
            _require(
                row["configuration_fingerprint"] == lane["configuration_fingerprint"],
                "Progress fingerprint mismatch.",
            )
            for field in ENERGY_FIELDS:
                _require(
                    _canonical_float_token(row[field]),
                    f"Progress token is not canonical float64: {field}={row[field]}",
                )
            rows.append(row)
    _require(len(rows) == expected, "Progress sample count mismatch.")
    return rows


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "diagnostic manifest")
    payload = _verify_fixed_inputs(manifest_path, manifest)
    payload["status"] = "PASS"
    payload["state_csv"] = {}
    for lane in manifest["input_lanes"]:
        payload["state_csv"][lane["id"]] = _audit_state_csv(
            Path(lane["state_path"]), manifest, lane
        )
        _load_progress(Path(lane["progress_path"]), manifest, lane)
    return payload


def _iter_state_groups(
    path: Path, manifest: dict[str, Any], lane: dict[str, Any]
) -> Iterator[list[dict[str, str]]]:
    names = manifest["state_schema"]["body_order"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == STATE_SAMPLE_FIELDS, "State schema changed after audit.")
        group: list[dict[str, str]] = []
        for row in reader:
            group.append(row)
            if len(group) == len(names):
                sample_index = int(group[0]["sample_index"])
                _require(
                    [item["body_name"] for item in group] == names,
                    "State particle order changed after audit.",
                )
                _require(
                    all(int(item["sample_index"]) == sample_index for item in group),
                    "State group has mixed sample indices.",
                )
                yield group
                group = []
        _require(not group, "State CSV ends with an incomplete group.")


def float64_energy(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    gravitational_constant: float,
    speed_of_light: float,
    coefficient_scale: float,
) -> dict[str, float]:
    kinetic = 0.5 * float(np.sum(masses[:, np.newaxis] * velocities * velocities))
    pair_potential = 0.0
    pair_abs_sum = 0.0
    for i in range(len(masses) - 1):
        for j in range(i + 1, len(masses)):
            radius = float(np.linalg.norm(positions[j] - positions[i]))
            _require(radius != 0.0, "Coincident particles in Newtonian potential.")
            term = (
                -gravitational_constant
                * float(masses[i])
                * float(masses[j])
                / radius
            )
            pair_potential += term
            pair_abs_sum += abs(term)
    weighted_inverse_r2 = 0.0
    for body_index in range(1, len(masses)):
        displacement = positions[body_index] - positions[0]
        radius_squared = float(np.dot(displacement, displacement))
        _require(radius_squared != 0.0, "Coincident particle in GR potential.")
        weighted_inverse_r2 += float(masses[body_index]) / radius_squared
    prefactor = (
        -3.0
        * coefficient_scale
        * gravitational_constant**2
        * float(masses[0]) ** 2
        / speed_of_light**2
    )
    gr_potential = float(prefactor * weighted_inverse_r2)
    newtonian = float(kinetic + pair_potential)
    corrected = float(newtonian + gr_potential)
    return {
        "kinetic": kinetic,
        "pair_potential": pair_potential,
        "gr_potential": gr_potential,
        "newtonian": newtonian,
        "corrected": corrected,
        "newtonian_term_abs_sum": abs(kinetic) + pair_abs_sum,
        "gr_term_abs_sum": abs(prefactor) * sum(
            float(masses[index])
            / float(np.dot(positions[index] - positions[0], positions[index] - positions[0]))
            for index in range(1, len(masses))
        ),
    }


def compensated_energy(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    gravitational_constant: float,
    speed_of_light: float,
    coefficient_scale: float,
) -> dict[str, float]:
    kinetic_terms = [
        0.5 * float(masses[i]) * float(velocities[i, axis]) ** 2
        for i in range(len(masses))
        for axis in range(3)
    ]
    pair_terms = []
    for i in range(len(masses) - 1):
        for j in range(i + 1, len(masses)):
            delta = positions[j] - positions[i]
            radius = math.sqrt(math.fsum(float(value) ** 2 for value in delta))
            _require(radius != 0.0, "Coincident particles in compensated potential.")
            pair_terms.append(
                -gravitational_constant
                * float(masses[i])
                * float(masses[j])
                / radius
            )
    prefactor = (
        -3.0
        * coefficient_scale
        * gravitational_constant**2
        * float(masses[0]) ** 2
        / speed_of_light**2
    )
    gr_terms = []
    for index in range(1, len(masses)):
        delta = positions[index] - positions[0]
        radius_squared = math.fsum(float(value) ** 2 for value in delta)
        _require(radius_squared != 0.0, "Coincident particle in compensated GR potential.")
        gr_terms.append(prefactor * float(masses[index]) / radius_squared)
    kinetic = math.fsum(kinetic_terms)
    pair_potential = math.fsum(pair_terms)
    gr_potential = math.fsum(gr_terms)
    newtonian = math.fsum((kinetic, pair_potential))
    corrected = math.fsum((newtonian, gr_potential))
    return {
        "kinetic": kinetic,
        "pair_potential": pair_potential,
        "gr_potential": gr_potential,
        "newtonian": newtonian,
        "corrected": corrected,
        "newtonian_term_abs_sum": math.fsum(abs(value) for value in kinetic_terms + pair_terms),
        "gr_term_abs_sum": math.fsum(abs(value) for value in gr_terms),
    }


def decimal_energy(
    rows: Sequence[dict[str, str]],
    *,
    gravitational_constant: Decimal,
    speed_of_light: Decimal,
    coefficient_scale: Decimal,
) -> dict[str, Decimal]:
    masses = [Decimal(row["mass_kg"]) for row in rows]
    positions = [
        (Decimal(row["x_m"]), Decimal(row["y_m"]), Decimal(row["z_m"]))
        for row in rows
    ]
    velocities = [
        (
            Decimal(row["vx_m_per_s"]),
            Decimal(row["vy_m_per_s"]),
            Decimal(row["vz_m_per_s"]),
        )
        for row in rows
    ]
    half = Decimal("0.5")
    kinetic_terms = [
        half * masses[i] * velocities[i][axis] * velocities[i][axis]
        for i in range(len(masses))
        for axis in range(3)
    ]
    pair_terms = []
    for i in range(len(masses) - 1):
        for j in range(i + 1, len(masses)):
            delta = tuple(positions[j][axis] - positions[i][axis] for axis in range(3))
            radius = sum((value * value for value in delta), Decimal(0)).sqrt()
            _require(radius != 0, "Coincident particles in Decimal potential.")
            pair_terms.append(-gravitational_constant * masses[i] * masses[j] / radius)
    prefactor = (
        -Decimal(3)
        * coefficient_scale
        * gravitational_constant
        * gravitational_constant
        * masses[0]
        * masses[0]
        / (speed_of_light * speed_of_light)
    )
    gr_terms = []
    for index in range(1, len(masses)):
        delta = tuple(positions[index][axis] - positions[0][axis] for axis in range(3))
        radius_squared = sum((value * value for value in delta), Decimal(0))
        _require(radius_squared != 0, "Coincident particle in Decimal GR potential.")
        gr_terms.append(prefactor * masses[index] / radius_squared)
    kinetic = sum(kinetic_terms, Decimal(0))
    pair_potential = sum(pair_terms, Decimal(0))
    gr_potential = sum(gr_terms, Decimal(0))
    newtonian = kinetic + pair_potential
    corrected = newtonian + gr_potential
    return {
        "kinetic": kinetic,
        "pair_potential": pair_potential,
        "gr_potential": gr_potential,
        "newtonian": newtonian,
        "corrected": corrected,
        "newtonian_term_abs_sum": sum((abs(value) for value in kinetic_terms + pair_terms), Decimal(0)),
        "gr_term_abs_sum": sum((abs(value) for value in gr_terms), Decimal(0)),
    }


def decimal_statistics(times: Sequence[Decimal], values: Sequence[Decimal]) -> dict[str, Any]:
    _require(len(times) == len(values) and len(values) >= 2, "Invalid Decimal history.")
    absolute = [abs(value) for value in values]
    maximum = max(absolute)
    worst_index = absolute.index(maximum)
    count = Decimal(len(values))
    rms = (sum((value * value for value in values), Decimal(0)) / count).sqrt()
    p99 = sorted(absolute)[9900]
    mean_time = sum(times, Decimal(0)) / count
    mean_value = sum(values, Decimal(0)) / count
    centered_times = [value - mean_time for value in times]
    denominator = sum((value * value for value in centered_times), Decimal(0))
    slope = sum(
        (
            centered_times[index] * (values[index] - mean_value)
            for index in range(len(values))
        ),
        Decimal(0),
    ) / denominator
    return {
        "max_abs": maximum,
        "max_abs_worst_epoch_years": times[worst_index],
        "rms": rms,
        "p99_abs": p99,
        "fitted_trend_per_year": slope,
        "fitted_change_over_1myr": slope * Decimal(1_000_000),
        "final": values[-1],
    }


def _decimal_stats_json(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {"decimal": str(value), "float": float(value)}
            if isinstance(value, Decimal)
            else value
        )
        for key, value in stats.items()
    }


def _update_worst(
    tracker: dict[str, Any], value: float, epoch: float, *, absolute: bool = True
) -> None:
    score = abs(value) if absolute else value
    if tracker.get("score", -math.inf) < score:
        tracker.update(score=score, value=value, worst_epoch_years=epoch)


def _drift(total: float, reference: float) -> float:
    scale = abs(reference) if reference != 0.0 else 1.0
    return (total - reference) / scale


def _decimal_drift(total: Decimal, reference: Decimal) -> Decimal:
    scale = abs(reference) if reference != 0 else Decimal(1)
    return (total - reference) / scale


def _lane_diagnosis(
    manifest_path: Path, manifest: dict[str, Any], lane: dict[str, Any]
) -> LaneResult:
    progress = _load_progress(Path(lane["progress_path"]), manifest, lane)
    output_path = Path(lane["timeseries_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    _require(not output_path.exists() and not temporary.exists(), f"Output already exists: {output_path}")

    constants = manifest["authoritative_constants"]
    g_float = float(constants["gravitational_constant_decimal"])
    c_float = float(constants["speed_of_light_decimal"])
    scale_float = float(constants["coefficient_scale_decimal"])
    gamma = float(manifest["numeric_tolerances"]["gamma_1024"])
    unit_roundoff = float(constants["float64_unit_roundoff"])
    safety = float(manifest["numeric_tolerances"]["raw_energy_safety_factor"])
    method_agreement = float(
        manifest["numeric_tolerances"]["method_history_agreement_abs"]
    )
    times_float: list[float] = []
    times_decimal: list[Decimal] = []
    histories_float: dict[str, list[float]] = {
        "recorded": [],
        "float64": [],
        "compensated": [],
    }
    decimal_history: list[Decimal] = []
    references_float: dict[str, float] = {}
    decimal_reference: Decimal | None = None
    tolerance_reference = 0.0
    telemetry_violations = 0
    trackers: dict[str, dict[str, Any]] = {
        name: {}
        for name in (
            "float_minus_recorded_drift",
            "compensated_minus_recorded_drift",
            "decimal_minus_recorded_drift",
            "float_minus_decimal_drift",
            "compensated_minus_decimal_drift",
            "telemetry_tolerance_ratio",
            "float64_corrected_cancellation",
            "compensated_corrected_cancellation",
            "decimal_corrected_cancellation",
            "float64_drift_cancellation",
            "compensated_drift_cancellation",
            "decimal_drift_cancellation",
        )
    }
    manifest_hash = sha256_file(manifest_path)

    with localcontext() as context:
        context.prec = int(constants["decimal_precision_digits"])
        g_decimal = Decimal(constants["gravitational_constant_decimal"])
        c_decimal = Decimal(constants["speed_of_light_decimal"])
        scale_decimal = Decimal(constants["coefficient_scale_decimal"])
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS)
            writer.writeheader()
            groups = _iter_state_groups(Path(lane["state_path"]), manifest, lane)
            for sample_index, rows in enumerate(groups):
                progress_row = progress[sample_index]
                time_token = rows[0]["time_years"]
                time_years = float(time_token)
                time_decimal = Decimal(time_token)
                masses = np.asarray([float(row["mass_kg"]) for row in rows], dtype=np.float64)
                positions = np.asarray(
                    [
                        [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                velocities = np.asarray(
                    [
                        [
                            float(row["vx_m_per_s"]),
                            float(row["vy_m_per_s"]),
                            float(row["vz_m_per_s"]),
                        ]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                energy_float = float64_energy(
                    masses,
                    positions,
                    velocities,
                    gravitational_constant=g_float,
                    speed_of_light=c_float,
                    coefficient_scale=scale_float,
                )
                energy_compensated = compensated_energy(
                    masses,
                    positions,
                    velocities,
                    gravitational_constant=g_float,
                    speed_of_light=c_float,
                    coefficient_scale=scale_float,
                )
                energy_decimal = decimal_energy(
                    rows,
                    gravitational_constant=g_decimal,
                    speed_of_light=c_decimal,
                    coefficient_scale=scale_decimal,
                )
                for method, energy in (
                    ("float64", energy_float),
                    ("compensated", energy_compensated),
                ):
                    _require(
                        all(math.isfinite(value) for value in energy.values()),
                        f"Nonfinite {method} energy at sample {sample_index}.",
                    )
                _require(
                    all(value.is_finite() for value in energy_decimal.values()),
                    f"Nonfinite Decimal energy at sample {sample_index}.",
                )
                newton_tolerance = (
                    safety * gamma * energy_float["newtonian_term_abs_sum"]
                )
                gr_tolerance = safety * gamma * energy_float["gr_term_abs_sum"]
                corrected_tolerance = (
                    newton_tolerance
                    + gr_tolerance
                    + safety
                    * unit_roundoff
                    * (abs(energy_float["newtonian"]) + abs(energy_float["gr_potential"]))
                )
                if sample_index == 0:
                    references_float = {
                        "float64": energy_float["corrected"],
                        "compensated": energy_compensated["corrected"],
                    }
                    decimal_reference = energy_decimal["corrected"]
                    tolerance_reference = corrected_tolerance
                _require(decimal_reference is not None, "Decimal reference was not initialized.")
                drift_float = _drift(energy_float["corrected"], references_float["float64"])
                drift_compensated = _drift(
                    energy_compensated["corrected"], references_float["compensated"]
                )
                drift_decimal = _decimal_drift(energy_decimal["corrected"], decimal_reference)
                _require(
                    math.isfinite(drift_float)
                    and math.isfinite(drift_compensated)
                    and drift_decimal.is_finite(),
                    f"Nonfinite energy drift at sample {sample_index}.",
                )
                recorded = {field: float(progress_row[field]) for field in ENERGY_FIELDS}
                recorded_decimal_drift = Decimal(progress_row["corrected_energy_rel_change"])
                drift_tolerance = (
                    (corrected_tolerance + tolerance_reference)
                    / abs(references_float["float64"])
                    + 16.0 * gamma
                )
                raw_checks = (
                    (
                        energy_float["newtonian"] - recorded["newtonian_energy_j"],
                        newton_tolerance,
                    ),
                    (
                        energy_float["gr_potential"] - recorded["gr_potential_energy_j"],
                        gr_tolerance,
                    ),
                    (
                        energy_float["corrected"] - recorded["corrected_energy_j"],
                        corrected_tolerance,
                    ),
                    (
                        drift_float - recorded["corrected_energy_rel_change"],
                        drift_tolerance,
                    ),
                )
                sample_ratio = 0.0
                for difference, tolerance in raw_checks:
                    ratio = abs(difference) / tolerance if tolerance > 0.0 else math.inf
                    sample_ratio = max(sample_ratio, ratio)
                    if abs(difference) > tolerance:
                        telemetry_violations += 1
                _update_worst(
                    trackers["telemetry_tolerance_ratio"],
                    sample_ratio,
                    time_years,
                    absolute=False,
                )
                float_minus_decimal = Decimal.from_float(drift_float) - drift_decimal
                compensated_minus_decimal = (
                    Decimal.from_float(drift_compensated) - drift_decimal
                )
                float_minus_recorded = (
                    drift_float - recorded["corrected_energy_rel_change"]
                )
                compensated_minus_recorded = (
                    drift_compensated - recorded["corrected_energy_rel_change"]
                )
                decimal_minus_recorded = drift_decimal - recorded_decimal_drift
                _update_worst(
                    trackers["float_minus_recorded_drift"],
                    float_minus_recorded,
                    time_years,
                )
                _update_worst(
                    trackers["compensated_minus_recorded_drift"],
                    compensated_minus_recorded,
                    time_years,
                )
                _update_worst(
                    trackers["decimal_minus_recorded_drift"],
                    float(decimal_minus_recorded),
                    time_years,
                )
                _update_worst(
                    trackers["float_minus_decimal_drift"],
                    float(float_minus_decimal),
                    time_years,
                )
                _update_worst(
                    trackers["compensated_minus_decimal_drift"],
                    float(compensated_minus_decimal),
                    time_years,
                )
                float_cancellation = (
                    energy_float["newtonian_term_abs_sum"]
                    + energy_float["gr_term_abs_sum"]
                ) / max(abs(energy_float["corrected"]), 1.0)
                compensated_cancellation = (
                    energy_compensated["newtonian_term_abs_sum"]
                    + energy_compensated["gr_term_abs_sum"]
                ) / max(abs(energy_compensated["corrected"]), 1.0)
                decimal_cancellation = (
                    energy_decimal["newtonian_term_abs_sum"]
                    + energy_decimal["gr_term_abs_sum"]
                ) / max(abs(energy_decimal["corrected"]), Decimal(1))
                float_drift_cancellation = (
                    None
                    if energy_float["corrected"] == references_float["float64"]
                    else (
                        abs(energy_float["corrected"])
                        + abs(references_float["float64"])
                    )
                    / abs(
                        energy_float["corrected"] - references_float["float64"]
                    )
                )
                compensated_drift_cancellation = (
                    None
                    if energy_compensated["corrected"]
                    == references_float["compensated"]
                    else (
                        abs(energy_compensated["corrected"])
                        + abs(references_float["compensated"])
                    )
                    / abs(
                        energy_compensated["corrected"]
                        - references_float["compensated"]
                    )
                )
                decimal_drift_cancellation = (
                    None
                    if energy_decimal["corrected"] == decimal_reference
                    else (
                        abs(energy_decimal["corrected"]) + abs(decimal_reference)
                    )
                    / abs(energy_decimal["corrected"] - decimal_reference)
                )
                composition_cancellations = {
                    "float64": float_cancellation,
                    "compensated": compensated_cancellation,
                    "decimal": float(decimal_cancellation),
                }
                drift_cancellations = {
                    "float64": float_drift_cancellation,
                    "compensated": compensated_drift_cancellation,
                    "decimal": (
                        float(decimal_drift_cancellation)
                        if decimal_drift_cancellation is not None
                        else None
                    ),
                }
                for method, value in composition_cancellations.items():
                    _update_worst(
                        trackers[f"{method}_corrected_cancellation"],
                        value,
                        time_years,
                        absolute=False,
                    )
                for method, value in drift_cancellations.items():
                    if value is not None:
                        _update_worst(
                            trackers[f"{method}_drift_cancellation"],
                            value,
                            time_years,
                            absolute=False,
                        )
                times_float.append(time_years)
                times_decimal.append(time_decimal)
                histories_float["recorded"].append(
                    recorded["corrected_energy_rel_change"]
                )
                histories_float["float64"].append(drift_float)
                histories_float["compensated"].append(drift_compensated)
                decimal_history.append(drift_decimal)
                writer.writerow(
                    {
                        "schema_version": manifest["output_policy"]["timeseries_schema_version"],
                        "manifest_sha256": manifest_hash,
                        "lane_id": lane["id"],
                        "sample_index": sample_index,
                        "time_years": time_token,
                        "recorded_newtonian_j": progress_row["newtonian_energy_j"],
                        "recorded_gr_potential_j": progress_row["gr_potential_energy_j"],
                        "recorded_corrected_j": progress_row["corrected_energy_j"],
                        "recorded_rel_change": progress_row["corrected_energy_rel_change"],
                        "float64_kinetic_j": energy_float["kinetic"],
                        "float64_pair_potential_j": energy_float["pair_potential"],
                        "float64_gr_potential_j": energy_float["gr_potential"],
                        "float64_newtonian_j": energy_float["newtonian"],
                        "float64_corrected_j": energy_float["corrected"],
                        "float64_rel_change": drift_float,
                        "float64_rel_change_minus_recorded": float_minus_recorded,
                        "float64_corrected_minus_decimal_j": str(
                            Decimal.from_float(energy_float["corrected"])
                            - energy_decimal["corrected"]
                        ),
                        "compensated_kinetic_j": energy_compensated["kinetic"],
                        "compensated_pair_potential_j": energy_compensated[
                            "pair_potential"
                        ],
                        "compensated_gr_potential_j": energy_compensated["gr_potential"],
                        "compensated_newtonian_j": energy_compensated["newtonian"],
                        "compensated_corrected_j": energy_compensated["corrected"],
                        "compensated_rel_change": drift_compensated,
                        "compensated_rel_change_minus_recorded": compensated_minus_recorded,
                        "compensated_corrected_minus_decimal_j": str(
                            Decimal.from_float(energy_compensated["corrected"])
                            - energy_decimal["corrected"]
                        ),
                        "decimal_kinetic_j": str(energy_decimal["kinetic"]),
                        "decimal_pair_potential_j": str(energy_decimal["pair_potential"]),
                        "decimal_gr_potential_j": str(energy_decimal["gr_potential"]),
                        "decimal_newtonian_j": str(energy_decimal["newtonian"]),
                        "decimal_corrected_j": str(energy_decimal["corrected"]),
                        "decimal_rel_change": str(drift_decimal),
                        "decimal_rel_change_minus_recorded": str(decimal_minus_recorded),
                        "float64_rel_change_minus_decimal": str(float_minus_decimal),
                        "compensated_rel_change_minus_decimal": str(
                            compensated_minus_decimal
                        ),
                        "float64_corrected_cancellation_factor": float_cancellation,
                        "float64_drift_subtraction_cancellation_factor": (
                            float_drift_cancellation
                            if float_drift_cancellation is not None
                            else ""
                        ),
                        "compensated_corrected_cancellation_factor": (
                            compensated_cancellation
                        ),
                        "compensated_drift_subtraction_cancellation_factor": (
                            compensated_drift_cancellation
                            if compensated_drift_cancellation is not None
                            else ""
                        ),
                        "decimal_corrected_cancellation_factor": str(decimal_cancellation),
                        "decimal_drift_subtraction_cancellation_factor": (
                            str(decimal_drift_cancellation)
                            if decimal_drift_cancellation is not None
                            else ""
                        ),
                        "float64_newtonian_abs_tolerance_j": newton_tolerance,
                        "float64_gr_abs_tolerance_j": gr_tolerance,
                        "float64_corrected_abs_tolerance_j": corrected_tolerance,
                        "float64_drift_abs_tolerance": drift_tolerance,
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)

        statistics_float = {
            method: _series_metrics(
                np.asarray(times_float, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
            for method, values in histories_float.items()
        }
        statistics_decimal = decimal_statistics(times_decimal, decimal_history)
    agreement_ok = (
        trackers["compensated_minus_decimal_drift"]["score"] <= method_agreement
    )
    summary = {
        "step_days": lane["step_days"],
        "samples": len(times_float),
        "state_rows": len(times_float) * manifest["state_schema"]["real_particle_count"],
        "telemetry_reproduction": {
            "passed": telemetry_violations == 0,
            "violation_count": telemetry_violations,
            "worst_tolerance_ratio": trackers["telemetry_tolerance_ratio"],
            "float64_minus_recorded_drift": trackers["float_minus_recorded_drift"],
        },
        "method_agreement": {
            "passed": agreement_ok,
            "absolute_drift_limit": method_agreement,
            "float64_minus_decimal_drift": trackers["float_minus_decimal_drift"],
            "compensated_minus_decimal_drift": trackers[
                "compensated_minus_decimal_drift"
            ],
            "compensated_minus_recorded_drift": trackers[
                "compensated_minus_recorded_drift"
            ],
            "decimal_minus_recorded_drift": trackers["decimal_minus_recorded_drift"],
        },
        "conditioning": {
            "worst_corrected_composition_cancellation": {
                method: trackers[f"{method}_corrected_cancellation"]
                for method in ("float64", "compensated", "decimal")
            },
            "worst_drift_subtraction_cancellation": {
                method: trackers[f"{method}_drift_cancellation"]
                for method in ("float64", "compensated", "decimal")
            },
        },
        "statistics": {
            **statistics_float,
            "decimal": _decimal_stats_json(statistics_decimal),
        },
    }
    inventory = {
        "path": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }
    return LaneResult(
        lane_id=lane["id"],
        summary=summary,
        statistics_internal={
            **statistics_float,
            "decimal": statistics_decimal,
        },
        timeseries_inventory=inventory,
    )


def classify_diagnosis(
    manifest: dict[str, Any], lane_results: dict[str, LaneResult]
) -> tuple[str, dict[str, Any]]:
    coarse = lane_results[manifest["input_lanes"][0]["id"]]
    fine = lane_results[manifest["input_lanes"][1]["id"]]
    telemetry_ok = all(
        result.summary["telemetry_reproduction"]["passed"]
        for result in lane_results.values()
    )
    agreement_ok = all(
        result.summary["method_agreement"]["passed"] for result in lane_results.values()
    )
    margin = Decimal(str(manifest["numeric_tolerances"]["lane_worsening_significance_margin"]))
    metrics = ("max_abs", "rms", "p99_abs")
    decimal_differences = {
        metric: fine.statistics_internal["decimal"][metric]
        - coarse.statistics_internal["decimal"][metric]
        for metric in metrics
    }
    compensated_differences = {
        metric: Decimal.from_float(fine.statistics_internal["compensated"][metric])
        - Decimal.from_float(coarse.statistics_internal["compensated"][metric])
        for metric in metrics
    }
    decimal_trend_difference = abs(
        fine.statistics_internal["decimal"]["fitted_change_over_1myr"]
    ) - abs(coarse.statistics_internal["decimal"]["fitted_change_over_1myr"])
    compensated_trend_difference = Decimal.from_float(
        abs(fine.statistics_internal["compensated"]["fitted_change_over_1myr"])
        - abs(coarse.statistics_internal["compensated"]["fitted_change_over_1myr"])
    )
    trend_floor = Decimal("1e-10")
    energy_bound = Decimal(str(manifest["statistics"]["unchanged_energy_bound"]))

    decimal_trend_pass = {}
    decimal_bound_pass = {}
    for result in lane_results.values():
        stats = result.statistics_internal["decimal"]
        decimal_trend_pass[result.lane_id] = abs(stats["fitted_change_over_1myr"]) <= max(
            Decimal("0.25") * stats["max_abs"], trend_floor
        )
        decimal_bound_pass[result.lane_id] = stats["max_abs"] <= energy_bound
    decimal_nonincreasing = all(value <= 0 for value in decimal_differences.values())
    decimal_precision_limited = (
        agreement_ok
        and decimal_nonincreasing
        and all(decimal_trend_pass.values())
        and all(decimal_bound_pass.values())
    )
    confirmed = (
        agreement_ok
        and all(value > margin for value in decimal_differences.values())
        and all(value > margin for value in compensated_differences.values())
        and decimal_trend_difference > margin
        and compensated_trend_difference > margin
    )
    if not telemetry_ok:
        diagnosis = "BLOCKED"
    elif decimal_precision_limited:
        diagnosis = "ENERGY_TELEMETRY_PRECISION_LIMITED"
    elif confirmed:
        diagnosis = "ENERGY_DRIFT_CONFIRMED"
    else:
        diagnosis = "INCONCLUSIVE"
    evidence = {
        "telemetry_reproduction_passed": telemetry_ok,
        "compensated_decimal_agreement_passed": agreement_ok,
        "lane_worsening_significance_margin": float(margin),
        "decimal_fine_minus_coarse": {
            key: {"decimal": str(value), "float": float(value)}
            for key, value in decimal_differences.items()
        },
        "compensated_fine_minus_coarse": {
            key: {"decimal": str(value), "float": float(value)}
            for key, value in compensated_differences.items()
        },
        "decimal_abs_trend_fine_minus_coarse": {
            "decimal": str(decimal_trend_difference),
            "float": float(decimal_trend_difference),
        },
        "compensated_abs_trend_fine_minus_coarse": {
            "decimal": str(compensated_trend_difference),
            "float": float(compensated_trend_difference),
        },
        "decimal_trend_pass": decimal_trend_pass,
        "decimal_bound_pass": decimal_bound_pass,
        "decimal_nonincreasing": decimal_nonincreasing,
        "precision_limited_rule_passed": decimal_precision_limited,
        "confirmed_drift_rule_passed": confirmed,
    }
    return diagnosis, evidence


def _next_action(diagnosis: str) -> str:
    if diagnosis == "ENERGY_TELEMETRY_PRECISION_LIMITED":
        return (
            "Preregister a separate decision analysis for the 0.25-day production candidate; "
            "do not revise the historical Step 3 or Step 3b statuses."
        )
    if diagnosis == "ENERGY_DRIFT_CONFIRMED":
        return (
            "Before any further timestep halving, preregister a bounded integrator/roundoff "
            "diagnosis that separates WHFast truncation, synchronization, and accumulated "
            "state roundoff over shorter fixed horizons."
        )
    if diagnosis == "INCONCLUSIVE":
        return (
            "Identify the smallest conditioning or stored-state-precision ambiguity with the "
            "existing artifacts before requesting any new trajectory."
        )
    return "Resolve the recorded reconstruction discrepancy without launching an integration."


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['diagnosis']}",
        "",
        "Corrected energy was reconstructed offline from the existing lossless-float64 state CSVs.",
        "",
        "## Statistics",
        "",
        "| Lane | Method | Maximum | Worst epoch (yr) | RMS | P99 | Fitted change over 1 Myr |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lane_id, lane in payload.get("lanes", {}).items():
        for method in ("recorded", "float64", "compensated", "decimal"):
            stats = lane["statistics"][method]
            if method == "decimal":
                values = {
                    key: stats[key]["float"]
                    for key in (
                        "max_abs",
                        "max_abs_worst_epoch_years",
                        "rms",
                        "p99_abs",
                        "fitted_change_over_1myr",
                    )
                }
            else:
                values = stats
            lines.append(
                f"| {lane_id} | {method} | {values['max_abs']:.12g} | "
                f"{values['max_abs_worst_epoch_years']:.12g} | "
                f"{values['rms']:.12g} | {values['p99_abs']:.12g} | "
                f"{values['fitted_change_over_1myr']:.12g} |"
            )
    lines.extend(["", "## Method Checks", ""])
    historical = payload.get("audit", {}).get("historical_artifact_inventory", {})
    if historical:
        lines.append(
            "- Historical artifact audit: "
            f"`{historical['entries_verified']['step3_summary']}` Step 3 entries and "
            f"`{historical['entries_verified']['step3b_summary']}` Step 3b entries verified "
            f"(`{historical['unique_artifacts_verified']}` unique artifacts)."
        )
    for lane_id, lane in payload.get("lanes", {}).items():
        telemetry = lane["telemetry_reproduction"]
        agreement = lane["method_agreement"]
        cancellation = lane["conditioning"]
        lines.extend(
            [
                f"- {lane_id}: telemetry reproduction `{telemetry['passed']}`; worst tolerance ratio "
                f"`{telemetry['worst_tolerance_ratio']['value']:.12g}` at "
                f"`{telemetry['worst_tolerance_ratio']['worst_epoch_years']:.12g}` years.",
                f"- {lane_id}: compensated/Decimal agreement `{agreement['passed']}`; worst drift "
                f"difference `{agreement['compensated_minus_decimal_drift']['value']:.12g}` at "
                f"`{agreement['compensated_minus_decimal_drift']['worst_epoch_years']:.12g}` years.",
            ]
        )
        for method in ("float64", "compensated", "decimal"):
            composition = cancellation["worst_corrected_composition_cancellation"][
                method
            ]
            subtraction = cancellation["worst_drift_subtraction_cancellation"][method]
            lines.append(
                f"- {lane_id} {method}: worst corrected-energy composition cancellation "
                f"`{composition['value']:.12g}` at "
                f"`{composition['worst_epoch_years']:.12g}` years; "
                f"worst drift-subtraction cancellation `{subtraction['value']:.12g}` at "
                f"`{subtraction['worst_epoch_years']:.12g}` years."
            )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            f"- Telemetry reconstruction passed: `{payload.get('classification_evidence', {}).get('telemetry_reproduction_passed')}`.",
            f"- Compensated/Decimal agreement passed: `{payload.get('classification_evidence', {}).get('compensated_decimal_agreement_passed')}`.",
            f"- Confirmed-drift rule passed: `{payload.get('classification_evidence', {}).get('confirmed_drift_rule_passed')}`.",
            f"- Evidence supports future telemetry evaluation changes: `{payload.get('future_telemetry_change_supported')}`.",
            "- Historical Step 3 and Step 3b statuses remain unchanged.",
            "- No integration or Stage 4 command was run or produced.",
            "",
            "## Next Action",
            "",
            payload.get("next_action", ""),
            "",
        ]
    )
    return "\n".join(lines)


def render_existing_report(manifest_path: Path) -> None:
    manifest = _load_json(manifest_path, "diagnostic manifest")
    report_json = Path(manifest["paths"]["report_json"])
    report_markdown = Path(manifest["paths"]["report_markdown"])
    payload = _load_json(report_json, "diagnostic summary")
    _require(
        payload.get("manifest_sha256") == sha256_file(manifest_path),
        "Diagnostic summary does not match the frozen manifest.",
    )
    _require(payload.get("diagnosis") in FINAL_DIAGNOSES, "Invalid diagnosis.")
    _atomic_write_text(report_markdown, _markdown_report(payload))
    print(f"[m0-energy-diagnosis] wrote {report_markdown}")


def diagnose(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "diagnostic manifest")
    base_payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "model_id": manifest["model_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "historical_statuses_unchanged": manifest["provenance"][
            "historical_statuses_immutable"
        ],
    }
    output_root = Path(manifest["paths"]["output_root"])
    _require(not output_root.exists(), f"Fresh output root already exists: {output_root}")
    try:
        audit_payload = audit(manifest_path)
        lane_results = {
            lane["id"]: _lane_diagnosis(manifest_path, manifest, lane)
            for lane in manifest["input_lanes"]
        }
        diagnosis, classification = classify_diagnosis(manifest, lane_results)
        future_telemetry_change_supported = diagnosis == "ENERGY_TELEMETRY_PRECISION_LIMITED"
        base_payload.update(
            diagnosis=diagnosis,
            audit=audit_payload,
            lanes={key: value.summary for key, value in lane_results.items()},
            classification_evidence=classification,
            future_telemetry_change_supported=future_telemetry_change_supported,
            next_action=_next_action(diagnosis),
            timeseries_inventory=[
                value.timeseries_inventory for value in lane_results.values()
            ],
        )
    except Exception as exc:
        base_payload.update(
            diagnosis="BLOCKED",
            failures=[str(exc)],
            future_telemetry_change_supported=False,
            next_action=_next_action("BLOCKED"),
        )
    report_json = Path(manifest["paths"]["report_json"])
    report_markdown = Path(manifest["paths"]["report_markdown"])
    _require(base_payload["diagnosis"] in FINAL_DIAGNOSES, "Invalid diagnosis.")
    _atomic_write_json(report_json, base_payload)
    _atomic_write_text(report_markdown, _markdown_report(base_payload))
    print(f"[m0-energy-diagnosis] wrote {report_json}")
    print(f"[m0-energy-diagnosis] wrote {report_markdown}")
    print(f"[m0-energy-diagnosis] diagnosis={base_payload['diagnosis']}")
    return 0 if base_payload["diagnosis"] != "BLOCKED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline M0 corrected-energy precision diagnosis.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("diagnose")
    subparsers.add_parser("render-report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            print(json.dumps(audit(args.manifest), indent=2, sort_keys=True))
            return
        if args.command == "render-report":
            render_existing_report(args.manifest)
            return
        raise SystemExit(diagnose(args.manifest))
    except EnergyDiagnosisError as exc:
        raise SystemExit(f"m0 energy diagnosis error: {exc}") from exc


if __name__ == "__main__":
    main()
