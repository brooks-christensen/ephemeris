from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest import mock

import numpy as np

from .m0_energy_precision_diagnosis import float64_energy
from .nbody import G_SI
from .stability_diagnostics import total_angular_momentum_vector
from .nbody import NBodyState


C_M_PER_S = 299_792_458.0
JULIAN_YEAR_S = 365.25 * 86_400.0


class Step3g0AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_strict_finite(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            require_strict_finite(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_strict_finite(item, location=f"{location}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise Step3g0AuditError(f"Nonfinite value at {location}: {value!r}")


def unit_vector(vector: Sequence[float]) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("vector must be a finite three-vector")
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("zero-length vector has no direction")
    return values / norm


def direction_angles(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    u = unit_vector(left)
    v = unit_vector(right)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(np.cross(u, v)))
    chord_argument = float(np.clip(0.5 * np.linalg.norm(u - v), 0.0, 1.0))
    return {
        "acos_rad": math.acos(dot),
        "atan2_rad": math.atan2(cross_norm, dot),
        "chord_rad": 2.0 * math.asin(chord_argument),
    }


def newtonian_accelerations(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    *,
    gravitational_constant: float = G_SI,
) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    masses = np.asarray(masses_kg, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or masses.shape != (positions.shape[0],):
        raise ValueError("positions must be (N,3) and masses must be (N,)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(masses)):
        raise ValueError("Newtonian inputs must be finite")
    accelerations = np.zeros_like(positions)
    for left in range(len(masses)):
        for right in range(left + 1, len(masses)):
            displacement = positions[right] - positions[left]
            radius_squared = float(np.dot(displacement, displacement))
            if radius_squared == 0.0:
                raise ValueError("Newtonian point masses may not coincide")
            inverse_radius_cubed = radius_squared ** -1.5
            accelerations[left] += gravitational_constant * masses[right] * inverse_radius_cubed * displacement
            accelerations[right] -= gravitational_constant * masses[left] * inverse_radius_cubed * displacement
    return accelerations


def decimal_gr_pair_oracle(
    relative_position_m: Sequence[float],
    delta_relative_position_m: Sequence[float],
    *,
    gravitational_constant: float,
    central_mass_kg: float,
    coefficient_scale: float = 1.0,
    c_m_per_s: float = C_M_PER_S,
    precision: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    with localcontext() as context:
        context.prec = precision
        d = [Decimal(str(value)) for value in relative_position_m]
        delta = [Decimal(str(value)) for value in delta_relative_position_m]
        g = Decimal(str(gravitational_constant))
        mass = Decimal(str(central_mass_kg))
        scale = Decimal(str(coefficient_scale))
        c = Decimal(str(c_m_per_s))
        radius_squared = sum(value * value for value in d)
        if radius_squared <= 0:
            raise ValueError("GR point masses may not coincide")
        radius_fourth = radius_squared * radius_squared
        radius_sixth = radius_fourth * radius_squared
        prefactor = -Decimal(6) * scale * (g * mass) ** 2 / c**2
        acceleration = [prefactor * value / radius_fourth for value in d]
        projection = sum(d[index] * delta[index] for index in range(3))
        tangent = [
            prefactor * (delta[index] / radius_fourth - Decimal(4) * d[index] * projection / radius_sixth)
            for index in range(3)
        ]
    return np.asarray([float(value) for value in acceleration]), np.asarray([float(value) for value in tangent])


def decimal_gr_pair_central_difference(
    relative_position_m: Sequence[float],
    delta_relative_position_m: Sequence[float],
    *,
    gravitational_constant: float,
    central_mass_kg: float,
    coefficient_scale: float = 1.0,
    c_m_per_s: float = C_M_PER_S,
    parameter_step: str = "1e-20",
    precision: int = 70,
) -> np.ndarray:
    with localcontext() as context:
        context.prec = precision
        position = [Decimal(str(value)) for value in relative_position_m]
        direction = [Decimal(str(value)) for value in delta_relative_position_m]
        epsilon = Decimal(parameter_step)
        g = Decimal(str(gravitational_constant))
        mass = Decimal(str(central_mass_kg))
        scale = Decimal(str(coefficient_scale))
        c = Decimal(str(c_m_per_s))
        prefactor = -Decimal(6) * scale * (g * mass) ** 2 / c**2

        def acceleration(sign: Decimal) -> list[Decimal]:
            shifted = [position[index] + sign * epsilon * direction[index] for index in range(3)]
            radius_squared = sum(value * value for value in shifted)
            if radius_squared <= 0:
                raise ValueError("GR point masses may not coincide")
            return [prefactor * value / radius_squared**2 for value in shifted]

        plus = acceleration(Decimal(1))
        minus = acceleration(Decimal(-1))
        derivative = [(plus[index] - minus[index]) / (Decimal(2) * epsilon) for index in range(3)]
    return np.asarray([float(value) for value in derivative])


def complex_step_gr_pair_jvp(
    relative_position_m: Sequence[float],
    delta_relative_position_m: Sequence[float],
    *,
    gravitational_constant: float,
    central_mass_kg: float,
    coefficient_scale: float = 1.0,
    c_m_per_s: float = C_M_PER_S,
    step: float = 1.0e-20,
) -> np.ndarray:
    position = np.asarray(relative_position_m, dtype=np.complex128)
    direction = np.asarray(delta_relative_position_m, dtype=np.float64)
    shifted = position + 1j * step * direction
    # Use an analytic polynomial dot, not np.vdot, so complex arithmetic is
    # preserved for this independent formula-level oracle.
    radius_squared = np.sum(shifted * shifted)
    prefactor = -6.0 * coefficient_scale * (gravitational_constant * central_mass_kg) ** 2 / c_m_per_s**2
    acceleration = prefactor * shifted / radius_squared**2
    return np.imag(acceleration) / step


def gr_system_oracle(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    delta_positions_m: np.ndarray | None,
    *,
    gravitational_constant: float = G_SI,
    coefficient_scale: float = 1.0,
    c_m_per_s: float = C_M_PER_S,
    include_central_response: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    positions = np.asarray(positions_m, dtype=np.float64)
    masses = np.asarray(masses_kg, dtype=np.float64)
    delta = None if delta_positions_m is None else np.asarray(delta_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or masses.shape != (len(positions),):
        raise ValueError("invalid GR state shape")
    if delta is not None and delta.shape != positions.shape:
        raise ValueError("invalid tangent state shape")
    arrays = (positions, masses) if delta is None else (positions, masses, delta)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("GR inputs must be finite")
    if masses[0] <= 0.0 or not math.isfinite(gravitational_constant) or not math.isfinite(c_m_per_s):
        raise ValueError("GR scalar inputs must be finite and positive")
    acceleration = np.zeros_like(positions)
    tangent = None if delta is None else np.zeros_like(positions)
    zero_delta = np.zeros(3)
    for index in range(1, len(positions)):
        relative = positions[index] - positions[0]
        if float(np.dot(relative, relative)) == 0.0:
            raise ValueError("GR point masses may not coincide")
        pair_acceleration, pair_tangent = decimal_gr_pair_oracle(
            relative,
            zero_delta if delta is None else delta[index] - delta[0],
            gravitational_constant=gravitational_constant,
            central_mass_kg=float(masses[0]),
            coefficient_scale=coefficient_scale,
            c_m_per_s=c_m_per_s,
        )
        acceleration[index] = pair_acceleration
        if include_central_response:
            acceleration[0] -= masses[index] / masses[0] * pair_acceleration
        if tangent is not None:
            tangent[index] = pair_tangent
            if include_central_response:
                tangent[0] -= masses[index] / masses[0] * pair_tangent
    return acceleration, tangent


def callback_accounting_model(
    *,
    total_steps: int = 14_610_000,
    positive_output_events: int = 100,
    live_evaluations_per_step: int = 2,
    corrector_evaluations: int = 32,
) -> dict[str, int]:
    live_map = total_steps * live_evaluations_per_step
    first_map_corrector = corrector_evaluations
    pre_endpoint_sync = positive_output_events * corrector_evaluations
    return_sync = positive_output_events * corrector_evaluations
    total = live_map + first_map_corrector + pre_endpoint_sync + return_sync
    historical_expected = live_map + first_map_corrector + return_sync
    return {
        "total_steps": total_steps,
        "live_map": live_map,
        "first_map_forward_corrector": first_map_corrector,
        "pre_final_step_inverse_correctors": pre_endpoint_sync,
        "integrate_return_inverse_correctors": return_sync,
        "diagnostic_copy": 0,
        "checkpoint_serialization": 0,
        "analysis_only": 0,
        "source_schedule_total": total,
        "historical_expected": historical_expected,
        "difference": total - historical_expected,
    }


def restart_callback_accounting_model(
    *,
    continuation_steps: int = 146_100,
    live_evaluations_per_step: int = 2,
    corrector_evaluations: int = 32,
) -> dict[str, int]:
    live_map = continuation_steps * live_evaluations_per_step
    total = live_map + 2 * corrector_evaluations
    return {
        "continuation_steps": continuation_steps,
        "live_map": live_map,
        "pre_final_step_inverse_corrector": corrector_evaluations,
        "integrate_return_inverse_corrector": corrector_evaluations,
        "source_schedule_total": total,
        "historical_expected": live_map + corrector_evaluations,
        "difference": corrector_evaluations,
    }


@dataclass
class NoIntegrationGuard:
    calls: list[str]
    patches: list[mock._patch]

    def assert_unused(self) -> None:
        if self.calls:
            raise AssertionError(f"integration guard intercepted calls: {self.calls}")


@contextlib.contextmanager
def no_integration_guard() -> Iterator[NoIntegrationGuard]:
    import rebound

    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> None:
            calls.append(name)
            raise Step3g0AuditError(f"Step 3g0 prohibits {name}")

        return fail

    targets: list[tuple[Any, str, str]] = [
        (rebound.Simulation, "integrate", "rebound.Simulation.integrate"),
        (rebound.Simulation, "step", "rebound.Simulation.step"),
    ]
    for symbol in ("reb_simulation_integrate", "reb_simulation_step"):
        if hasattr(rebound.clibrebound, symbol):
            targets.append((rebound.clibrebound, symbol, f"rebound.clibrebound.{symbol}"))
    project_targets = (
        ("mini_ephemeris.m0_step3f1_runner", "run"),
        ("mini_ephemeris.rebound_gr_tangent_cli", "run"),
        ("mini_ephemeris.long_term_stability_cli", "run_stability_analysis"),
    )
    for module_name, symbol in project_targets:
        try:
            module = __import__(module_name, fromlist=[symbol])
        except ImportError:
            continue
        if hasattr(module, symbol):
            targets.append((module, symbol, f"{module_name}.{symbol}"))
    patches = [mock.patch.object(owner, symbol, forbidden(name)) for owner, symbol, name in targets]
    guard = NoIntegrationGuard(calls=calls, patches=patches)
    for patcher in patches:
        patcher.start()
    try:
        yield guard
    finally:
        for patcher in reversed(patches):
            patcher.stop()


def _load_frozen_state(path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise Step3g0AuditError(f"No state rows in {path}")
    sample_count = max(int(row["sample_index"]) for row in rows) + 1
    body_count = max(int(row["body_index"]) for row in rows) + 1
    if len(rows) != sample_count * body_count:
        raise Step3g0AuditError(f"Incomplete frozen state grid in {path}")
    positions = np.zeros((sample_count, body_count, 3))
    velocities = np.zeros_like(positions)
    masses = np.zeros(body_count)
    times = np.zeros(sample_count)
    names = [""] * body_count
    for row in rows:
        sample = int(row["sample_index"])
        body = int(row["body_index"])
        positions[sample, body] = [float(row[key]) for key in ("x_m", "y_m", "z_m")]
        velocities[sample, body] = [float(row[key]) for key in ("vx_m_per_s", "vy_m_per_s", "vz_m_per_s")]
        masses[body] = float(row["mass_kg"])
        names[body] = row["body_name"]
        times[sample] = float(row["time_years"])
    return {"positions": positions, "velocities": velocities, "masses": masses, "times": times, "names": names}


def recompute_frozen_orientation(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _load_frozen_state(left_path)
    right = _load_frozen_state(right_path)
    if left["names"] != right["names"] or not np.array_equal(left["times"], right["times"]):
        raise Step3g0AuditError("Frozen orientation inputs do not share identity and timestamps")
    records: list[dict[str, Any]] = []
    for sample, time_years in enumerate(left["times"]):
        for body in range(1, len(left["names"])):
            directions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            eccentricity_vectors = []
            for state in (left, right):
                r = state["positions"][sample, body] - state["positions"][sample, 0]
                v = state["velocities"][sample, body] - state["velocities"][sample, 0]
                angular_momentum = np.cross(r, v)
                mu = G_SI * (state["masses"][0] + state["masses"][body])
                eccentricity_vectors.append(np.cross(v, angular_momentum) / mu - r / np.linalg.norm(r))
                if state is left:
                    left_h = angular_momentum
                else:
                    right_h = angular_momentum
            directions["orbital_plane"] = (left_h, right_h)
            directions["apsidal_direction"] = (eccentricity_vectors[0], eccentricity_vectors[1])
            for metric, pair in directions.items():
                records.append({
                    "sample_index": sample,
                    "time_years": float(time_years),
                    "body": left["names"][body],
                    "metric": metric,
                    **direction_angles(*pair),
                })
    output: dict[str, Any] = {"comparisons": len(records), "metrics": {}}
    for metric in ("orbital_plane", "apsidal_direction"):
        selected = [row for row in records if row["metric"] == metric]
        metric_output: dict[str, Any] = {}
        for estimator in ("acos_rad", "atan2_rad", "chord_rad"):
            worst = max(selected, key=lambda row: row[estimator])
            metric_output[estimator] = {
                "max": worst[estimator],
                "worst_body": worst["body"],
                "worst_epoch_years": worst["time_years"],
                "zero_count": sum(row[estimator] == 0.0 for row in selected),
            }
        metric_output["max_atan2_chord_abs_difference"] = max(
            abs(row["atan2_rad"] - row["chord_rad"]) for row in selected
        )
        output["metrics"][metric] = metric_output
    require_strict_finite(output)
    return output


def recompute_frozen_conservation(state_path: Path) -> dict[str, Any]:
    state = _load_frozen_state(state_path)
    energies = []
    angular_momenta = []
    for sample in range(len(state["times"])):
        diagnostic = float64_energy(
            state["masses"],
            state["positions"][sample],
            state["velocities"][sample],
            gravitational_constant=G_SI,
            speed_of_light=C_M_PER_S,
            coefficient_scale=1.0,
        )
        energies.append(diagnostic)
        nbody = NBodyState(state["positions"][sample], state["velocities"][sample], state["masses"])
        angular_momenta.append(float(np.linalg.norm(total_angular_momentum_vector(nbody))))
    corrected = np.asarray([row["corrected"] for row in energies])
    newtonian = np.asarray([row["newtonian"] for row in energies])
    angular = np.asarray(angular_momenta)
    corrected_drift = (corrected - corrected[0]) / abs(corrected[0])
    newtonian_drift = (newtonian - newtonian[0]) / abs(newtonian[0])
    angular_drift = (angular - angular[0]) / abs(angular[0])
    output = {
        "samples": len(corrected),
        "corrected_energy_max_abs_relative_drift": float(np.max(np.abs(corrected_drift))),
        "newtonian_energy_max_abs_relative_drift": float(np.max(np.abs(newtonian_drift))),
        "angular_momentum_norm_max_abs_relative_drift": float(np.max(np.abs(angular_drift))),
        "initial_gr_potential_j": float(energies[0]["gr_potential"]),
    }
    require_strict_finite(output)
    return output


def audit_initial_physical_state(state_path: Path) -> dict[str, Any]:
    state = _load_frozen_state(state_path)
    positions = state["positions"][0]
    velocities = state["velocities"][0]
    masses = state["masses"]
    total_mass = float(np.sum(masses))
    center_of_mass = np.sum(masses[:, None] * positions, axis=0) / total_mass
    center_of_mass_velocity = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    mercury = 1
    r = positions[mercury] - positions[0]
    v = velocities[mercury] - velocities[0]
    mu = G_SI * (masses[0] + masses[mercury])
    specific_energy = 0.5 * float(np.dot(v, v)) - mu / float(np.linalg.norm(r))
    semimajor_axis = -mu / (2.0 * specific_energy)
    eccentricity_vector = np.cross(v, np.cross(r, v)) / mu - r / np.linalg.norm(r)
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    period_seconds = 2.0 * math.pi * math.sqrt(semimajor_axis**3 / mu)
    advance_per_orbit = 6.0 * math.pi * G_SI * masses[0] / (
        semimajor_axis * (1.0 - eccentricity**2) * C_M_PER_S**2
    )
    rate_arcsec_per_century = advance_per_orbit * 206_264.80624709636 * (
        100.0 * JULIAN_YEAR_S / period_seconds
    )
    output = {
        "body_count": len(masses),
        "total_mass_kg": total_mass,
        "center_of_mass_position_m": center_of_mass.tolist(),
        "center_of_mass_velocity_m_per_s": center_of_mass_velocity.tolist(),
        "total_linear_momentum_kg_m_per_s": momentum.tolist(),
        "mercury_initial_osculating_semimajor_axis_m": semimajor_axis,
        "mercury_initial_osculating_eccentricity": eccentricity,
        "analytic_gr_perihelion_rate_arcsec_per_century": rate_arcsec_per_century,
    }
    require_strict_finite(output)
    return output


def inspect_archive_readonly(path: Path) -> dict[str, Any]:
    import rebound

    before = sha256_file(path)
    archive = rebound.Simulationarchive(str(path))
    snapshots = len(archive)
    times = [float(archive[index].t) for index in range(snapshots)]
    after = sha256_file(path)
    if before != after:
        raise Step3g0AuditError(f"Read-only archive inspection changed {path}")
    output = {
        "path": str(path),
        "sha256_before": before,
        "sha256_after": after,
        "snapshots": snapshots,
        "first_time_seconds": times[0],
        "last_time_seconds": times[-1],
        "times_strictly_increasing": all(right > left for left, right in zip(times, times[1:])),
    }
    require_strict_finite(output)
    return output


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    require_strict_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)
