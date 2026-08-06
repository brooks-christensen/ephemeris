from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .gr_potential_tangent import C_M_PER_S
from .nbody import G_SI, NBodyState
from .orbital_elements import seconds_to_years


STATE_SAMPLE_SCHEMA_VERSION = 1
STATE_SAMPLE_FIELDS = [
    "state_sample_schema_version",
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
    "variation_config_index",
    "variation_x_m",
    "variation_y_m",
    "variation_z_m",
    "variation_vx_m_per_s",
    "variation_vy_m_per_s",
    "variation_vz_m_per_s",
]


class TelemetrySchemaError(RuntimeError):
    pass


def gr_potential_energy(
    state: NBodyState,
    *,
    sun_index: int = 0,
    coefficient_scale: float = 1.0,
    gravitational_constant: float = G_SI,
    c_m_per_s: float = C_M_PER_S,
) -> float:
    """Return the conservative potential represented by the GR-potential force."""
    if coefficient_scale == 0.0:
        return 0.0
    masses = np.asarray(state.masses, dtype=np.float64)
    positions = np.asarray(state.positions, dtype=np.float64)
    if positions.shape != (len(masses), 3):
        raise ValueError("Positions and masses have incompatible shapes.")
    if not 0 <= sun_index < len(masses):
        raise IndexError("sun_index is outside the state.")
    central_mass = float(masses[sun_index])
    weighted_inverse_r2 = 0.0
    for body_index, mass in enumerate(masses):
        if body_index == sun_index:
            continue
        displacement = positions[body_index] - positions[sun_index]
        radius_squared = float(np.dot(displacement, displacement))
        if radius_squared == 0.0:
            raise ValueError("GR-potential energy is undefined for coincident bodies.")
        weighted_inverse_r2 += float(mass) / radius_squared
    prefactor = (
        -3.0
        * coefficient_scale
        * gravitational_constant**2
        * central_mass**2
        / c_m_per_s**2
    )
    return float(prefactor * weighted_inverse_r2)


def state_sample_rows(
    sim: Any,
    body_names: Sequence[str],
    *,
    sample_index: int,
    configuration_fingerprint: str,
) -> list[dict[str, Any]]:
    """Serialize all real particles and the first variation block at one epoch."""
    names = tuple(body_names)
    n_real = int(sim.N_real)
    if n_real != len(names):
        raise TelemetrySchemaError("Real-particle count does not match body names.")
    if int(sim.N_var) < n_real or int(sim.N) < 2 * n_real:
        raise TelemetrySchemaError("The first variation block is incomplete.")
    time_seconds = float(sim.t)
    time_years = seconds_to_years(time_seconds)
    rows: list[dict[str, Any]] = []
    for body_index, body_name in enumerate(names):
        particle = sim.particles[body_index]
        variation = sim.particles[n_real + body_index]
        rows.append(
            {
                "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
                "configuration_fingerprint": configuration_fingerprint,
                "sample_index": sample_index,
                "time_seconds": time_seconds,
                "time_years": time_years,
                "body_index": body_index,
                "body_name": body_name,
                "mass_kg": float(particle.m),
                "x_m": float(particle.x),
                "y_m": float(particle.y),
                "z_m": float(particle.z),
                "vx_m_per_s": float(particle.vx),
                "vy_m_per_s": float(particle.vy),
                "vz_m_per_s": float(particle.vz),
                "variation_config_index": 0,
                "variation_x_m": float(variation.x),
                "variation_y_m": float(variation.y),
                "variation_z_m": float(variation.z),
                "variation_vx_m_per_s": float(variation.vx),
                "variation_vy_m_per_s": float(variation.vy),
                "variation_vz_m_per_s": float(variation.vz),
            }
        )
    return rows


def read_state_samples(
    path: Path,
    *,
    body_names: Sequence[str],
    configuration_fingerprint: str,
) -> list[dict[str, str]]:
    if b"\x00" in path.read_bytes():
        raise TelemetrySchemaError(f"NUL byte in state sample CSV: {path}")
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != STATE_SAMPLE_FIELDS:
                raise TelemetrySchemaError(
                    f"State sample schema mismatch in {path}: {reader.fieldnames}"
                )
            rows = list(reader)
    except (csv.Error, OSError) as exc:
        raise TelemetrySchemaError(f"Unreadable state sample CSV {path}: {exc}") from exc
    names = tuple(body_names)
    n_real = len(names)
    if len(rows) % n_real:
        raise TelemetrySchemaError("State sample CSV ends with an incomplete body group.")
    previous_time = -math.inf
    for offset in range(0, len(rows), n_real):
        group = rows[offset : offset + n_real]
        expected_sample_index = offset // n_real
        try:
            group_times = {float(row["time_years"]) for row in group}
            sample_indices = {int(row["sample_index"]) for row in group}
            body_indices = [int(row["body_index"]) for row in group]
            schema_versions = {
                int(row["state_sample_schema_version"]) for row in group
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise TelemetrySchemaError("Malformed state sample row.") from exc
        if group_times == set() or len(group_times) != 1:
            raise TelemetrySchemaError("State sample body group has inconsistent times.")
        current_time = next(iter(group_times))
        if not math.isfinite(current_time) or current_time <= previous_time:
            raise TelemetrySchemaError("State sample times are not strictly increasing.")
        if sample_indices != {expected_sample_index}:
            raise TelemetrySchemaError("State sample index sequence is incomplete or duplicated.")
        if body_indices != list(range(n_real)):
            raise TelemetrySchemaError("State sample body sequence is incomplete or duplicated.")
        if [row["body_name"] for row in group] != list(names):
            raise TelemetrySchemaError("State sample body names do not match configuration.")
        if schema_versions != {STATE_SAMPLE_SCHEMA_VERSION}:
            raise TelemetrySchemaError("State sample schema version is incompatible.")
        if any(
            row["configuration_fingerprint"] != configuration_fingerprint
            for row in group
        ):
            raise TelemetrySchemaError("State sample configuration fingerprint mismatch.")
        previous_time = current_time
    return rows
