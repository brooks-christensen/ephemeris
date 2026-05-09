from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .nbody import NBodyState
from .orbital_elements import AU_M, JULIAN_YEAR_S, seconds_to_years


def make_position_perturbed_state(
    state: NBodyState,
    *,
    body_index: int,
    displacement_m: float = 1.0,
    component: int = 0,
) -> NBodyState:
    """Return a copy with one Cartesian position component displaced."""
    if component not in (0, 1, 2):
        raise ValueError("component must be 0, 1, or 2")
    if body_index < 0 or body_index >= state.positions.shape[0]:
        raise IndexError("body_index out of range")

    perturbed = state.copy()
    perturbed.positions[body_index, component] += float(displacement_m)
    return perturbed


def phase_space_separation_m(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> float:
    """
    Return a scaled phase-space separation in meters.

    Velocity differences are multiplied by ``velocity_time_scale_s`` so that
    position and velocity terms have common length units.
    """
    if reference.positions.shape != perturbed.positions.shape:
        raise ValueError("State position arrays must have the same shape.")
    if reference.velocities.shape != perturbed.velocities.shape:
        raise ValueError("State velocity arrays must have the same shape.")

    dpos = perturbed.positions - reference.positions
    dvel = (perturbed.velocities - reference.velocities) * velocity_time_scale_s
    return float(np.linalg.norm(np.concatenate([dpos.ravel(), dvel.ravel()])))


def finite_time_lyapunov_exponent(
    initial_separation_m: float,
    current_separation_m: float,
    elapsed_s: float,
) -> float:
    """Return log(delta/delta0)/elapsed_s, or NaN for degenerate inputs."""
    if initial_separation_m <= 0.0 or current_separation_m <= 0.0 or elapsed_s <= 0.0:
        return math.nan
    return math.log(current_separation_m / initial_separation_m) / elapsed_s


@dataclass(frozen=True)
class SeparationDiagnostics:
    time_years: float
    separation_au: float
    finite_time_lyapunov_per_year: float


def separation_diagnostics(
    time_s: float,
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    initial_separation_m: float,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> SeparationDiagnostics:
    separation_m = phase_space_separation_m(
        reference,
        perturbed,
        velocity_time_scale_s=velocity_time_scale_s,
    )
    exponent_per_s = finite_time_lyapunov_exponent(
        initial_separation_m,
        separation_m,
        time_s,
    )
    return SeparationDiagnostics(
        time_years=seconds_to_years(time_s),
        separation_au=separation_m / AU_M,
        finite_time_lyapunov_per_year=exponent_per_s * JULIAN_YEAR_S,
    )
