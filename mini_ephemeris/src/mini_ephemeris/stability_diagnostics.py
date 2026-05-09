from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .nbody import G_SI, NBodyState
from .orbital_elements import ARCSEC_PER_RAD, AU_M, JULIAN_YEAR_S, seconds_to_years


def total_newtonian_energy(state: NBodyState, *, G: float = G_SI) -> float:
    """Return kinetic plus Newtonian point-mass potential energy in joules."""
    masses = state.masses
    velocities = state.velocities
    positions = state.positions

    kinetic = 0.5 * float(np.sum(masses[:, np.newaxis] * velocities * velocities))

    potential = 0.0
    n_bodies = len(masses)
    for i in range(n_bodies - 1):
        for j in range(i + 1, n_bodies):
            r_ij = float(np.linalg.norm(positions[j] - positions[i]))
            if r_ij == 0.0:
                potential = -math.inf
            else:
                potential -= G * float(masses[i]) * float(masses[j]) / r_ij

    return float(kinetic + potential)


def total_angular_momentum_vector(state: NBodyState) -> np.ndarray:
    """Return total angular momentum vector about the coordinate origin."""
    return np.sum(
        state.masses[:, np.newaxis] * np.cross(state.positions, state.velocities),
        axis=0,
    )


def center_of_mass_position_velocity(state: NBodyState) -> tuple[np.ndarray, np.ndarray]:
    """Return center-of-mass position and velocity."""
    total_mass = float(np.sum(state.masses))
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive.")

    weights = state.masses[:, np.newaxis] / total_mass
    position = np.sum(weights * state.positions, axis=0)
    velocity = np.sum(weights * state.velocities, axis=0)
    return position, velocity


@dataclass(frozen=True)
class InvariantReference:
    energy_j: float
    angular_momentum: np.ndarray
    angular_momentum_norm: float
    com_position_m: np.ndarray
    com_velocity_m_s: np.ndarray


def invariant_reference(state: NBodyState, *, G: float = G_SI) -> InvariantReference:
    angular_momentum = total_angular_momentum_vector(state)
    com_position, com_velocity = center_of_mass_position_velocity(state)
    return InvariantReference(
        energy_j=total_newtonian_energy(state, G=G),
        angular_momentum=angular_momentum,
        angular_momentum_norm=float(np.linalg.norm(angular_momentum)),
        com_position_m=com_position,
        com_velocity_m_s=com_velocity,
    )


def invariant_diagnostics_row(
    time_s: float,
    state: NBodyState,
    reference: InvariantReference,
    *,
    G: float = G_SI,
) -> dict[str, float]:
    """Return one CSV-ready invariant diagnostics row."""
    energy = total_newtonian_energy(state, G=G)
    energy_abs_drift = energy - reference.energy_j
    energy_scale = abs(reference.energy_j) if reference.energy_j != 0.0 else 1.0

    angular_momentum = total_angular_momentum_vector(state)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    angular_delta = angular_momentum - reference.angular_momentum
    angular_abs_drift = float(np.linalg.norm(angular_delta))
    angular_scale = (
        reference.angular_momentum_norm
        if reference.angular_momentum_norm != 0.0
        else 1.0
    )

    if angular_momentum_norm > 0.0 and reference.angular_momentum_norm > 0.0:
        cos_angle = float(
            np.dot(angular_momentum, reference.angular_momentum)
            / (angular_momentum_norm * reference.angular_momentum_norm)
        )
        angular_direction_drift_arcsec = math.acos(
            max(-1.0, min(1.0, cos_angle))
        ) * ARCSEC_PER_RAD
    else:
        angular_direction_drift_arcsec = math.nan

    com_position, com_velocity = center_of_mass_position_velocity(state)
    com_position_drift = com_position - reference.com_position_m
    com_velocity_drift = com_velocity - reference.com_velocity_m_s

    return {
        "time_years": seconds_to_years(time_s),
        "energy_j": energy,
        "energy_abs_drift_j": energy_abs_drift,
        "energy_rel_drift": energy_abs_drift / energy_scale,
        "angular_momentum_norm_kg_m2_s": angular_momentum_norm,
        "angular_momentum_abs_drift_kg_m2_s": angular_abs_drift,
        "angular_momentum_rel_drift": angular_abs_drift / angular_scale,
        "angular_momentum_direction_drift_arcsec": angular_direction_drift_arcsec,
        "com_x_au": com_position[0] / AU_M,
        "com_y_au": com_position[1] / AU_M,
        "com_z_au": com_position[2] / AU_M,
        "com_vx_au_per_year": com_velocity[0] * JULIAN_YEAR_S / AU_M,
        "com_vy_au_per_year": com_velocity[1] * JULIAN_YEAR_S / AU_M,
        "com_vz_au_per_year": com_velocity[2] * JULIAN_YEAR_S / AU_M,
        "com_position_drift_au": float(np.linalg.norm(com_position_drift)) / AU_M,
        "com_velocity_drift_au_per_year": (
            float(np.linalg.norm(com_velocity_drift)) * JULIAN_YEAR_S / AU_M
        ),
    }


def pairwise_distance_m(positions_m: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions_m[j] - positions_m[i]))


@dataclass
class PairwiseMinimumTracker:
    """Track minimum body-body separations without storing trajectory samples."""

    body_names: tuple[str, ...]
    pair_indices: tuple[tuple[int, int], ...]
    min_distance_m: np.ndarray
    min_time_s: np.ndarray

    @classmethod
    def create(cls, body_names: Sequence[str]) -> "PairwiseMinimumTracker":
        names = tuple(body_names)
        pairs = tuple(
            (i, j)
            for i in range(len(names) - 1)
            for j in range(i + 1, len(names))
        )
        return cls(
            body_names=names,
            pair_indices=pairs,
            min_distance_m=np.full(len(pairs), math.inf, dtype=float),
            min_time_s=np.full(len(pairs), math.nan, dtype=float),
        )

    def update(self, time_s: float, positions_m: np.ndarray) -> None:
        for pair_index, (i, j) in enumerate(self.pair_indices):
            distance = pairwise_distance_m(positions_m, i, j)
            if distance < self.min_distance_m[pair_index]:
                self.min_distance_m[pair_index] = distance
                self.min_time_s[pair_index] = time_s

    def rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for pair_index, (i, j) in enumerate(self.pair_indices):
            distance_m = float(self.min_distance_m[pair_index])
            rows.append(
                {
                    "body_i": self.body_names[i],
                    "body_j": self.body_names[j],
                    "min_separation_au": distance_m / AU_M,
                    "min_separation_km": distance_m / 1.0e3,
                    "time_years": seconds_to_years(float(self.min_time_s[pair_index])),
                }
            )
        return rows
