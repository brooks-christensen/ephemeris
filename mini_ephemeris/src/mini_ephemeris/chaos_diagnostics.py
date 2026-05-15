from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .nbody import G_SI, NBodyState
from .orbital_elements import AU_M, JULIAN_YEAR_S, seconds_to_years
from .stability_diagnostics import center_of_mass_position_velocity


@dataclass(frozen=True)
class RadialPerturbationResult:
    state: NBodyState
    target_norm: float
    body_indices: tuple[int, ...]
    body_names: tuple[str, ...]
    displacement_m_by_body: dict[str, float]
    sun_position_compensation_m: tuple[float, float, float]


@dataclass(frozen=True)
class RenormalizationResult:
    state: NBodyState
    separation_norm_before: float
    separation_norm_after: float
    scale_factor: float


@dataclass(frozen=True)
class PhaseSpaceComponentDiagnostics:
    separation_norm: float
    max_position_separation_m: float
    max_velocity_separation_m_s: float
    dominant_body_name: str
    dominant_component_type: str
    dominant_component_norm_contribution: float


def tangent_acceleration_newtonian(
    state: NBodyState,
    delta_positions_m: np.ndarray,
    *,
    G: float = G_SI,
) -> np.ndarray:
    """
    Return the Newtonian variational acceleration for position perturbations.

    For Newtonian point masses the acceleration has no velocity dependence, so
    the tangent acceleration depends only on the current reference positions and
    the tangent position vector.
    """
    positions = np.asarray(state.positions, dtype=float)
    masses = np.asarray(state.masses, dtype=float)
    delta_positions = np.asarray(delta_positions_m, dtype=float)
    if positions.shape != delta_positions.shape:
        raise ValueError("delta_positions_m must have the same shape as state.positions.")

    tangent_acc = np.zeros_like(positions)
    n_bodies = positions.shape[0]
    for i in range(n_bodies - 1):
        for j in range(i + 1, n_bodies):
            r_ij = positions[j] - positions[i]
            delta_r_ij = delta_positions[j] - delta_positions[i]
            r2 = float(np.dot(r_ij, r_ij))
            if r2 == 0.0:
                continue
            inv_r = 1.0 / math.sqrt(r2)
            inv_r3 = inv_r**3
            inv_r5 = inv_r**5
            projection = float(np.dot(r_ij, delta_r_ij))
            pair_term = delta_r_ij * inv_r3 - 3.0 * projection * r_ij * inv_r5
            tangent_acc[i] += G * masses[j] * pair_term
            tangent_acc[j] -= G * masses[i] * pair_term
    return tangent_acc


def velocity_verlet_tangent_step_newtonian(
    reference_state_new: NBodyState,
    delta_positions_m: np.ndarray,
    delta_velocities_m_s: np.ndarray,
    tangent_acceleration_m_s2: np.ndarray,
    dt_s: float,
    *,
    G: float = G_SI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance a tangent vector through one Newtonian velocity-Verlet step."""
    delta_vel_half = delta_velocities_m_s + 0.5 * dt_s * tangent_acceleration_m_s2
    delta_positions_new = delta_positions_m + dt_s * delta_vel_half
    tangent_acceleration_new = tangent_acceleration_newtonian(
        reference_state_new,
        delta_positions_new,
        G=G,
    )
    delta_velocities_new = delta_vel_half + 0.5 * dt_s * tangent_acceleration_new
    return delta_positions_new, delta_velocities_new, tangent_acceleration_new


def scaled_phase_space_delta_vector(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> np.ndarray:
    """Return perturbed-reference as [dr/AU, dv/(AU/year)] flat coordinates."""
    if reference.positions.shape != perturbed.positions.shape:
        raise ValueError("State position arrays must have the same shape.")
    if reference.velocities.shape != perturbed.velocities.shape:
        raise ValueError("State velocity arrays must have the same shape.")

    if body_indices is None:
        indices = np.arange(reference.positions.shape[0], dtype=int)
    else:
        indices = np.asarray(tuple(body_indices), dtype=int)

    dpos_au = (perturbed.positions[indices] - reference.positions[indices]) / AU_M
    dvel_au_per_year = (
        (perturbed.velocities[indices] - reference.velocities[indices])
        * velocity_time_scale_s
        / AU_M
    )
    return np.concatenate([dpos_au.ravel(), dvel_au_per_year.ravel()])


def scaled_phase_space_delta_vector_from_arrays(
    delta_positions_m: np.ndarray,
    delta_velocities_m_s: np.ndarray,
    *,
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> np.ndarray:
    """Return tangent arrays as flat [dr/AU, dv/(AU/year)] coordinates."""
    if delta_positions_m.shape != delta_velocities_m_s.shape:
        raise ValueError("delta position and velocity arrays must have matching shapes.")
    if body_indices is None:
        indices = np.arange(delta_positions_m.shape[0], dtype=int)
    else:
        indices = np.asarray(tuple(body_indices), dtype=int)
    dpos_au = delta_positions_m[indices] / AU_M
    dvel_au_per_year = delta_velocities_m_s[indices] * velocity_time_scale_s / AU_M
    return np.concatenate([dpos_au.ravel(), dvel_au_per_year.ravel()])


def cosine_between_scaled_deltas(delta_a: np.ndarray, delta_b: np.ndarray) -> float:
    """Return cosine between two scaled phase-space delta vectors."""
    norm_a = float(np.linalg.norm(delta_a))
    norm_b = float(np.linalg.norm(delta_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return math.nan
    return float(np.dot(delta_a, delta_b) / (norm_a * norm_b))


def state_from_scaled_delta_vector(
    reference: NBodyState,
    scaled_delta: np.ndarray,
    *,
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> NBodyState:
    """Build ``reference + scaled_delta`` from [dr/AU, dv/(AU/year)] coordinates."""
    if body_indices is None:
        indices = np.arange(reference.positions.shape[0], dtype=int)
    else:
        indices = np.asarray(tuple(body_indices), dtype=int)

    expected_size = 6 * len(indices)
    scaled_delta = np.asarray(scaled_delta, dtype=float)
    if scaled_delta.size != expected_size:
        raise ValueError(
            f"scaled_delta has size {scaled_delta.size}; expected {expected_size}."
        )

    n_coords = 3 * len(indices)
    dpos_au = scaled_delta[:n_coords].reshape((len(indices), 3))
    dvel_au_per_year = scaled_delta[n_coords:].reshape((len(indices), 3))

    state = reference.copy()
    state.positions[indices] = reference.positions[indices] + dpos_au * AU_M
    state.velocities[indices] = (
        reference.velocities[indices]
        + dvel_au_per_year * AU_M / velocity_time_scale_s
    )
    return state


def delta_arrays_from_scaled_delta_vector(
    reference: NBodyState,
    scaled_delta: np.ndarray,
    *,
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full SI tangent arrays from flat scaled phase-space coordinates."""
    if body_indices is None:
        indices = np.arange(reference.positions.shape[0], dtype=int)
    else:
        indices = np.asarray(tuple(body_indices), dtype=int)

    expected_size = 6 * len(indices)
    scaled_delta = np.asarray(scaled_delta, dtype=float)
    if scaled_delta.size != expected_size:
        raise ValueError(
            f"scaled_delta has size {scaled_delta.size}; expected {expected_size}."
        )

    n_coords = 3 * len(indices)
    dpos_au = scaled_delta[:n_coords].reshape((len(indices), 3))
    dvel_au_per_year = scaled_delta[n_coords:].reshape((len(indices), 3))

    delta_positions = np.zeros_like(reference.positions)
    delta_velocities = np.zeros_like(reference.velocities)
    delta_positions[indices] = dpos_au * AU_M
    delta_velocities[indices] = dvel_au_per_year * AU_M / velocity_time_scale_s
    return delta_positions, delta_velocities


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
    return scaled_phase_space_norm(
        reference,
        perturbed,
        velocity_time_scale_s=velocity_time_scale_s,
    ) * AU_M


def scaled_phase_space_norm(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> float:
    """
    Return a dimensionless phase-space norm using AU and AU/year units.

    The norm is

        sqrt(sum((dr / AU)^2 + (dv / (AU/year))^2)).

    This avoids mixing position meters and velocity meters/second directly.
    """
    return float(
        np.linalg.norm(
            scaled_phase_space_delta_vector(
                reference,
                perturbed,
                body_indices=body_indices,
                velocity_time_scale_s=velocity_time_scale_s,
            )
        )
    )


def scaled_phase_space_component_diagnostics(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    body_names: Sequence[str],
    body_indices: Sequence[int] | None = None,
    velocity_time_scale_s: float = JULIAN_YEAR_S,
) -> PhaseSpaceComponentDiagnostics:
    """Return component-level diagnostics for the scaled phase-space norm."""
    if body_indices is None:
        indices = tuple(range(reference.positions.shape[0]))
    else:
        indices = tuple(int(index) for index in body_indices)

    names = tuple(body_names)
    dpos = perturbed.positions - reference.positions
    dvel = perturbed.velocities - reference.velocities

    max_position = 0.0
    max_velocity = 0.0
    dominant_body = ""
    dominant_type = "position"
    dominant_contribution = -1.0

    total = 0.0
    for index in indices:
        pos_norm_m = float(np.linalg.norm(dpos[index]))
        vel_norm_m_s = float(np.linalg.norm(dvel[index]))
        pos_contribution = pos_norm_m / AU_M
        vel_contribution = vel_norm_m_s * velocity_time_scale_s / AU_M
        total += pos_contribution * pos_contribution + vel_contribution * vel_contribution

        if pos_norm_m > max_position:
            max_position = pos_norm_m
        if vel_norm_m_s > max_velocity:
            max_velocity = vel_norm_m_s
        if pos_contribution > dominant_contribution:
            dominant_contribution = pos_contribution
            dominant_body = names[index]
            dominant_type = "position"
        if vel_contribution > dominant_contribution:
            dominant_contribution = vel_contribution
            dominant_body = names[index]
            dominant_type = "velocity"

    return PhaseSpaceComponentDiagnostics(
        separation_norm=math.sqrt(total),
        max_position_separation_m=max_position,
        max_velocity_separation_m_s=max_velocity,
        dominant_body_name=dominant_body,
        dominant_component_type=dominant_type,
        dominant_component_norm_contribution=dominant_contribution,
    )


def finite_time_lyapunov_exponent(
    initial_separation_m: float,
    current_separation_m: float,
    elapsed_s: float,
) -> float:
    """Return log(delta/delta0)/elapsed_s, or NaN for degenerate inputs."""
    if initial_separation_m <= 0.0 or current_separation_m <= 0.0 or elapsed_s <= 0.0:
        return math.nan
    return math.log(current_separation_m / initial_separation_m) / elapsed_s


def lyapunov_time_years(lambda_1_per_year: float) -> float:
    if not math.isfinite(lambda_1_per_year) or lambda_1_per_year <= 0.0:
        return math.inf
    return 1.0 / lambda_1_per_year


def _radial_unit_vector(state: NBodyState, *, body_index: int, sun_index: int) -> np.ndarray:
    radial = state.positions[body_index] - state.positions[sun_index]
    norm = float(np.linalg.norm(radial))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Cannot build radial perturbation from a degenerate Sun-body vector.")
    return radial / norm


def make_radial_perturbed_state(
    state: NBodyState,
    *,
    body_indices: Sequence[int],
    body_names: Sequence[str],
    sun_index: int,
    perturbation_m: float,
    seed: int | None = None,
    preserve_barycenter: bool = True,
) -> RadialPerturbationResult:
    """
    Perturb selected bodies radially and optionally compensate with the Sun.

    For one selected body the perturbation is a positive outward radial shift.
    For multiple selected bodies, seeded random radial weights are normalized
    so the selected-body displacement vector has the requested meter scale.
    """
    indices = tuple(int(index) for index in body_indices)
    names = tuple(str(name) for name in body_names)
    if len(indices) == 0:
        raise ValueError("At least one body index is required for Lyapunov perturbation.")
    if len(indices) != len(names):
        raise ValueError("body_indices and body_names must have the same length.")
    if perturbation_m <= 0.0:
        raise ValueError("perturbation_m must be positive.")

    deltas = np.zeros_like(state.positions)
    if len(indices) == 1:
        weights = np.array([1.0], dtype=float)
    else:
        rng = np.random.default_rng(seed)
        weights = rng.normal(size=len(indices))
        weight_norm = float(np.linalg.norm(weights))
        if weight_norm == 0.0 or not math.isfinite(weight_norm):
            weights = np.zeros(len(indices), dtype=float)
            weights[0] = 1.0
        else:
            weights = weights / weight_norm

    displacement_m_by_body: dict[str, float] = {}
    for index, name, weight in zip(indices, names, weights):
        unit = _radial_unit_vector(state, body_index=index, sun_index=sun_index)
        delta = perturbation_m * float(weight) * unit
        deltas[index] += delta
        displacement_m_by_body[name] = float(np.linalg.norm(delta))

    sun_compensation = np.zeros(3, dtype=float)
    if preserve_barycenter:
        weighted_position_delta = np.sum(
            state.masses[:, np.newaxis] * deltas,
            axis=0,
        )
        sun_mass = float(state.masses[sun_index])
        if sun_mass <= 0.0:
            raise ValueError("Sun mass must be positive for barycenter compensation.")
        sun_compensation = -weighted_position_delta / sun_mass
        deltas[sun_index] += sun_compensation

    perturbed = state.copy()
    perturbed.positions += deltas
    target_norm = scaled_phase_space_norm(state, perturbed)

    return RadialPerturbationResult(
        state=perturbed,
        target_norm=target_norm,
        body_indices=indices,
        body_names=names,
        displacement_m_by_body=displacement_m_by_body,
        sun_position_compensation_m=tuple(float(x) for x in sun_compensation),
    )


def match_sun_to_reference_barycenter(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    sun_index: int,
) -> NBodyState:
    """Shift the perturbed Sun so total barycenter position and velocity match."""
    corrected = perturbed.copy()
    ref_com_pos, ref_com_vel = center_of_mass_position_velocity(reference)
    pert_com_pos, pert_com_vel = center_of_mass_position_velocity(corrected)
    total_mass = float(np.sum(corrected.masses))
    sun_mass = float(corrected.masses[sun_index])
    if sun_mass <= 0.0:
        raise ValueError("Sun mass must be positive for barycenter compensation.")

    corrected.positions[sun_index] -= (total_mass / sun_mass) * (pert_com_pos - ref_com_pos)
    corrected.velocities[sun_index] -= (total_mass / sun_mass) * (pert_com_vel - ref_com_vel)
    return corrected


def renormalize_to_scaled_norm(
    reference: NBodyState,
    perturbed: NBodyState,
    *,
    target_norm: float,
    sun_index: int,
    preserve_barycenter: bool = True,
) -> RenormalizationResult:
    """Rescale perturbed-reference separation back to ``target_norm``."""
    scaled_delta = scaled_phase_space_delta_vector(reference, perturbed)
    separation_norm = float(np.linalg.norm(scaled_delta))
    if separation_norm <= 0.0 or not math.isfinite(separation_norm):
        raise ValueError("Cannot renormalize a degenerate Lyapunov separation.")
    if target_norm <= 0.0 or not math.isfinite(target_norm):
        raise ValueError("target_norm must be finite and positive.")

    scale = target_norm / separation_norm

    # Barycenter compensation is a projection that can slightly change the
    # scaled norm. Iterate the scale/projection pair so the post-renorm norm is
    # genuinely the requested target.
    renormalized = reference.copy()
    for _ in range(6):
        renormalized = state_from_scaled_delta_vector(
            reference,
            scaled_delta * scale,
        )

        if preserve_barycenter:
            renormalized = match_sun_to_reference_barycenter(
                reference,
                renormalized,
                sun_index=sun_index,
            )

        after_norm = scaled_phase_space_norm(reference, renormalized)
        if after_norm <= 0.0 or not math.isfinite(after_norm):
            raise ValueError("Renormalized Lyapunov separation became degenerate.")
        correction = target_norm / after_norm
        scale *= correction
        if abs(correction - 1.0) < 1.0e-12:
            break

    after_norm = scaled_phase_space_norm(reference, renormalized)
    return RenormalizationResult(
        state=renormalized,
        separation_norm_before=separation_norm,
        separation_norm_after=after_norm,
        scale_factor=scale,
    )


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
