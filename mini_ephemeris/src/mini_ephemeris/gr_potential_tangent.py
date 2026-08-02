from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


C_M_PER_S = 299_792_458.0


@dataclass(frozen=True)
class ReboundVariationApiBehavior:
    rebound_version: str
    arbitrary_cartesian_assignment_supported: bool
    readback_exact: bool
    n_real: int
    n_total_after_variation: int
    n_variation_particles: int


def gr_potential_pair_acceleration(
    relative_position_m: np.ndarray,
    *,
    gravitational_constant: float,
    central_mass_kg: float,
    c_m_per_s: float = C_M_PER_S,
    coefficient_scale: float = 1.0,
) -> np.ndarray:
    r2 = float(np.dot(relative_position_m, relative_position_m))
    if r2 <= 0.0 or coefficient_scale == 0.0:
        return np.zeros(3)
    r = math.sqrt(r2)
    coefficient = -6.0 * coefficient_scale * (gravitational_constant * central_mass_kg) ** 2 / (
        c_m_per_s**2 * r**4
    )
    return coefficient * relative_position_m


def gr_potential_pair_jacobian(
    relative_position_m: np.ndarray,
    *,
    gravitational_constant: float,
    central_mass_kg: float,
    c_m_per_s: float = C_M_PER_S,
    coefficient_scale: float = 1.0,
) -> np.ndarray:
    r2 = float(np.dot(relative_position_m, relative_position_m))
    if r2 <= 0.0 or coefficient_scale == 0.0:
        return np.zeros((3, 3))
    r = math.sqrt(r2)
    identity = np.eye(3)
    outer = np.outer(relative_position_m, relative_position_m)
    prefactor = -6.0 * coefficient_scale * (gravitational_constant * central_mass_kg) ** 2 / c_m_per_s**2
    return prefactor * (identity / r**4 - 4.0 * outer / r**6)


def gr_potential_accelerations_and_tangent(
    positions_m: np.ndarray,
    masses_kg: np.ndarray,
    delta_positions_m: np.ndarray | None,
    *,
    gravitational_constant: float,
    central_index: int = 0,
    c_m_per_s: float = C_M_PER_S,
    coefficient_scale: float = 1.0,
    include_central_response: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    positions = np.asarray(positions_m, dtype=float)
    masses = np.asarray(masses_kg, dtype=float)
    accelerations = np.zeros_like(positions)
    tangent = np.zeros_like(positions) if delta_positions_m is not None else None
    central_mass = float(masses[central_index])
    for index in range(len(masses)):
        if index == central_index:
            continue
        relative = positions[index] - positions[central_index]
        planet_acc = gr_potential_pair_acceleration(
            relative,
            gravitational_constant=gravitational_constant,
            central_mass_kg=central_mass,
            c_m_per_s=c_m_per_s,
            coefficient_scale=coefficient_scale,
        )
        accelerations[index] += planet_acc
        if include_central_response:
            accelerations[central_index] -= masses[index] / central_mass * planet_acc
        if tangent is not None:
            jacobian = gr_potential_pair_jacobian(
                relative,
                gravitational_constant=gravitational_constant,
                central_mass_kg=central_mass,
                c_m_per_s=c_m_per_s,
                coefficient_scale=coefficient_scale,
            )
            delta_relative = delta_positions_m[index] - delta_positions_m[central_index]
            planet_delta_acc = jacobian @ delta_relative
            tangent[index] += planet_delta_acc
            if include_central_response:
                tangent[central_index] -= masses[index] / central_mass * planet_delta_acc
    return accelerations, tangent


def verify_rebound_variation_api(rebound_module) -> ReboundVariationApiBehavior:
    sim = rebound_module.Simulation()
    sim.G = 1.0
    sim.add(m=1.0)
    sim.add(m=1.0e-3, x=1.0, vy=1.0)
    variation = sim.add_variation()
    assigned = []
    for index, particle in enumerate(variation.particles):
        values = (10.0 + index, 20.0 + index, 30.0 + index, 40.0 + index, 50.0 + index, 60.0 + index)
        particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz = values
        assigned.append(values)
    readback = [
        (particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz)
        for particle in variation.particles
    ]
    return ReboundVariationApiBehavior(
        rebound_version=getattr(rebound_module, "__version__", "unknown"),
        arbitrary_cartesian_assignment_supported=True,
        readback_exact=assigned == readback,
        n_real=int(sim.N_real),
        n_total_after_variation=int(sim.N),
        n_variation_particles=len(variation.particles),
    )


def attach_gr_potential_tangent_force(
    sim,
    *,
    coefficient_scale: float = 1.0,
    c_m_per_s: float = C_M_PER_S,
    include_central_response: bool = True,
) -> Callable:
    stats = {
        "callback_invocations": 0,
        "real_gr_accel_norm_max": 0.0,
        "real_gr_accel_norm_sum": 0.0,
        "real_gr_accel_norm_count": 0,
        "tangent_gr_accel_norm_max": 0.0,
        "tangent_gr_accel_norm_sum": 0.0,
        "tangent_gr_accel_norm_count": 0,
    }
    if coefficient_scale == 0.0:
        sim._mini_ephemeris_gr_potential_tangent_force = None
        sim._mini_ephemeris_gr_potential_tangent_stats = stats
        sim._mini_ephemeris_gr_potential_tangent_config = {
            "coefficient_scale": coefficient_scale,
            "c_m_per_s": c_m_per_s,
            "include_central_response": include_central_response,
            "zero_limit_callback_skipped": True,
        }
        return lambda _sim_pointer: None

    def additional_forces(sim_pointer):
        rebound_sim = sim_pointer.contents
        n_real = int(rebound_sim.N_real)
        particles = rebound_sim.particles
        stats["callback_invocations"] += 1
        central = particles[0]
        central_mass = float(central.m)
        mu2_over_c2 = coefficient_scale * (float(rebound_sim.G) * central_mass) ** 2 / c_m_per_s**2
        real_accels: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * n_real
        central_ax = 0.0
        central_ay = 0.0
        central_az = 0.0
        for i in range(1, n_real):
            particle = particles[i]
            dx = float(particle.x - central.x)
            dy = float(particle.y - central.y)
            dz = float(particle.z - central.z)
            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                ax = ay = az = 0.0
            else:
                inv_r2 = 1.0 / r2
                coefficient = -6.0 * mu2_over_c2 * inv_r2 * inv_r2
                ax = coefficient * dx
                ay = coefficient * dy
                az = coefficient * dz
            particle.ax += ax
            particle.ay += ay
            particle.az += az
            real_accels[i] = (ax, ay, az)
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            stats["real_gr_accel_norm_max"] = max(stats["real_gr_accel_norm_max"], norm)
            stats["real_gr_accel_norm_sum"] += norm
            stats["real_gr_accel_norm_count"] += 1
            if include_central_response:
                mass_ratio = float(particle.m) / central_mass
                central_ax -= mass_ratio * ax
                central_ay -= mass_ratio * ay
                central_az -= mass_ratio * az
        if include_central_response:
            central.ax += central_ax
            central.ay += central_ay
            central.az += central_az
        for start in range(n_real, int(rebound_sim.N), n_real):
            if start + n_real > int(rebound_sim.N):
                break
            central_var = particles[start]
            central_tax = 0.0
            central_tay = 0.0
            central_taz = 0.0
            for i in range(1, n_real):
                particle = particles[i]
                var_particle = particles[start + i]
                dx = float(particle.x - central.x)
                dy = float(particle.y - central.y)
                dz = float(particle.z - central.z)
                ddx = float(var_particle.x - central_var.x)
                ddy = float(var_particle.y - central_var.y)
                ddz = float(var_particle.z - central_var.z)
                r2 = dx * dx + dy * dy + dz * dz
                if r2 <= 0.0:
                    tax = tay = taz = 0.0
                else:
                    inv_r2 = 1.0 / r2
                    inv_r4 = inv_r2 * inv_r2
                    dot = dx * ddx + dy * ddy + dz * ddz
                    prefactor = -6.0 * mu2_over_c2
                    common = 4.0 * dot * inv_r4 * inv_r2
                    tax = prefactor * (ddx * inv_r4 - dx * common)
                    tay = prefactor * (ddy * inv_r4 - dy * common)
                    taz = prefactor * (ddz * inv_r4 - dz * common)
                var_particle.ax += tax
                var_particle.ay += tay
                var_particle.az += taz
                norm = math.sqrt(tax * tax + tay * tay + taz * taz)
                stats["tangent_gr_accel_norm_max"] = max(stats["tangent_gr_accel_norm_max"], norm)
                stats["tangent_gr_accel_norm_sum"] += norm
                stats["tangent_gr_accel_norm_count"] += 1
                if include_central_response:
                    mass_ratio = float(particle.m) / central_mass
                    central_tax -= mass_ratio * tax
                    central_tay -= mass_ratio * tay
                    central_taz -= mass_ratio * taz
            if include_central_response:
                central_var.ax += central_tax
                central_var.ay += central_tay
                central_var.az += central_taz

    sim.additional_forces = additional_forces
    sim.force_is_velocity_dependent = 0
    sim._mini_ephemeris_gr_potential_tangent_force = additional_forces
    sim._mini_ephemeris_gr_potential_tangent_stats = stats
    sim._mini_ephemeris_gr_potential_tangent_config = {
        "coefficient_scale": coefficient_scale,
        "c_m_per_s": c_m_per_s,
        "include_central_response": include_central_response,
    }
    return additional_forces
