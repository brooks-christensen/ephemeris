from __future__ import annotations

import math
import unittest

import numpy as np

from mini_ephemeris.ephem import BODY_MASSES
from mini_ephemeris.gr_potential_tangent import (
    attach_gr_potential_tangent_force,
    gr_potential_accelerations_and_tangent,
    verify_rebound_variation_api,
)
from mini_ephemeris.nbody import G_SI
from mini_ephemeris.orbital_elements import AU_M, DAY_S
from mini_ephemeris.long_term_stability_cli import build_rebound_simulation
from mini_ephemeris.nbody import NBodyState


def _require_rebound():
    try:
        import rebound
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"REBOUND unavailable: {exc}")
    return rebound


def _random_state(rng: np.random.Generator, n_planets: int = 4):
    masses = [BODY_MASSES["sun"]]
    positions = [rng.normal(0.0, 1.0e7, size=3)]
    for name in ["mercury barycenter", "venus barycenter", "earth barycenter", "mars barycenter"][:n_planets]:
        masses.append(BODY_MASSES[name])
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = rng.uniform(0.3, 1.6) * AU_M
        positions.append(positions[0] + direction * radius)
    return np.asarray(positions), np.asarray(masses)


class ReboundVariationApiTests(unittest.TestCase):
    def test_arbitrary_cartesian_variation_assignment_readback(self) -> None:
        rebound = _require_rebound()
        behavior = verify_rebound_variation_api(rebound)
        print(
            "REBOUND variation API:",
            behavior.rebound_version,
            behavior.arbitrary_cartesian_assignment_supported,
            behavior.readback_exact,
            behavior.n_real,
            behavior.n_total_after_variation,
            behavior.n_variation_particles,
        )
        self.assertEqual(behavior.rebound_version, "4.6.0")
        self.assertTrue(behavior.arbitrary_cartesian_assignment_supported)
        self.assertTrue(behavior.readback_exact)
        self.assertEqual(behavior.n_real, 2)
        self.assertEqual(behavior.n_variation_particles, 2)


class GrPotentialJacobianTests(unittest.TestCase):
    def test_jacobian_matches_centered_finite_difference(self) -> None:
        rng = np.random.default_rng(123)
        for _ in range(8):
            positions, masses = _random_state(rng, n_planets=4)
            direction = rng.normal(size=positions.shape)
            direction -= np.mean(direction, axis=0)
            direction /= np.linalg.norm(direction)
            _, analytic = gr_potential_accelerations_and_tangent(
                positions,
                masses,
                direction,
                gravitational_constant=G_SI,
                include_central_response=True,
            )
            self.assertIsNotNone(analytic)
            best = math.inf
            for epsilon in [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0]:
                plus, _ = gr_potential_accelerations_and_tangent(
                    positions + epsilon * direction,
                    masses,
                    None,
                    gravitational_constant=G_SI,
                    include_central_response=True,
                )
                minus, _ = gr_potential_accelerations_and_tangent(
                    positions - epsilon * direction,
                    masses,
                    None,
                    gravitational_constant=G_SI,
                    include_central_response=True,
                )
                finite_difference = (plus - minus) / (2.0 * epsilon)
                denom = max(np.linalg.norm(analytic), 1.0e-30)
                best = min(best, float(np.linalg.norm(finite_difference - analytic) / denom))
            self.assertLess(best, 1.0e-6)

    def test_translation_rotation_central_response_and_zero_limit(self) -> None:
        rng = np.random.default_rng(456)
        positions, masses = _random_state(rng, n_planets=3)
        accelerations, _ = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            None,
            gravitational_constant=G_SI,
            include_central_response=True,
        )
        net_force = np.sum(accelerations * masses[:, None], axis=0)
        force_scale = np.sum(np.linalg.norm(accelerations * masses[:, None], axis=1))
        self.assertLess(np.linalg.norm(net_force) / max(force_scale, 1.0), 1.0e-14)

        translation = np.ones_like(positions) * np.array([1.0, -2.0, 0.5])
        _, translated_tangent = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            translation,
            gravitational_constant=G_SI,
            include_central_response=True,
        )
        self.assertLess(np.linalg.norm(translated_tangent), 1.0e-30)

        omega = np.array([0.2, -0.1, 0.3])
        rotation_delta = np.cross(np.broadcast_to(omega, positions.shape), positions)
        _, rotation_tangent = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            rotation_delta,
            gravitational_constant=G_SI,
            include_central_response=True,
        )
        expected = np.cross(np.broadcast_to(omega, accelerations.shape), accelerations)
        self.assertLess(np.linalg.norm(rotation_tangent - expected) / max(np.linalg.norm(expected), 1.0e-30), 1.0e-12)

        zero_acc, zero_tangent = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            rotation_delta,
            gravitational_constant=G_SI,
            coefficient_scale=0.0,
            include_central_response=True,
        )
        self.assertEqual(float(np.linalg.norm(zero_acc)), 0.0)
        self.assertEqual(float(np.linalg.norm(zero_tangent)), 0.0)


class NewtonianZeroLimitTests(unittest.TestCase):
    def test_zero_gr_variation_matches_native_rebound(self) -> None:
        rebound = _require_rebound()
        masses = np.array([BODY_MASSES["sun"], BODY_MASSES["jupiter barycenter"]], dtype=float)
        radius = 5.2 * AU_M
        speed = math.sqrt(G_SI * np.sum(masses) / radius)
        state = NBodyState(
            positions=np.array(
                [[-masses[1] / np.sum(masses) * radius, 0.0, 0.0], [masses[0] / np.sum(masses) * radius, 0.0, 0.0]],
                dtype=float,
            ),
            velocities=np.array(
                [[0.0, -masses[1] / np.sum(masses) * speed, 0.0], [0.0, masses[0] / np.sum(masses) * speed, 0.0]],
                dtype=float,
            ),
            masses=masses,
        )
        native = build_rebound_simulation(rebound, state, integrator="whfast", step_s=16.0 * DAY_S, ias15_epsilon=1.0e-10)
        custom = build_rebound_simulation(rebound, state, integrator="whfast", step_s=16.0 * DAY_S, ias15_epsilon=1.0e-10)
        native_var = native.add_variation()
        custom_var = custom.add_variation()
        known = [(1.0, 2.0, 3.0, 4.0e-3, 5.0e-3, 6.0e-3), (-1.0, -2.0, 0.5, -4.0e-3, 2.0e-3, -1.0e-3)]
        for particles in [native_var.particles, custom_var.particles]:
            for particle, values in zip(particles, known):
                particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz = values
        attach_gr_potential_tangent_force(custom, coefficient_scale=0.0)
        for years in [1.0, 5.0, 10.0]:
            target = years * 365.25 * DAY_S
            native.integrate(target, exact_finish_time=1)
            custom.integrate(target, exact_finish_time=1)
            for n_particle, c_particle in zip(native_var.particles, custom_var.particles):
                native_values = np.array([n_particle.x, n_particle.y, n_particle.z, n_particle.vx, n_particle.vy, n_particle.vz])
                custom_values = np.array([c_particle.x, c_particle.y, c_particle.z, c_particle.vx, c_particle.vy, c_particle.vz])
                self.assertLess(np.linalg.norm(native_values - custom_values), 1.0e-8 * max(np.linalg.norm(native_values), 1.0))


if __name__ == "__main__":
    unittest.main()
