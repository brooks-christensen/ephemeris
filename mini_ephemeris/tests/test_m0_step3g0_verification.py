from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.gr_potential_tangent import gr_potential_accelerations_and_tangent
from mini_ephemeris.gr_potential_tangent_c import load_c_backend
from mini_ephemeris.m0_step3g0_verification import (
    C_M_PER_S,
    Step3g0AuditError,
    audit_initial_physical_state,
    callback_accounting_model,
    complex_step_gr_pair_jvp,
    decimal_gr_pair_central_difference,
    decimal_gr_pair_oracle,
    direction_angles,
    gr_system_oracle,
    inspect_archive_readonly,
    newtonian_accelerations,
    no_integration_guard,
    recompute_frozen_conservation,
    recompute_frozen_orientation,
    restart_callback_accounting_model,
)
from mini_ephemeris.nbody import G_SI


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "stability" / "m0_step3f1_two_lane_architecture_screen_v1"
LANE_P = OUTPUT / "m0_step3f1_physical_whckl_0p25d_10k"
LANE_T = OUTPUT / "m0_step3f1_tangent_whfast_0p25d_10k"


class Step3g0VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard_context = no_integration_guard()
        self.guard = self.guard_context.__enter__()

    def tearDown(self) -> None:
        try:
            self.guard.assert_unused()
        finally:
            self.guard_context.__exit__(None, None, None)

    def test_guard_intercepts_without_executing_integrator(self) -> None:
        import rebound

        simulation = rebound.Simulation()
        with self.assertRaisesRegex(Step3g0AuditError, "prohibits"):
            simulation.integrate(1.0)
        self.guard.calls.clear()

    def test_newtonian_two_body_and_momentum_closure(self) -> None:
        masses = np.asarray([2.0e30, 3.0e24])
        positions = np.asarray([[0.0, 0.0, 0.0], [2.0e11, 0.0, 0.0]])
        acceleration = newtonian_accelerations(positions, masses)
        expected_left = G_SI * masses[1] / positions[1, 0] ** 2
        expected_right = -G_SI * masses[0] / positions[1, 0] ** 2
        self.assertEqual(acceleration[0, 0], expected_left)
        self.assertEqual(acceleration[1, 0], expected_right)
        np.testing.assert_allclose(np.sum(masses[:, None] * acceleration, axis=0), 0.0, atol=1.0e-18)

    def test_gr_pair_and_jvp_against_decimal_oracle(self) -> None:
        position = np.asarray([5.7e10, -1.1e10, 2.5e9])
        delta = np.asarray([3.0e3, -4.0e3, 1.2e3])
        masses = np.asarray([1.9884098713264225e30, 3.3009873694619664e23])
        positions = np.vstack((np.zeros(3), position))
        deltas = np.vstack((np.zeros(3), delta))
        acceleration, tangent = gr_potential_accelerations_and_tangent(
            positions, masses, deltas, gravitational_constant=G_SI
        )
        expected_acceleration, expected_tangent = decimal_gr_pair_oracle(
            position,
            delta,
            gravitational_constant=G_SI,
            central_mass_kg=masses[0],
        )
        np.testing.assert_allclose(acceleration[1], expected_acceleration, rtol=3.0e-15)
        np.testing.assert_allclose(tangent[1], expected_tangent, rtol=4.0e-15)
        complex_step = complex_step_gr_pair_jvp(
            position,
            delta,
            gravitational_constant=G_SI,
            central_mass_kg=masses[0],
        )
        np.testing.assert_allclose(tangent[1], complex_step, rtol=4.0e-15)
        decimal_difference = decimal_gr_pair_central_difference(
            position,
            delta,
            gravitational_constant=G_SI,
            central_mass_kg=masses[0],
        )
        np.testing.assert_allclose(tangent[1], decimal_difference, rtol=4.0e-15)
        np.testing.assert_allclose(np.sum(masses[:, None] * acceleration, axis=0), 0.0, atol=4.0e-7)
        np.testing.assert_allclose(np.sum(masses[:, None] * tangent, axis=0), 0.0, atol=5.0e-9)

    def test_gr_jvp_central_difference_and_linearity(self) -> None:
        rng = np.random.default_rng(20260813)
        masses = np.asarray([1.9884098713264225e30, 3.3e23, 4.87e24, 5.97e24])
        positions = rng.normal(size=(4, 3)) * np.asarray([[1.0], [5.8e10], [1.08e11], [1.50e11]])
        u = rng.normal(size=(4, 3)) * 1.0e4
        v = rng.normal(size=(4, 3)) * 1.0e4
        _, tangent_u = gr_system_oracle(positions, masses, u)
        _, tangent_v = gr_system_oracle(positions, masses, v)
        _, tangent_combo = gr_system_oracle(positions, masses, 1.7 * u - 0.4 * v)
        np.testing.assert_allclose(tangent_combo, 1.7 * tangent_u - 0.4 * tangent_v, rtol=4.0e-15, atol=1.0e-30)
        epsilon = 1.0e-3
        plus, _ = gr_system_oracle(positions + epsilon * u, masses, None)
        minus, _ = gr_system_oracle(positions - epsilon * u, masses, None)
        finite_difference = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(tangent_u, finite_difference, rtol=2.0e-7, atol=1.0e-20)

    def test_gr_covariances_limits_and_no_backreaction(self) -> None:
        rng = np.random.default_rng(451)
        masses = np.asarray([1.9e30, 4.0e23, 5.0e24])
        positions = np.asarray([[1.0e8, -2.0e8, 4.0e8], [5.8e10, 2.0e10, -1.0e10], [-8.0e10, 7.0e10, 3.0e10]])
        delta = rng.normal(size=(3, 3)) * 2.0e3
        acceleration, tangent = gr_system_oracle(positions, masses, delta)
        shifted, shifted_tangent = gr_system_oracle(positions + np.asarray([3.0e12, -7.0e11, 9.0e10]), masses, delta)
        np.testing.assert_allclose(shifted, acceleration, rtol=3.0e-15, atol=1.0e-30)
        np.testing.assert_allclose(shifted_tangent, tangent, rtol=4.0e-15, atol=1.0e-30)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        rotated, rotated_tangent = gr_system_oracle(positions @ q.T, masses, delta @ q.T)
        np.testing.assert_allclose(rotated, acceleration @ q.T, rtol=4.0e-15, atol=1.0e-30)
        np.testing.assert_allclose(rotated_tangent, tangent @ q.T, rtol=5.0e-15, atol=1.0e-30)
        reflected = np.diag([-1.0, 1.0, 1.0])
        reflected_acceleration, _ = gr_system_oracle(positions @ reflected, masses, None)
        np.testing.assert_allclose(reflected_acceleration, acceleration @ reflected, rtol=4.0e-15, atol=1.0e-30)
        zero_acceleration, zero_tangent = gr_system_oracle(positions, masses, delta, coefficient_scale=0.0)
        self.assertTrue(np.array_equal(zero_acceleration, np.zeros_like(positions)))
        self.assertTrue(np.array_equal(zero_tangent, np.zeros_like(delta)))
        changed_delta = delta.copy()
        changed_delta[1] *= 10.0
        changed_acceleration, _ = gr_system_oracle(positions, masses, changed_delta)
        self.assertTrue(np.array_equal(changed_acceleration, acceleration))

    def test_python_c_equality_deterministic_random_cases(self) -> None:
        backend = load_c_backend()
        rng = np.random.default_rng(123456)
        for case in range(12):
            masses = np.concatenate(([1.9884098713264225e30], 10.0 ** rng.uniform(20.0, 28.0, 5)))
            radii = 10.0 ** rng.uniform(10.5, 13.0, 5)
            directions = rng.normal(size=(5, 3))
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            positions = np.vstack((rng.normal(size=3) * 1.0e9, directions * radii[:, None]))
            positions[1:] += positions[0]
            delta = rng.normal(size=(6, 3)) * 1.0e5
            expected_acceleration, expected_tangent = gr_potential_accelerations_and_tangent(
                positions, masses, delta, gravitational_constant=G_SI
            )
            actual_acceleration, actual_tangent = backend.pointwise(
                positions, masses, delta, gravitational_constant=G_SI
            )
            with self.subTest(case=case):
                np.testing.assert_allclose(actual_acceleration, expected_acceleration, rtol=4.0e-15, atol=1.0e-30)
                np.testing.assert_allclose(actual_tangent, expected_tangent, rtol=6.0e-15, atol=1.0e-30)

    def test_oracle_rejects_singular_and_nonfinite_inputs(self) -> None:
        masses = np.asarray([1.0e30, 1.0e24])
        coincident = np.zeros((2, 3))
        with self.assertRaises(ValueError):
            gr_system_oracle(coincident, masses, None)
        nonfinite = coincident.copy()
        nonfinite[1, 0] = math.nan
        with self.assertRaises(ValueError):
            gr_system_oracle(nonfinite, masses, None)

    def test_current_c_input_contract_gap_is_reproducible(self) -> None:
        backend = load_c_backend()
        masses = np.asarray([1.0e30, 1.0e24])
        coincident = np.zeros((2, 3))
        acceleration, tangent = backend.pointwise(
            coincident, masses, coincident, gravitational_constant=G_SI
        )
        self.assertTrue(np.array_equal(acceleration, np.zeros_like(coincident)))
        self.assertTrue(np.array_equal(tangent, np.zeros_like(coincident)))
        nonfinite = coincident.copy()
        nonfinite[1, 0] = math.nan
        acceleration, tangent = backend.pointwise(
            nonfinite, masses, coincident, gravitational_constant=G_SI
        )
        self.assertFalse(np.all(np.isfinite(acceleration)))
        self.assertFalse(np.all(np.isfinite(tangent)))

    def test_direction_estimators_conditioning(self) -> None:
        for angle in (0.0, 1.0e-12, math.sqrt(np.finfo(float).eps), 3.0e-8, 0.3, math.pi):
            right = np.asarray([math.cos(angle), math.sin(angle), 0.0])
            result = direction_angles([2.0, 0.0, 0.0], 7.0 * right)
            self.assertAlmostEqual(result["atan2_rad"], angle, delta=3.0e-16)
            self.assertAlmostEqual(result["chord_rad"], angle, delta=3.0e-16)
        with self.assertRaises(ValueError):
            direction_angles([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    def test_callback_schedule_exactly_recovers_observed_counts(self) -> None:
        accounting = callback_accounting_model()
        self.assertEqual(accounting["source_schedule_total"], 29_226_432)
        self.assertEqual(accounting["historical_expected"], 29_223_232)
        self.assertEqual(accounting["difference"], 3_200)
        restart = restart_callback_accounting_model()
        self.assertEqual(restart["source_schedule_total"], 292_264)
        self.assertEqual(restart["historical_expected"], 292_232)
        self.assertEqual(restart["difference"], 32)

    def test_frozen_state_recomputations_are_finite(self) -> None:
        orientation = recompute_frozen_orientation(LANE_P / "state.csv", LANE_T / "state.csv")
        self.assertEqual(orientation["comparisons"], 1818)
        self.assertLess(orientation["metrics"]["orbital_plane"]["max_atan2_chord_abs_difference"], 1.0e-12)
        conservation = recompute_frozen_conservation(LANE_P / "state.csv")
        self.assertEqual(conservation["samples"], 101)
        self.assertTrue(all(math.isfinite(value) for value in conservation.values()))
        initial = audit_initial_physical_state(LANE_P / "state.csv")
        self.assertEqual(initial["body_count"], 10)
        self.assertGreater(initial["analytic_gr_perihelion_rate_arcsec_per_century"], 42.0)
        self.assertLess(initial["analytic_gr_perihelion_rate_arcsec_per_century"], 44.0)

    def test_archives_are_read_only_and_complete(self) -> None:
        for lane in (LANE_P, LANE_T):
            audit = inspect_archive_readonly(lane / "simulationarchive.bin")
            self.assertEqual(audit["sha256_before"], audit["sha256_after"])
            self.assertEqual(audit["snapshots"], 11)
            self.assertTrue(audit["times_strictly_increasing"])


if __name__ == "__main__":
    unittest.main()
