"""Pin the MEGNO convention as algebra, so it cannot drift silently.

The conversion factor from a MEGNO slope to a Lyapunov exponent was 0.5 in this
repository, which is wrong under both conventions. It is now 2.0. The evidence
for 2.0 is not a fit -- it is the definition REBOUND implements, verified
against REBOUND 4.6.0 by reconstructing <Y> from the tangent vector alone
(scripts/measure_megno_convention.py). These tests hold the algebra in place
without needing REBOUND, SciPy or a kernel.

They also pin the safeguard that stops the previous calibration mistake from
recurring: Simulation.lyapunov() is the least-squares slope of
Simulation.megno(), so using it as the "independent" lambda divides a quantity
by itself and reports 1.0 -- a confident wrong answer that would have halved
every Lyapunov exponent in the project.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    MEGNO_INSTANTANEOUS_TO_LYAPUNOV,
    MEGNO_MEAN_TO_LYAPUNOV,
    REBOUND_LYAPUNOV_TO_LAMBDA,
    calibrate_megno_factor,
    lambda_from_rebound_lyapunov,
    megno_from_log_tangent,
)


class ChaoticLimit(unittest.TestCase):
    """L = lambda t  =>  Y = lambda t  and  <Y> = lambda t / 2, exactly."""

    def setUp(self) -> None:
        self.lam = 3.0e-3
        self.times = np.linspace(0.0, 2000.0, 4001)
        self.log_norm = self.lam * self.times

    def test_instantaneous_Y_is_lambda_t(self) -> None:
        y_inst, _ = megno_from_log_tangent(self.times, self.log_norm)
        np.testing.assert_allclose(y_inst, self.lam * self.times, rtol=1e-12, atol=1e-12)

    def test_mean_Y_is_half_lambda_t(self) -> None:
        _, y_mean = megno_from_log_tangent(self.times, self.log_norm)
        np.testing.assert_allclose(
            y_mean, 0.5 * self.lam * self.times, rtol=1e-12, atol=1e-12
        )

    def test_mean_slope_times_two_recovers_lambda(self) -> None:
        _, y_mean = megno_from_log_tangent(self.times, self.log_norm)
        slope = float(np.polyfit(self.times, y_mean, 1)[0])
        self.assertAlmostEqual(slope * MEGNO_MEAN_TO_LYAPUNOV, self.lam, places=12)

    def test_instantaneous_slope_times_one_recovers_lambda(self) -> None:
        y_inst, _ = megno_from_log_tangent(self.times, self.log_norm)
        slope = float(np.polyfit(self.times, y_inst, 1)[0])
        self.assertAlmostEqual(
            slope * MEGNO_INSTANTANEOUS_TO_LYAPUNOV, self.lam, places=12
        )

    def test_the_discarded_factor_would_be_wrong_by_four(self) -> None:
        """The old 0.5 is not a near miss: it is 4x off the mean convention."""
        _, y_mean = megno_from_log_tangent(self.times, self.log_norm)
        slope = float(np.polyfit(self.times, y_mean, 1)[0])
        self.assertAlmostEqual(slope * 0.5, self.lam / 4.0, places=12)

    def test_tangent_normalization_is_irrelevant(self) -> None:
        """A constant offset in ln|delta| cancels; delta0 need not be a unit vector."""
        _, base = megno_from_log_tangent(self.times, self.log_norm)
        _, shifted = megno_from_log_tangent(self.times, self.log_norm + 12.75)
        np.testing.assert_allclose(shifted, base, rtol=1e-10, atol=1e-10)


class RegularLimit(unittest.TestCase):
    """Regular motion: |delta| ~ t, so Y -> 2 and <Y> -> 2.

    This is the signature that identifies the Cincotta-Simo normalization --
    the factor of two inside Y is exactly what makes quasi-periodic motion sit
    at 2 rather than at 1.
    """

    def setUp(self) -> None:
        self.times = np.linspace(0.0, 5000.0, 20001)
        self.log_norm = np.log1p(self.times)

    def test_instantaneous_Y_tends_to_two(self) -> None:
        y_inst, _ = megno_from_log_tangent(self.times, self.log_norm)
        self.assertAlmostEqual(float(y_inst[-1]), 2.0, places=2)

    def test_mean_Y_tends_to_two(self) -> None:
        """<Y> approaches 2 from below as 2 - (ln t)^2 / t: slow, but monotone.

        The slowness is the point. A run that stops at <Y> = 1.98 has not shown
        chaos and has not shown regularity either; it has shown that the double
        time average has not converged. That is why the calibration script
        refuses to work with a system whose final <Y> is below 5.
        """
        _, y_mean = megno_from_log_tangent(self.times, self.log_norm)
        self.assertLess(abs(float(y_mean[-1]) - 2.0), 0.02)

        longer = np.linspace(0.0, 50000.0, 50001)
        _, y_mean_longer = megno_from_log_tangent(longer, np.log1p(longer))
        self.assertLess(
            abs(float(y_mean_longer[-1]) - 2.0), abs(float(y_mean[-1]) - 2.0)
        )
        self.assertLess(abs(float(y_mean_longer[-1]) - 2.0), 0.005)

    def test_mean_Y_slope_is_negligible(self) -> None:
        """No exponent to report: the slope is ~ln(t)/t^2, not a lambda."""
        _, y_mean = megno_from_log_tangent(self.times, self.log_norm)
        half = len(self.times) // 2
        slope = float(np.polyfit(self.times[half:], y_mean[half:], 1)[0])
        self.assertLess(abs(slope) * MEGNO_MEAN_TO_LYAPUNOV, 1.0e-4)


class ReboundLyapunovIsNotIndependent(unittest.TestCase):
    """The circularity that broke the first calibration attempt."""

    def test_source_argument_is_required(self) -> None:
        with self.assertRaises(TypeError):
            calibrate_megno_factor(1.0e-7, 2.0e-7)  # type: ignore[call-arg]

    def test_rebound_lyapunov_is_rejected_by_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            calibrate_megno_factor(
                1.0e-7, 1.0e-7, lambda_source="rebound_lyapunov"
            )
        self.assertIn("not independent", str(ctx.exception))

    def test_megno_derived_lambda_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_megno_factor(1.0e-7, 2.0e-7, lambda_source="megno")

    def test_unknown_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_megno_factor(1.0e-7, 2.0e-7, lambda_source="vibes")

    def test_provenance_is_recorded_in_the_payload(self) -> None:
        result = calibrate_megno_factor(
            1.0e-7, 2.0e-7, lambda_source="shadow_orbit"
        )
        self.assertEqual(result["lambda_source"], "shadow_orbit")

    def test_the_circular_calculation_would_have_reported_instantaneous(self) -> None:
        """Demonstrate the failure the guard now prevents.

        sim.lyapunov() is the OLS slope of <Y> over the whole record; the
        calibration fitted the slope of <Y> over its second half. On a chaotic
        system both are lambda/2, so their ratio is 1.0 -- indistinguishable
        from the instantaneous convention, and a 2x error in every exponent.
        """
        lam = 4.0e-4
        times = np.linspace(0.0, 5000.0, 10001)
        _, y_mean = megno_from_log_tangent(times, lam * times)
        whole_record_slope = float(np.polyfit(times, y_mean, 1)[0])
        half = len(times) // 2
        tail_slope = float(np.polyfit(times[half:], y_mean[half:], 1)[0])
        circular = whole_record_slope / tail_slope
        self.assertAlmostEqual(circular, 1.0, places=6)
        self.assertAlmostEqual(circular, MEGNO_INSTANTANEOUS_TO_LYAPUNOV, places=6)


class ReboundLyapunovConversion(unittest.TestCase):
    def test_factor_is_two(self) -> None:
        self.assertEqual(REBOUND_LYAPUNOV_TO_LAMBDA, 2.0)

    def test_doubling(self) -> None:
        self.assertAlmostEqual(lambda_from_rebound_lyapunov(3.5e-6), 7.0e-6, places=15)

    def test_non_finite_input(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                self.assertTrue(np.isnan(lambda_from_rebound_lyapunov(value)))

    def test_recovers_lambda_from_a_whole_record_slope(self) -> None:
        """What lyapunov() computes, and what it takes to make it an exponent."""
        lam = 8.0e-5
        times = np.linspace(0.0, 20000.0, 20001)
        _, y_mean = megno_from_log_tangent(times, lam * times)
        native = float(np.polyfit(times, y_mean, 1)[0])   # what lyapunov() returns
        self.assertAlmostEqual(native, lam / 2.0, places=12)
        self.assertAlmostEqual(lambda_from_rebound_lyapunov(native), lam, places=12)


class InputValidation(unittest.TestCase):
    def test_times_must_start_at_zero(self) -> None:
        times = np.linspace(1.0, 100.0, 50)
        with self.assertRaises(ValueError):
            megno_from_log_tangent(times, times)

    def test_times_must_increase(self) -> None:
        times = np.array([0.0, 2.0, 1.0, 3.0])
        with self.assertRaises(ValueError):
            megno_from_log_tangent(times, times)

    def test_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            megno_from_log_tangent(np.linspace(0.0, 1.0, 10), np.zeros(9))

    def test_rejects_non_finite(self) -> None:
        times = np.linspace(0.0, 10.0, 11)
        log_norm = times.copy()
        log_norm[5] = float("nan")
        with self.assertRaises(ValueError):
            megno_from_log_tangent(times, log_norm)

    def test_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            megno_from_log_tangent([0.0, 1.0], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
