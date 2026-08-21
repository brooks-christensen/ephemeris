"""The combined estimator, and the consistency criterion Codex asked for.

The specification I wrote asked for "the two estimators to agree" without
saying agree to what, and their defaults did not read the same data -- the
windowed slope discarded the first 25% of the record and the two-term fit the
first 5%. Codex refused to launch an 800 Myr run against that, correctly.

These tests pin the repair: one discard fraction for both, and a criterion that
is a statement about the model rather than a tolerance band. The windowed slope
must sit where the fitted transient predicts it should -- above the fitted
lambda by A times the least-squares slope of ln t over that window -- because
that is what the two-term form implies. A record the form does not describe
fails it even when both estimators happen to return similar numbers.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    ESTIMATOR_DISCARD_FRACTION,
    MEGNO_AGREEMENT_TOLERANCE,
    WINDOW_CONSISTENCY_TOLERANCE,
    estimate_lyapunov_exponent,
    fit_transient_and_exponent,
    analyze_window_slopes,
)

LAMBDA = 7.19e-8


def clean_record(years: float = 8.0e8, n: int = 4000, amplitude: float = 1.0):
    times = np.linspace(1.0e6, years, n)
    growth = amplitude * np.log(times) + LAMBDA * times + 14.0
    return times, growth


class BothEstimatorsSeeTheSameData(unittest.TestCase):
    def test_one_discard_fraction_is_used_for_both(self) -> None:
        times, growth = clean_record()
        result = estimate_lyapunov_exponent(times, growth)
        self.assertEqual(result.discard_fraction, ESTIMATOR_DISCARD_FRACTION)

        # The reported windows must be the ones the shared fraction produces.
        windows = analyze_window_slopes(
            times, growth, discard_fraction=ESTIMATOR_DISCARD_FRACTION
        )
        self.assertEqual(
            result.window_slopes_1_per_year, windows.slopes_1_per_year
        )

    def test_the_old_mismatched_defaults_would_have_differed(self) -> None:
        """Why this mattered: 5% and 25% are not the same record."""
        times, growth = clean_record()
        five = fit_transient_and_exponent(times, growth, discard_fraction=0.05)
        twenty_five = fit_transient_and_exponent(
            times, growth, discard_fraction=ESTIMATOR_DISCARD_FRACTION
        )
        self.assertNotEqual(five.n_samples, twenty_five.n_samples)


class ConsistencyIsAModelStatement(unittest.TestCase):
    def test_a_clean_two_term_record_is_consistent(self) -> None:
        times, growth = clean_record()
        result = estimate_lyapunov_exponent(times, growth)
        self.assertTrue(result.consistent)
        self.assertLess(result.split_half_disagreement, WINDOW_CONSISTENCY_TOLERANCE)
        self.assertLess(result.window_consistency, 1.0e-6)
        self.assertAlmostEqual(result.lambda_1_per_year / LAMBDA, 1.0, places=6)

    def test_window_slopes_sit_above_the_fitted_lambda(self) -> None:
        """The estimators must not agree exactly -- the offset is predictable."""
        times, growth = clean_record()
        result = estimate_lyapunov_exponent(times, growth)
        for measured in result.window_slopes_1_per_year:
            self.assertGreater(measured, result.lambda_1_per_year)
        for measured, predicted in zip(
            result.window_slopes_1_per_year,
            result.predicted_window_slopes_1_per_year,
        ):
            self.assertAlmostEqual(measured / predicted, 1.0, places=6)

    def test_a_record_the_model_does_not_describe_is_rejected(self) -> None:
        """Stretched-exponential growth: both estimators return numbers anyway."""
        times = np.linspace(1.0e6, 8.0e8, 4000)
        growth = 14.0 + 3.0e-5 * times ** 0.75      # not A ln t + lambda t
        result = estimate_lyapunov_exponent(times, growth)

        # The window-versus-prediction check does NOT catch this: the two-term
        # form approximates t^0.75 well enough locally that the window slopes
        # land where it predicts. Split-half stability of the fit is what
        # catches it, because the local rate falls as t^-0.25.
        self.assertLess(result.window_consistency, WINDOW_CONSISTENCY_TOLERANCE)
        self.assertGreater(result.split_half_disagreement, WINDOW_CONSISTENCY_TOLERANCE)
        self.assertFalse(result.consistent)
        self.assertTrue(math.isnan(result.lambda_1_per_year))

    def test_no_lambda_is_returned_when_inconsistent(self) -> None:
        times = np.linspace(1.0e6, 1.0e8, 400)
        growth = 2.0 * np.log(times)                 # regular motion
        result = estimate_lyapunov_exponent(times, growth)
        self.assertFalse(result.consistent)
        self.assertTrue(math.isnan(result.lyapunov_time_years))

    def test_a_short_record_is_rejected_on_trend_to_scatter(self) -> None:
        rng = np.random.default_rng(11)
        times = np.linspace(1.0e6, 1.0e8, 400)
        growth = (np.log(times) + LAMBDA * times + 14.0
                  + 1.3 * np.sin(2 * np.pi * times / 3.0e7)
                  + 0.5 * rng.standard_normal(times.size))
        result = estimate_lyapunov_exponent(times, growth)
        # Threshold-independent claims: the record is rejected and says why.
        self.assertFalse(result.consistent)
        self.assertNotEqual(result.notes, ())
        self.assertTrue(math.isnan(result.lambda_1_per_year))


class MegnoAsAThirdEstimator(unittest.TestCase):
    def test_consistent_megno_passes(self) -> None:
        times, growth = clean_record()
        megno = LAMBDA * times / 2.0                 # <Y> -> lambda t / 2
        result = estimate_lyapunov_exponent(times, growth, mean_megno=megno)
        self.assertIsNotNone(result.megno_lambda_1_per_year)
        self.assertLess(result.megno_disagreement, 1.0e-6)
        self.assertTrue(result.consistent)

    def test_megno_off_by_the_old_factor_is_caught(self) -> None:
        """The 0.5 factor this project shipped would show up here as 4x."""
        times, growth = clean_record()
        megno = LAMBDA * times / 2.0
        result = estimate_lyapunov_exponent(
            times, growth, mean_megno=megno * 0.25
        )
        self.assertGreater(result.megno_disagreement, MEGNO_AGREEMENT_TOLERANCE)
        self.assertFalse(result.consistent)

    def test_length_mismatch_raises(self) -> None:
        times, growth = clean_record()
        with self.assertRaises(ValueError):
            estimate_lyapunov_exponent(times, growth, mean_megno=np.zeros(7))

    def test_megno_is_optional(self) -> None:
        times, growth = clean_record()
        result = estimate_lyapunov_exponent(times, growth)
        self.assertIsNone(result.megno_lambda_1_per_year)
        self.assertIsNone(result.megno_disagreement)
        self.assertTrue(result.consistent)


class LongEnoughRecord(unittest.TestCase):
    """800 Myr passes where 200 Myr does not, with the measured scatter."""

    @staticmethod
    def _record(years: float, seed: int = 5):
        rng = np.random.default_rng(seed)
        n = int(years / 2.0e5)
        times = np.linspace(1.0e6, years, n)
        growth = (np.log(times) + LAMBDA * times + 14.0
                  + 1.3 * np.sin(2 * np.pi * times / 3.0e7)
                  + 0.5 * rng.standard_normal(times.size))
        return times, growth

    def test_two_hundred_megayears_is_not_enough(self) -> None:
        result = estimate_lyapunov_exponent(*self._record(2.0e8))
        self.assertFalse(result.consistent)

    def test_eight_hundred_megayears_is(self) -> None:
        result = estimate_lyapunov_exponent(*self._record(8.0e8))
        self.assertTrue(result.consistent)
        self.assertLess(abs(result.lambda_1_per_year - LAMBDA) / LAMBDA, 0.10)


if __name__ == "__main__":
    unittest.main()
