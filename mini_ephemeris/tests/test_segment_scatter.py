"""Error bars from the record itself, validated against a known answer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.chaos_estimator_diagnostics import segment_scatter  # noqa: E402
from mini_ephemeris.ladder_rungs import (  # noqa: E402
    FAST_CHAOS_LAMBDA,
    FAST_CHAOS_RECORD,
)


class AgainstTheKnownAnswer(unittest.TestCase):
    """The whole point: does the spread predict the actual error?"""

    def setUp(self) -> None:
        table = np.loadtxt(FAST_CHAOS_RECORD, delimiter=",", skiprows=1)
        self.times, self.growth = table[:, 0], table[:, 1]

    def test_full_record_recovers_the_known_exponent(self) -> None:
        result = segment_scatter(self.times, self.growth, n_segments=4)
        error = abs(result.full_record_lambda_1_per_year - FAST_CHAOS_LAMBDA)
        self.assertLess(error / FAST_CHAOS_LAMBDA, 0.10)

    def test_the_true_error_lies_within_about_one_sigma(self) -> None:
        result = segment_scatter(self.times, self.growth, n_segments=4)
        error = abs(
            result.full_record_lambda_1_per_year - FAST_CHAOS_LAMBDA
        ) / FAST_CHAOS_LAMBDA
        self.assertLess(error, 2.0 * result.relative_spread)

    def test_the_quoted_uncertainty_is_not_absurd(self) -> None:
        result = segment_scatter(self.times, self.growth, n_segments=4)
        self.assertGreater(result.relative_spread, 0.01)
        self.assertLess(result.relative_spread, 0.60)

    def test_more_segments_means_shorter_segments_and_wider_spread(self) -> None:
        """Precision per segment falls as segments shorten. Sanity, not surprise."""
        few = segment_scatter(self.times, self.growth, n_segments=3)
        many = segment_scatter(self.times, self.growth, n_segments=12)
        self.assertLess(few.segment_length_years, many.segment_length_years * 5)
        self.assertGreater(many.relative_spread, few.relative_spread * 0.5)


class CleanSignals(unittest.TestCase):
    def test_a_perfect_exponential_has_almost_no_spread(self) -> None:
        times = np.linspace(1.0e6, 4.0e8, 4000)
        result = segment_scatter(times, 7.19e-8 * times, n_segments=4)
        self.assertLess(result.relative_spread, 1.0e-6)
        self.assertAlmostEqual(
            result.full_record_lambda_1_per_year / 7.19e-8, 1.0, places=8
        )

    def test_notes_quote_the_uncertainty(self) -> None:
        times = np.linspace(1.0e6, 4.0e8, 4000)
        result = segment_scatter(times, 7.19e-8 * times, n_segments=4)
        self.assertTrue(any("1-sigma" in note for note in result.notes))


class InputValidation(unittest.TestCase):
    def test_one_segment_is_rejected(self) -> None:
        times = np.linspace(1.0, 100.0, 400)
        with self.assertRaises(ValueError):
            segment_scatter(times, times, n_segments=1)

    def test_too_short_a_record_is_rejected(self) -> None:
        times = np.linspace(1.0, 100.0, 20)
        with self.assertRaises(ValueError):
            segment_scatter(times, times, n_segments=8)

    def test_nonincreasing_times_rejected(self) -> None:
        times = np.array([1.0, 3.0, 2.0] * 40)
        with self.assertRaises(ValueError):
            segment_scatter(times, np.arange(120, dtype=float), n_segments=3)


if __name__ == "__main__":
    unittest.main()
