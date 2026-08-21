"""Tests for the windowed-slope estimator.

The point of these is that the new estimator must NOT be the old line-fit bug
wearing a hat. The line fit ran from t = 0 and reported the slope of A ln(t) as
an exponent. The windowed slope removes the additive transient but has to earn
the right to report a number, and the cross-window spread is what it earns it
with: for logarithmic growth the window slopes fall off as 1/t and disagree
enormously, so `converged` is False and no lambda is returned.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    SLOPE_WINDOW_MAX_SPREAD,
    analyze_growth,
    analyze_window_slopes,
)


class PureExponential(unittest.TestCase):
    """S = lambda t. The windows must agree and recover lambda exactly."""

    def setUp(self) -> None:
        self.lam = 7.19e-8
        self.times = np.linspace(1.0e6, 4.0e8, 2000)
        self.growth = self.lam * self.times

    def test_recovers_lambda(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(
            result.lambda_estimate_1_per_year / self.lam, 1.0, places=9
        )

    def test_windows_agree(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertLess(result.relative_spread, 1.0e-9)

    def test_lyapunov_time(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertAlmostEqual(result.lyapunov_time_years, 1.0 / self.lam, places=3)


class PureLogarithmic(unittest.TestCase):
    """S = A ln t -- regular motion. This is what broke the original estimator."""

    def setUp(self) -> None:
        self.times = np.linspace(1.0e6, 4.0e8, 2000)
        self.growth = np.log(self.times)

    def test_refuses_to_report_a_lambda(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertFalse(result.converged)
        self.assertTrue(math.isnan(result.lambda_estimate_1_per_year))
        self.assertTrue(math.isnan(result.lyapunov_time_years))

    def test_window_slopes_fall_off(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        slopes = result.slopes_1_per_year
        for earlier, later in zip(slopes[:-1], slopes[1:]):
            self.assertLess(later, earlier)

    def test_spread_is_large(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertGreater(result.relative_spread, 0.3)

    def test_note_names_the_failure(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        self.assertTrue(any("logarithmic" in note for note in result.notes))


class TransientPlusExponential(unittest.TestCase):
    """S = A ln t + lambda t, the real case. Reproduces the Pluto measurement.

    With lambda = 7.19e-8 per year over 400 Myr, A ln T is about 20 and
    lambda T is about 29, so S(T)/T is badly contaminated while the window
    slopes are not.
    """

    def setUp(self) -> None:
        self.lam = 7.19e-8
        self.times = np.linspace(1.0e6, 4.0e8, 4000)
        self.growth = np.log(self.times) + self.lam * self.times

    def test_residual_bias_is_exactly_the_predicted_logarithmic_term(self) -> None:
        """The windowed slope is biased high, and by a computable amount.

        Over a window [t1, t2] with S = A ln t + lambda t,

            slope = lambda + A ln(t2/t1) / (t2 - t1)

        so the estimate always overshoots lambda while A > 0 -- which it is for
        any Hamiltonian flow, since the tangent vector does not shrink on
        average. The reported lambda is therefore an UPPER bound and the
        reported Lyapunov time a LOWER bound. Here the median window runs from
        2e8 to 3e8 yr, giving a predicted bias of ln(1.5)/1e8 = 4.05e-9, about
        5.6% of lambda. This asserts that prediction rather than tolerating the
        discrepancy.
        """
        result = analyze_window_slopes(self.times, self.growth)
        self.assertTrue(result.converged)
        edges = result.window_edges_years
        lo, hi = edges[1], edges[2]                      # the median window
        predicted_bias = math.log(hi / lo) / (hi - lo)   # A = 1 in this fixture
        recovered = result.lambda_estimate_1_per_year
        self.assertAlmostEqual(
            (recovered - self.lam) / predicted_bias, 1.0, places=6
        )
        self.assertGreater(recovered, self.lam)          # always an upper bound

    def test_recovers_lambda_within_ten_percent(self) -> None:
        result = analyze_window_slopes(self.times, self.growth)
        error = abs(result.lambda_estimate_1_per_year - self.lam) / self.lam
        self.assertLess(error, 0.10)

    def test_the_ratio_statistic_is_badly_biased(self) -> None:
        """S(T)/T overstates lambda by a third here, which is the whole point."""
        legacy = analyze_growth(self.times, self.growth).lambda_running_final
        self.assertGreater(legacy / self.lam, 1.25)

    def test_the_ratio_statistic_drifts_with_duration(self) -> None:
        """Same dynamics, longer run, different answer -- the reported artifact."""
        half = len(self.times) // 2
        short = analyze_growth(self.times[:half], self.growth[:half])
        full = analyze_growth(self.times, self.growth)
        self.assertGreater(
            abs(full.lambda_running_final - short.lambda_running_final)
            / short.lambda_running_final,
            0.15,
        )

    def test_the_windowed_slope_drifts_far_less_than_the_ratio(self) -> None:
        """Both statistics move as the run lengthens; one moves much less.

        The windowed slope still drifts, because its residual logarithmic bias
        shrinks as the windows move later. The claim worth making is relative:
        that drift is several times smaller than the ratio statistic's, on
        identical data.
        """
        half = len(self.times) // 2
        short = analyze_window_slopes(self.times[:half], self.growth[:half])
        full = analyze_window_slopes(self.times, self.growth)
        self.assertTrue(short.converged and full.converged)
        slope_drift = abs(
            full.lambda_estimate_1_per_year - short.lambda_estimate_1_per_year
        ) / short.lambda_estimate_1_per_year

        ratio_short = analyze_growth(
            self.times[:half], self.growth[:half]
        ).lambda_running_final
        ratio_full = analyze_growth(self.times, self.growth).lambda_running_final
        ratio_drift = abs(ratio_full - ratio_short) / ratio_short

        self.assertLess(slope_drift, ratio_drift / 3.0)
        self.assertLess(slope_drift, 0.10)


class PlutoMeasurement(unittest.TestCase):
    """The overnight numbers, held as a regression.

    dt = 0.4, five tangent seeds, cumulative log growth at 100/200/400 Myr
    reconstructed from the published summaries. Every seed gave the same window
    slope to five significant figures; only the additive transient differed.
    """

    SEEDS = (
        (23.662, 30.780, 45.154),
        (20.211, 27.176, 41.550),
        (24.693, 31.811, 46.186),
        (23.634, 30.747, 45.122),
        (24.049, 31.162, 45.537),
    )

    def test_every_seed_gives_the_same_slope(self) -> None:
        slopes = [(s400 - s200) / 2.0e8 for _s100, s200, s400 in self.SEEDS]
        self.assertLess((max(slopes) - min(slopes)) / np.median(slopes), 0.005)

    def test_slope_lands_in_the_preregistered_window(self) -> None:
        slopes = [(s400 - s200) / 2.0e8 for _s100, s200, s400 in self.SEEDS]
        lyapunov_time = 1.0 / float(np.median(slopes))
        self.assertGreater(lyapunov_time, 1.0e7)
        self.assertLess(lyapunov_time, 4.0e7)

    def test_consecutive_windows_agree(self) -> None:
        """100-200 Myr against 200-400 Myr: converged in time, at this timestep."""
        for s100, s200, s400 in self.SEEDS:
            early = (s200 - s100) / 1.0e8
            late = (s400 - s200) / 2.0e8
            with self.subTest(seed=s100):
                self.assertLess(abs(late - early) / early, SLOPE_WINDOW_MAX_SPREAD)

    def test_the_ratio_statistic_moved_with_duration(self) -> None:
        """6.50 Myr at 200 Myr, 8.86 Myr at 400 Myr, same run."""
        for _s100, s200, s400 in self.SEEDS:
            with self.subTest(seed=s200):
                ratio_200 = s200 / 2.0e8
                ratio_400 = s400 / 4.0e8
                self.assertGreater(abs(ratio_400 - ratio_200) / ratio_200, 0.20)


class InputValidation(unittest.TestCase):
    def test_needs_at_least_two_windows(self) -> None:
        times = np.linspace(1.0, 100.0, 100)
        with self.assertRaises(ValueError):
            analyze_window_slopes(times, times, n_windows=1)

    def test_needs_enough_samples(self) -> None:
        times = np.linspace(1.0, 10.0, 8)
        with self.assertRaises(ValueError):
            analyze_window_slopes(times, times, n_windows=3)

    def test_rejects_nonincreasing_times(self) -> None:
        times = np.array([1.0, 3.0, 2.0] * 10)
        with self.assertRaises(ValueError):
            analyze_window_slopes(times, np.arange(30, dtype=float))

    def test_rejects_non_finite(self) -> None:
        times = np.linspace(1.0, 100.0, 100)
        growth = times.copy()
        growth[50] = float("nan")
        with self.assertRaises(ValueError):
            analyze_window_slopes(times, growth)

    def test_flat_growth_does_not_report_a_lambda(self) -> None:
        times = np.linspace(1.0, 100.0, 100)
        result = analyze_window_slopes(times, np.zeros_like(times))
        self.assertFalse(result.converged)
        self.assertTrue(math.isnan(result.lambda_estimate_1_per_year))


if __name__ == "__main__":
    unittest.main()
