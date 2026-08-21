"""Tests for the circular-span statistic, built around the bug it replaced."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.angle_statistics import (  # noqa: E402
    minimum_circular_span_degrees,
    naive_linear_span_degrees,
)


def libration(centre: float, amplitude: float, n: int = 2000) -> np.ndarray:
    """A clean libration: centre +/- amplitude, sampled over many cycles."""
    phase = np.linspace(0.0, 40.0 * np.pi, n)
    return centre + amplitude * np.sin(phase)


class TheBugThisReplaces(unittest.TestCase):
    """Libration about 180 degrees is where the naive wrap fails."""

    def test_naive_wrap_reports_circulation_for_a_resonant_orbit(self) -> None:
        angles = libration(180.0, 83.5)          # Pluto's actual 3:2 libration
        self.assertGreater(naive_linear_span_degrees(angles), 355.0)

    def test_circular_span_gets_it_right(self) -> None:
        angles = libration(180.0, 83.5)
        span, centre = minimum_circular_span_degrees(angles)
        self.assertAlmostEqual(span, 167.0, delta=1.0)
        self.assertAlmostEqual(centre, 180.0, delta=1.0)

    def test_the_two_disagree_by_more_than_a_factor_of_two(self) -> None:
        angles = libration(180.0, 83.5)
        naive = naive_linear_span_degrees(angles)
        span, _ = minimum_circular_span_degrees(angles)
        self.assertGreater(naive / span, 2.0)

    def test_naive_wrap_happens_to_work_about_zero(self) -> None:
        """Which is why it survived: it is right for the other centre."""
        angles = libration(0.0, 83.5)
        self.assertAlmostEqual(naive_linear_span_degrees(angles), 167.0, delta=1.0)


class LibrationAtEveryCentre(unittest.TestCase):
    """The statistic must be orientation-free."""

    def test_span_is_independent_of_centre(self) -> None:
        for centre in (0.0, 45.0, 90.0, 179.0, 180.0, 181.0, 270.0, 359.0):
            with self.subTest(centre=centre):
                span, found = minimum_circular_span_degrees(libration(centre, 60.0))
                self.assertAlmostEqual(span, 120.0, delta=1.0)
                offset = (found - centre + 180.0) % 360.0 - 180.0
                self.assertLess(abs(offset), 1.0)

    def test_rotating_all_samples_does_not_change_the_span(self) -> None:
        angles = libration(180.0, 83.5)
        base, _ = minimum_circular_span_degrees(angles)
        for shift in (17.0, 123.0, -256.0):
            with self.subTest(shift=shift):
                shifted, _ = minimum_circular_span_degrees(angles + shift)
                self.assertAlmostEqual(shifted, base, places=6)


class Circulation(unittest.TestCase):
    def test_uniform_coverage_spans_nearly_the_full_circle(self) -> None:
        angles = np.linspace(0.0, 360.0, 3601)[:-1]
        span, _ = minimum_circular_span_degrees(angles)
        self.assertGreater(span, 359.0)

    def test_a_circulating_argument_is_detected(self) -> None:
        """Monotonic drift right around the circle, many times over."""
        angles = np.linspace(0.0, 360.0 * 25.0, 5000)
        span, _ = minimum_circular_span_degrees(angles)
        self.assertGreater(span, 355.0)

    def test_libration_and_circulation_are_separated_by_a_wide_margin(self) -> None:
        librating, _ = minimum_circular_span_degrees(libration(180.0, 83.5))
        circulating, _ = minimum_circular_span_degrees(
            np.linspace(0.0, 360.0 * 25.0, 5000)
        )
        self.assertLess(librating, 200.0)
        self.assertGreater(circulating, 340.0)


class EdgeCases(unittest.TestCase):
    def test_single_sample(self) -> None:
        span, centre = minimum_circular_span_degrees([42.0])
        self.assertEqual(span, 0.0)
        self.assertAlmostEqual(centre, 42.0)

    def test_identical_samples(self) -> None:
        span, centre = minimum_circular_span_degrees([90.0] * 50)
        self.assertAlmostEqual(span, 0.0)
        self.assertAlmostEqual(centre, 90.0)

    def test_two_samples_takes_the_short_way_round(self) -> None:
        span, _ = minimum_circular_span_degrees([350.0, 10.0])
        self.assertAlmostEqual(span, 20.0)

    def test_negative_and_out_of_range_inputs_are_normalised(self) -> None:
        span, _ = minimum_circular_span_degrees([-10.0, 10.0, 730.0])
        self.assertAlmostEqual(span, 20.0)

    def test_empty_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            minimum_circular_span_degrees([])

    def test_non_finite_raises(self) -> None:
        with self.assertRaises(ValueError):
            minimum_circular_span_degrees([1.0, float("nan")])

    def test_two_dimensional_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            minimum_circular_span_degrees(np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
