"""Tests for the ladder's result machinery.

Most of these assert that something *cannot* be done. That is the point: the
defect this machinery replaces was a reporting path that wrote
``"result": "PASS"`` for 124 nodes without running anything, in a codebase where
no module could emit a failure status at all.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mini_ephemeris.ladder_rungs import (  # noqa: E402
    CAT_MAP_LAMBDA,
    rung2a_cat_map,
    rung2b_standard_map,
)
from mini_ephemeris.validation_ladder import (  # noqa: E402
    LadderReport,
    RungResult,
    RungStatus,
    evaluate_rung,
    run_ladder,
)


class PassCannotBeStamped(unittest.TestCase):
    def test_pass_outside_its_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(rung="X", name="stamped", status=RungStatus.PASS,
                       measured=5.0, acceptance=(0.9, 1.1))

    def test_pass_with_no_measurement_and_no_conditions_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(rung="X", name="empty", status=RungStatus.PASS)

    def test_pass_with_an_unmet_condition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(rung="X", name="unmet", status=RungStatus.PASS,
                       measured=1.0, acceptance=(0.9, 1.1),
                       conditions=(("required", False),))

    def test_pass_with_a_nan_measurement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(rung="X", name="nan", status=RungStatus.PASS,
                       measured=float("nan"), acceptance=(0.9, 1.1))

    def test_a_failure_may_carry_any_measurement(self) -> None:
        result = RungResult(rung="X", name="fail", status=RungStatus.FAIL,
                            measured=5.0, acceptance=(0.9, 1.1))
        self.assertIs(result.status, RungStatus.FAIL)

    def test_malformed_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RungResult(rung="X", name="bad window", status=RungStatus.FAIL,
                       measured=1.0, acceptance=(1.1, 0.9))


class EvaluateRung(unittest.TestCase):
    def test_inside_the_window_passes(self) -> None:
        self.assertIs(
            evaluate_rung("A", "a", measured=1.0, acceptance=(0.9, 1.1)).status,
            RungStatus.PASS,
        )

    def test_outside_the_window_fails(self) -> None:
        self.assertIs(
            evaluate_rung("A", "a", measured=2.0, acceptance=(0.9, 1.1)).status,
            RungStatus.FAIL,
        )

    def test_boundaries_are_inclusive(self) -> None:
        for value in (0.9, 1.1):
            with self.subTest(value=value):
                self.assertIs(
                    evaluate_rung("A", "a", measured=value,
                                  acceptance=(0.9, 1.1)).status,
                    RungStatus.PASS,
                )

    def test_non_finite_measurement_fails(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIs(
                    evaluate_rung("A", "a", measured=value,
                                  acceptance=(0.9, 1.1)).status,
                    RungStatus.FAIL,
                )

    def test_missing_measurement_fails(self) -> None:
        self.assertIs(
            evaluate_rung("A", "a", measured=None, acceptance=(0.9, 1.1)).status,
            RungStatus.FAIL,
        )

    def test_no_window_and_no_conditions_cannot_pass(self) -> None:
        result = evaluate_rung("A", "a", measured=None, acceptance=None)
        self.assertIs(result.status, RungStatus.FAIL)

    def test_an_unmet_condition_fails_a_good_measurement(self) -> None:
        result = evaluate_rung("A", "a", measured=1.0, acceptance=(0.9, 1.1),
                               conditions=(("required", False),))
        self.assertIs(result.status, RungStatus.FAIL)


class LadderHalting(unittest.TestCase):
    @staticmethod
    def _good() -> RungResult:
        return evaluate_rung("A", "good", measured=1.0, acceptance=(0.9, 1.1))

    @staticmethod
    def _bad() -> RungResult:
        return evaluate_rung("B", "bad", measured=9.0, acceptance=(0.9, 1.1))

    @staticmethod
    def _raises() -> RungResult:
        raise RuntimeError("boom")

    def test_halts_at_the_first_failure(self) -> None:
        report = run_ladder((("A", "a", self._good), ("B", "b", self._bad),
                             ("C", "c", self._good)))
        self.assertEqual(
            [r.status for r in report.results],
            [RungStatus.PASS, RungStatus.FAIL, RungStatus.NOT_RUN],
        )
        self.assertEqual(report.halted_at, "B")

    def test_not_run_is_not_a_pass(self) -> None:
        report = run_ladder((("A", "a", self._bad), ("B", "b", self._good)))
        self.assertIs(report.overall_status, RungStatus.FAIL)

    def test_an_exception_becomes_an_error(self) -> None:
        report = run_ladder((("A", "a", self._raises),))
        self.assertIs(report.results[0].status, RungStatus.ERROR)
        self.assertIs(report.overall_status, RungStatus.ERROR)

    def test_continue_on_failure_still_reports_failure(self) -> None:
        report = run_ladder((("A", "a", self._bad), ("B", "b", self._good)),
                            halt_on_failure=False)
        self.assertIs(report.results[1].status, RungStatus.PASS)
        self.assertIs(report.overall_status, RungStatus.FAIL)

    def test_an_empty_ladder_is_not_a_pass(self) -> None:
        self.assertIs(run_ladder(()).overall_status, RungStatus.FAIL)

    def test_a_rung_returning_the_wrong_type_is_an_error(self) -> None:
        report = run_ladder((("A", "a", lambda: "PASS"),))
        self.assertIs(report.results[0].status, RungStatus.ERROR)

    def test_all_pass_is_a_pass(self) -> None:
        report = run_ladder((("A", "a", self._good), ("B", "b", self._good)))
        self.assertIs(report.overall_status, RungStatus.PASS)
        self.assertIsNone(report.halted_at)


class ReportSerialisation(unittest.TestCase):
    def test_json_round_trip_keeps_the_status(self) -> None:
        report = run_ladder((("A", "a", lambda: evaluate_rung(
            "A", "a", measured=1.0, acceptance=(0.9, 1.1))),))
        import json
        payload = json.loads(report.to_json())
        self.assertEqual(payload["overall_status"], "PASS")
        self.assertEqual(payload["rungs"][0]["acceptance"], [0.9, 1.1])

    def test_report_renders_without_error(self) -> None:
        report = LadderReport(results=(), halted_at=None, total_seconds=0.0)
        self.assertIn("OVERALL", report.render())


class MapRungsAreHonest(unittest.TestCase):
    """The rungs themselves, on the cheap maps. Seconds, no REBOUND."""

    def test_cat_map_recovers_its_exact_exponent(self) -> None:
        """5000 iterations get within 0.1% of an exponent known in closed form.

        The residual is finite-sample, not bias: the Benettin sum converges as
        1/N, so this is the accuracy the method is entitled to at this length.
        """
        result = rung2a_cat_map(n_steps=5_000)
        self.assertIs(result.status, RungStatus.PASS)
        relative_error = abs(result.measured - CAT_MAP_LAMBDA) / CAT_MAP_LAMBDA
        self.assertLess(relative_error, 1.0e-3)

    def test_cat_map_converges_as_the_run_lengthens(self) -> None:
        short = rung2a_cat_map(n_steps=2_000)
        long = rung2a_cat_map(n_steps=50_000)
        self.assertLess(
            abs(long.measured - CAT_MAP_LAMBDA),
            abs(short.measured - CAT_MAP_LAMBDA),
        )

    def test_cat_map_target_is_the_golden_ratio_squared(self) -> None:
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        self.assertAlmostEqual(CAT_MAP_LAMBDA, 2.0 * math.log(golden), places=12)

    def test_standard_map_recovers_the_asymptotic_exponent(self) -> None:
        result = rung2b_standard_map(k=97.0, n_steps=40_000)
        self.assertIs(result.status, RungStatus.PASS)


if __name__ == "__main__":
    unittest.main()
