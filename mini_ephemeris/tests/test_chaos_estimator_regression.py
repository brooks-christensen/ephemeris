"""Regression tests for the chaos estimators.

These encode behaviour that was found to be wrong in an external review on
2026-08-19. Each test names the defect it guards against so a future reader can
tell what the test is for without reading the review.

The tests use analytic growth laws rather than integrations, so they run in
milliseconds and need neither REBOUND, SciPy, nor an ephemeris kernel:

* regular / quasi-periodic motion    ->  S(t) = A ln(t) + B
* chaotic motion                     ->  S(t) = lambda t + B

Both laws were confirmed against real integrations: an integrable Sun+Jupiter
two-body system reproduces the logarithmic law with slope 1.0001 and R^2 =
1.000000, and the tangent-Benettin loop applied to it yields the artifact
documented in :class:`RegularMotionMustNotReadAsChaotic`.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from mini_ephemeris.chaos_estimator_diagnostics import (
    CHAOTIC_RATIO_MIN,
    REGULAR_RATIO_MAX,
    ENERGY_DRIFT_CHAOS_GATE,
    analyze_growth,
    analyze_running_lambda,
    classify_growth,
    diagnostics_payload,
)


def regular_growth(duration_years: float, n: int = 400, amplitude: float = 1.0):
    """S(t) = amplitude * ln(t) + const -- linear tangent growth, lambda = 0."""
    times = np.linspace(duration_years / n, duration_years, n)
    return times, amplitude * np.log(times) + 0.5874


def chaotic_growth(duration_years: float, lam: float, n: int = 400):
    """S(t) = lam * t + const -- exponential tangent growth."""
    times = np.linspace(duration_years / n, duration_years, n)
    return times, lam * times + 0.25


class RegularMotionMustNotReadAsChaotic(unittest.TestCase):
    """DEFECT: fitting a straight line to S(t) returns a positive slope for
    regular motion, because S grows as ln(t). The reported Lyapunov time comes
    out at a fixed fraction (~0.35) of the run duration regardless of dynamics.
    """

    def test_line_fit_artifact_is_detected(self) -> None:
        for duration in (1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e8):
            with self.subTest(duration=duration):
                result = analyze_growth(*regular_growth(duration))
                self.assertEqual(result.classification, "regular_likely")
                self.assertTrue(result.artifact_suspected)

    def test_line_fit_lyapunov_time_tracks_duration(self) -> None:
        """The signature itself: T_lyap / duration is constant across a 100x
        span of durations. A real exponent cannot behave this way."""
        fractions = [
            analyze_growth(*regular_growth(d)).line_fit_time_fraction
            for d in (1.0e4, 1.0e5, 1.0e6)
        ]
        for value in fractions:
            self.assertGreater(value, 0.20)
            self.assertLess(value, 0.55)
        self.assertLess(max(fractions) - min(fractions), 0.05)

    def test_running_estimate_decays_toward_zero(self) -> None:
        """The correct estimator, S(T)/T, does go to zero for regular motion."""
        previous = math.inf
        for duration in (1.0e4, 1.0e5, 1.0e6, 1.0e7):
            value = analyze_growth(*regular_growth(duration)).lambda_running_final
            self.assertGreater(previous, value)
            previous = value
        self.assertLess(previous, 1.0e-5)

    def test_halving_ratio_approaches_one_half(self) -> None:
        for duration in (1.0e5, 1.0e7, 1.0e9):
            ratio = analyze_growth(*regular_growth(duration)).halving_ratio
            self.assertLess(ratio, REGULAR_RATIO_MAX)
            self.assertGreater(ratio, 0.45)

    def test_logarithmic_model_is_preferred(self) -> None:
        result = analyze_growth(*regular_growth(1.0e6))
        self.assertTrue(result.log_model_preferred)
        self.assertGreater(result.log_r_squared, 0.999)


class ChaoticMotionMustReadAsChaotic(unittest.TestCase):
    """The discriminator must not achieve its safety by calling everything
    regular. A genuine exponent has to survive."""

    def test_chaotic_growth_is_classified_chaotic(self) -> None:
        for lam in (1.0e-7, 2.0e-7, 1.0e-6):
            with self.subTest(lam=lam):
                result = analyze_growth(*chaotic_growth(1.0e8, lam))
                self.assertEqual(result.classification, "chaotic_candidate")
                self.assertFalse(result.artifact_suspected)

    def test_running_estimate_recovers_lambda(self) -> None:
        """S(T)/T recovers lambda to within the O(1/T) transient bias.

        The Benettin running estimate carries the initial transient as an
        additive S(0)/T term, so it approaches lambda from above rather than
        landing on it exactly. A few percent at 100 Myr is expected.
        """
        lam = 2.0e-7
        result = analyze_growth(*chaotic_growth(1.0e8, lam))
        self.assertLess(abs(result.lambda_running_final / lam - 1.0), 0.02)

    def test_transient_bias_decays_as_one_over_t(self) -> None:
        """The residual bias must shrink with duration -- that is what
        distinguishes a transient offset from a wrong exponent."""
        lam = 2.0e-7
        errors = [
            abs(analyze_growth(*chaotic_growth(d, lam)).lambda_running_final / lam - 1.0)
            for d in (1.0e7, 1.0e8, 1.0e9)
        ]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[-1], 2.0e-3)

    def test_halving_ratio_approaches_one(self) -> None:
        result = analyze_growth(*chaotic_growth(1.0e8, 2.0e-7))
        self.assertGreater(result.halving_ratio, CHAOTIC_RATIO_MIN)
        self.assertLess(result.halving_ratio, 1.05)

    def test_five_myr_lyapunov_time_is_recovered(self) -> None:
        """The literature value for inner-Solar-System chaos, over a 100 Myr
        run -- the case the artifact is most likely to be confused with."""
        lam = 1.0 / 5.0e6
        result = analyze_growth(*chaotic_growth(1.0e8, lam))
        self.assertEqual(result.classification, "chaotic_candidate")
        recovered = 1.0 / result.lambda_running_final
        # 2% tolerance: the O(1/T) transient bias, not an estimator error.
        self.assertLess(abs(recovered - 5.0e6) / 5.0e6, 0.02)


class ClassifierMustHaveDiscriminatingPower(unittest.TestCase):
    """DEFECT: the previous criterion was ``lcn * elapsed_years > 1.0``, which
    is algebraically just "total log growth exceeded one e-fold". Regular linear
    tangent growth passes that after about three renormalization intervals, so
    every full-scope run was classified chaotic regardless of dynamics.
    """

    def test_old_criterion_would_have_fired_on_regular_motion(self) -> None:
        """Documents the defect: the old test passes on data we know is
        regular, which is why it had no discriminating power."""
        times, growth = regular_growth(2.0e4)
        lcn = growth[-1] / times[-1]
        self.assertGreater(lcn * times[-1], 1.0)          # old criterion fires
        self.assertEqual(                                  # new one does not
            classify_growth(times, growth, model_scope="full"), "regular_likely"
        )

    def test_two_body_scope_never_reports_chaotic(self) -> None:
        """A two-body system is integrable by construction; a chaotic verdict
        there is an estimator defect, not a physical finding."""
        times, growth = chaotic_growth(1.0e8, 2.0e-7)
        self.assertEqual(
            classify_growth(times, growth, model_scope="two_body"), "ambiguous"
        )
        self.assertEqual(
            classify_growth(times, growth, model_scope="full"), "chaotic_candidate"
        )

    def test_scopes_are_not_silently_conflated(self) -> None:
        times, growth = regular_growth(1.0e6)
        for scope in ("full", "full_with_pluto", "inner", "two_body"):
            with self.subTest(scope=scope):
                self.assertNotEqual(
                    classify_growth(times, growth, model_scope=scope),
                    "chaotic_candidate",
                )


class UnderResolvedIntegrationMustNotReadAsChaotic(unittest.TestCase):
    """An under-resolved integration manufactures a positive exponent whose
    growth curve is genuinely linear, so no analysis of the curve alone can
    tell it from physical chaos.

    Measured on an integrable Sun+Jupiter system with a variational leapfrog,
    the verdict tracks the timestep directly: at 1,000 steps/orbit
    (max|dE/E| = 1.96e-6) the halving ratio is 0.852 and the run reads
    chaotic, while at 32,000 steps/orbit (1.92e-9) it is 0.546 and reads
    regular. The ratio converges monotonically to the 0.5 regular asymptote as
    dt shrinks.
    """

    def test_large_energy_drift_withholds_a_chaotic_verdict(self) -> None:
        times, growth = chaotic_growth(1.0e8, 2.0e-7)
        clean = analyze_growth(times, growth, max_relative_energy_drift=1.0e-11)
        self.assertEqual(clean.classification, "chaotic_candidate")

        drifting = analyze_growth(times, growth, max_relative_energy_drift=1.0e-5)
        self.assertEqual(drifting.classification, "ambiguous")
        self.assertTrue(
            any("energy drift" in note for note in drifting.notes),
            msg=f"expected an energy-drift note, got {drifting.notes}",
        )

    def test_gate_is_applied_at_the_documented_threshold(self) -> None:
        times, growth = chaotic_growth(1.0e8, 2.0e-7)
        just_under = analyze_growth(
            times, growth, max_relative_energy_drift=ENERGY_DRIFT_CHAOS_GATE * 0.5
        )
        just_over = analyze_growth(
            times, growth, max_relative_energy_drift=ENERGY_DRIFT_CHAOS_GATE * 2.0
        )
        self.assertEqual(just_under.classification, "chaotic_candidate")
        self.assertEqual(just_over.classification, "ambiguous")

    def test_gate_does_not_manufacture_a_chaotic_verdict(self) -> None:
        """The gate may only downgrade. Regular motion with clean energy must
        still read regular."""
        times, growth = regular_growth(1.0e6)
        result = analyze_growth(times, growth, max_relative_energy_drift=1.0e-14)
        self.assertEqual(result.classification, "regular_likely")

    def test_absent_drift_information_leaves_the_verdict_alone(self) -> None:
        times, growth = chaotic_growth(1.0e8, 2.0e-7)
        self.assertEqual(
            analyze_growth(times, growth).classification, "chaotic_candidate"
        )
        self.assertEqual(
            analyze_growth(
                times, growth, max_relative_energy_drift=float("nan")
            ).classification,
            "chaotic_candidate",
        )


class RunningLambdaEntryPoint(unittest.TestCase):
    """Call sites that carry lambda_running(t) rather than S(t) must get the
    same verdict."""

    def test_agrees_with_growth_entry_point(self) -> None:
        for maker in (lambda: regular_growth(1.0e6), lambda: chaotic_growth(1.0e8, 2.0e-7)):
            times, growth = maker()
            direct = analyze_growth(times, growth)
            via_running = analyze_running_lambda(times, growth / times)
            self.assertEqual(direct.classification, via_running.classification)
            self.assertAlmostEqual(
                direct.halving_ratio, via_running.halving_ratio, places=9
            )

    def test_payload_is_json_serializable(self) -> None:
        import json

        payload = diagnostics_payload(analyze_growth(*regular_growth(1.0e6)))
        json.loads(json.dumps(payload))
        self.assertEqual(payload["classification"], "regular_likely")
        self.assertTrue(payload["artifact_suspected"])
        self.assertIsNotNone(payload["lambda_running_final_1_per_year"])


class InputHandling(unittest.TestCase):
    def test_rejects_too_few_samples(self) -> None:
        with self.assertRaises(ValueError):
            analyze_growth([1.0, 2.0], [0.1, 0.2])

    def test_rejects_nonincreasing_times(self) -> None:
        with self.assertRaises(ValueError):
            analyze_growth([1.0, 3.0, 2.0, 4.0], [0.1, 0.2, 0.3, 0.4])

    def test_drops_nonfinite_and_nonpositive_times(self) -> None:
        times, growth = regular_growth(1.0e5, n=50)
        times = np.concatenate([[0.0, -1.0], times])
        growth = np.concatenate([[0.0, 0.0], growth])
        result = analyze_growth(times, growth)
        self.assertEqual(result.n_points, 50)


if __name__ == "__main__":
    unittest.main()
