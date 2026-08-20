"""Discriminators that separate a genuine Lyapunov exponent from the
finite-time artifact produced by regular (quasi-periodic) motion.

Background
----------
For a chaotic trajectory the Benettin cumulative log growth is asymptotically
linear in time::

    S(t) = sum of log renormalization factors  ~  lambda * t

For a regular or quasi-periodic trajectory the tangent vector grows *linearly*
in time rather than exponentially, so::

    S(t) ~ A * ln(t)

Fitting a straight line to S(t) therefore returns a positive slope for regular
motion too. That slope is not an exponent: it scales as 1/T_run, so the reported
"Lyapunov time" comes out as a fixed fraction of the integration length no matter
what the dynamics are. Measured on an integrable Sun+Jupiter two-body system,
where lambda is exactly zero, the line fit returns T_lyap ~ 0.35 * duration,
constant to 2% across a 20x span of durations, with R^2 of 0.78-0.87.

The quantity that does behave correctly is the running Benettin estimate::

    lambda_running(t) = S(t) / t

which tends to lambda for chaotic motion and to zero (as ln(t)/t) for regular
motion.

The halving ratio
-----------------
The cheapest robust discriminator compares the running estimate at the end of
the run against its value at the halfway point::

    ratio = lambda_running(T) / lambda_running(T/2)

* Chaotic:  S ~ lambda*t, so lambda_running is constant and ratio -> 1.
* Regular:  S ~ A*ln(t), so

      lambda_running(T)   = A*ln(T)/T
      lambda_running(T/2) = 2*A*(ln(T) - ln 2)/T

  and the ratio is ln(T) / (2*(ln(T) - ln 2)), which tends to 1/2 from above.

The two regimes are separated by a factor of two, which is a wide margin, and the
ratio needs no model fitting and no choice of fit window.

This module is pure ``numpy``/stdlib and has no project dependencies, so it can
be exercised without REBOUND, SciPy, or an ephemeris kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

__all__ = [
    "GrowthDiagnostics",
    "REGULAR_RATIO_MAX",
    "CHAOTIC_RATIO_MIN",
    "ENERGY_DRIFT_CHAOS_GATE",
    "analyze_growth",
    "analyze_running_lambda",
    "classify_growth",
    "diagnostics_payload",
]

# Halving-ratio decision boundaries.
#
# The asymptotes are 0.5 (regular) and 1.0 (chaotic). The regular ratio
# approaches 0.5 from above, reaching ~0.58 at ln(T) = 6 and ~0.55 at
# ln(T) = 10, so the regular band is placed generously at <= 0.70. The chaotic
# band is placed at >= 0.85 to leave room for the slow transient in a real
# chaotic run. Anything between is reported as ambiguous rather than forced.
REGULAR_RATIO_MAX = 0.70
CHAOTIC_RATIO_MIN = 0.85

# A line fit to a logarithm yields T_lyap / duration in a narrow band around
# 1/e. Observing a value in this band is a positive signature of the artifact.
_ARTIFACT_BAND = (0.25, 0.50)

# Energy-drift backstop.
#
# An under-resolved integration manufactures a positive exponent that no
# post-processing can distinguish from physical chaos: the growth curve really
# is linear, because the numerics really are expanding the tangent vector.
# Measured on an integrable Sun+Jupiter system with a variational leapfrog, the
# verdict tracks the timestep directly --
#
#     steps/orbit   max|dE/E|   halving ratio   verdict
#           1 000    1.96e-06           0.852   chaotic_candidate  (FALSE)
#           2 000    4.90e-07           0.765   ambiguous
#           4 000    1.23e-07           0.671   regular_likely
#           8 000    3.06e-08           0.601   regular_likely
#          32 000    1.92e-09           0.546   regular_likely
#
# -- converging monotonically to the 0.5 regular asymptote as dt shrinks. The
# false verdicts sit above 1e-6 and the correct ones at or below 1.2e-7, so the
# gate is placed at 1e-7. A symplectic solar-system integration normally holds
# bounded drift several orders below that, so this rejects gross
# under-resolution without touching legitimate runs.
#
# This is a backstop, NOT a substitute for showing that lambda is convergent
# under timestep refinement. Nothing in a single run can establish that.
ENERGY_DRIFT_CHAOS_GATE = 1.0e-7


@dataclass(frozen=True)
class GrowthDiagnostics:
    """Result of :func:`analyze_growth`.

    Attributes
    ----------
    lambda_running_final:
        ``S(T) / T`` -- the Benettin estimate. This is the value to report as
        the Lyapunov exponent.
    lambda_running_half:
        ``S(T/2) / (T/2)``.
    halving_ratio:
        ``lambda_running_final / lambda_running_half``. ~1 chaotic, ~0.5 regular.
    classification:
        ``"chaotic_candidate"``, ``"regular_likely"``, or ``"ambiguous"``.
    linear_r_squared, log_r_squared:
        Goodness of fit of ``S = a*t + b`` and ``S = a*ln(t) + b``. The
        logarithmic model fitting better is corroborating evidence of regular
        motion.
    log_model_preferred:
        ``log_r_squared > linear_r_squared``.
    line_fit_lambda:
        The slope of the straight-line fit -- i.e. the legacy estimator. Kept so
        callers can report it alongside, and so the artifact can be quantified.
    line_fit_lyapunov_time_years, line_fit_time_fraction:
        The legacy Lyapunov time and its ratio to the run duration.
    artifact_suspected:
        True when the evidence says the line fit is measuring ln(t) rather than
        an exponent.
    notes:
        Human-readable findings, suitable for emitting as warnings.
    """

    n_points: int
    duration_years: float
    lambda_running_final: float
    lambda_running_half: float
    halving_ratio: float
    classification: str
    linear_r_squared: float
    log_r_squared: float
    log_model_preferred: bool
    line_fit_lambda: float
    line_fit_lyapunov_time_years: float
    line_fit_time_fraction: float
    artifact_suspected: bool
    notes: tuple[str, ...]


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = y - fitted
    ss_res = float(np.dot(residual, residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.dot(centered, centered))
    if ss_tot == 0.0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def analyze_growth(
    times_years: Sequence[float],
    cumulative_log_growth: Sequence[float],
    *,
    max_relative_energy_drift: float | None = None,
) -> GrowthDiagnostics:
    """Diagnose whether cumulative log growth is exponential or logarithmic.

    ``times_years`` must be strictly positive and increasing; the pair are the
    per-renormalization samples from a Benettin run.

    ``max_relative_energy_drift``, when supplied, gates a chaotic verdict: an
    under-resolved integration produces genuinely linear growth, so no analysis
    of the growth curve alone can tell it from physical chaos. Exceeding
    :data:`ENERGY_DRIFT_CHAOS_GATE` downgrades ``chaotic_candidate`` to
    ``ambiguous``.
    """

    times = np.asarray(times_years, dtype=float)
    growth = np.asarray(cumulative_log_growth, dtype=float)
    if times.shape != growth.shape or times.ndim != 1:
        raise ValueError("times and growth must be 1-D arrays of equal length")
    finite = np.isfinite(times) & np.isfinite(growth) & (times > 0.0)
    times, growth = times[finite], growth[finite]
    if times.size < 4:
        raise ValueError("need at least 4 finite samples with t > 0")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")

    duration = float(times[-1])
    notes: list[str] = []

    # Running Benettin estimate, and the same at the halfway point.
    lambda_final = float(growth[-1] / times[-1])
    half_index = int(np.searchsorted(times, duration / 2.0))
    half_index = min(max(half_index, 1), times.size - 1)
    lambda_half = float(growth[half_index] / times[half_index])

    if lambda_half == 0.0 or not math.isfinite(lambda_half):
        ratio = math.nan
    else:
        ratio = lambda_final / lambda_half

    # Straight-line fit -- the legacy estimator, retained for comparison.
    line_slope, line_intercept = np.polyfit(times, growth, deg=1)
    linear_r2 = _r_squared(growth, line_slope * times + line_intercept)

    # Logarithmic fit.
    log_slope, log_intercept = np.polyfit(np.log(times), growth, deg=1)
    log_r2 = _r_squared(growth, log_slope * np.log(times) + log_intercept)

    line_lambda = float(line_slope)
    if line_lambda > 0.0 and math.isfinite(line_lambda):
        line_time = 1.0 / line_lambda
    else:
        line_time = math.inf
    time_fraction = line_time / duration if duration > 0.0 else math.nan

    log_preferred = bool(
        math.isfinite(log_r2) and math.isfinite(linear_r2) and log_r2 > linear_r2
    )

    # Classification from the halving ratio.
    if not math.isfinite(ratio):
        classification = "ambiguous"
        notes.append(
            "running Lyapunov estimate at the half-way point is zero or "
            "non-finite; cannot form the halving ratio"
        )
    elif ratio <= REGULAR_RATIO_MAX:
        classification = "regular_likely"
        notes.append(
            f"halving ratio {ratio:.3f} <= {REGULAR_RATIO_MAX}: the running "
            "estimate is still decaying as ln(t)/t, consistent with regular "
            "or quasi-periodic motion"
        )
    elif ratio >= CHAOTIC_RATIO_MIN:
        classification = "chaotic_candidate"
        notes.append(
            f"halving ratio {ratio:.3f} >= {CHAOTIC_RATIO_MIN}: the running "
            "estimate has stabilized, consistent with a positive exponent"
        )
    else:
        classification = "ambiguous"
        notes.append(
            f"halving ratio {ratio:.3f} lies between {REGULAR_RATIO_MAX} and "
            f"{CHAOTIC_RATIO_MIN}: run longer before classifying"
        )

    if (
        classification == "chaotic_candidate"
        and max_relative_energy_drift is not None
        and math.isfinite(max_relative_energy_drift)
        and abs(max_relative_energy_drift) > ENERGY_DRIFT_CHAOS_GATE
    ):
        classification = "ambiguous"
        notes.append(
            f"maximum relative energy drift {abs(max_relative_energy_drift):.3e} "
            f"exceeds {ENERGY_DRIFT_CHAOS_GATE:.0e}: the integration may be "
            "manufacturing the exponent. A chaotic verdict is withheld until "
            "lambda is shown convergent under timestep refinement."
        )

    if log_preferred:
        notes.append(
            f"a logarithmic model fits the growth better than a linear one "
            f"(R^2 {log_r2:.4f} vs {linear_r2:.4f})"
        )

    in_band = (
        math.isfinite(time_fraction)
        and _ARTIFACT_BAND[0] <= time_fraction <= _ARTIFACT_BAND[1]
    )
    if in_band:
        notes.append(
            f"the straight-line fit gives a Lyapunov time of "
            f"{time_fraction:.3f} x the run duration, which is the signature "
            "of fitting a line to ln(t) rather than measuring an exponent"
        )

    artifact = classification == "regular_likely" and (log_preferred or in_band)
    if artifact:
        notes.append(
            "ARTIFACT SUSPECTED: report lambda_running_final, not the "
            "straight-line slope"
        )

    return GrowthDiagnostics(
        n_points=int(times.size),
        duration_years=duration,
        lambda_running_final=lambda_final,
        lambda_running_half=lambda_half,
        halving_ratio=ratio,
        classification=classification,
        linear_r_squared=linear_r2,
        log_r_squared=log_r2,
        log_model_preferred=log_preferred,
        line_fit_lambda=line_lambda,
        line_fit_lyapunov_time_years=line_time,
        line_fit_time_fraction=time_fraction,
        artifact_suspected=artifact,
        notes=tuple(notes),
    )


def classify_growth(
    times_years: Sequence[float],
    cumulative_log_growth: Sequence[float],
    *,
    model_scope: str = "full",
) -> str:
    """Classification only, from the same evidence as :func:`analyze_growth`.

    A ``two_body`` scope is integrable by construction, so a chaotic verdict
    there indicates a defect in the estimator rather than a physical finding and
    is downgraded to ``"ambiguous"``.
    """

    result = analyze_growth(times_years, cumulative_log_growth)
    if model_scope.startswith("two_body") and result.classification == "chaotic_candidate":
        return "ambiguous"
    return result.classification


def analyze_running_lambda(
    times_years: Sequence[float],
    running_lambda_1_per_year: Sequence[float],
    *,
    max_relative_energy_drift: float | None = None,
) -> GrowthDiagnostics:
    """Same analysis, from a running-estimate history rather than raw growth.

    Several call sites carry ``lambda_running(t) = S(t)/t`` rather than ``S(t)``
    itself. The two are equivalent -- ``S(t) = lambda_running(t) * t`` -- so this
    reconstructs the cumulative growth and delegates to :func:`analyze_growth`.
    """

    times = np.asarray(times_years, dtype=float)
    running = np.asarray(running_lambda_1_per_year, dtype=float)
    if times.shape != running.shape:
        raise ValueError("times and running lambda must have equal length")
    return analyze_growth(
        times, running * times,
        max_relative_energy_drift=max_relative_energy_drift,
    )


def diagnostics_payload(result: GrowthDiagnostics) -> dict:
    """JSON-serializable view, for embedding in a run summary."""

    def finite(value: float):
        return float(value) if math.isfinite(value) else None

    return {
        "schema": "chaos.growth_diagnostics/1",
        "n_points": result.n_points,
        "duration_years": finite(result.duration_years),
        "lambda_running_final_1_per_year": finite(result.lambda_running_final),
        "lambda_running_half_1_per_year": finite(result.lambda_running_half),
        "halving_ratio": finite(result.halving_ratio),
        "classification": result.classification,
        "linear_r_squared": finite(result.linear_r_squared),
        "log_r_squared": finite(result.log_r_squared),
        "log_model_preferred": result.log_model_preferred,
        "line_fit_lambda_1_per_year": finite(result.line_fit_lambda),
        "line_fit_lyapunov_time_years": finite(result.line_fit_lyapunov_time_years),
        "line_fit_time_fraction_of_duration": finite(result.line_fit_time_fraction),
        "artifact_suspected": result.artifact_suspected,
        "notes": list(result.notes),
        "regular_ratio_max": REGULAR_RATIO_MAX,
        "chaotic_ratio_min": CHAOTIC_RATIO_MIN,
        "energy_drift_chaos_gate": ENERGY_DRIFT_CHAOS_GATE,
    }
