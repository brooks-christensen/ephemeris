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
    "SATURATION_SLOPE_FRACTION",
    "SaturationWindow",
    "find_saturation_onset",
    "MEGNO_MEAN_TO_LYAPUNOV",
    "MEGNO_INSTANTANEOUS_TO_LYAPUNOV",
    "REBOUND_LYAPUNOV_TO_LAMBDA",
    "SLOPE_WINDOW_MAX_SPREAD",
    "WindowedSlopeAnalysis",
    "analyze_window_slopes",
    "TransientFit",
    "fit_transient_and_exponent",
    "ESTIMATOR_DISCARD_FRACTION",
    "WINDOW_CONSISTENCY_TOLERANCE",
    "MEGNO_AGREEMENT_TOLERANCE",
    "LyapunovEstimate",
    "estimate_lyapunov_exponent",
    "megno_from_log_tangent",
    "lambda_from_rebound_lyapunov",
    "calibrate_megno_factor",
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
    #
    # Guard first on the sign. lambda_max >= 0 for a Hamiltonian flow, so a
    # negative running estimate is never physical -- and two negative values
    # produce a POSITIVE halving ratio, which would otherwise sail through the
    # chaotic band. Found by running a legacy two-trajectory estimator on an
    # integrable system: running lambda -1.455e-04 with ratio 2.174.
    if math.isfinite(lambda_final) and lambda_final < 0.0:
        classification = "ambiguous"
        notes.append(
            f"running Lyapunov estimate is negative ({lambda_final:.3e}), which "
            "is unphysical for a Hamiltonian flow: the estimate has not "
            "converged and no classification is possible"
        )
    elif not math.isfinite(ratio):
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


# ---------------------------------------------------------------- saturation --
#
# A shadow-particle run measures ln|separation|, which grows linearly with slope
# lambda only while the separation is small. Once the shadow has wandered a
# significant fraction of the system size the growth stops and the curve
# flattens. Including that plateau in a straight-line fit biases lambda LOW --
# and the bias grows with run duration, so it is worst on the longest runs.
#
# Measured on a synthetic 100 Myr shadow run with a true Lyapunov time of 2 Myr
# saturating at 51 Myr, fitting the whole record gave 3.837 Myr (1.92x too long)
# with R^2 = 0.813 -- above the 0.8 "not strongly linear" warning threshold, so
# nothing fired.

# Saturation is declared where the local slope has fallen below this fraction of
# the slope measured over the early, unambiguously exponential part of the run.
SATURATION_SLOPE_FRACTION = 0.25

# Fraction of the record used to establish the reference (pre-saturation) slope.
_EARLY_FRACTION = 0.25

# The early window must itself be convincingly exponential before its slope is
# trusted as a reference. Without this, a record with no growth at all -- noise
# with a randomly positive early trend -- would have "saturation" declared
# partway through and most of its samples discarded.
_EARLY_LINEARITY_MIN_R2 = 0.90


@dataclass(frozen=True)
class SaturationWindow:
    """Where a separation record stops growing exponentially."""

    onset_index: int | None
    onset_time: float | None
    reference_slope: float
    n_excluded: int
    saturated: bool
    note: str


def find_saturation_onset(
    times: Sequence[float],
    log_separation: Sequence[float],
    *,
    slope_fraction: float = SATURATION_SLOPE_FRACTION,
) -> SaturationWindow:
    """Locate the point where exponential separation growth stops.

    Returns the index of the first sample judged saturated, or ``None`` if the
    record never saturates. Callers should fit only up to ``onset_index``.
    """

    t = np.asarray(times, dtype=float)
    y = np.asarray(log_separation, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("times and log separation must be 1-D arrays of equal length")
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    n = t.size
    if n < 8:
        return SaturationWindow(None, None, math.nan, 0, False, "too few samples to test")

    early = max(4, int(n * _EARLY_FRACTION))
    early_slope, early_intercept = np.polyfit(t[:early], y[:early], 1)
    reference_slope = float(early_slope)
    if not math.isfinite(reference_slope) or reference_slope <= 0.0:
        return SaturationWindow(
            None, None, reference_slope, 0, False,
            "early slope is not positive; no exponential phase to protect",
        )
    early_r2 = _r_squared(y[:early], early_slope * t[:early] + early_intercept)
    if not math.isfinite(early_r2) or early_r2 < _EARLY_LINEARITY_MIN_R2:
        return SaturationWindow(
            None, None, reference_slope, 0, False,
            (
                f"early window is not convincingly exponential (R^2 "
                f"{early_r2:.4f} < {_EARLY_LINEARITY_MIN_R2}); no reference "
                "slope established, so nothing is excluded"
            ),
        )

    # Rolling local slope over a window of ~1/10 of the record.
    span = max(3, n // 10)
    threshold = slope_fraction * reference_slope
    onset = None
    for i in range(early, n - span):
        local = float(np.polyfit(t[i : i + span], y[i : i + span], 1)[0])
        if local < threshold:
            onset = i
            break

    if onset is None:
        return SaturationWindow(
            None, None, reference_slope, 0, False,
            "no saturation detected; the whole record is usable",
        )
    return SaturationWindow(
        onset_index=int(onset),
        onset_time=float(t[onset]),
        reference_slope=reference_slope,
        n_excluded=int(n - onset),
        saturated=True,
        note=(
            f"local slope fell below {slope_fraction:.2f} x the early slope at "
            f"t = {t[onset]:.6g}; {n - onset} of {n} samples excluded from the fit"
        ),
    )


# --------------------------------------------------------------------- MEGNO --
#
# CONVENTION HAZARD. Two different quantities are called "MEGNO":
#
#   Y(t)     the instantaneous MEGNO, which grows as lambda * t
#   <Y>(t)   the time-averaged MEGNO, which grows as lambda * t / 2
#            and tends to 2 for regular orbits
#
# REBOUND's Simulation.megno() returns the time-averaged <Y>; this repository's
# own build_fli_megno_samples produces the instantaneous Y (verified: its slope
# recovers lambda to five significant figures on a synthetic with known lambda).
# The two are plotted against the same Y = 2 reference line and cross-compared,
# so their slopes differ by a factor of two before any conversion is applied.
#
# The previous conversion here was 0.5, which is wrong under BOTH conventions.
MEGNO_MEAN_TO_LYAPUNOV = 2.0
MEGNO_INSTANTANEOUS_TO_LYAPUNOV = 1.0

# REBOUND's Simulation.lyapunov() is documented as the Lyapunov Characteristic
# Number, but src/tools.c computes it as
#
#     megno_cov_Yt / megno_var_t
#
# where the covariance is accumulated against the value returned by
# reb_simulation_megno(), i.e. against <Y>. It is therefore the
# ordinary-least-squares slope of <Y> against time over the whole integration,
# which is lambda/2 and not lambda. Verified two ways against REBOUND 4.6.0
# (and the code is byte-identical in 5.1.1):
#
#   * sampling <Y> once per fixed WHFast step and running polyfit on the whole
#     record reproduces Simulation.lyapunov() to a relative 1.4e-5;
#   * on chaotic systems Simulation.lyapunov() divided by lambda measured from
#     the tangent vector itself lands at 0.39-0.55, never near 1.
#
# Multiply by this constant to obtain lambda. Prefer a late-window fit of <Y>
# (or a direct Benettin estimate) for anything quantitative: the REBOUND value
# regresses over the entire history, transient included, and is biased low
# whenever chaos sets in after t = 0.
REBOUND_LYAPUNOV_TO_LAMBDA = 2.0

# Provenances that are acceptable as an "independent" lambda in
# calibrate_megno_factor, and those that are structurally circular.
_ACCEPTED_LAMBDA_SOURCES = frozenset(
    {"tangent_vector", "shadow_orbit", "benettin", "analytic"}
)
_FORBIDDEN_LAMBDA_SOURCES = {
    "rebound_lyapunov": (
        "Simulation.lyapunov() is the least-squares slope of Simulation.megno() "
        "against time, which is the very series whose slope is being "
        "calibrated. Their ratio is approximately 1 on any chaotic system under "
        "either convention, so it cannot distinguish them; it would report "
        "'instantaneous_Y' and halve every Lyapunov exponent in the project. "
        "Supply a tangent-vector or shadow-orbit Benettin lambda instead."
    ),
    "megno": (
        "A lambda derived from the same MEGNO series cannot calibrate that "
        "series. Supply an independent estimate."
    ),
}


def _cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, same length as the inputs, starting 0."""

    out = np.zeros_like(values, dtype=float)
    out[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    return out


def megno_from_log_tangent(
    times: Sequence[float],
    log_tangent_norm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct Cincotta-Simo MEGNO from a tangent-vector history.

    ``Y(t)`` and its time average ``<Y>(t)`` are defined by

        Y(t)   = (2/t) * integral_0^t s * dL/ds ds,     L = ln |delta|
        <Y>(t) = (1/t) * integral_0^t Y(s) ds

    Integrating the first by parts removes the derivative of a noisy series::

        Y(t) = 2 * [ L(t) - (1/t) * integral_0^t L ds ]

    which is what this function evaluates, by trapezoid, from ``L`` alone. Any
    constant offset in ``L`` cancels, so the normalization of the initial
    tangent vector does not matter.

    Two limits fall out analytically and are pinned by the regression tests:

    * Regular motion, ``|delta| ~ t``: ``Y -> 2`` and ``<Y> -> 2``.
    * Chaos, ``L = lambda * t``: ``Y = lambda t`` and ``<Y> = lambda t / 2``,
      hence ``lambda = MEGNO_MEAN_TO_LYAPUNOV * d<Y>/dt``.

    Reconstructing ``<Y>`` this way reproduces REBOUND's ``Simulation.megno()``
    to a relative 2.4e-4 on chaotic and regular three-body systems under both
    WHFast and IAS15, which is what establishes that ``Simulation.megno()`` is
    the time-averaged quantity and that the conversion factor is 2.0.

    Parameters
    ----------
    times:
        Strictly increasing, starting at exactly 0.
    log_tangent_norm:
        ``ln |delta(t)|`` sampled at ``times``, with no renormalization applied
        (or, if renormalized, with the log of each rescaling added back in).

    Returns
    -------
    ``(Y, Y_mean)``, both the same length as ``times``, with the value at
    ``t = 0`` set to 0.
    """

    t = np.asarray(times, dtype=float)
    log_norm = np.asarray(log_tangent_norm, dtype=float)
    if t.ndim != 1 or log_norm.shape != t.shape:
        raise ValueError("times and log_tangent_norm must be 1-D and the same length")
    if t.size < 3:
        raise ValueError("need at least three samples")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(log_norm)):
        raise ValueError("times and log_tangent_norm must be finite")
    if t[0] != 0.0:
        raise ValueError(
            "times must start at t = 0 with the initial ln|delta|; the double "
            "time average is taken from the start of the integration"
        )
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("times must be strictly increasing")

    integral_log = _cumulative_trapezoid(log_norm, t)
    y_inst = np.zeros_like(t)
    y_inst[1:] = 2.0 * (log_norm[1:] - integral_log[1:] / t[1:])

    integral_y = _cumulative_trapezoid(y_inst, t)
    y_mean = np.zeros_like(t)
    y_mean[1:] = integral_y[1:] / t[1:]
    return y_inst, y_mean


def lambda_from_rebound_lyapunov(rebound_lyapunov_1_per_time: float) -> float:
    """Convert ``Simulation.lyapunov()`` to a maximal Lyapunov exponent.

    See ``REBOUND_LYAPUNOV_TO_LAMBDA``. The result is still a whole-history
    regression and is biased low when the chaotic phase starts after t = 0;
    treat it as a diagnostic, not as a reportable exponent.
    """

    value = float(rebound_lyapunov_1_per_time)
    if not math.isfinite(value):
        return math.nan
    return value * REBOUND_LYAPUNOV_TO_LAMBDA



# --------------------------------------------------------------------------
# Windowed slope: the estimator for slow chaos
# --------------------------------------------------------------------------
#
# lambda_running(T) = S(T)/T is the right statistic when S is dominated by the
# exponential term, and the wrong one when it is not. Real tangent growth is
#
#     S(t) = A ln(t) + lambda t + c
#
# and S(T)/T carries the whole transient as a 1/T tail. For Pluto, lambda is
# about 7e-8 per year and A ln T is about 20 at T = 4e8 yr, so at 400 Myr the
# logarithmic term is still a third of the signal: S(T)/T overestimates lambda
# by roughly that much, and it keeps changing as the run lengthens. Measured on
# the Pluto rung, the reported Lyapunov time moved from 6.50 Myr at 200 Myr to
# 8.86 Myr at 400 Myr -- same seeds, same timestep, only the duration changed.
#
# A slope over a late window removes the additive transient entirely, and the
# same data gave 13.91 Myr from every one of five tangent seeds, agreeing with
# the independent MEGNO estimate (12.60 Myr) to 10%.
#
# This is NOT a return to the line fit that started all of this. That fit ran
# from t = 0 and reported the slope of A ln(t) as an exponent. The guard here is
# cross-window agreement: for S = A ln t the window slopes fall off as
# 1/t and disagree by order 100% across consecutive windows, while for genuine
# exponential growth they agree. Convergence has to be demonstrated by the
# windows, not assumed -- and `converged` is False unless it is.
SLOPE_WINDOW_MAX_SPREAD = 0.10


@dataclass(frozen=True)
class WindowedSlopeAnalysis:
    """Lambda from consecutive late-window slopes of S(t), plus its convergence.

    Attributes
    ----------
    window_edges_years:
        The ``n_windows + 1`` boundaries used, starting after the discarded
        first quarter of the record.
    slopes_1_per_year:
        dS/dt within each window.
    lambda_estimate_1_per_year:
        Median window slope. Report this, not ``S(T)/T``, once ``converged``.
    relative_spread:
        ``(max - min) / |median|`` across the windows. This is the convergence
        statistic: it is near zero for exponential growth and of order 1 for
        logarithmic growth.
    converged:
        ``relative_spread <= SLOPE_WINDOW_MAX_SPREAD`` and every slope positive.
        A False here means the number is not yet a Lyapunov exponent.
    """

    n_windows: int
    window_edges_years: tuple[float, ...]
    slopes_1_per_year: tuple[float, ...]
    lambda_estimate_1_per_year: float
    lyapunov_time_years: float
    relative_spread: float
    converged: bool
    notes: tuple[str, ...]


def analyze_window_slopes(
    times_years: Sequence[float],
    cumulative_log_growth: Sequence[float],
    *,
    n_windows: int = 3,
    discard_fraction: float = 0.25,
) -> WindowedSlopeAnalysis:
    """Estimate lambda from consecutive equal-duration late windows of S(t).

    The first ``discard_fraction`` of the record is dropped, and the remainder
    is split into ``n_windows`` equal spans of time. Each window contributes
    ``(S(end) - S(start)) / (end - start)``.
    """

    times = np.asarray(times_years, dtype=float)
    growth = np.asarray(cumulative_log_growth, dtype=float)
    if times.ndim != 1 or growth.shape != times.shape:
        raise ValueError("times and growth must be 1-D and the same length")
    if n_windows < 2:
        raise ValueError("need at least two windows to test convergence")
    if times.size < 4 * n_windows:
        raise ValueError("not enough samples for the requested number of windows")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(growth)):
        raise ValueError("times and growth must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in [0, 1)")

    start = times[0] + discard_fraction * (times[-1] - times[0])
    edges = np.linspace(start, times[-1], n_windows + 1)
    slopes: list[float] = []
    notes: list[str] = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # Least squares across every sample in the window, NOT a difference of
        # the two endpoints. S(t) for a real system carries secular oscillation
        # on top of the exponential trend, and an endpoint difference is
        # maximally sensitive to wherever those two instants happen to sit in
        # that oscillation. Measured on the Pluto rung, endpoint differencing
        # made consecutive window slopes disagree by 117% on a record whose
        # least-squares slopes agree far better -- a spurious "not converged"
        # verdict produced entirely by the estimator's own noise.
        mask = (times >= lo) & (times <= hi)
        if int(np.count_nonzero(mask)) < 3:
            raise ValueError(
                f"window {index} spans fewer than three samples; reduce "
                "n_windows or sample the history more finely"
            )
        slopes.append(float(np.polyfit(times[mask], growth[mask], 1)[0]))

    array = np.asarray(slopes, dtype=float)
    median = float(np.median(array))
    if median > 0.0 and np.all(array > 0.0):
        spread = float((array.max() - array.min()) / median)
    else:
        spread = math.inf
        notes.append(
            "at least one window slope is not positive: the tangent vector is "
            "not growing steadily, so no exponent can be reported"
        )
    converged = spread <= SLOPE_WINDOW_MAX_SPREAD
    if not converged and math.isfinite(spread):
        notes.append(
            f"window slopes disagree by {spread:.1%} (limit "
            f"{SLOPE_WINDOW_MAX_SPREAD:.0%}): lambda has not converged in time. "
            "Falling slopes across consecutive windows are the signature of "
            "logarithmic growth, i.e. regular motion."
        )
    if converged:
        notes.append(
            f"window slopes agree to {spread:.2%}; lambda = {median:.4e} per "
            f"year, Lyapunov time {1.0 / median / 1.0e6:.3f} Myr"
        )
        notes.append(
            "this lambda is an UPPER bound and the Lyapunov time a LOWER "
            "bound: the residual logarithmic term contributes "
            "A*ln(t2/t1)/(t2-t1) > 0 to every window slope, and it shrinks as "
            "the windows move later. Report the bound as a bound."
        )
    return WindowedSlopeAnalysis(
        n_windows=n_windows,
        window_edges_years=tuple(float(e) for e in edges),
        slopes_1_per_year=tuple(slopes),
        lambda_estimate_1_per_year=median if converged else math.nan,
        lyapunov_time_years=(1.0 / median) if (converged and median > 0) else math.nan,
        relative_spread=spread,
        converged=converged,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class TransientFit:
    """Least squares of S(t) = A ln(t) + lambda*t + c, the asymptotic form itself.

    The windowed slope is assumption-light but throws away most of the record:
    it sees only the trend within each window and still carries a residual
    logarithmic bias. When the run is long enough that the two-term form
    actually describes the data, fitting it directly uses every sample and
    separates the transient from the exponent instead of tolerating it.

    The price is model dependence, and ``r_squared`` and ``residual_sigma`` are
    how you check whether you are paying it. Measured on a 100 Myr Pluto record,
    R^2 was 0.78 and the residual scatter about the model was 1.0 in S -- the
    tangent norm swinging by a factor of e -- with dominant residual periods of
    25 to 34 Myr, comparable to the Lyapunov time. Over that span the fit is
    meaningless and returns a negative lambda. It becomes usable only once
    lambda*T is large against that scatter.

    Attributes
    ----------
    transient_amplitude:
        ``A``, the coefficient of ln(t). Positive for a tangent vector that
        also grows polynomially, which is the generic case.
    lambda_1_per_time:
        The exponent, with the logarithmic term removed rather than absorbed.
    residual_sigma:
        Scatter of S about the fitted model, in the same units as S. Compare it
        against ``lambda*T``: if they are comparable, the run is too short.
    trend_to_scatter:
        ``lambda*T / residual_sigma``. A rough signal-to-noise for the exponent.
    """

    transient_amplitude: float
    lambda_1_per_time: float
    intercept: float
    r_squared: float
    residual_sigma: float
    trend_to_scatter: float
    n_samples: int
    notes: tuple[str, ...]


def fit_transient_and_exponent(
    times_years: Sequence[float],
    cumulative_log_growth: Sequence[float],
    *,
    discard_fraction: float = 0.05,
) -> TransientFit:
    """Fit S = A ln(t) + lambda*t + c by least squares and report the diagnostics."""

    times = np.asarray(times_years, dtype=float)
    growth = np.asarray(cumulative_log_growth, dtype=float)
    if times.ndim != 1 or growth.shape != times.shape:
        raise ValueError("times and growth must be 1-D and the same length")
    if times.size < 10:
        raise ValueError("need at least ten samples")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(growth)):
        raise ValueError("times and growth must be finite")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in [0, 1)")

    start = times[0] + discard_fraction * (times[-1] - times[0])
    keep = times >= start
    times = times[keep]
    growth = growth[keep]
    if np.any(times <= 0.0):
        raise ValueError("times must be positive to fit a logarithmic term")

    design = np.column_stack([np.log(times), times, np.ones_like(times)])
    coefficients, *_ = np.linalg.lstsq(design, growth, rcond=None)
    residual = growth - design @ coefficients
    ss_res = float(residual @ residual)
    centred = growth - float(growth.mean())
    ss_tot = float(centred @ centred)
    r_squared = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else math.nan
    sigma = float(np.std(residual))

    amplitude, lam, intercept = (float(c) for c in coefficients)
    trend = lam * float(times[-1])
    ratio = trend / sigma if sigma > 0.0 else math.inf

    notes: list[str] = []
    if lam <= 0.0:
        notes.append(
            f"fitted lambda is not positive ({lam:.3e}): over this span the "
            "two-term model does not describe the record, and no exponent "
            "should be taken from it"
        )
    if ratio < 10.0:
        notes.append(
            f"lambda*T = {trend:.2f} against residual scatter {sigma:.2f} "
            f"(ratio {ratio:.1f}): the exponential trend does not yet dominate "
            "the secular oscillation in the tangent norm. Integrate longer."
        )
    if math.isfinite(r_squared) and r_squared < 0.95:
        notes.append(
            f"R^2 = {r_squared:.3f}: the two-term form is not describing this "
            "record well, so its separation of A from lambda is not meaningful"
        )
    return TransientFit(
        transient_amplitude=amplitude,
        lambda_1_per_time=lam,
        intercept=intercept,
        r_squared=r_squared,
        residual_sigma=sigma,
        trend_to_scatter=ratio,
        n_samples=int(times.size),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------
# The two estimators, over the same data, with a stated consistency criterion
# --------------------------------------------------------------------------
#
# Codex refused to launch an 800 Myr run against a specification that asked for
# "the two estimators to agree" without saying agree to what, and pointed out
# that their defaults did not even look at the same data: the windowed slope
# discarded the first 25% of the record and the two-term fit the first 5%. An
# agreement criterion between estimators reading different spans is not a
# criterion. That was a real defect, and this is the repair.
#
# One discard fraction, used by both.
ESTIMATOR_DISCARD_FRACTION = 0.25

# The two estimators must NOT agree exactly, and demanding that they do would be
# wrong. The windowed slope over [t1, t2] of S = A ln t + lambda t is
#
#     lambda + A * (least-squares slope of ln t over that window)
#
# so it sits above the two-term fit's lambda by an amount that is *predictable
# once A is known*. The test is therefore not "are they close" but "does the
# windowed slope sit where the fitted transient says it should", which is a
# statement about whether the two-term model describes the record at all. A
# record whose tangent growth is not A ln t + lambda t will fail it even if both
# estimators happen to return similar numbers.
WINDOW_CONSISTENCY_TOLERANCE = 0.15

# MEGNO is a third, structurally independent estimator: a different accumulator
# on the same variational particles. It carries its own double-time-average
# transient, so exact agreement is not expected here either.
MEGNO_AGREEMENT_TOLERANCE = 0.20


@dataclass(frozen=True)
class LyapunovEstimate:
    """Both tangent estimators over one record, plus MEGNO, plus the verdict.

    ``lambda_1_per_year`` comes from the two-term fit, which removes the
    logarithmic transient rather than tolerating it. The windowed slope is
    reported alongside as a cross-check, not as the answer.
    """

    lambda_1_per_year: float
    lyapunov_time_years: float
    transient_amplitude: float
    r_squared: float
    residual_sigma: float
    trend_to_scatter: float
    window_slopes_1_per_year: tuple[float, ...]
    predicted_window_slopes_1_per_year: tuple[float, ...]
    window_consistency: float
    split_half_disagreement: float
    megno_lambda_1_per_year: float | None
    megno_disagreement: float | None
    discard_fraction: float
    consistent: bool
    notes: tuple[str, ...]


def estimate_lyapunov_exponent(
    times_years: Sequence[float],
    cumulative_log_growth: Sequence[float],
    *,
    mean_megno: Sequence[float] | None = None,
    n_windows: int = 3,
    discard_fraction: float = ESTIMATOR_DISCARD_FRACTION,
) -> LyapunovEstimate:
    """Run both tangent estimators over the same window and check consistency."""

    times = np.asarray(times_years, dtype=float)
    growth = np.asarray(cumulative_log_growth, dtype=float)
    fit = fit_transient_and_exponent(
        times, growth, discard_fraction=discard_fraction
    )
    windows = analyze_window_slopes(
        times, growth, n_windows=n_windows, discard_fraction=discard_fraction
    )

    # Where each window slope should sit, given the fitted transient.
    predicted: list[float] = []
    edges = windows.window_edges_years
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (times >= lo) & (times <= hi)
        span = times[mask]
        log_slope = float(np.polyfit(span, np.log(span), 1)[0])
        predicted.append(fit.lambda_1_per_time + fit.transient_amplitude * log_slope)

    notes = list(fit.notes) + list(windows.notes)
    scale = abs(fit.lambda_1_per_time)

    # Split-half stability of the FIT, which is what catches a record the
    # two-term form does not describe.
    #
    # The window-versus-prediction check below is necessary but not sufficient,
    # and a test caught that: on stretched-exponential growth S ~ t^0.75 the
    # two-term model is a good enough local approximation that the window slopes
    # land exactly where it predicts, and the check passes a record it should
    # reject. Fitting the early and late halves separately does catch it,
    # because a genuine exponent is the same in both halves and a spurious one
    # is not -- for t^0.75 the local rate falls as t^-0.25.
    start = times[0] + discard_fraction * (times[-1] - times[0])
    analysed = times[times >= start]
    midpoint = analysed[0] + 0.5 * (analysed[-1] - analysed[0])
    early_mask = (times >= start) & (times <= midpoint)
    late_mask = times >= midpoint
    split = math.inf
    if int(np.count_nonzero(early_mask)) >= 10 and int(np.count_nonzero(late_mask)) >= 10:
        early = fit_transient_and_exponent(
            times[early_mask], growth[early_mask], discard_fraction=0.0
        )
        late = fit_transient_and_exponent(
            times[late_mask], growth[late_mask], discard_fraction=0.0
        )
        if scale > 0.0 and math.isfinite(scale):
            split = abs(late.lambda_1_per_time - early.lambda_1_per_time) / scale
    if split > WINDOW_CONSISTENCY_TOLERANCE:
        notes.append(
            f"lambda fitted on the early and late halves of the analysed span "
            f"differ by {split:.1%} (limit {WINDOW_CONSISTENCY_TOLERANCE:.0%}): "
            "the exponent is not stable across the record, so the two-term form "
            "is not describing it"
        )
    if scale > 0.0 and math.isfinite(scale):
        deviations = [
            abs(measured - expected) / scale
            for measured, expected in zip(windows.slopes_1_per_year, predicted)
        ]
        consistency = max(deviations)
    else:
        consistency = math.inf
    if consistency > WINDOW_CONSISTENCY_TOLERANCE:
        notes.append(
            f"window slopes sit {consistency:.1%} away from where the fitted "
            f"transient puts them (limit {WINDOW_CONSISTENCY_TOLERANCE:.0%}): "
            "the two-term model is not describing this record, so neither "
            "estimator's lambda should be reported"
        )

    megno_lambda: float | None = None
    megno_gap: float | None = None
    if mean_megno is not None:
        series = np.asarray(mean_megno, dtype=float)
        if series.shape != times.shape:
            raise ValueError("mean_megno must be the same length as times")
        start = times[0] + discard_fraction * (times[-1] - times[0])
        mask = (times >= start) & np.isfinite(series)
        if int(np.count_nonzero(mask)) >= 10:
            slope = float(np.polyfit(times[mask], series[mask], 1)[0])
            megno_lambda = slope * MEGNO_MEAN_TO_LYAPUNOV
            if scale > 0.0:
                megno_gap = abs(megno_lambda - fit.lambda_1_per_time) / scale
                if megno_gap > MEGNO_AGREEMENT_TOLERANCE:
                    notes.append(
                        f"MEGNO differs from the tangent fit by {megno_gap:.1%} "
                        f"(limit {MEGNO_AGREEMENT_TOLERANCE:.0%})"
                    )

    consistent = (
        fit.lambda_1_per_time > 0.0
        and consistency <= WINDOW_CONSISTENCY_TOLERANCE
        and split <= WINDOW_CONSISTENCY_TOLERANCE
        and fit.trend_to_scatter >= 10.0
        and (megno_gap is None or megno_gap <= MEGNO_AGREEMENT_TOLERANCE)
    )
    lam = fit.lambda_1_per_time
    return LyapunovEstimate(
        lambda_1_per_year=lam if consistent else math.nan,
        lyapunov_time_years=(1.0 / lam) if (consistent and lam > 0.0) else math.nan,
        transient_amplitude=fit.transient_amplitude,
        r_squared=fit.r_squared,
        residual_sigma=fit.residual_sigma,
        trend_to_scatter=fit.trend_to_scatter,
        window_slopes_1_per_year=windows.slopes_1_per_year,
        predicted_window_slopes_1_per_year=tuple(predicted),
        window_consistency=consistency,
        split_half_disagreement=split,
        megno_lambda_1_per_year=megno_lambda,
        megno_disagreement=megno_gap,
        discard_fraction=discard_fraction,
        consistent=consistent,
        notes=tuple(notes),
    )

def calibrate_megno_factor(
    megno_slope_per_year: float,
    independent_lambda_1_per_year: float,
    *,
    lambda_source: str,
) -> dict:
    """Determine empirically which MEGNO convention a series follows.

    Run a MEGNO integration and an independent Benettin estimate on the same
    chaotic system, then pass the MEGNO slope and the Benettin lambda here. The
    implied factor identifies the convention, settling it by measurement rather
    than by reading library documentation.

    ``lambda_source`` is required and names where the lambda came from, because
    the obvious candidate is not independent: REBOUND's
    ``Simulation.lyapunov()`` is itself the least-squares slope of
    ``Simulation.megno()``, so feeding it here divides a quantity by itself and
    returns about 1.0 no matter which convention holds. That is not a noisy
    answer, it is a confident wrong one -- it reports ``instantaneous_Y`` and
    invites halving every Lyapunov exponent in the project. Accepted values are
    ``tangent_vector``, ``shadow_orbit``, ``benettin`` and ``analytic``.
    """

    source = str(lambda_source).strip().lower()
    if source in _FORBIDDEN_LAMBDA_SOURCES:
        raise ValueError(
            f"lambda_source={source!r} is not independent of the MEGNO series: "
            + _FORBIDDEN_LAMBDA_SOURCES[source]
        )
    if source not in _ACCEPTED_LAMBDA_SOURCES:
        raise ValueError(
            f"unknown lambda_source={lambda_source!r}; expected one of "
            + ", ".join(sorted(_ACCEPTED_LAMBDA_SOURCES))
        )

    if not (
        math.isfinite(megno_slope_per_year)
        and math.isfinite(independent_lambda_1_per_year)
        and megno_slope_per_year > 0.0
    ):
        return {"implied_factor": None, "convention": "undetermined",
                "note": "need a positive finite MEGNO slope and a finite lambda"}
    implied = independent_lambda_1_per_year / megno_slope_per_year
    if abs(implied - MEGNO_MEAN_TO_LYAPUNOV) < 0.25:
        convention = "mean_Y"
    elif abs(implied - MEGNO_INSTANTANEOUS_TO_LYAPUNOV) < 0.25:
        convention = "instantaneous_Y"
    else:
        convention = "unrecognized"
    return {
        "implied_factor": implied,
        "convention": convention,
        "lambda_source": source,
        "mean_Y_factor": MEGNO_MEAN_TO_LYAPUNOV,
        "instantaneous_Y_factor": MEGNO_INSTANTANEOUS_TO_LYAPUNOV,
        "note": (
            f"implied factor {implied:.4f}; expected "
            f"{MEGNO_MEAN_TO_LYAPUNOV} for mean MEGNO or "
            f"{MEGNO_INSTANTANEOUS_TO_LYAPUNOV} for instantaneous"
        ),
    }
