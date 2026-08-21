"""Rungs 0-2 of the validation ladder: cheap, self-contained, decisive.

These three rungs need no REBOUND, no SPICE kernel and no solar system. They
run in seconds. Between them they would have caught every estimator defect
found in this codebase:

* Rung 0 executes the unit tests instead of transcribing their names.
* Rung 1 puts the estimator on an integrable two-body orbit, where lambda is
  exactly zero. The legacy line-fit estimator failed this by reporting a
  Lyapunov time of 0.35 x run duration, at every duration.
* Rung 2 puts the Benettin machinery on maps whose Lyapunov exponent is known
  in closed form, so a wrong renormalisation or a wrong accumulation shows up
  as a number that misses a target rather than as a plausible-looking result.

Rungs 3-5 (Pluto, the inner solar system, the GR sign test) need the real
pipeline and live elsewhere.
"""

from __future__ import annotations

import io
import math
import time
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from .chaos_estimator_diagnostics import analyze_growth
from .validation_ladder import RungResult, RungStatus, evaluate_rung

__all__ = [
    "PENDING_RUNGS",
    "rung0_unit_tests",
    "rung1_integrable_two_body",
    "rung2a_cat_map",
    "rung2b_standard_map",
    "DEPENDENCY_FREE_TEST_MODULES",
    "CAT_MAP_LAMBDA",
]

# Rungs that docs/PLAN.md declares but this harness does not implement. They
# need REBOUND, a kernel and real integration time, so they run in WSL, not
# here. Listed so a green rung 0-2 report cannot be mistaken for a validated
# pipeline: the report prints them as explicitly not run.
PENDING_RUNGS = (
    ("3", "Pluto Lyapunov time in 10-40 Myr (target ~20 Myr)"),
    ("4", "inner solar system Lyapunov time in 3-10 Myr (target ~5 Myr), GR on"),
    ("5", "GR sign test: g1-g5 detuning, secular diffusion increases with GR off"),
)

# Rung 0 runs these by name. The list is explicit so the report can state
# exactly what was executed; there is no hidden scope and no "and everything
# else looked fine".
DEPENDENCY_FREE_TEST_MODULES = (
    "test_chaos_estimator_regression",
    "test_megno_convention",
    "test_gr_physics_regression",
    "test_validation_ladder",
    "test_window_slopes",
)

# Arnold's cat map: the tangent map is the constant matrix [[2,1],[1,1]], whose
# larger eigenvalue is the golden ratio squared. lambda is exact, identical for
# every trajectory, with no asymptotics and no finite-size correction.
CAT_MAP_LAMBDA = math.log((3.0 + math.sqrt(5.0)) / 2.0)


# --------------------------------------------------------------------------
# Rung 0 -- actually run the tests
# --------------------------------------------------------------------------

def rung0_unit_tests(
    test_dir: Path | None = None,
    modules: Sequence[str] = DEPENDENCY_FREE_TEST_MODULES,
) -> RungResult:
    """Execute the dependency-free unit tests and report real counts."""

    started = time.monotonic()
    if test_dir is None:
        test_dir = Path(__file__).resolve().parents[2] / "tests"
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    missing: list[str] = []
    for module in modules:
        if not (test_dir / f"{module}.py").is_file():
            missing.append(module)
            continue
        suite.addTests(loader.discover(str(test_dir), pattern=f"{module}.py"))

    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    broken = len(outcome.failures) + len(outcome.errors)

    return evaluate_rung(
        "0",
        "unit tests actually executed",
        measured=float(broken),
        target=0.0,
        acceptance=(0.0, 0.0),
        unit="failures+errors",
        conditions=(
            ("every named test module was found", not missing),
            ("at least one test ran", outcome.testsRun > 0),
        ),
        duration_seconds=time.monotonic() - started,
        evidence={
            "modules": list(modules),
            "missing_modules": missing,
            "tests_run": outcome.testsRun,
            "failures": len(outcome.failures),
            "errors": len(outcome.errors),
            "skipped": len(outcome.skipped),
        },
        notes=tuple(
            f"{kind}: {test}" for kind, group in
            (("FAILURE", outcome.failures), ("ERROR", outcome.errors))
            for test, _ in group
        ),
    )


# --------------------------------------------------------------------------
# Rung 1 -- integrable two-body, lambda = 0 exactly
# --------------------------------------------------------------------------

# Yoshida 4th-order symplectic coefficients (Forest-Ruth). Two reasons for
# using them here rather than velocity-Verlet:
#
#   1. Accuracy per force evaluation. At 1000 steps per orbit the energy
#      oscillation is ~1e-10, far below the estimator's 1e-7 chaos gate, so a
#      "regular" verdict on this rung is never an artifact of the gate firing.
#      An under-resolved two-body harness manufactures fake chaos; that has
#      already happened once in this project.
#   2. No cached acceleration. Every force evaluation happens inside the step,
#      so nothing survives across a renormalisation boundary. The first run of
#      this rung failed because velocity-Verlet's cached variational
#      acceleration was NOT recomputed after the tangent vector was rescaled:
#      the stale value, of order the pre-rescaling norm, was applied as a kick
#      to the rescaled vector, injecting growth of ~exp(0.03 t) into an
#      integrable system. Structure the integrator so the bug is unwriteable.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)
_YOSHIDA_DRIFTS = (_W1 / 2.0, (_W0 + _W1) / 2.0, (_W0 + _W1) / 2.0, _W1 / 2.0)
_YOSHIDA_KICKS = (_W1, _W0, _W1, 0.0)


def _two_body_benettin(
    n_orbits: int,
    steps_per_orbit: int,
    renorm_steps: int,
    eccentricity: float = 0.2,
) -> dict:
    """Yoshida-4 on the Kepler problem with its exact variational flow.

    The variational acceleration

        d(delta_a) = -mu [ delta_r / r^3 - 3 (r . delta_r) r / r^5 ]

    is position-only, so the symplectic structure is preserved and the tangent
    vector grows *linearly* in time, not exponentially -- neighbouring orbits
    with slightly different semi-major axes drift apart at a constant rate.
    That linear growth makes S(t) = ln|delta| ~ ln(t), which is exactly the case
    the legacy line-fit estimator misread as an exponent.
    """

    mu = 1.0
    semi_major = 1.0
    period = 2.0 * math.pi * math.sqrt(semi_major ** 3 / mu)
    dt = period / steps_per_orbit
    total_steps = n_orbits * steps_per_orbit

    x, y = semi_major * (1.0 - eccentricity), 0.0        # start at pericentre
    vx = 0.0
    vy = math.sqrt(mu * (1.0 + eccentricity) / (semi_major * (1.0 - eccentricity)))
    dx, dy, dvx, dvy = 0.5, 0.5, 0.5, 0.5                # unit norm in 4-D phase space

    energy0 = 0.5 * (vx * vx + vy * vy) - mu / math.hypot(x, y)
    max_excursion = 0.0

    times: list[float] = []
    growth: list[float] = []
    cumulative = 0.0

    for step in range(1, total_steps + 1):
        for drift, kick in zip(_YOSHIDA_DRIFTS, _YOSHIDA_KICKS):
            cdt = drift * dt
            x += cdt * vx
            y += cdt * vy
            dx += cdt * dvx
            dy += cdt * dvy
            if kick == 0.0:
                continue
            ddt = kick * dt
            r2 = x * x + y * y
            r = math.sqrt(r2)
            r3 = r2 * r
            r5 = r3 * r2
            vx -= ddt * mu * x / r3
            vy -= ddt * mu * y / r3
            dot = x * dx + y * dy
            dvx -= ddt * mu * (dx / r3 - 3.0 * dot * x / r5)
            dvy -= ddt * mu * (dy / r3 - 3.0 * dot * y / r5)

        if step % renorm_steps == 0:
            norm = math.sqrt(dx * dx + dy * dy + dvx * dvx + dvy * dvy)
            if not math.isfinite(norm) or norm <= 0.0:
                raise ArithmeticError("tangent vector collapsed; harness is broken")
            cumulative += math.log(norm)
            scale = 1.0 / norm
            dx *= scale
            dy *= scale
            dvx *= scale
            dvy *= scale

            energy = 0.5 * (vx * vx + vy * vy) - mu / math.hypot(x, y)
            max_excursion = max(max_excursion, abs((energy - energy0) / energy0))
            times.append(step * dt)
            growth.append(cumulative)

    return {
        "times": times,
        "growth": growth,
        "duration": total_steps * dt,
        "orbits": n_orbits,
        "max_relative_energy_drift": max_excursion,
        "steps_per_orbit": steps_per_orbit,
    }


def rung1_integrable_two_body(
    short_orbits: int = 250,
    long_orbits: int = 1000,
    steps_per_orbit: int = 1000,
) -> RungResult:
    """lambda is exactly zero here. The estimator must not report chaos."""

    started = time.monotonic()
    short = _two_body_benettin(short_orbits, steps_per_orbit, steps_per_orbit // 10)
    long = _two_body_benettin(long_orbits, steps_per_orbit, steps_per_orbit // 10)

    short_result = analyze_growth(
        short["times"], short["growth"],
        max_relative_energy_drift=short["max_relative_energy_drift"],
    )
    long_result = analyze_growth(
        long["times"], long["growth"],
        max_relative_energy_drift=long["max_relative_energy_drift"],
    )

    resolved = max(
        short["max_relative_energy_drift"], long["max_relative_energy_drift"]
    ) < 1.0e-9

    # The legacy artifact signature: the line-fit Lyapunov time is a fixed
    # fraction of the run, so it barely moves when the run gets 4x longer.
    fractions = (short_result.line_fit_time_fraction, long_result.line_fit_time_fraction)
    artifact_fraction_spread = (
        abs(fractions[0] - fractions[1]) / max(abs(fractions[0]), 1e-30)
        if all(math.isfinite(f) for f in fractions) else math.nan
    )

    return evaluate_rung(
        "1",
        "integrable two-body reads as regular",
        measured=long_result.halving_ratio,
        target=0.5,
        # Two-sided, and the lower bound matters. For logarithmic tangent growth
        # the halving ratio is ln(T)/(2(ln T - ln 2)), which approaches 0.5 from
        # ABOVE and cannot sit near zero. A one-sided (0, 0.70) window let a
        # mutation swap in the line-fit slope (4.8e-4) and still pass, because
        # any small number satisfied it. A headline measurement that accepts
        # almost anything is decorative, which is the failure mode this whole
        # ladder exists to remove.
        acceptance=(0.40, 0.70),
        unit="halving ratio",
        conditions=(
            # "regular_likely", not merely "not chaotic". lambda is zero here by
            # theorem, so "ambiguous" is a failure to measure, not a pass.
            ("short run classified regular",
             short_result.classification == "regular_likely"),
            ("long run classified regular",
             long_result.classification == "regular_likely"),
            ("integration resolved (energy excursion < 1e-9)", resolved),
            ("logarithmic model preferred over linear", long_result.log_model_preferred),
        ),
        duration_seconds=time.monotonic() - started,
        evidence={
            "short": {
                "orbits": short_orbits,
                "classification": short_result.classification,
                "halving_ratio": short_result.halving_ratio,
                "lambda_running_final": short_result.lambda_running_final,
                "line_fit_lambda": short_result.line_fit_lambda,
                "line_fit_time_fraction": short_result.line_fit_time_fraction,
                "max_relative_energy_drift": short["max_relative_energy_drift"],
            },
            "long": {
                "orbits": long_orbits,
                "classification": long_result.classification,
                "halving_ratio": long_result.halving_ratio,
                "lambda_running_final": long_result.lambda_running_final,
                "line_fit_lambda": long_result.line_fit_lambda,
                "line_fit_time_fraction": long_result.line_fit_time_fraction,
                "max_relative_energy_drift": long["max_relative_energy_drift"],
            },
            "legacy_artifact": {
                "line_fit_time_fraction_short": fractions[0],
                "line_fit_time_fraction_long": fractions[1],
                "relative_spread_over_4x_duration": artifact_fraction_spread,
                "comment": (
                    "A genuine Lyapunov time is a property of the dynamics and "
                    "does not scale with how long you ran. These two numbers "
                    "staying together across a 4x change in duration is the "
                    "signature of the line-fit artifact, recorded here so the "
                    "defect stays visible after it was fixed."
                ),
            },
        },
    )


# --------------------------------------------------------------------------
# Rung 2 -- maps with a known Lyapunov exponent
# --------------------------------------------------------------------------

def _map_benettin(step_fn, tangent_fn, state, tangent, n_steps: int) -> dict:
    """Benettin with renormalisation every iteration."""

    cumulative = 0.0
    times: list[float] = []
    growth: list[float] = []
    for step in range(1, n_steps + 1):
        tangent = tangent_fn(state, tangent)
        state = step_fn(state)
        norm = math.hypot(*tangent)
        if not math.isfinite(norm) or norm <= 0.0:
            raise ArithmeticError("tangent vector collapsed; harness is broken")
        cumulative += math.log(norm)
        tangent = (tangent[0] / norm, tangent[1] / norm)
        if step % 10 == 0:
            times.append(float(step))
            growth.append(cumulative)
    return {"times": times, "growth": growth, "final_state": state}


def rung2a_cat_map(n_steps: int = 20_000) -> RungResult:
    """Arnold's cat map. lambda = ln((3+sqrt 5)/2), exactly, for every orbit."""

    started = time.monotonic()

    def step_fn(s):
        x, y = s
        return ((2.0 * x + y) % 1.0, (x + y) % 1.0)

    def tangent_fn(_s, t):
        dx, dy = t
        return (2.0 * dx + dy, dx + dy)

    run = _map_benettin(step_fn, tangent_fn, (0.1234567, 0.7654321), (1.0, 0.0), n_steps)
    result = analyze_growth(run["times"], run["growth"])
    relative_error = abs(result.lambda_running_final - CAT_MAP_LAMBDA) / CAT_MAP_LAMBDA

    return evaluate_rung(
        "2a",
        "cat map recovers its exact Lyapunov exponent",
        measured=result.lambda_running_final,
        target=CAT_MAP_LAMBDA,
        acceptance=(CAT_MAP_LAMBDA * 0.99, CAT_MAP_LAMBDA * 1.01),
        unit="per iteration",
        conditions=(
            ("classified chaotic", result.classification == "chaotic_candidate"),
            ("halving ratio near 1", 0.85 <= result.halving_ratio <= 1.15),
        ),
        duration_seconds=time.monotonic() - started,
        evidence={
            "iterations": n_steps,
            "exact_lambda": CAT_MAP_LAMBDA,
            "relative_error": relative_error,
            "halving_ratio": result.halving_ratio,
            "classification": result.classification,
            "linear_r_squared": result.linear_r_squared,
        },
    )


def rung2b_standard_map(k: float = 97.0, n_steps: int = 200_000) -> RungResult:
    """Chirikov standard map. For K >> 1, lambda -> ln(K/2).

    K is deliberately not near a multiple of 2*pi: accelerator modes there
    create stable islands that a single trajectory can be trapped in, which
    would make this rung fail for dynamical reasons rather than code reasons.
    """

    started = time.monotonic()
    target = math.log(k / 2.0)
    two_pi = 2.0 * math.pi

    def step_fn(s):
        p, theta = s
        p_new = p + k * math.sin(theta)
        return (p_new, (theta + p_new) % two_pi)

    def tangent_fn(s, t):
        _p, theta = s
        dp, dtheta = t
        cos_term = k * math.cos(theta)
        dp_new = dp + cos_term * dtheta
        return (dp_new, dp_new + dtheta)

    run = _map_benettin(step_fn, tangent_fn, (0.31, 1.7), (1.0, 0.0), n_steps)
    result = analyze_growth(run["times"], run["growth"])
    relative_error = abs(result.lambda_running_final - target) / target

    return evaluate_rung(
        "2b",
        "standard map recovers ln(K/2)",
        measured=result.lambda_running_final,
        target=target,
        acceptance=(target * 0.95, target * 1.05),
        unit="per iteration",
        conditions=(
            ("classified chaotic", result.classification == "chaotic_candidate"),
            ("halving ratio near 1", 0.85 <= result.halving_ratio <= 1.15),
        ),
        duration_seconds=time.monotonic() - started,
        evidence={
            "K": k,
            "iterations": n_steps,
            "asymptotic_lambda_ln_K_over_2": target,
            "relative_error": relative_error,
            "halving_ratio": result.halving_ratio,
            "classification": result.classification,
        },
    )
