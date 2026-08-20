#!/usr/bin/env python3
"""Settle the MEGNO convention by measurement.

Two different quantities are called "MEGNO":

    Y(t)      instantaneous, grows as lambda * t
    <Y>(t)    time-averaged,  grows as lambda * t / 2, tends to 2 for regular

The conversion factor from MEGNO slope to Lyapunov exponent is therefore 1.0
for the first and 2.0 for the second. This repository previously used 0.5,
which is wrong under both.

`long_term_stability_cli` now assumes the time-averaged convention (factor
2.0), which is what REBOUND's documentation describes -- but assumption is not
measurement, and the output says so. This script replaces the assumption with
a measurement.

METHOD
    REBOUND exposes both quantities from the same set of variational
    particles: `sim.megno()` and `sim.lyapunov()`. Integrate a chaotic system,
    fit d(megno)/dt, and compare against sim.lyapunov(). The ratio is the
    conversion factor, and it identifies the convention directly.

USAGE
    cd ~/ephemeris
    env PYTHONPATH=mini_ephemeris/src .venv/bin/python calibrate_megno.py

    Runs in a couple of minutes. Prints the implied factor and the verdict.

WHAT TO DO WITH THE RESULT
    If it reports mean_Y, MEGNO_MEAN_TO_LYAPUNOV = 2.0 in
    chaos_estimator_diagnostics.py is correct and nothing changes -- but the
    assumption is now measured, and that fact belongs in the closeout.

    If it reports instantaneous_Y, change the factor used in
    `long_term_stability_cli.megno_slope_estimates` from
    MEGNO_MEAN_TO_LYAPUNOV to MEGNO_INSTANTANEOUS_TO_LYAPUNOV, and treat every
    previously reported lyapunov_proxy_1_per_year as wrong by 2x.

    Either way, record the measured factor and the REBOUND version, because
    the convention is a property of the library and could change across
    versions.
"""

from __future__ import annotations

import math
import sys

import numpy as np

try:
    import rebound
except ImportError:  # pragma: no cover
    sys.exit("REBOUND is not installed in this environment; run inside .venv")

from mini_ephemeris.chaos_estimator_diagnostics import calibrate_megno_factor


def chaotic_simulation() -> rebound.Simulation:
    """Two massive planets close enough to overlap resonances.

    This is deliberately, unambiguously chaotic -- the point is to get a
    Lyapunov time short enough to measure in a few thousand orbits, not to
    model anything real.
    """
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.dt = 0.01
    sim.add(m=1.0)
    sim.add(m=1.0e-3, a=1.00, e=0.05, f=0.0)
    sim.add(m=1.0e-3, a=1.35, e=0.05, f=1.7)
    sim.move_to_com()
    sim.init_megno()
    return sim


def main() -> int:
    print(f"REBOUND version: {rebound.__version__}")
    sim = chaotic_simulation()

    total_years = 20_000.0
    n_samples = 400
    times = np.linspace(total_years / n_samples, total_years, n_samples)

    megno_values = []
    for t in times:
        sim.integrate(t)
        megno_values.append(sim.megno())
    megno_values = np.asarray(megno_values, dtype=float)

    lyapunov = float(sim.lyapunov())          # 1/yr, REBOUND's own MLE

    # Fit the MEGNO slope over the second half, past the transient.
    half = len(times) // 2
    slope = float(np.polyfit(times[half:], megno_values[half:], 1)[0])

    print()
    print(f"  integrated             : {total_years:,.0f} yr")
    print(f"  final MEGNO            : {megno_values[-1]:.4f}   (-> 2 for regular)")
    print(f"  d(MEGNO)/dt            : {slope:.6e} per yr")
    print(f"  sim.lyapunov()         : {lyapunov:.6e} per yr")

    if not (math.isfinite(slope) and slope > 0.0 and math.isfinite(lyapunov)):
        print()
        print("  The system did not become measurably chaotic. Increase")
        print("  total_years, or bring the planets closer together.")
        return 1

    result = calibrate_megno_factor(slope, lyapunov)
    print()
    print(f"  IMPLIED FACTOR         : {result['implied_factor']:.4f}")
    print(f"  CONVENTION             : {result['convention']}")
    print(f"  {result['note']}")
    print()

    if result["convention"] == "mean_Y":
        print("  -> MEGNO_MEAN_TO_LYAPUNOV = 2.0 is correct. No code change needed.")
    elif result["convention"] == "instantaneous_Y":
        print("  -> The factor must be 1.0. Change long_term_stability_cli to use")
        print("     MEGNO_INSTANTANEOUS_TO_LYAPUNOV, and treat every previously")
        print("     reported lyapunov_proxy_1_per_year as wrong by 2x.")
    else:
        print("  -> Neither convention matches. Do not use the MEGNO proxy until")
        print("     this is understood; something else is wrong.")

    print()
    print("  Record this output, with the REBOUND version, in the closeout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
