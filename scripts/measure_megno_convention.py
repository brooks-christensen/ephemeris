#!/usr/bin/env python3
"""Settle the MEGNO -> Lyapunov conversion factor by measurement.

WHY THE FIRST ATTEMPT AT THIS WAS WRONG
    The earlier `calibrate_megno.py` divided `sim.lyapunov()` by the fitted
    slope of `sim.megno()` and called the quotient the conversion factor. Those
    two numbers are not independent. REBOUND accumulates `lyapunov()` as
    cov(<Y>, t) / var(t) -- the least-squares slope of the very series being
    fitted -- so on any chaotic system the quotient is approximately 1.0
    whatever the convention is. The script could only ever have reported
    "instantaneous_Y", which would have halved every Lyapunov exponent in the
    project. Adding a chaos guard to it did not fix that; the guard addressed
    the noise, not the circularity.

WHAT THIS SCRIPT DOES INSTEAD
    The tangent vector is the ground truth. REBOUND's MEGNO uses first-order
    variational particles and never renormalizes them, so |delta(t)| is
    directly readable from `sim.particles[sim.N_real:]` and

        lambda = d ln|delta| / dt

    is a measurement of the maximal Lyapunov exponent that involves no MEGNO
    convention at all. Three checks are run against it:

    A. IS lyapunov() INDEPENDENT?  Sample <Y> once per fixed WHFast step and
       fit a line to the whole record. If that equals `sim.lyapunov()`, the old
       calibration was circular and `lyapunov()` is a <Y> slope, i.e. lambda/2.

    B. WHAT IS megno() MEASURING?  Reconstruct <Y> from ln|delta| alone using
       `megno_from_log_tangent`. If the reconstruction matches `sim.megno()`,
       then megno() is the time-averaged <Y>, and lambda = 2 * d<Y>/dt follows
       as algebra rather than as a fit.

    C. DOES THE FACTOR COME OUT AT 2?  On a chaotic system whose ln|delta| is
       actually a straight line, compare lambda from the tangent vector against
       d<Y>/dt. Refuses to conclude unless final <Y> >= 5 and the ln|delta| fit
       has R^2 >= 0.95, so a regular system cannot be calibrated on by mistake.

USAGE
    cd ~/ephemeris
    env PYTHONPATH=mini_ephemeris/src .venv/bin/python scripts/measure_megno_convention.py

    A few minutes. Record the full output and the REBOUND version: the
    convention is a property of the library and could change across versions.
"""

from __future__ import annotations

import math
import sys

import numpy as np

try:
    import rebound
except ImportError:  # pragma: no cover - environment guard
    sys.exit("REBOUND is not installed in this environment; run inside .venv")

from mini_ephemeris.chaos_estimator_diagnostics import (
    MEGNO_MEAN_TO_LYAPUNOV,
    REBOUND_LYAPUNOV_TO_LAMBDA,
    calibrate_megno_factor,
    megno_from_log_tangent,
)

MIN_FINAL_MEGNO = 5.0
MIN_TANGENT_R2 = 0.95


def log_tangent_norm(sim: "rebound.Simulation") -> float:
    """ln|delta| over the variational particles, REBOUND's own inner product."""

    total = 0.0
    for particle in sim.particles[sim.N_real:]:
        total += particle.x * particle.x + particle.y * particle.y + particle.z * particle.z
        total += particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz
    return 0.5 * math.log(total)


def two_planet(a2: float, integrator: str, dt: float = 0.01) -> "rebound.Simulation":
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = integrator
    if integrator == "whfast":
        sim.dt = dt
    sim.add(m=1.0)
    sim.add(m=1.0e-3, a=1.00, e=0.05, f=0.0)
    sim.add(m=1.0e-3, a=a2, e=0.05, f=1.7)
    sim.move_to_com()
    sim.init_megno()
    return sim


def test_particle_in_chaotic_zone(a_tp: float) -> "rebound.Simulation":
    """Massless particle in Jupiter's resonance-overlap zone.

    The massive bodies stay on a fixed two-body orbit, so there is no
    scattering event to make lambda non-stationary and ln|delta| stays linear.
    """

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.dt = 0.08
    sim.add(m=1.0)
    sim.add(m=9.5458e-4, a=5.2027, e=0.0484, f=0.0)
    sim.add(m=0.0, a=a_tp, e=0.15, f=1.1, omega=0.4)
    sim.move_to_com()
    sim.init_megno()
    return sim


def line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return float(slope), r2


def check_a_lyapunov_is_a_megno_slope() -> bool:
    print("A. Is Simulation.lyapunov() independent of Simulation.megno()?")
    worst = 0.0
    for label, a2 in (("chaotic", 1.30), ("regular", 1.45)):
        sim = two_planet(a2, "whfast", dt=0.01)
        n = 40_000
        times = np.empty(n)
        mean_megno = np.empty(n)
        for i in range(n):
            sim.step()
            times[i] = sim.t
            mean_megno[i] = sim.megno()
        ols, _ = line_fit(times, mean_megno)
        native = float(sim.lyapunov())
        rel = abs(ols - native) / max(abs(ols), 1e-300)
        worst = max(worst, rel)
        print(f"     {label:8}  OLS slope of <Y> {ols:14.6e}   "
              f"lyapunov() {native:14.6e}   rel diff {rel:.2e}")
    verdict = worst < 1.0e-3
    print(f"     -> lyapunov() {'IS' if verdict else 'is NOT'} the least-squares slope of <Y>.")
    if verdict:
        print("        It is therefore lambda/2, not lambda, and must never be used")
        print("        as the independent lambda in a MEGNO calibration.")
    print()
    return verdict


def check_b_megno_is_the_time_average() -> bool:
    print("B. Is Simulation.megno() the time-averaged <Y>?")
    ok = True
    for label, a2, integrator in (
            ("chaotic", 1.30, "ias15"),
            ("regular", 1.45, "ias15"),
            ("chaotic", 1.30, "whfast"),
    ):
        sim = two_planet(a2, integrator, dt=0.002)
        n = 8_000
        times = np.linspace(0.0, 400.0, n + 1)
        log_norm = np.empty(n + 1)
        native = np.empty(n + 1)
        log_norm[0] = log_tangent_norm(sim)
        native[0] = 0.0
        for i in range(1, n + 1):
            sim.integrate(times[i], exact_finish_time=1)
            log_norm[i] = log_tangent_norm(sim)
            native[i] = sim.megno()
        _, reconstructed = megno_from_log_tangent(times, log_norm)
        tail = slice(n // 2, None)
        denom = np.maximum(np.abs(native[tail]), 1e-12)
        rel = float(np.max(np.abs(reconstructed[tail] - native[tail]) / denom))
        ok = ok and rel < 1.0e-2
        print(f"     {label:8} [{integrator:6}]  final <Y> REBOUND {native[-1]:11.5f}   "
              f"reconstructed {reconstructed[-1]:11.5f}   max rel diff {rel:.2e}")
    print(f"     -> megno() {'IS' if ok else 'is NOT'} the double time average <Y>.")
    if ok:
        print(f"        For steady growth <Y> -> lambda t / 2, so the factor is "
              f"{MEGNO_MEAN_TO_LYAPUNOV}.")
    print()
    return ok


def check_c_measured_factor() -> float | None:
    print("C. Measured factor on a chaotic system with stationary lambda")
    header = (f"     {'a_tp':>6} {'final<Y>':>10} {'R2(ln|d|)':>10} {'lambda':>12} "
              f"{'d<Y>/dt':>12} {'factor':>8} {'lyap()/lam':>11}")
    print(header)
    accepted = []
    for a_tp in (2.90, 2.95, 3.00):
        sim = test_particle_in_chaotic_zone(a_tp)
        total, n = 1.0e6, 2000
        times = np.linspace(total / n, total, n)
        log_norm = np.empty(n)
        mean_megno = np.empty(n)
        escaped = False
        for i, t in enumerate(times):
            sim.integrate(t, exact_finish_time=1)
            log_norm[i] = log_tangent_norm(sim)
            mean_megno[i] = sim.megno()
            if not math.isfinite(log_norm[i]) or abs(sim.particles[2].x) > 100.0:
                escaped = True
                break
        if escaped:
            print(f"     {a_tp:6.2f}   test particle left the system; discarded")
            continue
        cut = (2 * n) // 3
        lam, r2 = line_fit(times[cut:], log_norm[cut:])
        d_mean, _ = line_fit(times[cut:], mean_megno[cut:])
        native = float(sim.lyapunov())
        factor = lam / d_mean if d_mean > 0.0 else float("nan")
        print(f"     {a_tp:6.2f} {mean_megno[-1]:10.3f} {r2:10.5f} {lam:12.4e} "
              f"{d_mean:12.4e} {factor:8.4f} "
              f"{native / lam if lam > 0 else float('nan'):11.4f}")
        if mean_megno[-1] >= MIN_FINAL_MEGNO and r2 >= MIN_TANGENT_R2 and lam > 0 and d_mean > 0:
            accepted.append(factor)
    print()
    if not accepted:
        print("     No system met final <Y> >= %.1f with R^2 >= %.2f; nothing measured here."
              % (MIN_FINAL_MEGNO, MIN_TANGENT_R2))
        return None
    mean_factor = float(np.mean(accepted))
    print(f"     accepted systems: {len(accepted)}   measured factor {mean_factor:.4f}")
    return mean_factor


def main() -> int:
    print(f"REBOUND version: {rebound.__version__}")
    print()
    a_ok = check_a_lyapunov_is_a_megno_slope()
    b_ok = check_b_megno_is_the_time_average()
    measured = check_c_measured_factor()

    print()
    print("VERDICT")
    if not b_ok:
        print("  megno() is not the time average this repository assumes it is.")
        print("  Do not use the MEGNO proxy until this is understood.")
        return 1
    print(f"  MEGNO_MEAN_TO_LYAPUNOV = {MEGNO_MEAN_TO_LYAPUNOV} is correct: megno() is <Y>,")
    print("  which grows as lambda*t/2 for chaos and tends to 2 for regular motion.")
    if measured is not None:
        result = calibrate_megno_factor(
            1.0, measured, lambda_source="tangent_vector"
        )
        print(f"  Independent measurement gives {measured:.4f} -> {result['convention']}.")
    else:
        print("  The direct measurement did not find a usable system this run; the")
        print("  identity in check B is what establishes the factor.")
    if a_ok:
        print()
        print(f"  SEPARATE FINDING: Simulation.lyapunov() is a <Y> slope, so it returns")
        print(f"  lambda/2. Multiply by REBOUND_LYAPUNOV_TO_LAMBDA = "
              f"{REBOUND_LYAPUNOV_TO_LAMBDA} to get lambda.")
        print("  Every value this repository stores as 'lcn' from Simulation.lyapunov()")
        print("  is a factor of two low, and every Lyapunov time from it a factor of")
        print("  two long. That is a separate decision from the MEGNO factor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
