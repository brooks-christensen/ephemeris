#!/usr/bin/env python3
"""Rung 3 of the validation ladder: Pluto's Lyapunov time.

Target ~20 Myr (Applegate et al. 1986; Sussman & Wisdom 1988), acceptance
window 10-40 Myr as fixed in docs/PLAN.md before the run.

Configuration: Sun (with the terrestrial planet masses folded in) plus the four
giant planets and Pluto -- the same system Sussman & Wisdom used. Pluto's chaos
comes from the 3:2 resonance with Neptune and the associated Kozai libration,
so the inner planets are not needed.

Two independent exponents from one integration:

  * Benettin, from the tangent vector itself. REBOUND's MEGNO machinery carries
    first-order variational particles and never renormalises them, so
    ln|delta(t)| is readable directly and lambda = d ln|delta|/dt. The running
    estimate S(T)/T and its halving ratio go through the project's own
    analyze_growth.
  * MEGNO, as 2 x d<Y>/dt, using the factor measured in
    scripts/measure_megno_convention.py.

sim.lyapunov() is deliberately NOT used: it is the least-squares slope of <Y>
over the whole history, so it returns lambda/2 and it is not independent of the
MEGNO estimate.

    python3 scripts/ladder_rung3_pluto.py --years 4e8 --dt 0.4 --json rung3.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_ephemeris" / "src"))

try:
    import rebound
except ImportError:  # pragma: no cover
    sys.exit("REBOUND is not installed in this environment; run inside .venv")

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    MEGNO_MEAN_TO_LYAPUNOV,
    analyze_growth,
)
from mini_ephemeris.validation_ladder import evaluate_rung  # noqa: E402

DEG = math.pi / 180.0

# Sun plus Mercury..Mars, as in the classical outer-solar-system integrations.
SOLAR_MASS_WITH_TERRESTRIALS = 1.0000059

# m [Msun], a [AU], e, inc, Omega, omega, M  (degrees, J2000 heliocentric)
BODIES = (
    ("Jupiter", 9.5479e-4, 5.2044, 0.0489, 1.303, 100.464, 273.867, 20.020),
    ("Saturn",  2.8588e-4, 9.5826, 0.0565, 2.485, 113.665, 339.392, 317.020),
    ("Uranus",  4.3662e-5, 19.2184, 0.0457, 0.773, 74.006, 96.998, 142.238),
    ("Neptune", 5.1514e-5, 30.110, 0.0113, 1.770, 131.784, 276.336, 256.228),
    ("Pluto",   6.55e-9,  39.482, 0.2488, 17.16, 110.299, 113.834, 14.53),
)

TARGET_LYAPUNOV_TIME_YEARS = 2.0e7
ACCEPTANCE_YEARS = (1.0e7, 4.0e7)

# Pre-flight physics check, added after the first attempt at this rung.
#
# Pluto's ~20 Myr Lyapunov time is a property of a Pluto *protected by the 3:2
# mean-motion resonance with Neptune*: the resonant argument
#
#     phi = 3*lambda_Pluto - 2*lambda_Neptune - varpi_Pluto
#
# librates about 180 deg, which keeps Pluto away from Neptune even though its
# perihelion (29.7 AU) lies inside Neptune's orbit (30.1 AU). Take Pluto out of
# the resonance and you are measuring a different system.
#
# Built from the rounded J2000 osculating elements in BODIES, phi CIRCULATES
# through the full 360 deg -- the resonance is not captured. So this rung
# cannot run on tabulated mean elements; it needs real ephemeris initial
# conditions (HORIZONS, or a DE44x kernel through skyfield). The check below is
# a hard precondition so that a number produced from an unprotected Pluto can
# never be reported as Pluto's Lyapunov time.
MAX_LIBRATION_AMPLITUDE_DEG = 330.0
RESONANCE_CHECK_YEARS = 3.0e5
RESONANCE_CHECK_SAMPLES = 3000


def build(dt: float) -> "rebound.Simulation":
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.dt = dt
    sim.add(m=SOLAR_MASS_WITH_TERRESTRIALS)
    for _name, m, a, e, inc, node, peri, mean in BODIES:
        sim.add(m=m, a=a, e=e, inc=inc * DEG, Omega=node * DEG,
                omega=peri * DEG, M=mean * DEG)
    sim.move_to_com()
    sim.init_megno()
    return sim


def check_pluto_resonance(
    dt: float,
    years: float = RESONANCE_CHECK_YEARS,
    samples: int = RESONANCE_CHECK_SAMPLES,
) -> dict:
    """Does the 3:2 resonant argument librate? Precondition, not a diagnostic."""

    sim = build(dt)
    phis: list[float] = []
    separations: list[float] = []
    semi_major: list[float] = []
    for t in np.linspace(years / samples, years, samples):
        sim.integrate(t, exact_finish_time=0)
        pluto = sim.particles[5].orbit(primary=sim.particles[0])
        neptune = sim.particles[4].orbit(primary=sim.particles[0])
        phi = (3.0 * pluto.l - 2.0 * neptune.l
               - (pluto.Omega + pluto.omega)) % (2.0 * math.pi)
        phis.append(math.degrees(phi))
        offset = sim.particles[5] - sim.particles[4]
        separations.append(math.sqrt(offset.x ** 2 + offset.y ** 2 + offset.z ** 2))
        semi_major.append(pluto.a)

    wrapped = np.where(np.array(phis) > 180.0, np.array(phis) - 360.0, np.array(phis))
    amplitude = float(wrapped.max() - wrapped.min())
    return {
        "libration_amplitude_deg": amplitude,
        "librating": amplitude < MAX_LIBRATION_AMPLITUDE_DEG,
        "min_pluto_neptune_separation_au": float(min(separations)),
        "pluto_a_range_au": [float(min(semi_major)), float(max(semi_major))],
        "checked_years": years,
    }


def log_tangent_norm(sim: "rebound.Simulation") -> float:
    total = 0.0
    for p in sim.particles[sim.N_real:]:
        total += p.x * p.x + p.y * p.y + p.z * p.z
        total += p.vx * p.vx + p.vy * p.vy + p.vz * p.vz
    return 0.5 * math.log(total)


def run(years: float, dt: float, samples: int, progress: Path | None) -> dict:
    sim = build(dt)
    energy0 = sim.energy()
    log0 = log_tangent_norm(sim)

    times = np.linspace(years / samples, years, samples)
    growth = np.empty(samples)
    mean_megno = np.empty(samples)
    drift = 0.0
    started = time.time()

    for i, t in enumerate(times):
        sim.integrate(t, exact_finish_time=0)
        growth[i] = log_tangent_norm(sim) - log0
        mean_megno[i] = sim.megno()
        drift = max(drift, abs((sim.energy() - energy0) / energy0))
        if progress is not None and i % 20 == 0:
            elapsed = time.time() - started
            progress.write_text(
                f"{i + 1}/{samples}  t={t:.3e} yr  S={growth[i]:.4f}  "
                f"<Y>={mean_megno[i]:.4f}  dE/E={drift:.2e}  "
                f"elapsed={elapsed / 60:.1f} min\n"
            )

    return {
        "times": times,
        "growth": growth,
        "mean_megno": mean_megno,
        "max_relative_energy_drift": drift,
        "wall_seconds": time.time() - started,
        "dt": dt,
        "years": years,
    }


def summarise(data: dict) -> dict:
    result = analyze_growth(
        data["times"], data["growth"],
        max_relative_energy_drift=data["max_relative_energy_drift"],
    )
    half = len(data["times"]) // 2
    megno_slope = float(np.polyfit(data["times"][half:], data["mean_megno"][half:], 1)[0])
    lambda_megno = megno_slope * MEGNO_MEAN_TO_LYAPUNOV
    lambda_benettin = result.lambda_running_final
    agreement = (
        abs(lambda_megno - lambda_benettin) / abs(lambda_benettin)
        if lambda_benettin > 0 else math.nan
    )
    return {
        "lambda_benettin_1_per_year": lambda_benettin,
        "lambda_megno_1_per_year": lambda_megno,
        "lyapunov_time_years": 1.0 / lambda_benettin if lambda_benettin > 0 else math.inf,
        "lyapunov_time_myr": (1.0 / lambda_benettin) / 1.0e6 if lambda_benettin > 0 else math.inf,
        "halving_ratio": result.halving_ratio,
        "classification": result.classification,
        "final_mean_megno": float(data["mean_megno"][-1]),
        "megno_slope_per_year": megno_slope,
        "estimator_relative_disagreement": agreement,
        "max_relative_energy_drift": data["max_relative_energy_drift"],
        "final_cumulative_log_growth": float(data["growth"][-1]),
        "dt_years": data["dt"],
        "duration_years": data["years"],
        "wall_minutes": data["wall_seconds"] / 60.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=4.0e8)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument("--skip-resonance-check", action="store_true",
                        help="diagnostic only; the rung cannot PASS without it")
    args = parser.parse_args(argv)

    print(f"REBOUND {rebound.__version__}   {args.years:.3e} yr at dt = {args.dt} yr")

    if args.skip_resonance_check:
        resonance = {"librating": False, "skipped": True,
                     "note": "precondition skipped; this run cannot pass"}
    else:
        print("  pre-flight: is Pluto in the 3:2 resonance?", flush=True)
        resonance = check_pluto_resonance(args.dt)
        for key, value in resonance.items():
            print(f"    {key}: {value}")
        if not resonance["librating"]:
            print()
            print("  STOP. The 3:2 resonant argument circulates: this Pluto is not")
            print("  resonance-protected, so its Lyapunov time is not the ~20 Myr")
            print("  figure this rung targets. Supply real ephemeris initial")
            print("  conditions (HORIZONS or a DE44x kernel) and try again. Do not")
            print("  widen the acceptance window to accommodate a different system.")
            return 1

    summary = summarise(run(args.years, args.dt, args.samples, args.progress))
    for key, value in summary.items():
        print(f"  {key}: {value}")

    result = evaluate_rung(
        "3",
        "Pluto Lyapunov time",
        measured=summary["lyapunov_time_years"],
        target=TARGET_LYAPUNOV_TIME_YEARS,
        acceptance=ACCEPTANCE_YEARS,
        unit="years",
        conditions=(
            ("classified chaotic", summary["classification"] == "chaotic_candidate"),
            ("halving ratio near 1", 0.85 <= summary["halving_ratio"] <= 1.15),
            ("energy drift < 1e-9", summary["max_relative_energy_drift"] < 1.0e-9),
            ("Benettin and MEGNO agree within 20%",
             summary["estimator_relative_disagreement"] < 0.20),
            ("Pluto is protected by the 3:2 resonance", bool(resonance["librating"])),
        ),
        evidence={**summary, "resonance_precondition": resonance},
    )
    print()
    print(result.one_line())
    for label, ok in result.conditions:
        print(f"    {'ok  ' if ok else 'BAD '} {label}")
    if args.json is not None:
        args.json.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"  wrote {args.json}")
    return 0 if result.status.value == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
