from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import time

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .long_term_stability_cli import (
    MODEL_SCOPES,
    ReboundMegnoSample,
    classify_megno_result,
    plot_megno_growth,
    stability_body_list,
)
from .orbital_elements import AU_M, DAY_S, JULIAN_YEAR_S, seconds_to_years


SOLAR_SCOPES = ("two_body_jupiter", "two_body_saturn", "inner", "full")
TOY_SCOPES = ("chaotic_three_body", "resonant_test_particle")


def parse_start_datetime(text: str) -> dt.datetime:
    if "T" in text:
        value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        value = dt.datetime.combine(dt.date.fromisoformat(text), dt.time(), tzinfo=dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run REBOUND-native MEGNO positive-control and regular-control validations."
    )
    parser.add_argument("--kernel-path", default="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument(
        "--model-scope",
        choices=SOLAR_SCOPES + TOY_SCOPES,
        default="chaotic_three_body",
    )
    parser.add_argument("--duration-years", type=float, default=None)
    parser.add_argument("--step-days", type=float, default=None)
    parser.add_argument("--record-every-years", type=float, default=None)
    parser.add_argument("--integrator", choices=["whfast", "ias15"], default="ias15")
    parser.add_argument("--megno-seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability/chaos_positive_control")
    parser.add_argument("--tag", default="chaos_positive_control")
    parser.add_argument("--no-progress-bar", action="store_true", help="Accepted for script compatibility; no progress bar is used.")
    return parser


def add_solar_system(rebound, sim, args: argparse.Namespace) -> tuple[float, str]:
    bodies = stability_body_list(args.model_scope, include_pluto=False)
    state = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    sim.G = 6.67430e-11
    for position, velocity, mass in zip(state.positions, state.velocities, state.masses):
        sim.add(
            m=float(mass),
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            vz=float(velocity[2]),
        )
    step_s = (args.step_days if args.step_days is not None else 4.0) * DAY_S
    sim.dt = step_s
    return JULIAN_YEAR_S, "SI years"


def add_chaotic_three_body(rebound, sim, args: argparse.Namespace) -> tuple[float, str]:
    # Compact equal-mass triple in crossing geometry. This is intentionally a
    # toy positive control, not a Solar System model.
    sim.G = 1.0
    sim.add(m=1.0, x=-1.0, y=0.0, vx=0.05, vy=0.42)
    sim.add(m=1.0, x=1.0, y=0.0, vx=0.05, vy=-0.42)
    sim.add(m=1.0, x=0.0, y=0.25, vx=-0.10, vy=0.0)
    sim.move_to_com()
    sim.dt = args.step_days if args.step_days is not None else 0.01
    return 1.0, "dimensionless toy time"


def add_resonant_test_particle(rebound, sim, args: argparse.Namespace) -> tuple[float, str]:
    sim.G = 4.0 * math.pi * math.pi
    sim.add(m=1.0)
    sim.add(m=9.545e-4, a=5.2, e=0.048)
    sim.add(m=0.0, a=2.5, e=0.15, inc=math.radians(5.0))
    sim.move_to_com()
    sim.dt = (args.step_days if args.step_days is not None else 5.0) / 365.25
    return 1.0, "Julian years with AU/Msun units"


def classify(final_megno: float, lcn: float, duration: float, max_radius: float, min_sep: float, scope: str) -> str:
    if scope in SOLAR_SCOPES:
        return classify_megno_result(
            final_megno=final_megno,
            estimated_lyapunov_per_year=lcn,
            duration_years=duration,
            model_scope=scope,
        )
    if math.isfinite(max_radius) and max_radius > 100.0:
        return "unstable_or_escape"
    if math.isfinite(min_sep) and min_sep < 1.0e-4:
        return "unstable_or_escape"
    if math.isfinite(final_megno) and final_megno > 10.0:
        return "chaotic_candidate"
    if math.isfinite(lcn) and lcn > 0.0 and lcn * max(duration, 1.0) > 1.0:
        return "chaotic_candidate"
    if math.isfinite(final_megno) and abs(final_megno - 2.0) <= 3.0:
        return "regular_likely"
    return "ambiguous"


def pairwise_stats(sim) -> tuple[float, float]:
    particles = sim.particles[: sim.N_real]
    max_radius = 0.0
    min_sep = math.inf
    for i, particle in enumerate(particles):
        radius = math.sqrt(particle.x * particle.x + particle.y * particle.y + particle.z * particle.z)
        max_radius = max(max_radius, radius)
        for other in particles[i + 1 :]:
            dx = particle.x - other.x
            dy = particle.y - other.y
            dz = particle.z - other.z
            min_sep = min(min_sep, math.sqrt(dx * dx + dy * dy + dz * dz))
    return max_radius, min_sep


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration_years is None:
        args.duration_years = 200.0 if args.model_scope in TOY_SCOPES else 1000.0
    if args.record_every_years is None:
        args.record_every_years = max(args.duration_years / 200.0, 1.0e-3)

    try:
        import rebound
    except ImportError as exc:
        raise SystemExit("REBOUND is not installed; install rebound to run this validation.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in args.tag)
    csv_path = output_dir / f"chaos_positive_control_{tag}.csv"
    summary_path = output_dir / f"chaos_positive_control_summary_{tag}.json"
    plot_path = output_dir / f"chaos_positive_control_{tag}.png"

    sim = rebound.Simulation()
    if args.model_scope in SOLAR_SCOPES:
        time_scale, unit_note = add_solar_system(rebound, sim, args)
    elif args.model_scope == "chaotic_three_body":
        time_scale, unit_note = add_chaotic_three_body(rebound, sim, args)
    else:
        time_scale, unit_note = add_resonant_test_particle(rebound, sim, args)

    sim.integrator = args.integrator
    if args.integrator == "whfast" and sim.dt == 0.0:
        sim.dt = 0.01
    sim.init_megno(seed=args.megno_seed)

    duration = args.duration_years * time_scale
    record_interval = args.record_every_years * time_scale
    times = list(np.arange(0.0, duration + 0.5 * record_interval, record_interval))
    if not times or abs(times[-1] - duration) > 1.0e-9:
        times.append(duration)

    samples: list[ReboundMegnoSample] = []
    max_radius_seen = 0.0
    min_sep_seen = math.inf
    start = time.perf_counter()
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "time_years",
                "megno",
                "finite_time_lyapunov_estimate",
                "max_radius",
                "min_pairwise_separation",
                "classification_hint",
            ],
        )
        writer.writeheader()
        for target in times:
            sim.integrate(float(target), exact_finish_time=1)
            max_radius, min_sep = pairwise_stats(sim)
            max_radius_seen = max(max_radius_seen, max_radius)
            min_sep_seen = min(min_sep_seen, min_sep)
            megno = float(sim.megno())
            lcn = float(sim.lyapunov()) * time_scale
            time_out = float(sim.t) / time_scale
            hint = classify(megno, lcn, time_out, max_radius_seen, min_sep_seen, args.model_scope)
            writer.writerow(
                {
                    "time_years": time_out,
                    "megno": megno,
                    "finite_time_lyapunov_estimate": lcn,
                    "max_radius": max_radius_seen,
                    "min_pairwise_separation": min_sep_seen,
                    "classification_hint": hint,
                }
            )
            samples.append(
                ReboundMegnoSample(
                    time_years=time_out,
                    megno=megno,
                    mean_megno=math.nan,
                    finite_time_lyapunov_estimate=lcn,
                )
            )
            if hint == "unstable_or_escape":
                break

    runtime = time.perf_counter() - start
    final = samples[-1] if samples else ReboundMegnoSample(0.0, math.nan, math.nan, math.nan)
    classification = classify(
        final.megno,
        final.finite_time_lyapunov_estimate,
        final.time_years,
        max_radius_seen,
        min_sep_seen,
        args.model_scope,
    )
    warnings: list[str] = []
    if args.model_scope in TOY_SCOPES and classification not in {"chaotic_candidate", "unstable_or_escape"}:
        warnings.append("positive-control toy did not trigger a chaotic/unstable classification")
    if args.model_scope in SOLAR_SCOPES and classification != "regular_likely":
        warnings.append("regular Solar/two-body control did not classify as regular_likely")

    plot_megno_growth(samples, plot_path)
    summary = {
        "diagnostic": "REBOUND-native MEGNO positive-control validation",
        "model_scope": args.model_scope,
        "toy_system": args.model_scope in TOY_SCOPES,
        "unit_note": unit_note,
        "integrator": args.integrator,
        "duration_years_requested": args.duration_years,
        "duration_years_actual": final.time_years,
        "step_days_or_toy_dt": sim.dt,
        "megno_seed": args.megno_seed,
        "final_megno": final.megno,
        "final_lcn": final.finite_time_lyapunov_estimate,
        "classification": classification,
        "runtime_seconds": runtime,
        "max_radius": max_radius_seen,
        "min_pairwise_separation": min_sep_seen,
        "warnings": warnings,
        "outputs": {"csv": str(csv_path), "summary": str(summary_path), "plot": str(plot_path)},
        "caveat": "Toy positive controls validate the diagnostic plumbing; they are not Solar System models.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"classification: {classification}")
    print(f"final_megno: {final.megno:.8g}")
    print(f"final_lcn: {final.finite_time_lyapunov_estimate:.8g}")
    print(f"runtime_seconds: {runtime:.3f}")
    print(f"wrote csv: {csv_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote plot: {plot_path}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
