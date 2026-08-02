from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent import attach_gr_potential_tangent_force, verify_rebound_variation_api
from .long_term_stability_cli import (
    build_rebound_simulation,
    configure_rebound_simulationarchive,
    optional_import_module,
    parse_start_datetime,
    rebound_state_from_sim,
    sanitize_tag,
    stability_body_list,
)
from .nbody import G_SI
from .orbital_elements import ARCSEC_PER_RAD, DAY_S, JULIAN_YEAR_S, heliocentric_elements_for_state, seconds_to_years
from .stability_diagnostics import invariant_diagnostics_row, invariant_reference


FIELDS = [
    "wall_time_utc",
    "wall_time_monotonic_seconds",
    "recent_throughput_years_per_wall_second",
    "worker_pid",
    "time_years",
    "megno",
    "lcn_1_per_year",
    "energy_rel_drift",
    "angular_momentum_rel_drift",
    "mercury_a_au",
    "mercury_e",
    "mercury_varpi_deg_unwrapped",
    "callback_invocations",
    "real_gr_accel_norm_max",
    "real_gr_accel_norm_mean",
    "tangent_gr_accel_norm_max",
    "tangent_gr_accel_norm_mean",
]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def wrapped_delta(angle: float, previous: float) -> float:
    return (angle - previous + math.pi) % (2.0 * math.pi) - math.pi


def classify(final_lcn: float, final_megno: float, duration_years: float, model_scope: str) -> str:
    if model_scope.startswith("two_body"):
        if math.isfinite(final_lcn) and abs(final_lcn) < 1.0e-5 and math.isfinite(final_megno) and final_megno < 8.0:
            return "regular_likely"
        return "ambiguous"
    if math.isfinite(final_lcn) and final_lcn * duration_years > 1.0:
        return "chaotic_candidate"
    return "regular_likely" if math.isfinite(final_lcn) else "ambiguous"


def variation_metadata(sim) -> dict[str, Any]:
    n_real = int(getattr(sim, "N_real", 0))
    n_total = int(getattr(sim, "N", 0))
    variation_particles = max(0, n_total - n_real)
    ranges = []
    for start in range(n_real, n_total, n_real if n_real else 1):
        end = min(start + n_real, n_total)
        if start < end:
            ranges.append({"start_index": start, "end_index_exclusive": end, "count": end - start})
    return {
        "n_real": n_real,
        "n_total": n_total,
        "n_var": int(getattr(sim, "N_var", 0)),
        "n_var_config": int(getattr(sim, "N_var_config", 0)),
        "variation_particle_count": variation_particles,
        "variation_particle_ranges": ranges,
    }


def callback_stats(sim) -> dict[str, float | int | None]:
    stats = getattr(sim, "_mini_ephemeris_gr_potential_tangent_stats", {}) or {}
    real_count = int(stats.get("real_gr_accel_norm_count", 0) or 0)
    tangent_count = int(stats.get("tangent_gr_accel_norm_count", 0) or 0)
    return {
        "callback_invocations": int(stats.get("callback_invocations", 0) or 0),
        "real_gr_accel_norm_max": float(stats.get("real_gr_accel_norm_max", 0.0) or 0.0),
        "real_gr_accel_norm_mean": (
            float(stats.get("real_gr_accel_norm_sum", 0.0)) / real_count if real_count else None
        ),
        "tangent_gr_accel_norm_max": float(stats.get("tangent_gr_accel_norm_max", 0.0) or 0.0),
        "tangent_gr_accel_norm_mean": (
            float(stats.get("tangent_gr_accel_norm_sum", 0.0)) / tangent_count if tangent_count else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REBOUND WHFast tangent-aware gr_potential validation worker.")
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument("--model-scope", choices=["two_body_mercury", "inner", "full", "full_with_pluto"], default="two_body_mercury")
    parser.add_argument("--duration-years", type=float, default=1000.0)
    parser.add_argument("--step-days", type=float, default=1.0)
    parser.add_argument("--record-every-years", type=float, default=10.0)
    parser.add_argument("--megno-seed", type=int, default=12345)
    parser.add_argument("--gr-scale", type=float, default=1.0, help="Scale applied to the custom gr_potential acceleration and tangent Jacobian.")
    parser.add_argument("--no-central-response", action="store_true", help="Disable equal-and-opposite central-body response for diagnostic comparisons only.")
    parser.add_argument("--simulationarchive", default=None)
    parser.add_argument("--archive-interval-years", type=float, default=None)
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability/gr_tangent")
    parser.add_argument("--tag", default="gr_tangent")
    parser.add_argument("--status-every-record", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip if final summary already exists.")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration_years <= 0 or args.step_days <= 0 or args.record_every_years <= 0:
        parser.error("duration, step, and record cadence must be positive.")
    rebound = optional_import_module("rebound")
    if rebound is None:
        raise RuntimeError("REBOUND is not installed.")

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"gr_tangent_progress_{tag}.csv"
    summary_path = output_dir / f"gr_tangent_summary_{tag}.json"
    status_path = output_dir / f"gr_tangent_status_{tag}.json"
    if args.resume and summary_path.exists():
        print(f"[gr-tangent] RESUME=1 and summary exists; skipping {summary_path}")
        return

    behavior = verify_rebound_variation_api(rebound)
    if not behavior.readback_exact:
        raise RuntimeError("Installed REBOUND variation API did not exactly read back assigned Cartesian components.")

    bodies = stability_body_list(args.model_scope, include_pluto=args.model_scope == "full_with_pluto")
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    sim = build_rebound_simulation(
        rebound,
        state0,
        integrator="whfast",
        step_s=args.step_days * DAY_S,
        ias15_epsilon=1.0e-10,
    )
    sim.init_megno(seed=args.megno_seed)
    sim.lyapunov()
    attach_gr_potential_tangent_force(
        sim,
        coefficient_scale=args.gr_scale,
        include_central_response=not args.no_central_response,
    )
    archive_status = "disabled"
    if args.simulationarchive and args.archive_interval_years:
        archive_status = configure_rebound_simulationarchive(
            sim,
            Path(args.simulationarchive),
            interval_s=args.archive_interval_years * JULIAN_YEAR_S,
            delete_existing=True,
        )

    reference = invariant_reference(state0, G=G_SI)
    duration_s = args.duration_years * JULIAN_YEAR_S
    record_s = args.record_every_years * JULIAN_YEAR_S
    n_records = int(math.floor(duration_s / record_s))
    sun_index = bodies.index("sun")
    previous_varpi = None
    unwrapped_varpi = None
    varpi_times: list[float] = []
    varpi_values: list[float] = []
    lcn_history: list[tuple[float, float]] = []
    megno_history: list[tuple[float, float]] = []
    element_extrema: dict[str, dict[str, float]] = {}
    max_energy = 0.0
    max_angular = 0.0
    rows_written = 0
    start_wall = time.perf_counter()
    previous_progress: tuple[float, float] | None = None
    recent_rates: list[float] = []
    metadata = variation_metadata(sim)

    with progress_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record_index in range(n_records + 1):
            target_s = min(record_index * record_s, duration_s)
            sim.integrate(target_s, exact_finish_time=1)
            now_monotonic = time.perf_counter()
            current_years = seconds_to_years(target_s)
            recent_rate = math.nan
            if previous_progress is not None:
                previous_years, previous_wall = previous_progress
                dt_wall = now_monotonic - previous_wall
                dt_years = current_years - previous_years
                if dt_wall > 0.0 and dt_years > 0.0:
                    recent_rate = dt_years / dt_wall
                    recent_rates.append(recent_rate)
                    recent_rates = recent_rates[-8:]
            previous_progress = (current_years, now_monotonic)
            state = rebound_state_from_sim(sim, state0.masses)
            inv = invariant_diagnostics_row(target_s, state, reference)
            energy = float(inv["energy_rel_drift"])
            angular = float(inv["angular_momentum_rel_drift"])
            max_energy = max(max_energy, abs(energy))
            max_angular = max(max_angular, abs(angular))
            mercury_a = math.nan
            mercury_e = math.nan
            elements = heliocentric_elements_for_state(state, bodies, sun_index=sun_index)
            for element in elements:
                extrema = element_extrema.setdefault(
                    element.body_name,
                    {
                        "max_e": 0.0,
                        "max_i_deg": 0.0,
                        "min_a_au": math.inf,
                        "max_a_au": -math.inf,
                    },
                )
                a_au = element.semi_major_axis_m / 149_597_870_700.0
                extrema["max_e"] = max(extrema["max_e"], float(element.eccentricity))
                extrema["max_i_deg"] = max(extrema["max_i_deg"], abs(math.degrees(element.inclination_rad)))
                extrema["min_a_au"] = min(extrema["min_a_au"], a_au)
                extrema["max_a_au"] = max(extrema["max_a_au"], a_au)
                if element.body_name != "mercury barycenter":
                    continue
                mercury_a = a_au
                mercury_e = element.eccentricity
                varpi = element.longitude_perihelion_rad
                if previous_varpi is None:
                    previous_varpi = varpi
                    unwrapped_varpi = varpi
                else:
                    assert unwrapped_varpi is not None
                    unwrapped_varpi += wrapped_delta(varpi, previous_varpi)
                    previous_varpi = varpi
                varpi_times.append(seconds_to_years(target_s))
                varpi_values.append(float(unwrapped_varpi))
            try:
                megno = float(sim.megno())
            except Exception:
                megno = math.nan
            try:
                lcn = float(sim.lyapunov())
            except Exception:
                lcn = math.nan
            if math.isfinite(lcn):
                lcn_history.append((current_years, lcn))
            if math.isfinite(megno):
                megno_history.append((current_years, megno))
            stats_now = callback_stats(sim)
            row = {
                "wall_time_utc": dt.datetime.utcnow().isoformat() + "Z",
                "wall_time_monotonic_seconds": now_monotonic,
                "recent_throughput_years_per_wall_second": recent_rate if math.isfinite(recent_rate) else "",
                "worker_pid": os.getpid(),
                "time_years": current_years,
                "megno": megno if math.isfinite(megno) else "",
                "lcn_1_per_year": lcn if math.isfinite(lcn) else "",
                "energy_rel_drift": energy,
                "angular_momentum_rel_drift": angular,
                "mercury_a_au": mercury_a if math.isfinite(mercury_a) else "",
                "mercury_e": mercury_e if math.isfinite(mercury_e) else "",
                "mercury_varpi_deg_unwrapped": (
                    math.degrees(unwrapped_varpi) if unwrapped_varpi is not None else ""
                ),
                "callback_invocations": stats_now["callback_invocations"],
                "real_gr_accel_norm_max": stats_now["real_gr_accel_norm_max"],
                "real_gr_accel_norm_mean": stats_now["real_gr_accel_norm_mean"] if stats_now["real_gr_accel_norm_mean"] is not None else "",
                "tangent_gr_accel_norm_max": stats_now["tangent_gr_accel_norm_max"],
                "tangent_gr_accel_norm_mean": stats_now["tangent_gr_accel_norm_mean"] if stats_now["tangent_gr_accel_norm_mean"] is not None else "",
            }
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            rows_written += 1
            elapsed = now_monotonic - start_wall
            percent = 100.0 * target_s / duration_s if duration_s else 100.0
            rate = float(np.median(recent_rates)) if recent_rates else (current_years / elapsed if elapsed > 0 else math.nan)
            eta = (args.duration_years - seconds_to_years(target_s)) / rate if rate and rate > 0 else math.nan
            status = {
                "tag": tag,
                "current_time_years": current_years,
                "percent_complete": percent,
                "eta_seconds": eta if math.isfinite(eta) else None,
                "latest_megno": megno if math.isfinite(megno) else None,
                "latest_lcn_1_per_year": lcn if math.isfinite(lcn) else None,
                "worker_pid": os.getpid(),
                "variation_metadata": metadata,
                "callback_stats": stats_now,
            }
            if args.status_every_record or record_index == n_records:
                atomic_write_json(status_path, status)
            if record_index == 0 or args.status_every_record or record_index == n_records:
                print(
                    f"[gr-tangent] {tag}: t={current_years:.6g} yr "
                    f"{percent:.2f}% rate={rate:.6g} yr/s ETA={eta if math.isfinite(eta) else math.nan:.1f}s "
                    f"MEGNO={megno:.6g} LCN={lcn:.6e} callbacks={stats_now['callback_invocations']}",
                    flush=True,
                )

    runtime = time.perf_counter() - start_wall
    final_stats = callback_stats(sim)
    final_megno = float(row["megno"]) if row.get("megno") not in {"", None} else math.nan
    final_lcn = float(row["lcn_1_per_year"]) if row.get("lcn_1_per_year") not in {"", None} else math.nan
    if len(varpi_times) >= 2:
        slope = float(np.polyfit(varpi_times, varpi_values, 1)[0])
        perihelion = slope * ARCSEC_PER_RAD * 100.0
    else:
        perihelion = math.nan
    late_cut = 0.5 * args.duration_years
    early_lcn_abs = [abs(value) for time_years, value in lcn_history if time_years <= late_cut]
    late_lcn_abs = [abs(value) for time_years, value in lcn_history if time_years >= late_cut]
    late_megno_values = [value for time_years, value in megno_history if time_years >= late_cut]
    early_lcn_median = float(np.median(early_lcn_abs)) if early_lcn_abs else math.nan
    late_lcn_median = float(np.median(late_lcn_abs)) if late_lcn_abs else math.nan
    late_megno_median = float(np.median(late_megno_values)) if late_megno_values else math.nan
    lcn_trends_toward_zero = bool(
        math.isfinite(early_lcn_median)
        and math.isfinite(late_lcn_median)
        and (late_lcn_median <= early_lcn_median or late_lcn_median <= 1.0e-8)
    )
    stable_positive_lcn_plateau = bool(
        late_lcn_abs
        and np.median([value for time_years, value in lcn_history if time_years >= late_cut]) > 1.0e-5
    )
    summary = {
        "mode": "REBOUND tangent-aware gr_potential",
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "tag": tag,
        "rebound_version": behavior.rebound_version,
        "variation_api": behavior.__dict__,
        "configuration": {
            "kernel_path": args.kernel_path,
            "start_date": args.start_date.isoformat(),
            "model_scope": args.model_scope,
            "duration_years": args.duration_years,
            "step_days": args.step_days,
            "record_every_years": args.record_every_years,
            "megno_seed": args.megno_seed,
            "gr_scale": args.gr_scale,
            "include_central_response": not args.no_central_response,
            "body_names": bodies,
            "simulationarchive": args.simulationarchive,
            "archive_interval_years": args.archive_interval_years,
            "archive_status": archive_status,
        },
        "production_metadata": {
            **metadata,
            "worker_pid": os.getpid(),
            "callback_stats": final_stats,
        },
        "diagnostics": {
            "runtime_seconds": runtime,
            "rows_written": rows_written,
            "actual_time_years": args.duration_years,
            "final_megno": final_megno if math.isfinite(final_megno) else None,
            "final_lcn_1_per_year": final_lcn if math.isfinite(final_lcn) else None,
            "classification_hint": classify(final_lcn, final_megno, args.duration_years, args.model_scope),
            "late_window_megno_median": late_megno_median if math.isfinite(late_megno_median) else None,
            "early_abs_lcn_median": early_lcn_median if math.isfinite(early_lcn_median) else None,
            "late_abs_lcn_median": late_lcn_median if math.isfinite(late_lcn_median) else None,
            "lcn_trends_toward_zero": lcn_trends_toward_zero,
            "stable_positive_lcn_plateau": stable_positive_lcn_plateau,
            "max_energy_rel_drift": max_energy,
            "max_angular_momentum_rel_drift": max_angular,
            "mercury_perihelion_drift_arcsec_per_century": perihelion if math.isfinite(perihelion) else None,
            "orbital_element_extrema": element_extrema,
        },
        "caveats": [
            "Finite-time tangent/MEGNO diagnostic; not an asymptotic Lyapunov proof.",
            "Custom gr_potential tangent path is separate from ordinary REBOUNDx gr_potential and must pass validation before production use.",
        ],
        "outputs": {
            "progress_csv": str(progress_path),
            "status_json": str(status_path),
            "summary_json": str(summary_path),
        },
    }
    atomic_write_json(summary_path, summary)
    print(f"[gr-tangent] wrote {progress_path}")
    print(f"[gr-tangent] wrote {summary_path}")
    print(f"[gr-tangent] final_lcn={final_lcn:.6e} final_megno={final_megno:.6e}")


if __name__ == "__main__":
    main(sys.argv[1:])
