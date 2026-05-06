from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
from skyfield.api import load
from scipy.optimize import minimize_scalar

from .advanced_integrators import (
    acceleration_newtonian,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_gr_sun_earth_j2,
    integrate_dop853_with_accel,
)
from .american_ephemeris import (
    circular_angle_diff_deg,
    ecliptic_lon_lat_from_icrf_vectors_m,
    jpl_geometric_and_apparent_ecliptic,
)
from .ephem import (
    EphemerisConfig,
    apply_lunar_tangential_velocity_correction,
    initial_state_solar_system_barycentric_time,
    solar_system_body_list_earth_moon,
)


DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def daily_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    n = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(n + 1)]


def make_tt_times_for_dates(ts, dates: list[dt.date]):
    years = np.array([d.year for d in dates], dtype=int)
    months = np.array([d.month for d in dates], dtype=int)
    days = np.array([d.day for d in dates], dtype=int)
    return ts.tt(years, months, days, 0, 0, 0)


def rms(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def choose_acceleration(args, bodies: tuple[str, ...]):
    sun_index = bodies.index("sun")
    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")

    if args.gr_model == "none":
        if args.earth_j2:
            return acceleration_newtonian_earth_j2, {
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        return acceleration_newtonian, {}

    if args.gr_model == "sun":
        if args.earth_j2:
            return acceleration_newtonian_gr_sun_earth_j2, {
                "sun_index": sun_index,
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        return acceleration_newtonian_gr_sun, {"sun_index": sun_index}

    raise ValueError(f"Unsupported gr_model: {args.gr_model!r}")


def linear_diagnostics(years: np.ndarray, lon_err_arcsec: np.ndarray) -> tuple[float, float]:
    coeff = np.polyfit(years, lon_err_arcsec, deg=1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    trend = slope * years + intercept
    detrended = lon_err_arcsec - trend
    return slope, rms(detrended)


def evaluate_dv(
    *,
    dv_t_mm_s: float,
    base_state,
    bodies: tuple[str, ...],
    earth_index: int,
    moon_index: int,
    times,
    offsets_s: np.ndarray,
    years_since_start: np.ndarray,
    jpl_moon: dict[str, np.ndarray],
    accel_func,
    accel_kwargs: dict,
    args,
) -> dict[str, float]:
    state0 = apply_lunar_tangential_velocity_correction(
        base_state,
        earth_index=earth_index,
        moon_index=moon_index,
        dv_t_m_s=dv_t_mm_s * 1.0e-3,
        preserve_emb_momentum=not args.no_preserve_emb_momentum,
    )

    t_end = float(offsets_s[-1])

    times_s, positions_m, _ = integrate_dop853_with_accel(
        state0=state0,
        t_span=(0.0, t_end),
        dt=DAY_S,
        accel_func=accel_func,
        accel_kwargs=accel_kwargs,
        record_every=1,
        rtol=args.rtol,
        atol=args.atol,
        chunk_duration=args.chunk_years * YEAR_S,
        show_progress=not args.no_progress_bar,
        max_step=args.max_step_days * DAY_S if args.max_step_days is not None else None,
    )

    sample_indices = np.rint(offsets_s / DAY_S).astype(int)
    max_alignment_error_s = float(np.max(np.abs(times_s[sample_indices] - offsets_s)))
    if max_alignment_error_s > 1e-5:
        raise RuntimeError(
            f"Model output is not aligned with requested TT midnights. "
            f"Max alignment error = {max_alignment_error_s} s"
        )

    model_positions = positions_m[sample_indices]
    model_geo_vec = model_positions[:, moon_index, :] - model_positions[:, earth_index, :]

    model_lon, model_lat, model_distance_km = ecliptic_lon_lat_from_icrf_vectors_m(
        model_geo_vec,
        times,
    )

    lon_err_arcsec = circular_angle_diff_deg(
        model_lon,
        jpl_moon["geom_lon_deg"],
    ) * 3600.0
    lat_err_arcsec = (model_lat - jpl_moon["geom_lat_deg"]) * 3600.0
    dist_err_km = model_distance_km - jpl_moon["geom_distance_km"]

    slope, detrended_rms = linear_diagnostics(years_since_start, lon_err_arcsec)

    return {
        "dv_t_mm_s": float(dv_t_mm_s),
        "lon_rms_arcsec": rms(lon_err_arcsec),
        "lon_mean_arcsec": float(np.mean(lon_err_arcsec)),
        "lon_peak_abs_arcsec": float(np.max(np.abs(lon_err_arcsec))),
        "linear_slope_arcsec_per_year": slope,
        "linear_detrended_rms_arcsec": detrended_rms,
        "lat_rms_arcsec": rms(lat_err_arcsec),
        "dist_rms_km": rms(dist_err_km),
    }


def write_csv(rows: list[dict[str, float]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def objective_from_row(row: dict[str, float], objective: str) -> float:
    if objective == "rms":
        return row["lon_rms_arcsec"]
    if objective == "peak":
        return row["lon_peak_abs_arcsec"]
    if objective == "slope_abs":
        return abs(row["linear_slope_arcsec_per_year"])
    if objective == "rms_plus_peak":
        return row["lon_rms_arcsec"] + row["lon_peak_abs_arcsec"]
    raise ValueError(f"Unsupported objective: {objective!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a tiny lunar initial tangential velocity correction."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2050-01-31")
    parser.add_argument("--output", required=True)

    parser.add_argument("--gr-model", default="sun", choices=["none", "sun"])
    parser.add_argument("--earth-j2", action="store_true")
    parser.add_argument("--include-pluto", action="store_true")

    parser.add_argument("--chunk-years", type=float, default=1.0)
    parser.add_argument("--max-step-days", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-15)
    parser.add_argument("--no-progress-bar", action="store_true")

    parser.add_argument("--dv-min-mm-s", type=float, default=-0.25)
    parser.add_argument("--dv-max-mm-s", type=float, default=0.02)
    parser.add_argument("--dv-count", type=int, default=8)

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="After the grid search, run a bounded 1-D optimization over dv_t.",
    )
    parser.add_argument(
        "--objective",
        choices=["rms", "peak", "slope_abs", "rms_plus_peak"],
        default="rms",
        help=(
            "Objective for --optimize. "
            "'rms' minimizes Moon longitude RMS; "
            "'peak' minimizes max absolute Moon longitude residual; "
            "'slope_abs' minimizes absolute secular slope; "
            "'rms_plus_peak' minimizes lon_rms + lon_peak_abs."
        ),
    )
    parser.add_argument(
        "--opt-xatol-mm-s",
        type=float,
        default=1e-5,
        help="Optimizer absolute tolerance in mm/s.",
    )
    parser.add_argument(
        "--opt-maxiter",
        type=int,
        default=20,
        help="Maximum optimizer iterations.",
    )

    parser.add_argument(
        "--no-preserve-emb-momentum",
        action="store_true",
        help="Apply correction to Moon only instead of preserving Earth-Moon barycenter momentum.",
    )

    args = parser.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    dates_dt = daily_dates(start, end)

    ts = load.timescale()
    times = make_tt_times_for_dates(ts, dates_dt)
    t0 = ts.tt(2000, 1, 1, 0, 0, 0)

    offsets_s = (times.tt - t0.tt) * DAY_S
    if np.any(offsets_s < -1e-9):
        raise ValueError("This script assumes requested dates are >= 2000-01-01.")

    years_since_start = (times.tt - times.tt[0]) / 365.25

    bodies = tuple(solar_system_body_list_earth_moon())
    if args.include_pluto and "pluto barycenter" not in bodies:
        bodies = bodies + ("pluto barycenter",)

    config = EphemerisConfig(kernel_path=args.kernel_path)
    base_state = initial_state_solar_system_barycentric_time(
        t0,
        bodies=bodies,
        config=config,
        verbose=True,
    )

    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")
    accel_func, accel_kwargs = choose_acceleration(args, bodies)

    print("[Lunar Initial Tangential Velocity Fit]")
    print(f"  start_date     : {args.start_date}")
    print(f"  end_date       : {args.end_date}")
    print(f"  sample_days    : {len(dates_dt)}")
    print(f"  gr_model       : {args.gr_model}")
    print(f"  earth_j2       : {args.earth_j2}")
    print(f"  preserve_emb   : {not args.no_preserve_emb_momentum}")
    print(f"  dv_min_mm_s    : {args.dv_min_mm_s}")
    print(f"  dv_max_mm_s    : {args.dv_max_mm_s}")
    print(f"  dv_count       : {args.dv_count}")
    print()

    print("Precomputing JPL Moon geometric reference...")
    jpl_moon = jpl_geometric_and_apparent_ecliptic(
        kernel_path=args.kernel_path,
        times=times,
        bodies=["moon"],
    )["moon"]

    trial_values = np.linspace(args.dv_min_mm_s, args.dv_max_mm_s, args.dv_count)
    rows: list[dict[str, float]] = []

    for dv in trial_values:
        print(f"Trial dv_t = {dv:.9f} mm/s")
        row = evaluate_dv(
            dv_t_mm_s=float(dv),
            base_state=base_state,
            bodies=bodies,
            earth_index=earth_index,
            moon_index=moon_index,
            times=times,
            offsets_s=offsets_s,
            years_since_start=years_since_start,
            jpl_moon=jpl_moon,
            accel_func=accel_func,
            accel_kwargs=accel_kwargs,
            args=args,
        )
        rows.append(row)
        print(
            f"  lon_rms={row['lon_rms_arcsec']:.6f} arcsec, "
            f"slope={row['linear_slope_arcsec_per_year']:.6f} arcsec/yr, "
            f"peak={row['lon_peak_abs_arcsec']:.6f} arcsec, "
            f"lat_rms={row['lat_rms_arcsec']:.6f} arcsec, "
            f"dist_rms={row['dist_rms_km']:.6f} km"
        )

    # rows_sorted = sorted(rows, key=lambda r: r["dv_t_mm_s"])
    # write_csv(rows_sorted, args.output)

    # best = min(rows_sorted, key=lambda r: r["lon_rms_arcsec"])

    if args.optimize:
        print()
        print("[Bounded 1-D optimization]")
        print(f"  objective      : {args.objective}")
        print(f"  bounds mm/s    : [{args.dv_min_mm_s}, {args.dv_max_mm_s}]")
        print(f"  xatol mm/s     : {args.opt_xatol_mm_s}")
        print(f"  maxiter        : {args.opt_maxiter}")
        print()

        cache: dict[float, dict[str, float]] = {
            round(row["dv_t_mm_s"], 12): row for row in rows
        }

        def eval_cached(dv_t_mm_s: float) -> dict[str, float]:
            key = round(float(dv_t_mm_s), 12)
            if key not in cache:
                print(f"Optimizer trial dv_t = {dv_t_mm_s:.12f} mm/s")
                row = evaluate_dv(
                    dv_t_mm_s=float(dv_t_mm_s),
                    base_state=base_state,
                    bodies=bodies,
                    earth_index=earth_index,
                    moon_index=moon_index,
                    times=times,
                    offsets_s=offsets_s,
                    years_since_start=years_since_start,
                    jpl_moon=jpl_moon,
                    accel_func=accel_func,
                    accel_kwargs=accel_kwargs,
                    args=args,
                )
                cache[key] = row
                rows.append(row)
                print(
                    f"  objective={objective_from_row(row, args.objective):.9f}, "
                    f"lon_rms={row['lon_rms_arcsec']:.6f}, "
                    f"peak={row['lon_peak_abs_arcsec']:.6f}, "
                    f"slope={row['linear_slope_arcsec_per_year']:.9f}, "
                    f"lat_rms={row['lat_rms_arcsec']:.6f}, "
                    f"dist_rms={row['dist_rms_km']:.6f}"
                )
            return cache[key]

        def scalar_objective(dv_t_mm_s: float) -> float:
            row = eval_cached(float(dv_t_mm_s))
            return objective_from_row(row, args.objective)

        result = minimize_scalar(
            scalar_objective,
            bounds=(args.dv_min_mm_s, args.dv_max_mm_s),
            method="bounded",
            options={
                "xatol": args.opt_xatol_mm_s,
                "maxiter": args.opt_maxiter,
            },
        )

        opt_row = eval_cached(float(result.x))

        print()
        print("Optimizer result")
        print(f"  success                       : {result.success}")
        print(f"  message                       : {result.message}")
        print(f"  objective                     : {args.objective}")
        print(f"  dv_t_mm_s                     : {opt_row['dv_t_mm_s']:.12f}")
        print(f"  objective_value               : {objective_from_row(opt_row, args.objective):.9f}")
        print(f"  lon_rms_arcsec                : {opt_row['lon_rms_arcsec']:.9f}")
        print(f"  lon_peak_abs_arcsec           : {opt_row['lon_peak_abs_arcsec']:.9f}")
        print(f"  linear_slope_arcsec_per_year  : {opt_row['linear_slope_arcsec_per_year']:.9f}")
        print(f"  linear_detrended_rms_arcsec   : {opt_row['linear_detrended_rms_arcsec']:.9f}")
        print(f"  lat_rms_arcsec                : {opt_row['lat_rms_arcsec']:.9f}")
        print(f"  dist_rms_km                   : {opt_row['dist_rms_km']:.9f}")

    rows_sorted = sorted(rows, key=lambda r: r["dv_t_mm_s"])
    write_csv(rows_sorted, args.output)

    best = min(rows_sorted, key=lambda r: objective_from_row(r, args.objective))

    print()
    # print("Best trial by Moon longitude RMS")
    print(f"Best trial by objective: {args.objective}")
    # print(f"  dv_t_mm_s                    : {best['dv_t_mm_s']:.9f}")
    print(f"  dv_t_mm_s                    : {best['dv_t_mm_s']:.12f}")
    print(f"  objective_value              : {objective_from_row(best, args.objective):.9f}")
    print(f"  lon_rms_arcsec               : {best['lon_rms_arcsec']:.6f}")
    print(f"  lon_peak_abs_arcsec          : {best['lon_peak_abs_arcsec']:.6f}")
    print(f"  linear_slope_arcsec_per_year : {best['linear_slope_arcsec_per_year']:.6f}")
    print(f"  linear_detrended_rms_arcsec  : {best['linear_detrended_rms_arcsec']:.6f}")
    print(f"  lat_rms_arcsec               : {best['lat_rms_arcsec']:.6f}")
    print(f"  dist_rms_km                  : {best['dist_rms_km']:.6f}")
    print(f"Wrote trial CSV: {args.output}")


if __name__ == "__main__":
    main()