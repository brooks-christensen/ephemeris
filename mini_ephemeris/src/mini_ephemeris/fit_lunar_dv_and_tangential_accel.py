from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, minimize
from skyfield.api import load

from .lunar_calibration import LunarCalibration, save_lunar_calibration_profile

from .advanced_integrators import (
    acceleration_newtonian,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_gr_sun_earth_j2,
    integrate_dop853_with_accel,
    make_acceleration_with_earth_moon_tangential_term,
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


def choose_base_acceleration(args, bodies: tuple[str, ...]):
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


def quadratic_diagnostics(years: np.ndarray, lon_err_arcsec: np.ndarray) -> tuple[float, float]:
    coeff = np.polyfit(years, lon_err_arcsec, deg=2)
    quad = float(coeff[0])
    trend = np.polyval(coeff, years)
    detrended = lon_err_arcsec - trend
    return quad, rms(detrended)


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


def evaluate_params(
    *,
    dv_t_mm_s: float,
    a_t_1e_15_m_s2: float,
    base_state,
    bodies: tuple[str, ...],
    earth_index: int,
    moon_index: int,
    times,
    offsets_s: np.ndarray,
    years_since_start: np.ndarray,
    jpl_moon: dict[str, np.ndarray],
    base_accel_func,
    base_accel_kwargs: dict,
    args,
) -> dict[str, float]:
    state0 = apply_lunar_tangential_velocity_correction(
        base_state,
        earth_index=earth_index,
        moon_index=moon_index,
        dv_t_m_s=dv_t_mm_s * 1.0e-3,
        preserve_emb_momentum=not args.no_preserve_emb_momentum,
    )

    accel_func = base_accel_func
    accel_kwargs = dict(base_accel_kwargs)

    if a_t_1e_15_m_s2 != 0.0:
        accel_func = make_acceleration_with_earth_moon_tangential_term(
            base_accel_func,
            earth_index=earth_index,
            moon_index=moon_index,
            a_t_m_s2=a_t_1e_15_m_s2 * 1.0e-15,
            base_accel_kwargs=base_accel_kwargs,
        )
        accel_kwargs = {}

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

    linear_slope, linear_detrended_rms = linear_diagnostics(
        years_since_start,
        lon_err_arcsec,
    )
    quadratic_coeff, quadratic_detrended_rms = quadratic_diagnostics(
        years_since_start,
        lon_err_arcsec,
    )

    return {
        "dv_t_mm_s": float(dv_t_mm_s),
        "a_t_1e_15_m_s2": float(a_t_1e_15_m_s2),
        "lon_rms_arcsec": rms(lon_err_arcsec),
        "lon_mean_arcsec": float(np.mean(lon_err_arcsec)),
        "lon_max_arcsec": float(np.max(lon_err_arcsec)),
        "lon_min_arcsec": float(np.min(lon_err_arcsec)),
        "lon_peak_abs_arcsec": float(np.max(np.abs(lon_err_arcsec))),
        "linear_slope_arcsec_per_year": linear_slope,
        "linear_detrended_rms_arcsec": linear_detrended_rms,
        "quadratic_coeff_arcsec_per_year2": quadratic_coeff,
        "quadratic_detrended_rms_arcsec": quadratic_detrended_rms,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit lunar dv_t plus empirical tangential acceleration."
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

    parser.add_argument("--initial-dv-mm-s", type=float, default=0.042417969864)
    parser.add_argument("--initial-at-1e-15", type=float, default=0.0)

    parser.add_argument("--dv-min-mm-s", type=float, default=0.038)
    parser.add_argument("--dv-max-mm-s", type=float, default=0.046)
    parser.add_argument("--at-min-1e-15", type=float, default=-20.0)
    parser.add_argument("--at-max-1e-15", type=float, default=20.0)

    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Run a rectangular dv_t/a_t grid scan instead of Powell optimization.",
    )
    parser.add_argument("--grid-dv-count", type=int, default=9)
    parser.add_argument("--grid-at-count", type=int, default=9)

    parser.add_argument(
        "--objective",
        choices=["rms", "peak", "slope_abs", "rms_plus_peak"],
        default="peak",
    )
    parser.add_argument("--opt-maxiter", type=int, default=30)
    parser.add_argument("--opt-xtol", type=float, default=1e-4)
    parser.add_argument("--opt-ftol", type=float, default=1e-4)

    parser.add_argument(
        "--no-preserve-emb-momentum",
        action="store_true",
        help="Apply initial dv_t to Moon only instead of preserving Earth-Moon barycenter momentum.",
    )

    parser.add_argument(
        "--save-calibration-file",
        default=None,
        help="Optional JSON file to update with the best-fit lunar calibration profile.",
    )
    parser.add_argument(
        "--save-calibration-name",
        default=None,
        help="Profile name to save when --save-calibration-file is used.",
    )
    parser.add_argument(
        "--save-calibration-description",
        default="",
        help="Optional description for the saved lunar calibration profile.",
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

    base_accel_func, base_accel_kwargs = choose_base_acceleration(args, bodies)

    print("[Lunar dv_t + tangential acceleration fit]")
    print(f"  start_date     : {args.start_date}")
    print(f"  end_date       : {args.end_date}")
    print(f"  sample_days    : {len(dates_dt)}")
    print(f"  gr_model       : {args.gr_model}")
    print(f"  earth_j2       : {args.earth_j2}")
    print(f"  objective      : {args.objective}")
    print(f"  initial_dv     : {args.initial_dv_mm_s} mm/s")
    print(f"  initial_at     : {args.initial_at_1e_15} x 1e-15 m/s^2")
    print(f"  dv bounds      : [{args.dv_min_mm_s}, {args.dv_max_mm_s}] mm/s")
    print(f"  at bounds      : [{args.at_min_1e_15}, {args.at_max_1e_15}] x 1e-15 m/s^2")
    print()

    print("Precomputing JPL Moon geometric reference...")
    jpl_moon = jpl_geometric_and_apparent_ecliptic(
        kernel_path=args.kernel_path,
        times=times,
        bodies=["moon"],
    )["moon"]

    rows: list[dict[str, float]] = []
    cache: dict[tuple[float, float], dict[str, float]] = {}

    def eval_cached(x: np.ndarray) -> dict[str, float]:
        dv_t_mm_s = float(x[0])
        a_t_1e_15 = float(x[1])

        key = (round(dv_t_mm_s, 12), round(a_t_1e_15, 12))
        if key not in cache:
            print(
                f"Trial dv_t={dv_t_mm_s:.12f} mm/s, "
                f"a_t={a_t_1e_15:.12f} x 1e-15 m/s^2"
            )
            row = evaluate_params(
                dv_t_mm_s=dv_t_mm_s,
                a_t_1e_15_m_s2=a_t_1e_15,
                base_state=base_state,
                bodies=bodies,
                earth_index=earth_index,
                moon_index=moon_index,
                times=times,
                offsets_s=offsets_s,
                years_since_start=years_since_start,
                jpl_moon=jpl_moon,
                base_accel_func=base_accel_func,
                base_accel_kwargs=base_accel_kwargs,
                args=args,
            )
            cache[key] = row
            rows.append(row)

            print(
                f"  objective={objective_from_row(row, args.objective):.9f}, "
                f"rms={row['lon_rms_arcsec']:.6f}, "
                f"peak={row['lon_peak_abs_arcsec']:.6f}, "
                f"mean={row['lon_mean_arcsec']:.6f}, "
                f"max={row['lon_max_arcsec']:.6f}, "
                f"min={row['lon_min_arcsec']:.6f}, "
                f"slope={row['linear_slope_arcsec_per_year']:.9f}, "
                f"quad={row['quadratic_coeff_arcsec_per_year2']:.12f}, "
                f"lat_rms={row['lat_rms_arcsec']:.6f}, "
                f"dist_rms={row['dist_rms_km']:.6f}"
            )

        return cache[key]

    def scalar_objective(x: np.ndarray) -> float:
        row = eval_cached(np.asarray(x, dtype=float))
        return objective_from_row(row, args.objective)

    # x0 = np.array([args.initial_dv_mm_s, args.initial_at_1e_15], dtype=float)
    # bounds = Bounds(
    #     [args.dv_min_mm_s, args.at_min_1e_15],
    #     [args.dv_max_mm_s, args.at_max_1e_15],
    # )

    # result = minimize(
    #     scalar_objective,
    #     x0=x0,
    #     method="Powell",
    #     bounds=bounds,
    #     options={
    #         "maxiter": args.opt_maxiter,
    #         "xtol": args.opt_xtol,
    #         "ftol": args.opt_ftol,
    #         "disp": True,
    #     },
    # )

    # best_row = eval_cached(result.x)
    if args.grid_only:
        dv_values = np.linspace(args.dv_min_mm_s, args.dv_max_mm_s, args.grid_dv_count)
        at_values = np.linspace(args.at_min_1e_15, args.at_max_1e_15, args.grid_at_count)

        print()
        print("[Grid scan]")
        print(f"  dv count       : {args.grid_dv_count}")
        print(f"  at count       : {args.grid_at_count}")
        print(f"  total trials   : {args.grid_dv_count * args.grid_at_count}")
        print()

        for dv_t in dv_values:
            for a_t in at_values:
                eval_cached(np.array([dv_t, a_t], dtype=float))

        best_row = min(rows, key=lambda r: objective_from_row(r, args.objective))

        class GridResult:
            success = True
            message = "Grid scan complete."

        result = GridResult()

    else:
        x0 = np.array([args.initial_dv_mm_s, args.initial_at_1e_15], dtype=float)
        bounds = Bounds(
            [args.dv_min_mm_s, args.at_min_1e_15],
            [args.dv_max_mm_s, args.at_max_1e_15],
        )

        result = minimize(
            scalar_objective,
            x0=x0,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": args.opt_maxiter,
                "xtol": args.opt_xtol,
                "ftol": args.opt_ftol,
                "disp": True,
            },
        )

        best_row = eval_cached(result.x)

    rows_sorted = sorted(rows, key=lambda r: (r["dv_t_mm_s"], r["a_t_1e_15_m_s2"]))
    write_csv(rows_sorted, args.output)

    print()
    print("Optimizer result")
    print(f"  success                       : {result.success}")
    print(f"  message                       : {result.message}")
    print(f"  objective                     : {args.objective}")
    print(f"  dv_t_mm_s                     : {best_row['dv_t_mm_s']:.12f}")
    print(f"  a_t_1e_15_m_s2                : {best_row['a_t_1e_15_m_s2']:.12f}")
    print(f"  objective_value               : {objective_from_row(best_row, args.objective):.9f}")
    print(f"  lon_rms_arcsec                : {best_row['lon_rms_arcsec']:.9f}")
    print(f"  lon_peak_abs_arcsec           : {best_row['lon_peak_abs_arcsec']:.9f}")
    print(f"  lon_mean_arcsec               : {best_row['lon_mean_arcsec']:.9f}")
    print(f"  lon_max_arcsec                : {best_row['lon_max_arcsec']:.9f}")
    print(f"  lon_min_arcsec                : {best_row['lon_min_arcsec']:.9f}")
    print(f"  linear_slope_arcsec_per_year  : {best_row['linear_slope_arcsec_per_year']:.9f}")
    print(f"  linear_detrended_rms_arcsec   : {best_row['linear_detrended_rms_arcsec']:.9f}")
    print(f"  quadratic_coeff_arcsec_yr2    : {best_row['quadratic_coeff_arcsec_per_year2']:.12f}")
    print(f"  quadratic_detrended_rms       : {best_row['quadratic_detrended_rms_arcsec']:.9f}")
    print(f"  lat_rms_arcsec                : {best_row['lat_rms_arcsec']:.9f}")
    print(f"  dist_rms_km                   : {best_row['dist_rms_km']:.9f}")
    print(f"Wrote trial CSV: {args.output}")

    if args.save_calibration_file is not None:
        if args.save_calibration_name is None:
            raise ValueError(
                "--save-calibration-name is required when --save-calibration-file is used."
            )

        profile = LunarCalibration(
            name=args.save_calibration_name,
            moon_dv_t_mm_s=best_row["dv_t_mm_s"],
            moon_a_t_1e_15_m_s2=best_row["a_t_1e_15_m_s2"],
            description=args.save_calibration_description,
            fit_start_date=args.start_date,
            fit_end_date=args.end_date,
            validation_start_date=args.start_date,
            validation_end_date=args.end_date,
            objective=args.objective,
            model_notes=(
                "Newtonian N-body + optional Sun 1PN GR + optional Earth J2 + "
                "empirical lunar along-track acceleration."
            ),
            lon_rms_arcsec=best_row["lon_rms_arcsec"],
            lon_peak_abs_arcsec=best_row["lon_peak_abs_arcsec"],
            lat_rms_arcsec=best_row["lat_rms_arcsec"],
            dist_rms_km=best_row["dist_rms_km"],
        )

        save_lunar_calibration_profile(profile, args.save_calibration_file)
        print()
        print(f"Saved lunar calibration profile: {args.save_calibration_name}")
        print(f"Calibration file               : {args.save_calibration_file}")


if __name__ == "__main__":
    main()