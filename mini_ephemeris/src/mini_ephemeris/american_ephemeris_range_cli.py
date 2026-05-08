from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skyfield.api import load

from .lunar_calibration import resolve_lunar_correction_values

from .american_ephemeris import (
    build_model_vs_jpl_ephemeris_rows,
    write_rows_csv,
)
from .ephem import (
    EphemerisConfig,
    apply_lunar_velocity_correction,
    initial_state_solar_system_barycentric_time,
    solar_system_body_list_earth_moon,
)
from .advanced_integrators import (
    integrate_dop853_with_accel,
    acceleration_newtonian,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun_earth_j2,
    make_acceleration_with_earth_moon_tangential_term,
)


DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S


def rms(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


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


# def plot_moon_longitude_residual(rows: list[dict], output_path: str) -> None:
#     moon_rows = [r for r in rows if r["body"] == "Moon"]

#     dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
#     lon_err = np.array([float(r["lon_error_arcsec"]) for r in moon_rows])

#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(12, 5))
#     plt.plot(dates, lon_err, linewidth=0.8)
#     plt.axhline(0.0, linestyle="--", linewidth=0.8)
#     plt.axhline(60.0, linestyle=":", linewidth=0.8)
#     plt.axhline(-60.0, linestyle=":", linewidth=0.8)
#     plt.title("Moon longitude residual vs JPL / American Ephemeris convention")
#     plt.xlabel("Date")
#     plt.ylabel("Longitude error [arcsec]")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()


# def plot_moon_longitude_residual(
#     rows: list[dict],
#     output_path: str,
#     y_limit_arcsec: float | None = None,
# ) -> None:
#     moon_rows = [r for r in rows if r["body"] == "Moon"]

#     dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
#     lon_err = np.array([float(r["lon_error_arcsec"]) for r in moon_rows])

#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     peak_abs = float(np.max(np.abs(lon_err)))

#     # If no explicit zoom is requested, choose a sensible automatic scale.
#     # Keep at least +/-1 arcsec so the plot is not too cramped.
#     if y_limit_arcsec is None:
#         y_limit_arcsec = max(1.0, 1.15 * peak_abs)

#     plt.figure(figsize=(12, 5))
#     plt.plot(dates, lon_err, linewidth=0.8)
#     plt.axhline(0.0, linestyle="--", linewidth=0.8)
#     plt.axhline(y_limit_arcsec, linestyle=":", linewidth=0.8)
#     plt.axhline(-y_limit_arcsec, linestyle=":", linewidth=0.8)
#     plt.ylim(-y_limit_arcsec, y_limit_arcsec)
#     plt.title("Moon longitude residual vs JPL / American Ephemeris convention")
#     plt.xlabel("Date")
#     plt.ylabel("Longitude error [arcsec]")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()


def plot_moon_longitude_residual(
    rows: list[dict],
    output_path: str,
    y_limit_arcsec: float | None = None,
) -> None:
    moon_rows = [r for r in rows if r["body"] == "Moon"]

    dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
    lon_err = np.array([float(r["lon_error_arcsec"]) for r in moon_rows])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    peak_abs = float(np.max(np.abs(lon_err)))
    if y_limit_arcsec is None:
        y_limit_arcsec = max(1.0, 1.15 * peak_abs)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, lon_err, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(y_limit_arcsec, linestyle=":", linewidth=0.8)
    plt.axhline(-y_limit_arcsec, linestyle=":", linewidth=0.8)
    plt.ylim(-y_limit_arcsec, y_limit_arcsec)
    plt.title("Moon longitude residual vs JPL / American Ephemeris convention")
    plt.xlabel("Date")
    plt.ylabel("Longitude error [arcsec]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# def plot_moon_latitude_residual(rows: list[dict], output_path: str) -> None:
#     moon_rows = [r for r in rows if r["body"] == "Moon"]

#     dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
#     lat_err = np.array([float(r["lat_error_arcsec"]) for r in moon_rows])

#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(12, 5))
#     plt.plot(dates, lat_err, linewidth=0.8)
#     plt.axhline(0.0, linestyle="--", linewidth=0.8)
#     plt.title("Moon latitude residual vs JPL")
#     plt.xlabel("Date")
#     plt.ylabel("Latitude error [arcsec]")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()


# def plot_moon_latitude_residual(
#     rows: list[dict],
#     output_path: str,
#     y_limit_arcsec: float | None = None,
# ) -> None:
#     moon_rows = [r for r in rows if r["body"] == "Moon"]

#     dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
#     lat_err = np.array([float(r["lat_error_arcsec"]) for r in moon_rows])

#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     peak_abs = float(np.max(np.abs(lat_err)))

#     if y_limit_arcsec is None:
#         y_limit_arcsec = max(1.0, 1.15 * peak_abs)

#     plt.figure(figsize=(12, 5))
#     plt.plot(dates, lat_err, linewidth=0.8)
#     plt.axhline(0.0, linestyle="--", linewidth=0.8)
#     plt.axhline(y_limit_arcsec, linestyle=":", linewidth=0.8)
#     plt.axhline(-y_limit_arcsec, linestyle=":", linewidth=0.8)
#     plt.ylim(-y_limit_arcsec, y_limit_arcsec)
#     plt.title("Moon latitude residual vs JPL")
#     plt.xlabel("Date")
#     plt.ylabel("Latitude error [arcsec]")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close()


def plot_moon_latitude_residual(
    rows: list[dict],
    output_path: str,
    y_limit_arcsec: float | None = None,
) -> None:
    moon_rows = [r for r in rows if r["body"] == "Moon"]

    dates = [dt.date.fromisoformat(r["date"]) for r in moon_rows]
    lat_err = np.array([float(r["lat_error_arcsec"]) for r in moon_rows])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    peak_abs = float(np.max(np.abs(lat_err)))
    if y_limit_arcsec is None:
        y_limit_arcsec = max(1.0, 1.15 * peak_abs)

    plt.figure(figsize=(12, 5))
    plt.plot(dates, lat_err, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(y_limit_arcsec, linestyle=":", linewidth=0.8)
    plt.axhline(-y_limit_arcsec, linestyle=":", linewidth=0.8)
    plt.ylim(-y_limit_arcsec, y_limit_arcsec)
    plt.title("Moon latitude residual vs JPL")
    plt.xlabel("Date")
    plt.ylabel("Latitude error [arcsec]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full date-range model-vs-JPL American Ephemeris comparison."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2050-01-01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--moon-lon-plot", required=True)
    parser.add_argument("--moon-lat-plot", default=None)

    parser.add_argument("--gr-model", default="sun", choices=["none", "sun"])
    parser.add_argument("--earth-j2", action="store_true")
    parser.add_argument(
        "--moon-dv-r-mm-s",
        type=float,
        default=None,
        help=(
            "Relative geocentric lunar radial velocity correction at the "
            "2000-01-01 TT epoch, in mm/s. Positive points from Earth toward "
            "Moon. The correction preserves Earth-Moon barycenter momentum "
            "unless --no-preserve-emb-momentum is used."
        ),
    )
    parser.add_argument(
        "--moon-dv-t-mm-s",
        type=float,
        default=None,
        help=(
            "Relative geocentric lunar tangential velocity correction at the "
            "2000-01-01 TT epoch, in mm/s. Positive is prograde; negative "
            "slows the Moon along-track. The correction preserves Earth-Moon "
            "barycenter momentum unless --no-preserve-emb-momentum is used."
        ),
    )
    parser.add_argument(
        "--moon-dv-h-mm-s",
        type=float,
        default=None,
        help=(
            "Relative geocentric lunar out-of-plane velocity correction at the "
            "2000-01-01 TT epoch, in mm/s. Positive points along the Moon's "
            "instantaneous geocentric angular momentum."
        ),
    )
    parser.add_argument(
        "--moon-a-t-1e-15-m-s2",
        type=float,
        default=None,
        help=(
            "Empirical relative geocentric lunar tangential acceleration in units "
            "of 1e-15 m/s^2. Positive is prograde. This is split between Earth "
            "and Moon to preserve Earth-Moon barycenter momentum."
        ),
    )
    parser.add_argument(
        "--no-preserve-emb-momentum",
        action="store_true",
        help="Apply --moon-dv-t-mm-s to the Moon only instead of preserving EMB momentum.",
    )
    parser.add_argument("--include-pluto", action="store_true")

    parser.add_argument("--chunk-years", type=float, default=1.0)
    parser.add_argument("--max-step-days", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-15)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--moon-lon-ylim-arcsec",
        type=float,
        default=None,
        help=(
            "Optional absolute y-limit for Moon longitude residual plot in arcsec. "
            "If omitted, use automatic scaling based on the data."
        ),
    )
    parser.add_argument(
        "--moon-lat-ylim-arcsec",
        type=float,
        default=None,
        help=(
            "Optional absolute y-limit for Moon latitude residual plot in arcsec. "
            "If omitted, use automatic scaling based on the data."
        ),
    )
    parser.add_argument(
        "--lunar-calibration-profile",
        default=None,
        help="Named lunar calibration profile to load from --lunar-calibration-file.",
    )
    parser.add_argument(
        "--lunar-calibration-file",
        default=None,
        help="JSON file containing named lunar calibration profiles.",
    )
    # parser.add_argument(
    #     "--moon-lon-ylim-arcsec",
    #     type=float,
    #     default=None,
    #     help=(
    #         "Optional absolute y-limit for Moon longitude residual plot in arcsec. "
    #         "If omitted, use automatic scaling based on the data."
    #     ),
    # )
    # parser.add_argument(
    #     "--moon-lat-ylim-arcsec",
    #     type=float,
    #     default=None,
    #     help=(
    #         "Optional absolute y-limit for Moon latitude residual plot in arcsec. "
    #         "If omitted, use automatic scaling based on the data."
    #     ),
    # )

    args = parser.parse_args()

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    dates_dt = daily_dates(start, end)
    date_strings = [d.isoformat() for d in dates_dt]

    ts = load.timescale()
    t_range = make_tt_times_for_dates(ts, dates_dt)

    # Fixed integration epoch: American Ephemeris comparison starts from J2000-era TT.
    t0 = ts.tt(2000, 1, 1, 0, 0, 0)

    offsets_s = (t_range.tt - t0.tt) * DAY_S
    if np.any(offsets_s < -1e-9):
        raise ValueError("This script assumes requested dates are >= 2000-01-01.")

    t_end = float(offsets_s[-1])
    dt_s = DAY_S

    bodies = tuple(solar_system_body_list_earth_moon())
    if args.include_pluto and "pluto barycenter" not in bodies:
        bodies = bodies + ("pluto barycenter",)

    config = EphemerisConfig(kernel_path=args.kernel_path)

    state0 = initial_state_solar_system_barycentric_time(
        t0,
        bodies=bodies,
        config=config,
        verbose=True,
    )

    sun_index = bodies.index("sun")
    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")
    (
        moon_dv_r_mm_s,
        moon_dv_t_mm_s,
        moon_dv_h_mm_s,
        moon_a_t_1e_15_m_s2,
        lunar_profile,
    ) = resolve_lunar_correction_values(
        profile_name=args.lunar_calibration_profile,
        calibration_file=args.lunar_calibration_file,
        moon_dv_r_mm_s=args.moon_dv_r_mm_s,
        moon_dv_t_mm_s=args.moon_dv_t_mm_s,
        moon_dv_h_mm_s=args.moon_dv_h_mm_s,
        moon_a_t_1e_15_m_s2=args.moon_a_t_1e_15_m_s2,
    )
    if (
        moon_dv_r_mm_s != 0.0
        or moon_dv_t_mm_s != 0.0
        or moon_dv_h_mm_s != 0.0
    ):
        state0 = apply_lunar_velocity_correction(
            state0,
            earth_index=earth_index,
            moon_index=moon_index,
            dv_r_m_s=moon_dv_r_mm_s * 1.0e-3,
            dv_t_m_s=moon_dv_t_mm_s * 1.0e-3,
            dv_h_m_s=moon_dv_h_mm_s * 1.0e-3,
            preserve_emb_momentum=not args.no_preserve_emb_momentum,
        )

    if args.gr_model == "none":
        if args.earth_j2:
            accel_func = acceleration_newtonian_earth_j2
            accel_kwargs = {
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        else:
            accel_func = acceleration_newtonian
            accel_kwargs = {}

    elif args.gr_model == "sun":
        if args.earth_j2:
            accel_func = acceleration_newtonian_gr_sun_earth_j2
            accel_kwargs = {
                "sun_index": sun_index,
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        else:
            accel_func = acceleration_newtonian_gr_sun
            accel_kwargs = {
                "sun_index": sun_index,
            }
    else:
        raise ValueError(f"Unsupported gr_model: {args.gr_model!r}")
    
    if moon_a_t_1e_15_m_s2 != 0.0:
        accel_func = make_acceleration_with_earth_moon_tangential_term(
            accel_func,
            earth_index=earth_index,
            moon_index=moon_index,
            a_t_m_s2=moon_a_t_1e_15_m_s2 * 1.0e-15,
            base_accel_kwargs=accel_kwargs,
        )
        accel_kwargs = {}

    if lunar_profile is not None:
        print(f"  lunar_profile  : {lunar_profile.name}")
        if lunar_profile.description:
            print(f"  profile_desc   : {lunar_profile.description}")

    print(f"  moon_dv_r_mm_s : {moon_dv_r_mm_s}")
    print(f"  moon_dv_t_mm_s : {moon_dv_t_mm_s}")
    print(f"  moon_dv_h_mm_s : {moon_dv_h_mm_s}")
    print(f"  moon_a_t_1e-15 : {moon_a_t_1e_15_m_s2}")

    print("[American Ephemeris Range Compare]")
    print(f"  start_date     : {args.start_date}")
    print(f"  end_date       : {args.end_date}")
    print(f"  sample_days    : {len(dates_dt)}")
    print(f"  t_end_days     : {t_end / DAY_S:.1f}")
    print(f"  bodies         : {bodies}")
    print(f"  gr_model       : {args.gr_model}")
    print(f"  earth_j2       : {args.earth_j2}")
    # print(f"  moon_dv_t_mm_s : {moon_dv_t_mm_s}")
    # print(f"  moon_a_t_1e-15 : {moon_a_t_1e_15_m_s2}")
    print(f"  preserve_emb   : {not args.no_preserve_emb_momentum}")
    print(f"  max_step_days  : {args.max_step_days}")
    print(f"  chunk_years    : {args.chunk_years}")
    print(f"  rtol           : {args.rtol}")
    print(f"  atol           : {args.atol}")

    times_s, positions_m, velocities_m = integrate_dop853_with_accel(
        state0=state0,
        t_span=(0.0, t_end),
        dt=dt_s,
        accel_func=accel_func,
        accel_kwargs=accel_kwargs,
        record_every=1,
        rtol=args.rtol,
        atol=args.atol,
        chunk_duration=args.chunk_years * YEAR_S,
        show_progress=not args.no_progress_bar,
        max_step=args.max_step_days * DAY_S if args.max_step_days is not None else None,
    )

    sample_indices = np.rint(offsets_s / dt_s).astype(int)
    max_alignment_error_s = float(np.max(np.abs(times_s[sample_indices] - offsets_s)))

    if max_alignment_error_s > 1e-5:
        raise RuntimeError(
            f"Model output is not aligned with requested TT midnights. "
            f"Max alignment error = {max_alignment_error_s} s"
        )

    model_positions = positions_m[sample_indices]

    rows = build_model_vs_jpl_ephemeris_rows(
        dates=date_strings,
        times=t_range,
        model_positions_m=model_positions,
        bodies=bodies,
        kernel_path=args.kernel_path,
    )

    write_rows_csv(rows, args.output)
    # plot_moon_longitude_residual(rows, args.moon_lon_plot)

    # if args.moon_lat_plot:
    #     plot_moon_latitude_residual(rows, args.moon_lat_plot)
    plot_moon_longitude_residual(
        rows,
        args.moon_lon_plot,
        y_limit_arcsec=args.moon_lon_ylim_arcsec,
    )

    if args.moon_lat_plot is not None:
        plot_moon_latitude_residual(
            rows,
            args.moon_lat_plot,
            y_limit_arcsec=args.moon_lat_ylim_arcsec,
        )

    print(f"Wrote CSV: {args.output}")
    print(f"Wrote Moon longitude plot: {args.moon_lon_plot}")
    if args.moon_lat_plot:
        print(f"Wrote Moon latitude plot: {args.moon_lat_plot}")

    by_body_lon = defaultdict(list)
    by_body_lat = defaultdict(list)
    by_body_dist = defaultdict(list)

    for row in rows:
        body = row["body"]
        by_body_lon[body].append(row["lon_error_arcsec"])
        by_body_lat[body].append(row["lat_error_arcsec"])
        by_body_dist[body].append(row["distance_error_km"])

    print()
    print("RMS comparison against JPL/book-style longitude")
    print("body        lon_rms_arcsec   lat_rms_arcsec   dist_rms_km")
    print("----------  --------------   --------------   -----------")

    for body in sorted(by_body_lon.keys()):
        print(
            f"{body:10s}  "
            f"{rms(by_body_lon[body]):14.3f}   "
            f"{rms(by_body_lat[body]):14.3f}   "
            f"{rms(by_body_dist[body]):11.3f}"
        )

    moon_lon = np.array(by_body_lon["Moon"], dtype=float)
    print()
    print("Moon longitude residual diagnostics")
    print(f"  mean_arcsec : {float(np.mean(moon_lon)):.3f}")
    print(f"  rms_arcsec  : {rms(moon_lon):.3f}")
    print(f"  max_arcsec  : {float(np.max(moon_lon)):.3f}")
    print(f"  min_arcsec  : {float(np.min(moon_lon)):.3f}")
    print(f"  peak_abs    : {float(np.max(np.abs(moon_lon))):.3f}")


if __name__ == "__main__":
    main()
