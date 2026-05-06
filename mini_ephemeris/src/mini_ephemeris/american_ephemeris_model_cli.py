from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from skyfield.api import load

from .american_ephemeris import (
    build_model_vs_jpl_ephemeris_rows,
    make_tt_midnight_times,
    write_rows_csv,
)
from .ephem import (
    EphemerisConfig,
    initial_state_solar_system_barycentric_time,
    solar_system_body_list_earth_moon,
)
from .advanced_integrators import (
    integrate_dop853_with_accel,
    acceleration_newtonian,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun_earth_j2,
)


DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S


def rms(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare integrated model output against American Ephemeris-style JPL zodiac longitude."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--integrator", default="dop853", choices=["dop853"])
    parser.add_argument("--gr-model", default="sun", choices=["none", "sun"])
    parser.add_argument("--earth-j2", action="store_true")

    parser.add_argument("--chunk-years", type=float, default=1.0)
    parser.add_argument("--max-step-days", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-15)
    parser.add_argument("--include-pluto", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")

    args = parser.parse_args()

    ts = load.timescale()

    # Book-style ET/TT midnight samples for the requested month.
    days, t_month = make_tt_midnight_times(ts, args.year, args.month)
    dates = [f"{args.year:04d}-{args.month:02d}-{int(day):02d}" for day in days]

    # Start model at ET/TT midnight on 2000-01-01.
    t0 = ts.tt(2000, 1, 1, 0, 0, 0)
    offsets_s = (t_month.tt - t0.tt) * DAY_S

    if np.any(offsets_s < -1e-9):
        raise ValueError("This comparison CLI currently assumes dates >= 2000-01-01.")

    t_end = float(offsets_s[-1])

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

    # Use daily output; DOP853 still adapts internally, controlled by max_step.
    dt_s = DAY_S

    print("[American Ephemeris Model Compare]")
    print(f"  month          : {args.year:04d}-{args.month:02d}")
    print(f"  t_end_days     : {t_end / DAY_S:.1f}")
    print(f"  bodies         : {bodies}")
    print(f"  gr_model       : {args.gr_model}")
    print(f"  earth_j2       : {args.earth_j2}")
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

    # Select the model samples corresponding to the requested TT midnights.
    sample_indices = np.rint(offsets_s / dt_s).astype(int)
    max_alignment_error_s = float(np.max(np.abs(times_s[sample_indices] - offsets_s)))

    if max_alignment_error_s > 1e-5:
        raise RuntimeError(
            f"Model output is not aligned with requested TT midnights. "
            f"Max alignment error = {max_alignment_error_s} s"
        )

    model_positions_month = positions_m[sample_indices]

    rows = build_model_vs_jpl_ephemeris_rows(
        dates=dates,
        times=t_month,
        model_positions_m=model_positions_month,
        bodies=bodies,
        kernel_path=args.kernel_path,
    )

    write_rows_csv(rows, args.output)
    print(f"Wrote {args.output}")

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


if __name__ == "__main__":
    main()