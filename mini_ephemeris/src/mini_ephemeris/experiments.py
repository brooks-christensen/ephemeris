from __future__ import annotations
import argparse
import datetime as dt
import math
import time

import numpy as np

from .nbody import integrate, total_energy_series, NBodyState, G_SI
from .ephem import (
    EphemerisConfig,
    initial_state_sun_earth,
    truth_positions_sun_earth,
    initial_state_sun_earth_jupiter,
    truth_positions_sun_earth_jupiter,
    initial_state_sun_earth_moon,
    truth_positions_sun_earth_moon,
    initial_state_solar_system_barycentric,
    truth_positions_solar_system_barycentric,
    # SOLAR_SYSTEM_BODIES_DEFAULT,
    solar_system_body_list,
    solar_system_body_list_earth_moon,
)
from .plotting import (
    plot_trajectory_xy,
    plot_trajectory_xy_raw,
    plot_position_error,
    plot_log_error,
    plot_energy,
    plot_poincare_section,
)
from .error_metrics import position_error, rms
from .analysis_tools import poincare_section_y0, lyapunov_max
from .advanced_integrators import (
    integrate_with_accel,
    integrate_rk4_with_accel,
    integrate_dop853_with_accel,
    acceleration_newtonian,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_eih_1pn,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun_earth_j2,
)


def run_sun_earth(years: float,
                  dt_days: float,
                  with_truth: bool = True) -> None:
    t0 = dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    state0 = initial_state_sun_earth(t0)

    t1 = years * 365.25 * 86400.0
    dt_s = dt_days * 86400.0

    times, positions, velocities = integrate(state0, (0.0, t1), dt_s)

    plot_trajectory_xy(positions, labels=["Sun", "Earth"],
                       title="Sun–Earth Newtonian orbit (velocity-Verlet)")

    energies = total_energy_series(positions, velocities, state0.masses)
    plot_energy(times, energies, title="Sun–Earth total energy drift")

    if with_truth:
        truth = truth_positions_sun_earth(times, t0)
        err = position_error(positions[:, 1, :], truth)
        plot_position_error(times, err, title="Sun–Earth position error vs JPL DE")


def run_sun_earth_jupiter(years: float,
                           dt_days: float,
                           with_truth: bool = True,
                           with_lyapunov: bool = False) -> None:
    t0 = dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    state0 = initial_state_sun_earth_jupiter(t0)

    t1 = years * 365.25 * 86400.0
    dt_s = dt_days * 86400.0

    times, positions, velocities = integrate(state0, (0.0, t1), dt_s)

    plot_trajectory_xy(
        positions,
        labels=["Sun", "Earth", "Jupiter"],
        title="Sun–Earth–Jupiter Newtonian orbits (velocity-Verlet)",
    )

    energies = total_energy_series(positions, velocities, state0.masses)
    plot_energy(times, energies, title="Sun–Earth–Jupiter total energy drift")

    if with_truth:
        truth = truth_positions_sun_earth_jupiter(times, t0)
        err_e = position_error(positions[:, 1, :], truth[:, 1, :])
        err_j = position_error(positions[:, 2, :], truth[:, 2, :])

        plot_position_error(times, err_e,
                            title="Earth position error vs JPL DE")
        plot_position_error(times, err_j,
                            title="Jupiter position error vs JPL DE")

    if with_lyapunov:
        lam = lyapunov_max(state0, (0.0, t1), dt_s)
        print(f"Approximate max Lyapunov exponent (Sun–Earth–Jupiter, Newtonian): {lam:.3e} 1/s")


def run_sun_earth_moon(years: float,
                        dt_days: float,
                        with_truth: bool = True) -> None:
    t0 = dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    state0 = initial_state_sun_earth_moon(t0)

    t1 = years * 365.25 * 86400.0
    dt_s = dt_days * 86400.0

    times, positions, velocities = integrate(state0, (0.0, t1), dt_s)

    plot_trajectory_xy(
        positions,
        labels=["Sun", "Earth", "Moon"],
        title="Sun–Earth–Moon Newtonian orbits (velocity-Verlet)",
    )

    energies = total_energy_series(positions, velocities, state0.masses)
    plot_energy(times, energies, title="Sun–Earth–Moon total energy drift")

    if with_truth:
        truth = truth_positions_sun_earth_moon(times, t0)
        err_e = position_error(positions[:, 1, :], truth[:, 1, :])
        err_m = position_error(positions[:, 2, :], truth[:, 2, :])

        plot_position_error(times, err_e,
                            title="Earth position error vs JPL DE")
        plot_position_error(times, err_m,
                            title="Moon position error vs JPL DE")


def _equal_mass_euler_state() -> NBodyState:
    positions = np.array([
        [-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0],
        [ 1.0, 0.0, 0.0],
    ], dtype=float)

    m = 1.0
    masses = np.array([m, m, m], dtype=float)

    omega = math.sqrt(1.25)

    velocities = np.zeros_like(positions)
    for i in range(3):
        x, y, _ = positions[i]
        velocities[i, 0] = -omega * y
        velocities[i, 1] =  omega * x

    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def _equal_mass_lagrange_state() -> NBodyState:
    R = 1.0
    positions = np.array([
        [ R, 0.0, 0.0],
        [-0.5 * R,  math.sqrt(3) * 0.5 * R, 0.0],
        [-0.5 * R, -math.sqrt(3) * 0.5 * R, 0.0],
    ], dtype=float)

    m = 1.0
    masses = np.array([m, m, m], dtype=float)

    omega = 0.7598356856515925

    velocities = np.zeros_like(positions)
    for i in range(3):
        x, y, _ = positions[i]
        velocities[i, 0] = -omega * y
        velocities[i, 1] =  omega * x

    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def run_three_body_euler(years: float,
                         dt_days: float) -> None:
    state0 = _equal_mass_euler_state()
    t1 = years
    dt_s = dt_days

    times, positions, velocities = integrate(state0, (0.0, t1), dt_s, G=1.0)

    plot_trajectory_xy_raw(
        positions,
        labels=["m1", "m2", "m3"],
        title="Equal-mass Euler collinear three-body configuration",
    )

    x_sec, vx_sec = poincare_section_y0(times, positions, velocities,
                                        body_index=0, direction="positive")
    plot_poincare_section(
        x_sec,
        vx_sec,
        title="Poincaré section (Euler, body 1, y=0 crossings)",
    )


def run_three_body_lagrange(years: float,
                            dt_days: float) -> None:
    state0 = _equal_mass_lagrange_state()
    t1 = years
    dt_s = dt_days

    times, positions, velocities = integrate(state0, (0.0, t1), dt_s, G=1.0)

    plot_trajectory_xy_raw(
        positions,
        labels=["m1", "m2", "m3"],
        title="Equal-mass Lagrange equilateral three-body configuration",
    )

    x_sec, vx_sec = poincare_section_y0(times, positions, velocities,
                                        body_index=0, direction="positive")
    plot_poincare_section(
        x_sec,
        vx_sec,
        title="Poincaré section (Lagrange, body 1, y=0 crossings)",
    )


def run_solar_system_ephem(
    years: float,
    dt_days: float,
    kernel_path: str,
    with_lyapunov: bool = False,
    verbose: bool = True,
    gr_model: str = "sun",
    include_pluto: bool = False,
    truth_stride: int = 365,
    integrator: str = "velocity_verlet",
    rtol: float = 1e-12,
    atol: float = 1e-15,
    chunk_years: float = 1.0,
    show_progress: bool = True,
    show_plots: bool = False,
    max_step_days: float | None = None,
) -> None:
    config = EphemerisConfig(kernel_path=kernel_path)
    t0 = dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

    # bodies = SOLAR_SYSTEM_BODIES_DEFAULT
    bodies = solar_system_body_list(include_pluto=include_pluto)
    state0 = initial_state_solar_system_barycentric(t0, bodies=bodies, config=config)

    # Physical times
    t1 = years * 365.25 * 86400.0
    dt_s = dt_days * 86400.0

    # Save about one point per day
    record_every = max(1, int(86400.0 / dt_s))
    n_steps = int(np.floor(t1 / dt_s))

    # accel_func = acceleration_newtonian_gr_sun if use_gr else acceleration_newtonian
    # accel_kwargs = {"sun_index": 0} if use_gr else {}
    if gr_model == "none":
        accel_func = acceleration_newtonian
        accel_kwargs = {}
    elif gr_model == "sun":
        accel_func = acceleration_newtonian_gr_sun
        accel_kwargs = {"sun_index": 0}
    elif gr_model == "eih":
        accel_func = acceleration_newtonian_eih_1pn
        accel_kwargs = {}
    else:
        raise ValueError(f"Unknown gr_model: {gr_model!r}")

    if verbose:
        print("[Experiment] solar_system_ephem")
        print(f"  span        : {years:.1f} years")
        print(f"  dt          : {dt_days:.5f} days")
        print(f"  steps       : {n_steps}")
        print(f"  record_every: {record_every}")
        print(f"  kernel      : {kernel_path}")
        print(f"  bodies      : {bodies}")
        print(f"  gr_model     : {gr_model}")
        print(f"  include_pluto: {include_pluto}")
        print(f"  truth_stride : {truth_stride}")
        print(f"  integrator  : {integrator}")
        print(f"  show_plots  : {show_plots}")
        print(f"  rtol        : {rtol}")
        print(f"  atol        : {atol}")
        print(f"  chunk_years  : {chunk_years}")
        print(f"  show_progress: {show_progress}")
        print(f"  max_step_days: {max_step_days}")
        print("", flush=True)

    start_wall = time.time()

    def progress_callback(frac: float, i: int, t_model: float) -> None:
        elapsed = time.time() - start_wall
        eta = float("inf") if frac <= 0.0 else elapsed * (1.0 - frac) / frac
        t_years = t_model / (86400.0 * 365.25)
        msg = (
            f"\r[Progress] {frac * 100:6.2f}%  "
            f"step {i:9d}/{n_steps:<9d}  "
            f"t_model ≈ {t_years:9.1f} yr  "
            f"elapsed {elapsed/60:6.1f} min  "
            f"ETA {eta/60:6.1f} min"
        )
        print(msg, end="", flush=True)

    if verbose:
        print("[Experiment] Starting integration...", flush=True)

    if integrator == "velocity_verlet":
        integrator_func = integrate_with_accel
        integrator_kwargs = {}
    elif integrator == "rk4":
        integrator_func = integrate_rk4_with_accel
        integrator_kwargs = {}
    elif integrator == "dop853":
        integrator_func = integrate_dop853_with_accel
        integrator_kwargs = {
            "rtol": rtol,
            "atol": atol,
            "chunk_duration": chunk_years * 365.25 * 86400.0,
            "show_progress": show_progress,
            "max_step": None if max_step_days is None else max_step_days * 86400.0,
        }
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")

    times, positions, velocities = integrator_func(
        state0,
        (0.0, t1),
        dt_s,
        accel_func,
        accel_kwargs=accel_kwargs,
        G=G_SI,
        record_every=record_every,
        progress=progress_callback if verbose else None,
        **integrator_kwargs,
    )

    if verbose:
        print("\n[Experiment] Integration complete.", flush=True)

    labels = list(bodies)

    # Downsample trajectory plot to keep matplotlib responsive
    plot_stride = max(1, len(times) // 5000)
    idx_plot = np.arange(0, len(times), plot_stride, dtype=int)

    plot_trajectory_xy(
        positions[idx_plot],
        labels=labels,
        title=f"Barycentric Solar-System trajectories (GR model: {gr_model})",
        filename=f"/home/peacelovephysics/ephemeris/output/trajectories_solar_system_ephem_{integrator}_years{years}_dt{dt_days}_gr{gr_model}.png" if not show_plots else None,
        show=show_plots,
    )

    # Sparse, apples-to-apples validation against JPL truth
    # With ~1 saved sample/day, truth_stride=365 means ~1 comparison per year.
    # truth_stride = 365
    idx_cmp = np.arange(0, len(times), truth_stride, dtype=int)

    times_cmp = times[idx_cmp]
    positions_cmp = positions[idx_cmp]

    if verbose:
        print(
            f"[Experiment] Computing JPL truth on {len(times_cmp)} comparison points "
            f"(truth_stride={truth_stride}) ...",
            flush=True,
        )

    truth = truth_positions_solar_system_barycentric(
        times_cmp,
        t0,
        bodies=bodies,
        config=config,
    )

    idx_mercury = bodies.index("mercury barycenter")
    idx_earth = bodies.index("earth barycenter")
    idx_jupiter = bodies.index("jupiter barycenter")

    err_mercury = position_error(
        positions_cmp[:, idx_mercury, :],
        truth[:, idx_mercury, :],
    )
    err_earth = position_error(
        positions_cmp[:, idx_earth, :],
        truth[:, idx_earth, :],
    )
    err_jupiter = position_error(
        positions_cmp[:, idx_jupiter, :],
        truth[:, idx_jupiter, :],
    )

    print(f"RMS position error Mercury: {rms(err_mercury)/1e3:.3f} km")
    print(f"RMS position error Earth:   {rms(err_earth)/1e3:.3f} km")
    print(f"RMS position error Jupiter: {rms(err_jupiter)/1e3:.3f} km")

    plot_position_error(
        times_cmp,
        err_mercury,
        title="Mercury position error vs JPL (barycentric)",
        filename=f"/home/peacelovephysics/ephemeris/output/mercury_error_solar_system_ephem_{integrator}_years{years}_dt{dt_days}_gr{gr_model}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_earth,
        title="Earth position error vs JPL (barycentric)",
        filename=f"/home/peacelovephysics/ephemeris/output/earth_error_solar_system_ephem_{integrator}_years{years}_dt{dt_days}_gr{gr_model}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_jupiter,
        title="Jupiter position error vs JPL (barycentric)",
        filename=f"/home/peacelovephysics/ephemeris/output/jupiter_error_solar_system_ephem_{integrator}_years{years}_dt{dt_days}_gr{gr_model}.png" if not show_plots else None,
        show=show_plots,
    )

    if with_lyapunov:
        perturbed = state0.copy()
        perturbed.positions[idx_earth, 0] += 1.0

        if verbose:
            print("[Experiment] Starting perturbed integration for Lyapunov-style separation...", flush=True)

        times2, positions2, _ = integrator_func(
            perturbed,
            (0.0, t1),
            dt_s,
            accel_func,
            accel_kwargs=accel_kwargs,
            G=G_SI,
            record_every=record_every,
            **integrator_kwargs,
        )

        sep_earth = np.linalg.norm(
            positions2[:, idx_earth, :] - positions[:, idx_earth, :],
            axis=-1,
        )

        # Downsample the Lyapunov-style plot to keep memory/rendering reasonable
        idx_lya = idx_cmp
        plot_log_error(
            times[idx_lya],
            sep_earth[idx_lya],
            title="Earth trajectory separation (1 m initial perturbation)",
        )


def run_solar_system_ephem_earth_moon(
    years: float,
    dt_days: float,
    kernel_path: str,
    verbose: bool = True,
    gr_model: str = "sun",
    truth_stride: int = 30,
    integrator: str = "dop853",
    rtol: float = 1e-12,
    atol: float = 1e-15,
    chunk_years: float = 1.0,
    show_progress: bool = True,
    show_plots: bool = False,
    max_step_days: float | None = None,
    earth_j2: bool = False,
) -> None:
    config = EphemerisConfig(kernel_path=kernel_path)

    # Start at J2000-era window for ephemeris comparison mode.
    # You can later expose this as a CLI date argument if you want.
    t0 = dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)

    bodies = solar_system_body_list_earth_moon()
    state0 = initial_state_solar_system_barycentric(t0, bodies=bodies, config=config)

    t1 = years * 365.25 * 86400.0
    dt_s = dt_days * 86400.0

    record_every = max(1, int(86400.0 / dt_s))
    n_steps = int(np.floor(t1 / dt_s))

    # if gr_model == "none":
    #     accel_func = acceleration_newtonian
    #     accel_kwargs = {}
    # elif gr_model == "sun":
    #     accel_func = acceleration_newtonian_gr_sun
    #     accel_kwargs = {"sun_index": 0}
    # elif gr_model == "eih":
    #     accel_func = acceleration_newtonian_eih_1pn
    #     accel_kwargs = {}
    # else:
    #     raise ValueError(f"Unknown gr_model: {gr_model!r}")

    sun_index = bodies.index("sun")
    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")

    if gr_model == "none":
        if earth_j2:
            accel_func = acceleration_newtonian_earth_j2
            accel_kwargs = {
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        else:
            accel_func = acceleration_newtonian
            accel_kwargs = {}

    elif gr_model == "sun":
        if earth_j2:
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

    elif gr_model == "eih":
        if earth_j2:
            raise ValueError("earth_j2 is not wired for gr_model='eih' yet.")
        accel_func = acceleration_newtonian_eih_1pn
        accel_kwargs = {}

    else:
        raise ValueError(f"Unknown gr_model: {gr_model!r}")

    if integrator == "velocity_verlet":
        integrator_func = integrate_with_accel
        integrator_kwargs = {}
    elif integrator == "rk4":
        integrator_func = integrate_rk4_with_accel
        integrator_kwargs = {}
    elif integrator == "dop853":
        integrator_func = integrate_dop853_with_accel
        integrator_kwargs = {
            "rtol": rtol,
            "atol": atol,
            "chunk_duration": chunk_years * 365.25 * 86400.0,
            "show_progress": show_progress,
            "max_step": None if max_step_days is None else max_step_days * 86400.0,
        }
    else:
        raise ValueError(f"Unknown integrator: {integrator!r}")

    if verbose:
        print("[Experiment] solar_system_ephem_earth_moon")
        print(f"  span         : {years:.1f} years")
        print(f"  dt           : {dt_days:.5f} days")
        print(f"  steps        : {n_steps}")
        print(f"  record_every : {record_every}")
        print(f"  kernel       : {kernel_path}")
        print(f"  bodies       : {bodies}")
        print(f"  gr_model     : {gr_model}")
        print(f"  truth_stride : {truth_stride}")
        print(f"  integrator   : {integrator}")
        print(f"  rtol         : {rtol}")
        print(f"  atol         : {atol}")
        print(f"  chunk_years  : {chunk_years}")
        print(f"  show_progress: {show_progress}")
        print(f"  show_plots   : {show_plots}")
        print(f"  max_step_days: {max_step_days}")
        print(f"  earth_j2     : {earth_j2}")

    times, positions, velocities = integrator_func(
        state0,
        (0.0, t1),
        dt_s,
        accel_func,
        accel_kwargs=accel_kwargs,
        G=G_SI,
        record_every=record_every,
        **integrator_kwargs,
    )

    labels = list(bodies)
    plot_stride = max(1, len(times) // 5000)
    idx_plot = np.arange(0, len(times), plot_stride, dtype=int)

    plot_trajectory_xy(
        positions[idx_plot],
        labels=labels,
        title=f"Barycentric Solar-System trajectories (Earth+Moon explicit, GR model: {gr_model})",
        filename=f"/home/peacelovephysics/ephemeris/output/trajectories_solar_system_ephem_earth_moon_{integrator}_years{years}_dt{dt_days}_gr{gr_model}.png" if not show_plots else None,
        show=show_plots,
    )

    idx_cmp = np.arange(0, len(times), truth_stride, dtype=int)
    times_cmp = times[idx_cmp]
    positions_cmp = positions[idx_cmp]

    if verbose:
        print(
            f"[Experiment] Computing JPL truth on {len(times_cmp)} comparison points "
            f"(truth_stride={truth_stride}) ...",
            flush=True,
        )

    truth = truth_positions_solar_system_barycentric(
        times_cmp,
        t0,
        bodies=bodies,
        config=config,
    )

    idx_earth = bodies.index("earth")
    idx_moon = bodies.index("moon")
    idx_mercury = bodies.index("mercury barycenter")
    idx_jupiter = bodies.index("jupiter barycenter")

    err_earth = position_error(
        positions_cmp[:, idx_earth, :],
        truth[:, idx_earth, :],
    )
    err_moon = position_error(
        positions_cmp[:, idx_moon, :],
        truth[:, idx_moon, :],
    )
    err_mercury = position_error(
        positions_cmp[:, idx_mercury, :],
        truth[:, idx_mercury, :],
    )
    err_jupiter = position_error(
        positions_cmp[:, idx_jupiter, :],
        truth[:, idx_jupiter, :],
    )
    model_moon_geo = positions_cmp[:, idx_moon, :] - positions_cmp[:, idx_earth, :]
    truth_moon_geo = truth[:, idx_moon, :] - truth[:, idx_earth, :]

    err_moon_geo = position_error(model_moon_geo, truth_moon_geo)

    print(f"RMS position error Mercury: {rms(err_mercury)/1e3:.3f} km")
    print(f"RMS position error Earth:   {rms(err_earth)/1e3:.3f} km")
    print(f"RMS position error Moon:    {rms(err_moon)/1e3:.3f} km")
    print(f"RMS position error Jupiter: {rms(err_jupiter)/1e3:.3f} km")
    print(f"RMS position error Moon geo: {rms(err_moon_geo)/1e3:.3f} km")

    output_dir = "/home/peacelovephysics/ephemeris/output"
    tag = (
        f"solar_system_ephem_earth_moon_"
        f"{integrator}_years{years}_dt{dt_days}_gr{gr_model}"
    )

    plot_position_error(
        times_cmp,
        err_mercury,
        title="Mercury position error vs JPL",
        filename=f"{output_dir}/mercury_error_{tag}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_earth,
        title="Earth position error vs JPL",
        filename=f"{output_dir}/earth_error_{tag}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_moon,
        title="Moon position error vs JPL",
        filename=f"{output_dir}/moon_error_{tag}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_jupiter,
        title="Jupiter position error vs JPL",
        filename=f"{output_dir}/jupiter_error_{tag}.png" if not show_plots else None,
        show=show_plots,
    )
    plot_position_error(
        times_cmp,
        err_moon_geo,
        title="Geocentric Moon position error vs JPL",
        filename=f"{output_dir}/moon_geocentric_error_{tag}.png" if not show_plots else None,
        show=show_plots,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run N-body ephemeris and toy 3-body experiments.",
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "sun_earth",
            "sun_earth_jupiter",
            "sun_earth_moon",
            "three_body_euler",
            "three_body_lagrange",
            "solar_system_ephem",
            "solar_system_ephem_earth_moon",
        ],
        default="sun_earth",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Duration of the simulation (years for Solar System, dimensionless for toy 3-body).",
    )
    parser.add_argument(
        "--dt-days",
        type=float,
        default=1.0,
        help="Timestep (days for Solar System, dimensionless for toy 3-body).",
    )
    parser.add_argument(
        "--no-truth",
        action="store_true",
        help="(Kept for backward compatibility; unused in solar_system_ephem).",
    )
    parser.add_argument(
        "--with-lyapunov",
        action="store_true",
        help="Estimate Lyapunov separation for solar_system_ephem or Sun–Earth–Jupiter.",
    )
    parser.add_argument(
        "--kernel-path",
        type=str,
        default="de440s.bsp",
        help="Path/name of the JPL kernel to use (Solar-System experiments).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information during experiments.",
    )
    # parser.add_argument(
    #     "--no-gr",
    #     action="store_true",
    #     help="Disable the current Sun-centered GR correction in solar_system_ephem.",
    # )
    parser.add_argument(
        "--include-pluto",
        action="store_true",
        help="Include Pluto barycenter in the full Solar-System body list.",
    )
    parser.add_argument(
        "--truth-stride",
        type=int,
        default=365,
        help="Evaluate JPL truth every N saved samples in solar_system_ephem.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots rather than storing them."
    )
    parser.add_argument(
        "--integrator",
        type=str,
        default="velocity_verlet",
        choices=["velocity_verlet", "rk4", "dop853"],
        help="Integrator to use in solar_system_ephem.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-12,
        help="Relative tolerance for adaptive integrators like DOP853.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-15,
        help="Absolute tolerance for adaptive integrators like DOP853.",
    )
    parser.add_argument(
        "--gr-model",
        type=str,
        default="sun",
        choices=["none", "sun", "eih"],
        help="GR model for solar_system_ephem: none, sun, or eih.",
    )
    parser.add_argument(
        "--chunk-years",
        type=float,
        default=1.0,
        help="Chunk duration in years for DOP853 integration progress/checkpointing.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars for chunked adaptive integrators.",
    )
    parser.add_argument(
        "--max-step-days",
        type=float,
        default=None,
        help="Maximum internal DOP853 step size in days. If omitted, DOP853 chooses freely.",
    )
    parser.add_argument(
        "--earth-j2",
        action="store_true",
        help="Include Earth's J2 oblateness correction in the Earth+Moon ephemeris experiment.",
    )

    args = parser.parse_args(argv)

    if args.experiment == "sun_earth":
        run_sun_earth(args.years, args.dt_days, with_truth=not args.no_truth)
    elif args.experiment == "sun_earth_jupiter":
        run_sun_earth_jupiter(
            args.years,
            args.dt_days,
            with_truth=not args.no_truth,
            with_lyapunov=args.with_lyapunov,
        )
    elif args.experiment == "sun_earth_moon":
        run_sun_earth_moon(args.years, args.dt_days, with_truth=not args.no_truth)
    elif args.experiment == "three_body_euler":
        run_three_body_euler(args.years, args.dt_days)
    elif args.experiment == "three_body_lagrange":
        run_three_body_lagrange(args.years, args.dt_days)
    elif args.experiment == "solar_system_ephem":
        run_solar_system_ephem(
            args.years,
            args.dt_days,
            kernel_path=args.kernel_path,
            with_lyapunov=args.with_lyapunov,
            verbose=args.verbose,
            gr_model=args.gr_model,
            include_pluto=args.include_pluto,
            truth_stride=args.truth_stride,
            integrator=args.integrator,
            rtol=args.rtol,
            atol=args.atol,
            chunk_years=args.chunk_years,
            show_progress=not args.no_progress_bar,
            show_plots=args.show_plots,
            max_step_days=args.max_step_days,
        )
    elif args.experiment == "solar_system_ephem_earth_moon":
        run_solar_system_ephem_earth_moon(
            args.years,
            args.dt_days,
            kernel_path=args.kernel_path,
            verbose=args.verbose,
            gr_model=args.gr_model,
            truth_stride=args.truth_stride,
            integrator=args.integrator,
            rtol=args.rtol,
            atol=args.atol,
            chunk_years=args.chunk_years,
            show_progress=not args.no_progress_bar,
            show_plots=args.show_plots,
            max_step_days=args.max_step_days,
            earth_j2=args.earth_j2,
        )
    else:
        raise ValueError(f"Unknown experiment {args.experiment!r}")


if __name__ == "__main__":
    main()
