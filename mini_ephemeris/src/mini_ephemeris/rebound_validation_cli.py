from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .long_term_stability_cli import (
    MODE_DESCRIPTION,
    TWO_BODY_MODEL_SCOPES,
    parse_start_datetime,
    sanitize_tag,
    stability_body_list,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import AU_M, ARCSEC_PER_RAD, DAY_S, JULIAN_YEAR_S, heliocentric_elements_for_state, seconds_to_years
from .stability_diagnostics import center_of_mass_position_velocity, total_angular_momentum_vector, total_newtonian_energy


FIELDS = [
    "time_years",
    "body",
    "energy_rel_drift",
    "angular_momentum_rel_drift",
    "a_au",
    "e",
    "i_deg",
    "Omega_deg",
    "omega_deg",
    "varpi_deg",
    "mean_longitude_deg",
]


def optional_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optional REBOUND Newtonian validation runner for stability-mode initial states."
    )
    parser.add_argument("--kernel-path", default="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument(
        "--model-scope",
        choices=["two_body_jupiter", "two_body_saturn", "two_body_mercury", "inner", "full"],
        default="two_body_jupiter",
    )
    parser.add_argument("--duration-years", type=float, default=100.0)
    parser.add_argument("--step-days", type=float, default=4.0)
    parser.add_argument("--record-every-years", type=float, default=1.0)
    parser.add_argument("--integrator", choices=["whfast", "ias15", "leapfrog"], default="whfast")
    parser.add_argument(
        "--ias15-epsilon",
        type=float,
        default=None,
        help="Optional IAS15 accuracy parameter when supported by the installed REBOUND version.",
    )
    parser.add_argument("--gr-model", choices=["none", "gr", "gr_full", "gr_potential"], default="none")
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability")
    parser.add_argument("--tag", default="rebound_validation")
    parser.add_argument("--inhouse-summary", default=None)
    parser.add_argument(
        "--simulation-archive",
        default=None,
        help="Optional REBOUND SimulationArchive path for checkpoint/restart validation.",
    )
    parser.add_argument(
        "--simulation-archive-interval-years",
        type=float,
        default=None,
        help="Write REBOUND SimulationArchive snapshots at this cadence in Julian years.",
    )
    parser.add_argument(
        "--resume-from-simulation-archive",
        default=None,
        help="Resume from the latest snapshot in a REBOUND SimulationArchive.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.duration_years <= 0.0:
        parser.error("--duration-years must be positive.")
    if args.step_days <= 0.0:
        parser.error("--step-days must be positive.")
    if args.record_every_years <= 0.0:
        parser.error("--record-every-years must be positive.")
    if args.ias15_epsilon is not None and args.ias15_epsilon <= 0.0:
        parser.error("--ias15-epsilon must be positive.")
    if args.simulation_archive_interval_years is not None and args.simulation_archive_interval_years <= 0.0:
        parser.error("--simulation-archive-interval-years must be positive.")
    if args.resume_from_simulation_archive and args.simulation_archive:
        parser.error("Use either --simulation-archive or --resume-from-simulation-archive for a single run.")


def state_from_sim(sim, masses: np.ndarray) -> NBodyState:
    positions = []
    velocities = []
    for particle in sim.particles:
        positions.append([particle.x, particle.y, particle.z])
        velocities.append([particle.vx, particle.vy, particle.vz])
    return NBodyState(
        positions=np.array(positions, dtype=float),
        velocities=np.array(velocities, dtype=float),
        masses=masses.copy(),
    )


def build_simulation(
    rebound,
    state: NBodyState,
    *,
    integrator: str,
    step_s: float,
    ias15_epsilon: float | None = None,
):
    sim = rebound.Simulation()
    sim.G = G_SI
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
    sim.integrator = integrator
    if integrator in {"whfast", "leapfrog"}:
        sim.dt = step_s
    if integrator == "ias15" and ias15_epsilon is not None and hasattr(sim, "ri_ias15"):
        sim.ri_ias15.epsilon = ias15_epsilon
    return sim


def load_simulation_archive(rebound, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"SimulationArchive not found: {path}")
    if hasattr(rebound, "SimulationArchive"):
        archive = rebound.SimulationArchive(str(path))
        if len(archive) == 0:
            raise RuntimeError(f"SimulationArchive contains no snapshots: {path}")
        return archive[-1], f"resumed from SimulationArchive snapshot {len(archive) - 1}"
    try:
        return rebound.Simulation(str(path)), "resumed from REBOUND binary file"
    except Exception as exc:
        raise RuntimeError(
            "This REBOUND version does not expose SimulationArchive loading through "
            "rebound.SimulationArchive or rebound.Simulation(path)."
        ) from exc


def configure_simulation_archive(sim, path: Path, *, interval_s: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sim, "save_to_file"):
        sim.save_to_file(str(path), interval=interval_s, delete_file=True)
        return "save_to_file"
    if hasattr(sim, "automateSimulationArchive"):
        sim.automateSimulationArchive(str(path), interval=interval_s, deletefile=True)
        return "automateSimulationArchive"
    raise RuntimeError(
        "This REBOUND version does not expose save_to_file or automateSimulationArchive."
    )


def add_reboundx_gr(sim, gr_model: str) -> str:
    if gr_model == "none":
        return "none"
    reboundx = optional_import("reboundx")
    if reboundx is None:
        raise RuntimeError(
            "reboundx is not installed. Install reboundx to test GR forces, or use --gr-model none."
        )
    rebx = reboundx.Extras(sim)
    force = rebx.load_force(gr_model)
    rebx.add_force(force)
    force.params["c"] = 299_792_458.0
    return gr_model


def wrapped_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def linear_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return math.nan
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def load_inhouse_comparison(path_text: str | None) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return {"error": f"in-house summary not found: {path}"}
    with path.open() as file_obj:
        summary = json.load(file_obj)
    return {
        "path": str(path),
        "diagnostic_extrema_over_records": summary.get("diagnostic_extrema_over_records", {}),
        "runtime": summary.get("runtime", {}),
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    rebound = optional_import("rebound")
    if rebound is None:
        print("REBOUND is not installed; optional validation skipped.")
        print("Install with: python -m pip install rebound")
        raise SystemExit(0)

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"rebound_validation_{tag}.csv"
    summary_path = output_dir / f"rebound_validation_summary_{tag}.json"

    bodies = stability_body_list(args.model_scope, include_pluto=False)
    sun_index = bodies.index("sun")
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    archive_status = "disabled"
    if args.resume_from_simulation_archive:
        sim, archive_status = load_simulation_archive(
            rebound,
            Path(args.resume_from_simulation_archive),
        )
        sim.integrator = args.integrator
        if args.integrator in {"whfast", "leapfrog"}:
            sim.dt = args.step_days * DAY_S
        if args.integrator == "ias15" and args.ias15_epsilon is not None and hasattr(sim, "ri_ias15"):
            sim.ri_ias15.epsilon = args.ias15_epsilon
    else:
        sim = build_simulation(
            rebound,
            state0,
            integrator=args.integrator,
            step_s=args.step_days * DAY_S,
            ias15_epsilon=args.ias15_epsilon,
        )
    if args.simulation_archive and args.simulation_archive_interval_years is not None:
        archive_method = configure_simulation_archive(
            sim,
            Path(args.simulation_archive),
            interval_s=args.simulation_archive_interval_years * JULIAN_YEAR_S,
        )
        archive_status = f"writing {args.simulation_archive} via {archive_method}"
    gr_status = add_reboundx_gr(sim, args.gr_model)

    energy0 = total_newtonian_energy(state0, G=G_SI)
    angular0 = total_angular_momentum_vector(state0)
    angular0_norm = float(np.linalg.norm(angular0))
    _, com_velocity0 = center_of_mass_position_velocity(state0)
    duration_s = args.duration_years * JULIAN_YEAR_S
    record_s = args.record_every_years * JULIAN_YEAR_S
    n_records = int(math.floor(duration_s / record_s)) + 1
    if args.resume_from_simulation_archive:
        start_record_index = max(0, int(math.floor(max(0.0, float(sim.t)) / record_s)) + 1)
    else:
        start_record_index = 0

    max_energy_rel = 0.0
    max_angular_rel = 0.0
    max_com_velocity_drift = 0.0
    mercury_varpi_times: list[float] = []
    mercury_varpi_values: list[float] = []
    previous_varpi = None
    unwrapped_varpi = None

    start_wall = time.perf_counter()
    append_csv = bool(args.resume_from_simulation_archive and csv_path.exists())
    with csv_path.open("a" if append_csv else "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
        if not append_csv:
            writer.writeheader()
        for record_index in range(start_record_index, n_records + 1):
            target_s = min(record_index * record_s, duration_s)
            if target_s < float(sim.t) - 1.0e-9:
                continue
            sim.integrate(target_s, exact_finish_time=1)
            state = state_from_sim(sim, state0.masses)
            energy_rel = (total_newtonian_energy(state, G=G_SI) - energy0) / (abs(energy0) or 1.0)
            angular_rel = float(np.linalg.norm(total_angular_momentum_vector(state) - angular0)) / (
                angular0_norm or 1.0
            )
            max_energy_rel = max(max_energy_rel, abs(energy_rel))
            max_angular_rel = max(max_angular_rel, abs(angular_rel))
            _, com_velocity = center_of_mass_position_velocity(state)
            com_velocity_drift = float(np.linalg.norm(com_velocity - com_velocity0)) * JULIAN_YEAR_S / AU_M
            max_com_velocity_drift = max(max_com_velocity_drift, com_velocity_drift)
            time_years = seconds_to_years(target_s)
            elements = heliocentric_elements_for_state(state, bodies, sun_index=sun_index)
            for element in elements:
                row = element.as_output_row(time_years)
                writer.writerow(
                    {
                        "time_years": time_years,
                        "body": element.body_name,
                        "energy_rel_drift": energy_rel,
                        "angular_momentum_rel_drift": angular_rel,
                        "a_au": row["a_au"],
                        "e": row["e"],
                        "i_deg": row["i_deg"],
                        "Omega_deg": row["Omega_deg"],
                        "omega_deg": row["omega_deg"],
                        "varpi_deg": row["varpi_deg"],
                        "mean_longitude_deg": row["mean_longitude_deg"],
                    }
                )
                if element.body_name == "mercury barycenter":
                    varpi = element.longitude_perihelion_rad
                    if previous_varpi is None:
                        previous_varpi = varpi
                        unwrapped_varpi = varpi
                    else:
                        assert unwrapped_varpi is not None
                        unwrapped_varpi += wrapped_delta(varpi, previous_varpi)
                        previous_varpi = varpi
                    mercury_varpi_times.append(time_years)
                    mercury_varpi_values.append(float(unwrapped_varpi))
            if target_s >= duration_s:
                break

    runtime_s = time.perf_counter() - start_wall
    perihelion_drift = (
        linear_slope(mercury_varpi_times, mercury_varpi_values) * ARCSEC_PER_RAD * 100.0
        if mercury_varpi_times
        else math.nan
    )
    summary = {
        "mode": "optional REBOUND validation backend",
        "stability_mode_boundary": MODE_DESCRIPTION,
        "caution": (
            "Initial scope is Newtonian validation unless GR is explicitly requested. "
            "REBOUNDx GR options are exposed as scaffolding and are not the package default."
        ),
        "configuration": {
            "kernel_path": args.kernel_path,
            "start_date": args.start_date.isoformat(),
            "model_scope": args.model_scope,
            "integrator": args.integrator,
            "gr_model": args.gr_model,
            "gr_status": gr_status,
            "duration_years": args.duration_years,
            "step_days": args.step_days,
            "ias15_epsilon": args.ias15_epsilon,
            "record_every_years": args.record_every_years,
            "tag": tag,
            "body_names": bodies,
            "simulation_archive": args.simulation_archive,
            "simulation_archive_interval_years": args.simulation_archive_interval_years,
            "resume_from_simulation_archive": args.resume_from_simulation_archive,
            "simulation_archive_status": archive_status,
            "appended_csv_on_resume": append_csv,
        },
        "diagnostics": {
            "runtime_seconds": runtime_s,
            "runtime_minutes": runtime_s / 60.0,
            "max_energy_rel_drift": max_energy_rel,
            "max_angular_momentum_rel_drift": max_angular_rel,
            "max_com_velocity_drift_au_per_year": max_com_velocity_drift,
            "mercury_perihelion_drift_arcsec_per_century": perihelion_drift,
        },
        "comparison": {
            "inhouse_summary": load_inhouse_comparison(args.inhouse_summary),
        },
        "outputs": {
            "csv": str(csv_path),
            "summary": str(summary_path),
        },
    }
    with summary_path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"runtime_seconds={runtime_s:.3f}")
    print(f"max_energy_rel_drift={max_energy_rel:.6e}")
    print(f"max_angular_momentum_rel_drift={max_angular_rel:.6e}")
    if math.isfinite(perihelion_drift):
        print(f"mercury_perihelion_drift_arcsec_per_century={perihelion_drift:.6e}")


if __name__ == "__main__":
    main()
