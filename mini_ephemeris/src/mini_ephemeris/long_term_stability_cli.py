from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import TextIO

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from .advanced_integrators import (
    acceleration_newtonian,
    acceleration_newtonian_gr_sun,
    pack_state,
    rhs_solve_ivp,
    unpack_state,
    velocity_verlet_step_generic,
)
from .ephem import (
    EphemerisConfig,
    initial_state_solar_system_barycentric,
    solar_system_body_list,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import (
    AU_M,
    DAY_S,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
    seconds_to_years,
)
from .stability_diagnostics import (
    PairwiseMinimumTracker,
    invariant_diagnostics_row,
    invariant_reference,
)


MODE_DESCRIPTION = "stability mode: physical reduced model, no empirical lunar calibration"

EMPIRICAL_LUNAR_FLAGS = {
    "--earth-j2",
    "--lunar-calibration-file",
    "--lunar-calibration-profile",
    "--moon-dv-r-mm-s",
    "--moon-dv-t-mm-s",
    "--moon-dv-h-mm-s",
    "--moon-a-r-1e-15-m-s2",
    "--moon-a-t-1e-15-m-s2",
    "--moon-a-h-1e-15-m-s2",
    "--moon-lon-plot",
    "--moon-lat-plot",
    "--moon-lon-ylim-arcsec",
    "--moon-lat-ylim-arcsec",
    "--no-preserve-emb-momentum",
}

STABILITY_TIMESERIES_FIELDS = [
    "time_years",
    "body",
    "x_au",
    "y_au",
    "z_au",
    "vx_au_per_year",
    "vy_au_per_year",
    "vz_au_per_year",
    "heliocentric_x_au",
    "heliocentric_y_au",
    "heliocentric_z_au",
    "heliocentric_vx_au_per_year",
    "heliocentric_vy_au_per_year",
    "heliocentric_vz_au_per_year",
    "heliocentric_r_au",
    "heliocentric_speed_au_per_year",
]

ORBITAL_ELEMENT_FIELDS = [
    "time_years",
    "body",
    "reference_plane",
    "a_au",
    "e",
    "i_deg",
    "Omega_deg",
    "omega_deg",
    "varpi_deg",
    "true_anomaly_deg",
    "mean_anomaly_deg",
    "mean_longitude_deg",
    "perihelion_au",
    "aphelion_au",
    "specific_energy_j_kg",
]

INVARIANT_FIELDS = [
    "time_years",
    "energy_j",
    "energy_abs_drift_j",
    "energy_rel_drift",
    "angular_momentum_norm_kg_m2_s",
    "angular_momentum_abs_drift_kg_m2_s",
    "angular_momentum_rel_drift",
    "angular_momentum_direction_drift_arcsec",
    "com_x_au",
    "com_y_au",
    "com_z_au",
    "com_vx_au_per_year",
    "com_vy_au_per_year",
    "com_vz_au_per_year",
    "com_position_drift_au",
    "com_velocity_drift_au_per_year",
]

MIN_SEPARATION_FIELDS = [
    "body_i",
    "body_j",
    "min_separation_au",
    "min_separation_km",
    "time_years",
]


@dataclass
class CsvOutputs:
    stability_timeseries: csv.DictWriter
    orbital_elements: csv.DictWriter
    invariants: csv.DictWriter
    files: tuple[TextIO, ...]
    paths: dict[str, Path]

    def flush(self) -> None:
        for file_obj in self.files:
            file_obj.flush()

    def close(self) -> None:
        for file_obj in self.files:
            file_obj.close()


@dataclass
class IntegrationResult:
    final_state: NBodyState
    actual_duration_s: float
    n_steps: int
    n_records: int
    min_tracker: PairwiseMinimumTracker
    extrema: dict[str, float]
    min_separation_sampling: str


def parse_start_datetime(text: str) -> dt.datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        if "T" in value or " " in value:
            parsed = dt.datetime.fromisoformat(value)
        else:
            parsed_date = dt.date.fromisoformat(value)
            parsed = dt.datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=dt.timezone.utc,
            )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected ISO date or datetime, for example 2000-01-01."
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def sanitize_tag(tag: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in tag.strip()
    )
    return cleaned or "stability"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run long-term Solar System stability experiments using the "
            "physical reduced barycenter model."
        )
    )
    parser.add_argument(
        "--kernel-path",
        default="de431_part-2.bsp",
        help="Path to a JPL BSP kernel used only for the initial state.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_start_datetime,
        default=parse_start_datetime("2000-01-01"),
        help="UTC ISO start date or datetime for the initial state.",
    )
    parser.add_argument(
        "--duration-years",
        type=float,
        default=1000.0,
        help="Integration duration in Julian years.",
    )
    parser.add_argument(
        "--step-days",
        type=float,
        default=4.0,
        help=(
            "Fixed leapfrog timestep in days, or DOP853 maximum internal step "
            "for validation runs."
        ),
    )
    parser.add_argument(
        "--record-every-years",
        type=float,
        default=1.0,
        help="CSV output cadence in Julian years.",
    )
    parser.add_argument(
        "--include-pluto",
        action="store_true",
        help="Include Pluto barycenter in the reduced model.",
    )
    parser.add_argument(
        "--gr-model",
        choices=["none", "sun"],
        default="none",
        help="Relativity model: none or Sun-centered 1PN GR.",
    )
    parser.add_argument(
        "--integrator",
        choices=["leapfrog", "dop853"],
        default="leapfrog",
        help="Integrator. Leapfrog is the default long-term stability integrator.",
    )
    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Directory for stability CSV and JSON outputs.",
    )
    parser.add_argument(
        "--tag",
        default="stability",
        help="Tag inserted into output file names.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    return parser


def reject_empirical_lunar_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    rejected: list[str] = []
    for token in argv:
        flag = token.split("=", 1)[0]
        if (
            flag in EMPIRICAL_LUNAR_FLAGS
            or flag.startswith("--moon-dv-")
            or flag.startswith("--moon-a-")
            or flag.startswith("--lunar-calibration")
        ):
            rejected.append(flag)

    if rejected:
        unique = ", ".join(sorted(set(rejected)))
        parser.error(
            "Empirical lunar calibration and explicit Earth-Moon tuning flags "
            f"are not accepted in stability mode: {unique}. Use the American "
            "Ephemeris CLIs for fitted short-range reproduction."
        )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.duration_years <= 0.0:
        parser.error("--duration-years must be positive.")
    if args.step_days <= 0.0:
        parser.error("--step-days must be positive.")
    if args.record_every_years <= 0.0:
        parser.error("--record-every-years must be positive.")


def output_paths(output_dir: Path, tag: str) -> dict[str, Path]:
    return {
        "stability_timeseries": output_dir / f"stability_timeseries_{tag}.csv",
        "orbital_elements": output_dir / f"orbital_elements_{tag}.csv",
        "invariants": output_dir / f"invariants_{tag}.csv",
        "min_separations": output_dir / f"min_separations_{tag}.csv",
        "summary": output_dir / f"summary_{tag}.json",
    }


def open_csv_outputs(paths: dict[str, Path]) -> CsvOutputs:
    stability_file = paths["stability_timeseries"].open("w", newline="")
    elements_file = paths["orbital_elements"].open("w", newline="")
    invariants_file = paths["invariants"].open("w", newline="")

    stability_writer = csv.DictWriter(
        stability_file,
        fieldnames=STABILITY_TIMESERIES_FIELDS,
    )
    elements_writer = csv.DictWriter(
        elements_file,
        fieldnames=ORBITAL_ELEMENT_FIELDS,
    )
    invariants_writer = csv.DictWriter(
        invariants_file,
        fieldnames=INVARIANT_FIELDS,
    )

    stability_writer.writeheader()
    elements_writer.writeheader()
    invariants_writer.writeheader()

    return CsvOutputs(
        stability_timeseries=stability_writer,
        orbital_elements=elements_writer,
        invariants=invariants_writer,
        files=(stability_file, elements_file, invariants_file),
        paths=paths,
    )


def select_acceleration_model(
    gr_model: str,
    *,
    sun_index: int,
):
    if gr_model == "none":
        return acceleration_newtonian, {}
    if gr_model == "sun":
        return acceleration_newtonian_gr_sun, {"sun_index": sun_index}
    raise ValueError(f"Unsupported gr_model: {gr_model!r}")


def update_extrema(extrema: dict[str, float], invariant_row: dict[str, float]) -> None:
    extrema["max_abs_energy_abs_drift_j"] = max(
        extrema["max_abs_energy_abs_drift_j"],
        abs(float(invariant_row["energy_abs_drift_j"])),
    )
    extrema["max_abs_energy_rel_drift"] = max(
        extrema["max_abs_energy_rel_drift"],
        abs(float(invariant_row["energy_rel_drift"])),
    )
    extrema["max_angular_momentum_abs_drift_kg_m2_s"] = max(
        extrema["max_angular_momentum_abs_drift_kg_m2_s"],
        abs(float(invariant_row["angular_momentum_abs_drift_kg_m2_s"])),
    )
    extrema["max_angular_momentum_rel_drift"] = max(
        extrema["max_angular_momentum_rel_drift"],
        abs(float(invariant_row["angular_momentum_rel_drift"])),
    )
    extrema["max_com_position_drift_au"] = max(
        extrema["max_com_position_drift_au"],
        abs(float(invariant_row["com_position_drift_au"])),
    )
    extrema["max_com_velocity_drift_au_per_year"] = max(
        extrema["max_com_velocity_drift_au_per_year"],
        abs(float(invariant_row["com_velocity_drift_au_per_year"])),
    )


def write_snapshot(
    time_s: float,
    state: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    reference,
    extrema: dict[str, float],
    *,
    sun_index: int,
) -> dict[str, float]:
    time_years = seconds_to_years(time_s)
    sun_position = state.positions[sun_index]
    sun_velocity = state.velocities[sun_index]

    for index, name in enumerate(body_names):
        position = state.positions[index]
        velocity = state.velocities[index]
        heliocentric_position = position - sun_position
        heliocentric_velocity = velocity - sun_velocity

        outputs.stability_timeseries.writerow(
            {
                "time_years": time_years,
                "body": name,
                "x_au": position[0] / AU_M,
                "y_au": position[1] / AU_M,
                "z_au": position[2] / AU_M,
                "vx_au_per_year": velocity[0] * JULIAN_YEAR_S / AU_M,
                "vy_au_per_year": velocity[1] * JULIAN_YEAR_S / AU_M,
                "vz_au_per_year": velocity[2] * JULIAN_YEAR_S / AU_M,
                "heliocentric_x_au": heliocentric_position[0] / AU_M,
                "heliocentric_y_au": heliocentric_position[1] / AU_M,
                "heliocentric_z_au": heliocentric_position[2] / AU_M,
                "heliocentric_vx_au_per_year": (
                    heliocentric_velocity[0] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_vy_au_per_year": (
                    heliocentric_velocity[1] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_vz_au_per_year": (
                    heliocentric_velocity[2] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_r_au": float(np.linalg.norm(heliocentric_position)) / AU_M,
                "heliocentric_speed_au_per_year": (
                    float(np.linalg.norm(heliocentric_velocity)) * JULIAN_YEAR_S / AU_M
                ),
            }
        )

    for elements in heliocentric_elements_for_state(
        state,
        body_names,
        sun_index=sun_index,
    ):
        outputs.orbital_elements.writerow(elements.as_output_row(time_years))

    invariant_row = invariant_diagnostics_row(time_s, state, reference)
    outputs.invariants.writerow(invariant_row)
    update_extrema(extrema, invariant_row)
    return invariant_row


def initial_extrema() -> dict[str, float]:
    return {
        "max_abs_energy_abs_drift_j": 0.0,
        "max_abs_energy_rel_drift": 0.0,
        "max_angular_momentum_abs_drift_kg_m2_s": 0.0,
        "max_angular_momentum_rel_drift": 0.0,
        "max_com_position_drift_au": 0.0,
        "max_com_velocity_drift_au_per_year": 0.0,
    }


def integrate_leapfrog_streaming(
    state0: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    *,
    duration_s: float,
    dt_s: float,
    record_interval_s: float,
    accel_func,
    accel_kwargs: dict,
    show_progress: bool,
    sun_index: int,
) -> IntegrationResult:
    total_steps = int(math.ceil(duration_s / dt_s))
    record_every_steps = max(1, int(round(record_interval_s / dt_s)))

    state = state0.copy()
    acceleration = accel_func(state, G=G_SI, **accel_kwargs)
    reference = invariant_reference(state, G=G_SI)
    min_tracker = PairwiseMinimumTracker.create(body_names)
    min_tracker.update(0.0, state.positions)

    extrema = initial_extrema()
    write_snapshot(0.0, state, body_names, outputs, reference, extrema, sun_index=sun_index)

    n_records = 1
    time_s = 0.0
    final_invariant_row = None

    progress = tqdm(
        total=total_steps,
        desc="leapfrog steps",
        disable=not show_progress,
    )

    try:
        for step_index in range(1, total_steps + 1):
            step_dt = min(dt_s, duration_s - time_s)
            state, acceleration = velocity_verlet_step_generic(
                state,
                acceleration,
                step_dt,
                accel_func,
                accel_kwargs=accel_kwargs,
                G=G_SI,
            )
            time_s += step_dt
            min_tracker.update(time_s, state.positions)

            if step_index % record_every_steps == 0 or step_index == total_steps:
                final_invariant_row = write_snapshot(
                    time_s,
                    state,
                    body_names,
                    outputs,
                    reference,
                    extrema,
                    sun_index=sun_index,
                )
                n_records += 1
                if n_records % 50 == 0:
                    outputs.flush()

            progress.update(1)
    finally:
        progress.close()

    if final_invariant_row is None:
        final_invariant_row = invariant_diagnostics_row(time_s, state, reference)
        update_extrema(extrema, final_invariant_row)

    return IntegrationResult(
        final_state=state,
        actual_duration_s=time_s,
        n_steps=total_steps,
        n_records=n_records,
        min_tracker=min_tracker,
        extrema=extrema,
        min_separation_sampling="each leapfrog integration step",
    )


def integrate_dop853_streaming(
    state0: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    *,
    duration_s: float,
    max_step_s: float,
    record_interval_s: float,
    accel_func,
    accel_kwargs: dict,
    show_progress: bool,
    sun_index: int,
) -> IntegrationResult:
    state = state0.copy()
    y_current = pack_state(state)
    reference = invariant_reference(state, G=G_SI)
    min_tracker = PairwiseMinimumTracker.create(body_names)
    min_tracker.update(0.0, state.positions)

    extrema = initial_extrema()
    write_snapshot(0.0, state, body_names, outputs, reference, extrema, sun_index=sun_index)

    n_records = 1
    current_t = 0.0
    next_record_t = min(record_interval_s, duration_s)

    progress = tqdm(
        total=seconds_to_years(duration_s),
        desc="DOP853 years",
        disable=not show_progress,
    )

    try:
        while current_t < duration_s - 1.0e-9:
            target_t = min(next_record_t, duration_s)
            if target_t <= current_t:
                target_t = min(current_t + record_interval_s, duration_s)

            sol = solve_ivp(
                fun=lambda t, y: rhs_solve_ivp(
                    t,
                    y,
                    masses=state0.masses,
                    accel_func=accel_func,
                    accel_kwargs=accel_kwargs,
                    G=G_SI,
                ),
                t_span=(current_t, target_t),
                y0=y_current,
                method="DOP853",
                t_eval=[target_t],
                rtol=1.0e-12,
                atol=1.0e-15,
                vectorized=False,
                max_step=max_step_s,
            )

            if not sol.success:
                raise RuntimeError(f"DOP853 integration failed: {sol.message}")

            previous_t = current_t
            current_t = float(sol.t[-1])
            y_current = sol.y[:, -1].copy()
            state = unpack_state(y_current, state0.masses)
            min_tracker.update(current_t, state.positions)

            write_snapshot(
                current_t,
                state,
                body_names,
                outputs,
                reference,
                extrema,
                sun_index=sun_index,
            )
            n_records += 1
            if n_records % 20 == 0:
                outputs.flush()

            progress.update(seconds_to_years(current_t - previous_t))
            next_record_t += record_interval_s
    finally:
        progress.close()

    nominal_steps = int(math.ceil(duration_s / max_step_s))
    return IntegrationResult(
        final_state=state,
        actual_duration_s=current_t,
        n_steps=nominal_steps,
        n_records=n_records,
        min_tracker=min_tracker,
        extrema=extrema,
        min_separation_sampling="recorded DOP853 output samples",
    )


def write_min_separations(path: Path, tracker: PairwiseMinimumTracker) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MIN_SEPARATION_FIELDS)
        writer.writeheader()
        for row in tracker.rows():
            writer.writerow(row)


def make_summary(
    *,
    args: argparse.Namespace,
    tag: str,
    body_names: tuple[str, ...],
    paths: dict[str, Path],
    result: IntegrationResult,
    runtime_s: float,
) -> dict:
    return {
        "mode": MODE_DESCRIPTION,
        "scientific_boundary": {
            "uses_earth_moon_barycenter": True,
            "uses_explicit_earth_and_moon": False,
            "uses_empirical_lunar_calibration": False,
            "uses_american_ephemeris_apparent_geocentric_tropical_output": False,
        },
        "configuration": {
            "kernel_path": args.kernel_path,
            "start_date_utc": args.start_date.isoformat(),
            "duration_years_requested": args.duration_years,
            "duration_years_actual": seconds_to_years(result.actual_duration_s),
            "step_days": args.step_days,
            "record_every_years": args.record_every_years,
            "include_pluto": args.include_pluto,
            "gr_model": args.gr_model,
            "integrator": args.integrator,
            "tag": tag,
            "body_names": body_names,
            "orbital_elements_reference_plane": "ecliptic_j2000",
        },
        "integrator_notes": {
            "leapfrog_default_for_long_term": True,
            "dop853_role": "short validation/comparison runs",
            "gr_leapfrog_symplectic_note": (
                "Sun 1PN GR is included through the acceleration callback; "
                "with gr_model='sun' the leapfrog update is not exactly symplectic."
                if args.integrator == "leapfrog" and args.gr_model == "sun"
                else None
            ),
        },
        "counts": {
            "n_steps_or_nominal_max_steps": result.n_steps,
            "n_records": result.n_records,
        },
        "runtime": {
            "wall_clock_seconds": runtime_s,
            "wall_clock_minutes": runtime_s / 60.0,
        },
        "diagnostic_extrema_over_records": result.extrema,
        "min_separation_sampling": result.min_separation_sampling,
        "min_separations": result.min_tracker.rows(),
        "outputs": {key: str(path) for key, path in paths.items()},
    }


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    reject_empirical_lunar_args(parser, list(argv))
    args = parser.parse_args(argv)
    validate_args(parser, args)

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, tag)

    bodies = tuple(solar_system_body_list(include_pluto=args.include_pluto))
    if "earth" in bodies or "moon" in bodies:
        raise RuntimeError("Stability mode must not use explicit Earth or Moon bodies.")
    if "earth barycenter" not in bodies:
        raise RuntimeError("Stability mode requires the Earth-Moon barycenter.")

    sun_index = bodies.index("sun")
    accel_func, accel_kwargs = select_acceleration_model(
        args.gr_model,
        sun_index=sun_index,
    )

    print(f"[Stability] {MODE_DESCRIPTION}.", flush=True)
    print(
        "[Stability] American Ephemeris apparent/geocentric/tropical machinery is not used.",
        flush=True,
    )
    print("[Stability] Earth-Moon barycenter is included; explicit Earth and Moon are omitted.")
    print(f"[Stability] bodies: {bodies}")
    print(f"[Stability] integrator: {args.integrator}")
    print(f"[Stability] gr_model: {args.gr_model}")

    if args.integrator == "dop853":
        print(
            "[Stability] DOP853 selected for short validation/comparison runs; "
            "leapfrog remains the default long-term stability integrator.",
            flush=True,
        )
    if args.integrator == "leapfrog" and args.gr_model == "sun":
        print(
            "[Stability] Sun 1PN GR is included through the acceleration callback; "
            "the leapfrog method is no longer exactly symplectic.",
            flush=True,
        )

    config = EphemerisConfig(kernel_path=args.kernel_path)
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=config,
    )

    duration_s = args.duration_years * JULIAN_YEAR_S
    step_s = args.step_days * DAY_S
    record_interval_s = args.record_every_years * JULIAN_YEAR_S

    print(f"[Stability] duration: {args.duration_years:g} Julian years")
    print(f"[Stability] step: {args.step_days:g} days")
    print(f"[Stability] record cadence: {args.record_every_years:g} Julian years")
    print(f"[Stability] output directory: {output_dir}")

    outputs = open_csv_outputs(paths)
    start_wall = time.perf_counter()
    try:
        if args.integrator == "leapfrog":
            result = integrate_leapfrog_streaming(
                state0,
                bodies,
                outputs,
                duration_s=duration_s,
                dt_s=step_s,
                record_interval_s=record_interval_s,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                show_progress=not args.no_progress_bar,
                sun_index=sun_index,
            )
        elif args.integrator == "dop853":
            result = integrate_dop853_streaming(
                state0,
                bodies,
                outputs,
                duration_s=duration_s,
                max_step_s=step_s,
                record_interval_s=record_interval_s,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                show_progress=not args.no_progress_bar,
                sun_index=sun_index,
            )
        else:
            raise ValueError(f"Unsupported integrator: {args.integrator!r}")

        outputs.flush()
    finally:
        outputs.close()

    runtime_s = time.perf_counter() - start_wall
    write_min_separations(paths["min_separations"], result.min_tracker)
    summary = make_summary(
        args=args,
        tag=tag,
        body_names=bodies,
        paths=paths,
        result=result,
        runtime_s=runtime_s,
    )
    write_summary(paths["summary"], summary)

    print("[Stability] complete.")
    print(f"[Stability] wall-clock runtime: {runtime_s:.3f} s")
    print(f"[Stability] records written: {result.n_records}")
    for key, path in paths.items():
        print(f"[Stability] wrote {key}: {path}")


if __name__ == "__main__":
    main()
