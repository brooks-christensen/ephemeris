from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .long_term_stability_cli import (
    LYAPUNOV_BODY_NAME_MAP,
    LYAPUNOV_ALL_BODY_NAMES,
    build_lyapunov_config,
    build_fli_megno_samples,
    initial_extrema,
    integrate_leapfrog_streaming,
    lyapunov_summary_dict,
    make_summary,
    open_csv_outputs,
    open_lyapunov_outputs,
    output_paths,
    parse_start_datetime,
    plot_lyapunov_growth,
    run_frequency_map_analysis,
    sanitize_tag,
    select_acceleration_model,
    stability_body_list,
    write_fli_megno_outputs,
    write_min_separations,
    write_summary,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import DAY_S, JULIAN_YEAR_S


ENSEMBLE_FIELDS = [
    "member_id",
    "seed",
    "backend",
    "integrator",
    "perturbation_description",
    "duration_years",
    "step_days",
    "runtime_seconds",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "max_com_velocity_drift",
    "max_eccentricity_mercury",
    "max_eccentricity_venus",
    "max_eccentricity_earth",
    "max_eccentricity_mars",
    "min_pairwise_separation_au",
    "finite_time_lambda_1_per_year",
    "lyapunov_time_years",
    "fli",
    "megno_lite",
    "classification",
    "warnings",
]


@dataclass(frozen=True)
class MemberConfig:
    member_id: int
    seed: int
    ensemble_root: str
    tag: str
    kernel_path: str
    start_date_iso: str
    model_scope: str
    duration_years: float
    step_days: float
    record_every_years: float
    gr_model: str
    integrator: str
    ensemble_perturbation_m: float
    backend: str
    with_lyapunov: bool
    lyapunov_method: str
    lyapunov_body: str
    lyapunov_renorm_years: float
    with_fli: bool
    with_megno_lite: bool
    with_frequency_map: bool


def timestamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log_line(log_path: Path, message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with log_path.open("a") as file_obj:
        file_obj.write(line + "\n")


def _csv_has_nul(path: Path) -> bool:
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            if b"\x00" in chunk:
                return True
    return False


def _csv_parses(path: Path) -> bool:
    try:
        if _csv_has_nul(path):
            return False
        with path.open(newline="") as file_obj:
            for _ in csv.reader(file_obj):
                pass
        return True
    except (OSError, csv.Error, UnicodeDecodeError):
        return False


def _json_parses(path: Path) -> bool:
    try:
        with path.open() as file_obj:
            json.load(file_obj)
        return True
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def member_tag(root_tag: str, member_id: int) -> str:
    return f"{root_tag}_member_{member_id:03d}"


def member_dir(ensemble_root: Path, member_id: int) -> Path:
    return ensemble_root / f"member_{member_id:03d}"


def member_required_paths(
    config: MemberConfig,
) -> dict[str, Path]:
    directory = member_dir(Path(config.ensemble_root), config.member_id)
    tag = member_tag(config.tag, config.member_id)
    paths = output_paths(
        directory,
        tag,
        with_lyapunov=config.with_lyapunov,
        with_frequency_map=config.with_frequency_map,
        with_fli_megno=config.with_fli or config.with_megno_lite,
        model_scope=config.model_scope,
    )
    return paths


def validate_member_outputs(config: MemberConfig) -> tuple[bool, str]:
    paths = member_required_paths(config)
    required_json = [paths["summary"]]
    required_csv = [
        paths["stability_timeseries"],
        paths["orbital_elements"],
        paths["invariants"],
        paths["min_separations"],
    ]
    if config.with_lyapunov:
        required_json.append(paths["lyapunov_summary"])
        required_csv.append(paths["lyapunov"])
    if config.with_frequency_map:
        required_csv.append(paths["frequency_map"])
    if config.with_fli or config.with_megno_lite:
        required_json.append(paths["fli_megno_summary"])
        required_csv.append(paths["fli_megno"])

    for path in required_json:
        if not path.exists():
            return False, f"missing JSON {path}"
        if not _json_parses(path):
            return False, f"corrupt JSON {path}"
    for path in required_csv:
        if not path.exists():
            return False, f"missing CSV {path}"
        if not _csv_parses(path):
            return False, f"corrupt CSV or NUL bytes {path}"
    return True, "ok"


def backup_corrupt_member(config: MemberConfig, reason: str) -> Path | None:
    directory = member_dir(Path(config.ensemble_root), config.member_id)
    if not directory.exists():
        return None
    backup_root = Path(config.ensemble_root) / "corrupt_member_backup"
    backup_root.mkdir(parents=True, exist_ok=True)
    suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = backup_root / f"member_{config.member_id:03d}_{suffix}"
    destination.mkdir(parents=True, exist_ok=False)
    for child in directory.iterdir():
        shutil.move(str(child), str(destination / child.name))
    (destination / "corruption_reason.txt").write_text(reason + "\n")
    return destination


def selected_ensemble_indices(body_choice: str, body_names: tuple[str, ...]) -> tuple[int, ...]:
    if body_choice == "all":
        selected = tuple(name for name in LYAPUNOV_ALL_BODY_NAMES if name in body_names)
    else:
        selected = (LYAPUNOV_BODY_NAME_MAP[body_choice],)
    missing = [name for name in selected if name not in body_names]
    if missing:
        raise ValueError(f"Ensemble perturbation target missing from model: {missing}")
    return tuple(body_names.index(name) for name in selected)


def rtn_basis(state: NBodyState, *, body_index: int, sun_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial = state.positions[body_index] - state.positions[sun_index]
    velocity = state.velocities[body_index] - state.velocities[sun_index]
    r_norm = float(np.linalg.norm(radial))
    if r_norm == 0.0:
        raise ValueError("Cannot build RTN perturbation basis for zero radius.")
    r_hat = radial / r_norm
    normal = np.cross(radial, velocity)
    n_norm = float(np.linalg.norm(normal))
    if n_norm == 0.0:
        raise ValueError("Cannot build RTN perturbation basis for zero angular momentum.")
    n_hat = normal / n_norm
    t_hat = np.cross(n_hat, r_hat)
    return r_hat, t_hat, n_hat


def apply_ensemble_perturbation(
    state: NBodyState,
    body_names: tuple[str, ...],
    *,
    body_choice: str,
    perturbation_m: float,
    seed: int,
    sun_index: int,
) -> tuple[NBodyState, dict[str, Any], str]:
    indices = selected_ensemble_indices(body_choice, body_names)
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(len(indices), 3))
    coeff_norm = float(np.linalg.norm(coefficients))
    if coeff_norm == 0.0 or not math.isfinite(coeff_norm):
        coefficients = np.zeros_like(coefficients)
        coefficients[0, 0] = 1.0
        coeff_norm = 1.0
    coefficients = coefficients / coeff_norm

    deltas = np.zeros_like(state.positions)
    perturbations: dict[str, Any] = {}
    for row, index in enumerate(indices):
        r_hat, t_hat, n_hat = rtn_basis(state, body_index=index, sun_index=sun_index)
        radial_m, tangential_m, normal_m = (perturbation_m * coefficients[row]).tolist()
        delta = radial_m * r_hat + tangential_m * t_hat + normal_m * n_hat
        deltas[index] += delta
        name = body_names[index]
        perturbations[name] = {
            "radial_m": float(radial_m),
            "tangential_m": float(tangential_m),
            "normal_m": float(normal_m),
            "cartesian_m": [float(value) for value in delta],
            "norm_m": float(np.linalg.norm(delta)),
        }

    weighted_delta = np.sum(state.masses[:, np.newaxis] * deltas, axis=0)
    sun_mass = float(state.masses[sun_index])
    sun_compensation = -weighted_delta / sun_mass
    deltas[sun_index] += sun_compensation

    perturbed = state.copy()
    perturbed.positions += deltas
    perturbations["sun_barycenter_compensation_m"] = [float(value) for value in sun_compensation]
    description = "; ".join(
        f"{name}:R={values['radial_m']:.6g}m,T={values['tangential_m']:.6g}m,N={values['normal_m']:.6g}m"
        for name, values in perturbations.items()
        if isinstance(values, dict)
    )
    return perturbed, perturbations, description


def make_member_args(config: MemberConfig) -> SimpleNamespace:
    return SimpleNamespace(
        kernel_path=config.kernel_path,
        start_date=parse_start_datetime(config.start_date_iso),
        duration_years=config.duration_years,
        step_days=config.step_days,
        record_every_years=config.record_every_years,
        include_pluto=False,
        gr_model=config.gr_model,
        integrator=config.integrator,
        model_scope=config.model_scope,
        tag=member_tag(config.tag, config.member_id),
        with_lyapunov=config.with_lyapunov,
        lyapunov_body=config.lyapunov_body,
        lyapunov_perturbation_m=config.ensemble_perturbation_m,
        lyapunov_renorm_years=config.lyapunov_renorm_years,
        lyapunov_fit_start_years=min(config.lyapunov_renorm_years, 0.2 * config.duration_years),
        lyapunov_fit_end_years=config.duration_years,
        lyapunov_seed=config.seed,
        lyapunov_norm="scaled_phase_space",
        lyapunov_method=config.lyapunov_method,
        lyapunov_no_renorm=False,
        lyapunov_debug=True,
        with_fli=config.with_fli,
        with_megno_lite=config.with_megno_lite,
        fli_method="tangent",
        fli_record_every_renorm=True,
        megno_record_every_renorm=True,
        with_frequency_map=config.with_frequency_map,
        frequency_window_years=max(4.0 * config.record_every_years, 0.25 * config.duration_years),
        frequency_step_years=max(2.0 * config.record_every_years, 0.10 * config.duration_years),
        frequency_bodies="mercury,venus,earth,mars" if config.model_scope == "inner" else "all",
        frequency_min_samples=16,
    )


def run_member(config_dict: dict[str, Any]) -> dict[str, Any]:
    config = MemberConfig(**config_dict)
    directory = member_dir(Path(config.ensemble_root), config.member_id)
    directory.mkdir(parents=True, exist_ok=True)
    tag = member_tag(config.tag, config.member_id)
    member_log = directory / "member_run.log"
    start_wall = time.perf_counter()
    args = make_member_args(config)
    paths = member_required_paths(config)

    with member_log.open("a") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        print(f"[{timestamp()}] member {config.member_id:03d} start seed={config.seed}", flush=True)
        bodies = stability_body_list(config.model_scope, include_pluto=False)
        sun_index = bodies.index("sun")
        accel_func, accel_kwargs = select_acceleration_model(config.gr_model, sun_index=sun_index)
        state0 = initial_state_solar_system_barycentric(
            args.start_date,
            bodies=bodies,
            config=EphemerisConfig(kernel_path=config.kernel_path),
        )
        state0, perturbation_vectors, perturbation_description = apply_ensemble_perturbation(
            state0,
            bodies,
            body_choice=config.lyapunov_body,
            perturbation_m=config.ensemble_perturbation_m,
            seed=config.seed,
            sun_index=sun_index,
        )

        lyapunov_config = None
        lyapunov_initial_state = None
        tangent_diagnostics = config.with_lyapunov or config.with_fli or config.with_megno_lite
        if tangent_diagnostics:
            lyapunov_config, lyapunov_initial_state = build_lyapunov_config(
                args,
                state0,
                bodies,
                sun_index=sun_index,
            )

        outputs = open_csv_outputs(paths)
        lyapunov_outputs = open_lyapunov_outputs(paths) if config.with_lyapunov else None
        try:
            result = integrate_leapfrog_streaming(
                state0,
                bodies,
                outputs,
                duration_s=config.duration_years * JULIAN_YEAR_S,
                dt_s=config.step_days * DAY_S,
                record_interval_s=config.record_every_years * JULIAN_YEAR_S,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                show_progress=False,
                sun_index=sun_index,
                lyapunov_config=lyapunov_config,
                lyapunov_outputs=lyapunov_outputs,
                lyapunov_initial_state=lyapunov_initial_state,
            )
            outputs.flush()
            if lyapunov_outputs is not None:
                lyapunov_outputs.flush()
        finally:
            outputs.close()
            if lyapunov_outputs is not None:
                lyapunov_outputs.close()

        runtime_s = time.perf_counter() - start_wall
        write_min_separations(paths["min_separations"], result.min_tracker)
        if result.lyapunov_result is not None and lyapunov_outputs is not None:
            plot_lyapunov_growth(
                result.lyapunov_result.samples,
                result.lyapunov_result.fit,
                lyapunov_outputs.plot_path,
            )
            write_summary(
                lyapunov_outputs.summary_path,
                lyapunov_summary_dict(
                    args=args,
                    tag=tag,
                    body_names=bodies,
                    result=result.lyapunov_result,
                    outputs=lyapunov_outputs,
                    runtime_s=runtime_s,
                ),
            )

        fli_megno_summary = None
        if (config.with_fli or config.with_megno_lite) and result.lyapunov_result is not None:
            fli_samples = build_fli_megno_samples(
                result.lyapunov_result,
                model_scope=config.model_scope,
            )
            fli_megno_summary = write_fli_megno_outputs(
                csv_path=paths["fli_megno"],
                summary_path=paths["fli_megno_summary"],
                samples=fli_samples,
                lyapunov_result=result.lyapunov_result,
                args=args,
                runtime_s=runtime_s,
            )

        frequency_summary = None
        if config.with_frequency_map:
            frequency_summary = run_frequency_map_analysis(
                orbital_elements_path=paths["orbital_elements"],
                output_csv_path=paths["frequency_map"],
                output_dir=directory,
                tag=tag,
                body_names=bodies,
                args=args,
            )

        summary = make_summary(
            args=args,
            tag=tag,
            body_names=bodies,
            paths=paths,
            result=result,
            runtime_s=runtime_s,
        )
        summary["ensemble_member"] = {
            "member_id": config.member_id,
            "seed": config.seed,
            "ensemble_perturbation_m": config.ensemble_perturbation_m,
            "perturbation_vectors": perturbation_vectors,
            "perturbation_description": perturbation_description,
        }
        if frequency_summary is not None:
            summary["frequency_map"] = frequency_summary
        if fli_megno_summary is not None:
            summary["fli_megno_lite"]["final_fli"] = fli_megno_summary.get("final_fli")
            summary["fli_megno_lite"]["final_megno_lite"] = fli_megno_summary.get("final_megno_lite")
            summary["fli_megno_lite"]["final_finite_time_lambda"] = fli_megno_summary.get("final_finite_time_lambda")
        write_summary(paths["summary"], summary)
        print(f"[{timestamp()}] member {config.member_id:03d} complete runtime_s={runtime_s:.3f}", flush=True)

    return member_summary_row(config, status="completed")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def max_eccentricities(orbital_elements_path: Path) -> dict[str, float]:
    keys = {
        "mercury barycenter": "max_eccentricity_mercury",
        "venus barycenter": "max_eccentricity_venus",
        "earth barycenter": "max_eccentricity_earth",
        "mars barycenter": "max_eccentricity_mars",
    }
    maxima = {value: math.nan for value in keys.values()}
    if not orbital_elements_path.exists():
        return maxima
    with orbital_elements_path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            key = keys.get(row.get("body", ""))
            if key is None:
                continue
            try:
                maxima[key] = max(
                    0.0 if not math.isfinite(maxima[key]) else maxima[key],
                    float(row["e"]),
                )
            except (KeyError, TypeError, ValueError):
                pass
    return maxima


def min_pairwise_separation(path: Path) -> float:
    if not path.exists():
        return math.nan
    minimum = math.inf
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                minimum = min(minimum, float(row["min_separation_au"]))
            except (KeyError, TypeError, ValueError):
                pass
    return minimum if math.isfinite(minimum) else math.nan


def member_summary_row(config: MemberConfig, *, status: str = "") -> dict[str, Any]:
    paths = member_required_paths(config)
    summary = read_json(paths["summary"])
    runtime = summary.get("runtime", {})
    extrema = summary.get("diagnostic_extrema_over_records", {})
    ensemble_member = summary.get("ensemble_member", {})
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(extrema, dict):
        extrema = {}
    if not isinstance(ensemble_member, dict):
        ensemble_member = {}

    lyapunov_summary = read_json(paths["lyapunov_summary"]) if config.with_lyapunov else {}
    fit = lyapunov_summary.get("fit", {}) if isinstance(lyapunov_summary, dict) else {}
    warnings = lyapunov_summary.get("warnings", []) if isinstance(lyapunov_summary, dict) else []
    fli_summary = (
        read_json(paths["fli_megno_summary"])
        if (config.with_fli or config.with_megno_lite) and paths["fli_megno_summary"].exists()
        else {}
    )
    eccentricities = max_eccentricities(paths["orbital_elements"])
    row = {
        "member_id": config.member_id,
        "seed": config.seed,
        "backend": config.backend,
        "integrator": config.integrator,
        "perturbation_description": ensemble_member.get("perturbation_description", ""),
        "duration_years": config.duration_years,
        "step_days": config.step_days,
        "runtime_seconds": runtime.get("wall_clock_seconds", ""),
        "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
        "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
        "max_com_velocity_drift": extrema.get("max_com_velocity_drift_au_per_year", ""),
        "min_pairwise_separation_au": min_pairwise_separation(paths["min_separations"]),
        "finite_time_lambda_1_per_year": fit.get("lambda_1_per_year", "") if isinstance(fit, dict) else "",
        "lyapunov_time_years": fit.get("lyapunov_time_years", "") if isinstance(fit, dict) else "",
        "fli": fli_summary.get("final_fli", "") if isinstance(fli_summary, dict) else "",
        "megno_lite": fli_summary.get("final_megno_lite", "") if isinstance(fli_summary, dict) else "",
        "classification": summary.get("classification", ""),
        "warnings": "; ".join(str(item) for item in warnings[:4]) if isinstance(warnings, list) else "",
    }
    row.update(eccentricities)
    return row


def write_ensemble_outputs(root: Path, tag: str, rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[Path, Path]:
    csv_path = root / f"ensemble_summary_{tag}.csv"
    json_path = root / f"ensemble_summary_{tag}.json"
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ENSEMBLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "mode": "stability ensemble: physical reduced model, no empirical lunar calibration",
        "caution": (
            "Ensemble spread and finite-time diagnostics are workflow/statistical checks. "
            "They are not an asymptotic Solar System Lyapunov exponent without duration scaling."
        ),
        "configuration": config,
        "members": rows,
    }
    with json_path.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    return csv_path, json_path


def estimate_member_runtime_seconds(duration_years: float, step_days: float) -> float:
    return 141.0 * (duration_years / 1000.0) * (0.25 / step_days)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU-parallel stability-mode ensembles across independent initial perturbations."
    )
    parser.add_argument("--kernel-path", default="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument("--model-scope", choices=["inner", "full"], default="inner")
    parser.add_argument("--duration-years", type=float, default=1000.0)
    parser.add_argument("--step-days", type=float, default=0.25)
    parser.add_argument("--record-every-years", type=float, default=10.0)
    parser.add_argument("--gr-model", choices=["none"], default="none")
    parser.add_argument("--integrator", choices=["leapfrog"], default="leapfrog")
    parser.add_argument("--backend", choices=["inhouse", "rebound"], default="inhouse")
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--ensemble-perturbation-m", type=float, default=1000.0)
    parser.add_argument("--ensemble-seed", type=int, default=12345)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability")
    parser.add_argument("--tag", default="ensemble")
    parser.add_argument("--with-lyapunov", action="store_true")
    parser.add_argument("--lyapunov-method", choices=["tangent"], default="tangent")
    parser.add_argument("--lyapunov-body", choices=["mercury", "all"], default="mercury")
    parser.add_argument("--lyapunov-renorm-years", type=float, default=0.25)
    parser.add_argument("--with-fli", action="store_true")
    parser.add_argument("--with-megno-lite", action="store_true")
    parser.add_argument("--with-frequency-map", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.duration_years <= 0.0:
        parser.error("--duration-years must be positive.")
    if args.step_days <= 0.0:
        parser.error("--step-days must be positive.")
    if args.record_every_years <= 0.0:
        parser.error("--record-every-years must be positive.")
    if args.ensemble_size <= 0:
        parser.error("--ensemble-size must be positive.")
    if args.ensemble_perturbation_m <= 0.0:
        parser.error("--ensemble-perturbation-m must be positive.")
    if args.lyapunov_renorm_years <= 0.0:
        parser.error("--lyapunov-renorm-years must be positive.")
    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be positive.")
    if (args.with_fli or args.with_megno_lite) and not args.with_lyapunov:
        parser.error("--with-fli/--with-megno-lite require --with-lyapunov in ensemble mode.")
    if args.backend == "rebound":
        parser.error(
            "--backend rebound is reserved for future ensemble smoke runs. "
            "Use mini_ephemeris.rebound_validation_cli for optional REBOUND validation now."
        )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    tag = sanitize_tag(args.tag)
    root = Path(args.output_dir) / tag
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / f"ensemble_run_{tag}.log"
    workers = args.workers if args.workers is not None else min(6, args.ensemble_size)
    total = args.ensemble_size if args.max_cases is None else min(args.ensemble_size, args.max_cases)

    log_line(log_path, "stability ensemble mode: physical reduced model, no empirical lunar calibration")
    log_line(log_path, f"tag={tag} ensemble_size={args.ensemble_size} total_scheduled={total} workers={workers}")
    log_line(log_path, f"estimated_member_runtime_seconds={estimate_member_runtime_seconds(args.duration_years, args.step_days):.1f}")
    log_line(log_path, "finite-time diagnostics are not asymptotic Lyapunov exponent claims")

    configs = [
        MemberConfig(
            member_id=member_id,
            seed=args.ensemble_seed + member_id,
            ensemble_root=str(root),
            tag=tag,
            kernel_path=args.kernel_path,
            start_date_iso=args.start_date.isoformat(),
            model_scope=args.model_scope,
            duration_years=args.duration_years,
            step_days=args.step_days,
            record_every_years=args.record_every_years,
            gr_model=args.gr_model,
            integrator=args.integrator,
            ensemble_perturbation_m=args.ensemble_perturbation_m,
            backend=args.backend,
            with_lyapunov=args.with_lyapunov,
            lyapunov_method=args.lyapunov_method,
            lyapunov_body=args.lyapunov_body,
            lyapunov_renorm_years=args.lyapunov_renorm_years,
            with_fli=args.with_fli,
            with_megno_lite=args.with_megno_lite,
            with_frequency_map=args.with_frequency_map,
        )
        for member_id in range(total)
    ]

    rows: list[dict[str, Any]] = []
    to_run: list[MemberConfig] = []
    for config in configs:
        if args.resume:
            valid, reason = validate_member_outputs(config)
            if valid:
                log_line(log_path, f"member {config.member_id:03d} resume skip: valid outputs")
                rows.append(member_summary_row(config, status="skipped"))
                continue
            directory = member_dir(root, config.member_id)
            if directory.exists() and (member_required_paths(config)["summary"].exists() or any(directory.iterdir())):
                backup = backup_corrupt_member(config, reason)
                log_line(log_path, f"member {config.member_id:03d} invalid resume output: {reason}; backup={backup}")
        to_run.append(config)

    completed = len(rows)
    if to_run:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for config in to_run:
                log_line(log_path, f"member {config.member_id:03d} start seed={config.seed}")
                futures[executor.submit(run_member, asdict(config))] = config
            for future in as_completed(futures):
                config = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    log_line(log_path, f"member {config.member_id:03d} failed: {exc!r}")
                    raise
                rows.append(row)
                completed += 1
                runtime = row.get("runtime_seconds", "")
                log_line(
                    log_path,
                    f"member {config.member_id:03d} end runtime_seconds={runtime} completed={completed}/{total}",
                )

    rows.sort(key=lambda row: int(row["member_id"]))
    csv_path, json_path = write_ensemble_outputs(
        root,
        tag,
        rows,
        {
            "kernel_path": args.kernel_path,
            "start_date": args.start_date.isoformat(),
            "model_scope": args.model_scope,
            "duration_years": args.duration_years,
            "step_days": args.step_days,
            "record_every_years": args.record_every_years,
            "ensemble_size": args.ensemble_size,
            "scheduled_members": total,
            "ensemble_perturbation_m": args.ensemble_perturbation_m,
            "ensemble_seed": args.ensemble_seed,
            "workers": workers,
            "backend": args.backend,
            "integrator": args.integrator,
            "with_lyapunov": args.with_lyapunov,
            "lyapunov_body": args.lyapunov_body,
            "with_fli": args.with_fli,
            "with_megno_lite": args.with_megno_lite,
            "with_frequency_map": args.with_frequency_map,
        },
    )
    log_line(log_path, f"wrote {csv_path}")
    log_line(log_path, f"wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
