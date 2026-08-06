from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent import attach_gr_potential_tangent_force
from .gr_potential_tangent_c import CBackend, load_c_backend
from .long_term_stability_cli import (
    build_rebound_simulation,
    configure_rebound_simulationarchive,
    load_rebound_archive_snapshot,
    optional_import_module,
    parse_start_datetime,
    rebound_state_from_sim,
    sanitize_tag,
    stability_body_list,
)
from .m0_telemetry import (
    STATE_SAMPLE_FIELDS,
    STATE_SAMPLE_SCHEMA_VERSION,
    TelemetrySchemaError,
    gr_potential_energy,
    read_state_samples,
    state_sample_rows,
)
from .nbody import G_SI
from .orbital_elements import (
    ARCSEC_PER_RAD,
    DAY_S,
    JULIAN_YEAR_S,
    heliocentric_elements_for_state,
    seconds_to_years,
)
from .rebound_gr_tangent_cli import APSIDAL_DRIFT_DEFINITION, DIAGNOSTIC_DEFINITIONS, FIELDS
from .stability_diagnostics import (
    invariant_diagnostics_row,
    invariant_reference,
    total_angular_momentum_vector,
)


RUNNER_SCHEMA_VERSION = 2
MAX_VALIDATION_DURATION_YEARS = 100_000.0
INTENTIONAL_INCOMPLETE_EXIT = 75
PROGRESS_FIELDS = [
    *FIELDS,
    "target_time_years",
    "time_seconds",
    "model_id",
    "runner_schema_version",
    "state_sample_schema_version",
    "configuration_fingerprint",
    "newtonian_energy_j",
    "gr_potential_energy_j",
    "corrected_energy_j",
    "corrected_energy_rel_change",
    "angular_momentum_x_kg_m2_s",
    "angular_momentum_y_kg_m2_s",
    "angular_momentum_z_kg_m2_s",
    "angular_momentum_norm_kg_m2_s",
    "nonfinite_result_count",
]


class RunnerSafetyError(RuntimeError):
    pass


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return {"git_commit": commit, "git_dirty": bool(dirty)}
    except Exception as exc:
        return {"git_commit": None, "git_dirty": None, "git_error": str(exc)}


def initial_condition_hash(state: Any, bodies: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(bodies), separators=(",", ":")).encode())
    for array in (state.positions, state.velocities, state.masses):
        digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


def output_paths(output_dir: Path, tag: str, archive_arg: str | None) -> dict[str, Path]:
    archive = Path(archive_arg) if archive_arg else output_dir / f"gr_tangent_archive_{tag}.bin"
    return {
        "progress": output_dir / f"gr_tangent_progress_{tag}.csv",
        "state": output_dir / f"gr_tangent_state_{tag}.csv",
        "status": output_dir / f"gr_tangent_status_{tag}.json",
        "summary": output_dir / f"gr_tangent_summary_{tag}.json",
        "restart": output_dir / f"gr_tangent_restart_{tag}.json",
        "archive": archive,
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        raise RunnerSafetyError(f"Unreadable {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunnerSafetyError(f"Invalid {label} {path}: expected a JSON object.")
    return payload


def apply_output_policy(
    paths: dict[str, Path],
    *,
    skip_if_complete: bool,
    overwrite_existing_output: bool,
    resume: bool,
) -> str:
    if sum(bool(value) for value in (skip_if_complete, overwrite_existing_output, resume)) > 1:
        raise RunnerSafetyError(
            "Use only one of --skip-if-complete, --overwrite-existing-output, or --resume."
        )
    summary = paths["summary"]
    existing = [path for path in paths.values() if path.exists()]
    if skip_if_complete:
        if not summary.exists():
            raise RunnerSafetyError(
                f"--skip-if-complete requires a valid final summary, but none exists: {summary}"
            )
        payload = _load_json_object(summary, "summary")
        if payload.get("complete") is not True or payload.get("status") != "COMPLETED":
            raise RunnerSafetyError(f"Summary is not a valid completed run: {summary}")
        if payload.get("schema_version") != RUNNER_SCHEMA_VERSION:
            raise RunnerSafetyError(f"Summary runner schema is incompatible: {summary}")
        if payload.get("state_sample_schema_version") != STATE_SAMPLE_SCHEMA_VERSION:
            raise RunnerSafetyError(f"Summary state sample schema is incompatible: {summary}")
        required = [
            paths[name] for name in ("progress", "state", "status", "restart", "archive")
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise RunnerSafetyError(
                "Completed summary has missing output artifacts: "
                + ", ".join(str(path) for path in missing)
            )
        return "skip"
    if resume:
        required = [paths["archive"], paths["restart"], paths["progress"], paths["state"]]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise RunnerSafetyError(
                "--resume requires archive, restart sidecar, progress CSV, and state CSV; missing: "
                + ", ".join(str(path) for path in missing)
            )
        if summary.exists():
            raise RunnerSafetyError(
                f"Refusing resume because a final summary exists: {summary}. Use --skip-if-complete."
            )
        return "resume"
    if overwrite_existing_output:
        if existing:
            print("[gr-tangent-backend] --overwrite-existing-output removes only these exact files:")
            for path in existing:
                print(f"[gr-tangent-backend] remove {path}")
            for path in existing:
                path.unlink()
        return "fresh"
    if existing:
        raise RunnerSafetyError(
            "Tagged output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing)
        )
    return "fresh"


def python_callback_stats(sim: Any) -> dict[str, int | float | None]:
    raw = getattr(sim, "_mini_ephemeris_gr_potential_tangent_stats", {}) or {}
    real_count = int(raw.get("real_gr_accel_norm_count", 0))
    tangent_count = int(raw.get("tangent_gr_accel_norm_count", 0))
    return {
        "callback_invocations": int(raw.get("callback_invocations", 0)),
        "real_gr_accel_norm_max": float(raw.get("real_gr_accel_norm_max", 0.0)),
        "real_gr_accel_norm_sum": float(raw.get("real_gr_accel_norm_sum", 0.0)),
        "real_gr_accel_norm_count": real_count,
        "real_gr_accel_norm_mean": (
            float(raw.get("real_gr_accel_norm_sum", 0.0)) / real_count if real_count else None
        ),
        "tangent_gr_accel_norm_max": float(raw.get("tangent_gr_accel_norm_max", 0.0)),
        "tangent_gr_accel_norm_sum": float(raw.get("tangent_gr_accel_norm_sum", 0.0)),
        "tangent_gr_accel_norm_count": tangent_count,
        "tangent_gr_accel_norm_mean": (
            float(raw.get("tangent_gr_accel_norm_sum", 0.0)) / tangent_count
            if tangent_count
            else None
        ),
        "nonfinite_result_count": 0,
    }


def cumulative_stats(
    current: dict[str, int | float | None], baseline: dict[str, Any]
) -> dict[str, int | float | None]:
    output: dict[str, int | float | None] = {}
    for name in ("callback_invocations", "real_gr_accel_norm_count", "tangent_gr_accel_norm_count"):
        output[name] = int(current.get(name, 0) or 0) + int(baseline.get(name, 0) or 0)
    for name in ("real_gr_accel_norm_sum", "tangent_gr_accel_norm_sum"):
        output[name] = float(current.get(name, 0.0) or 0.0) + float(baseline.get(name, 0.0) or 0.0)
    for name in ("real_gr_accel_norm_max", "tangent_gr_accel_norm_max"):
        output[name] = max(float(current.get(name, 0.0) or 0.0), float(baseline.get(name, 0.0) or 0.0))
    output["nonfinite_result_count"] = int(current.get("nonfinite_result_count", 0) or 0) + int(
        baseline.get("nonfinite_result_count", 0) or 0
    )
    real_count = int(output["real_gr_accel_norm_count"] or 0)
    tangent_count = int(output["tangent_gr_accel_norm_count"] or 0)
    output["real_gr_accel_norm_mean"] = (
        float(output["real_gr_accel_norm_sum"] or 0.0) / real_count if real_count else None
    )
    output["tangent_gr_accel_norm_mean"] = (
        float(output["tangent_gr_accel_norm_sum"] or 0.0) / tangent_count
        if tangent_count
        else None
    )
    return output


def _read_progress(
    path: Path, *, configuration_fingerprint: str
) -> list[dict[str, str]]:
    if b"\x00" in path.read_bytes():
        raise RunnerSafetyError(f"NUL byte in progress CSV: {path}")
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PROGRESS_FIELDS:
                raise RunnerSafetyError(
                    f"Progress CSV schema mismatch in {path}: {reader.fieldnames}"
                )
            rows = list(reader)
    except (csv.Error, OSError) as exc:
        raise RunnerSafetyError(f"Unreadable progress CSV {path}: {exc}") from exc
    previous = -math.inf
    for index, row in enumerate(rows, start=2):
        try:
            current = float(row["time_years"])
        except (TypeError, ValueError, KeyError) as exc:
            raise RunnerSafetyError(f"Malformed time on CSV row {index}: {path}") from exc
        if not math.isfinite(current) or current <= previous:
            raise RunnerSafetyError(f"Non-monotonic progress CSV at row {index}: {path}")
        if row.get("configuration_fingerprint") != configuration_fingerprint:
            raise RunnerSafetyError(
                f"Progress CSV configuration fingerprint mismatch at row {index}: {path}"
            )
        if row.get("runner_schema_version") != str(RUNNER_SCHEMA_VERSION):
            raise RunnerSafetyError(
                f"Progress CSV runner schema mismatch at row {index}: {path}"
            )
        if row.get("state_sample_schema_version") != str(
            STATE_SAMPLE_SCHEMA_VERSION
        ):
            raise RunnerSafetyError(
                f"Progress CSV state schema mismatch at row {index}: {path}"
            )
        previous = current
    return rows


def _rewrite_progress(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rewrite_state_samples(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATE_SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _float_values(rows: list[dict[str, str]], name: str) -> list[float]:
    output = []
    for row in rows:
        try:
            value = float(row.get(name, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output


def _variation_metadata(sim: Any) -> dict[str, Any]:
    n_real = int(sim.N_real)
    n_total = int(sim.N)
    return {
        "n_real": n_real,
        "n_total": n_total,
        "n_var": int(sim.N_var),
        "n_var_config": int(sim.N_var_config),
        "variation_particle_count": n_total - n_real,
    }


def _record_targets(duration_years: float, cadence_years: float) -> list[float]:
    count = int(math.floor(duration_years / cadence_years))
    targets = [index * cadence_years for index in range(count + 1)]
    if not math.isclose(targets[-1], duration_years, rel_tol=0.0, abs_tol=1.0e-12):
        targets.append(duration_years)
    return targets


def _config_payload(args: argparse.Namespace, bodies: list[str], hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
        "model_id": args.model_id,
        "backend": args.gr_tangent_backend,
        "kernel_sha256": hashes["kernel_sha256"],
        "initial_conditions_sha256": hashes["initial_conditions_sha256"],
        "artifact_sha256": hashes.get("artifact_sha256"),
        "c_source_sha256": hashes.get("c_source_sha256"),
        "start_date": args.start_date.isoformat(),
        "model_scope": args.model_scope,
        "body_names": bodies,
        "duration_years": args.duration_years,
        "step_days": args.step_days,
        "record_every_years": args.record_every_years,
        "archive_interval_years": args.archive_interval_years or args.record_every_years,
        "megno_seed": args.megno_seed,
        "gr_scale": args.gr_scale,
        "include_central_response": not args.no_central_response,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restart-safe REBOUND GR tangent runner with selectable Python/C backend."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument(
        "--model-scope",
        choices=["two_body_mercury", "inner", "full", "full_with_pluto"],
        default="two_body_mercury",
    )
    parser.add_argument("--duration-years", type=float, default=100.0)
    parser.add_argument(
        "--production-duration-approved",
        action="store_true",
        help=(
            "Explicitly authorize a production run longer than the 100 kyr validation cap. "
            "Validation manifests must never set this flag."
        ),
    )
    parser.add_argument("--step-days", type=float, default=1.0)
    parser.add_argument("--record-every-years", type=float, default=1.0)
    parser.add_argument("--archive-interval-years", type=float, default=None)
    parser.add_argument("--megno-seed", type=int, default=12345)
    parser.add_argument("--gr-scale", type=float, default=1.0)
    parser.add_argument("--no-central-response", action="store_true")
    parser.add_argument("--gr-tangent-backend", choices=["python", "c"], default="c")
    parser.add_argument("--c-artifact", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", default="output/stability/gr_tangent_c_port_validation_v1/manual")
    parser.add_argument("--tag", default="gr_tangent_c")
    parser.add_argument("--simulationarchive", default=None)
    parser.add_argument("--skip-if-complete", action="store_true")
    parser.add_argument("--overwrite-existing-output", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from a compatible C-lane archive and sidecar.")
    parser.add_argument(
        "--stop-after-years",
        type=float,
        default=None,
        help="Validation-only intentional incomplete stop; leaves restartable output and exits 75.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration_years <= 0.0:
        parser.error("--duration-years must be positive.")
    if (
        args.duration_years > MAX_VALIDATION_DURATION_YEARS
        and not args.production_duration_approved
    ):
        parser.error(
            f"--duration-years above {MAX_VALIDATION_DURATION_YEARS:g} requires "
            "--production-duration-approved."
        )
    if args.step_days <= 0.0 or args.record_every_years <= 0.0:
        parser.error("--step-days and --record-every-years must be positive.")
    archive_interval = args.archive_interval_years or args.record_every_years
    args.archive_interval_years = archive_interval
    if archive_interval <= 0.0:
        parser.error("--archive-interval-years must be positive.")
    ratio = archive_interval / args.record_every_years
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-12):
        parser.error("Archive cadence must be an integer multiple of record cadence for exact restart accounting.")
    if args.stop_after_years is not None and not (0.0 < args.stop_after_years < args.duration_years):
        parser.error("--stop-after-years must be positive and less than --duration-years.")

    rebound = optional_import_module("rebound")
    if rebound is None:
        raise RuntimeError("REBOUND is not installed.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    paths = output_paths(output_dir, tag, args.simulationarchive)
    try:
        action = apply_output_policy(
            paths,
            skip_if_complete=args.skip_if_complete,
            overwrite_existing_output=args.overwrite_existing_output,
            resume=args.resume,
        )
    except RunnerSafetyError as exc:
        parser.error(str(exc))
    if action == "skip":
        print(f"[gr-tangent-backend] valid completed summary exists; skipping {paths['summary']}")
        return 0

    bodies = stability_body_list(args.model_scope, include_pluto=args.model_scope == "full_with_pluto")
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    backend: CBackend | None = None
    hashes = {
        "kernel_sha256": sha256_file(Path(args.kernel_path)),
        "initial_conditions_sha256": initial_condition_hash(state0, bodies),
    }
    if args.gr_tangent_backend == "c":
        backend = load_c_backend(
            artifact_path=Path(args.c_artifact) if args.c_artifact else None
        )
        hashes.update(
            artifact_sha256=backend.build_metadata["artifact_sha256"],
            c_source_sha256=backend.build_metadata["source_sha256"],
        )
    config = _config_payload(args, bodies, hashes)
    fingerprint = canonical_hash(config)
    baseline_stats: dict[str, Any] = {}
    rows: list[dict[str, str]] = []
    state_row_count = 0
    resume_info: dict[str, Any] | None = None

    if action == "resume":
        try:
            sidecar = _load_json_object(paths["restart"], "restart sidecar")
        except RunnerSafetyError as exc:
            parser.error(str(exc))
        if sidecar.get("schema_version") != RUNNER_SCHEMA_VERSION:
            parser.error("Restart sidecar schema is incompatible.")
        if sidecar.get("state_sample_schema_version") != STATE_SAMPLE_SCHEMA_VERSION:
            parser.error("Restart state sample schema is incompatible.")
        if sidecar.get("state") != "incomplete":
            parser.error("Restart sidecar does not describe an incomplete run.")
        if sidecar.get("configuration_fingerprint") != fingerprint:
            parser.error("Restart configuration fingerprint mismatch; refusing continuation.")
        sim = load_rebound_archive_snapshot(rebound, paths["archive"])
        archive_years = seconds_to_years(float(sim.t))
        if archive_years <= 0.0 or archive_years >= args.duration_years:
            parser.error("Restart archive time must be positive and below the requested duration.")
        if int(sim.N_real) != len(bodies) or int(sim.N_var) <= 0:
            parser.error("Restart archive particle/variation layout is incompatible.")
        try:
            float(sim.megno())
            float(sim.lyapunov())
        except Exception as exc:
            parser.error(f"Restart archive did not preserve usable MEGNO/LCN state: {exc}")
        try:
            rows = _read_progress(
                paths["progress"], configuration_fingerprint=fingerprint
            )
            state_rows = read_state_samples(
                paths["state"],
                body_names=bodies,
                configuration_fingerprint=fingerprint,
            )
        except (RunnerSafetyError, TelemetrySchemaError) as exc:
            parser.error(str(exc))
        kept = [row for row in rows if float(row["time_years"]) <= archive_years + 1.0e-9]
        removed = len(rows) - len(kept)
        kept_state_rows = [
            row for row in state_rows if float(row["time_years"]) <= archive_years + 1.0e-9
        ]
        removed_state_rows = len(state_rows) - len(kept_state_rows)
        if not kept:
            parser.error("Restart progress CSV has no sample at or before the archive snapshot.")
        if not math.isclose(float(kept[-1]["time_years"]), archive_years, abs_tol=1.0e-8):
            parser.error(
                "Restart archive is not aligned with a diagnostic sample; exact counter continuity is unavailable."
            )
        if removed:
            _rewrite_progress(paths["progress"], kept)
        if removed_state_rows:
            _rewrite_state_samples(paths["state"], kept_state_rows)
        if len(kept_state_rows) != len(kept) * len(bodies):
            parser.error("Restart state samples do not match retained diagnostic samples.")
        if not kept_state_rows or not math.isclose(
            float(kept_state_rows[-1]["time_years"]), archive_years, abs_tol=1.0e-8
        ):
            parser.error("Restart state samples do not align with the archive snapshot.")
        rows = kept
        state_row_count = len(kept_state_rows)
        if sidecar.get("checkpoint_state_row_count") != state_row_count:
            parser.error("Restart sidecar state row count does not match retained telemetry.")
        baseline_stats = dict(sidecar.get("checkpoint_callback_stats", {}))
        if sidecar.get("checkpoint_time_years") != float(kept[-1]["time_years"]):
            parser.error("Restart sidecar checkpoint does not match retained progress.")
        resume_info = {
            "resumed": True,
            "archive_time_years": archive_years,
            "duplicate_or_future_rows_removed": removed,
            "duplicate_or_future_state_rows_removed": removed_state_rows,
            "fresh_process_callback_reattached": True,
        }
    else:
        sim = build_rebound_simulation(
            rebound,
            state0,
            integrator="whfast",
            step_s=args.step_days * DAY_S,
            ias15_epsilon=1.0e-10,
        )
        sim.init_megno(seed=args.megno_seed)
        sim.lyapunov()

    if args.gr_tangent_backend == "c":
        assert backend is not None
        backend.attach(
            sim,
            coefficient_scale=args.gr_scale,
            include_central_response=not args.no_central_response,
        )
        raw_stats = lambda: backend.stats(sim)
        hot_path = backend.hot_path_proof(sim)
    else:
        attach_gr_potential_tangent_force(
            sim,
            coefficient_scale=args.gr_scale,
            include_central_response=not args.no_central_response,
        )
        raw_stats = lambda: python_callback_stats(sim)
        hot_path = {
            "kind": "python_ctypes_callback",
            "addresses_match": False,
            "python_callback_in_force_path": True,
            "c_owned_instrumentation": False,
        }
    callbacks_at_attach = int(raw_stats().get("callback_invocations", 0) or 0)
    configure_rebound_simulationarchive(
        sim,
        paths["archive"],
        interval_s=archive_interval * JULIAN_YEAR_S,
        delete_existing=False,
    )

    reference = invariant_reference(state0, G=G_SI)
    sun_index = bodies.index("sun")
    corrected_energy_reference_j = reference.energy_j + gr_potential_energy(
        state0, sun_index=sun_index, coefficient_scale=args.gr_scale, gravitational_constant=G_SI
    )
    targets = _record_targets(args.duration_years, args.record_every_years)
    start_index = len(rows)
    if rows:
        last_time = float(rows[-1]["time_years"])
        start_index = next((index for index, value in enumerate(targets) if value > last_time + 1.0e-9), len(targets))
    progress_mode = "a" if rows else "w"
    start_wall = time.perf_counter()
    previous_progress: tuple[float, float] | None = None
    callback_after_reattach = False
    with (
        paths["progress"].open(progress_mode, newline="") as handle,
        paths["state"].open(progress_mode, newline="") as state_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
        state_writer = csv.DictWriter(state_handle, fieldnames=STATE_SAMPLE_FIELDS)
        if not rows:
            writer.writeheader()
            state_writer.writeheader()
        for output_index in range(start_index, len(targets)):
            target_years = targets[output_index]
            sim.integrate(target_years * JULIAN_YEAR_S, exact_finish_time=1)
            now = time.perf_counter()
            recent_rate = math.nan
            if previous_progress is not None:
                delta_years = target_years - previous_progress[0]
                delta_wall = now - previous_progress[1]
                if delta_years > 0.0 and delta_wall > 0.0:
                    recent_rate = delta_years / delta_wall
            previous_progress = (target_years, now)
            actual_time_seconds = float(sim.t)
            actual_time_years = seconds_to_years(actual_time_seconds)
            state = rebound_state_from_sim(sim, state0.masses)
            inv = invariant_diagnostics_row(actual_time_seconds, state, reference)
            gr_energy_j = gr_potential_energy(
                state,
                sun_index=sun_index,
                coefficient_scale=args.gr_scale,
                gravitational_constant=G_SI,
            )
            corrected_energy_j = float(inv["energy_j"]) + gr_energy_j
            corrected_energy_scale = (
                abs(corrected_energy_reference_j)
                if corrected_energy_reference_j != 0.0
                else 1.0
            )
            corrected_energy_rel_change = (
                corrected_energy_j - corrected_energy_reference_j
            ) / corrected_energy_scale
            angular_momentum = total_angular_momentum_vector(state)
            elements = heliocentric_elements_for_state(
                state, bodies, sun_index=sun_index
            )
            mercury = next((element for element in elements if element.body_name == "mercury barycenter"), None)
            try:
                megno = float(sim.megno())
                lcn = float(sim.lyapunov())
            except Exception:
                megno = math.nan
                lcn = math.nan
            stats = cumulative_stats(raw_stats(), baseline_stats)
            callback_after_reattach = callback_after_reattach or (
                int(raw_stats().get("callback_invocations", 0) or 0) > callbacks_at_attach
            )
            row = {
                "wall_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "wall_time_monotonic_seconds": now,
                "recent_throughput_years_per_wall_second": recent_rate if math.isfinite(recent_rate) else "",
                "worker_pid": os.getpid(),
                "time_years": actual_time_years,
                "megno": megno if math.isfinite(megno) else "",
                "lcn_1_per_year": lcn if math.isfinite(lcn) else "",
                "newtonian_energy_component_rel_change": float(inv["energy_rel_drift"]),
                "angular_momentum_rel_drift": float(inv["angular_momentum_rel_drift"]),
                "mercury_a_au": mercury.semi_major_axis_m / 149_597_870_700.0 if mercury else "",
                "mercury_e": mercury.eccentricity if mercury else "",
                "mercury_varpi_deg_unwrapped": math.degrees(mercury.longitude_perihelion_rad) if mercury else "",
                "callback_invocations": stats["callback_invocations"],
                "real_gr_accel_norm_max": stats["real_gr_accel_norm_max"],
                "real_gr_accel_norm_mean": stats["real_gr_accel_norm_mean"] if stats["real_gr_accel_norm_mean"] is not None else "",
                "tangent_gr_accel_norm_max": stats["tangent_gr_accel_norm_max"],
                "tangent_gr_accel_norm_mean": stats["tangent_gr_accel_norm_mean"] if stats["tangent_gr_accel_norm_mean"] is not None else "",
                "target_time_years": target_years,
                "time_seconds": actual_time_seconds,
                "model_id": args.model_id or "",
                "runner_schema_version": RUNNER_SCHEMA_VERSION,
                "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
                "configuration_fingerprint": fingerprint,
                "newtonian_energy_j": float(inv["energy_j"]),
                "gr_potential_energy_j": gr_energy_j,
                "corrected_energy_j": corrected_energy_j,
                "corrected_energy_rel_change": corrected_energy_rel_change,
                "angular_momentum_x_kg_m2_s": float(angular_momentum[0]),
                "angular_momentum_y_kg_m2_s": float(angular_momentum[1]),
                "angular_momentum_z_kg_m2_s": float(angular_momentum[2]),
                "angular_momentum_norm_kg_m2_s": float(inv["angular_momentum_norm_kg_m2_s"]),
                "nonfinite_result_count": stats["nonfinite_result_count"],
            }
            sample_rows = state_sample_rows(
                sim,
                bodies,
                sample_index=output_index,
                configuration_fingerprint=fingerprint,
            )
            state_writer.writerows(sample_rows)
            state_handle.flush()
            os.fsync(state_handle.fileno())
            state_row_count += len(sample_rows)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            rows.append({key: str(value) for key, value in row.items()})
            sidecar = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
                "state": "incomplete",
                "configuration_fingerprint": fingerprint,
                "configuration": config,
                "checkpoint_time_years": actual_time_years,
                "checkpoint_state_row_count": state_row_count,
                "output_index": output_index,
                "checkpoint_callback_stats": stats,
                "hot_path_proof": hot_path,
                "archive_path": str(paths["archive"]),
                "progress_path": str(paths["progress"]),
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "state_path": str(paths["state"]),
            }
            atomic_write_json(paths["restart"], sidecar)
            atomic_write_json(
                paths["status"],
                {
                    "state": "RUNNING",
                    "time_years": actual_time_years,
                    "corrected_energy_rel_change": corrected_energy_rel_change,
                    "percent_complete": 100.0 * target_years / args.duration_years,
                    "callback_stats": stats,
                    "worker_pid": os.getpid(),
                },
            )
            print(
                f"[gr-tangent-backend] {tag}: {target_years:.9g}/{args.duration_years:.9g} yr "
                f"callbacks={stats['callback_invocations']}",
                flush=True,
            )
            if args.stop_after_years is not None and target_years >= args.stop_after_years:
                print("[gr-tangent-backend] intentional incomplete stop; outputs remain restartable.")
                return INTENTIONAL_INCOMPLETE_EXIT

    elapsed = time.perf_counter() - start_wall
    final_stats = cumulative_stats(raw_stats(), baseline_stats)
    times = _float_values(rows, "time_years")
    varpis = np.radians(_float_values(rows, "mercury_varpi_deg_unwrapped"))
    if len(times) == len(varpis) and len(times) >= 2:
        varpis = np.unwrap(varpis)
        apsidal = float(np.polyfit(times, varpis, 1)[0] * ARCSEC_PER_RAD * 100.0)
    else:
        apsidal = None
    energies = _float_values(rows, "newtonian_energy_component_rel_change")
    corrected_energies = _float_values(rows, "corrected_energy_rel_change")
    max_newtonian_energy_error = max(
        (abs(value) for value in energies), default=None
    )
    max_corrected_energy_error = max(
        (abs(value) for value in corrected_energies), default=None
    )
    energy_improvement_factor = (
        max_newtonian_energy_error / max_corrected_energy_error
        if max_newtonian_energy_error is not None
        and max_corrected_energy_error is not None
        and max_corrected_energy_error > 0.0
        else None
    )
    corrected_energy_better_conserved = (
        max_newtonian_energy_error is not None
        and max_corrected_energy_error is not None
        and max_corrected_energy_error < max_newtonian_energy_error
    )
    angular = _float_values(rows, "angular_momentum_rel_drift")
    megno_values = _float_values(rows, "megno")
    lcn_values = _float_values(rows, "lcn_1_per_year")
    provenance = {
        **git_metadata(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "rebound_version": rebound.__version__,
        "rebound_build": getattr(rebound, "__build__", None),
        "rebound_githash": getattr(rebound, "__githash__", None),
        "command_line": [Path(sys.argv[0]).name, *(argv if argv is not None else sys.argv[1:])],
        "manifest_path": args.manifest_path,
        "manifest_sha256": sha256_file(Path(args.manifest_path)) if args.manifest_path else None,
        "hashes": hashes,
        "c_build": backend.build_metadata if backend is not None else None,
        "c_abi": backend.abi_metadata if backend is not None else None,
    }
    complete_resume = {
        **(resume_info or {"resumed": False}),
        "callbacks_increased_after_reattachment": callback_after_reattach if resume_info else None,
    }
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
        "status": "COMPLETED",
        "complete": True,
        "backend": args.gr_tangent_backend,
        "configuration": config,
        "configuration_fingerprint": fingerprint,
        "output_schemas": {
            "progress_fields": PROGRESS_FIELDS,
            "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
            "state_fields": STATE_SAMPLE_FIELDS,
            "float_serialization": "Python round-trip float64 decimal",
        },
        "energy_diagnostic": {
            "formula": "E_corrected = E_Newtonian - (3*s*G^2*M_sun^2/c^2)*sum_i(m_i/r_i_sun^2)",
            "units": "joules",
            "reference_epoch": args.start_date.isoformat(),
        },
        "production_metadata": _variation_metadata(sim),
        "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
        "apsidal_drift_definition": APSIDAL_DRIFT_DEFINITION,
        "hot_path_proof": hot_path,
        "restart": complete_resume,
        "diagnostics": {
            "runtime_seconds_this_process": elapsed,
            "rows_written_total": len(rows),
            "state_rows_written_total": state_row_count,
            "actual_time_years": times[-1] if times else None,
            "final_megno": megno_values[-1] if megno_values else None,
            "final_lcn_1_per_year": lcn_values[-1] if lcn_values else None,
            "max_newtonian_energy_component_rel_change": max_newtonian_energy_error,
            "max_corrected_energy_rel_change": max_corrected_energy_error,
            "corrected_energy_improvement_factor": energy_improvement_factor,
            "corrected_energy_better_conserved": corrected_energy_better_conserved,
            "max_angular_momentum_rel_drift": max((abs(value) for value in angular), default=None),
            "mercury_total_apsidal_drift_arcsec_per_century": apsidal,
            "callback_stats": final_stats,
        },
        "provenance": provenance,
        "outputs": {name: str(path) for name, path in paths.items()},
        "caveats": [
            "Finite-time tangent/MEGNO diagnostic; not an asymptotic Lyapunov proof.",
            "Corrected energy includes only the conservative potential represented by the validated gr_potential force.",
            "Full-system apsidal drift is total motion, not isolated relativistic excess.",
        ],
    }
    atomic_write_json(paths["summary"], summary)
    atomic_write_json(
        paths["restart"],
        {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "state": "complete",
            "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
            "configuration_fingerprint": fingerprint,
            "summary_path": str(paths["summary"]),
            "checkpoint_state_row_count": state_row_count,
        },
    )
    atomic_write_json(
        paths["status"],
        {
            "state": "COMPLETED",
            "time_years": times[-1],
            "percent_complete": 100.0,
            "callback_stats": final_stats,
            "worker_pid": os.getpid(),
        },
    )
    print(f"[gr-tangent-backend] wrote {paths['summary']}")
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


if __name__ == "__main__":
    main()
