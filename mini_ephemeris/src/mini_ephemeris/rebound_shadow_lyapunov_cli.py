from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .long_term_stability_cli import (
    add_reboundx_gr_force,
    build_rebound_simulation,
    rebound_state_from_sim,
    stability_body_list,
)
from .nbody import NBodyState
from .orbital_elements import AU_M, DAY_S, JULIAN_YEAR_S, heliocentric_elements_for_state


BODY_CHOICES = {
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "earth": "earth barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto": "pluto barycenter",
}
ELEMENT_BODIES = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto")
LEGACY_ELEMENT_BODIES = ("mercury", "venus", "earth", "mars", "jupiter")


def parse_start_datetime(text: str) -> dt.datetime:
    if "T" in text:
        value = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        value = dt.datetime.combine(dt.date.fromisoformat(text), dt.time(), tzinfo=dt.timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="REBOUND shadow-trajectory finite-time divergence diagnostic."
    )
    parser.add_argument("--kernel-path", default="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument("--model-scope", choices=["full", "full_with_pluto", "inner"], default="full")
    parser.add_argument("--integrator", choices=["whfast"], default="whfast")
    parser.add_argument("--gr-model", choices=["none", "gr_potential"], default="none")
    parser.add_argument("--duration-years", type=float, default=1000.0)
    parser.add_argument("--step-days", type=float, default=1.0)
    parser.add_argument("--record-every-years", type=float, default=10.0)
    parser.add_argument("--perturb-body", choices=[*BODY_CHOICES.keys(), "all"], default="mercury")
    parser.add_argument("--perturbation-m", type=float, default=1.0)
    parser.add_argument(
        "--perturbation-mode",
        choices=["radial", "tangential", "normal", "cartesian", "random"],
        default="radial",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fit-start-years", type=float, default=0.0)
    parser.add_argument("--fit-end-years", type=float, default=None)
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability")
    parser.add_argument("--tag", default="shadow")
    parser.add_argument("--resume", action="store_true", help="Skip if the summary JSON already exists.")
    parser.add_argument("--checkpoint-every-years", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None, help="'latest' or a checkpoint directory path.")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--write-partial-every-record", action="store_true")
    parser.add_argument("--stop-after-years", type=float, default=None)
    parser.add_argument("--inspect-checkpoints", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true", help="Accepted for script compatibility.")
    return parser


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "shadow"


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def shadow_config_hash(
    args: argparse.Namespace,
    *,
    tag: str,
    body_names: tuple[str, ...],
    fit_end_years: float,
) -> str:
    payload = {
        "kernel_path": str(args.kernel_path),
        "start_date": args.start_date.isoformat(),
        "model_scope": args.model_scope,
        "body_names": body_names,
        "integrator": args.integrator,
        "gr_model": args.gr_model,
        "duration_years": float(args.duration_years),
        "step_days": float(args.step_days),
        "record_every_years": float(args.record_every_years),
        "perturb_body": args.perturb_body,
        "perturbation_m": float(args.perturbation_m),
        "perturbation_mode": args.perturbation_mode,
        "seed": int(args.seed),
        "fit_start_years": float(args.fit_start_years),
        "fit_end_years": float(fit_end_years),
        "tag": tag,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_path(path: Path) -> None:
    try:
        with path.open("rb") as file_obj:
            os.fsync(file_obj.fileno())
    except OSError:
        return


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    tmp_path.replace(path)
    fsync_directory(path.parent)


def checkpoint_label(time_years: float) -> str:
    return f"{time_years:.6f}".rstrip("0").rstrip(".")


def default_checkpoint_dir(output_dir: Path, tag: str) -> Path:
    return output_dir / "shadow_checkpoints" / tag


def checkpoint_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [
            path for path in root.iterdir()
            if path.is_dir() and path.name.startswith("checkpoint_") and not path.name.endswith(".tmp")
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def validate_checkpoint_bundle(
    checkpoint_path: Path,
    rebound,
    *,
    expected_config_hash: str | None = None,
    load_archives: bool = True,
) -> tuple[bool, dict, list[str], object | None, object | None]:
    warnings: list[str] = []
    state_path = checkpoint_path / "checkpoint_state.json"
    ref_path = checkpoint_path / "reference.bin"
    shadow_path = checkpoint_path / "shadow.bin"
    if not state_path.exists():
        return False, {}, ["missing checkpoint_state.json"], None, None
    if not ref_path.exists():
        return False, {}, ["missing reference.bin"], None, None
    if not shadow_path.exists():
        return False, {}, ["missing shadow.bin"], None, None
    try:
        metadata = json.loads(state_path.read_text())
    except Exception as exc:
        return False, {}, [f"invalid checkpoint_state.json: {exc}"], None, None
    if expected_config_hash is not None and metadata.get("config_hash") != expected_config_hash:
        return False, metadata, ["config hash mismatch"], None, None
    if not load_archives:
        return True, metadata, warnings, None, None
    try:
        ref_sim = rebound.Simulation(str(ref_path))
    except Exception as exc:
        return False, metadata, [f"reference archive load failed: {exc}"], None, None
    try:
        shadow_sim = rebound.Simulation(str(shadow_path))
    except Exception as exc:
        return False, metadata, [f"shadow archive load failed: {exc}"], None, None
    time_tolerance_s = 60.0
    if abs(float(ref_sim.t) - float(shadow_sim.t)) > time_tolerance_s:
        return False, metadata, ["reference/shadow times disagree"], None, None
    expected_time_s = float(metadata.get("checkpoint_time_years", math.nan)) * JULIAN_YEAR_S
    if math.isfinite(expected_time_s) and abs(float(ref_sim.t) - expected_time_s) > time_tolerance_s:
        return False, metadata, ["archive time does not match checkpoint metadata"], None, None
    return True, metadata, warnings, ref_sim, shadow_sim


def inspect_checkpoints(root: Path, rebound, *, expected_config_hash: str | None = None) -> dict:
    valid: list[dict] = []
    invalid: list[dict] = []
    for checkpoint_path in checkpoint_directories(root):
        ok, metadata, warnings, _, _ = validate_checkpoint_bundle(
            checkpoint_path,
            rebound,
            expected_config_hash=expected_config_hash,
            load_archives=True,
        )
        entry = {
            "path": str(checkpoint_path),
            "checkpoint_time_years": metadata.get("checkpoint_time_years"),
            "warnings": warnings,
        }
        if ok:
            valid.append(entry)
        else:
            invalid.append(entry)
    latest = max(
        valid,
        key=lambda item: float(item.get("checkpoint_time_years") or -math.inf),
        default=None,
    )
    return {
        "checkpoint_dir": str(root),
        "valid": valid,
        "invalid": invalid,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "latest_valid_checkpoint": latest,
    }


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not math.isfinite(norm):
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def perturbation_direction(state: NBodyState, sun_index: int, body_index: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    r_vec = state.positions[body_index] - state.positions[sun_index]
    v_vec = state.velocities[body_index] - state.velocities[sun_index]
    radial = unit_vector(r_vec)
    normal = unit_vector(np.cross(r_vec, v_vec))
    tangential = unit_vector(np.cross(normal, radial))
    if mode == "radial":
        return radial
    if mode == "tangential":
        return tangential
    if mode == "normal":
        return normal
    if mode == "cartesian":
        return np.array([1.0, 0.0, 0.0])
    return unit_vector(rng.normal(size=3))


def make_shadow_state(
    state: NBodyState,
    body_names: tuple[str, ...],
    *,
    sun_index: int,
    perturb_body: str,
    perturbation_m: float,
    mode: str,
    seed: int,
) -> tuple[NBodyState, dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    shadow = state.copy()
    if perturb_body == "all":
        target_indices = [index for index, name in enumerate(body_names) if index != sun_index]
    else:
        target_name = BODY_CHOICES[perturb_body]
        target_indices = [body_names.index(target_name)]
    applied: dict[str, list[float]] = {}
    sun_mass = float(state.masses[sun_index])
    for index in target_indices:
        direction = perturbation_direction(state, sun_index, index, mode, rng)
        displacement = perturbation_m * direction
        shadow.positions[index] += displacement
        if sun_mass > 0.0:
            shadow.positions[sun_index] -= displacement * float(state.masses[index]) / sun_mass
        applied[body_names[index]] = [float(x) for x in displacement]
    return shadow, applied


def body_list_for_scope(scope: str) -> tuple[str, ...]:
    if scope == "full_with_pluto":
        return stability_body_list("full", include_pluto=True)
    return stability_body_list(scope, include_pluto=False)


def element_map(state: NBodyState, body_names: tuple[str, ...], sun_index: int) -> dict[str, object]:
    return {
        element.body_name: element
        for element in heliocentric_elements_for_state(state, body_names, sun_index=sun_index)
    }


def body_metric_slug(body_name: str) -> str:
    return body_name.replace(" ", "_")


def short_body_slug(body_name: str) -> str:
    return body_name.replace(" barycenter", "").replace(" ", "_")


def wrapped_angle_delta(shadow_angle: float, reference_angle: float) -> float:
    if not (math.isfinite(shadow_angle) and math.isfinite(reference_angle)):
        return math.nan
    return (shadow_angle - reference_angle + math.pi) % (2.0 * math.pi) - math.pi


def eccentricity_vector_separation(ref, shadow) -> float:
    ref_x = ref.eccentricity * math.cos(ref.longitude_perihelion_rad)
    ref_y = ref.eccentricity * math.sin(ref.longitude_perihelion_rad)
    shadow_x = shadow.eccentricity * math.cos(shadow.longitude_perihelion_rad)
    shadow_y = shadow.eccentricity * math.sin(shadow.longitude_perihelion_rad)
    return math.hypot(shadow_x - ref_x, shadow_y - ref_y)


def inclination_vector_separation(ref, shadow) -> float:
    ref_amp = math.sin(0.5 * ref.inclination_rad)
    shadow_amp = math.sin(0.5 * shadow.inclination_rad)
    ref_x = ref_amp * math.cos(ref.longitude_ascending_node_rad)
    ref_y = ref_amp * math.sin(ref.longitude_ascending_node_rad)
    shadow_x = shadow_amp * math.cos(shadow.longitude_ascending_node_rad)
    shadow_y = shadow_amp * math.sin(shadow.longitude_ascending_node_rad)
    return math.hypot(shadow_x - ref_x, shadow_y - ref_y)


def shadow_csv_fields(body_names: tuple[str, ...], sun_index: int) -> list[str]:
    base_fields = [
        "time_years",
        "raw_position_separation_au",
        "raw_velocity_separation_au_per_year",
        "log_separation",
        "finite_time_lambda_1_per_year",
        "finite_time_lyapunov_time_years",
    ]
    body_fields = []
    for index, name in enumerate(body_names):
        if index == sun_index:
            continue
        slug = short_body_slug(name)
        body_fields.append(f"sep_{slug}_au")
    element_fields = []
    for short_name in LEGACY_ELEMENT_BODIES:
        if BODY_CHOICES[short_name] in body_names:
            element_fields.extend(
                [
                    f"delta_{short_name}_a_au",
                    f"delta_{short_name}_e",
                    f"delta_{short_name}_varpi_arcsec",
                ]
            )
    for short_name in ELEMENT_BODIES:
        body = BODY_CHOICES[short_name]
        if body not in body_names:
            continue
        slug = body_metric_slug(body)
        element_fields.extend(
            [
                f"{slug}_delta_a_au",
                f"{slug}_delta_e",
                f"{slug}_delta_i",
                f"{slug}_delta_Omega_wrapped",
                f"{slug}_delta_varpi_wrapped",
                f"{slug}_delta_lambda_wrapped",
                f"{slug}_eccentricity_vector_separation",
                f"{slug}_inclination_vector_separation",
            ]
        )
    return base_fields + body_fields + element_fields


def backup_corrupt_csv(csv_path: Path, tag: str) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = csv_path.parent / "corrupt_shadow_backup" / f"{tag}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / csv_path.name
    shutil.move(str(csv_path), str(target))
    return target


def prepare_csv_for_resume(
    csv_path: Path,
    *,
    checkpoint_time_years: float | None,
    fieldnames: list[str],
    tag: str,
    can_recreate_from_checkpoint: bool,
) -> tuple[list[dict[str, str]], int, list[str]]:
    warnings: list[str] = []
    duplicate_rows_removed = 0
    if not csv_path.exists():
        warnings.append("existing shadow CSV missing; recreating from checkpoint/current state")
        return [], duplicate_rows_removed, warnings
    raw = csv_path.read_bytes()
    if b"\0" in raw:
        backup = backup_corrupt_csv(csv_path, tag)
        if not can_recreate_from_checkpoint:
            raise RuntimeError(f"shadow CSV contains NUL bytes and no checkpoint can recreate it; backed up to {backup}")
        warnings.append(f"shadow CSV contained NUL bytes; backed up to {backup}")
        return [], duplicate_rows_removed, warnings
    kept: list[dict[str, str]] = []
    seen_times: set[str] = set()
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None or "time_years" not in reader.fieldnames:
            raise ValueError("missing time_years header")
        for row in reader:
            time_text = row.get("time_years")
            if time_text in (None, ""):
                raise ValueError("row missing time_years")
            time_years = float(time_text)
            if checkpoint_time_years is not None and time_years > checkpoint_time_years + 1.0e-8:
                duplicate_rows_removed += 1
                continue
            time_key = f"{time_years:.12g}"
            if time_key in seen_times:
                duplicate_rows_removed += 1
                continue
            seen_times.add(time_key)
            kept.append({field: row.get(field, "") for field in fieldnames})
    except Exception as exc:
        backup = backup_corrupt_csv(csv_path, tag)
        if not can_recreate_from_checkpoint:
            raise RuntimeError(f"shadow CSV is malformed and no checkpoint can recreate it; backed up to {backup}: {exc}") from exc
        warnings.append(f"shadow CSV was malformed; backed up to {backup}: {exc}")
        return [], duplicate_rows_removed, warnings
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    return kept, duplicate_rows_removed, warnings


def open_shadow_csv_for_append(csv_path: Path, fieldnames: list[str], *, append: bool) -> tuple[object, csv.DictWriter]:
    file_obj = csv_path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
        file_obj.flush()
        os.fsync(file_obj.fileno())
    return file_obj, writer


def write_checkpoint_bundle(
    *,
    checkpoint_root: Path,
    tag: str,
    ref,
    shadow,
    args: argparse.Namespace,
    current_time_years: float,
    next_record_time_years: float,
    config_hash: str,
    csv_path: Path,
    applied_displacements: dict[str, list[float]],
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    label = checkpoint_label(current_time_years)
    final_dir = checkpoint_root / f"checkpoint_{tag}_{label}yr"
    tmp_dir = checkpoint_root / f"checkpoint_{tag}_{label}yr.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    ref_path = tmp_dir / "reference.bin"
    shadow_path = tmp_dir / "shadow.bin"
    ref.save_to_file(str(ref_path), delete_file=True)
    shadow.save_to_file(str(shadow_path), delete_file=True)
    fsync_path(ref_path)
    fsync_path(shadow_path)
    metadata = {
        "tag": tag,
        "model_scope": args.model_scope,
        "integrator": args.integrator,
        "gr_model": args.gr_model,
        "duration_years": args.duration_years,
        "step_days": args.step_days,
        "record_every_years": args.record_every_years,
        "checkpoint_time_years": current_time_years,
        "current_time_years": current_time_years,
        "next_record_time_years": next_record_time_years,
        "perturbation_metadata": {
            "perturb_body": args.perturb_body,
            "perturbation_m": args.perturbation_m,
            "perturbation_mode": args.perturbation_mode,
        },
        "perturbation_vector": applied_displacements,
        "perturb_body": args.perturb_body,
        "perturbation_m": args.perturbation_m,
        "seed": args.seed,
        "fit_start_years": args.fit_start_years,
        "fit_end_years": args.fit_end_years if args.fit_end_years is not None else args.duration_years,
        "config_hash": config_hash,
        "output_csv_path": str(csv_path),
        "reference_archive_path": str(final_dir / "reference.bin"),
        "shadow_archive_path": str(final_dir / "shadow.bin"),
        "created_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
    }
    atomic_write_json(tmp_dir / "checkpoint_state.json", metadata)
    fsync_directory(tmp_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    fsync_directory(checkpoint_root)
    return final_dir


def prune_checkpoints(
    checkpoint_root: Path,
    rebound,
    *,
    keep: int,
    expected_config_hash: str,
) -> None:
    if keep < 1:
        return
    valid: list[tuple[float, Path]] = []
    for checkpoint_path in checkpoint_directories(checkpoint_root):
        ok, metadata, _, _, _ = validate_checkpoint_bundle(
            checkpoint_path,
            rebound,
            expected_config_hash=expected_config_hash,
            load_archives=False,
        )
        if ok:
            valid.append((float(metadata.get("checkpoint_time_years", -math.inf)), checkpoint_path))
    valid.sort(reverse=True, key=lambda item: item[0])
    for _, checkpoint_path in valid[keep:]:
        shutil.rmtree(checkpoint_path)


def next_record_after(current_time_s: float, record_s: float) -> float:
    if current_time_s <= 1.0e-6:
        return 0.0
    multiple = math.floor(current_time_s / record_s)
    candidate = (multiple + 1) * record_s
    if candidate <= current_time_s + 1.0e-6:
        candidate += record_s
    return candidate


def add_scheduled_events(schedule: dict[float, set[str]], name: str, times: list[float]) -> None:
    for value in times:
        key = round(float(value), 6)
        schedule.setdefault(key, set()).add(name)


def angular_delta_arcsec(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    return wrapped_angle_delta(a, b) * 206_264.80624709636


def separation_row(
    time_years: float,
    ref_state: NBodyState,
    shadow_state: NBodyState,
    body_names: tuple[str, ...],
    sun_index: int,
) -> dict[str, float | str]:
    delta_r = shadow_state.positions - ref_state.positions
    delta_v = shadow_state.velocities - ref_state.velocities
    pos_norm_au = float(np.linalg.norm(delta_r) / AU_M)
    vel_norm_au_per_year = float(np.linalg.norm(delta_v) / AU_M * JULIAN_YEAR_S)
    log_sep = math.log(max(pos_norm_au, 1.0e-300))
    row: dict[str, float | str] = {
        "time_years": time_years,
        "raw_position_separation_au": pos_norm_au,
        "raw_velocity_separation_au_per_year": vel_norm_au_per_year,
        "log_separation": log_sep,
        "finite_time_lambda_1_per_year": "",
        "finite_time_lyapunov_time_years": "",
    }
    for index, name in enumerate(body_names):
        if index == sun_index:
            continue
        slug = short_body_slug(name)
        row[f"sep_{slug}_au"] = float(np.linalg.norm(delta_r[index]) / AU_M)
    ref_elements = element_map(ref_state, body_names, sun_index)
    shadow_elements = element_map(shadow_state, body_names, sun_index)
    for short_name in LEGACY_ELEMENT_BODIES:
        body = BODY_CHOICES[short_name]
        if body not in ref_elements or body not in shadow_elements:
            continue
        ref = ref_elements[body]
        shadow = shadow_elements[body]
        row[f"delta_{short_name}_a_au"] = (shadow.semi_major_axis_m - ref.semi_major_axis_m) / AU_M
        row[f"delta_{short_name}_e"] = shadow.eccentricity - ref.eccentricity
        row[f"delta_{short_name}_varpi_arcsec"] = angular_delta_arcsec(
            shadow.longitude_perihelion_rad,
            ref.longitude_perihelion_rad,
        )
    for short_name in ELEMENT_BODIES:
        body = BODY_CHOICES[short_name]
        if body not in ref_elements or body not in shadow_elements:
            continue
        ref = ref_elements[body]
        shadow = shadow_elements[body]
        slug = body_metric_slug(body)
        row[f"{slug}_delta_a_au"] = (shadow.semi_major_axis_m - ref.semi_major_axis_m) / AU_M
        row[f"{slug}_delta_e"] = shadow.eccentricity - ref.eccentricity
        row[f"{slug}_delta_i"] = shadow.inclination_rad - ref.inclination_rad
        row[f"{slug}_delta_Omega_wrapped"] = wrapped_angle_delta(
            shadow.longitude_ascending_node_rad,
            ref.longitude_ascending_node_rad,
        )
        row[f"{slug}_delta_varpi_wrapped"] = wrapped_angle_delta(
            shadow.longitude_perihelion_rad,
            ref.longitude_perihelion_rad,
        )
        row[f"{slug}_delta_lambda_wrapped"] = wrapped_angle_delta(
            shadow.mean_longitude_rad,
            ref.mean_longitude_rad,
        )
        row[f"{slug}_eccentricity_vector_separation"] = eccentricity_vector_separation(ref, shadow)
        row[f"{slug}_inclination_vector_separation"] = inclination_vector_separation(ref, shadow)
    return row


def linear_fit(rows: list[dict[str, float | str]], fit_start: float, fit_end: float) -> dict[str, float | str]:
    points = [
        (float(row["time_years"]), float(row["log_separation"]))
        for row in rows
        if fit_start <= float(row["time_years"]) <= fit_end
        and math.isfinite(float(row["log_separation"]))
    ]
    if len(points) < 2:
        return {"lambda_1_per_year": math.nan, "r_squared": math.nan, "warning": "fewer than two fit samples"}
    xs = np.array([item[0] for item in points], dtype=float)
    ys = np.array([item[1] for item in points], dtype=float)
    coeff = np.polyfit(xs, ys, 1)
    slope = float(coeff[0])
    pred = coeff[0] * xs + coeff[1]
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    ss_res = float(np.sum((ys - pred) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    warnings: list[str] = []
    if r2 < 0.8:
        warnings.append("log-separation fit is not strongly linear; window may be non-exponential")
    if max(float(row["raw_position_separation_au"]) for row in rows) > 0.1:
        warnings.append("separation exceeded 0.1 AU; late samples may be saturated")
    return {
        "lambda_1_per_year": slope,
        "r_squared": r2,
        "warning": " | ".join(warnings),
    }


def plot_rows(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [float(row["time_years"]) for row in rows],
        [float(row["log_separation"]) for row in rows],
        marker="o",
        markersize=2,
        linewidth=1.0,
    )
    ax.set_xlabel("time [years]")
    ax.set_ylabel("log raw position separation [AU]")
    ax.set_title("REBOUND shadow trajectory divergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import rebound
    except ImportError as exc:
        raise SystemExit("REBOUND is required for shadow-trajectory diagnostics.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)
    checkpoint_root = args.checkpoint_dir or default_checkpoint_dir(output_dir, tag)
    if args.inspect_checkpoints:
        if args.checkpoint_dir is None:
            parser.error("--inspect-checkpoints requires --checkpoint-dir.")
        inspection = inspect_checkpoints(args.checkpoint_dir, rebound)
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return

    if args.duration_years <= 0 or args.step_days <= 0 or args.record_every_years <= 0:
        parser.error("duration, step, and record cadence must be positive.")
    if args.checkpoint_every_years is not None and args.checkpoint_every_years <= 0.0:
        parser.error("--checkpoint-every-years must be positive.")
    if args.keep_checkpoints < 1:
        parser.error("--keep-checkpoints must be at least 1.")
    if args.stop_after_years is not None and args.stop_after_years <= 0.0:
        parser.error("--stop-after-years must be positive.")
    fit_end = args.fit_end_years if args.fit_end_years is not None else args.duration_years
    if fit_end <= args.fit_start_years:
        parser.error("--fit-end-years must be greater than --fit-start-years.")

    csv_path = output_dir / f"shadow_separation_{tag}.csv"
    summary_path = output_dir / f"shadow_lyapunov_summary_{tag}.json"
    plot_path = output_dir / f"shadow_growth_{tag}.png"
    if args.resume and summary_path.exists():
        print(f"RESUME=1 equivalent: summary exists, skipping {summary_path}")
        return

    body_names = body_list_for_scope(args.model_scope)
    sun_index = body_names.index("sun")
    config_hash = shadow_config_hash(args, tag=tag, body_names=body_names, fit_end_years=fit_end)
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=body_names,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    dt_s = args.step_days * DAY_S
    duration_s = args.duration_years * JULIAN_YEAR_S
    record_s = args.record_every_years * JULIAN_YEAR_S
    checkpoint_every_s = (
        args.checkpoint_every_years * JULIAN_YEAR_S
        if args.checkpoint_every_years is not None
        else None
    )

    checkpoint_warnings: list[str] = []
    resumed_from_checkpoint: str | None = None
    resumed_from_time_years = 0.0
    duplicate_rows_removed_on_resume = 0

    resume_metadata: dict | None = None
    if args.resume_from_checkpoint is not None:
        if args.resume_from_checkpoint == "latest":
            candidates = checkpoint_directories(checkpoint_root)
            if not candidates:
                checkpoint_warnings.append(
                    f"no checkpoints found in {checkpoint_root}; starting from scratch"
                )
            else:
                for candidate in candidates:
                    ok, metadata, warnings, ref_candidate, shadow_candidate = validate_checkpoint_bundle(
                        candidate,
                        rebound,
                        expected_config_hash=config_hash,
                        load_archives=True,
                    )
                    if ok:
                        ref = ref_candidate
                        shadow = shadow_candidate
                        resume_metadata = metadata
                        resumed_from_checkpoint = str(candidate)
                        resumed_from_time_years = float(metadata["checkpoint_time_years"])
                        checkpoint_warnings.extend(
                            f"ignored invalid checkpoint {candidate}: {'; '.join(warnings)}"
                            for warnings in ([warnings] if warnings else [])
                        )
                        break
                    checkpoint_warnings.append(
                        f"invalid checkpoint {candidate}: {'; '.join(warnings)}"
                    )
                else:
                    raise RuntimeError(
                        f"all checkpoints in {checkpoint_root} were invalid; refusing to restart from zero"
                    )
        else:
            checkpoint_path = Path(args.resume_from_checkpoint)
            ok, metadata, warnings, ref_candidate, shadow_candidate = validate_checkpoint_bundle(
                checkpoint_path,
                rebound,
                expected_config_hash=config_hash,
                load_archives=True,
            )
            if not ok:
                raise RuntimeError(f"checkpoint {checkpoint_path} is invalid: {'; '.join(warnings)}")
            ref = ref_candidate
            shadow = shadow_candidate
            resume_metadata = metadata
            resumed_from_checkpoint = str(checkpoint_path)
            resumed_from_time_years = float(metadata["checkpoint_time_years"])

    if resume_metadata is None:
        shadow0, applied = make_shadow_state(
            state0,
            body_names,
            sun_index=sun_index,
            perturb_body=args.perturb_body,
            perturbation_m=args.perturbation_m,
            mode=args.perturbation_mode,
            seed=args.seed,
        )
        ref = build_rebound_simulation(rebound, state0, integrator="whfast", step_s=dt_s, ias15_epsilon=1.0e-9)
        shadow = build_rebound_simulation(rebound, shadow0, integrator="whfast", step_s=dt_s, ias15_epsilon=1.0e-9)
        applied = applied
    else:
        applied = dict(resume_metadata.get("perturbation_vector", {}))
        ref.dt = dt_s
        shadow.dt = dt_s

    if args.gr_model != "none":
        add_reboundx_gr_force(ref, args.gr_model)
        add_reboundx_gr_force(shadow, args.gr_model)

    current_time_s = float(ref.t)
    fieldnames = shadow_csv_fields(body_names, sun_index)
    rows, duplicate_rows_removed_on_resume, csv_warnings = (
        prepare_csv_for_resume(
            csv_path,
            checkpoint_time_years=resumed_from_time_years,
            fieldnames=fieldnames,
            tag=tag,
            can_recreate_from_checkpoint=True,
        )
        if resume_metadata is not None and csv_path.exists()
        else ([], 0, [])
    )
    checkpoint_warnings.extend(csv_warnings)
    append_csv = bool(rows)
    csv_file, writer = open_shadow_csv_for_append(csv_path, fieldnames, append=append_csv)

    if not rows:
        ref_state = rebound_state_from_sim(ref, state0.masses)
        shadow_state = rebound_state_from_sim(shadow, state0.masses)
        checkpoint_row = separation_row(current_time_s / JULIAN_YEAR_S, ref_state, shadow_state, body_names, sun_index)
        rows.append(checkpoint_row)
        writer.writerow(checkpoint_row)
        csv_file.flush()
        os.fsync(csv_file.fileno())

    record_start_s = (
        float(resume_metadata.get("next_record_time_years", 0.0)) * JULIAN_YEAR_S
        if resume_metadata is not None
        else 0.0
    )
    if record_start_s <= current_time_s + 1.0e-6:
        record_start_s = next_record_after(current_time_s, record_s)

    event_schedule: dict[float, set[str]] = {}
    record_times: list[float] = []
    t = record_start_s
    while t <= duration_s + 1.0:
        if t > current_time_s + 1.0e-6:
            record_times.append(min(t, duration_s))
        t += record_s
    if not record_times or abs(record_times[-1] - duration_s) > 1.0:
        record_times.append(duration_s)
    add_scheduled_events(event_schedule, "record", sorted(set(record_times)))
    if checkpoint_every_s is not None:
        checkpoint_times: list[float] = []
        t = math.floor(current_time_s / checkpoint_every_s + 1.0) * checkpoint_every_s
        while t < duration_s - 1.0:
            if t > current_time_s + 1.0e-6:
                checkpoint_times.append(t)
            t += checkpoint_every_s
        add_scheduled_events(event_schedule, "checkpoint", checkpoint_times)
    stop_s = args.stop_after_years * JULIAN_YEAR_S if args.stop_after_years is not None else None
    if stop_s is not None and stop_s < duration_s and stop_s > current_time_s + 1.0e-6:
        add_scheduled_events(event_schedule, "stop", [stop_s])
    event_times = sorted(t for t in event_schedule if t > current_time_s + 1.0e-6)

    checkpoints_written = 0
    latest_checkpoint: Path | None = None
    start_wall = time.perf_counter()
    stopped_early = False
    try:
        for target in event_times:
            events = event_schedule[round(target, 6)]
            ref.integrate(float(target), exact_finish_time=1)
            shadow.integrate(float(target), exact_finish_time=1)
            current_time_s = float(ref.t)
            if "record" in events:
                ref_state = rebound_state_from_sim(ref, state0.masses)
                shadow_state = rebound_state_from_sim(shadow, state0.masses)
                row = separation_row(current_time_s / JULIAN_YEAR_S, ref_state, shadow_state, body_names, sun_index)
                rows.append(row)
                writer.writerow(row)
                if args.write_partial_every_record:
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
            if "checkpoint" in events or "stop" in events:
                latest_checkpoint = write_checkpoint_bundle(
                    checkpoint_root=checkpoint_root,
                    tag=tag,
                    ref=ref,
                    shadow=shadow,
                    args=args,
                    current_time_years=current_time_s / JULIAN_YEAR_S,
                    next_record_time_years=next_record_after(current_time_s, record_s) / JULIAN_YEAR_S,
                    config_hash=config_hash,
                    csv_path=csv_path,
                    applied_displacements=applied,
                )
                checkpoints_written += 1
                prune_checkpoints(
                    checkpoint_root,
                    rebound,
                    keep=args.keep_checkpoints,
                    expected_config_hash=config_hash,
                )
            if "stop" in events:
                stopped_early = True
                break
    finally:
        csv_file.flush()
        os.fsync(csv_file.fileno())
        csv_file.close()
    runtime = time.perf_counter() - start_wall

    if stopped_early:
        print(
            f"stopped after {current_time_s / JULIAN_YEAR_S:g} years; "
            f"latest checkpoint: {latest_checkpoint}"
        )
        return

    initial_log_separation = float(rows[0]["log_separation"]) if rows else math.nan
    for row in rows:
        time_years = float(row["time_years"])
        if time_years > 0.0 and math.isfinite(initial_log_separation):
            growth = float(row["log_separation"]) - initial_log_separation
            row["finite_time_lambda_1_per_year"] = growth / time_years
            lam = float(row["finite_time_lambda_1_per_year"])
            row["finite_time_lyapunov_time_years"] = (1.0 / lam) if lam > 0.0 else ""

    fit = linear_fit(rows, args.fit_start_years, fit_end)
    # Rewrite once at completion so finite-time lambda columns are populated for
    # all rows, while partial rows remain available during long runs.
    with csv_path.open("w", newline="") as file_obj:
        final_writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        final_writer.writeheader()
        final_writer.writerows(rows)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    plot_rows(rows, plot_path)

    warnings = [
        "finite-time shadow divergence diagnostic; not an asymptotic Lyapunov exponent",
    ]
    if args.gr_model != "none":
        warnings.append("GR shadow trajectory is trajectory-only; no REBOUNDx variational MEGNO claim is made")
    if fit.get("warning"):
        warnings.append(str(fit["warning"]))
    summary = {
        "diagnostic": "REBOUND shadow-trajectory finite-time divergence",
        "model_scope": args.model_scope,
        "integrator": args.integrator,
        "gr_model": args.gr_model,
        "duration_years": args.duration_years,
        "step_days": args.step_days,
        "record_every_years": args.record_every_years,
        "perturb_body": args.perturb_body,
        "perturbation_m": args.perturbation_m,
        "perturbation_mode": args.perturbation_mode,
        "seed": args.seed,
        "applied_displacements_m": applied,
        "fit_start_years": args.fit_start_years,
        "fit_end_years": fit_end,
        "fit": fit,
        "runtime_seconds": runtime,
        "warnings": warnings,
        "checkpointing_enabled": checkpoint_every_s is not None,
        "checkpoint_every_years": args.checkpoint_every_years,
        "checkpoint_dir": str(checkpoint_root) if checkpoint_every_s is not None or args.resume_from_checkpoint else None,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resumed_from_time_years": resumed_from_time_years,
        "number_of_checkpoints_written": checkpoints_written,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "checkpoint_warnings": checkpoint_warnings,
        "config_hash": config_hash,
        "output_rows": len(rows),
        "duplicate_rows_removed_on_resume": duplicate_rows_removed_on_resume,
        "outputs": {"csv": str(csv_path), "summary": str(summary_path), "plot": str(plot_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote csv: {csv_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote plot: {plot_path}")
    print(f"fit_lambda_1_per_year: {fit.get('lambda_1_per_year')}")
    print(f"runtime_seconds: {runtime:.3f}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
