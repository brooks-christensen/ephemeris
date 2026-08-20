from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np

from .chaos_estimator_diagnostics import analyze_running_lambda

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .long_term_stability_cli import (
    build_rebound_simulation,
    optional_import_module,
    parse_start_datetime,
    rebound_state_from_sim,
    sanitize_tag,
    stability_body_list,
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
    invariant_diagnostics_row,
    invariant_reference,
)


MODE_DESCRIPTION = (
    "RETIRED production Lyapunov path: finite-difference two-trajectory Benettin mode, "
    "kept for documentation/debugging only"
)

BODY_CHOICE_MAP = {
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

PROGRESS_FIELDS = [
    "time_years",
    "renorm_count",
    "separation_norm_before",
    "target_separation_norm",
    "accumulated_log_growth",
    "finite_time_lcn_1_per_year",
    "fit_start_years",
    "fit_elapsed_years",
    "seed",
    "step_days",
    "integrator",
    "gr_model",
    "reference_relative_energy_error",
    "shadow_relative_energy_error",
    "reference_relative_angular_momentum_error",
    "shadow_relative_angular_momentum_error",
    "interval_log_growth",
    "interval_lcn_1_per_year",
    "pre_renorm_norm",
    "post_renorm_norm",
    "post_renorm_relative_norm_error",
    "deviation_direction_cosine_pre_vs_post",
    "reference_com_position_norm",
    "reference_com_velocity_norm",
    "shadow_com_position_norm",
    "shadow_com_velocity_norm",
    "delta_semimajor_axis",
    "delta_mean_longitude_wrapped",
    "reference_renorm_max_position_change_m",
    "reference_renorm_max_velocity_change_m_s",
    "reference_standalone_max_position_delta_m",
    "reference_standalone_max_velocity_delta_m_s",
]


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value_f = float(value)
    return value_f if math.isfinite(value_f) else None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GR-compatible finite-time two-trajectory Benettin worker."
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--start-date", type=parse_start_datetime, default=parse_start_datetime("2000-01-01"))
    parser.add_argument("--duration-years", type=float, default=1000.0)
    parser.add_argument("--step-days", type=float, default=4.0)
    parser.add_argument("--record-every-years", type=float, default=100.0)
    parser.add_argument(
        "--model-scope",
        choices=["full", "full_with_pluto", "inner", "two_body_mercury", "two_body_jupiter", "two_body_saturn"],
        default="full",
    )
    parser.add_argument("--include-pluto", action="store_true")
    parser.add_argument("--integrator", choices=["whfast", "ias15"], default="whfast")
    parser.add_argument("--gr-model", choices=["none", "gr_potential", "gr"], default="none")
    parser.add_argument("--ias15-epsilon", type=float, default=1.0e-10)
    parser.add_argument("--perturb-body", "--lyapunov-body", choices=[*BODY_CHOICE_MAP.keys(), "all"], default="mercury")
    parser.add_argument("--perturbation-m", "--lyapunov-perturbation-m", type=float, default=1.0)
    parser.add_argument(
        "--perturbation-mode",
        choices=["radial", "tangential", "normal", "cartesian", "random"],
        default="radial",
    )
    parser.add_argument("--renorm-years", "--lyapunov-renorm-years", type=float, default=1000.0)
    parser.add_argument("--fit-start-years", "--lyapunov-fit-start-years", type=float, default=0.0)
    parser.add_argument("--fit-end-years", "--lyapunov-fit-end-years", type=float, default=None)
    parser.add_argument("--seed", "--lyapunov-seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="../output")
    parser.add_argument("--tag", default="gr_benettin")
    parser.add_argument("--status-every-renorm", action="store_true")
    parser.add_argument("--progress-file-every-renorm", action="store_true")
    parser.add_argument("--progress-line-every-seconds", type=float, default=300.0)
    parser.add_argument("--status-file-every-seconds", type=float, default=300.0)
    parser.add_argument("--checkpoint-every-years", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--with-standalone-reference-check", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--with-lyapunov", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lyapunov-method", default="two_trajectory", help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="rebound", help=argparse.SUPPRESS)
    parser.add_argument("--rebound-integrator", choices=["whfast", "ias15"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rebound-gr-model", choices=["none", "gr_potential", "gr", "gr_full"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rebound-simulationarchive", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rebound-archive-interval-years", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def normalized_config(args: argparse.Namespace, bodies: tuple[str, ...]) -> dict[str, Any]:
    integrator = args.rebound_integrator or args.integrator
    gr_model = args.rebound_gr_model or args.gr_model
    if gr_model == "gr_full":
        gr_model = "gr"
    return {
        "kernel_path": str(args.kernel_path),
        "start_date": args.start_date.isoformat(),
        "step_days": float(args.step_days),
        "record_every_years": float(args.record_every_years),
        "model_scope": args.model_scope,
        "body_names": bodies,
        "integrator": integrator,
        "gr_model": gr_model,
        "ias15_epsilon": float(args.ias15_epsilon),
        "perturb_body": args.perturb_body,
        "perturbation_m": float(args.perturbation_m),
        "perturbation_mode": args.perturbation_mode,
        "renorm_years": float(args.renorm_years),
        "fit_start_years": float(args.fit_start_years),
        "fit_end_years": args.fit_end_years,
        "seed": int(args.seed),
        "with_standalone_reference_check": bool(args.with_standalone_reference_check),
    }


def config_hash(config: dict[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def attach_reboundx_force(sim, gr_model: str):
    if gr_model == "none":
        return None
    reboundx = optional_import_module("reboundx")
    if reboundx is None:
        raise RuntimeError("reboundx is not installed; use --gr-model none.")
    if gr_model not in {"gr_potential", "gr"}:
        raise RuntimeError(f"Unsupported GR model for this worker: {gr_model}")
    rebx = reboundx.Extras(sim)
    force = rebx.load_force(gr_model)
    rebx.add_force(force)
    force.params["c"] = 299_792_458.0
    return rebx


def build_sim_pair(rebound, state: NBodyState, args: argparse.Namespace, gr_model: str):
    ref = build_rebound_simulation(
        rebound,
        state,
        integrator=args.rebound_integrator or args.integrator,
        step_s=args.step_days * DAY_S,
        ias15_epsilon=args.ias15_epsilon,
    )
    shadow = build_rebound_simulation(
        rebound,
        state,
        integrator=args.rebound_integrator or args.integrator,
        step_s=args.step_days * DAY_S,
        ias15_epsilon=args.ias15_epsilon,
    )
    ref_rebx = attach_reboundx_force(ref, gr_model)
    shadow_rebx = attach_reboundx_force(shadow, gr_model)
    return ref, shadow, ref_rebx, shadow_rebx


def body_indices(choice: str, bodies: tuple[str, ...]) -> list[int]:
    if choice == "all":
        return [i for i, name in enumerate(bodies) if name != "sun"]
    target = BODY_CHOICE_MAP[choice]
    if target not in bodies:
        raise ValueError(f"Selected perturbation body {target!r} is not present in model.")
    return [bodies.index(target)]


def perturbation_direction(
    state: NBodyState,
    index: int,
    sun_index: int,
    mode: str,
    rng: random.Random,
) -> np.ndarray:
    r = state.positions[index] - state.positions[sun_index]
    v = state.velocities[index] - state.velocities[sun_index]
    r_norm = np.linalg.norm(r)
    if r_norm == 0.0:
        raise ValueError("Cannot define perturbation direction for zero heliocentric radius.")
    radial = r / r_norm
    tangential_raw = v - np.dot(v, radial) * radial
    t_norm = np.linalg.norm(tangential_raw)
    tangential = tangential_raw / t_norm if t_norm > 0.0 else np.array([0.0, 1.0, 0.0])
    normal_raw = np.cross(radial, tangential)
    n_norm = np.linalg.norm(normal_raw)
    normal = normal_raw / n_norm if n_norm > 0.0 else np.array([0.0, 0.0, 1.0])
    if mode == "radial":
        return radial
    if mode == "tangential":
        return tangential
    if mode == "normal":
        return normal
    if mode == "cartesian":
        return np.array([1.0, 0.0, 0.0])
    direction = np.array([rng.gauss(0.0, 1.0) for _ in range(3)])
    return direction / np.linalg.norm(direction)


def remove_com_modes(
    delta_pos: np.ndarray,
    delta_vel: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_mass = float(np.sum(masses))
    pos_com = np.sum(delta_pos * masses[:, None], axis=0) / total_mass
    vel_com = np.sum(delta_vel * masses[:, None], axis=0) / total_mass
    return delta_pos - pos_com, delta_vel - vel_com


def scaled_norm(delta_pos: np.ndarray, delta_vel: np.ndarray) -> float:
    vel_scale = AU_M / JULIAN_YEAR_S
    vector = np.concatenate([(delta_pos / AU_M).ravel(), (delta_vel / vel_scale).ravel()])
    return float(np.linalg.norm(vector))


def scaled_vector(delta_pos: np.ndarray, delta_vel: np.ndarray) -> np.ndarray:
    vel_scale = AU_M / JULIAN_YEAR_S
    return np.concatenate([(delta_pos / AU_M).ravel(), (delta_vel / vel_scale).ravel()])


def direction_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return math.nan
    return float(np.dot(a, b) / (norm_a * norm_b))


def synchronize_simulation(sim) -> None:
    if hasattr(sim, "synchronize"):
        sim.synchronize()


def mark_coordinates_rebuilt(sim) -> None:
    if hasattr(sim, "ri_whfast"):
        try:
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        except Exception:
            pass


def initial_shadow_state(
    state: NBodyState,
    bodies: tuple[str, ...],
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[NBodyState, float, dict[str, list[float]]]:
    sun_index = bodies.index("sun")
    delta_pos = np.zeros_like(state.positions)
    delta_vel = np.zeros_like(state.velocities)
    indices = body_indices(args.perturb_body, bodies)
    per_body_m = float(args.perturbation_m) / math.sqrt(len(indices))
    metadata: dict[str, list[float]] = {}
    for index in indices:
        direction = perturbation_direction(state, index, sun_index, args.perturbation_mode, rng)
        delta = direction * per_body_m
        delta_pos[index] += delta
        metadata[bodies[index]] = [float(x) for x in delta]
    delta_pos, delta_vel = remove_com_modes(delta_pos, delta_vel, state.masses)
    target_norm = scaled_norm(delta_pos, delta_vel)
    if target_norm <= 0.0:
        raise RuntimeError("Initial perturbation norm is zero after COM-mode removal.")
    return (
        NBodyState(
            positions=state.positions + delta_pos,
            velocities=state.velocities + delta_vel,
            masses=state.masses,
        ),
        target_norm,
        metadata,
    )


def apply_state_to_sim(sim, state: NBodyState) -> None:
    synchronize_simulation(sim)
    n_real = int(getattr(sim, "N_real", len(state.masses)) or len(state.masses))
    for index, particle in enumerate(sim.particles[:n_real]):
        particle.x, particle.y, particle.z = state.positions[index]
        particle.vx, particle.vy, particle.vz = state.velocities[index]
    mark_coordinates_rebuilt(sim)


def deviation_between_sims(ref_sim, shadow_sim, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    synchronize_simulation(ref_sim)
    synchronize_simulation(shadow_sim)
    n_real = int(getattr(ref_sim, "N_real", len(masses)) or len(masses))
    delta_pos = np.zeros((n_real, 3), dtype=float)
    delta_vel = np.zeros((n_real, 3), dtype=float)
    for i in range(n_real):
        rp = ref_sim.particles[i]
        sp = shadow_sim.particles[i]
        delta_pos[i] = [sp.x - rp.x, sp.y - rp.y, sp.z - rp.z]
        delta_vel[i] = [sp.vx - rp.vx, sp.vy - rp.vy, sp.vz - rp.vz]
    return remove_com_modes(delta_pos, delta_vel, masses)


def com_norms(state: NBodyState) -> tuple[float, float]:
    total_mass = float(np.sum(state.masses))
    pos = np.sum(state.positions * state.masses[:, None], axis=0) / total_mass
    vel = np.sum(state.velocities * state.masses[:, None], axis=0) / total_mass
    return float(np.linalg.norm(pos)), float(np.linalg.norm(vel))


def renormalize_shadow(ref_sim, shadow_sim, masses: np.ndarray, target_norm: float) -> dict[str, float]:
    synchronize_simulation(ref_sim)
    synchronize_simulation(shadow_sim)
    ref_state_before = rebound_state_from_sim(ref_sim, masses)
    delta_pos, delta_vel = deviation_between_sims(ref_sim, shadow_sim, masses)
    norm_before = scaled_norm(delta_pos, delta_vel)
    if norm_before <= 0.0 or not math.isfinite(norm_before):
        raise RuntimeError("Shadow/reference separation norm became invalid.")
    pre_vector = scaled_vector(delta_pos, delta_vel)
    scale = target_norm / norm_before
    desired_delta_pos = delta_pos * scale
    desired_delta_vel = delta_vel * scale
    post_delta_pos = desired_delta_pos
    post_delta_vel = desired_delta_vel
    post_norm = math.nan
    post_vector = pre_vector
    for _ in range(5):
        new_state = NBodyState(
            positions=ref_state_before.positions + desired_delta_pos,
            velocities=ref_state_before.velocities + desired_delta_vel,
            masses=masses,
        )
        apply_state_to_sim(shadow_sim, new_state)
        post_delta_pos, post_delta_vel = deviation_between_sims(ref_sim, shadow_sim, masses)
        post_norm = scaled_norm(post_delta_pos, post_delta_vel)
        post_vector = scaled_vector(post_delta_pos, post_delta_vel)
        if post_norm <= 0.0 or not math.isfinite(post_norm):
            break
        if abs(post_norm - target_norm) / target_norm <= 1.0e-10 or abs(post_norm - target_norm) <= 1.0e-14:
            break
        correction = target_norm / post_norm
        desired_delta_pos = post_delta_pos * correction
        desired_delta_vel = post_delta_vel * correction
    ref_state_after = rebound_state_from_sim(ref_sim, masses)
    return {
        "pre_renorm_norm": norm_before,
        "post_renorm_norm": post_norm,
        "post_renorm_relative_norm_error": abs(post_norm - target_norm) / target_norm,
        "deviation_direction_cosine_pre_vs_post": direction_cosine(pre_vector, post_vector),
        "reference_renorm_max_position_change_m": float(
            np.max(np.abs(ref_state_after.positions - ref_state_before.positions))
        ),
        "reference_renorm_max_velocity_change_m_s": float(
            np.max(np.abs(ref_state_after.velocities - ref_state_before.velocities))
        ),
    }


def relative_invariants(sim, masses: np.ndarray, reference) -> tuple[float, float]:
    state = rebound_state_from_sim(sim, masses)
    row = invariant_diagnostics_row(float(sim.t), state, reference)
    return float(row["energy_rel_drift"]), float(row["angular_momentum_rel_drift"])


def wrap_angle_rad(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def selected_orbital_delta(
    ref_sim,
    shadow_sim,
    masses: np.ndarray,
    bodies: tuple[str, ...],
    selected_body: str,
) -> tuple[float, float]:
    if selected_body == "all" or selected_body not in BODY_CHOICE_MAP:
        return math.nan, math.nan
    body_name = BODY_CHOICE_MAP[selected_body]
    if body_name not in bodies:
        return math.nan, math.nan
    ref_state = rebound_state_from_sim(ref_sim, masses)
    shadow_state = rebound_state_from_sim(shadow_sim, masses)
    sun_index = bodies.index("sun")
    ref_elements = {
        item.body_name: item
        for item in heliocentric_elements_for_state(ref_state, bodies, sun_index=sun_index)
    }
    shadow_elements = {
        item.body_name: item
        for item in heliocentric_elements_for_state(shadow_state, bodies, sun_index=sun_index)
    }
    ref_item = ref_elements.get(body_name)
    shadow_item = shadow_elements.get(body_name)
    if ref_item is None or shadow_item is None:
        return math.nan, math.nan
    return (
        float((shadow_item.semi_major_axis_m - ref_item.semi_major_axis_m) / AU_M),
        wrap_angle_rad(shadow_item.mean_longitude_rad - ref_item.mean_longitude_rad),
    )


def open_progress_csv(path: Path, append: bool) -> tuple[Any, csv.DictWriter]:
    needs_header = (not append) or (not path.exists()) or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
    if needs_header:
        writer.writeheader()
    return handle, writer


def truncate_progress_after(path: Path, checkpoint_time_years: float) -> int:
    if not path.exists():
        return 0
    rows = []
    removed = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or PROGRESS_FIELDS
        for row in reader:
            try:
                t = float(row.get("time_years", "nan"))
            except ValueError:
                removed += 1
                continue
            if t <= checkpoint_time_years + 1.0e-9:
                rows.append(row)
            else:
                removed += 1
    if removed:
        tmp = path.with_name(f"{path.name}.tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    return removed


def checkpoint_dirs(checkpoint_dir: Path, tag: str) -> list[Path]:
    return sorted(
        [p for p in checkpoint_dir.glob(f"benettin_checkpoint_{tag}_*yr") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )


def write_checkpoint(
    checkpoint_dir: Path,
    tag: str,
    ref_sim,
    shadow_sim,
    state_payload: dict[str, Any],
    keep: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    time_years = float(state_payload["current_time_years"])
    name = f"benettin_checkpoint_{tag}_{time_years:.9f}yr"
    tmp = checkpoint_dir / f"{name}.tmp"
    final = checkpoint_dir / name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ref_sim.save_to_file(str(tmp / "reference.bin"), delete_file=True)
    shadow_sim.save_to_file(str(tmp / "shadow.bin"), delete_file=True)
    atomic_write_json(tmp / "checkpoint_state.json", state_payload)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    old = checkpoint_dirs(checkpoint_dir, tag)
    for path in old[:-keep]:
        shutil.rmtree(path, ignore_errors=True)
    return final


def encode_random_state(rng: random.Random) -> str:
    return repr(rng.getstate())


def restore_random_state(rng: random.Random, encoded: str | None) -> None:
    if not encoded:
        return
    import ast

    state = ast.literal_eval(encoded)
    rng.setstate(state)


def load_checkpoint(path: Path, expected_hash: str, rebound):
    state_path = path / "checkpoint_state.json"
    if not state_path.exists():
        raise RuntimeError(f"Checkpoint lacks checkpoint_state.json: {path}")
    payload = json.loads(state_path.read_text())
    if payload.get("config_hash") != expected_hash:
        raise RuntimeError("Checkpoint configuration hash does not match current CLI configuration.")
    ref_path = path / "reference.bin"
    shadow_path = path / "shadow.bin"
    if not ref_path.exists() or not shadow_path.exists():
        raise RuntimeError(f"Checkpoint lacks reference.bin or shadow.bin: {path}")
    ref = rebound.Simulation(str(ref_path))
    shadow = rebound.Simulation(str(shadow_path))
    if abs(float(ref.t) - float(shadow.t)) > 1.0e-6:
        raise RuntimeError("Checkpoint reference and shadow times disagree.")
    return payload, ref, shadow


def current_scaled_deviation_direction(ref_sim, shadow_sim, masses: np.ndarray) -> list[float]:
    delta_pos, delta_vel = deviation_between_sims(ref_sim, shadow_sim, masses)
    vector = scaled_vector(delta_pos, delta_vel)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not math.isfinite(norm):
        raise RuntimeError("Checkpoint deviation direction is invalid.")
    return [float(value) for value in vector / norm]


def validate_checkpoint_direction(payload: dict[str, Any], ref_sim, shadow_sim, masses: np.ndarray) -> None:
    stored = payload.get("scaled_deviation_direction")
    if not stored:
        return
    current = np.asarray(current_scaled_deviation_direction(ref_sim, shadow_sim, masses), dtype=float)
    previous = np.asarray(stored, dtype=float)
    cosine = direction_cosine(current, previous)
    if cosine < 1.0 - 1.0e-10:
        raise RuntimeError(
            "Checkpoint restored a different shadow-reference deviation direction: "
            f"cosine={cosine:.16g}"
        )


def latest_checkpoint(checkpoint_dir: Path, tag: str) -> Path | None:
    dirs = checkpoint_dirs(checkpoint_dir, tag)
    return dirs[-1] if dirs else None


def classify_lcn(
    lcn_history: list[tuple[float, float]],
    model_scope: str,
) -> str:
    """Classify from the running-LCN history.

    The previous criterion was ``lcn * elapsed_years > 1.0``. Since
    ``lcn = fit_log / fit_elapsed``, that product is algebraically just
    ``fit_log`` -- "has the tangent grown by one e-fold". Regular linear tangent
    growth passes it after about three renormalization intervals, so every
    full-scope run was classified ``chaotic_candidate`` regardless of the
    dynamics. Measured on an integrable two-body system the statistic was
    already 5.19 at the first sample.

    The replacement compares the running estimate at the end of the run against
    its value at the half-way point: ~1 for a genuine exponent, ~0.5 while the
    estimate is still decaying as ln(t)/t.
    """

    if len(lcn_history) < 4:
        return "ambiguous"
    try:
        result = analyze_running_lambda(
            [entry[0] for entry in lcn_history],
            [entry[1] for entry in lcn_history],
        )
    except (ValueError, ZeroDivisionError):
        return "ambiguous"
    if model_scope.startswith("two_body") and result.classification == "chaotic_candidate":
        # Integrable by construction: a chaotic verdict here indicates an
        # estimator defect, not a physical finding.
        return "ambiguous"
    return result.classification


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rebound_integrator is not None:
        args.integrator = args.rebound_integrator
    if args.rebound_gr_model is not None:
        args.gr_model = "gr" if args.rebound_gr_model == "gr_full" else args.rebound_gr_model
    if args.backend != "rebound":
        parser.error("gr_benettin_cli requires the REBOUND backend.")
    if args.lyapunov_method != "two_trajectory":
        parser.error("gr_benettin_cli implements the two_trajectory Benettin path only.")
    if args.duration_years <= 0 or args.step_days <= 0 or args.renorm_years <= 0:
        parser.error("duration, step, and renormalization cadence must be positive.")
    if args.integrator == "whfast" and args.gr_model == "gr":
        parser.error("Use --integrator ias15 for velocity-dependent --gr-model gr validation.")
    if args.gr_model == "gr_potential" and args.integrator != "whfast":
        parser.error("Initial production gr_potential path is WHFast-only.")

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / f"dual_checkpoints_{tag}"
    progress_path = output_dir / f"benettin_progress_{tag}.csv"
    status_path = output_dir / f"run_status_{tag}.json"
    summary_path = output_dir / f"benettin_summary_{tag}.json"

    bodies = stability_body_list(args.model_scope, include_pluto=args.include_pluto)
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=args.kernel_path),
    )
    config = normalized_config(args, bodies)
    cfg_hash = config_hash(config)
    rebound = optional_import_module("rebound")
    if rebound is None:
        raise RuntimeError("REBOUND is not installed.")

    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if args.resume_latest:
        resume_path = latest_checkpoint(checkpoint_dir, tag)
    resume_payload = None
    append_progress = False
    if resume_path is not None:
        resume_payload, ref_sim, shadow_sim = load_checkpoint(resume_path, cfg_hash, rebound)
        validate_checkpoint_direction(resume_payload, ref_sim, shadow_sim, state0.masses)
        truncate_progress_after(progress_path, float(resume_payload["current_time_years"]))
        append_progress = True
        ref_rebx = attach_reboundx_force(ref_sim, config["gr_model"])
        shadow_rebx = attach_reboundx_force(shadow_sim, config["gr_model"])
        target_norm = float(resume_payload["target_separation_norm"])
        renorm_count = int(resume_payload["renorm_count"])
        total_log = float(resume_payload["accumulated_log_growth"])
        fit_log = float(resume_payload["fit_accumulated_log_growth"])
        fit_elapsed_years = float(resume_payload["fit_elapsed_years"])
        perturb_meta = resume_payload.get("perturbation_vectors_m", {})
        rng = random.Random(args.seed)
        restore_random_state(rng, resume_payload.get("rng_state_repr"))
    else:
        rng = random.Random(args.seed)
        shadow_state, target_norm, perturb_meta = initial_shadow_state(state0, bodies, args, rng)
        ref_sim, shadow_sim, ref_rebx, shadow_rebx = build_sim_pair(rebound, state0, args, config["gr_model"])
        apply_state_to_sim(shadow_sim, shadow_state)
        renorm_count = 0
        total_log = 0.0
        fit_log = 0.0
        fit_elapsed_years = 0.0

    ref_reference = invariant_reference(state0, G=G_SI)
    shadow_reference = invariant_reference(
        rebound_state_from_sim(shadow_sim, state0.masses),
        G=G_SI,
    )

    progress_handle, progress_writer = open_progress_csv(progress_path, append=append_progress)
    start_wall = time.time()
    last_line_wall = 0.0
    last_status_wall = 0.0
    last_checkpoint_years = seconds_to_years(float(ref_sim.t))
    checkpoints_written: list[str] = []
    warnings: list[str] = []
    if config["gr_model"] != "none":
        warnings.append("Energy diagnostics are Newtonian bookkeeping for GR trajectories.")
    standalone_ref = None
    standalone_rebx = None
    max_standalone_pos_delta = 0.0
    max_standalone_vel_delta = 0.0
    if args.with_standalone_reference_check:
        standalone_ref = build_rebound_simulation(
            rebound,
            state0,
            integrator=config["integrator"],
            step_s=args.step_days * DAY_S,
            ias15_epsilon=args.ias15_epsilon,
        )
        standalone_rebx = attach_reboundx_force(standalone_ref, config["gr_model"])
    print(
        f"[Benettin] {tag}: start t={seconds_to_years(float(ref_sim.t)):.6g} yr "
        f"target={args.duration_years:g} yr integrator={config['integrator']} gr={config['gr_model']}",
        flush=True,
    )

    duration_s = args.duration_years * JULIAN_YEAR_S
    renorm_s = args.renorm_years * JULIAN_YEAR_S
    next_t = (math.floor(float(ref_sim.t) / renorm_s) + 1) * renorm_s
    fit_end = args.fit_end_years if args.fit_end_years is not None else args.duration_years
    latest_lcn = math.nan
    latest_checkpoint_path = None
    max_abs_ref_energy = 0.0
    max_abs_shadow_energy = 0.0
    max_abs_ref_l = 0.0
    max_abs_shadow_l = 0.0
    lcn_history: list[tuple[float, float]] = []
    warnings.append(
        "This two-trajectory finite-difference Benettin worker is retired as a production Lyapunov diagnostic; "
        "do not use it to unlock GR experiments."
    )

    try:
        while float(ref_sim.t) < duration_s - 1.0e-6:
            target_t = min(next_t, duration_s)
            ref_sim.integrate(target_t, exact_finish_time=1)
            shadow_sim.integrate(target_t, exact_finish_time=1)
            if standalone_ref is not None:
                standalone_ref.integrate(target_t, exact_finish_time=1)
            current_years = seconds_to_years(float(ref_sim.t))
            renorm_diag = renormalize_shadow(ref_sim, shadow_sim, state0.masses, target_norm)
            norm_before = renorm_diag["pre_renorm_norm"]
            post_abs_norm_error = abs(renorm_diag["post_renorm_norm"] - target_norm)
            if (
                renorm_diag["post_renorm_relative_norm_error"] > 1.0e-10
                and post_abs_norm_error > 1.0e-14
            ):
                raise RuntimeError(
                    "Post-renormalization norm error exceeded 1e-10: "
                    f"{renorm_diag['post_renorm_relative_norm_error']:.3e}"
                )
            if renorm_diag["deviation_direction_cosine_pre_vs_post"] < 1.0 - 1.0e-10:
                warnings.append(
                    "Post-renormalization direction cosine missed 1-1e-10; "
                    "likely finite-precision cancellation for small perturbations: "
                    f"{renorm_diag['deviation_direction_cosine_pre_vs_post']:.16g}"
                )
            direction_hard_floor = 0.999 if target_norm < 1.0e-12 else 0.9999
            if renorm_diag["deviation_direction_cosine_pre_vs_post"] < direction_hard_floor:
                raise RuntimeError(
                    "Renormalization failed to preserve evolved deviation direction: "
                    f"{renorm_diag['deviation_direction_cosine_pre_vs_post']:.16g}"
                )
            if (
                renorm_diag["reference_renorm_max_position_change_m"] > 0.0
                or renorm_diag["reference_renorm_max_velocity_change_m_s"] > 0.0
            ):
                raise RuntimeError("Reference trajectory changed during shadow renormalization.")
            increment = math.log(norm_before / target_norm)
            total_log += increment
            renorm_count += 1
            previous_years = current_years - args.renorm_years
            overlap_start = max(previous_years, args.fit_start_years)
            overlap_end = min(current_years, fit_end)
            if overlap_end > overlap_start:
                fit_log += increment * ((overlap_end - overlap_start) / max(args.renorm_years, 1.0e-30))
                fit_elapsed_years += overlap_end - overlap_start
            latest_lcn = fit_log / fit_elapsed_years if fit_elapsed_years > 0 else math.nan
            ref_energy, ref_l = relative_invariants(ref_sim, state0.masses, ref_reference)
            shadow_energy, shadow_l = relative_invariants(shadow_sim, state0.masses, shadow_reference)
            max_abs_ref_energy = max(max_abs_ref_energy, abs(ref_energy))
            max_abs_shadow_energy = max(max_abs_shadow_energy, abs(shadow_energy))
            max_abs_ref_l = max(max_abs_ref_l, abs(ref_l))
            max_abs_shadow_l = max(max_abs_shadow_l, abs(shadow_l))
            if math.isfinite(latest_lcn):
                lcn_history.append((current_years, latest_lcn))
            ref_state_now = rebound_state_from_sim(ref_sim, state0.masses)
            shadow_state_now = rebound_state_from_sim(shadow_sim, state0.masses)
            ref_com_pos, ref_com_vel = com_norms(ref_state_now)
            shadow_com_pos, shadow_com_vel = com_norms(shadow_state_now)
            delta_a, delta_lambda = selected_orbital_delta(
                ref_sim,
                shadow_sim,
                state0.masses,
                bodies,
                args.perturb_body,
            )
            standalone_pos_delta = math.nan
            standalone_vel_delta = math.nan
            if standalone_ref is not None:
                standalone_state = rebound_state_from_sim(standalone_ref, state0.masses)
                standalone_pos_delta = float(
                    np.max(np.abs(ref_state_now.positions - standalone_state.positions))
                )
                standalone_vel_delta = float(
                    np.max(np.abs(ref_state_now.velocities - standalone_state.velocities))
                )
                max_standalone_pos_delta = max(max_standalone_pos_delta, standalone_pos_delta)
                max_standalone_vel_delta = max(max_standalone_vel_delta, standalone_vel_delta)
            row = {
                "time_years": current_years,
                "renorm_count": renorm_count,
                "separation_norm_before": norm_before,
                "target_separation_norm": target_norm,
                "accumulated_log_growth": total_log,
                "finite_time_lcn_1_per_year": latest_lcn if math.isfinite(latest_lcn) else "",
                "fit_start_years": args.fit_start_years,
                "fit_elapsed_years": fit_elapsed_years,
                "seed": args.seed,
                "step_days": args.step_days,
                "integrator": config["integrator"],
                "gr_model": config["gr_model"],
                "reference_relative_energy_error": ref_energy,
                "shadow_relative_energy_error": shadow_energy,
                "reference_relative_angular_momentum_error": ref_l,
                "shadow_relative_angular_momentum_error": shadow_l,
                "interval_log_growth": increment,
                "interval_lcn_1_per_year": increment / args.renorm_years,
                "pre_renorm_norm": norm_before,
                "post_renorm_norm": renorm_diag["post_renorm_norm"],
                "post_renorm_relative_norm_error": renorm_diag["post_renorm_relative_norm_error"],
                "deviation_direction_cosine_pre_vs_post": renorm_diag[
                    "deviation_direction_cosine_pre_vs_post"
                ],
                "reference_com_position_norm": ref_com_pos,
                "reference_com_velocity_norm": ref_com_vel,
                "shadow_com_position_norm": shadow_com_pos,
                "shadow_com_velocity_norm": shadow_com_vel,
                "delta_semimajor_axis": delta_a if math.isfinite(delta_a) else "",
                "delta_mean_longitude_wrapped": delta_lambda if math.isfinite(delta_lambda) else "",
                "reference_renorm_max_position_change_m": renorm_diag[
                    "reference_renorm_max_position_change_m"
                ],
                "reference_renorm_max_velocity_change_m_s": renorm_diag[
                    "reference_renorm_max_velocity_change_m_s"
                ],
                "reference_standalone_max_position_delta_m": (
                    standalone_pos_delta if math.isfinite(standalone_pos_delta) else ""
                ),
                "reference_standalone_max_velocity_delta_m_s": (
                    standalone_vel_delta if math.isfinite(standalone_vel_delta) else ""
                ),
            }
            progress_writer.writerow(row)
            progress_handle.flush()
            os.fsync(progress_handle.fileno())
            now = time.time()
            elapsed = now - start_wall
            rate = max(0.0, current_years - (resume_payload or {}).get("current_time_years", 0.0)) / elapsed if elapsed > 0 else math.nan
            eta = (args.duration_years - current_years) / rate if rate and rate > 0 else math.nan
            status = {
                "mode": MODE_DESCRIPTION,
                "tag": tag,
                "current_time_years": current_years,
                "percent_complete": 100.0 * current_years / args.duration_years,
                "elapsed_wall_seconds": elapsed,
                "recent_simulated_years_per_wall_second": rate,
                "eta_seconds": finite_or_none(eta),
                "latest_lcn_1_per_year": finite_or_none(latest_lcn),
                "latest_checkpoint": str(latest_checkpoint_path) if latest_checkpoint_path else None,
                "warnings": warnings,
                "configuration_hash": cfg_hash,
            }
            if args.status_every_renorm or now - last_status_wall >= args.status_file_every_seconds:
                atomic_write_json(status_path, status)
                last_status_wall = now
            if now - last_line_wall >= args.progress_line_every_seconds or renorm_count == 1:
                print(
                    f"[Benettin] {tag}: t={current_years:.6g} yr "
                    f"{status['percent_complete']:.3f}% elapsed={elapsed:.1f}s "
                    f"rate={rate:.6g} yr/s ETA={eta:.1f}s LCN={latest_lcn:.6e} "
                    f"dE_ref={ref_energy:.3e} dL_ref={ref_l:.3e}",
                    flush=True,
                )
                last_line_wall = now
            if args.checkpoint_every_years is not None and (
                current_years - last_checkpoint_years >= args.checkpoint_every_years - 1.0e-9
                or current_years >= args.duration_years - 1.0e-9
            ):
                payload = {
                    "tag": tag,
                    "config": config,
                    "config_hash": cfg_hash,
                    "current_time_years": current_years,
                    "target_separation_norm": target_norm,
                    "accumulated_log_growth": total_log,
                    "fit_accumulated_log_growth": fit_log,
                    "fit_elapsed_years": fit_elapsed_years,
                    "renorm_count": renorm_count,
                    "rng_state_repr": encode_random_state(rng),
                    "perturbation_vectors_m": perturb_meta,
                    "scaled_deviation_direction": current_scaled_deviation_direction(
                        ref_sim,
                        shadow_sim,
                        state0.masses,
                    ),
                    "created_utc": dt.datetime.utcnow().isoformat() + "Z",
                }
                latest_checkpoint_path = write_checkpoint(
                    checkpoint_dir,
                    tag,
                    ref_sim,
                    shadow_sim,
                    payload,
                    args.keep_checkpoints,
                )
                checkpoints_written.append(str(latest_checkpoint_path))
                last_checkpoint_years = current_years
            next_t += renorm_s
    finally:
        progress_handle.close()

    classification = classify_lcn(lcn_history, args.model_scope)
    late_values = [value for time_years, value in lcn_history if time_years >= 0.5 * args.duration_years]
    late_median_lcn = float(np.median(late_values)) if late_values else math.nan
    stable_positive_late_time_plateau = bool(
        math.isfinite(late_median_lcn)
        and late_median_lcn > 1.0e-5
        and args.model_scope.startswith("two_body")
    )
    summary = {
        "mode": MODE_DESCRIPTION,
        "tag": tag,
        "configuration": config,
        "configuration_hash": cfg_hash,
        "duration_years": args.duration_years,
        "actual_time_years": seconds_to_years(float(ref_sim.t)),
        "renorm_count": renorm_count,
        "target_separation_norm": target_norm,
        "accumulated_log_growth": total_log,
        "fit_accumulated_log_growth": fit_log,
        "fit_elapsed_years": fit_elapsed_years,
        "finite_time_lcn_1_per_year": finite_or_none(latest_lcn),
        "late_window_median_lcn_1_per_year": finite_or_none(late_median_lcn),
        "stable_positive_late_time_plateau": stable_positive_late_time_plateau,
        "max_abs_reference_relative_energy_error": max_abs_ref_energy,
        "max_abs_shadow_relative_energy_error": max_abs_shadow_energy,
        "max_abs_reference_relative_angular_momentum_error": max_abs_ref_l,
        "max_abs_shadow_relative_angular_momentum_error": max_abs_shadow_l,
        "classification_hint": classification,
        "reference_standalone_max_position_delta_m": max_standalone_pos_delta,
        "reference_standalone_max_velocity_delta_m_s": max_standalone_vel_delta,
        "perturbation_vectors_m": perturb_meta,
        "checkpoints_written": checkpoints_written,
        "warnings": warnings,
        "outputs": {
            "progress_csv": str(progress_path),
            "status_json": str(status_path),
            "summary_json": str(summary_path),
        },
    }
    atomic_write_json(summary_path, summary)
    print(f"[Benettin] wrote summary: {summary_path}", flush=True)
    print(
        f"[Benettin] complete: LCN={latest_lcn:.6e} 1/year classification={classification}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
