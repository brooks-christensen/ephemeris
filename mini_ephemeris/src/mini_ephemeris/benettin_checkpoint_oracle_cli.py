from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import gr_benettin_cli as gb
from .long_term_stability_cli import (
    build_rebound_simulation,
    optional_import_module,
    rebound_state_from_sim,
    sanitize_tag,
    stability_body_list,
)
from .nbody import NBodyState
from .orbital_elements import DAY_S, JULIAN_YEAR_S, seconds_to_years


FIELDS = [
    "time_years",
    "interval",
    "benettin_pre_norm",
    "benettin_log_growth_increment",
    "benettin_cumulative_lcn_1_per_year",
    "native_megno",
    "native_lcn_1_per_year",
    "lcn_difference_1_per_year",
    "normalized_direction_cosine_to_previous",
    "warning",
]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_archive_snapshot(rebound, archive_path: Path, snapshot_years: float):
    sim_archive_cls = getattr(rebound, "SimulationArchive", None) or getattr(rebound, "Simulationarchive", None)
    if sim_archive_cls is None:
        raise RuntimeError("This REBOUND build does not expose SimulationArchive/Simulationarchive.")
    archive = sim_archive_cls(str(archive_path))
    target_t = snapshot_years * JULIAN_YEAR_S
    return archive.getSimulation(target_t, mode="snapshot", keep_unsynchronized=1)


def real_state_from_archive_sim(sim, model_scope: str, include_pluto: bool) -> tuple[NBodyState, tuple[str, ...]]:
    bodies = stability_body_list(model_scope, include_pluto=include_pluto)
    n_real = int(getattr(sim, "N_real", len(sim.particles)) or len(sim.particles))
    if n_real < len(bodies):
        raise RuntimeError(f"Archive has N_real={n_real}, but {model_scope} expects {len(bodies)} bodies.")
    masses = np.array([sim.particles[i].m for i in range(len(bodies))], dtype=float)
    state = rebound_state_from_sim(sim, masses)
    if len(state.masses) != len(bodies):
        state = NBodyState(
            positions=state.positions[: len(bodies)].copy(),
            velocities=state.velocities[: len(bodies)].copy(),
            masses=masses,
        )
    return state, bodies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two-trajectory Benettin growth against native Newtonian REBOUND MEGNO from an archived physical snapshot."
    )
    parser.add_argument("--simulationarchive", required=True)
    parser.add_argument("--snapshot-years", type=float, default=300_000_000.0)
    parser.add_argument("--duration-years", type=float, default=100_000.0)
    parser.add_argument("--step-days", type=float, default=1.0)
    parser.add_argument("--renorm-years", type=float, default=1000.0)
    parser.add_argument("--model-scope", choices=["full", "full_with_pluto", "inner"], default="full_with_pluto")
    parser.add_argument("--include-pluto", action="store_true")
    parser.add_argument("--perturb-body", choices=[*gb.BODY_CHOICE_MAP.keys(), "all"], default="mercury")
    parser.add_argument("--perturbation-m", type=float, default=1.0)
    parser.add_argument("--perturbation-mode", choices=["radial", "tangential", "normal", "cartesian", "random"], default="radial")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="/home/peacelovephysics/ephemeris/output/stability/benettin_checkpoint_oracle")
    parser.add_argument("--tag", default="checkpoint_oracle")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration_years <= 0 or args.step_days <= 0 or args.renorm_years <= 0:
        parser.error("duration, step, and renormalization cadence must be positive.")
    rebound = optional_import_module("rebound")
    if rebound is None:
        raise RuntimeError("REBOUND is not installed.")

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"benettin_checkpoint_oracle_{tag}.csv"
    summary_path = output_dir / f"benettin_checkpoint_oracle_summary_{tag}.json"
    warnings = [
        "Native REBOUND MEGNO does not expose an arbitrary variational-vector setter in the validated Python path; exact initial deviation-vector matching is unavailable.",
        "This is a Newtonian diagnostic-oracle comparison, not a GR validation.",
    ]

    archived_sim = load_archive_snapshot(rebound, Path(args.simulationarchive), args.snapshot_years)
    state, bodies = real_state_from_archive_sim(archived_sim, args.model_scope, args.include_pluto)
    ref = build_rebound_simulation(
        rebound,
        state,
        integrator="whfast",
        step_s=args.step_days * DAY_S,
        ias15_epsilon=1.0e-10,
    )
    shadow = build_rebound_simulation(
        rebound,
        state,
        integrator="whfast",
        step_s=args.step_days * DAY_S,
        ias15_epsilon=1.0e-10,
    )
    native = build_rebound_simulation(
        rebound,
        state,
        integrator="whfast",
        step_s=args.step_days * DAY_S,
        ias15_epsilon=1.0e-10,
    )
    native.init_megno(seed=args.seed)
    native.lyapunov()

    shim = argparse.Namespace(
        perturb_body=args.perturb_body,
        perturbation_m=args.perturbation_m,
        perturbation_mode=args.perturbation_mode,
    )
    shadow_state, target_norm, perturb_meta = gb.initial_shadow_state(state, bodies, shim, random.Random(args.seed))
    gb.apply_state_to_sim(shadow, shadow_state)

    total_log = 0.0
    renorm_count = 0
    previous_direction: np.ndarray | None = None
    rows: list[dict[str, Any]] = []
    start = time.time()
    duration_s = args.duration_years * JULIAN_YEAR_S
    renorm_s = args.renorm_years * JULIAN_YEAR_S
    t = renorm_s
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        while t <= duration_s + 1.0e-6:
            ref.integrate(t, exact_finish_time=1)
            shadow.integrate(t, exact_finish_time=1)
            native.integrate(t, exact_finish_time=1)
            delta_pos, delta_vel = gb.deviation_between_sims(ref, shadow, state.masses)
            direction = gb.scaled_vector(delta_pos, delta_vel)
            norm = float(np.linalg.norm(direction))
            direction = direction / norm if norm > 0 else direction
            cosine = gb.direction_cosine(direction, previous_direction) if previous_direction is not None else math.nan
            previous_direction = direction.copy()
            diag = gb.renormalize_shadow(ref, shadow, state.masses, target_norm)
            increment = math.log(diag["pre_renorm_norm"] / target_norm)
            total_log += increment
            renorm_count += 1
            years = seconds_to_years(float(ref.t))
            benettin_lcn = total_log / years if years > 0 else math.nan
            try:
                native_megno = float(native.megno())
            except Exception:
                native_megno = math.nan
            try:
                native_lcn = float(native.lyapunov())
            except Exception:
                native_lcn = math.nan
            row = {
                "time_years": years,
                "interval": renorm_count,
                "benettin_pre_norm": diag["pre_renorm_norm"],
                "benettin_log_growth_increment": increment,
                "benettin_cumulative_lcn_1_per_year": benettin_lcn,
                "native_megno": native_megno if math.isfinite(native_megno) else "",
                "native_lcn_1_per_year": native_lcn if math.isfinite(native_lcn) else "",
                "lcn_difference_1_per_year": (
                    benettin_lcn - native_lcn if math.isfinite(benettin_lcn) and math.isfinite(native_lcn) else ""
                ),
                "normalized_direction_cosine_to_previous": cosine if math.isfinite(cosine) else "",
                "warning": "; ".join(warnings),
            }
            writer.writerow(row)
            rows.append(row)
            t += renorm_s

    finite_diffs = [
        abs(float(row["lcn_difference_1_per_year"]))
        for row in rows
        if row["lcn_difference_1_per_year"] not in {"", None}
    ]
    summary = {
        "tag": tag,
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "simulationarchive": str(args.simulationarchive),
        "snapshot_years_requested": args.snapshot_years,
        "snapshot_time_years_loaded": seconds_to_years(float(archived_sim.t)),
        "duration_years": args.duration_years,
        "step_days": args.step_days,
        "renorm_years": args.renorm_years,
        "model_scope": args.model_scope,
        "perturb_body": args.perturb_body,
        "perturbation_m": args.perturbation_m,
        "seed": args.seed,
        "perturbation_vectors_m": perturb_meta,
        "exact_initial_deviation_vector_match_available": False,
        "row_count": len(rows),
        "final_benettin_lcn_1_per_year": rows[-1]["benettin_cumulative_lcn_1_per_year"] if rows else None,
        "final_native_lcn_1_per_year": rows[-1]["native_lcn_1_per_year"] if rows else None,
        "max_abs_lcn_difference_1_per_year": max(finite_diffs) if finite_diffs else None,
        "runtime_seconds": time.time() - start,
        "warnings": warnings,
        "outputs": {"csv": str(csv_path), "summary_json": str(summary_path)},
    }
    atomic_write_json(summary_path, summary)
    print(f"[oracle] wrote {csv_path}")
    print(f"[oracle] wrote {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
