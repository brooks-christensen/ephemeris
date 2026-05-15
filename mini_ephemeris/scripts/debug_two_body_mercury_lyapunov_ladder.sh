#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/two_body_mercury_lyapunov_runs}"
LADDER_CSV="${LADDER_CSV:-${OUTPUT_BASE}/two_body_mercury_lyapunov_ladder.csv}"
DURATION_YEARS="${DURATION_YEARS:-1000}"
FIT_START_YEARS="${FIT_START_YEARS:-100}"
FIT_END_YEARS="${FIT_END_YEARS:-${DURATION_YEARS}}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
STEP_DAYS_VALUES="${STEP_DAYS_VALUES:-1 0.5 0.25 0.125}"
PERTURBATION_M_VALUES="${PERTURBATION_M_VALUES:-1e-6 1e-3 1 1e3}"
RENORM_YEARS_VALUES="${RENORM_YEARS_VALUES:-0.05 0.1 0.25 1.0}"
MAX_CASES="${MAX_CASES:-}"

mkdir -p "${RUN_OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH OUTPUT_BASE RUN_OUTPUT_DIR LADDER_CSV
export DURATION_YEARS FIT_START_YEARS FIT_END_YEARS RECORD_EVERY_YEARS
export STEP_DAYS_VALUES PERTURBATION_M_VALUES RENORM_YEARS_VALUES MAX_CASES

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess


python_bin = os.environ["PYTHON_BIN"]
kernel_path = os.environ["KERNEL_PATH"]
run_output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
ladder_csv = Path(os.environ["LADDER_CSV"])
duration_years = float(os.environ["DURATION_YEARS"])
fit_start_years = float(os.environ["FIT_START_YEARS"])
fit_end_years = float(os.environ["FIT_END_YEARS"])
record_every_years = float(os.environ["RECORD_EVERY_YEARS"])
steps = [float(value) for value in os.environ["STEP_DAYS_VALUES"].split()]
perturbations = [float(value) for value in os.environ["PERTURBATION_M_VALUES"].split()]
renorms = [float(value) for value in os.environ["RENORM_YEARS_VALUES"].split()]
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None

fieldnames = [
    "step_days",
    "perturbation_m",
    "renorm_years",
    "lambda_1_per_year",
    "lyapunov_time_years",
    "r_squared",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "mean_log_growth_increment",
    "max_log_growth_increment",
    "post_renorm_norm_rel_error_max",
    "direction_reset_suspected_count",
    "warning_count",
    "main_warnings",
]


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")


def read_json(path: Path) -> dict:
    with path.open() as file_obj:
        return json.load(file_obj)


def read_first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return next(reader)


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def lyapunov_csv_stats(path: Path) -> dict[str, float | int | str]:
    log_increments: list[float] = []
    post_errors: list[float] = []
    direction_reset_count = 0
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                log_increments.append(float(row["log_growth_increment"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                post = float(row["post_renorm_separation_norm"])
                target = float(row["target_norm"])
                if target > 0.0:
                    post_errors.append(abs(post - target) / target)
            except (KeyError, TypeError, ValueError):
                pass
            if str(row.get("direction_reset_suspected", "")).strip() in {"1", "true", "True"}:
                direction_reset_count += 1
    return {
        "mean_log_growth_increment": (
            sum(log_increments) / len(log_increments) if log_increments else ""
        ),
        "max_log_growth_increment": (
            max((abs(value) for value in log_increments), default="")
        ),
        "post_renorm_norm_rel_error_max": max(post_errors, default=""),
        "direction_reset_suspected_count": direction_reset_count,
    }


ladder_csv.parent.mkdir(parents=True, exist_ok=True)
with ladder_csv.open("w", newline="") as matrix_file:
    writer = csv.DictWriter(matrix_file, fieldnames=fieldnames)
    writer.writeheader()

    case_count = 0
    for step_days in steps:
        for perturbation_m in perturbations:
            for renorm_years in renorms:
                case_count += 1
                if max_cases is not None and case_count > max_cases:
                    break

                tag = (
                    "two_body_mercury_lyap_"
                    f"step{tag_float(step_days)}d_"
                    f"pert{tag_float(perturbation_m)}m_"
                    f"renorm{tag_float(renorm_years)}yr"
                )
                cmd = [
                    python_bin,
                    "-m",
                    "mini_ephemeris.long_term_stability_cli",
                    "--kernel-path",
                    kernel_path,
                    "--start-date",
                    "2000-01-01",
                    "--duration-years",
                    f"{duration_years:g}",
                    "--step-days",
                    f"{step_days:g}",
                    "--record-every-years",
                    f"{record_every_years:g}",
                    "--model-scope",
                    "two_body_mercury",
                    "--gr-model",
                    "none",
                    "--integrator",
                    "leapfrog",
                    "--output-dir",
                    str(run_output_dir),
                    "--tag",
                    tag,
                    "--with-lyapunov",
                    "--lyapunov-body",
                    "mercury",
                    "--lyapunov-perturbation-m",
                    f"{perturbation_m:g}",
                    "--lyapunov-renorm-years",
                    f"{renorm_years:g}",
                    "--lyapunov-fit-start-years",
                    f"{fit_start_years:g}",
                    "--lyapunov-fit-end-years",
                    f"{fit_end_years:g}",
                    "--lyapunov-debug",
                    "--no-progress-bar",
                ]
                print(
                    f"[{case_count:03d}] step={step_days:g} d "
                    f"pert={perturbation_m:g} m renorm={renorm_years:g} yr",
                    flush=True,
                )
                completed = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                (run_output_dir / f"{tag}.log").write_text(completed.stdout)

                row = {name: "" for name in fieldnames}
                row.update(
                    {
                        "step_days": step_days,
                        "perturbation_m": perturbation_m,
                        "renorm_years": renorm_years,
                    }
                )

                if completed.returncode == 0:
                    lyap_summary = read_json(run_output_dir / f"lyapunov_summary_{tag}.json")
                    validation = read_first_csv_row(
                        run_output_dir / f"two_body_validation_{tag}.csv"
                    )
                    csv_stats = lyapunov_csv_stats(run_output_dir / f"lyapunov_{tag}.csv")
                    fit = lyap_summary.get("fit", {})
                    warnings = list(lyap_summary.get("warnings", []))
                    row.update(
                        {
                            "lambda_1_per_year": finite_or_blank(fit.get("lambda_1_per_year")),
                            "lyapunov_time_years": finite_or_blank(fit.get("lyapunov_time_years")),
                            "r_squared": finite_or_blank(fit.get("r_squared")),
                            "max_energy_rel_drift": validation.get("max_energy_rel_drift", ""),
                            "max_angular_momentum_rel_drift": validation.get("max_angular_momentum_rel_drift", ""),
                            "mean_log_growth_increment": finite_or_blank(csv_stats["mean_log_growth_increment"]),
                            "max_log_growth_increment": finite_or_blank(csv_stats["max_log_growth_increment"]),
                            "post_renorm_norm_rel_error_max": finite_or_blank(csv_stats["post_renorm_norm_rel_error_max"]),
                            "direction_reset_suspected_count": csv_stats["direction_reset_suspected_count"],
                            "warning_count": len(warnings),
                            "main_warnings": "; ".join(warnings[:4]),
                        }
                    )
                else:
                    row["warning_count"] = 1
                    row["main_warnings"] = f"run_failed_returncode_{completed.returncode}"
                    print(
                        f"  run failed with return code {completed.returncode}; see {run_output_dir / (tag + '.log')}",
                        flush=True,
                    )

                writer.writerow(row)
                matrix_file.flush()

            if max_cases is not None and case_count >= max_cases:
                break
        if max_cases is not None and case_count >= max_cases:
            break

print(f"Wrote {ladder_csv}")
PY
