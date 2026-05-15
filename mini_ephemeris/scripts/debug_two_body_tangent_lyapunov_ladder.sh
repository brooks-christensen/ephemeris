#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/two_body_tangent_lyapunov_runs}"
LADDER_CSV="${LADDER_CSV:-${OUTPUT_BASE}/two_body_tangent_lyapunov_ladder.csv}"
DURATION_YEARS="${DURATION_YEARS:-1000}"
FIT_START_YEARS="${FIT_START_YEARS:-100}"
FIT_END_YEARS="${FIT_END_YEARS:-${DURATION_YEARS}}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
MAX_CASES="${MAX_CASES:-}"

mkdir -p "${RUN_OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH RUN_OUTPUT_DIR LADDER_CSV
export DURATION_YEARS FIT_START_YEARS FIT_END_YEARS RECORD_EVERY_YEARS MAX_CASES

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
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None

configs = {
    "two_body_jupiter": {
        "body": "jupiter",
        "steps": [16.0, 8.0, 4.0, 2.0],
        "renorms": [1.0, 5.0, 10.0, 25.0],
        "perturbations": [1.0, 1000.0],
    },
    "two_body_saturn": {
        "body": "saturn",
        "steps": [16.0, 8.0, 4.0, 2.0],
        "renorms": [1.0, 5.0, 10.0, 25.0],
        "perturbations": [1.0, 1000.0],
    },
    "two_body_mercury": {
        "body": "mercury",
        "steps": [1.0, 0.5, 0.25, 0.125, 0.0625],
        "renorms": [0.05, 0.1, 0.25, 1.0],
        "perturbations": [1.0, 1000.0],
    },
}

fieldnames = [
    "model_scope",
    "step_days",
    "duration_years",
    "renorm_years",
    "perturbation_m",
    "lambda_1_per_year",
    "lyapunov_time_years",
    "r_squared",
    "mean_log_growth_increment",
    "max_log_growth_increment",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "final_running_lambda_1_per_year",
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


def lyapunov_csv_stats(path: Path) -> dict[str, float | str]:
    increments: list[float] = []
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                increments.append(float(row["log_growth_increment"]))
            except (KeyError, TypeError, ValueError):
                pass
    return {
        "mean_log_growth_increment": (
            sum(increments) / len(increments) if increments else ""
        ),
        "max_log_growth_increment": (
            max((abs(value) for value in increments), default="")
        ),
    }


ladder_csv.parent.mkdir(parents=True, exist_ok=True)
with ladder_csv.open("w", newline="") as matrix_file:
    writer = csv.DictWriter(matrix_file, fieldnames=fieldnames)
    writer.writeheader()

    case_count = 0
    for model_scope, config in configs.items():
        for step_days in config["steps"]:
            for renorm_years in config["renorms"]:
                for perturbation_m in config["perturbations"]:
                    case_count += 1
                    if max_cases is not None and case_count > max_cases:
                        break

                    tag = (
                        "two_body_tangent_lyap_"
                        f"{model_scope}_step{tag_float(step_days)}d_"
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
                        model_scope,
                        "--gr-model",
                        "none",
                        "--integrator",
                        "leapfrog",
                        "--output-dir",
                        str(run_output_dir),
                        "--tag",
                        tag,
                        "--with-lyapunov",
                        "--lyapunov-method",
                        "tangent",
                        "--lyapunov-body",
                        str(config["body"]),
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
                        f"[{case_count:03d}] {model_scope} step={step_days:g} d "
                        f"renorm={renorm_years:g} yr pert={perturbation_m:g} m",
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
                            "model_scope": model_scope,
                            "step_days": step_days,
                            "duration_years": duration_years,
                            "renorm_years": renorm_years,
                            "perturbation_m": perturbation_m,
                        }
                    )

                    if completed.returncode == 0:
                        lyap_summary = read_json(run_output_dir / f"lyapunov_summary_{tag}.json")
                        validation = read_first_csv_row(
                            run_output_dir / f"two_body_validation_{tag}.csv"
                        )
                        stats = lyapunov_csv_stats(run_output_dir / f"lyapunov_{tag}.csv")
                        fit = lyap_summary.get("fit", {})
                        final_running = lyap_summary.get("final_running_estimate", {})
                        warnings = list(lyap_summary.get("warnings", []))
                        row.update(
                            {
                                "lambda_1_per_year": finite_or_blank(fit.get("lambda_1_per_year")),
                                "lyapunov_time_years": finite_or_blank(fit.get("lyapunov_time_years")),
                                "r_squared": finite_or_blank(fit.get("r_squared")),
                                "mean_log_growth_increment": finite_or_blank(stats["mean_log_growth_increment"]),
                                "max_log_growth_increment": finite_or_blank(stats["max_log_growth_increment"]),
                                "max_energy_rel_drift": validation.get("max_energy_rel_drift", ""),
                                "max_angular_momentum_rel_drift": validation.get("max_angular_momentum_rel_drift", ""),
                                "final_running_lambda_1_per_year": finite_or_blank(final_running.get("lambda_1_per_year")),
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
        if max_cases is not None and case_count >= max_cases:
            break

print(f"Wrote {ladder_csv}")
PY
