#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/lyapunov_validation_runs}"
MATRIX_CSV="${MATRIX_CSV:-${OUTPUT_BASE}/lyapunov_validation_matrix.csv}"
DURATION_YEARS="${DURATION_YEARS:-1000}"
STEP_DAYS="${STEP_DAYS:-4}"
FIT_START_YEARS="${FIT_START_YEARS:-100}"
FIT_END_YEARS="${FIT_END_YEARS:-1000}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
MAX_CASES="${MAX_CASES:-}"

mkdir -p "${RUN_OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH OUTPUT_BASE RUN_OUTPUT_DIR MATRIX_CSV
export DURATION_YEARS STEP_DAYS FIT_START_YEARS FIT_END_YEARS RECORD_EVERY_YEARS MAX_CASES

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


python_bin = os.environ.get("PYTHON_BIN", "/home/peacelovephysics/ephemeris/.venv/bin/python")
kernel_path = os.environ.get("KERNEL_PATH", "/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
run_output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
matrix_csv = Path(os.environ["MATRIX_CSV"])
duration_years = float(os.environ.get("DURATION_YEARS", "1000"))
step_days = float(os.environ.get("STEP_DAYS", "4"))
fit_start_years = float(os.environ.get("FIT_START_YEARS", "100"))
fit_end_years = float(os.environ.get("FIT_END_YEARS", str(duration_years)))
record_every_years = float(os.environ.get("RECORD_EVERY_YEARS", "100"))
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None

model_scopes = ["two_body_mercury", "inner", "full"]
gr_models = ["none", "sun"]
renorm_years_values = [0.25, 1.0, 5.0, 10.0, 100.0]
perturbation_values = [1.0e-3, 1.0, 1.0e3]

fieldnames = [
    "model_scope",
    "gr_model",
    "step_days",
    "perturbation_m",
    "renorm_years",
    "lambda_1_per_year",
    "lyapunov_time_years",
    "r_squared",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "max_com_velocity_drift_au_per_year",
    "warning_count",
    "main_warnings",
]


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return value if math.isfinite(value) else ""


def read_json(path: Path) -> dict:
    with path.open() as file_obj:
        return json.load(file_obj)


rows: list[dict] = []
case_count = 0

for model_scope in model_scopes:
    for gr_model in gr_models:
        for renorm_years in renorm_years_values:
            for perturbation_m in perturbation_values:
                case_count += 1
                if max_cases is not None and case_count > max_cases:
                    break

                tag = (
                    f"lyap_validation_{model_scope}_gr{gr_model}_"
                    f"step{step_days:g}d_pert{perturbation_m:g}_renorm{renorm_years:g}"
                )
                tag = tag.replace(".", "p").replace("+", "").replace("-", "m")

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
                    gr_model,
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
                    f"[{case_count:03d}] {model_scope} gr={gr_model} "
                    f"pert={perturbation_m:g} renorm={renorm_years:g}",
                    flush=True,
                )

                completed = subprocess.run(
                    cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                log_path = run_output_dir / f"{tag}.log"
                log_path.write_text(completed.stdout)

                row = {
                    "model_scope": model_scope,
                    "gr_model": gr_model,
                    "step_days": step_days,
                    "perturbation_m": perturbation_m,
                    "renorm_years": renorm_years,
                    "lambda_1_per_year": "",
                    "lyapunov_time_years": "",
                    "r_squared": "",
                    "max_energy_rel_drift": "",
                    "max_angular_momentum_rel_drift": "",
                    "max_com_velocity_drift_au_per_year": "",
                    "warning_count": "",
                    "main_warnings": "",
                }

                if completed.returncode != 0:
                    row["warning_count"] = 1
                    row["main_warnings"] = f"run_failed_returncode_{completed.returncode}"
                    rows.append(row)
                    continue

                lyap_summary_path = run_output_dir / f"lyapunov_summary_{tag}.json"
                summary_path = run_output_dir / f"summary_{tag}.json"
                lyap_summary = read_json(lyap_summary_path)
                summary = read_json(summary_path)

                fit = lyap_summary.get("fit", {})
                extrema = summary.get("diagnostic_extrema_over_records", {})
                warnings = list(lyap_summary.get("warnings", []))

                row.update(
                    {
                        "lambda_1_per_year": finite_or_blank(fit.get("lambda_1_per_year")),
                        "lyapunov_time_years": finite_or_blank(fit.get("lyapunov_time_years")),
                        "r_squared": finite_or_blank(fit.get("r_squared")),
                        "max_energy_rel_drift": finite_or_blank(extrema.get("max_abs_energy_rel_drift")),
                        "max_angular_momentum_rel_drift": finite_or_blank(extrema.get("max_angular_momentum_rel_drift")),
                        "max_com_velocity_drift_au_per_year": finite_or_blank(extrema.get("max_com_velocity_drift_au_per_year")),
                        "warning_count": len(warnings),
                        "main_warnings": "; ".join(warnings[:4]),
                    }
                )
                rows.append(row)

            if max_cases is not None and case_count >= max_cases:
                break
        if max_cases is not None and case_count >= max_cases:
            break
    if max_cases is not None and case_count >= max_cases:
        break


def positive_finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value) and value > 0.0:
        return value
    return None


grouped: dict[tuple, list[dict]] = {}
for row in rows:
    grouped.setdefault(
        (row["model_scope"], row["gr_model"], row["step_days"]),
        [],
    ).append(row)

for group_rows in grouped.values():
    lambdas = [
        value
        for value in (positive_finite(row["lambda_1_per_year"]) for row in group_rows)
        if value is not None
    ]
    if len(lambdas) >= 2:
        min_lambda = min(lambdas)
        max_lambda = max(lambdas)
        if min_lambda > 0.0 and max_lambda / min_lambda > 10.0:
            for row in group_rows:
                suffix = "lambda varies >10x across perturbation/renormalization settings"
                row["main_warnings"] = (
                    f"{row['main_warnings']}; {suffix}"
                    if row["main_warnings"]
                    else suffix
                )
                try:
                    row["warning_count"] = int(row["warning_count"]) + 1
                except (TypeError, ValueError):
                    row["warning_count"] = 1

for none_row in [row for row in rows if row.get("gr_model") == "none"]:
    scope = none_row["model_scope"]
    perturbation = none_row["perturbation_m"]
    renorm = none_row["renorm_years"]
    if none_row.get("gr_model") != "none":
        continue
    sun_row = next(
        (
            row
            for row in rows
            if row["model_scope"] == scope
            and row["gr_model"] == "sun"
            and row["perturbation_m"] == perturbation
            and row["renorm_years"] == renorm
        ),
        None,
    )
    if sun_row is None:
        continue

    none_ang = positive_finite(none_row["max_angular_momentum_rel_drift"]) or 0.0
    sun_ang = positive_finite(sun_row["max_angular_momentum_rel_drift"]) or 0.0
    none_com = positive_finite(none_row["max_com_velocity_drift_au_per_year"]) or 0.0
    sun_com = positive_finite(sun_row["max_com_velocity_drift_au_per_year"]) or 0.0
    messages = []
    if sun_ang > max(1.0e-12, 100.0 * none_ang):
        messages.append("gr_model=sun angular momentum drift is >100x matching gr_model=none case")
    if sun_com > max(1.0e-14, 100.0 * none_com):
        messages.append("gr_model=sun COM velocity drift is >100x matching gr_model=none case")
    if messages:
        suffix = "; ".join(messages)
        sun_row["main_warnings"] = (
            f"{sun_row['main_warnings']}; {suffix}"
            if sun_row["main_warnings"]
            else suffix
        )
        try:
            sun_row["warning_count"] = int(sun_row["warning_count"]) + len(messages)
        except (TypeError, ValueError):
            sun_row["warning_count"] = len(messages)

matrix_csv.parent.mkdir(parents=True, exist_ok=True)
with matrix_csv.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {matrix_csv}")
PY
