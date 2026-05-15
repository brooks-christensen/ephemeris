#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/inner_10kyr_duration_runs}"
LADDER_CSV="${LADDER_CSV:-${OUTPUT_BASE}/inner_10kyr_duration_ladder.csv}"
SCALING_CSV="${SCALING_CSV:-${OUTPUT_BASE}/inner_10kyr_duration_scaling_summary.csv}"
SCALING_JSON="${SCALING_JSON:-${OUTPUT_BASE}/inner_10kyr_duration_scaling_summary.json}"
STEP_DAYS="${STEP_DAYS:-0.25}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
MAX_CASES="${MAX_CASES:-}"
RESUME="${RESUME:-1}"

mkdir -p "${RUN_OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH OUTPUT_BASE RUN_OUTPUT_DIR LADDER_CSV SCALING_CSV SCALING_JSON
export STEP_DAYS RECORD_EVERY_YEARS MAX_CASES RESUME

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


python_bin = os.environ["PYTHON_BIN"]
kernel_path = os.environ["KERNEL_PATH"]
output_base = Path(os.environ["OUTPUT_BASE"])
run_output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
ladder_csv = Path(os.environ["LADDER_CSV"])
scaling_csv = Path(os.environ["SCALING_CSV"])
scaling_json = Path(os.environ["SCALING_JSON"])
step_days = float(os.environ.get("STEP_DAYS", "0.25"))
record_every_years = float(os.environ.get("RECORD_EVERY_YEARS", "100"))
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None
resume = os.environ.get("RESUME", "1") != "0"

durations = [100.0, 300.0, 1000.0, 3000.0, 10000.0]
renorms = [0.1, 0.25, 1.0]
bodies = ["mercury", "all"]
perturbations = [1.0, 1000.0]

ladder_fields = [
    "model_scope",
    "step_days",
    "duration_years",
    "renorm_years",
    "lyapunov_body",
    "perturbation_m",
    "lambda_1_per_year",
    "lyapunov_time_years",
    "accumulated_log_growth_final",
    "lambda_times_duration",
    "r_squared",
    "mean_log_growth_increment",
    "max_log_growth_increment",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "runtime_seconds",
    "warning_count",
    "main_warnings",
]

scaling_fields = [
    "model_scope",
    "step_days",
    "renorm_years",
    "lyapunov_body",
    "perturbation_m",
    "n_points",
    "duration_years_min",
    "duration_years_max",
    "best_fit_model",
    "estimated_lambda_infinite_time",
    "estimated_lambda_infinite_time_uncertainty",
    "slope_accumulated_log_growth_vs_time",
    "slope_accumulated_log_growth_vs_log_time",
    "r2_lambda_vs_inv_duration",
    "r2_lambda_vs_log_duration_over_duration",
    "r2_growth_vs_time",
    "r2_growth_vs_log_time",
    "classification",
    "reason",
]


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite_or_blank(value):
    number = safe_float(value)
    return number if math.isfinite(number) else ""


def read_json(path: Path) -> dict:
    with path.open() as file_obj:
        return json.load(file_obj)


def tag_for(duration: float, renorm: float, body: str, perturbation: float) -> str:
    return (
        "inner_10kyr_duration_"
        f"dur{tag_float(duration)}yr_"
        f"step{tag_float(step_days)}d_"
        f"body{body}_"
        f"pert{tag_float(perturbation)}m_"
        f"renorm{tag_float(renorm)}yr"
    )


def lyapunov_csv_stats(path: Path) -> dict[str, float | str]:
    increments: list[float] = []
    final_growth = math.nan
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                increments.append(float(row["log_growth_increment"]))
                final_growth = float(row["cumulative_log_growth"])
            except (KeyError, TypeError, ValueError):
                pass
    return {
        "mean_log_growth_increment": sum(increments) / len(increments) if increments else "",
        "max_log_growth_increment": max((abs(value) for value in increments), default=""),
        "accumulated_log_growth_final": final_growth if math.isfinite(final_growth) else "",
    }


def build_row(duration: float, renorm: float, body: str, perturbation: float, tag: str, returncode: int) -> dict:
    row = {name: "" for name in ladder_fields}
    row.update(
        {
            "model_scope": "inner",
            "step_days": step_days,
            "duration_years": duration,
            "renorm_years": renorm,
            "lyapunov_body": body,
            "perturbation_m": perturbation,
        }
    )
    if returncode != 0:
        row["warning_count"] = 1
        row["main_warnings"] = f"run_failed_returncode_{returncode}"
        return row

    summary = read_json(run_output_dir / f"summary_{tag}.json")
    lyap_summary = read_json(run_output_dir / f"lyapunov_summary_{tag}.json")
    stats = lyapunov_csv_stats(run_output_dir / f"lyapunov_{tag}.csv")
    fit = lyap_summary.get("fit", {})
    warnings = list(lyap_summary.get("warnings", []))
    extrema = summary.get("diagnostic_extrema_over_records", {})
    lambda_value = safe_float(fit.get("lambda_1_per_year"))
    accumulated = safe_float(stats["accumulated_log_growth_final"])
    row.update(
        {
            "lambda_1_per_year": finite_or_blank(lambda_value),
            "lyapunov_time_years": finite_or_blank(fit.get("lyapunov_time_years")),
            "accumulated_log_growth_final": finite_or_blank(accumulated),
            "lambda_times_duration": lambda_value * duration if math.isfinite(lambda_value) else "",
            "r_squared": finite_or_blank(fit.get("r_squared")),
            "mean_log_growth_increment": finite_or_blank(stats["mean_log_growth_increment"]),
            "max_log_growth_increment": finite_or_blank(stats["max_log_growth_increment"]),
            "max_energy_rel_drift": finite_or_blank(extrema.get("max_abs_energy_rel_drift")),
            "max_angular_momentum_rel_drift": finite_or_blank(extrema.get("max_angular_momentum_rel_drift")),
            "runtime_seconds": finite_or_blank(summary.get("runtime", {}).get("wall_clock_seconds")),
            "warning_count": len(warnings),
            "main_warnings": "; ".join(warnings[:4]),
        }
    )
    return row


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"slope": math.nan, "intercept": math.nan, "r2": math.nan, "sigma_intercept": math.nan}
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx == 0.0:
        return {"slope": 0.0, "intercept": y_mean, "r2": math.nan, "sigma_intercept": math.nan}
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    sigma_intercept = math.nan
    if x.size > 2:
        s2 = ss_res / (x.size - 2)
        sigma_intercept = math.sqrt(s2 * (1.0 / x.size + x_mean**2 / sxx))
    return {"slope": slope, "intercept": intercept, "r2": r2, "sigma_intercept": sigma_intercept}


def classify_group(rows: list[dict]) -> dict:
    finite = [
        row for row in rows
        if math.isfinite(safe_float(row["lambda_1_per_year"]))
        and math.isfinite(safe_float(row["accumulated_log_growth_final"]))
    ]
    if len(finite) < 3:
        return {"best_fit_model": "ambiguous", "classification": "ambiguous", "reason": "fewer than three finite duration samples"}
    durations_arr = np.array([safe_float(row["duration_years"]) for row in finite], dtype=float)
    lambdas = np.array([safe_float(row["lambda_1_per_year"]) for row in finite], dtype=float)
    growth = np.array([safe_float(row["accumulated_log_growth_final"]) for row in finite], dtype=float)
    fit_inv = linear_fit(1.0 / durations_arr, lambdas)
    fit_log_over_t = linear_fit(np.log(durations_arr) / durations_arr, lambdas)
    fit_growth_time = linear_fit(durations_arr, growth)
    fit_growth_log = linear_fit(np.log(durations_arr), growth)
    candidates = {
        "one_over_T": fit_inv["r2"],
        "logT_over_T": fit_log_over_t["r2"],
        "constant_plateau": 0.0,
    }
    finite_candidates = {key: value for key, value in candidates.items() if math.isfinite(value)}
    best = max(finite_candidates, key=finite_candidates.get) if finite_candidates else "ambiguous"
    estimate = math.nan
    uncertainty = math.nan
    if best == "one_over_T":
        estimate = fit_inv["intercept"]
        uncertainty = fit_inv["sigma_intercept"]
    elif best == "logT_over_T":
        estimate = fit_log_over_t["intercept"]
        uncertainty = fit_log_over_t["sigma_intercept"]
    else:
        estimate = float(np.mean(lambdas))
        uncertainty = float(np.std(lambdas, ddof=1)) if lambdas.size > 1 else math.nan
    max_lambda = float(np.max(np.abs(lambdas)))
    longest_lambda = float(lambdas[np.argmax(durations_arr)])
    if best in {"one_over_T", "logT_over_T"} and abs(longest_lambda) < 0.5 * max_lambda and (not math.isfinite(estimate) or abs(estimate) < 0.25 * max_lambda):
        classification = "near_integrable_likely"
        reason = f"finite-time lambda falls with duration; {best} beats a constant plateau."
    elif best == "constant_plateau" and math.isfinite(estimate) and estimate > 0.0 and (not math.isfinite(uncertainty) or estimate > 3.0 * uncertainty):
        classification = "chaotic_candidate"
        reason = "duration scaling is most consistent with a nonzero constant plateau."
    elif fit_growth_log["r2"] > fit_growth_time["r2"] + 0.05 and abs(longest_lambda) < 0.75 * max_lambda:
        best = "logT_over_T"
        estimate = fit_log_over_t["intercept"]
        uncertainty = fit_log_over_t["sigma_intercept"]
        classification = "near_integrable_likely"
        reason = "accumulated log growth is better explained by log(duration) than time."
    else:
        classification = "ambiguous"
        reason = "duration trend does not cleanly separate finite-time shear from a plateau."
    return {
        "best_fit_model": best,
        "estimated_lambda_infinite_time": estimate,
        "estimated_lambda_infinite_time_uncertainty": uncertainty,
        "slope_accumulated_log_growth_vs_time": fit_growth_time["slope"],
        "slope_accumulated_log_growth_vs_log_time": fit_growth_log["slope"],
        "r2_lambda_vs_inv_duration": fit_inv["r2"],
        "r2_lambda_vs_log_duration_over_duration": fit_log_over_t["r2"],
        "r2_growth_vs_time": fit_growth_time["r2"],
        "r2_growth_vs_log_time": fit_growth_log["r2"],
        "classification": classification,
        "reason": reason,
    }


def write_scaling(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["step_days"]), str(row["renorm_years"]), str(row["lyapunov_body"]), str(row["perturbation_m"]))
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (step, renorm, body, perturbation), group in sorted(grouped.items()):
        finite_durations = [safe_float(row["duration_years"]) for row in group if math.isfinite(safe_float(row["lambda_1_per_year"]))]
        classification = classify_group(group)
        row = {name: "" for name in scaling_fields}
        row.update(
            {
                "model_scope": "inner",
                "step_days": step,
                "renorm_years": renorm,
                "lyapunov_body": body,
                "perturbation_m": perturbation,
                "n_points": len(finite_durations),
                "duration_years_min": min(finite_durations) if finite_durations else "",
                "duration_years_max": max(finite_durations) if finite_durations else "",
            }
        )
        for key, value in classification.items():
            row[key] = finite_or_blank(value) if isinstance(value, (float, int)) else value
        summary_rows.append(row)
    with scaling_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=scaling_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with scaling_json.open("w") as file_obj:
        json.dump({"groups": summary_rows}, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def estimate_seconds(duration: float) -> float:
    return 141.0 * (duration / 1000.0) * (0.25 / step_days)


start_wall = time.perf_counter()
rows: list[dict] = []
case_count = 0
print("[Run] Inner 10 kyr duration-scaling ladder")
print("[Run] finite-time diagnostics only; broad interpretation requires duration and timestep convergence")
print(f"[Run] step_days={step_days}; RESUME={int(resume)}; MAX_CASES={max_cases if max_cases is not None else 'all'}")
for renorm in renorms:
    for body in bodies:
        for perturbation in perturbations:
            for duration in durations:
                if max_cases is not None and case_count >= max_cases:
                    break
                tag = tag_for(duration, renorm, body, perturbation)
                summary_path = run_output_dir / f"summary_{tag}.json"
                print(f"[Run] case {case_count + 1}: duration={duration:g} yr renorm={renorm:g} body={body} perturb={perturbation:g} m estimated={estimate_seconds(duration):.1f} s")
                if resume and summary_path.exists():
                    print(f"[Run] skip existing {summary_path}")
                    returncode = 0
                else:
                    cmd = [
                        python_bin,
                        "-m",
                        "mini_ephemeris.long_term_stability_cli",
                        "--kernel-path",
                        kernel_path,
                        "--start-date",
                        "2000-01-01",
                        "--duration-years",
                        f"{duration:g}",
                        "--step-days",
                        f"{step_days:g}",
                        "--record-every-years",
                        f"{record_every_years:g}",
                        "--gr-model",
                        "none",
                        "--integrator",
                        "leapfrog",
                        "--model-scope",
                        "inner",
                        "--output-dir",
                        str(run_output_dir),
                        "--tag",
                        tag,
                        "--with-lyapunov",
                        "--lyapunov-method",
                        "tangent",
                        "--lyapunov-body",
                        body,
                        "--lyapunov-perturbation-m",
                        f"{perturbation:g}",
                        "--lyapunov-renorm-years",
                        f"{renorm:g}",
                        "--lyapunov-fit-start-years",
                        f"{min(renorm, 0.2 * duration):g}",
                        "--lyapunov-fit-end-years",
                        f"{duration:g}",
                        "--lyapunov-debug",
                        "--no-progress-bar",
                    ]
                    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    (run_output_dir / f"{tag}.log").write_text(completed.stdout)
                    returncode = completed.returncode
                rows.append(build_row(duration, renorm, body, perturbation, tag, returncode))
                case_count += 1
            if max_cases is not None and case_count >= max_cases:
                break
        if max_cases is not None and case_count >= max_cases:
            break
    if max_cases is not None and case_count >= max_cases:
        break

with ladder_csv.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=ladder_fields)
    writer.writeheader()
    writer.writerows(rows)
write_scaling(rows)
elapsed = time.perf_counter() - start_wall
print(f"[Run] wrote {ladder_csv}")
print(f"[Run] wrote {scaling_csv}")
print(f"[Run] wrote {scaling_json}")
print(f"[Run] elapsed_seconds={elapsed:.3f}")
PY
