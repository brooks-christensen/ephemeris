#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/inner_tangent_duration_runs}"
LADDER_CSV="${LADDER_CSV:-${OUTPUT_BASE}/inner_tangent_duration_ladder.csv}"
SCALING_CSV="${SCALING_CSV:-${OUTPUT_BASE}/inner_tangent_duration_scaling_summary.csv}"
SCALING_JSON="${SCALING_JSON:-${OUTPUT_BASE}/inner_tangent_duration_scaling_summary.json}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
MAX_CASES="${MAX_CASES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
INCLUDE_INNER_10000="${INCLUDE_INNER_10000:-0}"

mkdir -p "${RUN_OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH OUTPUT_BASE RUN_OUTPUT_DIR LADDER_CSV SCALING_CSV SCALING_JSON
export RECORD_EVERY_YEARS MAX_CASES SKIP_EXISTING INCLUDE_INNER_10000

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np


python_bin = os.environ["PYTHON_BIN"]
kernel_path = os.environ["KERNEL_PATH"]
output_base = Path(os.environ["OUTPUT_BASE"])
run_output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
ladder_csv = Path(os.environ["LADDER_CSV"])
scaling_csv = Path(os.environ["SCALING_CSV"])
scaling_json = Path(os.environ["SCALING_JSON"])
record_every_years = float(os.environ.get("RECORD_EVERY_YEARS", "100"))
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None
skip_existing = os.environ.get("SKIP_EXISTING", "1") != "0"
include_inner_10000 = os.environ.get("INCLUDE_INNER_10000", "0") == "1"

duration_years_values = [100.0, 300.0, 1000.0, 3000.0]
if include_inner_10000:
    duration_years_values.append(10000.0)

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


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_json(path: Path) -> dict:
    with path.open() as file_obj:
        return json.load(file_obj)


def lyapunov_csv_stats(path: Path) -> dict[str, float | str]:
    increments: list[float] = []
    final_growth = math.nan
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                increments.append(float(row["log_growth_increment"]))
            except (KeyError, TypeError, ValueError):
                pass
            try:
                final_growth = float(row["cumulative_log_growth"])
            except (KeyError, TypeError, ValueError):
                pass
    return {
        "mean_log_growth_increment": (
            sum(increments) / len(increments) if increments else ""
        ),
        "max_log_growth_increment": (
            max((abs(value) for value in increments), default="")
        ),
        "accumulated_log_growth_final": final_growth if math.isfinite(final_growth) else "",
    }


def run_case(cmd: list[str], log_path: Path) -> int:
    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    return completed.returncode


def case_tag(step_days: float, duration_years: float, renorm_years: float, body: str, perturbation_m: float) -> str:
    return (
        "inner_tangent_duration_"
        f"dur{tag_float(duration_years)}yr_"
        f"step{tag_float(step_days)}d_"
        f"body{body}_"
        f"pert{tag_float(perturbation_m)}m_"
        f"renorm{tag_float(renorm_years)}yr"
    )


def build_row(
    *,
    step_days: float,
    duration_years: float,
    renorm_years: float,
    body: str,
    perturbation_m: float,
    tag: str,
    returncode: int,
) -> dict:
    row = {name: "" for name in ladder_fields}
    row.update(
        {
            "model_scope": "inner",
            "step_days": step_days,
            "duration_years": duration_years,
            "renorm_years": renorm_years,
            "lyapunov_body": body,
            "perturbation_m": perturbation_m,
        }
    )
    if returncode != 0:
        row["warning_count"] = 1
        row["main_warnings"] = f"run_failed_returncode_{returncode}"
        return row

    lyap_summary = read_json(run_output_dir / f"lyapunov_summary_{tag}.json")
    summary = read_json(run_output_dir / f"summary_{tag}.json")
    csv_stats = lyapunov_csv_stats(run_output_dir / f"lyapunov_{tag}.csv")
    fit = lyap_summary.get("fit", {})
    warnings = list(lyap_summary.get("warnings", []))
    extrema = summary.get("diagnostic_extrema_over_records", {})
    lambda_value = safe_float(fit.get("lambda_1_per_year"))
    accumulated = safe_float(csv_stats["accumulated_log_growth_final"])
    row.update(
        {
            "lambda_1_per_year": finite_or_blank(lambda_value),
            "lyapunov_time_years": finite_or_blank(fit.get("lyapunov_time_years")),
            "accumulated_log_growth_final": finite_or_blank(accumulated),
            "lambda_times_duration": (
                lambda_value * duration_years if math.isfinite(lambda_value) else ""
            ),
            "r_squared": finite_or_blank(fit.get("r_squared")),
            "mean_log_growth_increment": finite_or_blank(csv_stats["mean_log_growth_increment"]),
            "max_log_growth_increment": finite_or_blank(csv_stats["max_log_growth_increment"]),
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
        return {"n": int(x.size), "slope": math.nan, "intercept": math.nan, "r2": math.nan, "sigma_intercept": math.nan}
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx == 0.0:
        intercept = y_mean
        fitted = np.full_like(y, intercept)
        residual = y - fitted
        ss_res = float(np.sum(residual * residual))
        centered = y - y_mean
        ss_tot = float(np.sum(centered * centered))
        return {
            "n": int(x.size),
            "slope": 0.0,
            "intercept": intercept,
            "r2": math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot,
            "sigma_intercept": float(np.std(y, ddof=1) / math.sqrt(x.size)) if x.size > 1 else math.nan,
        }
    sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual * residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.sum(centered * centered))
    r2 = math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    sigma_intercept = math.nan
    if x.size > 2:
        s2 = ss_res / (x.size - 2)
        sigma_intercept = math.sqrt(s2 * (1.0 / x.size + x_mean**2 / sxx))
    return {"n": int(x.size), "slope": float(slope), "intercept": float(intercept), "r2": r2, "sigma_intercept": sigma_intercept}


def classify_group(group_rows: list[dict]) -> dict:
    finite = [
        row for row in group_rows
        if math.isfinite(safe_float(row["lambda_1_per_year"]))
        and math.isfinite(safe_float(row["accumulated_log_growth_final"]))
    ]
    if len(finite) < 3:
        return {"best_fit_model": "ambiguous", "classification": "ambiguous", "reason": "fewer than three finite duration samples"}

    durations = np.array([safe_float(row["duration_years"]) for row in finite], dtype=float)
    lambdas = np.array([safe_float(row["lambda_1_per_year"]) for row in finite], dtype=float)
    growth = np.array([safe_float(row["accumulated_log_growth_final"]) for row in finite], dtype=float)

    fit_inv = linear_fit(1.0 / durations, lambdas)
    fit_log_over_t = linear_fit(np.log(durations) / durations, lambdas)
    fit_growth_time = linear_fit(durations, growth)
    fit_growth_log = linear_fit(np.log(durations), growth)

    candidates = {
        "constant_plateau": 0.0,
        "one_over_T": fit_inv["r2"],
        "logT_over_T": fit_log_over_t["r2"],
    }
    finite_candidates = {
        name: value for name, value in candidates.items() if math.isfinite(value)
    }
    best_fit_model = (
        max(finite_candidates, key=finite_candidates.get)
        if finite_candidates
        else "ambiguous"
    )
    estimated_lambda = math.nan
    uncertainty = math.nan
    if best_fit_model == "one_over_T":
        estimated_lambda = fit_inv["intercept"]
        uncertainty = fit_inv["sigma_intercept"]
    elif best_fit_model == "logT_over_T":
        estimated_lambda = fit_log_over_t["intercept"]
        uncertainty = fit_log_over_t["sigma_intercept"]
    elif best_fit_model == "constant_plateau":
        estimated_lambda = float(np.mean(lambdas))
        uncertainty = float(np.std(lambdas, ddof=1)) if lambdas.size > 1 else math.nan

    max_lambda = float(np.max(np.abs(lambdas)))
    max_duration = float(np.max(durations))
    min_duration = float(np.min(durations))
    longest_lambda = float(lambdas[np.argmax(durations)])
    if (
        best_fit_model in {"one_over_T", "logT_over_T"}
        and (not math.isfinite(estimated_lambda) or abs(estimated_lambda) < 0.25 * max_lambda)
        and abs(longest_lambda) < 0.5 * max_lambda
    ):
        classification = "near_integrable_likely"
        reason = (
            f"finite-time lambda falls with duration from T={min_duration:g} to {max_duration:g}; "
            f"{best_fit_model} explains lambda_T better than a constant plateau."
        )
    elif (
        best_fit_model == "constant_plateau"
        and math.isfinite(estimated_lambda)
        and estimated_lambda > 0.0
        and (not math.isfinite(uncertainty) or estimated_lambda > 3.0 * uncertainty)
    ):
        classification = "chaotic_candidate"
        reason = "duration scaling is most consistent with a nonzero constant plateau."
    elif (
        fit_growth_log["r2"] > fit_growth_time["r2"] + 0.05
        and abs(longest_lambda) < 0.75 * max_lambda
    ):
        classification = "near_integrable_likely"
        best_fit_model = "logT_over_T"
        estimated_lambda = fit_log_over_t["intercept"]
        uncertainty = fit_log_over_t["sigma_intercept"]
        reason = "accumulated log growth is better explained by log(duration) than by linear time growth."
    else:
        classification = "ambiguous"
        reason = "duration trend does not cleanly separate finite-time shear from a plateau."

    return {
        "best_fit_model": best_fit_model,
        "estimated_lambda_infinite_time": estimated_lambda,
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


def write_scaling_outputs(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row["model_scope"]),
            str(row["step_days"]),
            str(row["renorm_years"]),
            str(row["lyapunov_body"]),
            str(row["perturbation_m"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key, group_rows in sorted(grouped.items()):
        model_scope, step_days, renorm_years, body, perturbation_m = key
        finite_durations = [
            safe_float(row["duration_years"])
            for row in group_rows
            if math.isfinite(safe_float(row["lambda_1_per_year"]))
        ]
        classification = classify_group(group_rows)
        summary_row = {name: "" for name in scaling_fields}
        summary_row.update(
            {
                "model_scope": model_scope,
                "step_days": step_days,
                "renorm_years": renorm_years,
                "lyapunov_body": body,
                "perturbation_m": perturbation_m,
                "n_points": len(finite_durations),
                "duration_years_min": min(finite_durations) if finite_durations else "",
                "duration_years_max": max(finite_durations) if finite_durations else "",
            }
        )
        for name, value in classification.items():
            summary_row[name] = finite_or_blank(value) if isinstance(value, (float, int)) else value
        summary_rows.append(summary_row)

    with scaling_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=scaling_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with scaling_json.open("w") as file_obj:
        json.dump({"groups": summary_rows}, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def main() -> None:
    rows: list[dict] = []
    count = 0
    for step_days in [0.5, 0.25, 0.125]:
        for renorm_years in [0.1, 0.25, 1.0]:
            for body in ["mercury", "all"]:
                for perturbation_m in [1.0, 1000.0]:
                    for duration_years in duration_years_values:
                        if max_cases is not None and count >= max_cases:
                            break
                        tag = case_tag(step_days, duration_years, renorm_years, body, perturbation_m)
                        summary_path = run_output_dir / f"lyapunov_summary_{tag}.json"
                        if skip_existing and summary_path.exists():
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
                                f"{duration_years:g}",
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
                                f"{perturbation_m:g}",
                                "--lyapunov-renorm-years",
                                f"{renorm_years:g}",
                                "--lyapunov-fit-start-years",
                                f"{min(duration_years * 0.2, renorm_years):g}",
                                "--lyapunov-fit-end-years",
                                f"{duration_years:g}",
                                "--lyapunov-debug",
                                "--no-progress-bar",
                            ]
                            returncode = run_case(cmd, run_output_dir / f"{tag}.log")
                        rows.append(
                            build_row(
                                step_days=step_days,
                                duration_years=duration_years,
                                renorm_years=renorm_years,
                                body=body,
                                perturbation_m=perturbation_m,
                                tag=tag,
                                returncode=returncode,
                            )
                        )
                        count += 1
                    if max_cases is not None and count >= max_cases:
                        break
                if max_cases is not None and count >= max_cases:
                    break
            if max_cases is not None and count >= max_cases:
                break
        if max_cases is not None and count >= max_cases:
            break

    with ladder_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ladder_fields)
        writer.writeheader()
        writer.writerows(rows)
    write_scaling_outputs(rows)
    print(f"Wrote {ladder_csv}")
    print(f"Wrote {scaling_csv}")
    print(f"Wrote {scaling_json}")


if __name__ == "__main__":
    main()
PY
