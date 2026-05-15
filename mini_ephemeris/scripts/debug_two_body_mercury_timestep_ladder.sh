#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/peacelovephysics/ephemeris/output/stability}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_BASE}/two_body_mercury_timestep_runs}"
LADDER_CSV="${LADDER_CSV:-${OUTPUT_BASE}/two_body_mercury_timestep_ladder.csv}"
DURATION_YEARS_VALUES="${DURATION_YEARS_VALUES:-100 1000}"
STEP_DAYS_VALUES="${STEP_DAYS_VALUES:-4 2 1 0.5 0.25 0.125}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10}"
MAX_CASES="${MAX_CASES:-}"

mkdir -p "${RUN_OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

export PYTHON_BIN KERNEL_PATH OUTPUT_BASE RUN_OUTPUT_DIR LADDER_CSV
export DURATION_YEARS_VALUES STEP_DAYS_VALUES RECORD_EVERY_YEARS MAX_CASES

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import subprocess


python_bin = os.environ["PYTHON_BIN"]
kernel_path = os.environ["KERNEL_PATH"]
run_output_dir = Path(os.environ["RUN_OUTPUT_DIR"])
ladder_csv = Path(os.environ["LADDER_CSV"])
durations = [float(value) for value in os.environ["DURATION_YEARS_VALUES"].split()]
steps = [float(value) for value in os.environ["STEP_DAYS_VALUES"].split()]
record_every_years = float(os.environ["RECORD_EVERY_YEARS"])
max_cases_text = os.environ.get("MAX_CASES", "").strip()
max_cases = int(max_cases_text) if max_cases_text else None

fieldnames = [
    "step_days",
    "duration_years",
    "n_steps",
    "max_energy_rel_drift",
    "final_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "final_angular_momentum_rel_drift",
    "max_a_drift_au",
    "max_e_drift",
    "estimated_perihelion_drift_arcsec_per_century",
    "runtime_seconds",
]


def tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def read_first_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return next(reader)


ladder_csv.parent.mkdir(parents=True, exist_ok=True)
with ladder_csv.open("w", newline="") as matrix_file:
    writer = csv.DictWriter(matrix_file, fieldnames=fieldnames)
    writer.writeheader()

    case_count = 0
    for duration_years in durations:
        for step_days in steps:
            case_count += 1
            if max_cases is not None and case_count > max_cases:
                break

            tag = (
                "two_body_mercury_timestep_"
                f"{tag_float(duration_years)}yr_step{tag_float(step_days)}d"
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
                "--no-progress-bar",
            ]
            print(
                f"[{case_count:03d}] duration={duration_years:g} yr "
                f"step={step_days:g} d",
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
                    "duration_years": duration_years,
                    "n_steps": math.ceil(duration_years * 365.25 / step_days),
                }
            )
            if completed.returncode == 0:
                validation_path = run_output_dir / f"two_body_validation_{tag}.csv"
                validation = read_first_row(validation_path)
                row.update(
                    {
                        "n_steps": validation.get("n_steps", row["n_steps"]),
                        "max_energy_rel_drift": validation.get("max_energy_rel_drift", ""),
                        "final_energy_rel_drift": validation.get("final_energy_rel_drift", ""),
                        "max_angular_momentum_rel_drift": validation.get("max_angular_momentum_rel_drift", ""),
                        "final_angular_momentum_rel_drift": validation.get("final_angular_momentum_rel_drift", ""),
                        "max_a_drift_au": validation.get("max_a_drift_au", ""),
                        "max_e_drift": validation.get("max_e_drift", ""),
                        "estimated_perihelion_drift_arcsec_per_century": validation.get("estimated_perihelion_drift_arcsec_per_century", ""),
                        "runtime_seconds": validation.get("runtime_seconds", ""),
                    }
                )
            else:
                row["runtime_seconds"] = ""
                print(
                    f"  run failed with return code {completed.returncode}; see {run_output_dir / (tag + '.log')}",
                    flush=True,
                )

            writer.writerow(row)
            matrix_file.flush()

        if max_cases is not None and case_count >= max_cases:
            break

print(f"Wrote {ladder_csv}")
PY
