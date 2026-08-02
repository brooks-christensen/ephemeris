#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_benettin_mercury_parameter_sweep}"
DURATION_YEARS="${DURATION_YEARS:-10000}"
STEP_DAYS="${STEP_DAYS:-0.125}"
MAX_CASES="${MAX_CASES:-}"

RENORM_YEARS_LIST="${RENORM_YEARS_LIST:-1 10 100 1000}"
PERTURBATION_M_LIST="${PERTURBATION_M_LIST:-0.01 1 100}"
SEED="${SEED:-12345}"
SUMMARY_CSV="${OUTPUT_DIR}/benettin_mercury_parameter_sweep.csv"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf "renorm_years,perturbation_m,duration_years,step_days,final_lcn_1_per_year,late_window_median_lcn_1_per_year,classification_hint,max_abs_reference_relative_energy_error,max_abs_reference_relative_angular_momentum_error,stable_positive_late_time_plateau,summary_path\n" > "${SUMMARY_CSV}"
fi

case_count=0
for renorm in ${RENORM_YEARS_LIST}; do
  for perturb in ${PERTURBATION_M_LIST}; do
    if [[ -n "${MAX_CASES}" && "${case_count}" -ge "${MAX_CASES}" ]]; then
      echo "[sweep] MAX_CASES=${MAX_CASES} reached"
      exit 0
    fi
    case_count=$((case_count + 1))
    tag="mercury_sweep_${DURATION_YEARS}yr_dt${STEP_DAYS}_renorm${renorm}_pert${perturb}_seed${SEED}"
    safe_tag="${tag//./p}"
    case_dir="${OUTPUT_DIR}/${safe_tag}"
    summary="${case_dir}/benettin_summary_${safe_tag}.json"
    if [[ -f "${summary}" && "${RESUME:-1}" == "1" ]]; then
      echo "[sweep] skip existing ${safe_tag}"
    else
      echo "[sweep] run ${safe_tag}"
      "${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli \
        --kernel-path "${KERNEL_PATH}" \
        --start-date 2000-01-01 \
        --duration-years "${DURATION_YEARS}" \
        --step-days "${STEP_DAYS}" \
        --record-every-years "${renorm}" \
        --model-scope two_body_mercury \
        --integrator whfast \
        --gr-model none \
        --perturb-body mercury \
        --perturbation-m "${perturb}" \
        --renorm-years "${renorm}" \
        --fit-start-years 0 \
        --seed "${SEED}" \
        --output-dir "${case_dir}" \
        --tag "${safe_tag}" \
        --checkpoint-every-years "${renorm}" \
        --checkpoint-dir "${case_dir}/dual_checkpoints" \
        --resume-latest \
        --progress-line-every-seconds 120 \
        --status-file-every-seconds 120 \
        --no-progress-bar
    fi
    "${PYTHON_BIN}" - "${summary}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
data = json.loads(summary_path.read_text())
config = data["configuration"]
row = {
    "renorm_years": config["renorm_years"],
    "perturbation_m": config["perturbation_m"],
    "duration_years": data["duration_years"],
    "step_days": config["step_days"],
    "final_lcn_1_per_year": data.get("finite_time_lcn_1_per_year"),
    "late_window_median_lcn_1_per_year": data.get("late_window_median_lcn_1_per_year"),
    "classification_hint": data.get("classification_hint"),
    "max_abs_reference_relative_energy_error": data.get("max_abs_reference_relative_energy_error"),
    "max_abs_reference_relative_angular_momentum_error": data.get("max_abs_reference_relative_angular_momentum_error"),
    "stable_positive_late_time_plateau": data.get("stable_positive_late_time_plateau"),
    "summary_path": str(summary_path),
}
with csv_path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writerow(row)
PY
  done
done

echo "[sweep] wrote ${SUMMARY_CSV}"
