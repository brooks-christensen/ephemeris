#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/final_benettin_v2/integrable_mercury_10kyr}"
DURATION_YEARS="${DURATION_YEARS:-10000}"
STEP_DAYS="${STEP_DAYS:-0.125}"
RENORM_YEARS_LIST="${RENORM_YEARS_LIST:-1 10 100}"
PERTURBATION_M_LIST="${PERTURBATION_M_LIST:-0.01 1}"
SEED="${SEED:-12345}"
MAX_CASES="${MAX_CASES:-}"
RESUME="${RESUME:-1}"
SUMMARY_CSV="${OUTPUT_DIR}/integrable_mercury_10kyr_control.csv"
SUMMARY_JSON="${OUTPUT_DIR}/integrable_mercury_10kyr_control.json"
RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS=(--resume-latest)
fi

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

printf "renorm_years,perturbation_m,duration_years,step_days,final_lcn_1_per_year,late_window_median_lcn_1_per_year,classification_hint,max_abs_reference_relative_energy_error,max_abs_reference_relative_angular_momentum_error,stable_positive_late_time_plateau,lcn_trends_toward_zero,summary_path\n" > "${SUMMARY_CSV}"

case_count=0
for renorm in ${RENORM_YEARS_LIST}; do
  for perturb in ${PERTURBATION_M_LIST}; do
    if [[ -n "${MAX_CASES}" && "${case_count}" -ge "${MAX_CASES}" ]]; then
      echo "[integrable] MAX_CASES=${MAX_CASES} reached"
      break 2
    fi
    case_count=$((case_count + 1))
    tag="integrable_mercury_${DURATION_YEARS}yr_dt${STEP_DAYS}_renorm${renorm}_pert${perturb}_seed${SEED}"
    safe_tag="${tag//./p}"
    case_dir="${OUTPUT_DIR}/${safe_tag}"
    summary="${case_dir}/benettin_summary_${safe_tag}.json"
    progress="${case_dir}/benettin_progress_${safe_tag}.csv"
    if [[ -f "${summary}" && "${RESUME}" == "1" ]]; then
      echo "[integrable] skip existing ${safe_tag}"
    else
      echo "[integrable] run ${safe_tag}"
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
        "${RESUME_ARGS[@]}" \
        --progress-file-every-renorm \
        --status-every-renorm \
        --progress-line-every-seconds 120 \
        --status-file-every-seconds 120 \
        --with-standalone-reference-check \
        --no-progress-bar
    fi
    "${PYTHON_BIN}" - "${summary}" "${progress}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
progress_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
data = json.loads(summary_path.read_text())
config = data["configuration"]
samples = []
with progress_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            t = float(row["time_years"])
            lcn = float(row["finite_time_lcn_1_per_year"])
        except (KeyError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(lcn):
            samples.append((t, abs(lcn)))
if len(samples) >= 4:
    split = samples[0][0] + 0.5 * (samples[-1][0] - samples[0][0])
    early = [v for t, v in samples if t <= split]
    late = [v for t, v in samples if t >= split]
    early_median = sorted(early)[len(early)//2] if early else math.inf
    late_median = sorted(late)[len(late)//2] if late else math.inf
    trends = late_median <= early_median or late_median <= 1e-5
else:
    trends = False
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
    "lcn_trends_toward_zero": trends,
    "summary_path": str(summary_path),
}
with csv_path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writerow(row)
PY
  done
done

"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open(newline="")))
failures = []
for row in rows:
    try:
        lcn = abs(float(row["final_lcn_1_per_year"]))
        d_e = abs(float(row["max_abs_reference_relative_energy_error"]))
        d_l = abs(float(row["max_abs_reference_relative_angular_momentum_error"]))
    except ValueError:
        failures.append((row, "non-finite metric"))
        continue
    if row["stable_positive_late_time_plateau"] == "True":
        failures.append((row, "stable positive plateau"))
    if row["lcn_trends_toward_zero"] != "True":
        failures.append((row, "LCN did not trend toward zero"))
    if lcn > 1e-5:
        failures.append((row, f"abs LCN {lcn:.6g} > 1e-5"))
    if d_e > 1e-6:
        failures.append((row, f"energy drift {d_e:.6g} > 1e-6"))
    if d_l > 1e-10:
        failures.append((row, f"angular momentum drift {d_l:.6g} > 1e-10"))
payload = {
    "passed": not failures,
    "case_count": len(rows),
    "failure_count": len(failures),
    "failures": [
        {
            "renorm_years": item[0].get("renorm_years"),
            "perturbation_m": item[0].get("perturbation_m"),
            "reason": item[1],
        }
        for item in failures
    ],
    "summary_csv": str(csv_path),
}
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[integrable] wrote {csv_path}")
print(f"[integrable] wrote {json_path}")
if failures:
    raise SystemExit(1)
PY
