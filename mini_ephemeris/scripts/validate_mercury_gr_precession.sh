#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
DURATION_YEARS="${DURATION_YEARS:-100}"
STEP_DAYS="${STEP_DAYS:-0.25}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-1}"
INTEGRATOR="${INTEGRATOR:-ias15}"
SUMMARY_CSV="${OUTPUT_DIR}/mercury_gr_precession_validation.csv"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c "import rebound" >/dev/null 2>&1; then
  echo "gr_model,status,runtime_seconds,max_energy_rel_drift,max_angular_momentum_rel_drift,mercury_perihelion_drift_arcsec_per_century,summary_path" > "${SUMMARY_CSV}"
  echo "none,rebound_not_installed,,,,," >> "${SUMMARY_CSV}"
  echo "[GR] rebound is not installed; wrote ${SUMMARY_CSV}"
  exit 0
fi

GR_MODELS=(none)
if "${PYTHON_BIN}" -c "import reboundx" >/dev/null 2>&1; then
  GR_MODELS+=(gr gr_potential gr_full)
else
  echo "[GR] reboundx is not installed; running Newtonian baseline only."
fi

for GR_MODEL in "${GR_MODELS[@]}"; do
  TAG="mercury_gr_${GR_MODEL}_${DURATION_YEARS}yr"
  echo "[GR] running ${GR_MODEL}"
  if ! "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_mercury \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --integrator "${INTEGRATOR}" \
    --gr-model "${GR_MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}"
  then
    echo "[GR] ${GR_MODEL} failed; recording missing summary row."
  fi
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${DURATION_YEARS}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
duration_years = sys.argv[2]
summary_csv = Path(sys.argv[3])
rows = []
for gr_model in ["none", "gr", "gr_potential", "gr_full"]:
    path = output_dir / f"rebound_validation_summary_mercury_gr_{gr_model}_{duration_years}yr.json"
    if not path.exists():
        rows.append({
            "gr_model": gr_model,
            "status": "not_run_or_failed",
            "runtime_seconds": "",
            "max_energy_rel_drift": "",
            "max_angular_momentum_rel_drift": "",
            "mercury_perihelion_drift_arcsec_per_century": "",
            "summary_path": "",
        })
        continue
    with path.open() as file_obj:
        summary = json.load(file_obj)
    diagnostics = summary.get("diagnostics", {})
    rows.append({
        "gr_model": gr_model,
        "status": "ok",
        "runtime_seconds": diagnostics.get("runtime_seconds", ""),
        "max_energy_rel_drift": diagnostics.get("max_energy_rel_drift", ""),
        "max_angular_momentum_rel_drift": diagnostics.get("max_angular_momentum_rel_drift", ""),
        "mercury_perihelion_drift_arcsec_per_century": diagnostics.get("mercury_perihelion_drift_arcsec_per_century", ""),
        "summary_path": str(path),
    })

with summary_csv.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=[
        "gr_model",
        "status",
        "runtime_seconds",
        "max_energy_rel_drift",
        "max_angular_momentum_rel_drift",
        "mercury_perihelion_drift_arcsec_per_century",
        "summary_path",
    ])
    writer.writeheader()
    writer.writerows(rows)
print(f"[GR] wrote {summary_csv}")
PY
