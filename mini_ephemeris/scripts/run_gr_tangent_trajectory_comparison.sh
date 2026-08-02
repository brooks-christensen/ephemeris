#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_tangent_v1/trajectory_comparison}"
DURATION_YEARS="${DURATION_YEARS:-100}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-1}"
STEPS="${STEPS:-1 0.5}"
CSV_OUT="${OUTPUT_DIR}/gr_tangent_trajectory_comparison.csv"
JSON_OUT="${OUTPUT_DIR}/gr_tangent_trajectory_comparison.json"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

for step in ${STEPS}; do
  safe_step="${step//./p}"
  custom_tag="custom_gr_tangent_mercury_${DURATION_YEARS}yr_step${safe_step}"
  whfast_tag="reboundx_whfast_gr_potential_mercury_${DURATION_YEARS}yr_step${safe_step}"
  ias15_tag="reboundx_ias15_gr_mercury_${DURATION_YEARS}yr_step${safe_step}"
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_gr_tangent_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_mercury \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${step}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --megno-seed 12345 \
    --gr-scale 1 \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${custom_tag}" \
    --no-progress-bar
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_mercury \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${step}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --integrator whfast \
    --gr-model gr_potential \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${whfast_tag}"
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_mercury \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${step}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --integrator ias15 \
    --gr-model gr \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${ias15_tag}"
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${DURATION_YEARS}" "${CSV_OUT}" "${JSON_OUT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
duration = sys.argv[2]
csv_out = Path(sys.argv[3])
json_out = Path(sys.argv[4])
rows = []
for custom_path in sorted(out.glob(f"gr_tangent_summary_custom_gr_tangent_mercury_{duration}yr_step*.json")):
    step = custom_path.stem.split("step")[-1].replace("p", ".")
    custom = json.loads(custom_path.read_text())
    whfast_path = out / f"rebound_validation_summary_reboundx_whfast_gr_potential_mercury_{duration}yr_step{step.replace('.', 'p')}.json"
    ias15_path = out / f"rebound_validation_summary_reboundx_ias15_gr_mercury_{duration}yr_step{step.replace('.', 'p')}.json"
    whfast = json.loads(whfast_path.read_text()) if whfast_path.exists() else {}
    ias15 = json.loads(ias15_path.read_text()) if ias15_path.exists() else {}
    custom_precession = custom["diagnostics"].get("mercury_perihelion_drift_arcsec_per_century")
    whfast_precession = whfast.get("diagnostics", {}).get("mercury_perihelion_drift_arcsec_per_century")
    ias15_precession = ias15.get("diagnostics", {}).get("mercury_perihelion_drift_arcsec_per_century")
    rows.append({
        "step_days": step,
        "custom_precession_arcsec_per_century": custom_precession,
        "reboundx_whfast_gr_potential_precession_arcsec_per_century": whfast_precession,
        "reboundx_ias15_gr_precession_arcsec_per_century": ias15_precession,
        "custom_minus_whfast_precession": None if custom_precession is None or whfast_precession is None else float(custom_precession) - float(whfast_precession),
        "custom_minus_ias15_precession": None if custom_precession is None or ias15_precession is None else float(custom_precession) - float(ias15_precession),
        "custom_max_energy_rel_drift": custom["diagnostics"].get("max_energy_rel_drift"),
        "custom_max_angular_momentum_rel_drift": custom["diagnostics"].get("max_angular_momentum_rel_drift"),
        "custom_summary": str(custom_path),
        "whfast_summary": str(whfast_path),
        "ias15_summary": str(ias15_path),
    })
passed = bool(rows)
for row in rows:
    c = row["custom_precession_arcsec_per_century"]
    w = row["reboundx_whfast_gr_potential_precession_arcsec_per_century"]
    if c is None or w is None or abs(float(c) - float(w)) > 5.0:
        passed = False
with csv_out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
payload = {
    "passed": passed,
    "rows": rows,
    "warning": "Tolerances are short-run and timestep-convergence based; this is not a production chaos validation.",
}
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[gr-tangent-compare] wrote {csv_out}")
print(f"[gr-tangent-compare] wrote {json_out}")
if not passed:
    raise SystemExit(1)
PY
