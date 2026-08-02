#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_tangent_v1/mercury_null_control}"
DURATION_YEARS="${DURATION_YEARS:-1000}"
STEP_DAYS="${STEP_DAYS:-0.25}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10}"
SEEDS="${SEEDS:-12345 67890}"
SUMMARY_CSV="${OUTPUT_DIR}/gr_tangent_mercury_null_control.csv"
SUMMARY_JSON="${OUTPUT_DIR}/gr_tangent_mercury_null_control.json"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"
printf "seed,duration_years,step_days,final_megno,final_lcn_1_per_year,classification_hint,max_energy_rel_drift,max_angular_momentum_rel_drift,mercury_perihelion_drift_arcsec_per_century,summary_path\n" > "${SUMMARY_CSV}"

for seed in ${SEEDS}; do
  tag="gr_tangent_mercury_null_${DURATION_YEARS}yr_seed${seed}"
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_gr_tangent_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_mercury \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --megno-seed "${seed}" \
    --gr-scale 1 \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${tag}" \
    --status-every-record \
    --no-progress-bar
done

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${SUMMARY_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
json_path = Path(sys.argv[3])
rows = []
for path in sorted(out.glob("gr_tangent_summary_gr_tangent_mercury_null_*_seed*.json")):
    data = json.loads(path.read_text())
    cfg = data["configuration"]
    diag = data["diagnostics"]
    rows.append({
        "seed": cfg["megno_seed"],
        "duration_years": cfg["duration_years"],
        "step_days": cfg["step_days"],
        "final_megno": diag.get("final_megno"),
        "final_lcn_1_per_year": diag.get("final_lcn_1_per_year"),
        "classification_hint": diag.get("classification_hint"),
        "max_energy_rel_drift": diag.get("max_energy_rel_drift"),
        "max_angular_momentum_rel_drift": diag.get("max_angular_momentum_rel_drift"),
        "mercury_perihelion_drift_arcsec_per_century": diag.get("mercury_perihelion_drift_arcsec_per_century"),
        "summary_path": str(path),
    })
passed = bool(rows)
for row in rows:
    lcn = abs(float(row["final_lcn_1_per_year"])) if row["final_lcn_1_per_year"] is not None else math.inf
    precession = float(row["mercury_perihelion_drift_arcsec_per_century"]) if row["mercury_perihelion_drift_arcsec_per_century"] is not None else math.nan
    if row["classification_hint"] not in {"regular_likely", "ambiguous"}:
        passed = False
    if lcn > 1e-4:
        passed = False
    if not math.isfinite(precession) or not (20.0 <= abs(precession) <= 80.0):
        passed = False
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
payload = {
    "passed": passed,
    "row_count": len(rows),
    "warning": "Short relativistic null control; finite-time diagnostic only.",
    "rows": rows,
}
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[gr-tangent-null] wrote {csv_path}")
print(f"[gr-tangent-null] wrote {json_path}")
if not passed:
    raise SystemExit(1)
PY
