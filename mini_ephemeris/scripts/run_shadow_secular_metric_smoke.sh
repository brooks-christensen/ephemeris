#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_secular_metric_smoke}"
TAG_PREFIX="${TAG_PREFIX:-shadow_secular_metric_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

INNER_TAG="${TAG_PREFIX}_inner_mercury"
INNER_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${INNER_TAG}"
FULL_PLUTO_TAG="${TAG_PREFIX}_full_pluto"
FULL_PLUTO_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${FULL_PLUTO_TAG}"

COMMON_INNER=(
  --kernel-path "${KERNEL_PATH}"
  --model-scope inner
  --integrator whfast
  --gr-model none
  --duration-years 20000
  --step-days 64
  --record-every-years 2000
  --perturb-body mercury
  --perturbation-m 1
  --perturbation-mode radial
  --fit-start-years 2000
  --fit-end-years 18000
  --checkpoint-every-years 5000
  --checkpoint-dir "${INNER_CHECKPOINT_DIR}"
  --keep-checkpoints 5
  --write-partial-every-record
  --output-dir "${OUTPUT_DIR}"
  --tag "${INNER_TAG}"
  --no-progress-bar
)

echo "[Shadow secular smoke] inner Mercury first run stops after checkpoint"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  "${COMMON_INNER[@]}" \
  --stop-after-years 12000

echo "[Shadow secular smoke] inner Mercury resume from checkpoint"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  "${COMMON_INNER[@]}" \
  --resume-from-checkpoint latest

echo "[Shadow secular smoke] full+Pluto Pluto perturbation"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  --kernel-path "${KERNEL_PATH}" \
  --model-scope full_with_pluto \
  --integrator whfast \
  --gr-model none \
  --duration-years 10000 \
  --step-days 128 \
  --record-every-years 2000 \
  --perturb-body pluto \
  --perturbation-m 1 \
  --perturbation-mode radial \
  --fit-start-years 2000 \
  --fit-end-years 10000 \
  --checkpoint-every-years 5000 \
  --checkpoint-dir "${FULL_PLUTO_CHECKPOINT_DIR}" \
  --keep-checkpoints 5 \
  --write-partial-every-record \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${FULL_PLUTO_TAG}" \
  --no-progress-bar

"${PYTHON_BIN}" -m mini_ephemeris.shadow_fit_diagnostics \
  --shadow-csv "${OUTPUT_DIR}/shadow_separation_${INNER_TAG}.csv" \
  --tag "${INNER_TAG}_mercury_evec" \
  --output-dir "${OUTPUT_DIR}" \
  --metric mercury_barycenter_eccentricity_vector_separation \
  --windows "2000:10000,4000:18000"

"${PYTHON_BIN}" -m mini_ephemeris.shadow_metric_scan \
  --shadow-csv "${OUTPUT_DIR}/shadow_separation_${FULL_PLUTO_TAG}.csv" \
  --tag "${FULL_PLUTO_TAG}" \
  --output-dir "${OUTPUT_DIR}" \
  --windows "2000:6000,2000:10000" \
  --bodies mercury,jupiter,pluto

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${INNER_TAG}" "${FULL_PLUTO_TAG}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
inner_tag = sys.argv[2]
full_pluto_tag = sys.argv[3]

inner_csv = output_dir / f"shadow_separation_{inner_tag}.csv"
full_pluto_csv = output_dir / f"shadow_separation_{full_pluto_tag}.csv"
inner_summary = output_dir / f"shadow_lyapunov_summary_{inner_tag}.json"

for path in (inner_csv, full_pluto_csv):
    raw = path.read_bytes()
    if b"\0" in raw:
        raise SystemExit(f"CSV contains NUL bytes: {path}")

with inner_csv.open(newline="") as file_obj:
    inner_header = next(csv.reader(file_obj))
required_inner = {
    "mercury_barycenter_eccentricity_vector_separation",
    "mercury_barycenter_inclination_vector_separation",
    "mercury_barycenter_delta_varpi_wrapped",
    "mercury_barycenter_delta_lambda_wrapped",
}
missing_inner = sorted(required_inner - set(inner_header))
if missing_inner:
    raise SystemExit(f"inner CSV missing expected columns: {missing_inner}")

with full_pluto_csv.open(newline="") as file_obj:
    full_header = next(csv.reader(file_obj))
required_pluto = {
    "pluto_barycenter_eccentricity_vector_separation",
    "pluto_barycenter_inclination_vector_separation",
    "pluto_barycenter_delta_varpi_wrapped",
}
missing_pluto = sorted(required_pluto - set(full_header))
if missing_pluto:
    raise SystemExit(f"full+Pluto CSV missing expected columns: {missing_pluto}")

summary = json.loads(inner_summary.read_text())
if not summary.get("resumed_from_checkpoint"):
    raise SystemExit("inner smoke did not record checkpoint resume")
if float(summary.get("resumed_from_time_years") or 0.0) <= 0.0:
    raise SystemExit("inner smoke resume time was not greater than zero")

fit_csv = output_dir / f"shadow_fit_diagnostics_{inner_tag}_mercury_evec.csv"
scan_csv = output_dir / f"shadow_metric_scan_{full_pluto_tag}.csv"
scan_json = output_dir / f"shadow_metric_scan_{full_pluto_tag}.json"
for path in (fit_csv, scan_csv, scan_json):
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing or empty diagnostic output: {path}")

print("shadow secular metric smoke verification passed")
print(f"inner_resume_time_years={summary['resumed_from_time_years']}")
print(f"inner_csv={inner_csv}")
print(f"full_pluto_csv={full_pluto_csv}")
print(f"fit_csv={fit_csv}")
print(f"scan_csv={scan_csv}")
PY

echo "shadow secular metric smoke outputs verified in ${OUTPUT_DIR}"
