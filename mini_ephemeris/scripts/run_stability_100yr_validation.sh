#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STEP_DAYS="${STEP_DAYS:-1}"
DURATION_YEARS="${DURATION_YEARS:-100}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-1}"
TAG="${TAG:-stability_100yr_validation_step${STEP_DAYS}d}"
RESUME="${RESUME:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
START_EPOCH="$(date +%s)"

echo "[Run] 100 yr full-model stability validation"
echo "[Run] stability mode: physical reduced model, Earth-Moon barycenter, no empirical lunar calibration"
echo "[Run] tag=${TAG}"
echo "[Run] step_days=${STEP_DAYS}, duration_years=${DURATION_YEARS}, record_every_years=${RECORD_EVERY_YEARS}"

if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[Run] RESUME=1 and ${SUMMARY_PATH} exists; skipping integration."
else
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --gr-model none \
    --integrator leapfrog \
    --model-scope full \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --no-progress-bar
fi

"${PYTHON_BIN}" -m mini_ephemeris.stability_benchmark_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}"

END_EPOCH="$(date +%s)"
echo "[Run] elapsed_seconds=$((END_EPOCH - START_EPOCH))"
echo "[Run] summary=${SUMMARY_PATH}"
