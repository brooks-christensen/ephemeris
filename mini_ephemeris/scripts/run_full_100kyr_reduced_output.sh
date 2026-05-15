#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STEP_DAYS="${STEP_DAYS:-4}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-500}"
TAG="${TAG:-full_100kyr_reduced_output_step${STEP_DAYS}d}"
RESUME="${RESUME:-1}"
WITH_FREQUENCY="${WITH_FREQUENCY:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
START_EPOCH="$(date +%s)"
EXTRA_ARGS=()
if [[ "${WITH_FREQUENCY}" == "1" ]]; then
  EXTRA_ARGS+=(--with-frequency-map --frequency-window-years 20000 --frequency-step-years 5000 --frequency-bodies all)
fi

echo "[Run] 100 kyr full reduced-output survey"
echo "[Run] Broad survey only: step_days=${STEP_DAYS} is not validated Mercury-sensitive Lyapunov resolution."
echo "[Run] No finite-time Lyapunov/FLI/MEGNO claims are made by this script."
echo "[Run] expected desktop scale: about 4 hours at broad-survey cadence, depending on step_days"

if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[Run] RESUME=1 and ${SUMMARY_PATH} exists; skipping integration."
else
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --duration-years 100000 \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --gr-model none \
    --integrator leapfrog \
    --model-scope full \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --no-progress-bar \
    "${EXTRA_ARGS[@]}"
fi

"${PYTHON_BIN}" -m mini_ephemeris.stability_benchmark_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}"

END_EPOCH="$(date +%s)"
echo "[Run] elapsed_seconds=$((END_EPOCH - START_EPOCH))"
echo "[Run] summary=${SUMMARY_PATH}"
