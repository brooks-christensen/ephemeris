#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10}"
IAS15_EPSILON="${IAS15_EPSILON:-1e-10}"
RESUME="${RESUME:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

for GR_MODEL in none gr; do
  TAG="rebound_ias15_inner_1kyr_${GR_MODEL}"
  SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
  if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
    echo "[REBOUND IAS15] RESUME=1 and ${SUMMARY_PATH} exists; skipping."
    continue
  fi
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --backend rebound \
    --rebound-integrator ias15 \
    --rebound-gr-model "${GR_MODEL}" \
    --rebound-ias15-epsilon "${IAS15_EPSILON}" \
    --model-scope inner \
    --duration-years 1000 \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --gr-model none \
    --integrator leapfrog \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --no-progress-bar
done
