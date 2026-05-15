#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
TAG="${TAG:-rebound_inner_30kyr_newtonian}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-25}"
ARCHIVE_INTERVAL_YEARS="${ARCHIVE_INTERVAL_YEARS:-250}"
RESUME="${RESUME:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[REBOUND] RESUME=1 and ${SUMMARY_PATH} exists; skipping."
  exit 0
fi

"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --backend rebound \
  --rebound-integrator whfast \
  --rebound-gr-model none \
  --rebound-simulationarchive "${OUTPUT_DIR}/${TAG}.bin" \
  --rebound-archive-interval-years "${ARCHIVE_INTERVAL_YEARS}" \
  --model-scope inner \
  --duration-years 30000 \
  --step-days "${STEP_DAYS}" \
  --record-every-years "${RECORD_EVERY_YEARS}" \
  --gr-model none \
  --integrator leapfrog \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --no-progress-bar
