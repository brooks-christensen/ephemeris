#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_full_newtonian_100myr_megno}"
DURATION_YEARS="${DURATION_YEARS:-100000000}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10000}"
MEGNO_SEED="${MEGNO_SEED:-12345}"
MEGNO_RECORD_EVERY_YEARS="${MEGNO_RECORD_EVERY_YEARS:-${RECORD_EVERY_YEARS}}"
ARCHIVE_INTERVAL_YEARS="${ARCHIVE_INTERVAL_YEARS:-1000000}"
TAG="${TAG:-full_newtonian_100myr_megno_seed_${MEGNO_SEED}}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_DIR}/${TAG}.bin}"
RESUME="${RESUME:-1}"
LOG_PATH="${OUTPUT_DIR}/run_${TAG}_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[run] Summary already exists and RESUME=1, skipping: ${SUMMARY_PATH}"
  exit 0
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" && -f "${ARCHIVE_PATH}" ]]; then
  RESUME_ARGS=(--rebound-resume latest)
  echo "[run] Resuming from existing SimulationArchive: ${ARCHIVE_PATH}"
else
  echo "[run] Starting fresh; no existing SimulationArchive resume requested."
fi

{
  echo "[run] tag=${TAG}"
  echo "[run] output_dir=${OUTPUT_DIR}"
  echo "[run] archive=${ARCHIVE_PATH}"
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --backend rebound \
    --rebound-integrator whfast \
    --rebound-gr-model none \
    --model-scope full \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --with-megno \
    --with-rebound-lyapunov \
    --megno-seed "${MEGNO_SEED}" \
    --megno-record-every-years "${MEGNO_RECORD_EVERY_YEARS}" \
    --rebound-simulationarchive "${ARCHIVE_PATH}" \
    --rebound-archive-interval-years "${ARCHIVE_INTERVAL_YEARS}" \
    "${RESUME_ARGS[@]}" \
    --no-progress-bar
} 2>&1 | tee "${LOG_PATH}"
