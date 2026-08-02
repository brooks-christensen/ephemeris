#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_tangent_v1/full_1myr_smoke}"
DURATION_YEARS="${DURATION_YEARS:-1000000}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10000}"
ARCHIVE_INTERVAL_YEARS="${ARCHIVE_INTERVAL_YEARS:-100000}"
MEGNO_SEED="${MEGNO_SEED:-12345}"
TAG="${TAG:-full_with_pluto_gr_tangent_1myr_seed${MEGNO_SEED}}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m mini_ephemeris.rebound_gr_tangent_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --model-scope full_with_pluto \
  --duration-years "${DURATION_YEARS}" \
  --step-days "${STEP_DAYS}" \
  --record-every-years "${RECORD_EVERY_YEARS}" \
  --megno-seed "${MEGNO_SEED}" \
  --gr-scale 1 \
  --simulationarchive "${OUTPUT_DIR}/${TAG}.bin" \
  --archive-interval-years "${ARCHIVE_INTERVAL_YEARS}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --status-every-record \
  --resume \
  --no-progress-bar
