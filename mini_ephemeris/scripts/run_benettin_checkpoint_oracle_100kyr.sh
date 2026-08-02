#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
ARCHIVE_PATH="${ARCHIVE_PATH:-/home/peacelovephysics/ephemeris/output/stability/rebound_full_with_pluto_newtonian_500myr_megno_seed12345/full_with_pluto_newtonian_500myr_megno_1d_seed_12345.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/final_benettin_v2/checkpoint_oracle_100kyr}"
SNAPSHOT_YEARS="${SNAPSHOT_YEARS:-300000000}"
DURATION_YEARS="${DURATION_YEARS:-100000}"
STEP_DAYS="${STEP_DAYS:-1}"
RENORM_YEARS="${RENORM_YEARS:-1000}"
SEED="${SEED:-12345}"
TAG="${TAG:-checkpoint_oracle_100kyr_seed${SEED}}"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "[oracle] missing archive: ${ARCHIVE_PATH}" >&2
  exit 2
fi

"${PYTHON_BIN}" -m mini_ephemeris.benettin_checkpoint_oracle_cli \
  --simulationarchive "${ARCHIVE_PATH}" \
  --snapshot-years "${SNAPSHOT_YEARS}" \
  --duration-years "${DURATION_YEARS}" \
  --step-days "${STEP_DAYS}" \
  --renorm-years "${RENORM_YEARS}" \
  --model-scope full_with_pluto \
  --perturb-body mercury \
  --perturbation-m 1 \
  --perturbation-mode radial \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --no-progress-bar
