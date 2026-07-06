#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
SCRIPT="${PROJECT_DIR}/scripts/run_rebound_full_newtonian_100myr_megno.sh"
MAX_CASES="${MAX_CASES:-}"
CASE_COUNT=0

run_case() {
  local duration="$1"
  local step="$2"
  local seed="$3"
  local tag="$4"
  if [[ -n "${MAX_CASES}" && "${CASE_COUNT}" -ge "${MAX_CASES}" ]]; then
    return
  fi
  CASE_COUNT=$((CASE_COUNT + 1))
  echo "[convergence] case ${CASE_COUNT}: duration=${duration} step=${step} seed=${seed}"
  DURATION_YEARS="${duration}" \
  STEP_DAYS="${step}" \
  MEGNO_SEED="${seed}" \
  TAG="${tag}" \
  OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_full_newtonian_megno_convergence}" \
  bash "${SCRIPT}"
}

run_case 100000000 1 12345 full_newtonian_100myr_megno_1d_seed_12345
run_case 100000000 0.5 12345 full_newtonian_100myr_megno_0p5d_seed_12345

# Enable only after the 100 Myr cases have been inspected.
# run_case 200000000 1 12345 full_newtonian_200myr_megno_1d_seed_12345
