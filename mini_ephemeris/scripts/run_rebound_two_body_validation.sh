#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
DURATION_YEARS="${DURATION_YEARS:-100}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-1}"
INTEGRATOR="${INTEGRATOR:-whfast}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c "import rebound" >/dev/null 2>&1; then
  echo "[REBOUND] rebound is not installed; optional two-body validation skipped."
  echo "[REBOUND] Install with: ${PYTHON_BIN} -m pip install rebound"
  exit 0
fi

echo "[REBOUND] running optional Newtonian two-body validation"

for scope_step in \
  "two_body_jupiter 4" \
  "two_body_saturn 4" \
  "two_body_mercury 0.25"
do
  read -r MODEL_SCOPE STEP_DAYS <<< "${scope_step}"
  TAG="rebound_${MODEL_SCOPE}_${INTEGRATOR}_${DURATION_YEARS}yr"
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope "${MODEL_SCOPE}" \
    --duration-years "${DURATION_YEARS}" \
    --step-days "${STEP_DAYS}" \
    --record-every-years "${RECORD_EVERY_YEARS}" \
    --integrator "${INTEGRATOR}" \
    --gr-model none \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}"
done

echo "[REBOUND] summaries written under ${OUTPUT_DIR}"
