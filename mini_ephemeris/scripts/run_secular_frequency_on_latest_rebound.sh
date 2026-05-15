#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
WINDOW_YEARS="${WINDOW_YEARS:-100000}"
STEP_YEARS="${STEP_YEARS:-50000}"
BODIES="${BODIES:-all}"

cd "${PROJECT_ROOT}"

LATEST="$(find "${OUTPUT_DIR}" -type f -name 'orbital_elements_rebound*.csv' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${LATEST}" ]]; then
  echo "No orbital_elements_rebound*.csv file found under ${OUTPUT_DIR}."
  exit 1
fi

TAG="$(basename "${LATEST}" .csv)"
TAG="${TAG#orbital_elements_}"
"${PYTHON_BIN}" -m mini_ephemeris.secular_frequency_summary \
  --orbital-elements "${LATEST}" \
  --bodies "${BODIES}" \
  --window-years "${WINDOW_YEARS}" \
  --step-years "${STEP_YEARS}" \
  --output-prefix "$(dirname "${LATEST}")/secular_frequency_summary_${TAG}"
