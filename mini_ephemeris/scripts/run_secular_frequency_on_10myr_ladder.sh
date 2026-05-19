#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
LADDER_DIR="${LADDER_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_full_newtonian_megno_research_ladder}"
WINDOW_YEARS="${WINDOW_YEARS:-1000000}"
STEP_YEARS="${STEP_YEARS:-500000}"

cd "${PROJECT_ROOT}"

found=0
while IFS= read -r elements_path; do
  found=1
  tag="$(basename "${elements_path}" .csv)"
  tag="${tag#orbital_elements_}"
  echo "[Secular frequency] ${tag}"
  "${PYTHON_BIN}" -m mini_ephemeris.secular_frequency_summary \
    --orbital-elements "${elements_path}" \
    --bodies mercury,jupiter \
    --window-years "${WINDOW_YEARS}" \
    --step-years "${STEP_YEARS}" \
    --output-prefix "$(dirname "${elements_path}")/secular_frequency_summary_${tag}"
done < <(find "${LADDER_DIR}" -type f -name 'orbital_elements_*10000000*.csv' | sort)

if [[ "${found}" == "0" ]]; then
  echo "No 10 Myr orbital_elements files found under ${LADDER_DIR}."
fi
