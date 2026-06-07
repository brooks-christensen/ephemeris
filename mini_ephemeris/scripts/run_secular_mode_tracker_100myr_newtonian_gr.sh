#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
NEWTONIAN_CSV="${NEWTONIAN_CSV:-/home/peacelovephysics/ephemeris/output/stability/frequency_100myr_newtonian/orbital_elements_frequency_full_newtonian_100myr.csv}"
GR_CSV="${GR_CSV:-/home/peacelovephysics/ephemeris/output/stability/frequency_100myr_gr_potential/orbital_elements_frequency_full_gr_potential_100myr.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/frequency_100myr_mode_tracking}"
TAG="${TAG:-frequency_full_newtonian_vs_gr_potential_100myr}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m mini_ephemeris.secular_mode_tracker \
  --orbital-elements "${NEWTONIAN_CSV}" \
  --tag "full_newtonian_100myr_modes" \
  --output-dir "${OUTPUT_DIR}" \
  --bodies all \
  --window-years 20000000 \
  --step-years 5000000 \
  --top-k 5

"${PYTHON_BIN}" -m mini_ephemeris.secular_mode_tracker \
  --orbital-elements "${GR_CSV}" \
  --tag "full_gr_potential_100myr_modes" \
  --output-dir "${OUTPUT_DIR}" \
  --bodies all \
  --window-years 20000000 \
  --step-years 5000000 \
  --top-k 5

"${PYTHON_BIN}" -m mini_ephemeris.compare_secular_modes \
  --newtonian-peaks "${OUTPUT_DIR}/secular_mode_peaks_full_newtonian_100myr_modes.csv" \
  --gr-peaks "${OUTPUT_DIR}/secular_mode_peaks_full_gr_potential_100myr_modes.csv" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}"

echo "wrote mode-tracking outputs in ${OUTPUT_DIR}"
