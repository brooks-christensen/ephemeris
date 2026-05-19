#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_100myr}"
SHADOW_CSV="${SHADOW_CSV:-/home/peacelovephysics/ephemeris/output/stability/shadow_100myr/shadow_separation_full_newtonian_shadow_100myr.csv}"
TAG="${TAG:-full_newtonian_shadow_100myr}"
WINDOWS="${WINDOWS:-1e6:1e7,1e6:2e7,2e6:2e7,5e6:3e7,1e7:4e7,1e6:5e7}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m mini_ephemeris.shadow_fit_diagnostics \
  --shadow-csv "${SHADOW_CSV}" \
  --tag "${TAG}" \
  --output-dir "${OUTPUT_DIR}" \
  --windows "${WINDOWS}"
