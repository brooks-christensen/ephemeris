#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_smoke}"
TAG="${TAG:-shadow_1myr_smoke}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  --kernel-path "${KERNEL_PATH}" \
  --model-scope inner \
  --integrator whfast \
  --gr-model none \
  --duration-years 1000000 \
  --step-days 16 \
  --record-every-years 50000 \
  --perturb-body mercury \
  --perturbation-m 1 \
  --perturbation-mode radial \
  --fit-start-years 100000 \
  --fit-end-years 800000 \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --resume \
  --no-progress-bar

test -s "${OUTPUT_DIR}/shadow_separation_${TAG}.csv"
test -s "${OUTPUT_DIR}/shadow_lyapunov_summary_${TAG}.json"
test -s "${OUTPUT_DIR}/shadow_growth_${TAG}.png"
echo "shadow smoke outputs verified in ${OUTPUT_DIR}"
