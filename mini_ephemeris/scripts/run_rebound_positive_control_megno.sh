#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/chaos_positive_control}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "[Positive control] two-body Jupiter regular control"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_chaos_validation_cli \
  --kernel-path "${KERNEL_PATH}" \
  --model-scope two_body_jupiter \
  --duration-years 1000 \
  --step-days 4 \
  --record-every-years 10 \
  --integrator whfast \
  --output-dir "${OUTPUT_DIR}" \
  --tag two_body_jupiter_regular_control \
  --no-progress-bar

echo "[Positive control] compact chaotic three-body toy"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_chaos_validation_cli \
  --model-scope chaotic_three_body \
  --duration-years 200 \
  --step-days 0.01 \
  --record-every-years 1 \
  --integrator ias15 \
  --output-dir "${OUTPUT_DIR}" \
  --tag chaotic_three_body_positive_control \
  --no-progress-bar
