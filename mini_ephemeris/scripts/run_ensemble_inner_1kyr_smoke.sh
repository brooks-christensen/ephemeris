#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
TAG="${TAG:-ensemble_inner_1kyr_smoke}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-4}"
WORKERS="${WORKERS:-2}"
ENSEMBLE_SEED="${ENSEMBLE_SEED:-20260514}"
MAX_CASES="${MAX_CASES:-}"

cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  EXTRA_ARGS+=(--max-cases "${MAX_CASES}")
fi

echo "[Run] Ensemble inner 1 kyr smoke"
echo "[Run] ensemble_size=${ENSEMBLE_SIZE} workers=${WORKERS}"
echo "[Run] finite-time tangent Lyapunov enabled; no asymptotic chaos claim"

"${PYTHON_BIN}" -m mini_ephemeris.stability_ensemble_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --model-scope inner \
  --duration-years 1000 \
  --step-days 0.25 \
  --record-every-years 10 \
  --gr-model none \
  --integrator leapfrog \
  --ensemble-size "${ENSEMBLE_SIZE}" \
  --ensemble-perturbation-m 1000 \
  --ensemble-seed "${ENSEMBLE_SEED}" \
  --workers "${WORKERS}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body mercury \
  --lyapunov-renorm-years 0.25 \
  --resume \
  "${EXTRA_ARGS[@]}"
