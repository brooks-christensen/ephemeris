#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
TAG="${TAG:-inner_nonlinear_1kyr_smoke}"
RESUME="${RESUME:-1}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
START_EPOCH="$(date +%s)"

echo "[Run] 1 kyr inner Solar System nonlinear smoke"
echo "[Run] finite-time diagnostics only; not an asymptotic Lyapunov exponent"
echo "[Run] known scale on Ryzen 9 7900X: about 141 s at step_days=0.25"

if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[Run] RESUME=1 and ${SUMMARY_PATH} exists; skipping integration."
else
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --duration-years 1000 \
    --step-days 0.25 \
    --record-every-years 1 \
    --gr-model none \
    --integrator leapfrog \
    --model-scope inner \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --with-lyapunov \
    --lyapunov-method tangent \
    --lyapunov-body mercury \
    --lyapunov-perturbation-m 1000 \
    --lyapunov-renorm-years 0.25 \
    --with-poincare \
    --poincare-body mercury \
    --poincare-plane z \
    --poincare-direction positive \
    --with-frequency-map \
    --frequency-window-years 250 \
    --frequency-step-years 100 \
    --frequency-bodies mercury,venus,earth,mars \
    --with-fli \
    --with-megno-lite \
    --no-progress-bar
fi

"${PYTHON_BIN}" -m mini_ephemeris.stability_benchmark_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" -m mini_ephemeris.stability_scientific_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}"

END_EPOCH="$(date +%s)"
echo "[Run] elapsed_seconds=$((END_EPOCH - START_EPOCH))"
echo "[Run] summary=${SUMMARY_PATH}"
