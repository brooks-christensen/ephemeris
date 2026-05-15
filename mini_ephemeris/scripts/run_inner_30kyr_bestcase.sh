#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
TAG="${TAG:-inner_30kyr_bestcase_step0p25d_all_pert1000_renorm0p25}"
RESUME="${RESUME:-1}"
WITH_POINCARE="${WITH_POINCARE:-0}"
WITH_FLI_MEGNO="${WITH_FLI_MEGNO:-0}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
START_EPOCH="$(date +%s)"

EXTRA_ARGS=()
if [[ "${WITH_POINCARE}" == "1" ]]; then
  EXTRA_ARGS+=(--with-poincare --poincare-body mercury --poincare-plane z --poincare-direction positive)
fi
if [[ "${WITH_FLI_MEGNO}" == "1" ]]; then
  EXTRA_ARGS+=(--with-fli --with-megno-lite)
fi

echo "[Run] 30 kyr inner best-case finite-time diagnostic"
echo "[Run] Use after duration scaling suggests stable settings; not an asymptotic Lyapunov claim."
echo "[Run] expected desktop scale: roughly 1 to 1.5 hours at step_days=0.25"
echo "[Run] WITH_POINCARE=${WITH_POINCARE}; WITH_FLI_MEGNO=${WITH_FLI_MEGNO}"

if [[ "${RESUME}" == "1" && -f "${SUMMARY_PATH}" ]]; then
  echo "[Run] RESUME=1 and ${SUMMARY_PATH} exists; skipping integration."
else
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --duration-years 30000 \
    --step-days 0.25 \
    --record-every-years 10 \
    --gr-model none \
    --integrator leapfrog \
    --model-scope inner \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${TAG}" \
    --with-lyapunov \
    --lyapunov-method tangent \
    --lyapunov-body all \
    --lyapunov-perturbation-m 1000 \
    --lyapunov-renorm-years 0.25 \
    --with-frequency-map \
    --frequency-window-years 5000 \
    --frequency-step-years 1000 \
    --frequency-bodies mercury,venus,earth,mars \
    --no-progress-bar \
    "${EXTRA_ARGS[@]}"
fi

"${PYTHON_BIN}" -m mini_ephemeris.stability_benchmark_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}"

END_EPOCH="$(date +%s)"
echo "[Run] elapsed_seconds=$((END_EPOCH - START_EPOCH))"
echo "[Run] summary=${SUMMARY_PATH}"
