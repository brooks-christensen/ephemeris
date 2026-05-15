#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STAMP="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-checkpoint_resume_smoke_${STAMP}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints_${TAG}}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}"

echo "[Checkpoint] first leg: 0-2 years, two_body_jupiter, tangent state enabled"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 2 \
  --step-days 8 \
  --record-every-years 1 \
  --gr-model none \
  --integrator leapfrog \
  --model-scope two_body_jupiter \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body jupiter \
  --lyapunov-perturbation-m 1 \
  --lyapunov-renorm-years 1 \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --keep-checkpoints 3 \
  --no-progress-bar

LATEST_CHECKPOINT="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.npz' | sort | tail -n 1)"
if [[ -z "${LATEST_CHECKPOINT}" ]]; then
  echo "[Checkpoint] no checkpoint was written in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

echo "[Checkpoint] resume leg: ${LATEST_CHECKPOINT} -> 4 years"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 4 \
  --step-days 8 \
  --record-every-years 1 \
  --gr-model none \
  --integrator leapfrog \
  --model-scope two_body_jupiter \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body jupiter \
  --lyapunov-perturbation-m 1 \
  --lyapunov-renorm-years 1 \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --resume-from-checkpoint "${LATEST_CHECKPOINT}" \
  --keep-checkpoints 3 \
  --no-progress-bar

SUMMARY_PATH="${OUTPUT_DIR}/summary_${TAG}.json"
"${PYTHON_BIN}" -m mini_ephemeris.stability_benchmark_summary \
  "${SUMMARY_PATH}" \
  --output-dir "${OUTPUT_DIR}"

echo "[Checkpoint] summary=${SUMMARY_PATH}"
echo "[Checkpoint] checkpoint_dir=${CHECKPOINT_DIR}"
find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.npz' | sort
