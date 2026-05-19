#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_100myr}"
TAG="${TAG:-full_newtonian_shadow_100myr}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_checkpoints/${TAG}}"
LOG_PATH="${OUTPUT_DIR}/${TAG}_$(date -u +%Y%m%dT%H%M%SZ).log"
RESUME="${RESUME:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS=(--resume)
fi

"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  --kernel-path "${KERNEL_PATH}" \
  --model-scope full \
  --integrator whfast \
  --gr-model none \
  --duration-years 100000000 \
  --step-days 1 \
  --record-every-years 10000 \
  --perturb-body mercury \
  --perturbation-m 1 \
  --perturbation-mode radial \
  --fit-start-years 1000000 \
  --fit-end-years 50000000 \
  --checkpoint-every-years 1000000 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --resume-from-checkpoint latest \
  --keep-checkpoints 5 \
  --write-partial-every-record \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  "${RESUME_ARGS[@]}" >"${LOG_PATH}" 2>&1

echo "wrote log: ${LOG_PATH}"
