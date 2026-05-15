#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
BATCH_DIR="${BATCH_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
TAG="${TAG:-}"

cd "${PROJECT_ROOT}"

if [[ -n "${TAG}" ]]; then
  "${PYTHON_BIN}" -m mini_ephemeris.pack_stability_batch --batch-dir "${BATCH_DIR}" --tag "${TAG}"
else
  "${PYTHON_BIN}" -m mini_ephemeris.pack_stability_batch --batch-dir "${BATCH_DIR}"
fi
