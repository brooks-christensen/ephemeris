#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

if [[ -d "${WORKSPACE_ROOT}/mini_ephemeris/scripts" ]]; then
  SCRIPT_PREFIX="${WORKSPACE_ROOT}/mini_ephemeris/scripts"
else
  SCRIPT_PREFIX="${PROJECT_ROOT}/scripts"
fi

echo "[Batch] using script prefix: ${SCRIPT_PREFIX}"
bash "${SCRIPT_PREFIX}/run_rebound_inner_10kyr_newtonian.sh"
bash "${SCRIPT_PREFIX}/run_rebound_inner_10kyr_gr_potential.sh"
bash "${SCRIPT_PREFIX}/run_backend_comparison_ladder.sh"
