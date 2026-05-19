#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/shadow_checkpoint_smoke}"
TAG="${TAG:-shadow_checkpoint_resume_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${TAG}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
  --kernel-path "${KERNEL_PATH}"
  --model-scope inner
  --integrator whfast
  --gr-model none
  --duration-years 20000
  --step-days 64
  --record-every-years 2000
  --perturb-body mercury
  --perturbation-m 1
  --perturbation-mode radial
  --fit-start-years 2000
  --fit-end-years 18000
  --checkpoint-every-years 5000
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --keep-checkpoints 5
  --write-partial-every-record
  --output-dir "${OUTPUT_DIR}"
  --tag "${TAG}"
  --no-progress-bar
)

echo "[Shadow checkpoint smoke] first run stops after a checkpoint"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  "${COMMON_ARGS[@]}" \
  --stop-after-years 12000

latest_checkpoint="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name 'checkpoint_*yr' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${latest_checkpoint}" ]]; then
  echo "No checkpoint directories were written."
  exit 1
fi
test -s "${latest_checkpoint}/reference.bin"
test -s "${latest_checkpoint}/shadow.bin"
test -s "${latest_checkpoint}/checkpoint_state.json"

echo "[Shadow checkpoint smoke] intentionally corrupting latest checkpoint JSON: ${latest_checkpoint}"
cp "${latest_checkpoint}/checkpoint_state.json" "${latest_checkpoint}/checkpoint_state.json.original"
printf '{not valid json\n' > "${latest_checkpoint}/checkpoint_state.json"

echo "[Shadow checkpoint smoke] second run resumes from latest valid fallback"
"${PYTHON_BIN}" -m mini_ephemeris.rebound_shadow_lyapunov_cli \
  "${COMMON_ARGS[@]}" \
  --resume-from-checkpoint latest

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${TAG}" "${CHECKPOINT_DIR}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
tag = sys.argv[2]
checkpoint_dir = Path(sys.argv[3])
summary_path = output_dir / f"shadow_lyapunov_summary_{tag}.json"
csv_path = output_dir / f"shadow_separation_{tag}.csv"

if not summary_path.exists():
    raise SystemExit(f"missing final summary: {summary_path}")
summary = json.loads(summary_path.read_text())
if not summary.get("resumed_from_checkpoint"):
    raise SystemExit("summary did not record resumed_from_checkpoint")
if float(summary.get("resumed_from_time_years") or 0.0) <= 0.0:
    raise SystemExit("summary resumed_from_time_years was not greater than zero")
if not summary.get("checkpoint_warnings"):
    raise SystemExit("expected checkpoint_warnings from intentionally corrupted latest checkpoint")

valid_checkpoint_count = 0
for path in checkpoint_dir.glob("checkpoint_*yr"):
    if (path / "reference.bin").exists() and (path / "shadow.bin").exists() and (path / "checkpoint_state.json").exists():
        valid_checkpoint_count += 1
if valid_checkpoint_count == 0:
    raise SystemExit("no checkpoint bundles with expected files found")

raw = csv_path.read_bytes()
if b"\0" in raw:
    raise SystemExit("CSV contains NUL bytes")
times = []
with csv_path.open(newline="") as file_obj:
    for row in csv.DictReader(file_obj):
        times.append(float(row["time_years"]))
if times != sorted(times):
    raise SystemExit("CSV time_years are not monotonically increasing")
if len(times) != len(set(times)):
    raise SystemExit("CSV contains duplicate time_years rows")
if int(summary.get("duplicate_rows_removed_on_resume") or 0) < 1:
    raise SystemExit("expected at least one duplicate/truncated row removed on resume")

print("checkpoint resume smoke verification passed")
print(f"resumed_from_time_years={summary['resumed_from_time_years']}")
print(f"duplicate_rows_removed_on_resume={summary['duplicate_rows_removed_on_resume']}")
print(f"output_rows={summary['output_rows']}")
PY

echo "shadow checkpoint smoke outputs verified in ${OUTPUT_DIR}"
