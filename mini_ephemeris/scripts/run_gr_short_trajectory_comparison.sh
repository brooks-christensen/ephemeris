#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/final_gr_benettin_v2/gr_short_trajectory_comparison}"
DURATION_YEARS="${DURATION_YEARS:-100}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10}"
CSV_OUT="${OUTPUT_DIR}/gr_short_trajectory_comparison.csv"
JSON_OUT="${OUTPUT_DIR}/gr_short_trajectory_comparison.json"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

run_case() {
  local integrator="$1"
  local gr_model="$2"
  local tag="gr_short_${integrator}_${gr_model}_${DURATION_YEARS}yr"
  local summary="${OUTPUT_DIR}/summary_${tag}.json"
  if [[ "${RESUME:-1}" == "1" && -f "${summary}" ]]; then
    echo "[gr-short] skip existing ${tag}"
  else
    echo "[gr-short] run ${tag}"
    "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
      --kernel-path "${KERNEL_PATH}" \
      --start-date 2000-01-01 \
      --backend rebound \
      --rebound-integrator "${integrator}" \
      --rebound-gr-model "${gr_model}" \
      --model-scope inner \
      --duration-years "${DURATION_YEARS}" \
      --step-days "${STEP_DAYS}" \
      --record-every-years "${RECORD_EVERY_YEARS}" \
      --gr-model none \
      --integrator leapfrog \
      --output-dir "${OUTPUT_DIR}" \
      --tag "${tag}" \
      --no-progress-bar
  fi
}

run_case whfast gr_potential
run_case ias15 gr

"${PYTHON_BIN}" - "${OUTPUT_DIR}" "${DURATION_YEARS}" "${CSV_OUT}" "${JSON_OUT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
duration = sys.argv[2]
csv_out = Path(sys.argv[3])
json_out = Path(sys.argv[4])

def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

rows = []
for integrator, gr_model in [("whfast", "gr_potential"), ("ias15", "gr")]:
    tag = f"gr_short_{integrator}_{gr_model}_{duration}yr"
    path = out / f"summary_{tag}.json"
    if not path.exists():
        rows.append({"integrator": integrator, "gr_model": gr_model, "status": "missing", "runtime_seconds": "", "max_energy_rel_drift": "", "max_angular_momentum_rel_drift": "", "summary_path": ""})
        continue
    data = json.loads(path.read_text())
    rows.append({
        "integrator": integrator,
        "gr_model": gr_model,
        "status": "ok",
        "runtime_seconds": data.get("runtime_seconds", ""),
        "max_energy_rel_drift": data.get("max_energy_rel_drift", ""),
        "max_angular_momentum_rel_drift": data.get("max_angular_momentum_rel_drift", ""),
        "summary_path": str(path),
    })

passed = len(rows) == 2 and all(row["status"] == "ok" for row in rows)
if passed:
    max_l = max(abs(f(row["max_angular_momentum_rel_drift"])) for row in rows)
    passed = max_l < 1e-8
with csv_out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
payload = {
    "passed": passed,
    "duration_years": float(duration),
    "rows": rows,
    "warning": "Short trajectory comparison only; this is a GR plumbing gate, not a chaos validation.",
}
json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[gr-short] wrote {csv_out}")
print(f"[gr-short] wrote {json_out}")
if not passed:
    raise SystemExit(1)
PY
