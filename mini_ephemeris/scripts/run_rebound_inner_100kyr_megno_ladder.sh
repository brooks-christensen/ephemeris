#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_inner_100kyr_megno_ladder}"
STEP_DAYS="${STEP_DAYS:-1}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-100}"
MEGNO_RECORD_EVERY_YEARS="${MEGNO_RECORD_EVERY_YEARS:-100}"
ARCHIVE_INTERVAL_YEARS="${ARCHIVE_INTERVAL_YEARS:-1000}"
RESUME="${RESUME:-1}"
MAX_CASES="${MAX_CASES:-0}"

MANIFEST="${OUTPUT_DIR}/rebound_inner_100kyr_megno_ladder_manifest.tsv"
CSV_OUT="${OUTPUT_DIR}/rebound_inner_100kyr_megno_ladder.csv"
JSON_OUT="${OUTPUT_DIR}/rebound_inner_100kyr_megno_ladder.json"
MD_OUT="${OUTPUT_DIR}/rebound_inner_100kyr_megno_ladder.md"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

printf "tag\tgr_model\tduration_years\tsummary_path\tmegno_summary_path\n" > "${MANIFEST}"

case_count=0
run_case() {
  local gr_model="$1"
  local duration="$2"
  local tag="rebound_inner_whfast_${gr_model}_${duration}yr_megno"
  local summary_path="${OUTPUT_DIR}/summary_${tag}.json"
  local megno_summary_path="${OUTPUT_DIR}/megno_summary_${tag}.json"

  if [[ "${MAX_CASES}" != "0" && "${case_count}" -ge "${MAX_CASES}" ]]; then
    echo "[REBOUND MEGNO 100k] MAX_CASES=${MAX_CASES} reached; not scheduling ${tag}."
    return
  fi
  case_count=$((case_count + 1))

  if [[ "${RESUME}" == "1" && -f "${summary_path}" && -f "${megno_summary_path}" ]]; then
    echo "[REBOUND MEGNO 100k] RESUME=1 and ${tag} summaries exist; skipping."
  else
    "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
      --kernel-path "${KERNEL_PATH}" \
      --start-date 2000-01-01 \
      --backend rebound \
      --rebound-integrator whfast \
      --rebound-gr-model "${gr_model}" \
      --rebound-chaos-method megno \
      --rebound-simulationarchive "${OUTPUT_DIR}/${tag}.bin" \
      --rebound-archive-interval-years "${ARCHIVE_INTERVAL_YEARS}" \
      --model-scope inner \
      --duration-years "${duration}" \
      --step-days "${STEP_DAYS}" \
      --record-every-years "${RECORD_EVERY_YEARS}" \
      --megno-record-every-years "${MEGNO_RECORD_EVERY_YEARS}" \
      --gr-model none \
      --integrator leapfrog \
      --output-dir "${OUTPUT_DIR}" \
      --tag "${tag}" \
      --with-megno \
      --with-rebound-lyapunov \
      --megno-duration-scaling-mode \
      --no-progress-bar
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${gr_model}" "${duration}" "${summary_path}" "${megno_summary_path}" >> "${MANIFEST}"
}

for duration in 10000 30000 100000; do
  run_case "none" "${duration}"
  run_case "gr_potential" "${duration}"
done

"${PYTHON_BIN}" - "${MANIFEST}" "${CSV_OUT}" "${JSON_OUT}" "${MD_OUT}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
csv_out = Path(sys.argv[2])
json_out = Path(sys.argv[3])
md_out = Path(sys.argv[4])
fields = [
    "tag",
    "gr_model",
    "duration_years",
    "runtime_seconds",
    "final_megno",
    "estimated_lyapunov_if_available",
    "classification_hint",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "warnings",
]

def load(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.load(path.open())

def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def fmt(value):
    value = f(value)
    return f"{value:.8g}" if math.isfinite(value) else ""

rows = []
with manifest.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        summary = load(item["summary_path"])
        megno = load(item["megno_summary_path"])
        extrema = summary.get("diagnostic_extrema_over_records", {})
        warnings = list(summary.get("warnings", [])) + list(megno.get("caveats", []))
        rows.append(
            {
                "tag": item["tag"],
                "gr_model": item["gr_model"],
                "duration_years": item["duration_years"],
                "runtime_seconds": summary.get("runtime", {}).get("wall_clock_seconds", ""),
                "final_megno": megno.get("final_megno", ""),
                "estimated_lyapunov_if_available": megno.get("estimated_lyapunov_if_available", ""),
                "classification_hint": megno.get("classification_hint", "ambiguous"),
                "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
                "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
                "warnings": " | ".join(str(w) for w in warnings if w),
            }
        )

with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row[key]) if key not in {"tag", "gr_model", "classification_hint", "warnings"} else row[key] for key in fields})

json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
with md_out.open("w") as file_obj:
    file_obj.write("# REBOUND Inner 100 kyr MEGNO Ladder\n\n")
    file_obj.write("Finite-time MEGNO/LCN outputs are not asymptotic Lyapunov exponents without duration and timestep scaling.\n\n")
    file_obj.write("| tag | gr_model | duration_years | final_megno | LCN 1/yr | classification | runtime_s |\n")
    file_obj.write("| --- | --- | ---: | ---: | ---: | --- | ---: |\n")
    for row in rows:
        file_obj.write(
            f"| {row['tag']} | {row['gr_model']} | {fmt(row['duration_years'])} | "
            f"{fmt(row['final_megno'])} | {fmt(row['estimated_lyapunov_if_available'])} | "
            f"{row['classification_hint']} | {fmt(row['runtime_seconds'])} |\n"
        )

print(f"[REBOUND MEGNO 100k] wrote {csv_out}")
print(f"[REBOUND MEGNO 100k] wrote {json_out}")
print(f"[REBOUND MEGNO 100k] wrote {md_out}")
PY
