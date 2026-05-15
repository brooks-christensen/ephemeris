#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_full_newtonian_megno_research_ladder}"
RESUME="${RESUME:-1}"
MAX_CASES="${MAX_CASES:-0}"
RECORD_EVERY_YEARS="${RECORD_EVERY_YEARS:-10000}"
ARCHIVE_INTERVAL_YEARS="${ARCHIVE_INTERVAL_YEARS:-10000}"

MANIFEST="${OUTPUT_DIR}/rebound_full_newtonian_megno_research_ladder_manifest.tsv"
CSV_OUT="${OUTPUT_DIR}/rebound_full_newtonian_megno_research_ladder.csv"
JSON_OUT="${OUTPUT_DIR}/rebound_full_newtonian_megno_research_ladder.json"
MD_OUT="${OUTPUT_DIR}/rebound_full_newtonian_megno_research_ladder.md"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
printf "tag\tmodel_scope\tbackend\tintegrator\tgr_model\tduration_years\tstep_days\tmegno_seed\tstatus\tsummary_path\tmegno_summary_path\torbital_elements_path\tarchive_path\tlog_path\n" > "${MANIFEST}"

case_count=0
run_case() {
  local duration="$1"
  local step="$2"
  local seed="$3"
  local tag="rebound_full_whfast_none_${duration}yr_step${step}_seed${seed}"
  local summary_path="${OUTPUT_DIR}/summary_${tag}.json"
  local megno_summary_path="${OUTPUT_DIR}/megno_summary_${tag}.json"
  local elements_path="${OUTPUT_DIR}/orbital_elements_${tag}.csv"
  local archive_path="${OUTPUT_DIR}/${tag}.bin"
  local log_path="${OUTPUT_DIR}/${tag}_$(date -u +%Y%m%dT%H%M%SZ).log"
  local status="ok"

  if [[ "${MAX_CASES}" != "0" && "${case_count}" -ge "${MAX_CASES}" ]]; then
    return
  fi
  case_count=$((case_count + 1))

  if [[ "${RESUME}" == "1" && -f "${summary_path}" && -f "${megno_summary_path}" ]]; then
    echo "[Full MEGNO ladder] skipping completed ${tag}"
    status="skipped"
  else
    echo "[Full MEGNO ladder] running ${tag}"
    set +e
    "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
      --kernel-path "${KERNEL_PATH}" \
      --start-date 2000-01-01 \
      --backend rebound \
      --rebound-integrator whfast \
      --rebound-gr-model none \
      --rebound-simulationarchive "${archive_path}" \
      --rebound-archive-interval-years "${ARCHIVE_INTERVAL_YEARS}" \
      --rebound-chaos-method megno \
      --model-scope full \
      --duration-years "${duration}" \
      --step-days "${step}" \
      --record-every-years "${RECORD_EVERY_YEARS}" \
      --megno-record-every-years "${RECORD_EVERY_YEARS}" \
      --megno-seed "${seed}" \
      --gr-model none \
      --integrator leapfrog \
      --output-dir "${OUTPUT_DIR}" \
      --tag "${tag}" \
      --with-megno \
      --with-rebound-lyapunov \
      --megno-duration-scaling-mode \
      --no-progress-bar >"${log_path}" 2>&1
    exit_code=$?
    set -e
    if [[ "${exit_code}" != "0" ]]; then
      status="failed"
      echo "[Full MEGNO ladder] failed ${tag}; see ${log_path}"
    fi
  fi

  printf "%s\tfull\trebound\twhfast\tnone\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${duration}" "${step}" "${seed}" "${status}" \
    "${summary_path}" "${megno_summary_path}" "${elements_path}" "${archive_path}" "${log_path}" >> "${MANIFEST}"
}

for duration in 1000000 5000000 10000000; do
  for step in 1 0.5; do
    for seed in 101 202 303; do
      run_case "${duration}" "${step}" "${seed}"
    done
  done
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
    "model_scope", "backend", "integrator", "gr_model", "duration_years",
    "step_days", "megno_seed", "runtime_seconds", "final_megno", "final_lcn",
    "classification", "max_energy_rel_drift", "max_angular_momentum_rel_drift",
    "mercury_max_eccentricity", "mars_max_eccentricity", "min_pairwise_separation_au",
    "archive_path", "warnings",
]

def load(path):
    path = Path(path)
    return json.load(path.open()) if path.exists() else {}

def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def fmt(value):
    value = f(value)
    return f"{value:.8g}" if math.isfinite(value) else ""

def max_e(path, body):
    path = Path(path)
    if not path.exists():
        return math.nan
    out = math.nan
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("body") == body:
                value = f(row.get("e"))
                if math.isfinite(value):
                    out = value if not math.isfinite(out) else max(out, value)
    return out

def min_sep(summary):
    values = [f(row.get("min_separation_au")) for row in summary.get("min_separations", [])]
    values = [value for value in values if math.isfinite(value)]
    return min(values) if values else math.nan

rows = []
with manifest.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        summary = load(item["summary_path"])
        megno = load(item["megno_summary_path"])
        extrema = summary.get("diagnostic_extrema_over_records", {})
        warnings = list(summary.get("warnings", [])) + list(megno.get("caveats", []))
        if item.get("status") == "failed":
            warnings.append(f"case failed; see {item['log_path']}")
        rows.append({
            "model_scope": "full",
            "backend": "rebound",
            "integrator": "whfast",
            "gr_model": "none",
            "duration_years": item["duration_years"],
            "step_days": item["step_days"],
            "megno_seed": item["megno_seed"],
            "runtime_seconds": summary.get("runtime", {}).get("wall_clock_seconds", ""),
            "final_megno": megno.get("final_megno", ""),
            "final_lcn": megno.get("estimated_lyapunov_if_available", ""),
            "classification": megno.get("classification_hint", "ambiguous"),
            "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
            "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
            "mercury_max_eccentricity": max_e(item["orbital_elements_path"], "mercury barycenter"),
            "mars_max_eccentricity": max_e(item["orbital_elements_path"], "mars barycenter"),
            "min_pairwise_separation_au": min_sep(summary),
            "archive_path": item["archive_path"],
            "warnings": " | ".join(str(w) for w in warnings if w),
        })

with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row[key]) if key not in {"model_scope", "backend", "integrator", "gr_model", "classification", "archive_path", "warnings"} else row[key] for key in fields})
json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
with md_out.open("w") as file_obj:
    file_obj.write("# REBOUND Full Newtonian MEGNO Research Ladder\n\n")
    file_obj.write("Finite-time MEGNO/LCN results are not asymptotic Lyapunov exponents without duration, timestep, and seed convergence.\n\n")
    file_obj.write("| duration yr | step d | seed | final MEGNO | final LCN | classification | runtime s |\n")
    file_obj.write("| ---: | ---: | ---: | ---: | ---: | --- | ---: |\n")
    for row in rows:
        file_obj.write(f"| {fmt(row['duration_years'])} | {fmt(row['step_days'])} | {row['megno_seed']} | {fmt(row['final_megno'])} | {fmt(row['final_lcn'])} | {row['classification']} | {fmt(row['runtime_seconds'])} |\n")
print(f"wrote {csv_out}")
print(f"wrote {json_out}")
print(f"wrote {md_out}")
PY
