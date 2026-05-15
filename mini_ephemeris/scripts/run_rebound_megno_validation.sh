#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_megno_validation}"
RESUME="${RESUME:-1}"
MAX_CASES="${MAX_CASES:-0}"
INCLUDE_IAS15_10KYR="${INCLUDE_IAS15_10KYR:-1}"
INCLUDE_IAS15_GR="${INCLUDE_IAS15_GR:-1}"

MANIFEST="${OUTPUT_DIR}/rebound_megno_validation_manifest.tsv"
CSV_OUT="${OUTPUT_DIR}/rebound_megno_validation_ladder.csv"
JSON_OUT="${OUTPUT_DIR}/rebound_megno_validation_ladder.json"
MD_OUT="${OUTPUT_DIR}/rebound_megno_validation_ladder.md"
SCALING_CSV="${OUTPUT_DIR}/rebound_megno_duration_scaling_summary.csv"
SCALING_JSON="${OUTPUT_DIR}/rebound_megno_duration_scaling_summary.json"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" - <<'PY'
import rebound
PY
then
  echo "[REBOUND MEGNO] REBOUND is not installed; skipping validation ladder."
  exit 0
fi

HAS_REBOUNDX="$("${PYTHON_BIN}" - <<'PY'
try:
    import reboundx  # noqa: F401
except Exception:
    print("0")
else:
    print("1")
PY
)"

printf "tag\tmodel_scope\tintegrator\tgr_model\tduration_years\tstep_days\trecord_every_years\tmegno_record_every_years\tstatus\tsummary_path\tmegno_summary_path\torbital_elements_path\tlog_path\n" > "${MANIFEST}"

case_index=0
run_case() {
  local model_scope="$1"
  local integrator="$2"
  local gr_model="$3"
  local duration_years="$4"
  local step_days="$5"
  local record_every_years="$6"
  local megno_record_every_years="$7"
  local tag="$8"

  if [[ "${MAX_CASES}" != "0" && "${case_index}" -ge "${MAX_CASES}" ]]; then
    echo "[REBOUND MEGNO] MAX_CASES=${MAX_CASES} reached; not scheduling ${tag}."
    return
  fi
  case_index=$((case_index + 1))

  local summary_path="${OUTPUT_DIR}/summary_${tag}.json"
  local megno_summary_path="${OUTPUT_DIR}/megno_summary_${tag}.json"
  local elements_path="${OUTPUT_DIR}/orbital_elements_${tag}.csv"
  local log_path="${OUTPUT_DIR}/${tag}.log"
  local status="ok"

  if [[ "${RESUME}" == "1" && -f "${summary_path}" && -f "${megno_summary_path}" ]]; then
    echo "[REBOUND MEGNO] RESUME=1 and ${tag} summaries exist; skipping."
    status="skipped"
  else
    echo "[REBOUND MEGNO] Running ${tag} (${model_scope}, ${integrator}, ${gr_model}, ${duration_years} yr)."
    set +e
    "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
      --kernel-path "${KERNEL_PATH}" \
      --start-date 2000-01-01 \
      --backend rebound \
      --rebound-integrator "${integrator}" \
      --rebound-gr-model "${gr_model}" \
      --rebound-chaos-method megno \
      --model-scope "${model_scope}" \
      --duration-years "${duration_years}" \
      --step-days "${step_days}" \
      --record-every-years "${record_every_years}" \
      --megno-record-every-years "${megno_record_every_years}" \
      --gr-model none \
      --integrator leapfrog \
      --output-dir "${OUTPUT_DIR}" \
      --tag "${tag}" \
      --with-megno \
      --with-rebound-lyapunov \
      --megno-duration-scaling-mode \
      --no-progress-bar >"${log_path}" 2>&1
    local exit_code=$?
    set -e
    if [[ "${exit_code}" != "0" ]]; then
      echo "[REBOUND MEGNO] Case failed: ${tag}; see ${log_path}."
      status="failed"
    fi
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${tag}" "${model_scope}" "${integrator}" "${gr_model}" "${duration_years}" \
    "${step_days}" "${record_every_years}" "${megno_record_every_years}" "${status}" \
    "${summary_path}" "${megno_summary_path}" "${elements_path}" "${log_path}" >> "${MANIFEST}"
}

run_case "two_body_jupiter" "whfast" "none" "10000" "4" "100" "100" "rebound_megno_two_body_jupiter_whfast_none_10kyr"
run_case "two_body_saturn" "whfast" "none" "10000" "4" "100" "100" "rebound_megno_two_body_saturn_whfast_none_10kyr"
run_case "two_body_mercury" "whfast" "none" "10000" "0.125" "100" "100" "rebound_megno_two_body_mercury_whfast_none_10kyr"

for duration in 1000 10000 30000; do
  if [[ "${duration}" == "1000" ]]; then
    record="10"
  else
    record="100"
  fi
  run_case "inner" "whfast" "none" "${duration}" "1" "${record}" "${record}" "rebound_megno_inner_whfast_none_${duration}yr"
  if [[ "${HAS_REBOUNDX}" == "1" ]]; then
    run_case "inner" "whfast" "gr_potential" "${duration}" "1" "${record}" "${record}" "rebound_megno_inner_whfast_gr_potential_${duration}yr"
  else
    echo "[REBOUND MEGNO] reboundx unavailable; skipping whfast gr_potential ${duration} yr."
  fi
done

run_case "inner" "ias15" "none" "1000" "1" "10" "10" "rebound_megno_inner_ias15_none_1000yr"
if [[ "${INCLUDE_IAS15_10KYR}" == "1" ]]; then
  run_case "inner" "ias15" "none" "10000" "1" "100" "100" "rebound_megno_inner_ias15_none_10000yr"
fi
if [[ "${HAS_REBOUNDX}" == "1" && "${INCLUDE_IAS15_GR}" == "1" ]]; then
  run_case "inner" "ias15" "gr" "1000" "1" "10" "10" "rebound_megno_inner_ias15_gr_1000yr"
fi

"${PYTHON_BIN}" - "${MANIFEST}" "${CSV_OUT}" "${JSON_OUT}" "${MD_OUT}" "${SCALING_CSV}" "${SCALING_JSON}" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

manifest = Path(sys.argv[1])
csv_out = Path(sys.argv[2])
json_out = Path(sys.argv[3])
md_out = Path(sys.argv[4])
scaling_csv = Path(sys.argv[5])
scaling_json = Path(sys.argv[6])

FIELDS = [
    "model_scope",
    "backend",
    "integrator",
    "gr_model",
    "duration_years",
    "step_days",
    "runtime_seconds",
    "final_megno",
    "final_mean_megno",
    "estimated_lyapunov_if_available",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "mercury_max_eccentricity",
    "min_pairwise_separation_au",
    "classification",
    "warnings",
]

SCALING_FIELDS = [
    "model_scope",
    "integrator",
    "gr_model",
    "step_days",
    "duration_count",
    "durations_years",
    "final_megno_values",
    "estimated_lyapunov_values",
    "best_fit_model",
    "classification",
    "reason",
]

def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def fmt(value):
    value = f(value)
    return f"{value:.8g}" if math.isfinite(value) else ""

def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as file_obj:
        return json.load(file_obj)

def mercury_max_eccentricity(path):
    path = Path(path)
    if not path.exists():
        return math.nan
    maximum = math.nan
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("body") != "mercury barycenter":
                continue
            value = f(row.get("e"))
            if math.isfinite(value):
                maximum = value if not math.isfinite(maximum) else max(maximum, value)
    return maximum

def min_pairwise(summary):
    values = []
    for row in (summary or {}).get("min_separations", []):
        value = f(row.get("min_separation_au"))
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else math.nan

def linear_fit(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan, math.nan
    xs, ys = zip(*pairs)
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0.0:
        return math.nan, math.nan
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
    intercept = ybar - slope * xbar
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, r2

rows = []
with manifest.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        summary = load_json(item["summary_path"])
        megno = load_json(item["megno_summary_path"])
        warnings = []
        if item.get("status") == "failed":
            warnings.append(f"case failed; see {item.get('log_path')}")
        if summary is None:
            warnings.append("summary JSON missing")
            summary = {}
        if megno is None:
            warnings.append("MEGNO summary JSON missing")
            megno = {}
        warnings.extend(summary.get("warnings", []))
        warnings.extend(megno.get("caveats", []))
        extrema = summary.get("diagnostic_extrema_over_records", {})
        row = {
            "model_scope": item["model_scope"],
            "backend": "rebound",
            "integrator": item["integrator"],
            "gr_model": item["gr_model"],
            "duration_years": item["duration_years"],
            "step_days": item["step_days"],
            "runtime_seconds": summary.get("runtime", {}).get("wall_clock_seconds", ""),
            "final_megno": megno.get("final_megno", ""),
            "final_mean_megno": megno.get("final_mean_megno", ""),
            "estimated_lyapunov_if_available": megno.get("estimated_lyapunov_if_available", ""),
            "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
            "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
            "mercury_max_eccentricity": mercury_max_eccentricity(item["orbital_elements_path"]),
            "min_pairwise_separation_au": min_pairwise(summary),
            "classification": megno.get("classification_hint", "ambiguous"),
            "warnings": " | ".join(str(w) for w in warnings if w),
        }
        rows.append(row)

with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row[key]) if key not in {"model_scope", "backend", "integrator", "gr_model", "classification", "warnings"} else row[key] for key in FIELDS})

groups = defaultdict(list)
for row in rows:
    groups[(row["model_scope"], row["integrator"], row["gr_model"], str(row["step_days"]))].append(row)

scaling_rows = []
for (model_scope, integrator, gr_model, step_days), group in sorted(groups.items()):
    group = sorted(group, key=lambda row: f(row["duration_years"]))
    durations = [f(row["duration_years"]) for row in group]
    megnos = [f(row["final_megno"]) for row in group]
    lcns = [f(row["estimated_lyapunov_if_available"]) for row in group]
    finite_megnos = [value for value in megnos if math.isfinite(value)]
    slope, r2 = linear_fit(durations, megnos)
    if gr_model != "none":
        classification = "ambiguous"
        best_fit = "gr_variational_unvalidated"
        reason = "REBOUNDx warns that variational particles are not evolved self-consistently for GR"
    elif model_scope.startswith("two_body_"):
        if finite_megnos and max(abs(value - 2.0) for value in finite_megnos) <= 6.0:
            classification = "regular_likely"
            best_fit = "single_regular_gate" if len(group) < 2 else "bounded_regular"
            reason = "two-body validation MEGNO remains near regular expectation"
        else:
            classification = "ambiguous"
            best_fit = "ambiguous"
            reason = "two-body MEGNO did not pass the regular gate"
    elif len(group) < 2:
        classification = group[0].get("classification", "ambiguous")
        best_fit = "insufficient_duration_samples"
        reason = "only one duration in group"
    elif finite_megnos and max(finite_megnos) < 8.0:
        classification = "regular_likely"
        best_fit = "bounded_regular"
        reason = "MEGNO remains bounded near the regular range across durations"
    elif math.isfinite(slope) and slope > 0.0 and math.isfinite(r2) and r2 > 0.85 and finite_megnos and max(finite_megnos) > 10.0:
        classification = "chaotic_candidate"
        best_fit = "linear_growth_in_megno"
        reason = f"MEGNO grows approximately linearly with duration (r2={r2:.3f})"
    else:
        classification = "ambiguous"
        best_fit = "ambiguous"
        reason = "duration scaling does not show a robust bounded or linear-growth pattern"
    scaling_rows.append(
        {
            "model_scope": model_scope,
            "integrator": integrator,
            "gr_model": gr_model,
            "step_days": step_days,
            "duration_count": len(group),
            "durations_years": ",".join(fmt(value) for value in durations),
            "final_megno_values": ",".join(fmt(value) for value in megnos),
            "estimated_lyapunov_values": ",".join(fmt(value) for value in lcns),
            "best_fit_model": best_fit,
            "classification": classification,
            "reason": reason,
        }
    )

with scaling_csv.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=SCALING_FIELDS)
    writer.writeheader()
    writer.writerows(scaling_rows)

json_out.write_text(json.dumps({"rows": rows, "scaling_summary": scaling_rows}, indent=2, sort_keys=True) + "\n")
scaling_json.write_text(json.dumps(scaling_rows, indent=2, sort_keys=True) + "\n")

with md_out.open("w") as file_obj:
    file_obj.write("# REBOUND MEGNO Validation Ladder\n\n")
    file_obj.write("All MEGNO and LCN values are finite-time diagnostics. Two-body rows are validation gates, not chaos measurements.\n\n")
    file_obj.write("| model_scope | integrator | gr_model | duration_years | final_megno | LCN 1/yr | classification | runtime_s |\n")
    file_obj.write("| --- | --- | --- | ---: | ---: | ---: | --- | ---: |\n")
    for row in rows:
        file_obj.write(
            f"| {row['model_scope']} | {row['integrator']} | {row['gr_model']} | "
            f"{fmt(row['duration_years'])} | {fmt(row['final_megno'])} | "
            f"{fmt(row['estimated_lyapunov_if_available'])} | {row['classification']} | "
            f"{fmt(row['runtime_seconds'])} |\n"
        )
    file_obj.write("\n## Duration Scaling Summary\n\n")
    file_obj.write("| group | durations | MEGNO values | classification | reason |\n")
    file_obj.write("| --- | --- | --- | --- | --- |\n")
    for row in scaling_rows:
        group = f"{row['model_scope']} / {row['integrator']} / {row['gr_model']} / {row['step_days']} d"
        file_obj.write(
            f"| {group} | {row['durations_years']} | {row['final_megno_values']} | "
            f"{row['classification']} | {row['reason']} |\n"
        )

print(f"[REBOUND MEGNO] wrote {csv_out}")
print(f"[REBOUND MEGNO] wrote {json_out}")
print(f"[REBOUND MEGNO] wrote {md_out}")
print(f"[REBOUND MEGNO] wrote {scaling_csv}")
print(f"[REBOUND MEGNO] wrote {scaling_json}")
PY
