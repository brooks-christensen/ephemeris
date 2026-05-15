#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STAMP="${BENCHMARK_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TAG_PREFIX="${TAG_PREFIX:-backend_bench_${STAMP}}"
MAX_CASES="${MAX_CASES:-0}"
IAS15_EPSILON="${IAS15_EPSILON:-1e-10}"

MANIFEST="${OUTPUT_DIR}/backend_accuracy_benchmark_manifest_${STAMP}.tsv"
SKIPPED="${OUTPUT_DIR}/backend_accuracy_benchmark_skipped_${STAMP}.tsv"
CSV_OUT="${OUTPUT_DIR}/backend_accuracy_benchmark.csv"
JSON_OUT="${OUTPUT_DIR}/backend_accuracy_benchmark.json"
MD_OUT="${OUTPUT_DIR}/backend_accuracy_benchmark.md"

CASE_COUNT=0

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

printf "backend\tintegrator\tgr_model\tmodel_scope\tduration_years\tstep_days\ttolerance\ttag\tsummary_path\telements_path\twarnings\n" > "${MANIFEST}"
printf "backend\tintegrator\tgr_model\tmodel_scope\treason\n" > "${SKIPPED}"

has_module() {
  "${PYTHON_BIN}" -c "import ${1}" >/dev/null 2>&1
}

should_run_case() {
  if [[ "${MAX_CASES}" != "0" && "${CASE_COUNT}" -ge "${MAX_CASES}" ]]; then
    return 1
  fi
  CASE_COUNT=$((CASE_COUNT + 1))
  return 0
}

append_manifest() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${MANIFEST}"
}

append_skipped() {
  printf "%s\t%s\t%s\t%s\t%s\n" "$@" >> "${SKIPPED}"
}

run_inhouse_case() {
  local model_scope="$1"
  local duration_years="$2"
  local step_days="$3"
  local record_every_years="$4"
  local tag="${TAG_PREFIX}_inhouse_leapfrog_none_${model_scope}_${duration_years}yr"
  local summary_path="${OUTPUT_DIR}/summary_${tag}.json"
  local elements_path="${OUTPUT_DIR}/orbital_elements_${tag}.csv"

  should_run_case || return 0
  echo "[Benchmark] inhouse leapfrog none ${model_scope} duration=${duration_years}yr step=${step_days}d"
  "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --duration-years "${duration_years}" \
    --step-days "${step_days}" \
    --record-every-years "${record_every_years}" \
    --gr-model none \
    --integrator leapfrog \
    --model-scope "${model_scope}" \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${tag}" \
    --no-progress-bar
  append_manifest "inhouse" "leapfrog" "none" "${model_scope}" "${duration_years}" "${step_days}" "" "${tag}" "${summary_path}" "${elements_path}" ""
}

run_rebound_case() {
  local integrator="$1"
  local gr_model="$2"
  local model_scope="$3"
  local duration_years="$4"
  local step_days="$5"
  local record_every_years="$6"
  local tolerance=""
  local tag="${TAG_PREFIX}_rebound_${integrator}_${gr_model}_${model_scope}_${duration_years}yr"
  local summary_path="${OUTPUT_DIR}/rebound_validation_summary_${tag}.json"
  local elements_path="${OUTPUT_DIR}/rebound_validation_${tag}.csv"

  if [[ "${integrator}" == "ias15" ]]; then
    tolerance="${IAS15_EPSILON}"
  fi
  should_run_case || return 0
  echo "[Benchmark] rebound ${integrator} ${gr_model} ${model_scope} duration=${duration_years}yr step=${step_days}d tolerance=${tolerance:-n/a}"
  local args=(
    -m mini_ephemeris.rebound_validation_cli
    --kernel-path "${KERNEL_PATH}"
    --start-date 2000-01-01
    --model-scope "${model_scope}"
    --duration-years "${duration_years}"
    --step-days "${step_days}"
    --record-every-years "${record_every_years}"
    --integrator "${integrator}"
    --gr-model "${gr_model}"
    --output-dir "${OUTPUT_DIR}"
    --tag "${tag}"
  )
  if [[ -n "${tolerance}" ]]; then
    args+=(--ias15-epsilon "${tolerance}")
  fi
  "${PYTHON_BIN}" "${args[@]}"
  append_manifest "rebound" "${integrator}" "${gr_model}" "${model_scope}" "${duration_years}" "${step_days}" "${tolerance}" "${tag}" "${summary_path}" "${elements_path}" ""
}

echo "[Benchmark] writing outputs to ${OUTPUT_DIR}"
echo "[Benchmark] MAX_CASES=${MAX_CASES} (0 means all scheduled cases)"

REBOUND_AVAILABLE=0
REBOUNDX_AVAILABLE=0
if has_module rebound; then
  REBOUND_AVAILABLE=1
fi
if has_module reboundx; then
  REBOUNDX_AVAILABLE=1
fi

run_inhouse_case "two_body_jupiter" "1000" "4" "10"
run_inhouse_case "two_body_saturn" "1000" "4" "10"
run_inhouse_case "two_body_mercury" "100" "0.125" "1"
run_inhouse_case "inner" "1000" "1" "10"

if [[ "${REBOUND_AVAILABLE}" == "1" ]]; then
  for integrator in whfast ias15; do
    run_rebound_case "${integrator}" "none" "two_body_jupiter" "1000" "4" "10"
    run_rebound_case "${integrator}" "none" "two_body_saturn" "1000" "4" "10"
    run_rebound_case "${integrator}" "none" "two_body_mercury" "100" "0.125" "1"
    run_rebound_case "${integrator}" "none" "inner" "1000" "1" "10"
  done
else
  append_skipped "rebound" "whfast" "none" "all" "rebound is not installed"
  append_skipped "rebound" "ias15" "none" "all" "rebound is not installed"
fi

if [[ "${REBOUND_AVAILABLE}" == "1" && "${REBOUNDX_AVAILABLE}" == "1" ]]; then
  run_rebound_case "whfast" "gr" "two_body_mercury" "100" "0.125" "1"
  run_rebound_case "whfast" "gr_potential" "two_body_mercury" "100" "0.125" "1"
  run_rebound_case "ias15" "gr" "two_body_mercury" "100" "0.125" "1"
  run_rebound_case "whfast" "gr" "inner" "1000" "1" "10"
  run_rebound_case "whfast" "gr_potential" "inner" "1000" "1" "10"
  run_rebound_case "ias15" "gr" "inner" "1000" "1" "10"
else
  if [[ "${REBOUND_AVAILABLE}" == "1" ]]; then
    append_skipped "rebound" "whfast" "gr" "two_body_mercury,inner" "reboundx is not installed"
    append_skipped "rebound" "whfast" "gr_potential" "two_body_mercury,inner" "reboundx is not installed"
    append_skipped "rebound" "ias15" "gr" "two_body_mercury,inner" "reboundx is not installed"
  fi
fi

"${PYTHON_BIN}" - "${MANIFEST}" "${SKIPPED}" "${CSV_OUT}" "${JSON_OUT}" "${MD_OUT}" "${REBOUND_AVAILABLE}" "${REBOUNDX_AVAILABLE}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
skipped_path = Path(sys.argv[2])
csv_out = Path(sys.argv[3])
json_out = Path(sys.argv[4])
md_out = Path(sys.argv[5])
rebound_available = sys.argv[6] == "1"
reboundx_available = sys.argv[7] == "1"

FIELDS = [
    "backend",
    "integrator",
    "gr_model",
    "model_scope",
    "duration_years",
    "step_days_or_tolerance",
    "runtime_seconds",
    "steps_per_second",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "max_com_velocity_drift",
    "mercury_perihelion_drift_arcsec_per_century",
    "final_orbital_element_deltas",
    "warnings",
    "recommended_use",
]

def as_float(value):
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def finite(value):
    value = as_float(value)
    return math.isfinite(value)

def sort_value(value, missing=math.inf):
    value = as_float(value)
    return value if math.isfinite(value) else missing

def read_json(path):
    with Path(path).open() as file_obj:
        return json.load(file_obj)

def orbital_delta_summary(path):
    path = Path(path)
    if not path.exists():
        return "", math.nan, math.nan, math.nan
    first = {}
    last = {}
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if not body:
                continue
            if body not in first:
                first[body] = row
            last[body] = row
    max_da = 0.0
    max_de = 0.0
    max_di = 0.0
    parts = []
    for body in sorted(first):
        a0 = as_float(first[body].get("a_au"))
        a1 = as_float(last[body].get("a_au"))
        e0 = as_float(first[body].get("e"))
        e1 = as_float(last[body].get("e"))
        i0 = as_float(first[body].get("i_deg"))
        i1 = as_float(last[body].get("i_deg"))
        da = a1 - a0 if math.isfinite(a0) and math.isfinite(a1) else math.nan
        de = e1 - e0 if math.isfinite(e0) and math.isfinite(e1) else math.nan
        di = i1 - i0 if math.isfinite(i0) and math.isfinite(i1) else math.nan
        if math.isfinite(da):
            max_da = max(max_da, abs(da))
        if math.isfinite(de):
            max_de = max(max_de, abs(de))
        if math.isfinite(di):
            max_di = max(max_di, abs(di))
        short = body.replace(" barycenter", "").replace("earth", "emb")
        parts.append(f"{short}:da={da:.3e},de={de:.3e},di={di:.3e}")
    return "; ".join(parts[:8]), max_da, max_de, max_di

def recommended_use(row):
    backend = row["backend"]
    integrator = row["integrator"]
    gr_model = row["gr_model"]
    if backend == "inhouse" and gr_model == "none":
        return "transparent Newtonian baseline; checkpointed production candidate after timestep validation"
    if backend == "rebound" and integrator == "whfast" and gr_model == "none":
        return "candidate Newtonian symplectic production backend after matching conservation checks"
    if backend == "rebound" and integrator == "ias15" and gr_model == "none":
        return "high-accuracy short validation candidate"
    if backend == "rebound" and gr_model != "none":
        return "GR validation scaffold; require Mercury perihelion and conservation validation before production"
    return "exploratory benchmark case"

rows = []
with manifest_path.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        summary_path = item["summary_path"]
        if not summary_path or not Path(summary_path).exists():
            continue
        summary = read_json(summary_path)
        element_delta_text, max_da, max_de, max_di = orbital_delta_summary(item["elements_path"])
        if item["backend"] == "inhouse":
            config = summary.get("configuration", {})
            runtime = summary.get("runtime", {}).get("wall_clock_seconds", "")
            counts = summary.get("counts", {})
            n_steps = as_float(counts.get("n_steps_or_nominal_max_steps", ""))
            extrema = summary.get("diagnostic_extrema_over_records", {})
            two_body = summary.get("two_body_validation", {}).get("diagnostics", {})
            step_or_tol = item["step_days"]
            warnings = []
            if summary.get("integrator_notes", {}).get("gr_leapfrog_symplectic_note"):
                warnings.append(summary["integrator_notes"]["gr_leapfrog_symplectic_note"])
            row = {
                "backend": "inhouse",
                "integrator": item["integrator"],
                "gr_model": item["gr_model"],
                "model_scope": item["model_scope"],
                "duration_years": item["duration_years"],
                "step_days_or_tolerance": step_or_tol,
                "runtime_seconds": runtime,
                "steps_per_second": n_steps / as_float(runtime) if math.isfinite(n_steps) and as_float(runtime) > 0 else "",
                "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
                "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
                "max_com_velocity_drift": extrema.get("max_com_velocity_drift_au_per_year", ""),
                "mercury_perihelion_drift_arcsec_per_century": two_body.get("estimated_perihelion_drift_arcsec_per_century", ""),
                "final_orbital_element_deltas": element_delta_text,
                "warnings": "; ".join(warnings),
            }
        else:
            config = summary.get("configuration", {})
            diagnostics = summary.get("diagnostics", {})
            runtime = diagnostics.get("runtime_seconds", "")
            step_or_tol = item["tolerance"] if item["integrator"] == "ias15" and item["tolerance"] else item["step_days"]
            n_steps = as_float(item["duration_years"]) * 365.25 / as_float(item["step_days"])
            warnings = []
            if item["gr_model"] != "none":
                warnings.append("REBOUNDx GR scaffold; not yet production validated")
            if item["integrator"] == "whfast" and item["gr_model"] == "gr":
                warnings.append("WHFast plus velocity-dependent REBOUNDx gr should be validated as an operator path before production")
            row = {
                "backend": "rebound",
                "integrator": item["integrator"],
                "gr_model": item["gr_model"],
                "model_scope": item["model_scope"],
                "duration_years": item["duration_years"],
                "step_days_or_tolerance": step_or_tol,
                "runtime_seconds": runtime,
                "steps_per_second": (
                    n_steps / as_float(runtime)
                    if item["integrator"] in {"whfast", "leapfrog"} and math.isfinite(n_steps) and as_float(runtime) > 0
                    else ""
                ),
                "max_energy_rel_drift": diagnostics.get("max_energy_rel_drift", ""),
                "max_angular_momentum_rel_drift": diagnostics.get("max_angular_momentum_rel_drift", ""),
                "max_com_velocity_drift": diagnostics.get("max_com_velocity_drift_au_per_year", ""),
                "mercury_perihelion_drift_arcsec_per_century": diagnostics.get("mercury_perihelion_drift_arcsec_per_century", ""),
                "final_orbital_element_deltas": element_delta_text,
                "warnings": "; ".join(warnings),
            }
        row["recommended_use"] = recommended_use(row)
        row["_max_da_au"] = max_da
        row["_max_de"] = max_de
        row["_max_di_deg"] = max_di
        rows.append(row)

skipped = []
with skipped_path.open(newline="") as file_obj:
    for row in csv.DictReader(file_obj, delimiter="\t"):
        skipped.append(row)

def top(rows_in, key, limit=5):
    return [
        {k: v for k, v in row.items() if not k.startswith("_")}
        for row in sorted(rows_in, key=key)[:limit]
    ]

rankings = {
    "fastest": top(rows, lambda r: sort_value(r["runtime_seconds"])),
    "best_conservation": top(
        rows,
        lambda r: (
            sort_value(r["max_energy_rel_drift"]),
            sort_value(r["max_angular_momentum_rel_drift"]),
            sort_value(r["max_com_velocity_drift"]),
        ),
    ),
}

mercury_rows = [r for r in rows if finite(r["mercury_perihelion_drift_arcsec_per_century"])]
rankings["best_mercury_perihelion_behavior"] = top(
    mercury_rows,
    lambda r: abs(
        sort_value(r["mercury_perihelion_drift_arcsec_per_century"])
        - (42.98 if r["gr_model"] != "none" else 0.0)
    ),
)
production_candidates = [
    r for r in rows
    if r["gr_model"] == "none" and (
        (r["backend"] == "inhouse" and r["integrator"] == "leapfrog")
        or (r["backend"] == "rebound" and r["integrator"] == "whfast")
    )
]
def production_scope_priority(row):
    return 0 if row["model_scope"] in {"inner", "full"} else 1
rankings["best_week_long_production_candidates"] = top(
    production_candidates,
    lambda r: (
        production_scope_priority(r),
        sort_value(r["max_energy_rel_drift"]),
        sort_value(r["max_angular_momentum_rel_drift"]),
        sort_value(r["runtime_seconds"]),
    ),
)
validation_candidates = [r for r in rows if r["integrator"] == "ias15"] or rows
rankings["best_high_accuracy_short_validation_candidates"] = top(
    validation_candidates,
    lambda r: (
        sort_value(r["max_energy_rel_drift"]),
        sort_value(r["max_angular_momentum_rel_drift"]),
        sort_value(r["runtime_seconds"]),
    ),
)

recommendations = {
    "production_backend": (
        "inhouse leapfrog Newtonian remains the only available benchmarked backend; install REBOUND to make a backend decision."
        if not rebound_available
        else "Compare inhouse leapfrog and REBOUND WHFast within matching model_scope rows; prefer the best_week_long_production_candidates ranking."
    ),
    "gr_model": (
        "none; REBOUNDx is unavailable and the in-house gr_model=sun path is not the validated tangent/conservation path."
        if not reboundx_available
        else "REBOUNDx GR remains validation-only until Mercury perihelion and conservation rankings pass."
    ),
    "checkpoint_interval": "Start with 100 years for first long inner/full shakedowns; use 10 years during initial week-long restart testing.",
    "first_long_run": "Run a 10-30 kyr inner Newtonian checkpointed case before any expensive GR or ensemble run.",
}

public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(public_rows)

payload = {
    "backend_availability": {
        "rebound": rebound_available,
        "reboundx": reboundx_available,
    },
    "rows": public_rows,
    "rankings": rankings,
    "skipped_cases": skipped,
    "recommendations": recommendations,
}
with json_out.open("w") as file_obj:
    json.dump(payload, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")

def fmt(value):
    if value in ("", None):
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(f):
        return f"{f:.6g}"
    return ""

lines = [
    "# Backend Accuracy Benchmark",
    "",
    "Finite-time validation benchmark for stability mode only. This is not a final Solar System chaos claim.",
    "",
    f"- REBOUND available: {rebound_available}",
    f"- REBOUNDx available: {reboundx_available}",
    "",
    "## Results",
    "",
    "| backend | integrator | gr_model | model_scope | years | step/tol | runtime_s | energy_rel | angmom_rel | com_v_au_yr | Mercury perihelion arcsec/century | recommended_use |",
    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in public_rows:
    lines.append(
        "| {backend} | {integrator} | {gr_model} | {model_scope} | {duration_years} | {step_days_or_tolerance} | {runtime} | {energy} | {ang} | {com} | {peri} | {use} |".format(
            backend=row["backend"],
            integrator=row["integrator"],
            gr_model=row["gr_model"],
            model_scope=row["model_scope"],
            duration_years=row["duration_years"],
            step_days_or_tolerance=row["step_days_or_tolerance"],
            runtime=fmt(row["runtime_seconds"]),
            energy=fmt(row["max_energy_rel_drift"]),
            ang=fmt(row["max_angular_momentum_rel_drift"]),
            com=fmt(row["max_com_velocity_drift"]),
            peri=fmt(row["mercury_perihelion_drift_arcsec_per_century"]),
            use=row["recommended_use"].replace("|", "/"),
        )
    )
lines.extend(["", "## Rankings"])
for name, ranked_rows in rankings.items():
    lines.append("")
    lines.append(f"### {name}")
    if not ranked_rows:
        lines.append("No completed rows.")
        continue
    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            f"{index}. {row['backend']} {row['integrator']} {row['gr_model']} {row['model_scope']} "
            f"runtime={fmt(row['runtime_seconds'])} energy={fmt(row['max_energy_rel_drift'])}"
        )
lines.extend(["", "## Skipped Cases"])
if skipped:
    for row in skipped:
        lines.append(
            f"- {row['backend']} {row['integrator']} {row['gr_model']} {row['model_scope']}: {row['reason']}"
        )
else:
    lines.append("None.")
lines.extend(["", "## Recommendations"])
for key, value in recommendations.items():
    lines.append(f"- {key}: {value}")
md_out.write_text("\n".join(lines) + "\n")

print(f"[Benchmark] wrote {csv_out}")
print(f"[Benchmark] wrote {json_out}")
print(f"[Benchmark] wrote {md_out}")
PY

echo "[Benchmark] completed_cases=${CASE_COUNT}"
echo "[Benchmark] csv=${CSV_OUT}"
echo "[Benchmark] json=${JSON_OUT}"
echo "[Benchmark] md=${MD_OUT}"
