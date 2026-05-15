#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
RESUME="${RESUME:-1}"

MANIFEST="${OUTPUT_DIR}/backend_comparison_ladder_manifest.tsv"
CSV_OUT="${OUTPUT_DIR}/backend_comparison_ladder.csv"
JSON_OUT="${OUTPUT_DIR}/backend_comparison_ladder.json"
MD_OUT="${OUTPUT_DIR}/backend_comparison_ladder.md"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

printf "backend\tintegrator\tgr_model\tduration_years\ttag\tsummary_path\torbital_elements_path\n" > "${MANIFEST}"

run_case() {
  local backend="$1"
  local rebound_integrator="$2"
  local rebound_gr="$3"
  local duration="$4"
  local step="$5"
  local record="$6"
  local tag="$7"
  local summary_path="${OUTPUT_DIR}/summary_${tag}.json"
  local elements_path="${OUTPUT_DIR}/orbital_elements_${tag}.csv"

  if [[ "${RESUME}" == "1" && -f "${summary_path}" ]]; then
    echo "[Comparison] RESUME=1 and ${summary_path} exists; skipping."
  else
    if [[ "${backend}" == "inhouse" ]]; then
      "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
        --kernel-path "${KERNEL_PATH}" \
        --start-date 2000-01-01 \
        --backend inhouse \
        --integrator leapfrog \
        --gr-model none \
        --model-scope inner \
        --duration-years "${duration}" \
        --step-days "${step}" \
        --record-every-years "${record}" \
        --output-dir "${OUTPUT_DIR}" \
        --tag "${tag}" \
        --no-progress-bar
    else
      "${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
        --kernel-path "${KERNEL_PATH}" \
        --start-date 2000-01-01 \
        --backend rebound \
        --rebound-integrator "${rebound_integrator}" \
        --rebound-gr-model "${rebound_gr}" \
        --model-scope inner \
        --duration-years "${duration}" \
        --step-days "${step}" \
        --record-every-years "${record}" \
        --gr-model none \
        --integrator leapfrog \
        --output-dir "${OUTPUT_DIR}" \
        --tag "${tag}" \
        --no-progress-bar
    fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${backend}" "${rebound_integrator}" "${rebound_gr}" "${duration}" \
    "${tag}" "${summary_path}" "${elements_path}" >> "${MANIFEST}"
}

run_case "inhouse" "leapfrog" "none" "10000" "1" "10" "comparison_inhouse_inner_10kyr_newtonian"
run_case "rebound" "whfast" "none" "10000" "1" "10" "comparison_rebound_whfast_inner_10kyr_newtonian"
run_case "rebound" "whfast" "gr_potential" "10000" "1" "10" "comparison_rebound_whfast_inner_10kyr_gr_potential"
run_case "rebound" "ias15" "none" "1000" "1" "10" "comparison_rebound_ias15_inner_1kyr_newtonian"
run_case "rebound" "ias15" "gr" "1000" "1" "10" "comparison_rebound_ias15_inner_1kyr_gr"

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

FIELDS = [
    "backend",
    "integrator",
    "gr_model",
    "duration_years",
    "runtime_seconds",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "max_com_velocity_drift",
    "mercury_perihelion_drift_arcsec_per_century",
    "max_eccentricity_mercury",
    "max_eccentricity_venus",
    "max_eccentricity_earth",
    "max_eccentricity_mars",
    "min_pairwise_separation_au",
    "warnings",
    "recommended_use",
]

def f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def fmt(value):
    value = f(value)
    return f"{value:.6g}" if math.isfinite(value) else ""

def unwrap_delta_deg(current, previous):
    return (current - previous + 180.0) % 360.0 - 180.0

def mercury_perihelion_from_elements(path):
    rows = []
    with Path(path).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("body") == "mercury barycenter":
                t = f(row.get("time_years"))
                varpi = f(row.get("varpi_deg"))
                if math.isfinite(t) and math.isfinite(varpi):
                    rows.append((t, varpi))
    if len(rows) < 2:
        return math.nan
    unwrapped = []
    current = rows[0][1]
    previous = rows[0][1]
    for t, varpi in rows:
        if unwrapped:
            current += unwrap_delta_deg(varpi, previous)
            previous = varpi
        unwrapped.append((t, current))
    xs = [item[0] for item in unwrapped]
    ys = [item[1] for item in unwrapped]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0.0:
        return math.nan
    slope_deg_per_year = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom
    return slope_deg_per_year * 3600.0 * 100.0

def eccentricity_maxima(path):
    mapping = {
        "mercury barycenter": "max_eccentricity_mercury",
        "venus barycenter": "max_eccentricity_venus",
        "earth barycenter": "max_eccentricity_earth",
        "mars barycenter": "max_eccentricity_mars",
    }
    maxima = {key: "" for key in mapping.values()}
    with Path(path).open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            key = mapping.get(row.get("body"))
            if key is None:
                continue
            e = f(row.get("e"))
            if not math.isfinite(e):
                continue
            current = f(maxima[key])
            maxima[key] = e if not math.isfinite(current) else max(current, e)
    return maxima

def min_pairwise(summary):
    values = []
    for row in summary.get("min_separations", []):
        value = f(row.get("min_separation_au"))
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else math.nan

def recommended(row):
    if row["backend"] == "inhouse":
        return "tangent Lyapunov/FLI/MEGNO and current checkpointed diagnostics"
    if row["integrator"] == "whfast" and row["gr_model"] == "none":
        return "leading Newtonian production trajectory candidate"
    if row["integrator"] == "whfast" and row["gr_model"] == "gr_potential":
        return "leading WHFast-compatible GR candidate pending longer validation"
    if row["integrator"] == "ias15":
        return "high-accuracy short validation oracle"
    return "exploratory"

rows = []
with manifest.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        summary_path = Path(item["summary_path"])
        elements_path = Path(item["orbital_elements_path"])
        if not summary_path.exists():
            continue
        summary = json.load(summary_path.open())
        runtime = summary.get("runtime", {}).get("wall_clock_seconds", "")
        extrema = summary.get("diagnostic_extrema_over_records", {})
        warnings = summary.get("warnings", [])
        ecc = eccentricity_maxima(elements_path) if elements_path.exists() else {}
        row = {
            "backend": item["backend"],
            "integrator": item["integrator"],
            "gr_model": item["gr_model"],
            "duration_years": item["duration_years"],
            "runtime_seconds": runtime,
            "max_energy_rel_drift": extrema.get("max_abs_energy_rel_drift", ""),
            "max_angular_momentum_rel_drift": extrema.get("max_angular_momentum_rel_drift", ""),
            "max_com_velocity_drift": extrema.get("max_com_velocity_drift_au_per_year", ""),
            "mercury_perihelion_drift_arcsec_per_century": (
                mercury_perihelion_from_elements(elements_path)
                if elements_path.exists()
                else math.nan
            ),
            "min_pairwise_separation_au": min_pairwise(summary),
            "warnings": "; ".join(str(item) for item in warnings[:5]) if isinstance(warnings, list) else "",
        }
        row.update({key: ecc.get(key, "") for key in [
            "max_eccentricity_mercury",
            "max_eccentricity_venus",
            "max_eccentricity_earth",
            "max_eccentricity_mars",
        ]})
        row["recommended_use"] = recommended(row)
        rows.append(row)

with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "rows": rows,
    "recommendation": {
        "production_backend": "REBOUND WHFast Newtonian for trajectory production; keep in-house for tangent diagnostics until REBOUND tangent support exists.",
        "gr_model": "REBOUNDx gr_potential with WHFast is the preferred GR candidate pending longer validation.",
        "validation_oracle": "REBOUND IAS15 for short high-accuracy checks.",
    },
}
with json_out.open("w") as file_obj:
    json.dump(payload, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")

lines = [
    "# Backend Comparison Ladder",
    "",
    "| backend | integrator | gr | years | runtime_s | energy_rel | angmom_rel | com_v | Mercury perihelion | min_sep_au | recommended_use |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(
        f"| {row['backend']} | {row['integrator']} | {row['gr_model']} | "
        f"{row['duration_years']} | {fmt(row['runtime_seconds'])} | "
        f"{fmt(row['max_energy_rel_drift'])} | {fmt(row['max_angular_momentum_rel_drift'])} | "
        f"{fmt(row['max_com_velocity_drift'])} | {fmt(row['mercury_perihelion_drift_arcsec_per_century'])} | "
        f"{fmt(row['min_pairwise_separation_au'])} | {row['recommended_use']} |"
    )
lines.extend([
    "",
    "## Recommendation",
    "",
    "- Production trajectory backend: REBOUND WHFast Newtonian.",
    "- Tangent diagnostics: keep using the in-house backend.",
    "- GR candidate: WHFast + REBOUNDx gr_potential, pending longer validation.",
    "- Short validation oracle: REBOUND IAS15.",
])
md_out.write_text("\n".join(lines) + "\n")
print(f"[Comparison] wrote {csv_out}")
print(f"[Comparison] wrote {json_out}")
print(f"[Comparison] wrote {md_out}")
PY
