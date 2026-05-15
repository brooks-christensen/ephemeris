#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability}"
STAMP="${CHECKPOINT_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${OUTPUT_DIR}/checkpoint_restart_stress_manifest_${STAMP}.tsv"
CSV_OUT="${OUTPUT_DIR}/checkpoint_restart_stress_test.csv"
JSON_OUT="${OUTPUT_DIR}/checkpoint_restart_stress_test.json"
MD_OUT="${OUTPUT_DIR}/checkpoint_restart_stress_test.md"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

printf "backend\tstatus\ttag\tsummary_path\tcsv_path\tcheckpoint_dir\tarchive_path\tcorrupt_detection\twarning\n" > "${MANIFEST}"

append_manifest() {
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@" >> "${MANIFEST}"
}

has_module() {
  "${PYTHON_BIN}" -c "import ${1}" >/dev/null 2>&1
}

INHOUSE_TAG="checkpoint_stress_inhouse_${STAMP}"
INHOUSE_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints_${INHOUSE_TAG}"

echo "[CheckpointStress] in-house checkpoint create/resume/corrupt-check"
mkdir -p "${INHOUSE_CHECKPOINT_DIR}"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 2 \
  --step-days 8 \
  --record-every-years 1 \
  --gr-model none \
  --integrator leapfrog \
  --model-scope two_body_jupiter \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${INHOUSE_TAG}" \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body jupiter \
  --lyapunov-perturbation-m 1 \
  --lyapunov-renorm-years 1 \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${INHOUSE_CHECKPOINT_DIR}" \
  --keep-checkpoints 3 \
  --no-progress-bar

LATEST_CHECKPOINT="$(find "${INHOUSE_CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'checkpoint_*.npz' | sort | tail -n 1)"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 4 \
  --step-days 8 \
  --record-every-years 1 \
  --gr-model none \
  --integrator leapfrog \
  --model-scope two_body_jupiter \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${INHOUSE_TAG}" \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body jupiter \
  --lyapunov-perturbation-m 1 \
  --lyapunov-renorm-years 1 \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${INHOUSE_CHECKPOINT_DIR}" \
  --resume-from-checkpoint "${LATEST_CHECKPOINT}" \
  --keep-checkpoints 3 \
  --no-progress-bar

CORRUPT_CHECKPOINT="${INHOUSE_CHECKPOINT_DIR}/corrupt_checkpoint.npz"
printf '\0not-a-valid-checkpoint' > "${CORRUPT_CHECKPOINT}"
set +e
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 5 \
  --step-days 8 \
  --record-every-years 1 \
  --gr-model none \
  --integrator leapfrog \
  --model-scope two_body_jupiter \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${INHOUSE_TAG}" \
  --resume-from-checkpoint "${CORRUPT_CHECKPOINT}" \
  --no-progress-bar >/tmp/mini_ephemeris_corrupt_checkpoint_${STAMP}.log 2>&1
CORRUPT_EXIT=$?
set -e
if [[ "${CORRUPT_EXIT}" == "0" ]]; then
  CORRUPT_DETECTION="failed"
else
  CORRUPT_DETECTION="passed"
fi
append_manifest "inhouse" "completed" "${INHOUSE_TAG}" "${OUTPUT_DIR}/summary_${INHOUSE_TAG}.json" "${OUTPUT_DIR}/invariants_${INHOUSE_TAG}.csv" "${INHOUSE_CHECKPOINT_DIR}" "" "${CORRUPT_DETECTION}" ""

if has_module rebound; then
  REBOUND_TAG="checkpoint_stress_rebound_${STAMP}"
  REBOUND_ARCHIVE="${OUTPUT_DIR}/${REBOUND_TAG}.bin"
  echo "[CheckpointStress] REBOUND SimulationArchive create/resume"
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_jupiter \
    --duration-years 2 \
    --step-days 8 \
    --record-every-years 1 \
    --integrator whfast \
    --gr-model none \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${REBOUND_TAG}" \
    --simulation-archive "${REBOUND_ARCHIVE}" \
    --simulation-archive-interval-years 1
  "${PYTHON_BIN}" -m mini_ephemeris.rebound_validation_cli \
    --kernel-path "${KERNEL_PATH}" \
    --start-date 2000-01-01 \
    --model-scope two_body_jupiter \
    --duration-years 4 \
    --step-days 8 \
    --record-every-years 1 \
    --integrator whfast \
    --gr-model none \
    --output-dir "${OUTPUT_DIR}" \
    --tag "${REBOUND_TAG}" \
    --resume-from-simulation-archive "${REBOUND_ARCHIVE}"
  append_manifest "rebound" "completed" "${REBOUND_TAG}" "${OUTPUT_DIR}/rebound_validation_summary_${REBOUND_TAG}.json" "${OUTPUT_DIR}/rebound_validation_${REBOUND_TAG}.csv" "" "${REBOUND_ARCHIVE}" "not_tested" ""
else
  append_manifest "rebound" "skipped" "checkpoint_stress_rebound_${STAMP}" "" "" "" "" "not_tested" "rebound is not installed"
fi

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
    "status",
    "tag",
    "runtime_seconds",
    "checkpoint_count",
    "archive_exists",
    "output_rows",
    "time_monotonic",
    "final_time_years",
    "append_continuity",
    "corrupt_detection",
    "warning",
]

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def read_summary_runtime(path):
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    with p.open() as file_obj:
        summary = json.load(file_obj)
    if "runtime" in summary:
        return summary.get("runtime", {}).get("wall_clock_seconds", "")
    return summary.get("diagnostics", {}).get("runtime_seconds", "")

def csv_continuity(path):
    if not path:
        return 0, False, "", False
    p = Path(path)
    if not p.exists():
        return 0, False, "", False
    times = []
    with p.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            t = as_float(row.get("time_years"))
            if math.isfinite(t):
                times.append(t)
    monotonic = all(b >= a for a, b in zip(times, times[1:]))
    final_time = max(times) if times else math.nan
    append_continuity = bool(times and monotonic and final_time >= 3.9 and len(times) >= 5)
    return len(times), monotonic, final_time, append_continuity

rows = []
with manifest.open(newline="") as file_obj:
    for item in csv.DictReader(file_obj, delimiter="\t"):
        checkpoint_count = 0
        if item["checkpoint_dir"] and Path(item["checkpoint_dir"]).exists():
            checkpoint_count = len(list(Path(item["checkpoint_dir"]).glob("checkpoint_*.npz")))
        archive_exists = bool(item["archive_path"] and Path(item["archive_path"]).exists())
        output_rows, monotonic, final_time, append_continuity = csv_continuity(item["csv_path"])
        row = {
            "backend": item["backend"],
            "status": item["status"],
            "tag": item["tag"],
            "runtime_seconds": read_summary_runtime(item["summary_path"]),
            "checkpoint_count": checkpoint_count,
            "archive_exists": archive_exists,
            "output_rows": output_rows,
            "time_monotonic": monotonic,
            "final_time_years": final_time if math.isfinite(as_float(final_time)) else "",
            "append_continuity": append_continuity,
            "corrupt_detection": item["corrupt_detection"],
            "warning": item["warning"],
        }
        rows.append(row)

with csv_out.open("w", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
with json_out.open("w") as file_obj:
    json.dump({"rows": rows}, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")

lines = [
    "# Checkpoint Restart Stress Test",
    "",
    "| backend | status | checkpoint_count | archive_exists | rows | monotonic | final_time_years | append_continuity | corrupt_detection | warning |",
    "|---|---|---:|---|---:|---|---:|---|---|---|",
]
for row in rows:
    lines.append(
        f"| {row['backend']} | {row['status']} | {row['checkpoint_count']} | {row['archive_exists']} | "
        f"{row['output_rows']} | {row['time_monotonic']} | {row['final_time_years']} | "
        f"{row['append_continuity']} | {row['corrupt_detection']} | {row['warning']} |"
    )
md_out.write_text("\n".join(lines) + "\n")

print(f"[CheckpointStress] wrote {csv_out}")
print(f"[CheckpointStress] wrote {json_out}")
print(f"[CheckpointStress] wrote {md_out}")
PY

echo "[CheckpointStress] csv=${CSV_OUT}"
echo "[CheckpointStress] json=${JSON_OUT}"
echo "[CheckpointStress] md=${MD_OUT}"
