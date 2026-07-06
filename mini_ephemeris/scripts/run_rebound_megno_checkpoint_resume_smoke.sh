#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/rebound_megno_checkpoint_resume_smoke}"
TAG="${TAG:-rebound_megno_checkpoint_resume_smoke}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_DIR}/${TAG}.bin}"

mkdir -p "${OUTPUT_DIR}"
export OUTPUT_DIR TAG
cd "${PROJECT_DIR}"

COMMON_ARGS=(
  --kernel-path "${KERNEL_PATH}"
  --start-date 2000-01-01
  --backend rebound
  --rebound-integrator whfast
  --rebound-gr-model none
  --model-scope inner
  --step-days 8
  --record-every-years 100
  --output-dir "${OUTPUT_DIR}"
  --tag "${TAG}"
  --with-megno
  --with-rebound-lyapunov
  --megno-seed 123
  --megno-record-every-years 100
  --rebound-simulationarchive "${ARCHIVE_PATH}"
  --rebound-archive-interval-years 250
  --no-progress-bar
)

echo "[smoke] Phase 1: create partial 1000-year REBOUND MEGNO run"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  "${COMMON_ARGS[@]}" \
  --duration-years 1000

echo "[smoke] Phase 2: resume from SimulationArchive and continue to 2000 years"
"${PYTHON_BIN}" -m mini_ephemeris.long_term_stability_cli \
  "${COMMON_ARGS[@]}" \
  --duration-years 2000 \
  --rebound-resume latest

echo "[smoke] Verifying resume metadata and CSV continuity"
"${PYTHON_BIN}" - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
tag = os.environ["TAG"]
summary_path = output_dir / f"summary_{tag}.json"
megno_summary_path = output_dir / f"megno_summary_{tag}.json"
megno_csv = output_dir / f"megno_{tag}.csv"
stability_csv = output_dir / f"stability_timeseries_{tag}.csv"
elements_csv = output_dir / f"orbital_elements_{tag}.csv"
invariants_csv = output_dir / f"invariants_{tag}.csv"
archive_path = output_dir / f"{tag}.bin"

for path in (summary_path, megno_summary_path, megno_csv, stability_csv, elements_csv, invariants_csv, archive_path):
    if not path.exists():
        raise SystemExit(f"missing expected output: {path}")
    data = path.read_bytes()
    if b"\x00" in data and path.suffix == ".csv":
        raise SystemExit(f"NUL byte detected in {path}")

summary = json.loads(summary_path.read_text())
resume = summary.get("rebound_resume", {})
if not resume.get("archive_path"):
    raise SystemExit("summary missing rebound_resume archive_path")
if float(resume.get("resumed_from_time_years") or 0.0) <= 0.0:
    raise SystemExit("resume did not load from nonzero time")
if not resume.get("megno_state_validated"):
    raise SystemExit("MEGNO state was not validated after archive load")

megno_summary = json.loads(megno_summary_path.read_text())
if float(megno_summary.get("rebound_resume", {}).get("resumed_from_time_years") or 0.0) <= 0.0:
    raise SystemExit("MEGNO summary missing nonzero resume time")
final_megno = megno_summary.get("final_megno")
final_lcn = megno_summary.get("estimated_lyapunov_if_available")
if final_megno is None or not math.isfinite(float(final_megno)):
    raise SystemExit("final MEGNO is not finite")
if final_lcn is not None and not math.isfinite(float(final_lcn)):
    raise SystemExit("final LCN is present but not finite")

def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

megno_rows = rows(megno_csv)
megno_times = [float(row["time_years"]) for row in megno_rows]
if megno_times != sorted(megno_times) or len(megno_times) != len(set(megno_times)):
    raise SystemExit("MEGNO CSV times are not strictly monotonic unique")

invariant_times = [float(row["time_years"]) for row in rows(invariants_csv)]
if invariant_times != sorted(invariant_times) or len(invariant_times) != len(set(invariant_times)):
    raise SystemExit("invariants CSV times are not strictly monotonic unique")

for path, allowed in (
    (stability_csv, {"sun", "mercury barycenter", "venus barycenter", "earth barycenter", "mars barycenter"}),
    (elements_csv, {"mercury barycenter", "venus barycenter", "earth barycenter", "mars barycenter"}),
):
    seen = set()
    bodies = set()
    for row in rows(path):
        body = row.get("body", "")
        bodies.add(body)
        key = (row.get("time_years"), body)
        if key in seen:
            raise SystemExit(f"duplicate time/body row in {path}: {key}")
        seen.add(key)
    unexpected = bodies - allowed
    if unexpected:
        raise SystemExit(f"normal outputs include unexpected/variational bodies in {path}: {sorted(unexpected)}")

print("[smoke] resume loaded from %.6g years" % float(resume["resumed_from_time_years"]))
print("[smoke] MEGNO rows:", len(megno_rows))
print("[smoke] continuity checks passed")
PY
