#!/usr/bin/env bash
set -euo pipefail

cd /home/peacelovephysics/ephemeris
source .venv/bin/activate

ROOT=/home/peacelovephysics/ephemeris/output/stability
KERNEL=/home/peacelovephysics/ephemeris/data/de431_part-2.bsp
QUEUE_LOG="$ROOT/run_full_with_pluto_500myr_megno_queue_$(date -u +%Y%m%dT%H%M%SZ).log"

run_seed () {
  local SEED="$1"
  local BASE="$ROOT/rebound_full_with_pluto_newtonian_500myr_megno_seed${SEED}"
  local TAG="full_with_pluto_newtonian_500myr_megno_1d_seed_${SEED}"
  local ARCHIVE="$BASE/${TAG}.bin"
  local RUN_LOG="$BASE/run_${TAG}_$(date -u +%Y%m%dT%H%M%SZ).log"

  mkdir -p "$BASE"

  echo
  echo "============================================================"
  echo "[queue] Starting seed ${SEED}"
  echo "[queue] BASE=$BASE"
  echo "[queue] TAG=$TAG"
  echo "[queue] ARCHIVE=$ARCHIVE"
  echo "============================================================"
  echo

  RESUME_ARGS=()
  if [[ -f "$ARCHIVE" ]]; then
    echo "[queue] Existing SimulationArchive found; will resume latest snapshot."
    RESUME_ARGS=(--rebound-resume latest)
  else
    echo "[queue] No SimulationArchive found; starting fresh."
  fi

  PYTHONUNBUFFERED=1 python -m mini_ephemeris.long_term_stability_cli \
    --kernel-path "$KERNEL" \
    --start-date 2000-01-01 \
    --duration-years 500000000 \
    --step-days 1 \
    --record-every-years 10000 \
    --model-scope full_with_pluto \
    --backend rebound \
    --rebound-integrator whfast \
    --rebound-gr-model none \
    --gr-model none \
    --with-megno \
    --rebound-chaos-method megno \
    --megno-seed "$SEED" \
    --megno-record-every-years 10000 \
    --rebound-simulationarchive "$ARCHIVE" \
    --rebound-archive-interval-years 1000000 \
    "${RESUME_ARGS[@]}" \
    --output-dir "$BASE" \
    --tag "$TAG" \
    --no-progress-bar \
    2>&1 | tee "$RUN_LOG"

  echo
  echo "[queue] Finished seed ${SEED}; regenerating reports..."
  echo

  python -m mini_ephemeris.compare_megno_shadow_results \
    --output-dir "$ROOT" \
    --tag megno_lcn_fallback

  python -m mini_ephemeris.stability_research_master_report \
    --output-dir "$ROOT"

  local SNAP="$ROOT/report_snapshots_after_full_with_pluto_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$SNAP"

  cp "$ROOT/stability_research_master_report.md" "$SNAP/"
  cp "$ROOT/stability_research_master_summary.json" "$SNAP/"
  cp "$ROOT/megno_shadow_comparison_megno_lcn_fallback.md" "$SNAP/" || true
  cp "$ROOT/megno_shadow_comparison_megno_lcn_fallback.json" "$SNAP/" || true

  cat > "$SNAP/seed_${SEED}_run_manifest.txt" <<EOF
seed=${SEED}
model_scope=full_with_pluto
duration_years=500000000
step_days=1
backend=rebound
integrator=whfast
gr_model=none
archive=${ARCHIVE}
base=${BASE}
tag=${TAG}
run_log=${RUN_LOG}
snapshot_dir=${SNAP}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

  echo "[queue] Snapshot saved to $SNAP"
}

{
  echo "[queue] Queue started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_seed 12345
  run_seed 67890
  echo "[queue] Queue complete at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$QUEUE_LOG"
