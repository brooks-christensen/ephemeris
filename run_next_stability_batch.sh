#!/usr/bin/env bash
set -uo pipefail

cd /home/peacelovephysics/ephemeris || exit 1
source .venv/bin/activate

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="output/stability/batches/batch_${STAMP}"
mkdir -p "$BATCH_DIR"

exec > >(tee -a "$BATCH_DIR/batch_driver.log") 2>&1

echo "[Batch] started: $(date)"
echo "[Batch] output: $BATCH_DIR"
echo "job,status,start,end,seconds,log" > "$BATCH_DIR/batch_status.csv"

run_job () {
  local name="$1"
  shift
  local log="$BATCH_DIR/${name}.log"
  local start_epoch end_epoch status

  start_epoch="$(date +%s)"
  echo
  echo "================================================================"
  echo "[Batch] START $name at $(date)"
  echo "[Batch] CMD: $*"
  echo "================================================================"

  PYTHONUNBUFFERED=1 "$@" > "$log" 2>&1
  status=$?

  end_epoch="$(date +%s)"
  echo "[Batch] END $name at $(date), status=$status, seconds=$((end_epoch-start_epoch))"
  echo "${name},${status},${start_epoch},${end_epoch},$((end_epoch-start_epoch)),${log}" >> "$BATCH_DIR/batch_status.csv"

  if [[ "$status" -ne 0 ]]; then
    echo "[Batch] WARNING: $name failed. See $log"
  fi
}

run_shell_job () {
  local name="$1"
  shift
  local log="$BATCH_DIR/${name}.log"
  local start_epoch end_epoch status

  start_epoch="$(date +%s)"
  echo
  echo "================================================================"
  echo "[Batch] START $name at $(date)"
  echo "[Batch] CMD: $*"
  echo "================================================================"

  PYTHONUNBUFFERED=1 bash "$@" > "$log" 2>&1
  status=$?

  end_epoch="$(date +%s)"
  echo "[Batch] END $name at $(date), status=$status, seconds=$((end_epoch-start_epoch))"
  echo "${name},${status},${start_epoch},${end_epoch},$((end_epoch-start_epoch)),${log}" >> "$BATCH_DIR/batch_status.csv"

  if [[ "$status" -ne 0 ]]; then
    echo "[Batch] WARNING: $name failed. See $log"
  fi
}

run_job check_optional_backends \
  python -m mini_ephemeris.check_optional_backends

run_shell_job checkpoint_restart_stress \
  scripts/run_checkpoint_restart_stress_test.sh

run_shell_job rebound_two_body_validation \
  scripts/run_rebound_two_body_validation.sh

run_shell_job mercury_gr_precession_validation \
  scripts/validate_mercury_gr_precession.sh

run_job inner_30kyr_checkpointed_newtonian \
  python -m mini_ephemeris.long_term_stability_cli \
    --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
    --start-date 2000-01-01 \
    --duration-years 30000 \
    --step-days 0.25 \
    --record-every-years 100 \
    --gr-model none \
    --integrator leapfrog \
    --model-scope inner \
    --output-dir /home/peacelovephysics/ephemeris/output/stability/inner_30kyr_checkpointed \
    --tag inner_30kyr_checkpointed_newtonian \
    --with-lyapunov \
    --lyapunov-method tangent \
    --lyapunov-body all \
    --lyapunov-perturbation-m 1000 \
    --lyapunov-renorm-years 0.25 \
    --lyapunov-fit-start-years 1000 \
    --lyapunov-fit-end-years 30000 \
    --checkpoint-every-years 100 \
    --checkpoint-dir /home/peacelovephysics/ephemeris/output/stability/inner_30kyr_checkpointed/checkpoints \
    --keep-checkpoints 3 \
    --no-progress-bar

echo
echo "[Batch] finished: $(date)"
echo "[Batch] status file: $BATCH_DIR/batch_status.csv"
echo "[Batch] upload this whole folder if you want me to review it:"
echo "$BATCH_DIR"
