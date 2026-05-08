#!/usr/bin/env bash
set -euo pipefail

ANNEAL_MAXITER="${ANNEAL_MAXITER:-200}"
ANNEAL_INITIAL_TEMP="${ANNEAL_INITIAL_TEMP:-5230.0}"
ANNEAL_SEED="${ANNEAL_SEED:-20260508}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/peacelovephysics/ephemeris/output/lunar_4param_dual_annealing_full_book_trials.csv}"
PROFILE_NAME="${PROFILE_NAME:-american_ephemeris_2000_2050_full_book_empirical_4param_dual_annealing}"

python -m mini_ephemeris.fit_lunar_velocity_3d_and_accel \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output "${OUTPUT_PATH}" \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --method dual-annealing \
  --objective lon_rms_plus_peak_plus_lat_rms \
  --lat-weight 0.5 \
  --target-lon-peak-arcsec 0.5 \
  --anneal-maxiter "${ANNEAL_MAXITER}" \
  --anneal-initial-temp "${ANNEAL_INITIAL_TEMP}" \
  --anneal-seed "${ANNEAL_SEED}" \
  --anneal-local-powell \
  --initial-dv-r-mm-s -0.00367349 \
  --initial-dv-t-mm-s 0.03925477 \
  --initial-dv-h-mm-s -0.01243427 \
  --initial-at-1e-15 4.89216 \
  --dv-r-min-mm-s -0.0052 \
  --dv-r-max-mm-s -0.0024 \
  --dv-t-min-mm-s 0.039245 \
  --dv-t-max-mm-s 0.039285 \
  --dv-h-min-mm-s -0.028 \
  --dv-h-max-mm-s -0.006 \
  --at-min-1e-15 4.86 \
  --at-max-1e-15 4.94 \
  --opt-maxiter 40 \
  --opt-xtol 1e-5 \
  --opt-ftol 1e-5 \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0 \
  --save-calibration-file /home/peacelovephysics/ephemeris/calibrations/lunar_calibrations.json \
  --save-calibration-name "${PROFILE_NAME}" \
  --save-calibration-description "Full-book four-parameter empirical lunar calibration using dual annealing with Powell local search."
