#!/usr/bin/env bash
set -euo pipefail

ANNEAL_MAXITER="${ANNEAL_MAXITER:-600}"
ANNEAL_INITIAL_TEMP="${ANNEAL_INITIAL_TEMP:-5230.0}"
ANNEAL_SEED="${ANNEAL_SEED:-20260508}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/peacelovephysics/ephemeris/output/lunar_6param_v4_dual_annealing_full_book_trials.csv}"
PROFILE_NAME="${PROFILE_NAME:-american_ephemeris_2000_2050_full_book_6param_empirical_v4_dual_annealing}"

PYTHONUNBUFFERED=1 python -m mini_ephemeris.fit_lunar_velocity_3d_and_accel \
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
  --objective lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms \
  --lat-weight 0.5 \
  --target-lon-peak-arcsec 0.5 \
  --anneal-maxiter "${ANNEAL_MAXITER}" \
  --anneal-initial-temp "${ANNEAL_INITIAL_TEMP}" \
  --anneal-seed "${ANNEAL_SEED}" \
  --anneal-local-powell \
  --initial-dv-r-mm-s -0.0054546464726399554 \
  --initial-dv-t-mm-s 0.03924474618983251 \
  --initial-dv-h-mm-s -0.032555217119768595 \
  --initial-ar-1e-15 0.25789180988968186 \
  --initial-at-1e-15 4.9900490973754525 \
  --initial-ah-1e-15 -0.9731661094281016 \
  --dv-r-min-mm-s -0.0062 \
  --dv-r-max-mm-s -0.0050 \
  --dv-t-min-mm-s 0.039238 \
  --dv-t-max-mm-s 0.039253 \
  --dv-h-min-mm-s -0.0420 \
  --dv-h-max-mm-s -0.0260 \
  --ar-min-1e-15 0.10 \
  --ar-max-1e-15 0.50 \
  --at-min-1e-15 4.94 \
  --at-max-1e-15 5.12 \
  --ah-min-1e-15 -1.08 \
  --ah-max-1e-15 -0.88 \
  --opt-maxiter 60 \
  --opt-xtol 1e-5 \
  --opt-ftol 1e-5 \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0 \
  --save-calibration-file /home/peacelovephysics/ephemeris/calibrations/lunar_calibrations.json \
  --save-calibration-name "${PROFILE_NAME}" \
  --save-calibration-description "Full-book six-parameter v4 empirical lunar calibration using trend-aware dual annealing with Powell local search."
