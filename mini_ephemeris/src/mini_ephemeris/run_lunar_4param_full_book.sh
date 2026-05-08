#!/usr/bin/env bash
set -euo pipefail

python -m mini_ephemeris.fit_lunar_velocity_3d_and_accel \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output /home/peacelovephysics/ephemeris/output/lunar_4param_full_book_trials.csv \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --method powell \
  --objective lon_rms_plus_peak_plus_lat_rms \
  --lat-weight 0.5 \
  --target-lon-peak-arcsec 0.5 \
  --initial-dv-r-mm-s 0.0 \
  --initial-dv-t-mm-s 0.039220792423 \
  --initial-dv-h-mm-s 0.0 \
  --initial-at-1e-15 4.744123111671 \
  --dv-r-min-mm-s -0.05 \
  --dv-r-max-mm-s 0.05 \
  --dv-t-min-mm-s 0.038 \
  --dv-t-max-mm-s 0.041 \
  --dv-h-min-mm-s -0.05 \
  --dv-h-max-mm-s 0.05 \
  --at-min-1e-15 4.0 \
  --at-max-1e-15 5.5 \
  --opt-maxiter 40 \
  --opt-xtol 1e-5 \
  --opt-ftol 1e-5 \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0 \
  --save-calibration-file /home/peacelovephysics/ephemeris/calibrations/lunar_calibrations.json \
  --save-calibration-name american_ephemeris_2000_2050_full_book_empirical_4param \
  --save-calibration-description "Full-book four-parameter empirical lunar calibration."
