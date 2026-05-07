#!/usr/bin/env bash
set -euo pipefail

python -m mini_ephemeris.american_ephemeris_range_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output /home/peacelovephysics/ephemeris/output/model_vs_jpl_moon_profile_v2_full_book.csv \
  --moon-lon-plot /home/peacelovephysics/ephemeris/output/moon_longitude_residual_profile_v2_full_book.png \
  --moon-lat-plot /home/peacelovephysics/ephemeris/output/moon_latitude_residual_profile_v2_full_book.png \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --lunar-calibration-file calibrations/lunar_calibrations.json \
  --lunar-calibration-profile american_ephemeris_2000_2050_full_book_empirical_v2 \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0