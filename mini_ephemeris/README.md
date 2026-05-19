# mini_ephemeris

`mini_ephemeris` is a Python celestial-mechanics and ephemeris experimentation package. The current near-term goal is to reproduce the **apparent geocentric tropical zodiac longitudes** used by *The American Ephemeris for the 21st Century 2000-2050 at Midnight* by comparing a reduced N-body model against JPL/Skyfield reference positions.

The project currently has two distinct modes of use:

1. **Short-range ephemeris matching** over 2000-2050, using explicit Earth and Moon bodies and optional empirical lunar calibration.
2. **Long-term stability studies**, using a cleaner physical reduced model with the Earth-Moon barycenter rather than a fitted explicit Moon model.

These two goals should stay separate. The empirical lunar calibration is useful for reproducing a JPL/book-style ephemeris over a defined date range. It should not be treated as general-purpose lunar physics for million-year integrations.

---

## Project status

The current model is already extremely accurate for the Sun and planets over the 2000-2050 book range. The Moon was the difficult case. The raw model had a dominant lunar longitude drift, but empirical calibration has reduced the Moon longitude residual to sub-arcsecond scale over the full printed book range.

Current successful full-book lunar calibration profile:

```text
Profile name:
  american_ephemeris_2000_2050_full_book_4param_empirical_v3_peak

Fit / validation range:
  2000-01-01 to 2050-12-31

Model:
  Newtonian N-body
  + explicit Earth and Moon
  + Sun-centered 1PN GR
  + Earth J2
  + empirical lunar radial/tangential/out-of-plane initial velocity correction
  + empirical lunar along-track acceleration

Fitted lunar parameters:
  moon_dv_r_mm_s        = -0.003673491156
  moon_dv_t_mm_s        =  0.039254769119
  moon_dv_h_mm_s        = -0.012434267591
  moon_a_t_1e_15_m_s2   =  4.892158075549

Validated full-book residuals:
  Moon longitude RMS    ~= 0.239 arcsec
  Moon longitude peak   ~= 0.497 arcsec
  Moon latitude RMS     ~= 0.332 arcsec
  Moon distance RMS     ~= 0.061 km
```

The remaining Moon longitude residual still contains smooth curvature, while the Moon latitude residual shows a growing periodic envelope. The v4 exploratory path adds radial and out-of-plane empirical acceleration parameters to test whether those trends can be reduced without changing the force model, residual calculation, integrator, or time sampling.

---

## Local environment

Typical project root:

```bash
/home/peacelovephysics/ephemeris/mini_ephemeris
```

Typical JPL kernel path:

```bash
/home/peacelovephysics/ephemeris/data/de431_part-2.bsp
```

Typical output directory:

```bash
/home/peacelovephysics/ephemeris/output
```

Run commands from the package root with the virtual environment activated:

```bash
cd /home/peacelovephysics/ephemeris/mini_ephemeris
```

---

## Package layout

The package currently centers on these modules:

```text
src/mini_ephemeris/
  __init__.py
  nbody.py
  advanced_integrators.py
  ephem.py
  plotting.py
  error_metrics.py
  analysis_tools.py
  orbital_elements.py
  stability_diagnostics.py
  chaos_diagnostics.py
  experiments.py
  long_term_stability_cli.py

  american_ephemeris.py
  american_ephemeris_cli.py
  american_ephemeris_model_cli.py
  american_ephemeris_range_cli.py

  analyze_moon_residual.py
  fit_lunar_initial_velocity.py
  fit_lunar_dv_and_tangential_accel.py
  fit_lunar_velocity_3d_and_accel.py
  lunar_calibration.py
```

High-level purpose of the major files:

| File | Purpose |
|---|---|
| `nbody.py` | Core N-body state containers and basic mechanics utilities. |
| `advanced_integrators.py` | DOP853 integration and acceleration models, including Newtonian gravity, Sun 1PN GR, Earth J2, and empirical lunar basis-acceleration wrapping. |
| `ephem.py` | JPL/Skyfield kernel loading, initial barycentric state construction, body list construction, Earth/Moon setup, and lunar initial velocity correction helpers. |
| `orbital_elements.py` | Heliocentric osculating orbital elements for long-term stability diagnostics. |
| `stability_diagnostics.py` | Energy, angular momentum, center-of-mass, and pairwise separation diagnostics for stability runs. |
| `chaos_diagnostics.py` | Small phase-space separation and finite-time Lyapunov helper functions for future ensemble work. |
| `long_term_stability_cli.py` | Physical reduced-model stability CLI with streamed CSV diagnostics. |
| `american_ephemeris.py` | Converts JPL/model vectors into American-Ephemeris-style apparent/geocentric/tropical longitude comparison rows. |
| `american_ephemeris_cli.py` | Generates JPL-only American-Ephemeris-style CSVs. |
| `american_ephemeris_model_cli.py` | Compares one target month of integrated model output against JPL/book-style positions. |
| `american_ephemeris_range_cli.py` | Compares a full date range, writes CSVs, prints residual summaries, and saves Moon residual plots. |
| `analyze_moon_residual.py` | Post-processes model-vs-JPL CSVs and fits linear/quadratic lunar residual trends. |
| `fit_lunar_initial_velocity.py` | Fits a one-parameter lunar initial tangential velocity correction. |
| `fit_lunar_dv_and_tangential_accel.py` | Fits a two-parameter empirical lunar correction: initial tangential velocity plus along-track acceleration. |
| `fit_lunar_velocity_3d_and_accel.py` | Fits empirical lunar radial/tangential/out-of-plane velocity corrections plus optional radial/tangential/out-of-plane accelerations. Defaults keep the older four-parameter behavior. |
| `lunar_calibration.py` | Loads, resolves, and saves named empirical lunar calibration profiles. |

---

## Installed / runtime dependencies

The project uses:

- Python
- NumPy
- SciPy
- Matplotlib
- Skyfield
- tqdm, if progress bars are enabled in the integrator
- JPL BSP kernels, currently `de431_part-2.bsp`

Exact dependency management may depend on the local virtual environment. A future cleanup step should add or update `pyproject.toml` / `requirements.txt` if those are not already complete.

---

## Core model options

### Integrator

The preferred current integrator is DOP853 with chunked propagation.

Typical numerical settings:

```bash
--chunk-years 1
--max-step-days 1.0
--rtol 1e-12
--atol 1e-15
```

### Relativity

Available GR model option:

```bash
--gr-model none
--gr-model sun
```

Current preferred ephemeris-matching setting:

```bash
--gr-model sun
```

This applies the Sun-centered 1PN correction used in the current reduced model. Earlier experimentation with a more complex EIH-style model was not favored because it was slower and did not improve results.

### Earth J2

Earth J2 can be enabled with:

```bash
--earth-j2
```

Current preferred ephemeris-matching setting:

```bash
--earth-j2
```

This significantly improved lunar behavior.

### Pluto

Some commands support:

```bash
--include-pluto
```

Pluto is mainly for output completeness rather than fixing lunar dynamics.

### Progress bar

Some commands support:

```bash
--no-progress-bar
```

Use this for cleaner logs or scripted runs.

---

## American Ephemeris convention

The package's JPL/Skyfield output convention was validated against printed pages from *The American Ephemeris for the 21st Century 2000-2050 at Midnight*.

The target convention is:

```text
Time convention:
  TT / ET-style midnight samples

Position convention:
  apparent geocentric positions

Coordinate convention:
  tropical zodiac longitude
  ecliptic of date

Book precision:
  Sun and Moon: nearest arcsecond
  planets: nearest tenth arcminute = 6 arcsec
```

A JPL-only chart generator was checked against the printed January 2000 page. The generated values matched the book essentially exactly, with only a tiny Pluto rounding discrepancy of 0.1 arcminute on one checked date.

---

## JPL-only American Ephemeris chart generation

Generate a JPL/Skyfield American-Ephemeris-style chart for one month:

```bash
python -m mini_ephemeris.american_ephemeris_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --year 2000 \
  --month 1 \
  --output /home/peacelovephysics/ephemeris/output/jpl_american_ephemeris_2000_01.csv
```

Use this when you want a machine-readable JPL/book-style reference chart.

---

## One-month model-vs-JPL comparison

Compare the integrated model against JPL for a target month:

```bash
python -m mini_ephemeris.american_ephemeris_model_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --year 2050 \
  --month 1 \
  --output /home/peacelovephysics/ephemeris/output/model_vs_jpl_american_ephemeris_2050_01.csv \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15
```

Useful options include:

```bash
--moon-dv-t-mm-s <value>
--no-preserve-emb-momentum
```

The current calibration workflow has mainly focused on the full range CLI.

---

## Full-range model-vs-JPL comparison

Run the full printed book range with the current successful named calibration profile:

```bash
python -m mini_ephemeris.american_ephemeris_range_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output /home/peacelovephysics/ephemeris/output/model_vs_jpl_moon_profile_v3_full_book.csv \
  --moon-lon-plot /home/peacelovephysics/ephemeris/output/moon_longitude_residual_profile_v3_full_book.png \
  --moon-lat-plot /home/peacelovephysics/ephemeris/output/moon_latitude_residual_profile_v3_full_book.png \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --lunar-calibration-file calibrations/lunar_calibrations.json \
  --lunar-calibration-profile american_ephemeris_2000_2050_full_book_4param_empirical_v3_peak \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0
```

Important options:

| Option | Meaning |
|---|---|
| `--kernel-path` | Path to the JPL BSP kernel. |
| `--start-date` | Start date, ISO format. |
| `--end-date` | End date, ISO format. |
| `--output` | Output CSV path. |
| `--moon-lon-plot` | Output path for Moon longitude residual plot. |
| `--moon-lat-plot` | Optional output path for Moon latitude residual plot. |
| `--gr-model none/sun` | Select no GR or Sun-centered 1PN GR. |
| `--earth-j2` | Enable Earth J2. |
| `--include-pluto` | Include Pluto barycenter if supported. |
| `--chunk-years` | Chunk length for long DOP853 propagation. |
| `--max-step-days` | Maximum DOP853 step size in days. |
| `--rtol`, `--atol` | DOP853 tolerances. |
| `--moon-dv-r-mm-s`, `--moon-dv-t-mm-s`, `--moon-dv-h-mm-s` | Manual initial lunar radial/tangential/out-of-plane velocity corrections in mm/s. Override profile values if a profile is used. |
| `--moon-a-r-1e-15-m-s2`, `--moon-a-t-1e-15-m-s2`, `--moon-a-h-1e-15-m-s2` | Manual empirical lunar radial/tangential/out-of-plane accelerations in units of `1e-15 m/s^2`. Override profile values if a profile is used. |
| `--lunar-calibration-file` | JSON file containing named lunar calibration profiles. |
| `--lunar-calibration-profile` | Named calibration profile to load. |
| `--moon-lon-ylim-arcsec` | Manual absolute y-limit for Moon longitude residual plot. |
| `--moon-lat-ylim-arcsec` | Manual absolute y-limit for Moon latitude residual plot. |
| `--no-preserve-emb-momentum` | Apply the initial lunar velocity correction to the Moon only instead of preserving Earth-Moon barycenter momentum. |
| `--no-progress-bar` | Disable progress bar output. |

---

## Lunar calibration profiles

Calibration profiles are stored in JSON, for example:

```text
calibrations/lunar_calibrations.json
```

Current successful four-parameter profile:

```json
{
  "profiles": {
    "american_ephemeris_2000_2050_full_book_4param_empirical_v3_peak": {
      "description": "Full-book 2000-01-01 to 2050-12-31 four-parameter empirical lunar calibration. Best observed longitude peak from focused dual-annealing/Powell search.",
      "fit_start_date": "2000-01-01",
      "fit_end_date": "2050-12-31",
      "validation_start_date": "2000-01-01",
      "validation_end_date": "2050-12-31",
      "objective": "lon_rms_plus_peak_plus_lat_rms",
      "lat_weight": 0.5,
      "lat_peak_weight": 0.0,
      "model_notes": "Newtonian N-body + Sun 1PN GR + Earth J2 + empirical lunar radial/tangential/out-of-plane initial velocity correction + empirical lunar along-track acceleration. Do not use this calibration for long-term stability studies.",
      "moon_dv_r_mm_s": -0.003673491156,
      "moon_dv_t_mm_s": 0.039254769119,
      "moon_dv_h_mm_s": -0.012434267591,
      "moon_a_r_1e_15_m_s2": 0.0,
      "moon_a_t_1e_15_m_s2": 4.892158075549,
      "moon_a_h_1e_15_m_s2": 0.0,
      "lon_rms_arcsec": 0.239273,
      "lon_peak_abs_arcsec": 0.496638,
      "lat_rms_arcsec": 0.332383,
      "dist_rms_km": 0.060680
    }
  }
}
```

The profile loader resolves values using this rule:

```text
No profile:
  missing lunar correction values default to 0.

Profile selected:
  profile values are used.

Profile + explicit CLI values:
  explicit CLI values override profile values.
```

This makes it easy to run a known calibration while still testing small variations.

---

## Analyzing Moon residual trends

After a full-range CSV is generated, run:

```bash
python -m mini_ephemeris.analyze_moon_residual \
  --csv-path /home/peacelovephysics/ephemeris/output/model_vs_jpl_moon_profile_v3_full_book.csv \
  --output-dir /home/peacelovephysics/ephemeris/output \
  --tag moon_profile_v3_full_book
```

This prints:

```text
Raw mean
Raw RMS
Raw peak absolute error
Linear fit
Linear detrended RMS / peak
Quadratic fit
Quadratic detrended RMS / peak
Latitude RMS
Distance RMS
```

It also saves plots showing the raw Moon longitude residual, the linear detrended residual, and the quadratic detrended residual.

This script is diagnostic only. It does not rerun the integrator and does not change the model.

---

## One-parameter lunar initial velocity fitting

The first successful lunar calibration experiment fit a tiny initial geocentric tangential velocity correction for the Moon.

Example command:

```bash
python -m mini_ephemeris.fit_lunar_initial_velocity \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-01-31 \
  --output /home/peacelovephysics/ephemeris/output/lunar_tangent_fit_2000_2050.csv \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --dv-min-mm-s 0.020 \
  --dv-max-mm-s 0.070 \
  --dv-count 11
```

Later optimizer-enhanced usage may include:

```bash
--optimize
--objective rms
--objective peak
--objective slope_abs
--objective rms_plus_peak
--opt-xatol-mm-s 1e-5
--opt-maxiter 20
```

Interpretation:

- Positive `dv_t` is prograde.
- The correction is tiny: about `0.04 mm/s`.
- It is interpreted as a reduced-model initial-condition calibration.
- By default, it preserves Earth-Moon barycenter momentum by splitting the correction between Earth and Moon.

---

## Two-parameter lunar fitting

The older v2 calibration used two parameters:

1. Initial lunar tangential velocity correction.
2. Empirical geocentric lunar along-track acceleration.

Example full-book fit:

```bash
python -m mini_ephemeris.fit_lunar_dv_and_tangential_accel \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output /home/peacelovephysics/ephemeris/output/lunar_dv_plus_at_peak_full_book_v2.csv \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --objective peak \
  --initial-dv-mm-s 0.039211803399 \
  --initial-at-1e-15 4.697879236139 \
  --dv-min-mm-s 0.03918 \
  --dv-max-mm-s 0.03924 \
  --at-min-1e-15 4.67 \
  --at-max-1e-15 4.78 \
  --opt-maxiter 20 \
  --opt-xtol 1e-5 \
  --opt-ftol 1e-5
```

Supported objectives:

| Objective | Meaning |
|---|---|
| `rms` | Minimize Moon longitude RMS. |
| `peak` | Minimize maximum absolute Moon longitude residual. |
| `slope_abs` | Minimize absolute linear residual slope. |
| `rms_plus_peak` | Minimize longitude RMS plus peak absolute residual. Often a stable compromise. |

Recommended strategy:

1. Use `rms_plus_peak` for broad searches because it is smoother than pure peak.
2. Use `peak` only for a final tight local refinement.
3. Validate the final pair through `american_ephemeris_range_cli.py`.
4. Save a named calibration profile instead of relying on command history.

Some versions of the fitter also support saving directly:

```bash
--save-calibration-file calibrations/lunar_calibrations.json
--save-calibration-name american_ephemeris_2000_2050_full_book_empirical_v2
--save-calibration-description "Full-book 2000-01-01 to 2050-12-31 peak-optimized empirical lunar calibration."
```

---

## Current best results summary

The current successful full-book profile is:

```text
american_ephemeris_2000_2050_full_book_4param_empirical_v3_peak
```

Validated command:

```bash
python -m mini_ephemeris.american_ephemeris_range_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --end-date 2050-12-31 \
  --output /home/peacelovephysics/ephemeris/output/model_vs_jpl_moon_profile_v3_full_book.csv \
  --moon-lon-plot /home/peacelovephysics/ephemeris/output/moon_longitude_residual_profile_v3_full_book.png \
  --moon-lat-plot /home/peacelovephysics/ephemeris/output/moon_latitude_residual_profile_v3_full_book.png \
  --gr-model sun \
  --earth-j2 \
  --chunk-years 1 \
  --max-step-days 1.0 \
  --rtol 1e-12 \
  --atol 1e-15 \
  --lunar-calibration-file calibrations/lunar_calibrations.json \
  --lunar-calibration-profile american_ephemeris_2000_2050_full_book_4param_empirical_v3_peak \
  --moon-lon-ylim-arcsec 1.0 \
  --moon-lat-ylim-arcsec 1.0
```

Validated output:

```text
Moon longitude:
  RMS       ~=  0.239 arcsec
  peak abs  ~=  0.497 arcsec

Moon latitude:
  RMS       ~= 0.332 arcsec

Moon distance:
  RMS       ~= 0.061 km
```

The v3 residuals still show a longitude quadratic trend and a growing
latitude envelope. The v4 six-parameter run below is intended to explore
those remaining structures.

This is good enough for the current printed American Ephemeris matching milestone, and it is the seed for the exploratory v4 run.

---

## Plotting residuals

The range CLI supports plot zoom options:

```bash
--moon-lon-ylim-arcsec 1.0
--moon-lat-ylim-arcsec 1.0
```

Use this now that the Moon residual is sub-arcsecond. The old +/-60 arcsec plotting scale was useful for raw baseline runs but is too wide for calibrated runs.

If no y-limit is supplied, the plotter should choose an automatic scale based on the peak residual, with a minimum of +/-1 arcsec.

---

## Development checks

After code cleanup, run:

```bash
python -m compileall src/mini_ephemeris
```

Also smoke-test CLIs that use `argparse`, because duplicate argument definitions are not always caught by `compileall`:

```bash
python -m mini_ephemeris.american_ephemeris_range_cli --help
python -m mini_ephemeris.fit_lunar_dv_and_tangential_accel --help
python -m mini_ephemeris.fit_lunar_velocity_3d_and_accel --help
python -m mini_ephemeris.long_term_stability_cli --help
```

Small long-term stability smoke test:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 0.02 \
  --step-days 4 \
  --record-every-years 0.01 \
  --output-dir /home/peacelovephysics/ephemeris/output \
  --tag smoke_stability \
  --no-progress-bar
```

Small Lyapunov plumbing smoke test:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 0.05 \
  --step-days 2 \
  --record-every-years 0.025 \
  --with-lyapunov \
  --lyapunov-body mercury \
  --lyapunov-renorm-years 0.01 \
  --lyapunov-fit-start-years 0.01 \
  --lyapunov-fit-end-years 0.05 \
  --output-dir /home/peacelovephysics/ephemeris/output \
  --tag smoke_lyapunov \
  --no-progress-bar
```

Lyapunov validation matrix:

```bash
bash scripts/debug_lyapunov_validation.sh
```

Two-body timestep ladder:

```bash
bash scripts/debug_two_body_timestep_ladder.sh
```

Two-body tangent Lyapunov ladder:

```bash
bash scripts/debug_two_body_tangent_lyapunov_ladder.sh
```

Recommended Git checkpoint:

```bash
git status
git diff -- src/mini_ephemeris calibrations/lunar_calibrations.json

git add src/mini_ephemeris calibrations/lunar_calibrations.json
git commit -m "Add six-parameter lunar calibration exploration"
git tag lunar-calibration-v4-exploration
```

---

## Four-/six-parameter lunar velocity + acceleration fitting

The current multi-parameter optimizer fits the empirical lunar calibration in the instantaneous geocentric orbital basis. By default, radial and out-of-plane acceleration bounds are fixed at zero, which preserves the older four-parameter behavior. Opening those bounds enables the v4 six-parameter exploration.

```text
dv_r_mm_s:
  initial radial velocity correction, positive Earth -> Moon

dv_t_mm_s:
  initial tangential velocity correction, positive prograde

dv_h_mm_s:
  initial out-of-plane velocity correction, positive along lunar angular momentum

a_t_1e_15_m_s2:
  empirical geocentric along-track acceleration, in 1e-15 m/s^2

a_r_1e_15_m_s2:
  optional empirical geocentric radial acceleration, in 1e-15 m/s^2

a_h_1e_15_m_s2:
  optional empirical geocentric out-of-plane acceleration, in 1e-15 m/s^2
```

The initial velocity basis is:

```text
r_geo = r_moon - r_earth
v_geo = v_moon - v_earth

r_hat = r_geo / |r_geo|
h_hat = normalize(cross(r_geo, v_geo))
t_hat = normalize(cross(h_hat, r_hat))
```

The correction applies:

```text
dv_rel = dv_r * r_hat + dv_t * t_hat + dv_h * h_hat
a_rel  = a_r  * r_hat + a_t  * t_hat + a_h  * h_hat
```

To preserve Earth-Moon barycenter momentum:

```text
v_moon  += m_earth / (m_earth + m_moon) * dv_rel
v_earth -= m_moon  / (m_earth + m_moon) * dv_rel
```

Example four-parameter full-book run seeded from the v2 profile:

```bash
bash src/mini_ephemeris/run_lunar_4param_full_book.sh
```

Equivalent expanded command:

```bash
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
  --save-calibration-file calibrations/lunar_calibrations.json \
  --save-calibration-name american_ephemeris_2000_2050_full_book_empirical_4param \
  --save-calibration-description "Full-book four-parameter empirical lunar calibration."
```

Default sidecar outputs are derived from `--output`:

```text
lunar_4param_full_book_trials_summary.json
lunar_4param_full_book_trials_best_moon_residuals.csv
lunar_4param_full_book_trials_best_moon_longitude_residual.png
lunar_4param_full_book_trials_best_moon_latitude_residual.png
```

The trial CSV named by `--output` is also an incremental trial journal. It is
reset at run start, then one row is appended after each completed objective
evaluation for Powell, Nelder-Mead, dual annealing, and grid scans. This makes
interrupted long annealing runs inspectable. Each row includes:

```text
trial number and status
trial start/end UTC timestamps
trial runtime in seconds
optimizer mode and method label
parameter values
objective name and weights
objective value
longitude RMS, peak, mean, max, min
latitude RMS, peak, mean, max, min
distance RMS and extrema
trend diagnostics
```

Exploratory v4 six-parameter dual-annealing run:

```bash
bash src/mini_ephemeris/run_lunar_6param_v4_dual_annealing_full_book.sh
```

The v4 script uses:

```text
method:
  dual-annealing with Powell local search enabled

objective:
  lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms
  = lon_peak
    + 0.5 * lon_rms
    + 0.25 * linear_detrended_peak
    + 0.25 * quadratic_detrended_peak
    + 0.5 * lat_rms

initial values:
  moon_dv_r_mm_s      = -0.00471691895885988
  moon_dv_t_mm_s      =  0.039254442719099995
  moon_dv_h_mm_s      = -0.022806504495004626
  moon_a_r_1e_15_m_s2 =  0.6112252505297777
  moon_a_t_1e_15_m_s2 =  4.956823663287823
  moon_a_h_1e_15_m_s2 = -0.9546894580125809
```

Long-running annealing settings can be overridden from the environment:

```bash
ANNEAL_MAXITER=1000 \
ANNEAL_INITIAL_TEMP=7000 \
ANNEAL_SEED=12345 \
bash src/mini_ephemeris/run_lunar_6param_v4_dual_annealing_full_book.sh
```

Supported objectives:

```text
lon_peak
lon_rms
lon_rms_plus_peak
lon_peak_plus_lat_rms
lon_rms_plus_peak_plus_lat_rms
lon_rms_plus_peak_plus_lat_rms_plus_lat_peak
lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms
slope_abs
```

Supported optimizer methods:

```text
powell          default, backward-compatible behavior
nelder-mead     uses a custom initial simplex scaled to the parameter bounds
dual-annealing  uses scipy.optimize.dual_annealing over the parameter bounds
```

Dual annealing options:

```text
--anneal-maxiter <int>
--anneal-initial-temp <float>
--anneal-seed <int>
--anneal-local-powell
```

Recommended search strategy:

1. Use `lon_rms_plus_peak_plus_lat_rms` with `--lat-weight 0.5` for four-parameter compatibility runs.
2. Use `lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms` with `--lat-weight 0.5` for the trend-aware v4 six-parameter exploration.
3. Try `dual-annealing` for broad exploration, optionally with `--anneal-local-powell`.
4. Use `powell` or `nelder-mead` for final local refinement.
5. Validate the saved profile with `american_ephemeris_range_cli`.
6. If the peak stalls above the target, inspect the best residual plot before adding new physics.

Useful initial bounds:

```text
v2/v3 broad bounds:
  dv_r_mm_s: [-0.05, 0.05]
  dv_t_mm_s: [0.038, 0.041]
  dv_h_mm_s: [-0.05, 0.05]
  a_t_1e_15_m_s2: [4.0, 5.5]

v4 exploratory bounds:
  dv_r_mm_s: [-0.0058, -0.0038]
  dv_t_mm_s: [0.039244, 0.039260]
  dv_h_mm_s: [-0.0340, -0.0170]
  a_r_1e_15_m_s2: [0.15, 0.70]
  a_t_1e_15_m_s2: [4.92, 5.00]
  a_h_1e_15_m_s2: [-1.08, -0.86]
```

These are engineering starting points, not physical constants.

---

# Future enhancement: physically motivated lunar refinements

The empirical acceleration is useful, but it is not a final physical lunar model. Future work could replace or supplement it with more physically interpretable terms:

- Earth J2 axis orientation instead of a fixed simplified axis.
- Earth J3/J4 or additional zonal harmonics.
- Lunar figure effects.
- Earth-Moon tidal dissipation / secular lunar acceleration.
- More careful Earth orientation / precession / nutation handling.
- Additional perturbers, depending on scope.

The recommended order is:

1. First quantify what empirical terms are needed.
2. Then replace empirical knobs with physical terms one at a time.
3. Verify each addition against residual shape, not only total RMS.

---

# Long-term stability experiments

Long-term stability studies should be treated as a separate mode from the short-range ephemeris-matching mode.

For million-year integrations, do **not** use the empirical lunar calibration profile. It is fitted to JPL/book agreement over 2000-2050 and is not a general physical law.

Implemented reduced model:

```text
Sun
Mercury barycenter
Venus barycenter
Earth-Moon barycenter
Mars barycenter
Jupiter barycenter
Saturn barycenter
Uranus barycenter
Neptune barycenter
optional Pluto barycenter
```

Why use the Earth-Moon barycenter?

- The explicit Moon introduces a 27-day timescale.
- That forces small timesteps and makes million-year integrations expensive.
- For broad planetary stability, the Earth-Moon system mostly acts through its barycenter.
- Long-term studies care more about secular architecture than daily geocentric lunar positions.

Recommended numerical direction:

- Use the fixed-step leapfrog / velocity-Verlet integrator for long-term Newtonian work.
- Use DOP853 only for short validation/comparison runs.
- Use fixed timesteps appropriate to the shortest modeled orbital period.
- Track energy error and angular momentum error.
- Run ensembles of nearby initial conditions.
- Compare qualitative behavior, not exact long-term longitudes.
- Include Sun 1PN GR for Mercury-sensitive experiments.
- Treat exact planetary phase after very long times cautiously because the solar system is chaotic.

Run the stability CLI:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 10000 \
  --step-days 4 \
  --record-every-years 10 \
  --gr-model none \
  --integrator leapfrog \
  --output-dir /home/peacelovephysics/ephemeris/output \
  --tag stability_10kyr_emb
```

Optional Pluto and Sun 1PN GR:

```bash
--include-pluto
--gr-model sun
```

If `--gr-model sun` is used with `--integrator leapfrog`, the CLI prints that
the Sun 1PN GR perturbation is included through the acceleration callback and
the method is no longer exactly symplectic.

The stability CLI writes streamed numerical outputs:

```text
stability_timeseries_<tag>.csv
orbital_elements_<tag>.csv
invariants_<tag>.csv
min_separations_<tag>.csv
summary_<tag>.json
```

Diagnostics include Newtonian energy drift, total angular momentum drift,
center-of-mass position/velocity drift, pairwise minimum separations, and
J2000-ecliptic heliocentric osculating elements for each non-Sun body.

### Lyapunov diagnostic

The stability CLI can also run a two-trajectory Benettin-style maximum
Lyapunov estimate for the reduced barycenter model:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 10000 \
  --step-days 4 \
  --record-every-years 10 \
  --with-lyapunov \
  --lyapunov-body mercury \
  --lyapunov-perturbation-m 1.0 \
  --lyapunov-renorm-years 10 \
  --lyapunov-fit-start-years 1000 \
  --lyapunov-fit-end-years 10000 \
  --output-dir /home/peacelovephysics/ephemeris/output \
  --tag stability_10kyr_lyapunov
```

Lyapunov-specific outputs:

```text
lyapunov_<tag>.csv
lyapunov_summary_<tag>.json
lyapunov_growth_<tag>.png
```

The CSV records:

```text
time_years
separation_norm
pre_renorm_separation_norm
post_renorm_separation_norm
target_norm
growth_factor
log_growth_increment
cumulative_log_growth
local_lambda_1_per_year
running_lambda_1_per_year
lyapunov_time_years
max_position_separation_m
max_velocity_separation_m_s
dominant_body_in_norm
dominant_component_type
renorm_interval_years_actual
cosine_with_previous_delta_direction
cosine_with_initial_delta_direction
direction_reset_suspected
```

The reported value is a finite-time estimate in `1/year`; the JSON summary also
reports the corresponding finite-time scale in years and Myr. It is not an
asymptotic Lyapunov exponent unless duration scaling supports a nonzero plateau.
The phase-space norm is scaled: positions are measured in AU and velocities in
AU/year. The initial perturbation is radial on the selected body (`mercury`,
`venus`, `earth`, `mars`, `jupiter`, `saturn`, or `all` for the inner planetary
barycenters), with a compensating Sun shift to preserve the total barycenter
position.

Do not over-interpret a single run, and do not trust a clean-looking straight
line in accumulated log growth by itself. The estimator must pass
near-integrable checks, especially:

```bash
--model-scope two_body_mercury
```

That mode contains only Sun + Mercury barycenter and should not produce a
strong positive Lyapunov exponent. A fitted time of years or decades there is a
method/norm/renormalization warning, not a discovery.

The expected inner Solar System Lyapunov time is on the order of a few Myr, so
short runs are smoke tests only. Renormalization intervals must keep the
perturbation in the local tangent regime; very large per-interval growth factors
or dominant velocity components are red flags. Use `--lyapunov-no-renorm` to
inspect raw separation growth and `--lyapunov-debug` to make the summary more
explicit during validation.

Run timestep convergence checks before treating an estimate as meaningful:

```bash
for step in 8 4 2; do
  python -m mini_ephemeris.long_term_stability_cli \
    --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
    --start-date 2000-01-01 \
    --duration-years 10000 \
    --step-days "$step" \
    --record-every-years 10 \
    --with-lyapunov \
    --lyapunov-body mercury \
    --lyapunov-perturbation-m 1.0 \
    --lyapunov-renorm-years 10 \
    --lyapunov-fit-start-years 1000 \
    --lyapunov-fit-end-years 10000 \
    --output-dir /home/peacelovephysics/ephemeris/output \
    --tag "stability_10kyr_lyapunov_step${step}d" \
    --no-progress-bar
done
```

The summary JSON intentionally warns that Lyapunov estimates depend on the
timestep, force model, perturbation norm, perturbation target, renormalization
interval, fit window, and total duration.

For broader validation, run the matrix script:

```bash
bash scripts/debug_lyapunov_validation.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/lyapunov_validation_matrix.csv
```

The matrix varies `full` / `inner` / `two_body_mercury`, `gr_model` none/sun,
renormalization interval, and perturbation size. Rows with large changes across
perturbation size or renormalization interval should be treated as failed
convergence checks.

For conservation baselines, prefer `--gr-model none`. The current
`--gr-model sun` implementation is a Sun-centered 1PN acceleration callback
applied to the planets. It is useful as a Mercury-sensitive perturbation test,
but it is not exactly symplectic and is not momentum-conserving because the
equal-and-opposite Sun reaction is not applied.

### Lyapunov validation before Solar System interpretation

Do not move from the Lyapunov smoke tests to Poincare sections, frequency maps,
or MEGNO-style diagnostics until the near-integrable tests are boring.

#### Why validation starts with Jupiter/Saturn before Mercury

The clean two-body validation path starts with:

```text
--model-scope two_body_jupiter
--model-scope two_body_saturn
```

These are Sun + Jupiter barycenter and Sun + Saturn barycenter Newtonian
systems. They are cleaner first gates because Jupiter and Saturn have longer
periods, lower eccentricities, larger perihelion distances, and much smaller GR
sensitivity than Mercury. With a fixed leapfrog step, they give many more steps
per orbit at ordinary validation timesteps.

Mercury remains an important stress test, but it is numerically demanding: its
short period, eccentricity, close perihelion, and GR sensitivity make it a poor
first target for debugging the tangent/Lyapunov machinery. GR should remain
disabled during this clean validation path:

```text
--gr-model none
```

Passing Jupiter/Saturn is necessary but not sufficient for Mercury-sensitive
Solar System Lyapunov work. If Jupiter/Saturn fail, suspect the tangent
estimator or Newtonian Jacobian implementation. If Jupiter/Saturn pass but
Mercury fails, Mercury is likely exposing timestep and orbital-resolution
limits.

The two-body scopes contain only the Sun and one planetary barycenter with
Newtonian gravity when run with:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 1000 \
  --step-days 4 \
  --record-every-years 10 \
  --model-scope two_body_jupiter \
  --gr-model none \
  --integrator leapfrog \
  --output-dir /home/peacelovephysics/ephemeris/output/stability \
  --tag two_body_jupiter_validation \
  --no-progress-bar
```

This near-integrable case should not show true chaos. A radial perturbation can
still cause ordinary phase shear because the perturbed orbit has a slightly
different period. Phase shear is not chaos, and a clean linear accumulated-growth
plot is not sufficient evidence of a physical Lyapunov exponent.

The two-body validation CSV reports energy drift, angular momentum drift,
semi-major-axis drift, eccentricity drift, numerical perihelion drift, and an
orbital period estimate:

```text
two_body_validation_<tag>.csv
```

Run the timestep ladder first:

```bash
bash scripts/debug_two_body_timestep_ladder.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/two_body_timestep_ladder.csv
```

Only after the orbit ladder shows acceptable timestep convergence should the
two-body Lyapunov ladder be used:

```bash
bash scripts/debug_two_body_tangent_lyapunov_ladder.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/two_body_tangent_lyapunov_ladder.csv
```

The preferred validation method is:

```text
--lyapunov-method tangent
```

For Newtonian leapfrog, this propagates the tangent vector through the
velocity-Verlet map using the Newtonian variational acceleration. Benettin
renormalization must preserve the evolved perturbation direction: compute the
current scaled phase-space delta, scale that vector, unscale it back into SI
coordinates, and continue from `reference + renormalized_delta`. The debug CSV
records cosines with the previous and initial perturbation directions plus a
`direction_reset_suspected` flag. Repeated snapping back to the initial radial
direction means the estimator is wrong.

Use `--lyapunov-no-renorm` to write:

```text
no_renorm_separation_<tag>.csv
```

The no-renormalization summary includes both `log(separation)` vs time and
`log(separation)` vs `log(time)` fits. The log-log fit helps identify ordinary
algebraic or polynomial growth from phase shear.

#### Finite-time Lyapunov estimates and phase shear

Nearby Kepler orbits can separate by ordinary phase drift without chaos. A
small radial displacement changes the osculating orbit and therefore the orbital
period; over time, the two states can shear apart in phase even though the
underlying two-body system is integrable.

A clean positive finite-time slope over a short fit window is not enough. In an
integrable validation case, the finite-time estimate `lambda_T` should trend
toward zero as the total duration `T` increases. A strong `1/T` or
`log(T)/T` trend indicates finite-time shear, not a nonzero asymptotic Lyapunov
exponent.

Run the duration-scaling ladder:

```bash
bash scripts/debug_two_body_tangent_duration_ladder.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/two_body_tangent_duration_ladder.csv
/home/peacelovephysics/ephemeris/output/stability/two_body_tangent_duration_scaling_summary.csv
/home/peacelovephysics/ephemeris/output/stability/two_body_tangent_duration_scaling_summary.json
/home/peacelovephysics/ephemeris/output/stability/tangent_no_renorm_duration_summary.csv
```

It also writes simple per-scope plots:

```text
finite_time_lambda_vs_duration_<model_scope>.png
accumulated_log_growth_vs_duration_<model_scope>.png
```

The scaling classifier compares `lambda_T` against `1/T` and `log(T)/T`, and
compares accumulated log growth against both `T` and `log(T)`. A two-body case
should classify as `near_integrable_likely`. A `chaotic_candidate` classification
requires evidence of a nonzero plateau across duration, timestep, perturbation
size, and renormalization interval before a Solar System Lyapunov time should be
interpreted physically.

Solar System Lyapunov estimates require convergence across timestep,
perturbation size, renormalization interval, force model, and fit window.
`--gr-model sun` is currently not suitable for conservation-sensitive Lyapunov
validation because the Sun 1PN approximation is not symplectic or
momentum-conserving in this package.

#### Nonlinear diagnostics

The stability CLI includes optional nonlinear dynamics diagnostics, but they
inherit the same scientific boundary as the Lyapunov workflow: stability mode
only, Earth-Moon barycenter for planetary runs, no empirical lunar calibration,
and no American Ephemeris apparent/geocentric/tropical machinery.

Poincare-style sections are enabled with:

```bash
--with-poincare \
--poincare-body mercury \
--poincare-plane z \
--poincare-direction positive
```

The section detector uses heliocentric J2000 ecliptic coordinates and linearly
interpolates crossings of `z=0` or `y=0`. It writes
`poincare_<tag>_<body>.csv` and `poincare_<tag>_<body>.png`. These sections are
most rigorous for lower-dimensional systems; in the full Solar System they are
exploratory visual slices, not proof of chaos.

Frequency-map analysis is enabled with:

```bash
--with-frequency-map \
--frequency-window-years 250 \
--frequency-step-years 100 \
--frequency-bodies mercury,venus,earth,mars
```

This is NAFF-lite / FFT-lite, not full Laskar NAFF. It forms the secular complex
variables `z_e = e exp(i varpi)` and `z_i = sin(i/2) exp(i Omega)` from the
recorded orbital-element CSV, then estimates dominant FFT peaks in sliding
windows. Frequency drift across windows can indicate secular diffusion, but it
must be checked against timestep, output cadence, window length, and force
model choices.

FLI-lite and MEGNO-lite are enabled with:

```bash
--with-lyapunov \
--lyapunov-method tangent \
--with-fli \
--with-megno-lite
```

These indicators use the Newtonian tangent-vector machinery and are labeled as
finite-time tangent-growth diagnostics. FLI-lite records log tangent norm
growth; MEGNO-lite uses a discrete renormalization-interval approximation. Do
not treat either as a final chaos diagnosis by itself. Duration scaling remains
required before interpreting any finite-time Lyapunov, FLI, or MEGNO-like
quantity as evidence for a nonzero asymptotic exponent. Two-body validation
scopes must remain `near_integrable_likely`; if a nonlinear indicator reports
strong chaos there, the indicator has failed validation until proven otherwise.

Inner Solar System duration-scaling smoke tests are available through:

```bash
bash scripts/debug_inner_tangent_duration_ladder.sh
```

The full matrix is intentionally expensive. Use `MAX_CASES=<n>` for a bounded
workflow smoke. The script writes:

```text
/home/peacelovephysics/ephemeris/output/stability/inner_tangent_duration_ladder.csv
/home/peacelovephysics/ephemeris/output/stability/inner_tangent_duration_scaling_summary.csv
/home/peacelovephysics/ephemeris/output/stability/inner_tangent_duration_scaling_summary.json
```

Inner and full Solar System diagnostics require convergence across timestep,
duration, renormalization interval, perturbation size, fit window, and model
scope before physical interpretation. Short 1-10 kyr smoke tests are workflow
checks, not measurements of the real asymptotic Lyapunov exponent.

#### Practical run ladder

The desktop run ladder is designed for a Ryzen 9 7900X / 32 GB RAM machine and
assumes CPU-only execution. All scripts use:

```text
/home/peacelovephysics/ephemeris/data/de431_part-2.bsp
/home/peacelovephysics/ephemeris/output/stability
```

Start with the plumbing and conservation check:

```bash
bash scripts/run_stability_100yr_validation.sh
```

This is a full reduced Solar System run with no heavy nonlinear diagnostics. It
is meant to verify CSV/JSON output, invariants, min separations, and report
generation.

Then reproduce the standard inner nonlinear smoke:

```bash
bash scripts/run_inner_1kyr_nonlinear_smoke.sh
```

This runs the inner model with Mercury tangent finite-time Lyapunov, Poincare
crossings, NAFF-lite frequency maps, and FLI/MEGNO-lite. On the Ryzen 9 7900X,
the 1 kyr smoke at `step_days=0.25` took about 141 seconds. Rough scaling from
that run:

```text
1 kyr at 0.25 day: about 141 seconds
10 kyr at 0.25 day: about 20-30 minutes
30 kyr at 0.25 day: about 1-1.5 hours
100 kyr at 0.25 day: about 4 hours
0.125 day: roughly doubles runtime
```

Begin duration scaling before interpretation:

```bash
MAX_CASES=3 bash scripts/run_inner_10kyr_duration_scaling.sh
bash scripts/run_inner_10kyr_duration_scaling.sh
```

The bounded form is for smoke testing resume and summary behavior. The full
script runs inner `duration_years = 100, 300, 1000, 3000, 10000` at
`step_days=0.25` across Mercury/all perturbation targets, perturbation sizes,
and renormalization intervals. `RESUME=1` is the default and skips completed
cases whose `summary_<tag>.json` exists. The script writes:

```text
inner_10kyr_duration_ladder.csv
inner_10kyr_duration_scaling_summary.csv
inner_10kyr_duration_scaling_summary.json
```

After the ladder suggests a stable setting, run a longer single case:

```bash
bash scripts/run_inner_30kyr_bestcase.sh
```

This is still a finite-time diagnostic. It defaults to inner, Newtonian,
`step_days=0.25`, `duration_years=30000`, `lyapunov_body=all`, and a frequency
map with wider windows. Optional output-heavy diagnostics can be enabled with
`WITH_POINCARE=1` or `WITH_FLI_MEGNO=1`.

For broad full-system surveys:

```bash
bash scripts/run_full_10kyr_baseline.sh
bash scripts/run_full_100kyr_reduced_output.sh
```

The full 10 kyr and 100 kyr scripts are conservation/secular-survey baselines.
Default `step_days=2` or `4` full Solar System runs are not Mercury-sensitive
Lyapunov runs. They are useful for tracking invariants, orbital elements, min
separations, and coarse secular behavior. Inner Mercury-sensitive chaos
diagnostics require smaller steps and explicit duration scaling.

Benchmark and report helpers:

```bash
python -m mini_ephemeris.stability_benchmark_summary \
  /home/peacelovephysics/ephemeris/output/stability/summary_*.json \
  --output-dir /home/peacelovephysics/ephemeris/output/stability

python -m mini_ephemeris.stability_scientific_summary \
  /home/peacelovephysics/ephemeris/output/stability \
  --output-dir /home/peacelovephysics/ephemeris/output/stability
```

The benchmark helper writes compact runtime/conservation CSVs. The scientific
summary helper writes Markdown reports using finite-time language, duration
scaling classifications, and conservative next-run recommendations.

GR remains excluded from the validated tangent Lyapunov/FLI/MEGNO-lite path for
now. Use `--gr-model none` for conservation-sensitive nonlinear diagnostics
unless a momentum-consistent GR tangent model is added and validated.

#### Ensemble stability experiments

Ensemble mode runs independent nearby initial conditions in parallel across CPU
workers:

```bash
python -m mini_ephemeris.stability_ensemble_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --model-scope inner \
  --duration-years 1000 \
  --step-days 0.25 \
  --ensemble-size 4 \
  --workers 2 \
  --output-dir /home/peacelovephysics/ephemeris/output/stability \
  --tag ensemble_inner_1kyr_smoke \
  --with-lyapunov \
  --lyapunov-method tangent \
  --lyapunov-body mercury \
  --lyapunov-renorm-years 0.25 \
  --resume
```

Each member receives a reproducible random RTN position perturbation with the
requested meter scale. RTN means radial, tangential, and normal with respect to
the instantaneous heliocentric orbit. The selected-body displacement is
compensated by a Sun shift to preserve the barycenter position; no velocity
kick is applied, so total momentum is unchanged by the ensemble perturbation.

Outputs are written under:

```text
output_dir/<tag>/member_000/
output_dir/<tag>/member_001/
...
```

The ensemble root also receives:

```text
ensemble_summary_<tag>.csv
ensemble_summary_<tag>.json
ensemble_run_<tag>.log
```

`--resume` skips a member only after checking that required JSON/CSV files
exist, CSV files parse, and no NUL bytes are present. If a member looks corrupt,
its files are moved to `corrupt_member_backup/` and the member is rerun.

Desktop worker counts should be modest. The default is
`min(6, ensemble_size)` rather than all 24 hardware threads, and the supplied
scripts use smaller defaults:

```bash
bash scripts/run_ensemble_inner_1kyr_smoke.sh
bash scripts/run_ensemble_inner_10kyr_small.sh
```

Ensembles probe statistical robustness of nearby initial conditions and help
scale the workflow, but ensemble spread is not automatically a Lyapunov
exponent. Finite-time tangent Lyapunov, FLI, and MEGNO-lite values still require
duration scaling before interpretation. Broad 10 kyr ensembles are useful
workflow/statistical checks; they are not final Solar System chaos measurements.

#### Optional backends and restartable production runs

Optional backend packages are intentionally not required dependencies. Check the
local machine with:

```bash
bash scripts/check_backends.sh
python -m mini_ephemeris.check_optional_backends
```

The checker reports availability and versions for `rebound`, `reboundx`,
`numba`, and `cupy`. Missing optional packages do not fail the stability
package.

The in-house leapfrog backend now supports checkpoint/restart:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --model-scope inner \
  --gr-model none \
  --integrator leapfrog \
  --duration-years 1000 \
  --step-days 0.25 \
  --checkpoint-every-years 100 \
  --checkpoint-dir /home/peacelovephysics/ephemeris/output/stability/checkpoints_inner_run \
  --keep-checkpoints 3
```

Resume from either a specific checkpoint file or a checkpoint directory:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --resume-from-checkpoint /home/peacelovephysics/ephemeris/output/stability/checkpoints_inner_run
```

Use the same tag, output directory, timestep, model scope, and diagnostics when
resuming because CSV outputs are appended and the checkpoint carries a
configuration hash.

Run the in-house checkpoint smoke test with:

```bash
bash scripts/run_checkpoint_resume_smoke.sh
```

REBOUND support is optional, but when installed it is now the leading
production trajectory backend candidate:

```bash
bash scripts/run_rebound_two_body_validation.sh
python -m mini_ephemeris.rebound_validation_cli --help
```

REBOUNDx GR options are explicit non-default choices:
`none`, `gr`, `gr_full`, and `gr_potential`. `gr_potential` is the leading
WHFast-compatible candidate pending longer validation, and GR is still excluded
from the validated in-house tangent/FLI/MEGNO-lite path. Mercury GR precession
smoke testing lives in:

```bash
bash scripts/validate_mercury_gr_precession.sh
```

See `docs/long_term_integrator_strategy.md` for the production backend plan,
the current `gr_model=sun` limitation, and checkpoint requirements for
week-long runs.

#### Production REBOUND backend

The main stability CLI can now run production trajectory integrations through
REBOUND while preserving the stability-mode boundary:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --backend rebound \
  --rebound-integrator whfast \
  --rebound-gr-model none \
  --model-scope inner
```

The in-house backend remains the path for tangent Lyapunov, FLI-lite,
MEGNO-lite, and Poincare streaming diagnostics. REBOUND writes the normal
stability CSV/JSON outputs, and now also supports REBOUND-native finite-time
MEGNO/LCN diagnostics through REBOUND's variational equations. The old
in-house tangent diagnostics are still rejected clearly with `--backend
rebound`.

REBOUND-native chaos diagnostics:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --backend rebound \
  --rebound-integrator whfast \
  --rebound-gr-model none \
  --model-scope inner \
  --with-megno \
  --with-rebound-lyapunov
```

The outputs are:

```text
megno_<tag>.csv
megno_summary_<tag>.json
megno_growth_<tag>.png
```

These are finite-time diagnostics. A regular two-body system should remain
regular under MEGNO validation, and inner Solar System chaos claims still
require duration scaling, timestep comparison, perturbation/model checks, and
WHFast versus IAS15 comparison. REBOUNDx `gr_potential` is the leading
WHFast-compatible GR candidate, but GR MEGNO results should be interpreted as
exploratory until longer validation passes.

Recommended roles:

```text
in-house leapfrog:
  tangent diagnostics, current checkpointed diagnostic workflow

REBOUND WHFast:
  leading long Newtonian production trajectory backend

REBOUND IAS15:
  high-accuracy short validation oracle

REBOUND WHFast + REBOUNDx gr_potential:
  leading WHFast-compatible GR candidate pending longer validation

REBOUND WHFast + velocity-dependent gr/gr_full:
  use carefully; operator-style validation is still needed
```

Useful production wrappers:

```bash
bash mini_ephemeris/scripts/run_rebound_inner_10kyr_newtonian.sh
bash mini_ephemeris/scripts/run_rebound_inner_10kyr_gr_potential.sh
bash mini_ephemeris/scripts/run_backend_comparison_ladder.sh
bash mini_ephemeris/scripts/run_rebound_megno_validation.sh
```

#### Research-Grade Workflow

Before interpreting long regular-looking MEGNO runs, keep the workflow
reproducible and falsifiable:

- Run positive controls with `bash mini_ephemeris/scripts/run_rebound_positive_control_megno.sh`.
  Two-body controls should stay `regular_likely`; the toy compact three-body
  case should become `chaotic_candidate` or `unstable_or_escape`.
- Use duration/timestep/seed ladders before interpretation. The created
  full-system Newtonian ladder is
  `bash mini_ephemeris/scripts/run_rebound_full_newtonian_megno_research_ladder.sh`;
  it supports `MAX_CASES` and `RESUME=1`.
- Use `python -m mini_ephemeris.secular_frequency_summary` to compare
  FFT-lite / NAFF-lite secular frequency drift across Newtonian, GR trajectory,
  timestep, and duration runs. This is not full Laskar NAFF.
- Use `python -m mini_ephemeris.rebound_shadow_lyapunov_cli` for
  two-trajectory shadow-divergence experiments aimed at literature-style
  finite-time Lyapunov windows. The 100 Myr wrappers are created but should be
  launched outside Codex:
  `mini_ephemeris/scripts/run_rebound_full_newtonian_shadow_100myr.sh` and
  `mini_ephemeris/scripts/run_rebound_full_with_pluto_shadow_100myr.sh`.
  `--resume` only skips a completed final summary. For power-loss recovery use
  `--resume-from-checkpoint latest` with `--checkpoint-every-years`,
  `--checkpoint-dir`, and `--write-partial-every-record`. The 100 Myr scripts
  use 1 Myr checkpoint bundles and keep the last 5 valid checkpoints, so an
  interruption loses at most progress since the latest checkpoint, not the
  whole run. Inspect checkpoint bundles with
  `python -m mini_ephemeris.rebound_shadow_lyapunov_cli --inspect-checkpoints --checkpoint-dir PATH`.
- Use `python -m mini_ephemeris.stability_research_report` for cautious
  Markdown summaries and `python -m mini_ephemeris.pack_stability_batch` for
  uploadable artifact zips with manifests.
- REBOUND WHFast remains the production trajectory candidate; REBOUND IAS15 is
  the high-accuracy short-validation oracle. The in-house backend remains useful
  for older tangent diagnostics and checkpoint experiments.
- REBOUNDx `gr_potential` is useful for GR trajectory studies, but GR MEGNO is
  not validated in the current REBOUNDx path because variational particles are
  not evolved self-consistently there.
- SimulationArchive checkpoint `.bin` files are excluded from artifact upload
  packages by default unless `--include-archives` is explicitly passed.

The stability mode explicitly rejects empirical lunar calibration flags such as
`--moon-dv-t-mm-s`, `--moon-a-t-1e-15-m-s2`, and
`--lunar-calibration-profile`. It also does not import or use American
Ephemeris apparent/geocentric/tropical output machinery.

That separation keeps the project scientifically honest:

```text
Ephemeris mode:
  fitted, JPL/book matching, short range, explicit Moon

Stability mode:
  physical reduced model, long range, Earth-Moon barycenter, no empirical lunar fit
```

---

## Current recommendation

Keep the successful v3 full-book empirical lunar calibration as the stable short-range baseline. The next ephemeris-matching milestone is the v4 six-parameter run:

1. Run `src/mini_ephemeris/run_lunar_6param_v4_dual_annealing_full_book.sh`.
2. Let dual annealing explore the six-dimensional bounded space with Powell local search enabled.
3. Save the successful profile as `american_ephemeris_2000_2050_full_book_6param_empirical_v4_dual_annealing`.
4. Validate the saved profile with `american_ephemeris_range_cli` and compare the final residual plots against the v3 baseline.
5. If the longitude peak or latitude envelope does not improve meaningfully, use the residual shape to choose the next physical refinement.

The long-term dynamics milestone remains separate:

1. Start with short smoke tests.
2. Inspect energy, angular momentum, center-of-mass, and minimum-separation diagnostics before scaling up.
3. Scale through 10 kyr / 100 kyr reduced-model runs before attempting million-year runs.

The project now has a strong short-range ephemeris-matching baseline, implemented four- and six-parameter lunar optimization paths, and a clear separation between fitted ephemeris reproduction and long-term physical dynamics.
