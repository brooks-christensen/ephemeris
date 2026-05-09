# mini_ephemeris

`mini_ephemeris` is a Python celestial-mechanics and ephemeris experimentation package. The current near-term goal is to reproduce the **apparent geocentric tropical zodiac longitudes** used by *The American Ephemeris for the 21st Century 2000-2050 at Midnight* by comparing a reduced N-body model against JPL/Skyfield reference positions.

The project currently has two distinct modes of use:

1. **Short-range ephemeris matching** over 2000-2050, using explicit Earth and Moon bodies and optional empirical lunar calibration.
2. **Future long-term stability studies**, which should use a cleaner dynamical model, likely with the Earth-Moon barycenter rather than a fitted explicit Moon model.

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
  experiments.py

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

# Future mode: long-term stability experiments

Long-term stability studies should be treated as a separate mode from the short-range ephemeris-matching mode.

For million-year integrations, do **not** use the empirical lunar calibration profile. It is fitted to JPL/book agreement over 2000-2050 and is not a general physical law.

Recommended long-term model:

```text
Sun
Mercury
Venus
Earth-Moon barycenter
Mars
Jupiter
Saturn
Uranus
Neptune
optional Pluto
optional major asteroids
```

Why use the Earth-Moon barycenter?

- The explicit Moon introduces a 27-day timescale.
- That forces small timesteps and makes million-year integrations expensive.
- For broad planetary stability, the Earth-Moon system mostly acts through its barycenter.
- Long-term studies care more about secular architecture than daily geocentric lunar positions.

Recommended numerical direction:

- Add or use a symplectic integrator for long-term work.
- Use fixed timesteps appropriate to the shortest modeled orbital period.
- Track energy error and angular momentum error.
- Run ensembles of nearby initial conditions.
- Compare qualitative behavior, not exact long-term longitudes.
- Include Sun 1PN GR for Mercury-sensitive experiments.
- Treat exact planetary phase after very long times cautiously because the solar system is chaotic.

Possible long-term experiment CLI:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --kernel-path /home/peacelovephysics/ephemeris/data/de431_part-2.bsp \
  --start-date 2000-01-01 \
  --duration-years 1000000 \
  --model emb \
  --integrator symplectic \
  --step-days 4 \
  --include-gr \
  --output /home/peacelovephysics/ephemeris/output/stability_1Myr_emb.csv
```

Possible diagnostics:

```text
semi-major axis vs time
eccentricity vs time
inclination vs time
perihelion precession
minimum planet-planet separation
energy drift
angular momentum drift
Lyapunov-style divergence between nearby initial conditions
ensemble statistics
```

A long-term stability mode should have its own configuration and should explicitly disable:

```text
empirical lunar initial correction
empirical lunar acceleration
apparent geocentric output machinery
American Ephemeris formatting
```

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

The long-term dynamics milestone should remain separate:

1. Build Earth-Moon barycenter model mode.
2. Add long-term integrator diagnostics.
3. Start with 10 kyr / 100 kyr tests before attempting million-year runs.

The project now has a strong short-range ephemeris-matching baseline, implemented four- and six-parameter lunar optimization paths, and a clear separation between fitted ephemeris reproduction and long-term physical dynamics.
