# Long-Term Integrator Strategy

This document covers the stability subsystem only:

- physical reduced Solar System model
- Earth-Moon barycenter, not explicit Earth and Moon
- no empirical lunar calibration
- no American Ephemeris apparent/geocentric/tropical machinery

The short-range American Ephemeris reproduction mode remains a separate fitted
workflow and should not be mixed into long-term stability studies.

## Current In-House Baseline

The current production baseline is the in-house Newtonian fixed-step
kick-drift-kick/velocity-Verlet integrator exposed through:

```bash
python -m mini_ephemeris.long_term_stability_cli --integrator leapfrog
```

It is simple, transparent, and useful for conservation checks, duration-scaling
experiments, finite-time tangent diagnostics, Poincare-style sections, and
NAFF-lite frequency-map surveys. It streams CSV output and now supports
checkpoint/restart for long runs.

For validated tangent Lyapunov, FLI-lite, and MEGNO-lite workflows, the current
recommended force model is:

```bash
--gr-model none
```

That Newtonian-only path is the one that has passed the current two-body
Jupiter/Saturn/Mercury duration-scaling validation, where finite-time tangent
estimates trend toward zero as expected for near-integrable two-body systems.

## Leapfrog Limits For Mercury-Sensitive Myr Runs

Fixed-step leapfrog is a good baseline, but it is not automatically the final
choice for Mercury-sensitive million-year integrations.

Mercury is demanding because it has:

- short orbital period
- high eccentricity
- small perihelion distance
- strong sensitivity to timestep choice
- important relativistic perihelion precession in real Solar System dynamics

A broad full-system run at 2-4 day steps is useful as a secular survey, but it
is not a validated Mercury Lyapunov run. Mercury-sensitive diagnostics require
smaller timesteps, duration scaling, perturbation-size checks, renormalization
interval checks, and comparison against an independent backend before expensive
interpretation-level ensembles.

## Current GR Limitation

The current in-house `--gr-model sun` option adds a Sun-centered 1PN correction
through the acceleration callback. In the leapfrog path this means:

- the method is no longer exactly symplectic
- the approximation is not currently implemented as a pairwise
  momentum-conserving interaction
- center-of-mass and angular-momentum diagnostics can degrade relative to the
  Newtonian run

Because of that, `gr_model=sun` is not part of the validated conservation-
sensitive tangent/FLI/MEGNO-lite path. It remains useful as a documented
comparison mode, not as the default long-term chaos backend.

## Why Expensive Ensembles Should Wait

The ensemble CLI is useful for workflow scaling and statistical robustness, but
ensemble spread is not automatically a Lyapunov exponent. The current
100-10000 year inner duration-scaling results still look finite-time/shear
dominated rather than like a robust nonzero asymptotic plateau.

Before week-long scientific ensembles, settle:

- Newtonian versus GR physics target
- timestep ladder for the chosen model scope
- backend comparison strategy
- checkpoint/restart behavior under interruption
- duration-scaling classification for the selected diagnostics

The safe interpretation language is still "finite-time diagnostic" unless a
nonzero plateau survives convergence checks.

## Proposed REBOUND/REBOUNDx Strategy

REBOUND is still an optional dependency, but it is now the leading production
trajectory candidate when installed.

Near-term plan:

1. Use `python -m mini_ephemeris.check_optional_backends` to report whether
   `rebound`, `reboundx`, `numba`, and `cupy` are installed.
2. Use `python -m mini_ephemeris.rebound_validation_cli` for Newtonian
   two-body and short full/inner validation runs.
3. Compare REBOUND WHFast and IAS15 conservation diagnostics against the
   in-house leapfrog summaries for the same initial states.
4. Use REBOUNDx GR options only as an explicit scaffold:
   `none`, `gr`, `gr_full`, and `gr_potential`.
5. Validate Mercury GR perihelion precession before using any REBOUNDx GR path
   for physical interpretation.

REBOUND WHFast is a natural candidate for long Newtonian symplectic surveys.
REBOUND IAS15 is a useful high-accuracy reference for shorter comparisons.
REBOUNDx is the likely place to validate GR treatments, but those options should
not become default until the conservation and perihelion checks are understood.

## Checkpoint/Restart Requirements

Week-long runs must be restartable. The in-house checkpoint format stores:

- physical positions, velocities, and masses
- current time and record count
- invariant reference state
- invariant extrema accumulated so far
- pairwise minimum-separation tracker
- tangent/Lyapunov state when enabled
- accumulated finite-time Lyapunov sums and samples
- RNG state placeholder
- configuration hash

The write path is atomic:

1. write a temporary `.npz`
2. flush and fsync the file data
3. rename into place
4. prune old checkpoints according to `--keep-checkpoints`

Use:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --checkpoint-every-years 100 \
  --checkpoint-dir /home/peacelovephysics/ephemeris/output/stability/checkpoints_my_run \
  --keep-checkpoints 3
```

Resume with:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --resume-from-checkpoint /home/peacelovephysics/ephemeris/output/stability/checkpoints_my_run
```

The resume path checks that the checkpoint configuration hash matches the new
command. Existing output CSVs are opened in append mode, so keep the same tag,
output directory, model scope, timestep, and diagnostics when resuming.

For the optional REBOUND path, the validation CLI exposes REBOUND
SimulationArchive scaffolding when available:

```bash
python -m mini_ephemeris.rebound_validation_cli \
  --simulation-archive /home/peacelovephysics/ephemeris/output/stability/rebound_archive.bin \
  --simulation-archive-interval-years 10
```

and:

```bash
python -m mini_ephemeris.rebound_validation_cli \
  --resume-from-simulation-archive /home/peacelovephysics/ephemeris/output/stability/rebound_archive.bin
```

That path is for restart smoke tests and backend validation. It is not yet the
primary production ensemble backend.

## Backend Benchmark Results and Production Recommendation

Run the compact benchmark suite with:

```bash
bash scripts/run_backend_accuracy_benchmark.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/backend_accuracy_benchmark.csv
/home/peacelovephysics/ephemeris/output/stability/backend_accuracy_benchmark.json
/home/peacelovephysics/ephemeris/output/stability/backend_accuracy_benchmark.md
```

The suite always runs the in-house Newtonian leapfrog cases and adds REBOUND
WHFast, REBOUND IAS15, and REBOUNDx GR cases only when those optional packages
are installed. The benchmark ranks candidates separately by:

- fastest
- best conservation
- best Mercury perihelion behavior
- best candidate for week-long production runs
- best candidate for high-accuracy short validation runs

Do not collapse these into one scalar score. The fastest backend is not
necessarily the best validation oracle, and the best short validation integrator
is not automatically the right production integrator for a week-long run.

Run restart stress tests with:

```bash
bash scripts/run_checkpoint_restart_stress_test.sh
```

It writes:

```text
/home/peacelovephysics/ephemeris/output/stability/checkpoint_restart_stress_test.csv
/home/peacelovephysics/ephemeris/output/stability/checkpoint_restart_stress_test.json
/home/peacelovephysics/ephemeris/output/stability/checkpoint_restart_stress_test.md
```

Current default recommendation before optional REBOUND benchmarks pass:

- production backend: in-house Newtonian leapfrog
- GR model: none
- checkpoint cadence: 100 years for first long shakedowns, 10 years while
  testing restart behavior under interruption
- first long run: 10-30 kyr inner Newtonian checkpointed run before expensive
  GR or ensemble runs

REBOUND WHFast is now the leading candidate for long Newtonian production
surveys. If IAS15 is installed, treat it as a high-accuracy short validation
oracle, not automatically as the week-long production backend. If REBOUNDx is
installed, `gr_potential` is the leading WHFast-compatible GR candidate, while
velocity-dependent GR paths still need careful operator-style validation.

### Current Compact Benchmark Snapshot

On the current environment, the optional backend checker found:

- REBOUND 4.6.0
- REBOUNDx 4.6.1
- numba 0.65.0
- no cupy

The compact benchmark written to `backend_accuracy_benchmark.*` completed all
18 scheduled cases. Key observations:

- For the 1 kyr inner Newtonian case, in-house leapfrog at 1 day took about
  10.69 s with max relative energy drift about `8.59e-5`.
- For the same 1 kyr inner Newtonian case, REBOUND WHFast at 1 day took about
  0.30 s with max relative energy drift about `4.19e-10`.
- REBOUND IAS15 on the 1 kyr inner Newtonian case took about 1.67 s with max
  relative energy drift about `3.28e-15`, making it a strong short validation
  oracle.
- For two-body Mercury Newtonian, REBOUND WHFast and IAS15 showed near-zero
  numerical perihelion drift at the benchmark settings, while the in-house
  fixed-step Mercury stress case still showed large numerical perihelion drift.
- REBOUNDx `gr`, `gr_potential`, and IAS15+`gr` all produced Mercury two-body
  perihelion precession near the expected order of 43 arcsec/century.
- WHFast plus velocity-dependent REBOUNDx `gr` emitted a REBOUND warning that
  it should be applied as an operator path, so that combination is not a
  production recommendation yet.

Current recommendation after this benchmark:

- Backend for next Newtonian production candidate: REBOUND WHFast through the
  production stability CLI.
- Backend for immediate in-house tangent-diagnostic work: in-house leapfrog,
  because the historical tangent/FLI/MEGNO-lite validation path is integrated there.
- High-accuracy validation backend: REBOUND IAS15.
- GR model: keep `none` for validated tangent/chaos work. For GR validation,
  prefer REBOUNDx `gr_potential` with WHFast or `gr` with IAS15 until the
  operator-style WHFast+`gr` path is explicitly implemented and tested.
- First long run: a 10-30 kyr inner Newtonian checkpointed run, paired with a
  matching REBOUND WHFast validation survey before starting expensive ensembles
  or week-long GR runs.

## Production REBOUND Backend

The main stability CLI now accepts:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --backend rebound \
  --rebound-integrator whfast \
  --rebound-gr-model none
```

The REBOUND backend uses the same DE431 barycentric initial-state construction
as the in-house backend and writes the standard stability outputs:

```text
stability_timeseries_<tag>.csv
orbital_elements_<tag>.csv
invariants_<tag>.csv
min_separations_<tag>.csv
summary_<tag>.json
```

The in-house backend remains the active path for tangent Lyapunov, FLI-lite,
MEGNO-lite, and Poincare streaming diagnostics. If those tangent diagnostics
are requested with `--backend rebound`, the CLI fails clearly instead of
pretending they ran.

Current backend roles:

- in-house leapfrog: transparent baseline and nonlinear/tangent diagnostics
- REBOUND WHFast: leading long Newtonian production trajectory backend
- REBOUND IAS15: high-accuracy short validation oracle
- REBOUND WHFast + REBOUNDx `gr_potential`: leading WHFast-compatible GR
  candidate pending longer validation
- REBOUND WHFast + velocity-dependent `gr`/`gr_full`: treat carefully because
  REBOUND warns that operator-style handling is needed

Production wrappers:

```bash
bash mini_ephemeris/scripts/run_rebound_inner_10kyr_newtonian.sh
bash mini_ephemeris/scripts/run_rebound_inner_30kyr_newtonian.sh
bash mini_ephemeris/scripts/run_rebound_inner_10kyr_gr_potential.sh
bash mini_ephemeris/scripts/run_rebound_inner_30kyr_gr_potential.sh
bash mini_ephemeris/scripts/run_rebound_ias15_inner_1kyr_validation.sh
bash mini_ephemeris/scripts/run_backend_comparison_ladder.sh
```

The comparison ladder writes:

```text
/home/peacelovephysics/ephemeris/output/stability/backend_comparison_ladder.csv
/home/peacelovephysics/ephemeris/output/stability/backend_comparison_ladder.json
/home/peacelovephysics/ephemeris/output/stability/backend_comparison_ladder.md
```

## REBOUND-Native Chaos Diagnostics

REBOUND 4.6.0 exposes `Simulation.init_megno()`, `Simulation.megno()`, and
`Simulation.lyapunov()` for WHFast and IAS15. The production stability CLI now
uses those native variational-equation hooks when requested with:

```bash
python -m mini_ephemeris.long_term_stability_cli \
  --backend rebound \
  --rebound-integrator whfast \
  --rebound-gr-model none \
  --with-megno \
  --with-rebound-lyapunov
```

The CLI writes `megno_<tag>.csv`, `megno_summary_<tag>.json`, and
`megno_growth_<tag>.png`. The LCN reported by REBOUND is converted from the
simulation time unit to `1/year`, because the stability backend initializes
REBOUND in SI units. REBOUND 4.6.0 does not expose a separate Python accessor
for a distinct `mean_megno`, so that output is left blank unless a future API
provides it.

All REBOUND MEGNO and LCN outputs in this project remain finite-time
diagnostics unless duration scaling supports a nonzero plateau. The validation
gate is:

```bash
bash mini_ephemeris/scripts/run_rebound_megno_validation.sh
```

That ladder checks two-body Jupiter, Saturn, and Mercury as regular systems,
then compares inner Solar System WHFast and IAS15 cases with and without the
current GR candidates. Two-body systems must classify as `regular_likely`. If
they do not, the MEGNO setup, timestep, or GR coupling needs attention before
any inner/full Solar System interpretation.

Current production interpretation:

- REBOUND WHFast remains the leading Newtonian production trajectory backend.
- REBOUND IAS15 remains the high-accuracy short validation oracle.
- REBOUNDx `gr_potential` remains the leading WHFast-compatible GR candidate,
  but GR chaos diagnostics need careful comparison against Newtonian and IAS15
  cases.
- Velocity-dependent REBOUNDx `gr` and `gr_full` should still be treated with
  care under WHFast unless an operator-style path is explicitly implemented and
  benchmarked.
- The in-house backend remains useful for checkpointed baseline runs and the
  existing tangent/FLI/MEGNO-lite validation history, but REBOUND-native MEGNO
  is the preferred path for REBOUND-backed chaos analysis.

After the MEGNO validation ladder passes, the next created-but-not-default
production scripts are:

```bash
bash mini_ephemeris/scripts/run_rebound_inner_100kyr_newtonian.sh
bash mini_ephemeris/scripts/run_rebound_inner_100kyr_gr_potential.sh
bash mini_ephemeris/scripts/run_rebound_inner_100kyr_megno_ladder.sh
```

The first long MEGNO experiment should be a Newtonian WHFast inner duration
ladder before running GR or ensemble variants. A 100 kyr result that looks
interesting is still a candidate signal, not a final asymptotic Solar System
Lyapunov exponent, until timestep, integrator, duration, and force-model
comparisons agree.

## Research-Grade Workflow

The next long-run phase should prioritize reproducibility and controls before
larger compute jobs:

1. Positive controls. Run
   `mini_ephemeris/scripts/run_rebound_positive_control_megno.sh` before
   interpreting null MEGNO results. Two-body controls must remain
   `regular_likely`, while the compact toy three-body system should classify as
   `chaotic_candidate` or `unstable_or_escape`. The toy is not a Solar System
   model; it checks the diagnostic plumbing.
2. Duration/timestep/seed ladders. Use
   `mini_ephemeris/scripts/run_rebound_full_newtonian_megno_research_ladder.sh`
   for full Newtonian WHFast cases at 1, 5, and 10 Myr, with 1 day and 0.5 day
   steps and multiple MEGNO seeds. The script supports `MAX_CASES` and
   `RESUME=1`; do not run the full ladder casually.
3. Secular frequency diagnostics. Run
   `python -m mini_ephemeris.secular_frequency_summary` on orbital-elements
   CSVs to compare FFT-lite / NAFF-lite frequency drift across Newtonian, GR
   trajectory, timestep, and duration outputs. This is not full Laskar NAFF.
4. Literature-style shadow divergence. Run
   `python -m mini_ephemeris.rebound_shadow_lyapunov_cli` for two-trajectory
   finite-time divergence windows. The 100 Myr full and full+Pluto wrappers are
   intentionally scripts, not automatic tests:
   `mini_ephemeris/scripts/run_rebound_full_newtonian_shadow_100myr.sh` and
   `mini_ephemeris/scripts/run_rebound_full_with_pluto_shadow_100myr.sh`.
   Shadow fits must be checked for saturation and non-exponential windows.
   `--resume` only skips a completed final summary. Mid-run restart requires
   `--resume-from-checkpoint latest`. The 100 Myr scripts write independent
   checkpoint bundles every 1 Myr and keep the last 5 valid bundles. A power
   outage can lose progress since the most recent checkpoint, but it should not
   force a restart from zero. Checkpoints can be inspected with
   `--inspect-checkpoints --checkpoint-dir PATH`.
5. Reporting and packaging. Use
   `python -m mini_ephemeris.stability_research_report` for cautious Markdown
   summaries and `python -m mini_ephemeris.pack_stability_batch` to create
   uploadable zip bundles with manifests, commit hashes, logs, summaries,
   plots, and CSV outputs. SimulationArchive `.bin` checkpoint files are
   excluded from packages by default unless explicitly included.

Interpretation discipline:

- Say "regular-looking over this duration", not "stable forever".
- Treat MEGNO and LCN as finite-time diagnostics until duration scaling,
  timestep comparison, and seed comparison agree.
- Keep GR trajectory studies separate from GR MEGNO claims. REBOUNDx
  `gr_potential` is promising for WHFast GR trajectories, but current REBOUNDx
  variational-particle support does not validate GR MEGNO.
- Use REBOUND WHFast for production trajectories, REBOUND IAS15 for short
  high-accuracy checks, and the in-house backend for older tangent/checkpoint
  experiments.
