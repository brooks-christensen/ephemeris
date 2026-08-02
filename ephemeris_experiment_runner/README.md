# Ephemeris Experiment Runner

A small control-plane package for long Solar System stability experiments. It does not replace `mini_ephemeris`; it launches the existing CLI, reports progress and ETA, preserves state, and prevents failed stages from propagating into later stages.

## Main features

- Prints the experiment, objective, exact command, output directory, and target duration before each stage.
- Reports simulated time, percent complete, elapsed wall time, recent rate, ETA, CPU, memory, and progress source at a configurable cadence.
- Uses REBOUND SimulationArchive snapshots or incremental CSV output. The included manifests save archives at roughly 1% intervals for short runs and 0.5–1% intervals for long runs.
- Runs stages serially.
- Stops the queue after any nonzero exit or validation failure.
- Requires explicit human approval before expensive/scientifically consequential stages.
- Stores machine-readable state and progress history.
- Supports configured resume arguments, but they are intentionally not enabled in the supplied Benettin manifests until the project checkpoints both trajectories and the Benettin accumulator safely.

## Install

Copy or unzip this directory into the project, then:

```bash
cd /home/peacelovephysics/ephemeris
source .venv/bin/activate
pip install -e ./ephemeris_experiment_runner
```

Optional process-monitoring dependency:

```bash
pip install -e './ephemeris_experiment_runner[monitor]'
```

The runner itself uses the Linux `ps` command, so the optional dependency is not required under WSL.

## Preflight

```bash
cd /home/peacelovephysics/ephemeris
source .venv/bin/activate

python -m mini_ephemeris.long_term_stability_cli --help | grep -E \
  'full_with_pluto|with-lyapunov|lyapunov-method|rebound-simulationarchive'

ephem-exp plan ephemeris_experiment_runner/manifests/01_newtonian_benettin_smoke.json
ephem-exp run ephemeris_experiment_runner/manifests/01_newtonian_benettin_smoke.json --dry-run
```

The first real command should be the short smoke manifest, not the GR template.

## Start the smoke ladder in tmux

```bash
tmux new -s final_benettin
```

Inside tmux:

```bash
cd /home/peacelovephysics/ephemeris
source .venv/bin/activate

ephem-exp run \
  ephemeris_experiment_runner/manifests/01_newtonian_benettin_smoke.json
```

Detach with `Ctrl-b`, then `d`.

A typical update looks like:

```text
[progress] full_pluto_10myr_seed12345: 37.00% | 3.700/10.000 Myr |
elapsed 2h 41m | rate 1.38 Myr/h | ETA 4h 34m | CPU 99.7% |
RSS 612.4 MiB | source full_with_pluto_newtonian_10myr_seed12345.bin
```

ETA is based on recent checkpoint-to-checkpoint throughput. It will be noisy at the beginning and after a restart.

## Check status from another terminal

```bash
ephem-exp status \
  ephemeris_experiment_runner/manifests/01_newtonian_benettin_smoke.json
```

Or watch:

```bash
ephem-exp status \
  ephemeris_experiment_runner/manifests/01_newtonian_benettin_smoke.json \
  --watch --interval 60
```

## Approval gates

The calibration and GR manifests intentionally stop before costly stages. After reviewing the previous output:

```bash
ephem-exp approve \
  ephemeris_experiment_runner/manifests/02_newtonian_benettin_calibration.json \
  full_pluto_100myr_seed12345

ephem-exp run \
  ephemeris_experiment_runner/manifests/02_newtonian_benettin_calibration.json \
  --resume
```

Approval records permission to run one stage; it does not start anything itself.

## Failure behavior

The runner blocks downstream stages when:

- the process exits nonzero;
- a required summary/diagnostic file is missing;
- the selected CSV contains NUL bytes;
- simulated times are duplicated or nonmonotonic;
- the final simulated time does not reach the configured target;
- a configured scientific metric falls outside its allowed range.

The supplied manifests initially use structural integrity gates because the exact two-trajectory summary schema needs to be confirmed. After the worker writes stable metric keys, add `json_metrics` gates for LCN, invariant drift, and classification.

## Stalls and restarts

The runner warns when simulated time has not advanced for the configured period. It does not automatically kill a high-CPU calculation. This avoids destroying a legitimate long integration because an archive write was delayed.

After a hard reboot, a stale `runner.lock` may remain. Confirm no runner exists before removing it:

```bash
pgrep -af 'ephem-exp|long_term_stability_cli'
rm /home/peacelovephysics/ephemeris/output/stability/final_benettin/runner_state/<experiment>/runner.lock
```

Do not enable automatic Benettin resume using only the reference trajectory's SimulationArchive. A scientifically valid resume must restore both trajectories, the accumulated logarithmic growth, the fit-window state, and the last renormalized deviation. Use `CODEX_INTEGRATION_PROMPT.md` to add that support.

## Included files

- `manifests/01_newtonian_benettin_smoke.json` — short integrable and false-positive controls.
- `manifests/02_newtonian_benettin_calibration.json` — 100/200 Myr Newtonian calibration with manual gates.
- `manifests/03_gr_physics_ladder.template.json` — 1/10/100 Myr GR ladder; do not approve until validation is complete.
- `PHYSICS_PLAN.md` — recommended final physics hierarchy and stopping rules.
- `CODEX_INTEGRATION_PROMPT.md` — requirements for safe dual-trajectory checkpoints, incremental progress, and GR validation.

## Recommended immediate action

1. Install the runner.
2. Give Codex `CODEX_INTEGRATION_PROMPT.md` so the Benettin worker gains safe incremental output and dual-trajectory checkpoint/resume.
3. Run the smoke manifest.
4. Review it before approving the 100 Myr calibration.
5. Do not launch the GR template until Newtonian Benettin agrees broadly with native Newtonian MEGNO.
