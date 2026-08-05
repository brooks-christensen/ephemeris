# Compiled-C GR Tangent Backend

The C backend ports the frozen `gr-tangent-python-oracle-v2` `gr_potential`
acceleration and analytic position-Jacobian action. REBOUND invokes the exported
C callback directly through `reb_simulation.additional_forces`. Python is used
only to build and load the shared object, attach it once, sample diagnostics,
and write output.

Per-simulation configuration and instrumentation live in
`reb_simulation.extras` and are released by a C `extras_cleanup` callback. The
implementation has no mutable process-global state. The loader verifies source,
artifact, installed-header, REBOUND-version, structure-size, and field-offset
metadata before attachment.

The compiler command uses `-Wall -Wextra -Wpedantic -Werror`,
`-fno-fast-math`, and `-ffp-contract=off`.

## Reproducible Commands

Run from `/home/peacelovephysics/ephemeris`.

Clean forced build and ABI check:

```bash
PYTHONPATH=mini_ephemeris/src \
  .venv/bin/python -m mini_ephemeris.gr_potential_tangent_c_build --force
```

Focused tests:

```bash
PYTHONPATH=mini_ephemeris/src \
  .venv/bin/python -m unittest discover -s mini_ephemeris/tests \
  -p 'test_gr_potential_tangent*.py' -v

PYTHONPATH=mini_ephemeris/src \
  .venv/bin/python -m unittest discover -s mini_ephemeris/tests \
  -p 'test_rebound_gr_tangent_backend_cli.py' -v
```

Pointwise gates:

```bash
PYTHONPATH=mini_ephemeris/src .venv/bin/python \
  -m mini_ephemeris.gr_tangent_c_validation \
  --kernel-path data/de431_part-2.bsp \
  --output-dir /tmp/gr-tangent-c-pointwise pointwise-acceleration

PYTHONPATH=mini_ephemeris/src .venv/bin/python \
  -m mini_ephemeris.gr_tangent_c_validation \
  --kernel-path data/de431_part-2.bsp \
  --output-dir /tmp/gr-tangent-c-jacobian pointwise-jacobian
```

Short dynamics:

```bash
PYTHONPATH=mini_ephemeris/src .venv/bin/python \
  -m mini_ephemeris.gr_tangent_c_validation \
  --kernel-path data/de431_part-2.bsp \
  --output-dir /tmp/gr-tangent-c-short short-dynamic
```

Fresh-process restart gate:

```bash
PYTHONPATH=mini_ephemeris/src .venv/bin/python \
  -m mini_ephemeris.gr_tangent_c_validation \
  --kernel-path data/de431_part-2.bsp \
  --output-dir /tmp/gr-tangent-c-restart \
  --restart-duration-years 20 restart-equivalence
```

Matched production benchmark:

```bash
PYTHONPATH=mini_ephemeris/src .venv/bin/python \
  -m mini_ephemeris.gr_tangent_c_validation \
  --kernel-path data/de431_part-2.bsp \
  --output-dir /tmp/gr-tangent-c-benchmark benchmark
```

Plan and run the complete capped matrix:

```bash
PYTHONPATH=ephemeris_experiment_runner/src \
  .venv/bin/python -m ephemeris_experiments.cli plan \
  ephemeris_experiment_runner/manifests/08_gr_tangent_c_port_validation.json

PYTHONPATH=ephemeris_experiment_runner/src \
  .venv/bin/python -m ephemeris_experiments.cli run \
  ephemeris_experiment_runner/manifests/08_gr_tangent_c_port_validation.json
```

The hardened runner rejects tagged-output collisions by default. Use
`--skip-if-complete`, `--overwrite-existing-output`, and `--resume` only for
their distinct documented meanings. A duration above the 100 kyr validation
cap additionally requires the explicit `--production-duration-approved` flag;
the validation manifest forbids that flag.
