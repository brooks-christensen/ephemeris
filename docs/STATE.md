# Project state

**Read this file first and update it when a unit of work closes.**

Last updated: 2026-08-21

---

## Where things stand

| Item | State |
|---|---|
| Main line | `v2-whckl-tangent-core` @ `2af28a0` |
| Chaos-estimator series | `repair/chaos-estimators` @ `0edb9ea`, pushed |
| Active Codex work | `codex/rung3-real-ephemeris`, based on `0edb9ea` |
| Restore point | tag `review-baseline-2026-08-19` |
| Controlling plan | `docs/PLAN.md`; manifests 23-28 are superseded |
| Validation ladder | rungs 0, 1, 2a, and 2b PASS |
| Rung 3 | **FAIL; not scientifically converged or qualified** |
| Rungs 4-5 | not run |

The superseded Manifest 28 campaign and its provenance dispute are historical
context, not an open qualification path. Do not restore its pass status or use
it to certify the current pipeline.

---

## Rung 3 result

### Fixed configuration

- System: GM-weighted composite of Sun, Mercury, Venus, Earth-Moon barycenter,
  and Mars; Jupiter, Saturn, Uranus, Neptune, and Pluto barycenters.
- Epoch/frame: J2000 TT JD 2451545.0, Skyfield ICRF barycentric.
- State kernel: `data/de440s.bsp`.
- Kernel SHA-256:
  `c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2`.
- Masses: `mini_ephemeris.ephem` DE431 GM constants, explicitly recorded.
- Physical fingerprint:
  `7a06fd788272a0b9f430340b4a75e42627c153cd7970b0d54d37b0ddbd6fb1f3`.
- REBOUND: 4.6.0, WHFast.
- Tangent seeds: 12345, 23456, 34567, 45678, 56789.
- Coarse lane: 400 Myr, dt 0.4 yr, with an exact 200 Myr comparison anchor.
- Fine lane: 200 Myr, dt 0.2 yr.
- Samples: 2,000 per seed at deterministic golden-ratio-dithered times.
- Acceptance window: 10-40 Myr, unchanged.
- Existing energy limit: 1e-9, retained without weakening. The 1e-7 limit in
  `docs/PLAN.md` also fails at dt 0.2.

### Resonance precondition

The original linear wrap falsely measured libration centered near 180 degrees
as a nearly 360-degree circulation. The corrected minimum circular covering arc
gives:

| Lane | Span | Center | Minimum Pluto-Neptune separation |
|---|---:|---:|---:|
| dt 0.4 yr | 167.005892 deg | 185.333139 deg | 17.270780 AU |
| dt 0.2 yr | 167.063943 deg | 185.363300 deg | 17.268909 AU |

Both 300 kyr preflights pass. No Lyapunov result comes from an unprotected
Pluto.

### Artifacts

| File | SHA-256 | Status |
|---|---|---|
| `rung3-dt0.4.json` | `b067a5883173634b639d2bcb98e3c7fd21f5c338769f2b126210c66a32b8b934` | FAIL |
| `rung3-dt0.2.json` | `14d9d864bd3ef6ee689e31c6ed2f2f805bc8f61c43a60aef1fe950aca36aee46` | FAIL |

Both parse as strict JSON. The fine artifact records and verifies the coarse
artifact hash, physical fingerprint, seeds, 2:1 timestep ratio, and equal
200 Myr comparison duration.

### Measurements

| Seed | Coarse 400 Myr T_L | Coarse 200 Myr T_L | Fine 200 Myr T_L | Fine halving | Fine estimator disagreement | Benettin dt change | MEGNO dt change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12345 | 8.85850 Myr | 6.49779 Myr | 5.52071 Myr | 0.69353 | 40.06% | 15.04% | 30.94% |
| 23456 | 9.62697 Myr | 7.35950 Myr | 5.59665 Myr | 0.69798 | 37.20% | 23.95% | 61.11% |
| 34567 | 8.66071 Myr | 6.28714 Myr | 5.42052 Myr | 0.68858 | 42.12% | 13.78% | 28.03% |
| 45678 | 8.86490 Myr | 6.50467 Myr | 5.58029 Myr | 0.69654 | 38.65% | 14.21% | 30.73% |
| 56789 | 8.78411 Myr | 6.41804 Myr | 5.46510 Myr | 0.69118 | 40.87% | 14.85% | 29.19% |

Aggregate coarse 400 Myr:

- Median T_L: 8.85850 Myr.
- Tangent lambda range: 1.038749e-7 to 1.154640e-7 /yr.
- Halving range: 0.72594 to 0.76447.
- Energy drift: 1.038011e-6.
- All five classifications: `ambiguous`.

Aggregate fine 200 Myr:

- Median T_L: 5.52071 Myr.
- Tangent lambda range: 1.786783e-7 to 1.844842e-7 /yr.
- Halving range: 0.68858 to 0.69798.
- Energy drift: 2.596967e-7.
- All five classifications: `regular_likely`.

The fine energy error is 3.997 times smaller than coarse, consistent with an
approximately second-order timestep response, but it remains above both 1e-9
and 1e-7. Every seed fails the 10% timestep condition for both diagnostics.
Every seed also fails the halving and Benettin/MEGNO agreement conditions.
No tangent saturation was observed; maximum absolute log norm was 49.09 coarse
and 38.54 fine against the fixed limit of 300.

**Disposition: rung 3 FAIL.** The apparent 5-10 Myr times are not reportable
Lyapunov times because the running estimates are not duration-converged, the two
diagnostics disagree, timestep convergence fails, and the energy gate fails.
Do not compare them to the published approximately 20 Myr value as measurements
of the same quantity.

### Runtime

- Coarse five-seed wall time: 130.75 minutes.
- Fine five-seed wall time: 126.61 minutes.
- Available disk before launch: 777 GiB.
- No nonfinite or process failures occurred.

---

## Required next work

1. **Independent review before merge.** The author of the rung-3 harness does
   not certify it. Review the DE440s frame/mass folding, circular preflight,
   exact-time sampling, five-seed aggregation, same-duration dt formula, and
   all result consumers.
2. **Persist compact sampled histories before another long campaign.** The two
   JSON artifacts preserve per-seed summaries but not the sampled tangent,
   MEGNO, and energy series. Therefore the completed FAIL is auditable at the
   gate-output level but its slopes and halving values cannot be independently
   recomputed offline. Also move all JSON/progress collision checks before
   integration and add restart-safe checkpoints; the current JSON writer rejects
   overwrite only at final write and progress sidecars can overwrite. This run
   was manually verified collision-free. Do not reinterpret the current result;
   repair the apparatus first and rerun from clean only under a new campaign.
3. **Diagnose non-asymptotic growth before requesting more duration.** Use a
   bounded known-answer outer-system control and stored histories to separate
   finite transients from genuine exponential growth. Do not read S(T)/T alone.
4. **Resolve the energy ladder separately.** The observed factor-of-four
   improvement suggests a bounded dt ladder can determine the timestep needed
   for the 1e-7 screen. Do not weaken the inherited 1e-9 gate after observation.
5. **Do not run rungs 4 or 5.** Rung 3 has not passed.

---

## Estimator foundation

The installed REBOUND 4.6.0 convention was measured, not assumed:

- `sim.megno()` is the time-averaged mean MEGNO.
- `sim.lyapunov()` equals the OLS slope of mean MEGNO and is lambda/2.
- The conversion `lambda = 2 * d<Y>/dt` is correct for this runtime.
- `sim.lyapunov()` is not used as an independent estimator.

Rungs 0-2 currently pass:

- Rung 0: 93 tests, zero failures/errors.
- Rung 1: integrable two-body classified regular, halving 0.534082.
- Rung 2a: cat map relative lambda error 8.40e-6.
- Rung 2b: standard map relative lambda error 7.16e-4.

---

## Known issues outside the current rung

- `advanced_integrators.py` still contains a sign-flipped EIH 1PN path and
  missing EIH terms. This path is not part of rung 3.
- Velocity Verlet remains unsuitable for velocity-dependent 1PN acceleration.
- The old Manifest 23-28 qualification/reporting machinery contains inferred
  pass statuses and duplicated apparatus. It is superseded by `docs/PLAN.md`.
- Rungs 4 and 5 remain unimplemented and unrun.

---

## Verification commands for this unit

- `python3 scripts/check_undefined_names.py --self-test`
- `python3 scripts/check_undefined_names.py scripts/ladder_rung3_pluto.py mini_ephemeris/tests/test_ladder_rung3_pluto.py`
- `python3 scripts/run_validation_ladder.py --self-test`
- `python3 scripts/run_validation_ladder.py`
- `env PYTHONPATH=mini_ephemeris/src .venv/bin/python scripts/measure_megno_convention.py`
- `env PYTHONPATH=mini_ephemeris/src .venv/bin/python mini_ephemeris/tests/test_ladder_rung3_pluto.py -v`
- `env PYTHONPATH=mini_ephemeris/src .venv/bin/python -m py_compile scripts/ladder_rung3_pluto.py`
- `python3 -m json.tool rung3-dt0.4.json`
- `python3 -m json.tool rung3-dt0.2.json`
- `env PYTHONPATH=mini_ephemeris/src .venv/bin/python -m unittest discover -s mini_ephemeris/tests -p test_*.py`

The repository virtualenv does not include pytest; no dependency was installed.
Full unittest discovery ran 402 tests and is not green: 22 failures and 15 errors
come from superseded Manifest 23-28 branch/hash/artifact assumptions and
pre-existing v2 public-docstring checks. The rung-3 focused suite is 14/14 and
rung 0's enrolled suite is 93/93.
