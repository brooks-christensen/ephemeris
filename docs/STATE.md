# Project state

**Update this file when you finish a unit of work.** It is the source of truth
for where the project is. Both agents read it first.

Last updated: 2026-08-20

---

## Where things stand

| | |
|---|---|
| Main line | `v2-whckl-tangent-core` @ `2af28a0` |
| Active work | `repair/chaos-estimators` @ `2cc0c24` (3 commits, pushed, 43 tests green) |
| Restore point | tag `review-baseline-2026-08-19` |
| Step 3g1d | **not closed.** Manifest 28 campaign passed 124/124 but disposition is undecided — see below |

Throwaway branches `wip/manifest28-review` and `wip/manifest28-rerun` are review
snapshots, not results. Delete after the Manifest 28 closeout.

---

## Open items, highest value first

### 1. MEGNO convention — one measurement away

**Owner: Codex** (needs REBOUND). The conversion from MEGNO slope to Lyapunov
exponent used **0.5**, wrong under both conventions (2.0 for time-averaged ⟨Y⟩,
1.0 for instantaneous Y). Now uses 2.0, flagged `megno_convention_assumed:
"mean_Y"` in the output.

`calibrate_megno.py` settles it. First run failed usefully: final MEGNO = 1.9988,
i.e. ⟨Y⟩ → 2, i.e. the test system was **regular**, so the slope was residual
approach-to-2 and the implied factor (10.99) was noise over noise. The a₂ = 1.35
setup sits at 3.41 mutual Hill radii, just inside the 2√3 ≈ 3.46 threshold —
marginally chaotic with a Lyapunov time far longer than the run. Revised script
scans a₂ ∈ {1.25, 1.20, 1.15} and refuses to calibrate unless final ⟨Y⟩ ≥ 5.

**Action:** re-run, record the factor and the REBOUND version.

### 2. Manifest 28 disposition — needs Brooks's decision

The campaign passed 124/124 clean. Two reasons not to close it COMPLETE:

**(a) The preregistered command is unsatisfiable.** `generate_artifacts` runs
`python -m mini_ephemeris.m0_step3g1d_reporting`, which fails with
`RuntimeError: forbidden modules are loaded: ['mini_ephemeris.nbody']` —
`__init__.py` eagerly imports `nbody`, and `__init__.py` is frozen under
Manifest 23. Manifest 28 requires an action it also forbids. Artifacts were
generated through a `sys.modules` shim.

**(b) The gate that failed was rewritten 16 minutes later.**

```
11:28:08  ffe6475  Record failed Step 3g1d corrective campaign
11:44:15  e979c00  Correct affine finite-difference gate applicability
11:48:09  1713a2e  Preregister Step 3g1d requalification
```

Manifest 27 recorded one failure: `finite_difference.dense.acceptance = false`.
`e979c00` removed `_ladder_acceptance()` and added `AFFINE_EXACT`, which exempts
that fixture. Manifest 28's `parent_commit` is `e979c00`. Recomputing today gives
values **bit-identical to the ones that failed**, now passing. Manifest 28 states
the classification came from "the fixture mathematical definition… never
observed ladder values."

The mathematics is correct — for an affine map the central difference is exact,
so there is no U-curve to demand. The provenance claim is not.

**Manifest 26 precedent** for the same species of problem:

```
Final status:          STEP3G1D_BLOCKED
Verification envelope: NOT_ESTABLISHED
Blocking condition:    MANIFEST26_PROVENANCE_HASH_TABLE_INVALID
```

with its passing tests recorded as "nonqualifying diagnostic evidence only".

**Recommendation:** close BLOCKED the same way, disclose (b), record 124/124 as
diagnostic evidence for a successor.

### 3. Test artifacts stamp PASS without running tests

`m0_step3g1d_reporting.py:68-75` builds the 124-node inventory by transcribing
the manifest and stamping a literal `"result": "PASS"`. `_traceability_csv` does
the same. `generate_artifacts()` invokes no pytest. **The artifact would look
identical if every test failed.** Either run the tests and record real results,
or delete the artifact.

### 4. No qualification module can emit a failure

`FINAL_STATUS` and `PRIMARY_FINDING` are constants set to the COMPLETE values.
`STEP3G1D_REQUALIFICATION_FAILED`, `..._BLOCKED`, `..._NOT_EVALUATED`,
`SYNTHETIC_..._NOT_QUALIFIED` appear in **zero** `.py` files. An apparatus that
can only emit success is not a gate.

### 5. EIH 1PN acceleration is sign-flipped

`advanced_integrators.py:846-863`. Every sign in the β=γ=1 bracket is reversed;
Mercury precesses backwards (−4298 vs +4301 arcsec/century). Reached only via
`experiments.py:266,528`. **Decide whether `experiments.py` is live** before
investing in a fix. Two further EIH terms are also missing.

### 6. Structural re-baseline

The freeze is a byte hash over files with three roles — scientific subject (v2
kernels), verification apparatus (harness, reporting), packaging plumbing
(`__init__.py`). Any defect in any of them needs a whole new manifest to fix.
Four of the last seven manifests are repairs; 3g1d alone has consumed 26, 27, 28.

`__init__.py` is the clean case: one unused re-export with **zero consumers
anywhere in the repo**, forcing a forbidden module into `sys.modules` on every
import, frozen so Manifest 28 cannot fix what makes Manifest 28 impossible.

Proposal: tier the freeze. Subject byte-frozen; apparatus repairable under a
declared procedure that forces a re-run from clean; plumbing governed by a
machine-checkable rule. Acceptance criteria stay frozen either way.

### 7. Smaller, confirmed

- **Velocity-Verlet is not symplectic with velocity-dependent GR.**
  `acceleration_newtonian_gr_sun` is velocity-dependent; the step evaluates the
  new acceleration at `v_half`. Max energy error over 200/800/3200 orbits:
  Newtonian **1.00, 1.00** (bounded); with GR **1.12, 1.42** (growing).
  Magnitude at 100 Myr is *not* established — do not extrapolate.
- **λ needs a timestep-convergence ladder.** Under-resolution manufactures a
  positive exponent that no post-processing can detect: at 1,000 steps/orbit on
  an integrable system the growth curve fits a line with R² = 0.997.
  `ENERGY_DRIFT_CHAOS_GATE = 1e-7` is a backstop, not a substitute.
- **Unreachable bugfix.** The mismatch comprehension in `verify_inherited_integrity`
  was fixed in `3g1d` but not `3g1b`/`3g1c`; since `1d` delegates to `1c` → `1b`,
  the fix never runs. One line.
- **`acceptance_gates` blocks are never read** by any code; all entries `true`,
  including in Manifest 26 (BLOCKED) and 27 (FAILED).
- **Determinism gates in 3g1b/3g1c** run seed 0 twice where the manifests require
  1/C and 8675309/C.UTF-8.
- **20 of 23 shared function names** across the four qualification modules have
  diverged.

---

## Verified correct — do not re-litigate

- **v2 core**: Kepler tangent symplectic to 9.84e-12, cross-checked against an
  independent f&g propagator to 9.85e-15; Jacobi round-trip 9.08e-17; kick
  reversibility **bitwise exact**; 1,056 randomized/boundary cases, zero failures.
- **GR potential path** (`gr_potential_tangent.py` + C port): Mercury at
  **42.9808 vs 42.98** arcsec/century; Jacobian verified; C port agrees with
  Python to **5.6e-16** across 8 orders of magnitude in separation.
- **Newtonian integrators**: velocity-Verlet exactly 2nd order, RK4 exactly 4th,
  energy bounded to the digit over 3200 orbits.
- **Benettin core**: cadence, log accumulation, time normalization correct;
  variational equations momentum-conserving.
- **MEGNO-lite**: recovers λ to five significant figures.
- **Hash ledger**: 669 pins audited; 380 verified equal; the 12 wrong values are
  all Manifest 26 and **did not propagate**.

---

## Fixed this cycle (branch `repair/chaos-estimators`)

Four defects that were producing wrong scientific results:

1. **Lyapunov line-fit measured ln(t).** Reported T_lyap was 0.35 × run length
   regardless of dynamics — a 100 Myr run would report ~35 Myr from regular
   motion. Replaced with the running estimate plus a halving-ratio discriminator.
2. **LCN classifier had no power.** `lcn * elapsed > 1.0` reduces to "grew by one
   e-fold"; the statistic was 5.19 at the first sample on an integrable system.
3. **Shadow fit included the saturated tail.** 5.372 Myr → 2.001 Myr against a
   true 2.000 once saturation is excluded.
4. **MEGNO factor 0.5** → named constant + calibration hook (see item 1).

Plus: `best_megno_slope_fallback` max() → median; `analysis_tools.lyapunov_max`
barycenter and norm fixed, with a warning on negative returns; a sign guard in
the discriminator (two negative estimates gave a positive halving ratio and read
as chaotic).

New: `chaos_estimator_diagnostics.py`, 43 regression tests, ~2 seconds, no
REBOUND needed.
