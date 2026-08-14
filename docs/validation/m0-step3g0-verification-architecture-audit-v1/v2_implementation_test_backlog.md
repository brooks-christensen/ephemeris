# V2 Implementation and Test Backlog

## Gate 0: repository and contracts

- Create a new package namespace; do not modify protected M0 files.
- Freeze model/config/layout/unit schemas and canonical hashing.
- Implement `BodyId`, immutable `CompiledLayout`, typed SI boundaries, and integer timebase.
- Add a process-wide test guard that forbids trajectories in primitive-test jobs.
- Acceptance: serialization round-trip, body-swap mutant rejection, fractional-target rejection, and deterministic hashes.

## Gate 1: pure force/JVP

- Implement Newtonian and `gr_potential` pure kernels behind one force/JVP protocol.
- Enforce finite inputs, positive masses/constants, no coincidence, shape/layout/unit identity, and nonalias contracts.
- Preserve central reaction and force closure.
- Acceptance: decimal, complex-step where valid, finite-difference, covariance, linearity, zero-limit, no-backreaction, Python/C equality, strict warnings, ASan/UBSan.
- Mutants that must fail: sign, `r^-3`/`r^-5` confusion, central-response omission, body-index swap, tangent/physical array swap.

## Gate 2: coordinate primitives

- Implement immutable-mass inertial/Jacobi position and canonical momentum transforms with matching tangent transforms.
- Acceptance: forward/inverse round trip, linearity, finite-difference derivative closure, translation/boost behavior, and canonical mass convention.

## Gate 3: Kepler and basic kick

- Wrap or transcribe the audited WHFast universal-variable Kepler primitive and first variation.
- Implement the interaction/jump kick with exact coordinate ordering.
- Acceptance: analytic two-body states/Jacobian, one-step finite-difference closure, reversibility, determinant behavior, and canonical symplecticity.

## Gate 4: lazy kernel

- Implement the private `q_shift=q+h^2*A(q)/12` state and its JVP chain rule.
- Evaluate force/JVP at unshifted and shifted positions with distinct contexts.
- Restore live `q,dq` exactly after the kick.
- Acceptance: direct AD/test-map oracle, centered finite differences across step scales, tangent linearity, canonical symplecticity, exact evaluation counts, and no Hessian dependency.

## Gate 5: order-17 corrector

- Encode the 16 `Z(a,b)` stages from a source-locked coefficient fixture.
- Differentiate every Kepler, transform, and kick operation in identical order.
- Implement inverse by the exact source schedule.
- Acceptance: 32 force/JVP calls per application, forward/inverse closure, finite-difference tangent closure, canonical symplecticity, and coefficient/sign mutants killed.

## Gate 6: full WHCKL map

- Compose drift, jump, lazy kick, and corrector under one live-map owner.
- Implement `safe_mode=0`, `keep_unsynchronized=1` semantics explicitly.
- Acceptance: complete `D Phi` finite-difference closure, `D Phi^T Omega D Phi-Omega`, tangent linearity, determinant, reversibility, convergence order, and stable floating-point operation order.

## Gate 7: observer copy and counters

- Add synchronized inertial snapshot construction on a private copy.
- Add separate live, JVP, corrector, diagnostic, offline, and restart counters.
- Implement robust angle, energy, angular momentum, and element observers.
- Acceptance: live canonical bytes and counters unchanged by observation; context totals close exactly; `atan2` and chord agree over conditioning suite.

## Gate 8: checkpoint/restart

- Serialize canonical physical/tangent state, corrector/synchronization flags, integer scheduler, counters, and sample ledger atomically.
- Rebind kernels by immutable IDs in a fresh process without evaluation.
- Acceptance: exact restart before/after lazy, corrector, output, and rescale boundaries; no missing/duplicate sample; schema/fingerprint corruption rejected.

## Gate 9: tangent rescaling

- Add deterministic integer-tick rescaling and accumulated log scale.
- Acceptance: rescaled/unrescaled tangent equivalence, exact restart, sign/normalization metadata, and scale-independent finite-time growth.

## Gate 10: MEGNO and LCN

- Only after Gates 0-9, preregister a discrete growth/MEGNO/LCN definition and fit interval.
- Keep analysis outside force kernels and live synchronization.
- Acceptance: integrable analytic controls, known chaotic maps, same-map restart equality, timestep convergence, and no observer interference.

## Gate 11: bounded trajectory qualification

- Only after all primitive gates pass, preregister the smallest two-body and compact planetary trajectories.
- No Solar-System trajectory before analytic, finite-difference, canonical-symplecticity, reversibility, and restart unit gates pass.
- No production timestep, Stage 4, or 10 Myr action is implied by this backlog.

## Publication evidence package

- Mathematical operator graph and source-version fixture.
- Requirements-to-tests traceability and mutation score.
- Canonical symplecticity and discrete-map closure results.
- Exact counter and restart ledger.
- Benchmark split by physical force, JVP, lazy, corrector, observer, and serialization contexts.
- Reproducible model/ephemeris/configuration provenance and explicit relativity scope.
