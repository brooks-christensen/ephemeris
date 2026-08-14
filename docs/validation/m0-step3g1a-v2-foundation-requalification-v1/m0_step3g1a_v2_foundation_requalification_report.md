# M0 Step 3g1a V2 Foundation Requalification Report

## Result

- Final status: `STEP3G1A_REQUALIFICATION_COMPLETE`
- Primary finding: `V2_FOUNDATION_REQUALIFIED_READY_FOR_PRIMITIVES`
- Verification envelope: `FOUNDATION_ONLY_REQUALIFIED_WITHOUT_DYNAMICS_OR_PROTECTED_KERNEL_EVALUATION`
- Preregistration commit: `b7baa44efd8da5ae1d2c49e6f7ca4fd848d469e7`

## Historical Record

Manifest 22 remains permanently `STEP3G1A_V2_FOUNDATION_INCOMPLETE` with `V2_FOUNDATION_NOT_READY`. Its inherited wildcard selected 19 Step 3g0 tests, imported historical and protected kernel harnesses, and statically evaluated protected Python and compiled-C physical force/JVP paths. Those results are historical provenance and contribute no evidence to this requalification.

The prior calls operated on static arrays, the no-integration guard intercepted timestep entrypoints, archive access was read-only, and regeneration either used temporary destinations or reproduced byte-identical compact artifacts, changing no committed implementation or historical artifact bytes. That explains why the prior campaign violated scope without executing dynamics or mutating the v2 foundation.

## Requalification Evidence

Manifest 23 selected 26 byte-frozen historical foundation nodes, 10 requalification integrity nodes, and 6 artifact nodes by literal pytest node ID. The static gate audited the complete direct import graph and active subprocess closure before pytest. A synthetic package shell prevented the legacy `mini_ephemeris.__init__` and `mini_ephemeris.nbody` import. The deny-first guard, strict allowlist, `ctypes.dlopen` audit hook, and `/proc/self/maps` checks stayed clean, including guarded fresh interpreters.

All model, state, synthetic force/JVP, exact timebase, ownership, accounting, isolation, and both Step 3g0 high-finding contracts passed. A historical test that constructed but did not assert a reordered canonical tangent rejection was supplemented by the exact requalification-only node `test_canonical_layout_mismatch_is_rejected`; no historical test or v2 implementation file changed.

## Source Review

The byte-exact foundation was reviewed for nested mutability, writable NumPy aliases, body-ID/index conflation, velocity/momentum ambiguity, nondeterministic serialization, hidden globals/imports, live observer aliases, force/JVP accounting effects, weak assertions, missing requirement coverage, and claims beyond evidence. No material implementation defect remains within the foundation-only contract. Public immutable results still allocate, and optimized caller-owned buffers remain a future design gate.

## Evidence Boundary

No physical force model, Jacobi transform, integrator primitive, WHCKL map, tangent map, MEGNO/LCN calculation, timestep, IAS15 run, Solar-System trajectory, or archive creation was dynamically validated or executed in v2. Protected sources were read only for hashing and static review; no protected Python kernel, compiled-C tangent library, REBOUND, or REBOUNDx module was imported, loaded, called, or evaluated.

## Successor

Step 3g1b is justified only as a separately preregistered implementation and verification of isolated inertial/Jacobi coordinate transforms and their canonical tangent maps. Kepler, kick, lazy kernel, corrector, WHCKL composition, diagnostics, trajectories, and archives remain out of scope.
