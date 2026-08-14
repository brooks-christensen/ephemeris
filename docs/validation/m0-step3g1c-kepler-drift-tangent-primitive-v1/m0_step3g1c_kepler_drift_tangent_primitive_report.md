# M0 Step 3g1c Kepler Drift and Tangent Primitive

Final status: **STEP3G1C_KEPLER_DRIFT_TANGENT_COMPLETE**

Primary finding: **BOUND_ELLIPTIC_KEPLER_DRIFT_TANGENT_QUALIFIED**

Verification envelope: **ISOLATED_TWO_BODY_BOUND_ELLIPTIC_MAP_ONLY_NO_NBODY_DYNAMICS**

## Result

The immutable fixed-mass pair plan, exact bound-elliptic universal-variable flow,
and analytic initial-state tangent map passed every frozen physical, tangent,
solver, ownership, determinism, failure, safety, and integrity gate.

The campaign contains 73 pre-artifact nodes and 6 artifact nodes. The model
fingerprint is `97f6001ca3e19fdc3945c9e61667e5873ee63b0d90e48a92f7f8c52887797664`; the pair-plan fingerprint is
`76db4ed5663e9d9e96b0a2927c938df298c0ab45fadc65c3dd06fe4f50f1c127`.

## Numerical Summary

- Physical cases: 63; worst scaled error:
  2.3794873993006882e-13.
- Invariant cases: 30; worst normalized drift:
  6.021559238433838e-12.
- Symplectic matrices: 20; forward/reverse
  max residuals: 6.0885862714781705e-08 /
  2.2096541182232447e-08.
- Composition maximum scaled error:
  1.1274756047046615e-15.
- Solver cases: 63; maximum iterations:
  6; branch counts:
  {"elliptic_newton": 30, "elliptic_quartic": 26, "zero_duration": 7}.
- Forward finite-difference minimum:
  5.2089478047808886e-08;
  central minimum:
  5.092607121401355e-11.

## Scope

- Only an isolated two-body bound-elliptic map was evaluated.
- No protected force/JVP provider was invoked.
- No N-body dynamics, map, or trajectory was executed.
- No interaction kick, lazy kernel, corrector, or WHCKL composition exists yet.
- No MEGNO, LCN, or Solar-System result is qualified.
- Tangent qualification covers canonical initial-state derivatives only, at
  fixed masses, fixed parameters, and fixed duration.

No center-of-mass drift, multi-pair composition, synchronization, archive,
REBOUND, or REBOUNDx operation was executed. No qualified prior file or
protected/historical artifact changed. The claim remains limited to the frozen
isolated bound-elliptic domain. Step 3g1d may be proposed only for a synthetic
analytic interaction-kick force/JVP map; it is not implemented here.
