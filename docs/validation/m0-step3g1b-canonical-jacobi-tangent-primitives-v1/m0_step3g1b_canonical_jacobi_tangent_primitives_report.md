# M0 Step 3g1b Canonical Jacobi Tangent Primitives Report

## Result

- Final status: `STEP3G1B_JACOBI_TANGENT_PRIMITIVES_COMPLETE`
- Primary finding: `CANONICAL_JACOBI_TRANSFORMS_QUALIFIED_FOR_PRIMITIVE_COMPOSITION`
- Verification envelope: `COORDINATE_TRANSFORM_ONLY_NO_DYNAMICS_EXECUTED`
- Preregistration commit: `20d156839da5bed8d966b3a64b42ee73d7db788f`

## Convention And Implementation

The qualified operator retains the center-of-mass pair and maps fixed-mass inertial canonical `(x,p)` to Jacobi `(q,P)` with `P=A^(-T)p`. Body order and central-first identity are explicit. Production applies fixed left-to-right binary64 O(N) recurrences; dense matrices appear only in independent qualification code.

The Step 3g0 `(x,v)` wording remains a future adapter boundary. This step introduces no velocity-based Jacobi coordinate, no velocity-to-momentum conversion, and no claim that the complete REBOUND compatibility adapter is qualified.

## Numerical Evidence

The synthetic five-body mass range is 0.125 to 32 kg, ratio 256. `cond2(A)` is 5.0242795825410962; `cond2(S)` is 6.1993684124043602.

Maximum state forward/inverse round-trip component error is 1.7763568394002505e-15; maximum tangent forward/inverse component error is 4.4408920985006262e-16. The finite-difference ladder minimum relative L2 error is 8.6469450802247866e-15 against bound 7.047865906775269e-13. Nonmonotonic behavior after cancellation dominates is permitted exactly as preregistered.

Forward symplectic residuals are max 1.1102230246251565e-16, Frobenius 4.6248953598161032e-16, and scaled 1.3620524684437401e-17. Inverse residuals are max 5.5581360116065201e-17, Frobenius 1.8102927449330055e-16, and scaled 5.3313934911163855e-18. The reported determinant 1 is secondary only.

## Safety And Evidence Boundary

All tests use analytic or explicitly synthetic data. No Solar-System state is used. No physical force or JVP was evaluated. No dynamical map was implemented. No integration or timestep occurred. REBOUND and REBOUNDx were not imported. Protected and historical inputs remained byte exact.

Symplecticity applies only to this fixed-mass coordinate transformation. This result does not qualify a Kepler drift, kick, lazy kernel, corrector, WHCKL kernel, tangent evolution, MEGNO/LCN calculation, restart path, or Solar-System trajectory.

## Successor

The smallest justified successor is a separately preregistered Step 3g1c limited to a two-body Kepler drift and its canonical tangent map.
