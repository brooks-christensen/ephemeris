# M0 Step 3g1d Interaction Kick Requalification

Final status: **STEP3G1D_REQUALIFICATION_COMPLETE**

Primary finding: **SYNTHETIC_CONSERVATIVE_INTERACTION_KICK_QUALIFIED**

Verification envelope: **ISOLATED_SYNTHETIC_POSITION_ONLY_KICK_NO_PHYSICAL_FORCE_OR_INTEGRATION**

## Disposition

Manifest 26 remains permanently **STEP3G1D_BLOCKED** at commit
368cbac97940c093c261b374a9890d74c42c7666. Manifest 27 remains
permanently **STEP3G1D_CORRECTIVE_COMPLETION_FAILED** at commit
ffe6475afb94f980d61253ff32c35141f1465053, with 109 passes and three
affine finite-difference shape failures. Manifest 28 is a separate compact
delta requalification at commit 1713a2e7327bc292ca7358df3b90e76721acef7b;
it uses the method correction committed at
e979c0007edf0d8ddd57dd6efdbd1d7c24825485.

The isolated projected kick passed all
124 literal scientific, provenance,
historical-regression, artifact, safety, and integrity nodes. Four guarded
fresh subprocesses isolated the
118 pre-artifact nodes. This
is test-runner isolation, not production runtime behavior. Production kick.py
remained byte-identical to the failed-campaign snapshot during method correction
and qualification.

## Finite-Difference Method

Classification was fixed analytically before any ladder value was examined.
Dense quadratic force/JVP, complete kick tangent, and fixed linear projection
are **AFFINE_EXACT**. Their exact derivative is constant, so
they require independent oracle acceptance, the unchanged cap at the largest
epsilon and minimum, finite values, and the frozen binary64 roundoff envelope;
they do not require early improvements or a U-shaped curve.

The radial quartic fixture is **NONLINEAR_SMOOTH** and retains
Manifest 27 requirements unchanged: the same epsilon ladder, 2e-7 cap, at
least three early improvements, a resolved minimum, and a later
roundoff-dominated region.

- Dense kick largest/minimum errors:
  1.6216182874045567e-15 /
  1.6216182874045567e-15; early improvements
  0; roundoff consistent
  True.
- Dense force-JVP largest/minimum errors:
  2.1376941660688279e-15 /
  2.1376941660688279e-15; early improvements
  0; roundoff consistent
  True.
- Nonlinear kick minimum 0 at index
  6; early improvements
  3.
- Nonlinear force-JVP minimum 7.4722614613869985e-12 at index
  3; early improvements
  3.

## Numerical Summary

- Maximum physical / tangent scaled error: 1.4035195471874325e-16 /
  1.8078709969148304e-16.
- Maximum raw COM residual norm / derived bound norm:
  2.7194799110210365e-16 /
  6.1360395448588974e-14.
- Maximum COM norm ratio / component ratio:
  0.0068323720414402789 /
  0.0089936243841294266.
- Worst raw / scaled symplectic residual:
  4.5102810375396984e-16 /
  7.2164496600635175e-15.
- Negative-control asymmetry / symplectic residual:
  0.5 /
  0.5.
- Maximum reversal / composition scaled error:
  1.1842378929335003e-16 /
  1.3377282781857505e-16.

Every accepted physical force and JVP recorded its raw COM residual and frozen
derived bound, then projected the output COM row to exact zero. A COM-only
tangent direction produced no internal-force response. Above-bound nonclosing
and nonconservative controls were rejected.

## Evidence Boundary

The result covers only isolated synthetic position-only conservative
interaction kicks. No physical force provider, integration, timestep,
trajectory, archive, MEGNO, LCN, restart, or Solar-System state was evaluated.
The Step 3g1c raw symplectic residual remains an inherited risk and is not
repaired or reinterpreted here.

Success does not itself qualify drift-kick composition or production
models.

A separately preregistered Step 3g1e synthetic composition study is justified
as the next proposal only; it was not implemented or started.
