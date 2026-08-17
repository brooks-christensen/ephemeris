# Step 3g1d Manifest 27 Failed Campaign

## Classification

- Final status: STEP3G1D_CORRECTIVE_COMPLETION_FAILED
- Primary finding: SYNTHETIC_CONSERVATIVE_INTERACTION_KICK_NOT_QUALIFIED
- Verification envelope: NOT_ESTABLISHED
- Manifest 26 remains permanently STEP3G1D_BLOCKED.

This is a factual failed-campaign snapshot. It is not a qualification claim, and no full qualification artifact set was generated.

## Exact Result

The frozen pre-artifact campaign produced 109 passes and three failures. The Step 3g1d corrective group produced 32 passes and three failures; the isolated Step 3g1a, Step 3g1b, and Step 3g1c groups passed 25/25, 26/26, and 26/26 respectively.

The three failed nodes were:

- TangentKickTests::test_complete_kick_finite_difference_ladder
- TangentKickTests::test_force_and_jvp_finite_difference_closure
- TangentKickTests::test_projected_force_jvp_finite_difference_closure

All three failures were the frozen finite-difference shape assertions for the affine dense quadratic fixture. Its force/JVP minimum was 2.137694166068828e-15 and its complete-kick minimum was 1.6216182874045567e-15, both at the largest frozen epsilon and both below the unchanged 2e-7 cap. The required early-improvement count was zero. The nonlinear fixture retained the required convergence and roundoff pattern and passed.

## Passing Evidence

- Physical oracle maximum scaled error: 1.4035195471874325e-16.
- Tangent oracle maximum scaled error: 1.8078709969148304e-16.
- Maximum raw COM residual norm: 2.7194799110210365e-16; its case bound was 3.980286633290192e-14.
- Exact projected COM force and JVP rows were zero.
- Dense raw/scaled symplectic maxima: 4.510281037539698e-16 and 7.216449660063518e-15.
- Reversibility maximum scaled error: 1.1842378929335003e-16.
- Composition maximum scaled error: 1.3377282781857505e-16.
- Negative controls were detected with Jacobian asymmetry and raw symplectic maxima both 0.5.
- Evaluation accounting, static safety, Python compilation, strict JSON, git diff checking, and inherited integrity passed.

The static audit covered 118 literal nodes and four permitted subprocess sites. Inherited verification checked 165 hashes, 13 historical manifests, both protected annotated tags, and the exact 12 frozen Manifest 26 provenance errors.

## Scope

No integration, physical force/JVP provider, REBOUND, trajectory, archive, MEGNO/LCN, tag, Step 3g1e, or other forbidden operation occurred.
