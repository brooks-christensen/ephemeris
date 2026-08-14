# M0 Step 3g0 Verification and Architecture Audit

## Result

- Final status: `STEP3G0_VERIFICATION_ARCHITECTURE_AUDIT_COMPLETE`
- Primary finding: `V2_CORE_SPECIFICATION_READY`
- Verification envelope: `VERIFIED_WITHIN_DOCUMENTED_MODEL_AND_NUMERICAL_ENVELOPE`
- Callback classification: `CALLBACK_ACCOUNTING_EXACTLY_RECONCILED`
- Novelty classification: `NOVEL_COMBINATION_OR_EXTENSION`
- Current M0 production qualified: **no**

This audit completed the preregistered integrity, source, equation, force/JVP, callback, diagnostic, threshold, literature, and v2 design scope without a remaining ambiguity that blocks primitive implementation. It did not run a timestep, call an integration entrypoint, create a trajectory, modify production behavior, or revise any historical status.

## Integrity

The audit began at exact commit `997e5713aedb6e489c886e060af8fc6010b5c3b4` with a clean worktree. Manifest 21 was committed alone as `89a57748cdf5b16915f213dd72dd820e52295539` before new recomputation or conclusions.

Ancestry through the compiled-C baseline, Step 2, Steps 3b-3f1, and the Step 3f1 preregistration was verified. Both protected tags were annotated:

- `gr-tangent-compiled-c-v1` tag object `e12153e5...`, target `2d7778e1911c6f6ae97da24cdfe00ef45f21e73b`.
- `gr-tangent-python-oracle-v2` tag object `9661c72f...`, target `9933bc5e3d9bfe9ec07e72929c22e4406f12b441`.

Protected source hashes, Manifests 13-20, compact historical reports/summaries, and the frozen archive hashes matched Manifest 21. The two Step 3f1 archives remained byte-identical after read-only access and each contained 11 monotonically timed snapshots from 0 to 10 kyr. No transient Step 3f1 file was present.

Runtime identity was Python 3.10.12, NumPy 2.2.6, REBOUND 4.6.0 (`e3b07aa...`), REBOUNDx 4.6.1, GCC 11.4.0, glibc 2.35 on x86_64 WSL2. Rounding was `FE_TONEAREST`; binary64 epsilon was `2.220446049250313e-16`; NumPy long double exposed a 64-bit significand in 128-bit storage. Imported module/shared-library paths and hashes and the compiled callback's strict build flags are frozen in Manifest 21.

## Code Review

The complete review scope covered ephemeris extraction, constants, frame/unit transforms, body ordering, REBOUND construction, Newtonian gravity, protected GR force/JVP, native variations and MEGNO/LCN access, target/synchronization/copy behavior, telemetry, elements/orientation, checkpoint/restart, callback accounting, schema/fingerprints, and offline threshold logic. It also traced the exact relevant REBOUND 4.6.0 and REBOUNDx 4.6.1 symbols.

No critical finding or validated equation defect was found. The machine-readable register contains 11 findings:

| Severity | Count | Main consequence |
|---|---:|---|
| Critical | 0 | None |
| High | 2 | Ill-conditioned plane gate; misapplied different-map tangent/MEGNO thresholds |
| Medium | 4 | Fail-open kernel input domain; incomplete callback formula; positional identity; observer/live-map coupling |
| Low | 0 | None |
| Informational | 5 | Frozen model/runtime limitations and confirmed WHCKL derivative facts |

The findings do not establish that frozen finite runs exercised the singular/nonfinite kernel path. Every frozen scientific state was finite and the callback nonfinite counter was zero. The current behavior is therefore a v2 input-contract requirement, not retrospective evidence invalidation.

## Physical Model

M0 contains ten mutually active point masses ordered Sun, Mercury barycenter, Venus barycenter, Earth barycenter, Mars barycenter, Jupiter barycenter, Saturn barycenter, Uranus barycenter, Neptune barycenter, and Pluto barycenter. Initial positions/velocities come from DE431 at 2000-01-01 in Solar-System-barycentric ICRF and are converted to SI without recentering. Masses come from frozen DE431-consistent GM values divided by the repository `G`; Earth barycenter uses aggregate Earth-Moon GM.

Newtonian gravity is pairwise. For each body relative to the Sun, the additional model term is

```text
a_i = -6*s*(G*M_sun)^2/c^2 * d_i/r_i^4
a_sun = -sum_i (m_i/M_sun)*a_i
```

with potential

```text
U_GRpot = -3*s*G^2*M_sun^2/c^2 * sum_i m_i/r_i^2.
```

The tangent action is the derivative

```text
J(d) = -6*s*(G*M_sun)^2/c^2 * (I/r^4 - 4*d*d^T/r^6)
```

on `delta_x_i-delta_x_sun`, with the same mass-weighted solar reaction. The model obeys translation, O(3), and Galilean covariance; the physical force does not depend on tangent state.

The frozen initial Mercury two-body elements give the standard analytic GR advance `42.98066416845862 arcsec/century`. This validates the intended perihelion limit, not the mean motion: [REBOUNDx documents](https://reboundx.readthedocs.io/en/stable/effects.html) that `gr_potential` gets precession right but mean motion wrong by `O(GM/(a*c^2))`.

The separate Moon, solar J2, massive asteroids, and a more complete 1PN model are required in later production/sensitivity ladders. Solar mass evolution and stellar encounters are outside the initial methods baseline. DE440/DE441 is an initial-condition sensitivity question, not a reason to alter DE431 history.

## Force and JVP Verification

All new checks were static-state only and ran under a fail-fast guard over `Simulation.integrate`, `Simulation.step`, ctypes step/integrate symbols, and known project runners.

The checks covered:

- closed-form Newtonian two-body acceleration and momentum closure;
- protected GR force and reaction against an independent 70-digit Decimal oracle;
- JVP against Decimal analytic evaluation, complex-step formula differentiation, and centered finite differences;
- JVP linearity, translation, rotation, reflection, velocity-boost independence, zero-scale and two-body limits;
- deterministic random Python/C equality for 12 multi-body states;
- physical output independence from tangent input;
- explicit singular/nonfinite oracle rejection and reproduction of the protected fail-open input behavior;
- strict-warning and ASan/UBSan static C harnesses.

All comparisons passed within their stated binary64/differencing tolerances. The protected physical equations and compiled callback were not edited.

## Exact Callback Accounting

For Lane P, there are `14,610,000` lazy-kernel steps and two acceleration calls per step:

```text
live lazy map                    29,220,000
first forward order-17 corrector         32
100 exact-endpoint exit checks        3,200
100 integrate-return synchronizations 3,200
                                      ----------
source schedule total           29,226,432
```

Each order-17 corrector has 16 `Z` stages and two force calls per stage, hence 32 calls. The historical expected `29,223,232` included the first corrector and return synchronization but omitted the exact-endpoint exit-check synchronization. The omitted `100*32` is exactly the observed difference of 3,200.

The restart branch independently confirms the schedule: `2*146,100 + 32 + 32 = 292,264`, versus historical `292,232` when one synchronization is omitted.

With `keep_unsynchronized=1`, synchronization copies/restores internal `p_jh`; diagnostic copies and archive serialization do not add force calls or change the live physical map. The frozen restart state equality therefore remains valid. This is an incomplete historical accounting formula, not an equation defect.

## Diagnostic Validity

The historical orientation implementation normalized each vector and used `acos(clip(dot))`. For identical normalized binary64 vectors, the dot can be one ulp below 1, producing an artificial angle around `sqrt(epsilon)`.

Frozen Lane P/T state recomputation over 1,818 body/epoch/metric comparisons found:

| Orbital-plane estimator | Maximum (rad) | Zero count |
|---|---:|---:|
| Historical `acos` | `2.1073424255447017e-8` | 666 |
| `atan2(norm(cross),dot)` | `4.51049031315292e-11` | 9 |
| Chord `2*asin(norm(u-v)/2)` | `4.510494677284343e-11` | 9 |

The worst historical `acos` value occurs at the identical initial epoch. `atan2` and chord agree to at most `5.144191857205088e-17 rad`. The Manifest 20 `1e-8` plane gate had a method defect and is classified `ILL_CONDITIONED`. The historical recorded value/status remains unchanged.

The apsidal-direction maximum was about `1.08174e-7 rad` by robust estimators, but this angle is poorly interpretable for nearly circular states; the eccentricity vector remains the preferred nonsingular observable.

Offline energy recomputation from stored rows reproduced the corrected definition. Maximum corrected relative drift was `3.772e-13` (Lane P) and `3.184e-12` (Lane T), compared with Newtonian-only drift near `2.01e-9`; angular-momentum norm drift was `2.316e-13` and `1.791e-13`. These agree with the historical conclusion that conservation gates passed.

## Threshold Provenance

The `0.9999` direction cosine and derived `0.014142...` RMS threshold trace to Manifest 10 Python/C implementation equivalence. The `0.001` MEGNO and LCN gates likewise support identical-map implementation or restart comparison. Step 3f1 compared different physical maps, with different carrier trajectories and synchronization histories, over only 10 kyr in a system whose Lyapunov time is on a Myr scale.

Consequently:

- direction cosine/RMS: `VALID_ONLY_FOR_IMPLEMENTATION_EQUIVALENCE`;
- MEGNO/LCN equality: `VALID_ONLY_FOR_SAME_MAP_REPRODUCIBILITY`;
- use as a 10-kyr different-map physical architecture veto: `PHYSICALLY_UNJUSTIFIED`;
- tangent log-norm extension: `PROVENANCE_INSUFFICIENT`.

No replacement threshold was inferred from Step 3f1 observations. Raw tangent direction can remain useful for same-map regression. Different-map work needs preregistered finite-time growth, principal/subspace angles where applicable, convergence across intervals, and uncertainty tied to trajectory separation and rescaling history.

## Corrected Step 3f1 Interpretation

Manifest 20 remains `STEP3F1_TWO_LANE_SCREEN_FAILED` with `BOTH_LANES_UNQUALIFIED`. Step 3g0 does not revise it.

Prospectively, the plane-angle veto and different-map tangent/MEGNO equality veto do not support their intended architecture conclusions. Nevertheless Lane P still fails the valid Saturn RMS screen (`3.454e-9 > 2.183e-9`), and Lane T retains carrier/other assigned failures. Correcting diagnostic scope therefore does not qualify either lane and does not authorize a production timestep.

## WHCKL Tangent Feasibility

The [WHFast paper](https://academic.oup.com/mnras/article/452/1/376/1748797) establishes standard-kernel Jacobi/Kepler/interaction first variations. The [high-order kernel paper](https://academic.oup.com/mnras/article/489/4/4632/5565063) defines the lazy kernel and order-17 corrector and explicitly says those new kernels did not support variations.

Version-matched source resolves the derivative graph:

```text
A0       = A(q)
q_shift  = q + h^2*A0/12
A1       = A(q_shift)
kick with A1; restore q

dA0      = J_A(q)*dq
dq_shift = dq + h^2*dA0/12
dA1      = J_A(q_shift)*dq_shift
tangent kick with dA1; restore dq
```

Thus a first-order lazy tangent needs force and JVP at the unshifted and shifted positions. It does not need a Hessian. The validated `gr_potential` JVP is sufficient for that perturbation. Each of the 16 order-17 `Z` stages is a Kepler/kick composition and is differentiated by existing Kepler derivatives plus force JVPs; again no Hessian is required.

Correctors belong to internal integrator coordinates. With `safe_mode=0` and `keep_unsynchronized=1`, corrected output is built on a copy while the live map remains unsynchronized. Canonical symplecticity must be tested in Jacobi `(q,p)`, including mass conventions, not raw `(x,v)`.

## Novelty Review

Prior art is substantial:

- Mikkola and Innanen, DOI `10.1023/A:1008312912468`, established nearly Keplerian symplectic tangent maps.
- Rein and Tamayo, DOI `10.1093/mnras/stv1257`, implemented WHFast variations and MEGNO/LCN.
- Gozdziewski, Breiter, and Borczyk, DOI `10.1111/j.1365-2966.2007.12608.x`, implemented symplectic tangent-map MEGNO with a different SABA/Poincare architecture.
- Skokos and Gerlach, DOI `10.1103/PhysRevE.82.036704`, gave the general derivative-of-composed-symplectic-map method.
- Agol, Hernandez, and Langford, DOI `10.1093/mnras/stab2044`, propagated derivatives through a different symplectic N-body map.

The primary-source and official-repository search through 2026-08-13 found no close public match combining REBOUND's WHCKL lazy-kernel derivative, differentiated order-17 corrector, arbitrary position-potential JVP, noninterfering observer semantics, and exact tangent restart. A negative search is not universal proof, so the defensible classification is `NOVEL_COMBINATION_OR_EXTENSION`, not first-ever implementation or confirmed universal absence.

## Relativity Recommendation

The initial methods work should use Newtonian and position-only `gr_potential` to prove the WHCKL tangent architecture. Solar-System secular validation should compare `gr_potential`, central-body velocity-dependent 1PN (`gr`), full N-body 1PN/EIH (`gr_full`), and solar J2. Lense-Thirring is a later sensitivity. Velocity-dependent models need a separate operator/JVP architecture and cannot be inserted as a plain WHCKL position kick.

## V2 Architecture

The selected strategy is an external, test-first WHCKL tangent engine using audited utilities behind adapters, with a pinned REBOUND 4.6.0 historical compatibility layer and a possible later human-authored upstream contribution. The four layers are immutable physical model, pure force/JVP kernels, integrator/tangent adapters, and offline observers.

The design requires immutable `BodyId` layout, typed units, integer ticks, caller-owned buffers, explicit evaluation contexts, single-writer live-map ownership, observer copies, atomic content-addressed checkpoints, and separate live/JVP/corrector/diagnostic/offline/restart counters. Force kernels cannot synchronize, output, mutate settings, use mutable globals to infer purpose, or contain MEGNO logic.

## Remaining Risks

- The current M0 numerical lane is not production-qualified.
- The v2 WHCKL tangent map, corrector, canonical symplecticity, convergence, observer, and restart gates remain unimplemented.
- The protected kernels' singular/nonfinite input behavior must be rejected at the v2 boundary.
- The position-only relativity model omits mean-motion accuracy and full N-body 1PN physics.
- Separate Moon, solar J2, asteroids, and modern-ephemeris sensitivity remain future physical work.
- The novelty conclusion should receive independent scholarly and repository review before publication wording is finalized.

## Verification Commands

No integration command was run. Focused checks included:

```text
PYTHONPATH=mini_ephemeris/src .venv/bin/python -m unittest discover \
  -s mini_ephemeris/tests -p test_m0_step3g0_verification.py -v

cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -fno-fast-math -ffp-contract=off ... m0_step3g0_static_harness.c

cc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Werror \
  -fsanitize=address,undefined -fno-sanitize-recover=all ...
```

The closeout also ran protected/historical SHA-256 validation, read-only archive inventory, strict JSON/finite checks, Python compilation, current source/contract tests, deterministic artifact regeneration, `git diff --check`, and complete diff review.

## Smallest Next Step

Step 3g1 should implement only pure immutable model and force/JVP interfaces, deterministic timebase, observer-copy ownership, and isolated coordinate/Kepler/kick/lazy/corrector primitives. It must add no MEGNO until the primitive tangent map passes and run no Solar-System trajectory until analytic, finite-difference, canonical-symplecticity, reversibility, and restart unit gates pass.
