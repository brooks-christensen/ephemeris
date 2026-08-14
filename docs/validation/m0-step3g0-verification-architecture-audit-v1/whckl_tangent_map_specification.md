# WHCKL First-Order Tangent Map Specification

## Scope

This document specifies, but does not implement, the first derivative of the exact REBOUND 4.6.0 WHFast lazy-kernel map with Jacobi coordinates and first corrector order 17. The target is a discrete tangent map of the implemented floating-point operator sequence, not merely integration of the continuous variational ODE.

Let canonical internal state be `z=(q,p)` in Jacobi coordinates and tangent `dz=(dq,dp)`. A stored unsynchronized step is schematically:

```text
Phi_h = C_h^-1 o K_WHCKL,h o C_h
D Phi_h = D C_h^-1 o D K_WHCKL,h o D C_h
```

`C_h` is the first symplectic corrector representation transition. In the actual long-running map, its forward action is applied on initialization, corrected/inertial output is constructed by inverse action on a copy, and the stored `p_jh` map remains unsynchronized when `keep_unsynchronized=1`.

## State contracts

- Physical canonical arrays: Jacobi positions `q`, canonical Jacobi momenta `p`, masses, time tick, synchronization flag, corrector state.
- Tangent canonical arrays: `dq`, `dp` in the identical layout and mass convention.
- Cartesian force boundary: inertial positions and acceleration; a position-only force kernel also supplies `J_A(x)*dx`.
- The ordinary canonical matrix `Omega=[[0,I],[-I,0]]` applies only to canonical `(q,p)`. It must not be applied to unweighted `(x,v)`.
- All transforms are generated from frozen masses. Their first variations are the same linear operators on tangent arrays.

## Primitive graph

### 1. Inertial-to-Jacobi transform `T_IJ`

Physical: `(x,v) -> (q,p)` using the exact REBOUND mass-weighted Jacobi ordering. Tangent: `(dx,dv) -> (dq,dp)` using the same fixed linear transform. Temporary storage must not alias live input unless the exact source routine permits it. Force/JVP count: zero. Restart stores the resulting internal arrays and mass/layout fingerprint.

### 2. Kepler drift `D_K(tau)`

Physical: independently advance each hierarchical Jacobi two-body subsystem by the universal-variable WHFast Kepler solver; advance the center-of-mass drift in the source-defined place. Tangent: use the Mikkola-Innanen / WHFast first-variation equations with the same converged universal anomaly and exact source operation order. Force/JVP count: zero. Acceptance requires two-body analytic closure, tangent finite-difference closure, reversibility, and linearity.

### 3. Jump/center-of-mass drift

Physical: linear position update by the source-defined momentum combination. Tangent: apply its constant linear derivative. Force/JVP count: zero. It must preserve canonical mass weighting and round-trip with the selected coordinate convention.

### 4. Inertial force evaluation `F(x)`

Physical: transform the current Jacobi positions to inertial positions, then evaluate Newtonian interaction plus every supported position-only perturbation. Tangent: transform `dq` to `dx`, call each kernel JVP, sum in the same component order, and transform acceleration/JVP to the Jacobi kick representation. Force count: one; JVP count: one when a tangent is active. Kernels cannot synchronize, output, or inspect mutable invocation globals.

### 5. Interaction kick `K_B(tau)`

Physical: `q_out=q`, `p_out=p+tau*F_q(q)` with REBOUND's interaction/jump split. Tangent: `dq_out=dq`, `dp_out=dp+tau*J_Fq(q)*dq`. Force values and JVPs must refer to the same physical state and coordinate transforms.

### 6. Lazy-kernel shifted kick `K_L(h)`

The REBOUND 4.6.0 source sequence is:

```text
A0       = A(q)
q_saved  = q
q_shift  = q + (h^2/12)*A0
A1       = A(q_shift)
p_out    = interaction_kick(p, h, A1)
q_out    = q_saved
```

Its tangent action is exactly:

```text
dA0      = J_A(q) * dq
dq_shift = dq + (h^2/12)*dA0
dA1      = J_A(q_shift) * dq_shift
dp_out   = dp + h * interaction_JVP(q_shift, dq_shift, A1, dA1)
dq_out   = dq
```

The auxiliary shift is differentiated. A first-order tangent requires force values and force JVPs at `q` and `q_shift`; it does not require a Hessian or higher derivative. The existing analytic `gr_potential` JVP supplies the required perturbation derivative because that force is position-only. The combined Newtonian-plus-perturbation JVP must include derivatives of all coordinate transforms and the exact interaction-force subtraction used by WHFast.

Per live WHCKL step the physical lane performs two acceleration evaluations. With a tangent active, both evaluations need matching JVP contexts. The temporary shifted state must be private and the saved live positions restored exactly.

### 7. First corrector order 17

REBOUND 4.6.0 expresses the corrector as 16 coefficient-selected `Z(a,b)` stages. Each stage is the exact ordered composition:

```text
Kepler(+a)
transform positions to inertial
force evaluation
interaction kick(-b)
Kepler(-2a)
transform positions to inertial
force evaluation
interaction kick(+b)
Kepler(+a)
```

The tangent corrector applies the derivative of every operation in the same order. Each `Z` therefore has two force and two JVP evaluations; order 17 has 32 force evaluations and 32 JVP evaluations when tangent state is active. The inverse uses source-defined reversed signs/order, not a separately approximated formula. No Hessian is introduced: each force occurrence is differentiated once by its JVP.

Corrector state belongs to the integrator representation. With `keep_unsynchronized=1`, output correction occurs on an observer copy and the live stored `(q,p,dq,dp)` remains untouched. A fresh-process restart must preserve whether the forward corrector is already represented and must never apply it twice.

## Exact live step order

The implementation adapter must transcribe the version-matched sequence, including half drifts and time updates, from `reb_integrator_whfast_step`. At a conceptual level:

```text
initialize internal Jacobi state and forward corrector once
half/full Kepler and center-of-mass drift according to synchronization phase
jump half-step
construct inertial positions
evaluate A0 and J_A0
perform lazy shifted evaluation A1 and J_A1
interaction kick
restore saved q/dq
jump half-step
mark internal map unsynchronized
advance integer time tick
```

The implementation specification must include a source-line fixture that locks the exact 4.6.0 ordering before code is written. Floating-point reassociation is forbidden in reproducibility mode.

## Synchronization and observation

`safe_mode=0` means the adapter does not automatically synchronize after each step. `keep_unsynchronized=1` means synchronization for output copies internal `q,p,dq,dp`, applies remaining drift and inverse corrector to the copy, transforms to inertial Cartesian state, and restores/leaves the live map unchanged.

Observer contexts are explicit:

- `LIVE_MAP`: modifies live internal state and live counters.
- `CORRECTOR_INIT`: applies the one-time forward corrector.
- `OBSERVER_SYNC`: constructs a synchronized copy only.
- `CHECKPOINT_SERIALIZE`: serializes internal state without correction.
- `RESTART_RECONSTRUCT`: rebinds pure kernels and validates schema without advancing.
- `OFFLINE_ANALYSIS`: cannot access or increment live counters.

The historical 29,226,432 count is exactly `2*14,610,000 + 32 + 100*32 + 100*32`. The last two terms are pre-final-step exit-check and integrate-return inverse correctors. Diagnostic copy and serialization add zero. This schedule motivates explicit context counters.

## Tangent norm, rescaling, MEGNO, and LCN

The primitive tangent map owns only `dz`. A separate tangent-state service may rescale at deterministic integer ticks and stores:

- current canonical tangent vector;
- accumulated logarithmic scale;
- rescale count and exact tick ledger;
- norm definition and mass/unit scaling fingerprint;
- sign/orientation convention.

MEGNO is not part of force kernels. After the primitive map passes, a noninterfering diagnostic update may consume live-map tangent growth at exact ticks and update the discrete/continuous definition selected by a preregistered formula. LCN is derived from accumulated log growth and elapsed physical time with an explicit `t=0` unavailable state, never a silent fallback. MEGNO/LCN state must be checkpointed with formula/schema versions.

## Checkpoint and restart

Checkpoint schema must include immutable model/configuration hashes; integer tick and step; canonical physical and tangent arrays; synchronization flag; corrector-applied flag/order; temporary-cache version if required; tangent rescale ledger; MEGNO/LCN accumulators; counter ledger; target/sample ledger; compiler/runtime identities; and checksum. Writes are atomic and collision-safe. An incompatible fingerprint or schema fails before state construction.

Fresh-process acceptance requires byte-exact or explicitly bounded equality of internal canonical state, all tangent/diagnostic auxiliaries, next sample identity, and every context counter. No force call or synchronization may occur merely to inspect or serialize a checkpoint unless the schema explicitly records that event.

## Arbitrary position-dependent potentials

Supported perturbations implement a pure interface:

```text
evaluate(context, model, x, out_acceleration)
jvp(context, model, x, dx, out_delta_acceleration)
```

Both functions are deterministic, finite-checked, allocation-policy declared, and side-effect free except an explicitly passed context counter. A force without a JVP cannot be enabled in tangent mode. Velocity-dependent forces require a different operator and derivative contract and are not silently treated as position kicks.

## Future acceptance gates

1. Primitive analytic and finite-difference closure for every transform, Kepler, kick, lazy, and corrector operation.
2. Linearity of every tangent primitive and of one complete step.
3. `||D Phi^T Omega D Phi - Omega||` in canonical Jacobi `(q,p)` over deterministic state ranges.
4. Determinant behavior consistent with symplectic volume preservation, with conditioning reported.
5. Forward/backward reversibility for physical and tangent internal states.
6. Expected convergence order for the exact selected kernel/corrector.
7. Observer-copy noninterference and context-counter closure.
8. Exact fresh-process restart before and after output/corrector boundaries.
9. Temporary sign, body-swap, omitted-reaction, wrong-r-power, and tangent-array mutants must be killed.
10. No MEGNO and no Solar-System trajectory until gates 1-8 pass.
