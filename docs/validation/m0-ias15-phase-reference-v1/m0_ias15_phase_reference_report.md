# IAS15_REFERENCE_QUALIFIED_AT_ROUNDOFF_PHASE_FLOOR

Manifest-13 primary mechanism: **SYSTEMATIC_WHFAST_STEP_BIAS**

Step 3 diagnosis: **STEP3_NUMERICAL_FLOOR_CHARACTERIZED**

## Benchmark

- Observed 100-year runtime: 0.348435813 s.
- Projected 10-kyr runtime: 34.8435813 s (limit 3600 s).
- Accepted steps: 15957; nonfinite callbacks: 0.

## Qualification

- Integrity: `True`.
- Frozen corrected-energy gates: `True`.
- Frozen nonphase element gates: `True`.
- Phase signature: `True`.
- Angular-momentum conclusion unchanged: `True`.
- IAS envelope / WHFast global discrepancy: `0.00173960181465` (limit `0.1`).
- IAS worst body: `mercury barycenter`; body ratio `0.00301816165712` (limit `0.1`).

## IAS15 Lanes

| epsilon | accepted steps | rejected | callbacks | runtime s | median proposed dt d |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1e-09 | 1596849 | unavailable | 41556727 | 34.6792 | 4.15819384 |
| 1e-12 | 4284997 | unavailable | 74575651 | 63.0276 | 1.51575174 |
| 1e-13 | 5954089 | unavailable | 93655150 | 79.0623 | 1.0894745 |

Rejected IAS15 attempt counts are unavailable: the exact installed REBOUND build retries internally, exposes only successful `steps_done`, and serializes no rejected-attempt counter. No REBOUND or callback instrumentation was added.

| epsilon | endpoint energy | fitted slope / yr (95% CI) | max | RMS | p99 | same-sign blocks | angular max |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1e-09 | 1.65847752714e-16 | 6.45008140194e-20 [5.29815994006e-20, 7.60200286382e-20] | 6.62331340546e-16 | 2.59040559755e-16 | 6.07500590386e-16 | 5/10 | 4.74219159408e-16 |
| 1e-12 | 2.09701172527e-16 | 1.78705142198e-20 [8.30372385198e-21, 2.74373045877e-20] | 4.58997745161e-16 | 1.71251596426e-16 | 3.99388368982e-16 | 5/10 | 3.16146106272e-16 |
| 1e-13 | 5.24296202142e-17 | -4.72675482898e-21 [-1.30952795439e-20, 3.64176988591e-21] | 3.37076627284e-16 | 1.28012664697e-16 | 3.05652939825e-16 | 4/10 | 4.74219159408e-16 |

## Phase Evidence

- `m0_diag_phys_ias15_default_10k_vs_m0_diag_phys_ias15_eps1e12_10k`: global RMS `8.91256874948e-09`; transverse variance fraction `0.987317047`; worst body `mercury barycenter` at `2.74197101689e-08`.
- `m0_diag_phys_ias15_default_10k_vs_m0_diag_phys_ias15_eps1e13_10k`: global RMS `5.00705827624e-09`; transverse variance fraction `0.990927989`; worst body `mercury barycenter` at `1.49798359667e-08`.
- `m0_diag_phys_ias15_eps1e12_10k_vs_m0_diag_phys_ias15_eps1e13_10k`: global RMS `1.0004674719e-08`; transverse variance fraction `0.989935109`; worst body `mercury barycenter` at `3.05128180082e-08`.

The reported `t^0.5` and `t^1.5` residual comparisons are descriptive; neither model is forced as a gate.

## Frozen Classification

Only the manifest-13 raw pointwise IAS15 state-convergence condition is superseded. Every WHFast threshold and all other causal evidence are unchanged.

- All non-IAS causal evidence unchanged: True.
- Full MEGNO/tangent and physical-only current-sync states exactly match over 10 kyr at both timesteps: True.
- Synchronization material-reduction gate remains false independently of phase uncertainty: True.

Smallest next action: Proceed only through a separately preregistered Step 3e: one 0.125-day 1 Myr lane evaluated against the manifest-16 IAS15 roundoff/phase envelope.

## Bounded Step 3e Prompt

Proceed with Step 3e only. Preregister a versioned manifest before integration and run exactly one fresh full-M0 compiled-C tangent lane: 0.125-day WHFast, 1,000,000 years, 100-year scientific cadence, 100,000-year archive cadence, MEGNO seed 12345, identical DE431 state and unchanged validated equations. Compare the existing 0.25-day lane with the new 0.125-day lane using every frozen Step-3 physical, orbital, tangent, MEGNO, LCN, corrected-energy, angular-momentum, schema, fingerprint, callback, and artifact criterion. Treat manifest 16's three-lane IAS15 phase/roundoff envelope as a preregistered uncertainty floor; do not weaken a WHFast threshold or require monotonic IAS15 raw phase convergence. Run no other WHFast trajectory, no Stage 4, and no 10 Myr integration. Emit one convergence status, compact reports, focused checks, and the smallest evidence-based next action.
