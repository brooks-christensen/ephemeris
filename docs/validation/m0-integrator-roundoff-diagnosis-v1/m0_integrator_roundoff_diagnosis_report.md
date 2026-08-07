# M0 Integrator/Roundoff Diagnosis: BLOCKED

Primary mechanism: **BLOCKED**

Step 3 diagnosis status: **BLOCKED**

## Blocking Gate

The corrected two-body reversibility method passes every absolute gate, but the minimal-sync 0.25/0.5-day return-error ratio is 8.35827019, above the frozen maximum 4.

All four corrected-protocol two-body returns pass the absolute 1e-8 state-error gate, exact-time gate, callback-total gate, and nonfinite gate. The frozen timestep-ratio gate still fails:

| Mode | 0.5-day RMS | 0.25-day RMS | Fine/coarse | Limit |
| --- | ---: | ---: | ---: | ---: |
| current_sync | 2.26354582396e-13 | 3.66938140314e-13 | 1.6210767 | 4 |
| min_sync | 6.99726008176e-14 | 5.84849903499e-13 | 8.35827019 | 4 |

The first failed attempt is preserved separately. It demonstrated that flipping dt while retaining the unsynchronized internal Jacobi state starts the reverse leg from the wrong leapfrog phase. The corrected attempt uses the installed REBOUND 4.6.0 source-defined endpoint synchronization transition.

## Existing Histories

| Timestep | Fitted change / Myr | R2 | q per step | Same-sign blocks | Range exponent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0p5d | 2.89548283388e-10 | 0.99821607 | 3.96369997793e-19 | 10 | 0.616696 |
| 0p25d | 5.77548927044e-10 | 0.999939197 | 3.95310696129e-19 | 10 | 0.916896 |

## Candidate Mechanisms

- **BOUNDED_ENERGY_OSCILLATION**: not established. For: oscillatory residual power exists. Against: range growth and ten same-sign blocks fail the bounded-history interpretation.
- **RANDOM_WALK_ROUNDOFF**: not established. For: the 0.5-day range exponent overlaps a square-root-like regime. Against: the 0.25-day exponent and preregistered per-step signature are inconsistent with a pure random walk.
- **SYSTEMATIC_WHFAST_STEP_BIAS**: plausible but unclassified. For: existing full-history systematic signature passed=True. Against: physical-only synchronization controls and IAS15 were not authorized past the failed gate.
- **SYNCHRONIZATION_RECALCULATION_BIAS**: unclassified. For: the failed first method attempt confirms reversal is sensitive to internal leapfrog phase. Against: no 100 kyr current/min synchronization control ran.
- **VARIATION_MEGNO_COUPLING**: unclassified. For: the existing histories include variations and MEGNO. Against: no matched physical-only control ran.
- **CORRECTED_INVARIANT_OR_FORCE_PROBLEM**: unclassified. For: the existing corrected-energy drift is independently reconstructed. Against: no tolerance-converged IAS15 reference ran.
- **MIXED_OR_INCONCLUSIVE**: superseded by BLOCKED. For: several mechanisms remain unresolved. Against: the failed integrity gate requires BLOCKED rather than a causal conclusion.

## Configuration Matrix

No decisive M0 lane was launched. These frozen configurations remain preregistered but unobserved:

| Lane | Integrator | Purpose | Step / epsilon | Duration |
| --- | --- | --- | --- | ---: |
| m0_diag_phys_current_sync_0p5d_100k | whfast | current_sync | 0.5 | 100000 yr |
| m0_diag_phys_current_sync_0p25d_100k | whfast | current_sync | 0.25 | 100000 yr |
| m0_diag_phys_min_sync_0p5d_100k | whfast | min_sync | 0.5 | 100000 yr |
| m0_diag_phys_min_sync_0p25d_100k | whfast | min_sync | 0.25 | 100000 yr |
| m0_diag_phys_ias15_eps1e12_10k | ias15 | independent_integrator_tolerance_control | 1e-12 | 10000 yr |
| m0_diag_phys_ias15_eps1e13_10k | ias15 | independent_integrator_tolerance_control | 1e-13 | 10000 yr |
| m0_diag_reversibility_current_sync_0p5d_10k | whfast | current_sync | 0.5 | 10000 yr |
| m0_diag_reversibility_current_sync_0p25d_10k | whfast | current_sync | 0.25 | 10000 yr |
| m0_diag_reversibility_min_sync_0p5d_10k | whfast | min_sync | 0.5 | 10000 yr |
| m0_diag_reversibility_min_sync_0p25d_10k | whfast | min_sync | 0.25 | 10000 yr |

## Runtime And Scope

No decisive scientific lane ran, so scientific runtime and throughput are not applicable. The two bounded four-case method-validation commands each completed in under four wall-clock seconds; the driver did not instrument per-case throughput.

The requested full/current/min-sync, IAS15, and 10 kyr M0 reversibility figures are unavailable because producing them would require crossing the failed frozen gate.

## Decision

- Whether the production configuration must change is undetermined.
- A final 0.125-day lane is not justified.
- Historical Step 3, Step 3b, and Step 3c statuses remain unchanged.
- No Stage 4 command is provided.

## Smallest Next Action

In a separately preregistered follow-up, rerun only the four 365-day two-body method cases with a roundoff-aware reversibility scaling rule that retains the frozen absolute error, exact-time, callback, and nonfinite gates; do not launch an M0 trajectory until that gate passes.
