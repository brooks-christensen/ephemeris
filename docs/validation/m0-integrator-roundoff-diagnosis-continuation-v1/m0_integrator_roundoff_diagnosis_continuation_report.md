# BLOCKED

Step 3 diagnosis status: **BLOCKED**

## Classification Gates

- Artifact and telemetry integrity: True.
- Manifest-14 reversibility validity: True.
- IAS15 tolerance convergence: False.
- IAS15 global scaled-state RMS: 1.0004674719e-08 (limit 1e-10).
- IAS15 worst body: mercury barycenter at 3.05128180082e-08 (per-body limit 5e-10).
- IAS15 corrected-energy history difference: 6.07841718684e-16 (limit 5e-11); orbital elements agree: True.

## Long History

| Lane | Fitted change / Myr | R2 | q per step | Same-sign blocks | Range exponent |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0p5d | 2.89548283388e-10 | 0.99821607 | 3.96369997793e-19 | 10 | 0.616696 |
| 0p25d | 5.77548927044e-10 | 0.999939197 | 3.95310696129e-19 | 10 | 0.916896 |

## Controls

- m0_diag_phys_current_sync_0p5d_100k: fitted change/Myr `2.96667799062e-10`, max `3.46843111509e-11`, runtime `119.986` s.
- m0_diag_phys_current_sync_0p25d_100k: fitted change/Myr `5.8145799834e-10`, max `5.84707804891e-11`, runtime `220.026` s.
- m0_diag_phys_min_sync_0p5d_100k: fitted change/Myr `1.27282639815e-11`, max `1.36620926881e-11`, runtime `81.474` s.
- m0_diag_phys_min_sync_0p25d_100k: fitted change/Myr `1.87660342639e-11`, max `3.59617707081e-12`, runtime `146.060` s.
- m0_diag_phys_ias15_eps1e12_10k: fitted change/Myr `1.78705142198e-14`, max `4.58997745161e-16`, runtime `63.028` s.
- m0_diag_phys_ias15_eps1e13_10k: fitted change/Myr `-4.72675482898e-15`, max `3.37076627284e-16`, runtime `79.062` s.

## Reversibility Diagnostics

Fine/coarse ratios and apparent orders are diagnostic only and do not affect validity.

| Mode | Global RMS 0.5 d | Global RMS 0.25 d | Fine/coarse | Apparent order |
| --- | ---: | ---: | ---: | ---: |
| current_sync | 1.71208230571e-07 | 2.81593192593e-07 | 1.64474097801 | -0.717860399 |
| min_sync | 5.01499988871e-08 | 1.36503968029e-07 | 2.72191368013 | -1.44462132 |

## Candidate Mechanisms

- **BOUNDED_ENERGY_OSCILLATION**: fails. Evidence for: periodic residual peaks are present. Evidence against: range expansion and block/prefix gates.
- **RANDOM_WALK_ROUNDOFF**: fails. Evidence for: increment distributions are near centered. Evidence against: systematic q_h signature and sign-consistent blocks.
- **SYSTEMATIC_WHFAST_STEP_BIAS**: fails or is superseded. Evidence for: full signature=True, current signature=True. Evidence against: sync material reduction=False.
- **SYNCHRONIZATION_RECALCULATION_BIAS**: fails. Evidence for: current reproduces=True; min reduction=False. Evidence against: IAS15 reproduces=False.
- **VARIATION_MEGNO_COUPLING**: fails. Evidence for: current reduction=False. Evidence against: current reproduces full=True.
- **CORRECTED_INVARIANT_OR_FORCE_PROBLEM**: fails. Evidence for: IAS15 converged=False. Evidence against: significant IAS15 drift=False.
- **MIXED_OR_INCONCLUSIVE**: not needed. Evidence for: reserved for unresolved mixtures. Evidence against: selected mechanism=BLOCKED.

## Decision

- Present production configuration must change: `False`.
- A final 0.125-day lane is justified now: `False`.
- Historical Step 3, Step 3b, and Step 3c statuses remain unchanged.
- No production timestep is promoted, and no Stage 4 command is provided.

## Next Action

Preregister one bounded 10 kyr IAS15 epsilon=1e-14 tolerance lane, after a 100-year runtime benchmark, and compare epsilon=1e-13 versus 1e-14 against the unchanged convergence gates before another WHFast trajectory.
