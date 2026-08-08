# M0 Step 3e1 Offline State Diagnosis

**Final status:** STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE

**Primary classification:** TRUE_NONPHASE_NONCONVERGENCE

This analysis is offline and diagnostic only. It preserves Manifest 17's STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED status and does not retroactively validate 0.25 day.

## Frozen provenance

- Manifest 18 SHA-256: 088b55fa40cf0ccb7fa50f42d41f017fcdf560d7a8f7a7dfa69678717544021d.
- Manifest 16 mechanism remains SYSTEMATIC_WHFAST_STEP_BIAS.
- Manifest 16 diagnosis remains STEP3_NUMERICAL_FLOOR_CHARACTERIZED.
- Manifests 13 and 15 remain BLOCKED; Manifest 14 remains REVERSIBILITY_GATE_PASSED.
- No trajectory, IAS15 lane, benchmark, smoke integration, Stage 4, or 10 Myr command was run or provided.

## Manifest 17 reproduction

| Failed metric | Reproduced |
| --- | ---: |
| global_rms_ratio | 0.551548145246 |
| mercury_rms_ratio | 1.07191551941 |
| uranus_phase_orientation_ratio | 0.854837076986 |
| venus_orientation_rad | 8.30345456837e-08 |

## Physical-state defects

- Global coarse RMS: 0.000599197558712.
- Global fine RMS: 0.000330486302143.
- Fine/coarse ratio: 0.551548145246.
- Global position-component RMS (coarse, fine): 19532.6896 km, 17735.9861 km.
- Global velocity-component RMS (coarse, fine): 3.96907177 m/s, 2.14312498 m/s.

| Body | Coarse position RMS (km) | Fine position RMS (km) | Coarse velocity RMS (m/s) | Fine velocity RMS (m/s) | Ratio | Fine contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| earth barycenter | 49561.7 | 12031.8 | 9.89068 | 2.40095 | 0.24275 | 12.039% |
| jupiter barycenter | 28770.8 | 52183.2 | 0.484467 | 0.878706 | 1.81376 | 7.143% |
| mars barycenter | 15717.2 | 2543.78 | 1.67471 | 0.270868 | 0.161749 | 0.163% |
| mercury barycenter | 6477.02 | 6863.24 | 5.68568 | 6.09466 | 1.07192 | 75.766% |
| neptune barycenter | 951.436 | 1946.42 | 0.00114984 | 0.00235231 | 2.04577 | 0.008% |
| pluto barycenter | 877.718 | 1701.89 | 0.000764752 | 0.00148169 | 1.93899 | 0.006% |
| saturn barycenter | 2689.31 | 13770.8 | 0.0182596 | 0.0935166 | 5.12061 | 0.406% |
| sun | 27.5306 | 50.0881 | 0.000463701 | 0.000839415 | 1.81736 | 0.000% |
| uranus barycenter | 1336.99 | 2703.12 | 0.00318748 | 0.00644278 | 2.02179 | 0.015% |
| venus barycenter | 15213.8 | 4538.94 | 4.93387 | 1.47177 | 0.298301 | 4.455% |

The compact JSON and fixed CSV tables contain position-only, velocity-only, absolute SI scales, quantiles, worst epochs, ten windows, and ten cumulative endpoints for the full system and every body.

## Detailed diagnosis

- Mercury aggregate ratio: 1.07191551941.
- Fine global transverse-position fraction: 0.998554.
- Venus classical condition amplification: 0.985062.
- Uranus phase/orientation numerator: 3.84773e-09 rad; denominator: 4.50113e-09 rad.

Both mean-anomaly and mean-longitude phase alignments are reported. No method was selected after observing the result. Cartesian/element round trips and RTN basis reconstruction passed the preregistered tolerances.

## Classification evidence

- **WINDOWED_OR_PHASE_DOMINATED:** not fully supported; no_true_nonphase=FAIL, phase_or_transverse_concentration=PASS, phase_stripped_global_coherence=FAIL, phase_stripped_mercury_and_inner_nonphase=FAIL, window_dependence=PASS.
- **METRIC_OR_REPRESENTATION_ILL_CONDITIONED:** not fully supported; coordinate_free_converges=PASS, ill_conditioned_failed_metric=FAIL, no_threshold_change=PASS.
- **POINTWISE_PREDICTABILITY_FLOOR:** not fully supported; early_richardson_coherence=PASS, frozen_other_gates_preserved=PASS, no_persistent_nonphase=FAIL, tangent_growth_consistency=PASS, three_consecutive_late_loss=PASS, transition_reproduced_in_two_failed_bodies=FAIL.
- **TRUE_NONPHASE_NONCONVERGENCE:** supported; coherent_or_secular_component=PASS, coordinate_free_aggregate_nonconvergence=PASS, not_floor_or_representation=PASS, persistent_three_late_windows=PASS, phase_removal_does_not_restore=PASS.

## Conditioning and predictability

- Tangent/fine-defect Spearman correlation: 0.981546.
- Candidate final MEGNO: 2.0000876284.
- Candidate final finite-time LCN: -5.20392172231e-16 1/yr.
- Manifest 17's tangent, MEGNO, LCN, orbital, perihelion, invariant, and energy conclusions remain frozen.

## Window, RTN, and phase results

| Window end (kyr) | Global ratio | Mercury ratio | Uranus ratio |
| ---: | ---: | ---: | ---: |
| 100 | 0.262851 | 0.157848 | 8.26693 |
| 200 | 0.270459 | 0.0625738 | 7.63134 |
| 300 | 0.28207 | 0.177879 | 16.3071 |
| 400 | 0.307175 | 0.307156 | 4.84226 |
| 500 | 0.324859 | 0.392152 | 3.27321 |
| 600 | 0.377749 | 0.574516 | 2.84077 |
| 700 | 0.462588 | 0.829628 | 2.6028 |
| 800 | 0.541945 | 1.0587 | 2.20866 |
| 900 | 0.606223 | 1.27596 | 1.97447 |
| 1000 | 0.681922 | 1.55526 | 1.8099 |

### Cumulative ratios

| Cumulative end (kyr) | Global | Mercury | Venus | Uranus |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 0.262851 | 0.157848 | 0.35622 | 8.26693 |
| 200 | 0.269496 | 0.0824893 | 0.391339 | 7.6514 |
| 300 | 0.27856 | 0.158148 | 0.401679 | 13.0665 |
| 400 | 0.295562 | 0.256189 | 0.392902 | 5.58136 |
| 500 | 0.310085 | 0.325622 | 0.366766 | 3.74853 |
| 600 | 0.339613 | 0.432757 | 0.34674 | 3.14942 |
| 700 | 0.388509 | 0.585304 | 0.334021 | 2.85432 |
| 800 | 0.443674 | 0.746783 | 0.321726 | 2.49396 |
| 900 | 0.496458 | 0.903109 | 0.310127 | 2.24299 |
| 1000 | 0.551548 | 1.07192 | 0.298301 | 2.02179 |

- Mercury diagnosis: Mercury is not denominator- or floor-limited. Its raw adjacent defects are oppositely phased; after either phase alignment the residual remains coherently larger in the fine pair. Radial and transverse position plus radial velocity all have ratios above one.
- Mercury RTN fine/coarse ratios (position R/T/N; velocity R/T/N): 1.15944/1.0575/1.49311; 1.07193/1.64662/1.41268.
- Full-history Richardson alignment (global cosine/projection; Mercury cosine/projection): 0.0289584/0.015972; -0.907397/-0.972653. Both orders are ORDER_NOT_IDENTIFIABLE.
- Mean-anomaly-stripped global ratio: 1.86006.
- Mean-longitude-stripped global ratio: 1.78253.
- Venus first-10-kyr argument-of-periapsis and coordinate-free periapsis-direction differences: 8.30345e-08 rad and 8.42937e-08 rad; its full-history eccentricity-vector ratio is 0.252803.
- IAS15 overlap is limited to 10000 years and is not extrapolated.

## Persistent nonphase evidence

| Body | Nonphase fine/coarse | E-vector fine/coarse | h-vector fine/coarse |
| --- | ---: | ---: | ---: |
| mercury barycenter | 0.846452 | 0.828756 | 1.44477 |
| venus barycenter | 0.252799 | 0.252803 | 0.266742 |
| earth barycenter | 0.273833 | 0.272603 | 0.268362 |
| mars barycenter | 0.937438 | 0.960045 | 1.43741 |
| jupiter barycenter | 1.34603 | 1.34137 | 1.29234 |
| saturn barycenter | 1.49279 | 1.47328 | 1.30929 |
| uranus barycenter | 1.8094 | 1.80976 | 1.78869 |
| neptune barycenter | 1.81018 | 1.81001 | 1.80357 |
| pluto barycenter | 1.81008 | 1.8102 | 1.81079 |

The outer-planet nonphase vectors remain above one through repeated late windows, survive both phase alignments, lie far above reconstruction and stored-output floors, and retain positive coherent direction. This is the evidence that distinguishes the primary result from a phase-only or pointwise-predictability explanation.

## Smallest next step

Preregister two 100 kyr full-M0 controls at 0.25 and 0.125 day under one preselected alternative WHFast configuration. Compare their coordinate-free nonphase defects with the frozen baseline prefixes and with existing IAS15 only over its validated 10 kyr overlap. Use the result only to decide whether that configuration warrants qualification; do not assume 0.125 day or the alternative configuration is preferable.

Step 3e1 neither validates a production timestep nor authorizes Stage 4.
