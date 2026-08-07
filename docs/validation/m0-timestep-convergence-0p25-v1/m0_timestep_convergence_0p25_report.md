# M0_0P5DAY_NOT_CONVERGED

One fresh 0.25-day 1 Myr lane was compared with the unchanged 1-day and 0.5-day Step 3 artifacts.

## Runs

| Run | Step | Runtime | Throughput | Samples | State rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| m0_conv_1d_1myr_s12345 | 1 d | 1301.364 s | 768.425 yr/s | 10001 | 100010 |
| m0_conv_0p5d_1myr_s12345 | 0.5 d | 2393.812 s | 417.744 yr/s | 10001 | 100010 |
| m0_conv_0p25d_1myr_s12345 | 0.25 d | 4921.371 s | 203.195 yr/s | 10001 | 100010 |

## Criteria

| Criterion | Result |
| --- | ---: |
| physical_state | PASS |
| mercury_perihelion_rate | PASS |
| eccentricity_history | PASS |
| semimajor_axis_history | FAIL |
| tangent | PASS |
| megno | PASS |
| lcn | PASS |
| corrected_energy | FAIL |
| angular_momentum | PASS |
| run_integrity | PASS |

## Key Metrics

- Global scaled physical RMS: coarse `0.00245640453639`, fine `0.000599167601081`, ratio `0.243920572611`.
- Mercury mean-perihelion-rate fine-pair difference: `1.55063617058e-07` arcsec/century.
- Final tangent cosine: `0.999999956669`; direction RMS coarse/fine `0.000954709175919` / `0.000209784126428`.
- Final MEGNO difference: `8.14759938628e-06`.
- Final accumulated LCN difference: `2.5725846709e-13`.
- Worst fine-pair semimajor-axis difference: venus barycenter `1.14196873005e-07` at `993700` years (limit `1e-07`).

## Corrected Energy

| Run | Maximum | RMS | P99 | Fitted change over 1 Myr |
| --- | ---: | ---: | ---: | ---: |
| m0_conv_1d_1myr_s12345 | 1.71820003919e-10 | 6.33915373502e-11 | 1.34508201215e-10 | 1.43917226581e-10 |
| m0_conv_0p5d_1myr_s12345 | 2.96666364955e-10 | 1.63116276543e-10 | 2.80748636934e-10 | 2.89548318171e-10 |
| m0_conv_0p25d_1myr_s12345 | 5.77876207139e-10 | 3.32581788951e-10 | 5.69616050926e-10 | 5.77548936344e-10 |

## Integrity

- Manifest SHA-256: `c589820f11f6a7171d3b3b0f852ff58c9b7184b9c77426f9a61174a2163112da`.
- Reused Step 3 artifacts unchanged: `True`.
- Protected files unchanged: `True`.
- No prior convergence lane or 10 Myr integration was launched.

## Failures

- semimajor_axis_history
- corrected_energy

## Next Action

Do not request a 0.125-day lane. First recompute corrected energy epoch-by-epoch from the existing 0.5-day and new 0.25-day state artifacts using compensated or extended-precision summation and compare it with recorded telemetry.
