# M0_1DAY_NOT_CONVERGED

Three serial 1 Myr compiled-C M0 integrations were compared at matched 100-year samples.

## Runs

| Run | Step | Runtime | Throughput | Samples | State rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| m0_conv_2d_1myr_s12345 | 2 d | 972.441 s | 1028.340 yr/s | 10001 | 100010 |
| m0_conv_1d_1myr_s12345 | 1 d | 1301.364 s | 768.425 yr/s | 10001 | 100010 |
| m0_conv_0p5d_1myr_s12345 | 0.5 d | 2393.812 s | 417.744 yr/s | 10001 | 100010 |

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

- Global scaled physical RMS: coarse `0.0109760097936`, fine `0.00245640453639`, ratio `0.223797589705`.
- Mercury mean-perihelion-rate fine-pair difference: `9.21876107896e-07` arcsec/century.
- Final tangent cosine: coarse `0.999886140037`, fine `0.999998943901`.
- Final MEGNO difference: `3.77598769785e-05`.
- Final accumulated LCN difference: `1.21837677178e-12`.
- Worst 1 Myr angular-momentum drift: `1.20155564738e-10` for m0_conv_0p5d_1myr_s12345 at `999300` years, `19.305x` the accepted 100 kyr 1-day scale but below the `1e-09` bound; the horizons differ by a factor of ten.

## Integrity

- Manifest SHA-256: `5a45bd7967eeba7694a2bf076af33290cf113eefa5379ec439773fda0ad04363`.
- Protected files unchanged: `True`.
- No 10 Myr integration was launched.

## Failures

- semimajor_axis_history
- corrected_energy
- Worst fine-pair semimajor-axis difference: venus barycenter `4.53192926828e-07` at `991900` years (limit `1e-07`).
- Worst corrected-energy maximum: m0_conv_0p5d_1myr_s12345 `2.96666364955e-10` at `998700` years; all runs remain below the absolute bound, but reduction/trend rules fail.

## Next Action

Do not launch 10 Myr. Preregister one 0.25-day, 1 Myr lane and compare it with the existing 0.5-day lane, retaining the frozen semimajor-axis and corrected-energy criteria.
