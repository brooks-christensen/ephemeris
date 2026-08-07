# STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED

Step 3e ran one preregistered 0.125-day, 1 Myr compiled-C tangent/MEGNO WHFast lane and compared it with the unchanged 0.25-day production candidate.

## Decision

- Production candidate: `0.25 day`.
- Raw physical convergence: `False`; phase-aware state result: `False`.
- Corrected-energy prediction: `True`.
- Integrity: `True`.
- No Stage 4 or 10 Myr command was provided or executed.
- Smallest follow-up: Perform an offline windowed Mercury RTN and osculating-angle decomposition of the existing 0.5-day, 0.25-day, and 0.125-day stored rows to localize the nonmonotonic Mercury state error and test whether it is secular-orientation leakage or bounded phase beating; do not start another integration.

## Runtime And Integrity

- Runtime: `8996.598708` s; throughput `111.153118` yr/s.
- Samples/state rows/archive snapshots: `10001` / `100010` / `11`.
- Callback/nonfinite counts: `2922000000` / `0`.
- Fingerprint: `1064be4cb34530ca032ac958c99ae46c33def89fd2aaea8decc82f647649f50e`.
- In-lane prefix: 90.560787 s, projected 9056.078700 s, PASS; reconstructed conservatively from the persisted pre-Popen launch UTC after correcting the row-order parser.

## Physical And Orbital Convergence

- Global scaled RMS old/new: `0.000599167601081` / `0.000330469779068`; ratio `0.551548145246`.
- Worst inner ratio: `mercury barycenter` at `1.07191551941`.
- Worst semimajor-axis relative difference: `mars barycenter` `5.09125680166e-08`.
- Worst eccentricity difference: `mercury barycenter` `1.50568232882e-08`.
- Mercury perihelion-rate difference: `1.45140461427e-07` arcsec/century.

## Failed Thresholds

- Global RMS ratio 0.551548145246 exceeds 0.5 by 0.0515481452458 at 998900 years.
- Mercury RMS ratio 1.07191551941 misses strict less-than-1 improvement by 0.0719155194074 at 998900 years.
- Phase fallback orientation: venus barycenter 8.30345456837e-08 rad versus 1e-09 rad.
- Phase/orientation ratio: uranus barycenter 0.854837076986 versus minimum 10.

## Tangent And Chaos

- Final tangent cosine: `0.999999897842`; direction RMS old/new `0.000209784126428` / `0.00020372230391`.
- Final MEGNO difference: `3.31032825596e-06`.
- Final accumulated LCN difference: `1.05042862658e-13`.

## Corrected Energy

- Recomputed 0.125-day fitted change over 1 Myr: `1.14352955646e-09`.
- Recomputed q: `3.91351662034e-19`; interval `[3.6573194312830514e-19, 4.2153891282423703e-19]`.
- Same-sign 100-kyr blocks: `10` of 10.
- Maximum prediction-envelope excess: `0`.
- Historical Step 3 trend gate (diagnostic only after manifest 16): `False`.

## Criteria

| Criterion | Result |
| --- | ---: |
| integrity | PASS |
| physical_state_raw | FAIL |
| phase_aware_interpretation | FAIL |
| physical_state | FAIL |
| semimajor_axis | PASS |
| eccentricity | PASS |
| mercury_perihelion | PASS |
| tangent | PASS |
| megno | PASS |
| lcn | PASS |
| corrected_energy | PASS |
| angular_momentum | PASS |

## Artifacts

- Manifest SHA-256: `978ab813979ea6c728e113c1f473afabb54cd553d2097dd9d26add8391f5589b`.
- New raw artifact count: `7`.
- Figure count: `4`.

## Failures

- physical_state
