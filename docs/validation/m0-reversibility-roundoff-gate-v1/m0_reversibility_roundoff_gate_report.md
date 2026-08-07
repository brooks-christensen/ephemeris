# M0 Reversibility Roundoff Gate

Final status: **REVERSIBILITY_GATE_PASSED**

## Historical Result

Manifest 13 remains **BLOCKED**. Its minimal-sync fine/coarse ratio was `8.35827019` against the frozen `4.0` ratio gate. Neither manifest 13 nor its report has been reinterpreted or modified.

## Corrected Criterion

For an autonomous symmetric map with exact integer steps, `Phi(-h)^N Phi(h)^N = I` in exact arithmetic. The return error therefore measures roundoff, synchronization/reconstruction, and implementation asymmetry; it is not a second-order truncation convergence test.

The fine/coarse ratios below are diagnostic only and do not affect validity.

Manifest 13 froze 1e-8 for global scaled Cartesian RMS but no separate metric-specific numeric limits. Before this rerun, manifest 14 applied the same original dimensionless absolute scale uniformly to every requested normalized return metric; no observed endpoint value was used to set a threshold.

| Mode | 0.5-day RMS | 0.25-day RMS | Fine/coarse | Mode pass |
| --- | ---: | ---: | ---: | --- |
| current_sync | 2.26354582396e-13 | 3.66938140314e-13 | 1.6210767 | True |
| min_sync | 6.99726008176e-14 | 5.84849903499e-13 | 8.35827019 | True |

## Cases

| Case | RMS | RMS margin | Energy rel. | Angular rel. | Callbacks | Runtime (s) | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| m0_rev_roundoff_current_sync_0p5d_365d_v1 | 2.26354582396e-13 | 9.99977364542e-09 | 7.62010029992e-16 | 3.30268204001e-16 | 1460 | 0.000343 | True |
| m0_rev_roundoff_current_sync_0p25d_365d_v1 | 3.66938140314e-13 | 9.99963306186e-09 | 5.33407020994e-15 | 2.64214563201e-15 | 2920 | 0.000573 | True |
| m0_rev_roundoff_min_sync_0p5d_365d_v1 | 6.99726008176e-14 | 9.9999300274e-09 | 1.71452256748e-15 | 9.90804612003e-16 | 1460 | 0.000257 | True |
| m0_rev_roundoff_min_sync_0p25d_365d_v1 | 5.84849903499e-13 | 9.9994151501e-09 | 1.52402005998e-14 | 7.59616869202e-15 | 2920 | 0.000396 | True |

## Endpoint And Integrity

| Case | Forward time | Return time | Steps forward/back | Callbacks observed/expected | Nonfinite |
| --- | ---: | ---: | ---: | ---: | ---: |
| m0_rev_roundoff_current_sync_0p5d_365d_v1 | 31536000 | 0 | 730/730 | 1460/1460 | 0 |
| m0_rev_roundoff_current_sync_0p25d_365d_v1 | 31536000 | 0 | 1460/1460 | 2920/2920 | 0 |
| m0_rev_roundoff_min_sync_0p5d_365d_v1 | 31536000 | 0 | 730/730 | 1460/1460 | 0 |
| m0_rev_roundoff_min_sync_0p25d_365d_v1 | 31536000 | 0 | 1460/1460 | 2920/2920 | 0 |

## Historical Comparison

| Source case | Historical RMS | New RMS | Absolute difference |
| --- | ---: | ---: | ---: |
| two_body_current_sync_0p5d | 2.26354582396e-13 | 2.26354582396e-13 | 0 |
| two_body_current_sync_0p25d | 3.66938140314e-13 | 3.66938140314e-13 | 0 |
| two_body_min_sync_0p5d | 6.99726008176e-14 | 6.99726008176e-14 | 0 |
| two_body_min_sync_0p25d | 5.84849903499e-13 | 5.84849903499e-13 | 0 |

Every case report contains the full per-body position/velocity checks, center-of-mass checks, absolute limits and margins, exact endpoints and step counts, callback totals, finite-state checks, schema/fingerprint checks, and historical comparison.

## Decision

Step 3d decisive experiment may resume: **True**.

No million-year energy-drift mechanism is classified here. No Stage 4 command is provided.
