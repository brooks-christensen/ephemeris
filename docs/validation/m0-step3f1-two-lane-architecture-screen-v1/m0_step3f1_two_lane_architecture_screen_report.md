# M0 Step 3f1 Two-Lane Architecture Screen

**Final status:** STEP3F1_TWO_LANE_SCREEN_FAILED

**Primary finding:** `BOTH_LANES_UNQUALIFIED`

## Decision

Both proposed lanes failed material assigned screens. Lane P missed its frozen IAS15 physical threshold and callback-accounting integrity gate; Lane T missed carrier-consistency and tangent/MEGNO continuity gates.

This is an architecture screen only. Manifest 17 remains `STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED`, and Manifest 18 remains historically valid for its combined standard-kernel lane. This result does not retroactively validate the 0.25-day timestep. Stage 4 and the 10 Myr production experiment remain unauthorized.

## Exact Lane Contracts

| Lane | Role | Kernel | Corrector | Variations | MEGNO | safe_mode | keep_unsynchronized | Steps | Samples | State rows | Archives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P | candidate canonical physical trajectory for architecture screening only | lazy | 17 | False | False | 0 | 1 | 14610000 | 101 | 1010 | 11 |
| T | native first-variation, MEGNO, and finite-time LCN diagnostics | default | 0 | True | True | 0 | 1 | 14610000 | 101 | 1010 | 11 |

Lane T was freshly executed. The frozen historical prefix used `safe_mode=1` and `keep_unsynchronized=0`, so it was not contract-identical and was used only as the preregistered one-factor synchronization control.

Both lanes used the same DE431 state, Sun-through-Pluto M0 physics, 0.25-day step, 10,000-Julian-year interval, integer targets, and compiled GR callback. Lane P used physical-only WHCKL (`lazy`, order-17 corrector); Lane T used standard WHFast with native first variations, the compiled analytic Jacobian, MEGNO, and finite-time LCN.

## Integrity And Runtime

- Lane P: 29,226,432 callback evaluations, zero nonfinite results, 25.673 s wall time, 389.516 simulated years/s.
- Lane P restart: exact physical closure `True`, variation closure `True`, MEGNO/LCN closure `True/True`, archive hash unchanged.
- Lane T: 14,610,000 callback evaluations, zero nonfinite results, 48.534 s wall time, 206.042 simulated years/s.
- Lane T restart: exact physical closure `True`, variation closure `True`, MEGNO/LCN closure `True/True`, archive hash unchanged.

The live WHFast map remained unsynchronized after every positive-time output (`is_synchronized=0` before and after sampling). Output diagnostics used a disposable simulation copy and did not change the live particle state, step count, or time. Archive inspection did not mutate either archive.

Lane P observed 29,226,432 callback evaluations versus the preregistered 29,223,232. The exact +3,200 difference is 32 extra order-17 corrector evaluations for each of 100 diagnostic copies. Its restart similarly observed 292,264 versus 292,232. These remain failed integrity gates despite exact state closure. Lane T callback accounting and restart accounting passed; its final sidecars were recovered offline after a post-assertion NameError, without rerunning the trajectory.

The frozen IAS15 default reference was used only at its 101 stored 100-year timestamps over 0-10 kyr. Its characterized tolerance envelope was carried forward; it was not treated as exact truth beneath that floor. All frozen IAS15 and historical artifact hashes passed before and after analysis.

## Physical State

| Comparison | Global scaled RMS | Dominant body | Body RMS | Squared-error contribution | Mean-anomaly stripped RMS | Mean-longitude stripped RMS |
|---|---:|---|---:|---:|---:|---:|
| P_vs_IAS15 | 6.397111864e-08 | mercury barycenter | 1.840955441e-07 | 0.828168 | 9.794113909e-10 | 2.204603027e-11 |
| T_vs_IAS15 | 2.155813468e-06 | earth barycenter | 4.598406400e-06 | 0.454980 | 4.811545675e-08 | 7.434484905e-10 |
| T_vs_P | 2.192768179e-06 | earth barycenter | 4.565197046e-06 | 0.433445 | 4.856543791e-08 | 7.517245194e-10 |
| T_vs_old_T | 8.755469825e-08 | mercury barycenter | 2.104641901e-07 | 0.577827 | 1.354392115e-08 | 2.894736669e-10 |

- Lane P passed the global IAS15 scaled-state limit but failed Saturn: 3.453510540e-09 versus 2.182667455e-09.
- Lane T failed preregistered outer-body carrier limits against IAS15/Lane P and failed the historical synchronization-control limits for Sun, Jupiter, Uranus, Neptune, and Pluto.
Raw Cartesian differences were screened but were not the sole architecture veto. RTN decompositions, both preregistered shared-phase reconstructions, coordinate-free orbital histories, and ten fixed 1-kyr windows were evaluated for every planet.

## Orbital And Secular Results

- `P_vs_IAS15`: maximum relative semimajor-axis difference 9.825383662e-12 (pluto barycenter); maximum eccentricity-vector difference 1.099603373e-11 (saturn barycenter); Mercury perihelion-rate difference 7.543690117e-09 arcsec/century; persistent nonphase failures 1.
- `T_vs_IAS15`: maximum relative semimajor-axis difference 3.002159444e-10 (saturn barycenter); maximum eccentricity-vector difference 4.844189707e-10 (saturn barycenter); Mercury perihelion-rate difference 1.399989742e-07 arcsec/century; persistent nonphase failures 1.
- `T_vs_P`: maximum relative semimajor-axis difference 2.980479919e-10 (saturn barycenter); maximum eccentricity-vector difference 4.775713176e-10 (saturn barycenter); Mercury perihelion-rate difference 1.475426643e-07 arcsec/century; persistent nonphase failures 1.
- Frozen coordinate-free limits were 1.0e-08 for relative semimajor axis and 1.0e-08 for eccentricity, eccentricity-vector norm, inclination components, and orbital-plane direction; the Mercury pair-rate limit was 0.001 arcsec/century.
- Secular-frequency estimates and absolute Mercury rates remain contextual under the frozen 10-kyr/cadence policy. The binary64 floor of the frozen arccos plane-angle estimator is an essential unresolved method issue; pair-rate differences were still screened exactly as preregistered.

## Conservation

- Lane P: corrected-energy max |drift| 3.771973465e-13, fitted 10-kyr change 2.074541198e-13; angular-momentum max |drift| 2.315770228e-13.
- Lane T: corrected-energy max |drift| 3.184104415e-12, fitted 10-kyr change -3.720699773e-13; angular-momentum max |drift| 1.790967692e-13.
- Pair corrected-energy history max difference: 3.507283376e-12.
- Pair angular-momentum history max difference: 3.721039671e-13.
- Corrected energy was independently recomputed from every stored state row using the frozen Newtonian plus GR-potential definition; the recomputed and recorded histories agreed exactly in binary64.

## Tangent And Chaos Diagnostics

- Final direction cosine against the historical tangent prefix: 0.997759185619; direction discrepancy RMS: 5.567299988e-02.
- Maximum log tangent-norm difference: 1.000056811e-02; fitted new-lane log-norm growth: 3.391051341e-04 per year.
- Final MEGNO: 2.20948273024; final/history RMS MEGNO differences: 2.037540716e-01/1.895241681e-01.
- Final finite-time LCN: -2.444478535e-13 1/year; final/history RMS accumulated-LCN differences: 1.449634768e-09/1.241738725e-07.
- This tests numerical continuity of the already-validated tangent implementation; it does not establish long-duration chaos evidence.

## Classification Evidence

- `integrity`: `FAIL`.
- `raw_P_vs_IAS15`: `FAIL`.
- `carrier_T_vs_IAS15`: `FAIL`.
- `carrier_T_vs_P`: `FAIL`.
- `sync_T_vs_old_T`: `FAIL`.
- `orbit_P_vs_IAS15`: `FAIL`.
- `orbit_T_vs_IAS15`: `FAIL`.
- `orbit_T_vs_P`: `FAIL`.
- `orbit_T_vs_old_T`: `FAIL`.
- `mercury_perihelion_pairs`: `PASS`.
- `conservation`: `PASS`.
- `tangent`: `FAIL`.
- `lane_P_assigned`: `FAIL`.
- `lane_T_assigned`: `FAIL`.

Evidence supports BOTH_LANES_UNQUALIFIED and the emitted status STEP3F1_TWO_LANE_SCREEN_FAILED. Frequency and absolute-rate estimates are nonessential. The plane-angle estimator remains an essential unresolved method issue, but it does not change the classification because Lane P raw physical and integrity gates and Lane T carrier/tangent gates fail independently.

## Artifacts

The deterministic metrics table contains every coarse gate value and per-body coordinate-free result. The lane-configuration table records both effective contracts. Four PNG figures show physical defects, coordinate-free discrepancies, conservation histories, and tangent/MEGNO/LCN continuity. Raw trajectories and operational sidecars remain under the ignored Step 3f1 output root.

## Smallest Successor Action

Stop and perform a separately preregistered source-only and offline audit of the shared diagnostic-copy/synchronization representation against the frozen IAS15 state rows. This requires no new scientific integration and must resolve the order-17 copy callback accounting and coordinate-direction estimator floor before any architecture retest.
