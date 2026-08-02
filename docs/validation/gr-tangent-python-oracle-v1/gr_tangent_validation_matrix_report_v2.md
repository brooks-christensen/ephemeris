# GR Tangent Validation Matrix V2

## Overall Verdict
Status: **READY_FOR_C_PORT**

JSON summary: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_tangent_validation_matrix_summary_v2.json`
Expected stages: 11
Observed readable stages: 11

## Required-Stage Results
| Stage | Result | Source result |
| --- | --- | --- |
| existing_1myr_smoke_audit | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/existing_1myr_smoke_audit.json` |
| monitor_process_tree_audit | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/monitor_process_tree_audit/monitor_process_tree_audit.json` |
| dynamic_gr_tangent_oracle | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/dynamic_gr_tangent_oracle/dynamic_gr_tangent_oracle_summary.json` |
| newtonian_zero_limit_100kyr | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/newtonian_zero_limit_100kyr/newtonian_zero_limit_100kyr_summary.json` |
| gr_100kyr_1d_seed12345 | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_1d_seed12345/gr_tangent_summary_gr_100kyr_1d_seed12345.json` |
| gr_100kyr_1d_seed67890 | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_1d_seed67890/gr_tangent_summary_gr_100kyr_1d_seed67890.json` |
| seed_comparison | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/seed_comparison/gr_100kyr_1d_seed_comparison.json` |
| gr_100kyr_0p5d_seed12345 | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_0p5d_seed12345/gr_tangent_summary_gr_100kyr_0p5d_seed12345.json` |
| timestep_comparison | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/timestep_comparison/gr_100kyr_timestep_comparison.json` |
| physical_gr_trajectory_comparison_100kyr | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/physical_gr_trajectory_comparison_100kyr/physical_gr_trajectory_comparison_100kyr_summary.json` |
| gr_checkpoint_resume_equivalence_20kyr | passed | `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_checkpoint_resume_equivalence_20kyr/gr_checkpoint_resume_equivalence_20kyr_summary.json` |

## Tangent-Oracle Accuracy Summary
Best relative norm error across oracle groups: 7.04304e-07.
Best direction cosine across oracle groups: 1.

## Newtonian Zero-Limit Summary
Max variation norm relative difference: 0.
Minimum variation direction cosine: 1.

## Seed Comparison
Seed classifications: regular_likely and regular_likely.
Final MEGNO values: 2.01075 and 1.9819.

## Timestep Comparison
Timestep classifications: regular_likely and regular_likely.
0.5-day final LCN: -9.90848e-15.

## Physical-Trajectory Comparison
Max paired GR-minus-Newtonian difference: 5.28486e-05.
Max paired relative difference: 6.04693e-06.

## Checkpoint/Resume Comparison
Physical scaled phase difference: 0.
Tangent scaled phase difference: 0.

## Monitoring Verification
Monitor sample count: 18.

## Diagnostic-Semantics Notes
- Newtonian energy component is a consistency diagnostic and is not total conserved GR energy error for the custom Hamiltonian.
- The full-system Mercury apsidal drift includes Newtonian planetary secular perturbations plus GR and is not the isolated GR excess.
- Variation API smoke metadata reports a standalone API check; production_metadata is authoritative for production particle counts.

## Remaining Caveats
- Finite-time tangent and MEGNO diagnostics are not asymptotic Lyapunov proofs.
- Timestep comparison does not claim convergence of total GR energy from the Newtonian component diagnostic.
- Isolated Sun-Mercury GR precession remains the analytic approximately 43 arcsec/century validation; full-system totals must not be compared directly with that value.

## Next Authorized Action
Compiled-C port and C-versus-Python validation only.

## Source Result Paths
- existing_1myr_smoke_audit: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/existing_1myr_smoke_audit.json`
- monitor_process_tree_audit: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/monitor_process_tree_audit/monitor_process_tree_audit.json`
- dynamic_gr_tangent_oracle: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/dynamic_gr_tangent_oracle/dynamic_gr_tangent_oracle_summary.json`
- newtonian_zero_limit_100kyr: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/newtonian_zero_limit_100kyr/newtonian_zero_limit_100kyr_summary.json`
- gr_100kyr_1d_seed12345: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_1d_seed12345/gr_tangent_summary_gr_100kyr_1d_seed12345.json`
- gr_100kyr_1d_seed67890: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_1d_seed67890/gr_tangent_summary_gr_100kyr_1d_seed67890.json`
- seed_comparison: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/seed_comparison/gr_100kyr_1d_seed_comparison.json`
- gr_100kyr_0p5d_seed12345: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_100kyr_0p5d_seed12345/gr_tangent_summary_gr_100kyr_0p5d_seed12345.json`
- timestep_comparison: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/timestep_comparison/gr_100kyr_timestep_comparison.json`
- physical_gr_trajectory_comparison_100kyr: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/physical_gr_trajectory_comparison_100kyr/physical_gr_trajectory_comparison_100kyr_summary.json`
- gr_checkpoint_resume_equivalence_20kyr: `/home/peacelovephysics/ephemeris/output/stability/gr_tangent_validation_matrix_v1/gr_checkpoint_resume_equivalence_20kyr/gr_checkpoint_resume_equivalence_20kyr_summary.json`
