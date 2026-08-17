# M0 Step 3g1d Blocked Closeout

## Result

- Final status: `STEP3G1D_BLOCKED`
- Primary finding: `SYNTHETIC_CONSERVATIVE_INTERACTION_KICK_NOT_QUALIFIED`
- Verification envelope: `NOT_ESTABLISHED`
- Blocking condition: `MANIFEST26_PROVENANCE_HASH_TABLE_INVALID`

Manifest 26 is permanently closed as blocked. Its preregistered historical SHA-256 table contains 12 incorrect entries. The historical files themselves are unchanged, and their current hashes agree exactly with the authoritative Manifest 25 controlled baseline.

## Provenance Finding

| Historical manifest | Manifest 26 value | Actual and Manifest 25 value |
| --- | --- | --- |
| `13_m0_integrator_roundoff_diagnosis_v1.json` | `215bc071b7c3504a3d0ec8c25dfa075291e49759873f638740698c9eb3a17240` | `215bc0713aa17e2ff0986077cfadee8b40f520708d352593a6ffc04da9417240` |
| `14_m0_reversibility_roundoff_gate_v1.json` | `6d74db1ad9129389a0b5135383218d5e8e64fe2f69260a67015a727296ac7188` | `6d74db1a0a0a8d96a00295d0a4279d91c7d1811c194546de11f328c612f97188` |
| `15_m0_integrator_roundoff_diagnosis_continuation_v1.json` | `9fc06c7e0147efadf57f964823d1bd4c5f3735ca06f0d740bdbd1d550c52aafe` | `9fc06c7e0ae5811afc644df6f08ff75bfae52172eb5fc41d400c2bef6835aafe` |
| `16_m0_ias15_phase_reference_v1.json` | `3895ead3955956011895458a121ea97376f47365c417a227de14c51278391ec6` | `3895ead3f3641463d320143e2fa46abb6293d3f9c0461af4cc3f6d90fb591ec6` |
| `17_m0_step3e_whfast_0125d_convergence_v1.json` | `978ab813d75c00eb6c304a4e8c6c65b63f8d57fa58bd06aa3f18888af8c8f89b` | `978ab813979ea6c728e113c1f473afabb54cd553d2097dd9d26add8391f5589b` |
| `18_m0_step3e1_offline_state_diagnosis_v1.json` | `088b55fa0f7914de730584935d0a52ec973a5bd2c399b44da7afd63ba18921d` | `088b55fa40cf0ccb7fa50f42d41f017fcdf560d7a8f7a7dfa69678717544021d` |
| `19_m0_step3f0_whfast_configuration_audit_v1.json` | `28d8c39086b20636e43ecc872cb78f4aac87d54dfc3e6ac4bbd4a47e9f64b521` | `28d8c390690be7c1b98cfea1b5e22615926dc149b6e7c88640d8db4e5074b521` |
| `20_m0_step3f1_two_lane_architecture_screen_v1.json` | `46b19b54e543fccebeaf28dd0ccb7aa336442e1ac8d864d8eaee01f49d31a65c` | `46b19b54f278f6e174f7aa10d6fa6e2ed68e25394fc7839d1acecbcde601a65c` |
| `21_m0_step3g0_verification_architecture_audit_v1.json` | `b81495b1fb4d26baec33f00cb607ca1ffeced2293c22bf1f054542a3de9614f9` | `b81495b1b561ba10bddb9002912c9fdd03cb37113da29ecd479f03cb1e9614f9` |
| `22_m0_step3g1a_v2_foundation_v1.json` | `185948f12588211b77bf0422d50958ce11fb7d23c6e7dd7398450f129f0d4dd8` | `185948f1e95987f28aa80decbc0b423fa8455f31e43a661a60202d9abc3d4dd8` |
| `23_m0_step3g1a_v2_foundation_requalification_v1.json` | `c6dc3f3a9f22264173b8418150d271159e4337fa4073de1cb4aa73a0b7e9004c` | `c6dc3f3a42cb36d433279b0a4158a29ee3b40e2def766530ea7c151939e9004c` |
| `24_m0_step3g1b_canonical_jacobi_tangent_primitives_v1.json` | `9cd388ac9e2baf308dd90c0b223c4b6123696f5fc50c65ff1e7ff229a2e43f4c` | `9cd388aca62dcbeb8ae81c3485c62830f845798f2ca3edfcc6d64d702f743f4c` |

Manifest 25 itself is correct at `07be6ac74d30ee843abec0568ba0073847e719891d2248d0c52dccd9bb326dbc`. Manifest 26 remains unchanged at `5a7f610c232b336bf468dbda4ca1b8d52603fb4ea9278ec0b89314c20d046180`, committed as `0e646f0dbc79f0c04b68514025b1a480e7a8d773`.

## Diagnostic Evidence

The exact static safety audit passed with 108 literal nodes and no forbidden import or subprocess findings. The subsequent pre-artifact core invocation was nonqualifying diagnostic evidence only: 84 passed and 18 failed out of 102.

Two findings are preserved for a separately preregistered corrective completion:

1. Binary64 center-of-mass closure needs a mathematically derived bound, with raw residual and bound recorded and the physical and tangent projections applied consistently.
2. Step 3g1a, Step 3g1b, and Step 3g1c regression groups need fresh guarded subprocess isolation.

No qualification artifact campaign was run. The partial implementation, tests, qualification helper, runner, and reporting helper are retained as reviewable blocked-work evidence; they are not qualified by this closeout.

## Controlled Recovery

The untracked `kick.py.orig` sidecar was 22,582 bytes with SHA-256 `ffdfac77f7ddb911b6e112f43d344204a528b5881444fcba7edf8b4c526f0aba` and was byte-identical to the unchanged target. The untracked `kick.py.rej` sidecar was a 982-byte rejected unified diff with SHA-256 `8dd49a6ccd7be1ff17424639beb629eaddf1c87e69e73bb8cc956790917f55c4`; its fixed-`64 ulp` proposal was not applied. Neither sidecar was tracked or referenced. Exactly those two files were removed.

No physical force or JVP, integration, REBOUND/REBOUNDx operation, trajectory, archive, production run, or tag operation was executed. Protected kernels, qualified Step 3g1a/3g1b/3g1c files, historical manifests, artifacts, archives, trajectories, dependencies, and git history remain unchanged.
