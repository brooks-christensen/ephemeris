# M0 Step 3f0 WHFast Configuration Audit

## Status

- Final status: **STEP3F0_CONFIGURATION_AUDIT_COMPLETE**
- Primary finding: **COMBINED_LANE_CAPABILITY_CONSTRAINT_CONFIRMED**
- Integration steps: **0**
- Force evaluations: **0**
- Historical archives modified: **0**
- Manifest 18 remains: **TRUE_NONPHASE_NONCONVERGENCE**, scoped to the historical combined standard-kernel physical+tangent/MEGNO lane.

## Answer First

No material historical setting mismatch or undocumented effective default was found. The completed combined lanes consistently used REBOUND 4.6.0's supported standard-kernel Jacobi tangent map with `corrector=0`, `corrector2=0`, `safe_mode=1`, and `keep_unsynchronized=0`. Those settings match both the production runner and restored archives.

A material capability constraint is confirmed. REBOUND 4.6.0 rejects nonstandard WHFast kernels whenever native variational particles are present. Consequently, a single native physical+tangent/MEGNO lane cannot also use the literature's WHCKL-style physical kernel. Lane separation is technically necessary to use such a physical kernel while preserving REBOUND's native variation/MEGNO machinery. It is not necessary merely to enable a standard-kernel corrector or `safe_mode=0, keep_unsynchronized=1`; both remain feasible in the combined lane.

This inspection does not establish that the capability constraint caused the completed convergence anomaly. Manifest 15's physical-only current-sync controls reproduced the combined-lane behavior, while minimal synchronization did not materially improve both timesteps. Manifest 16's `SYSTEMATIC_WHFAST_STEP_BIAS` remains historical evidence, not a configuration defect newly proved here.

## Effective Configuration

The deterministic long-form matrix contains 777 rows: 21 lanes x 37 settings. Its SHA-256 is `06864e3d40f0b65c6dc68193fbbe02fe4420ad4974dee28d116d98c200aef3dd`. Values are labeled `DIRECT`, `CORROBORATED`, `INFERRED`, or `UNAVAILABLE`; absent archive values are never silently replaced with defaults.

The guarded zero-step construction recovered:

- before MEGNO: 10 real particles, no variations, Jacobi/default kernel, no correctors, `safe_mode=1`, `keep_unsynchronized=0`;
- after `init_megno(seed=12345)`: 20 particles, 10 first-variation particles in one full-system block, and unchanged WHFast settings;
- after GR attachment: the validated C callback was installed as a direct C function pointer, `force_is_velocity_dependent=0`, with zero callback invocations;
- the stored RNG field is the advanced generator state, so the requested seed is carried by configuration provenance rather than inferred from an archive.

## Source Findings

| Evidence | Finding | Historical application |
|---|---|---|
| `python_accessor_semantics` | The Python accessors read current C state or particle arrays and do not call the WHFast synchronization routine. | Historical scientific samples are nevertheless synchronized because the runner calls integrate for every target and REBOUND synchronizes before integrate returns. |
| `whckl_shortcut` | The WHCKL shortcut selects WHFast with the lazy kernel and first symplectic corrector order 17. | This is a concrete candidate for a future physical-only lane; it cannot be selected in a native variation/MEGNO lane because the nonstandard kernel is rejected. |
| `whfast_defaults` | Defaults are Jacobi coordinates, standard kernel, no correctors, safe_mode=1, keep_unsynchronized=0, synchronized state, and no forced coordinate recalculation. | Matches the three combined-lane archives and Manifest 13 current-sync controls. |
| `variation_capability` | WHFast first variations require Jacobi coordinates and the standard kernel; nonstandard kernels are rejected rather than silently downgraded. | A native tangent/MEGNO lane cannot select WHCKL or another nonstandard kernel in REBOUND 4.6.0. |
| `corrector_capability` | Corrector orders 3, 5, 7, 11, and 17 are accepted in Jacobi or barycentric coordinates, and Jacobi variation blocks are transformed during synchronization. | Correctors remain technically feasible for the standard-kernel combined lane, although none was requested historically. |
| `megno_per_step_sync` | With variations, WHFast constructs synchronized inertial x/v/a every timestep for MEGNO. keep_unsynchronized=1 preserves and restores the internal map around that calculation. | Historical combined lanes used safe_mode=1, so they already synchronized each step before the MEGNO-specific path. |
| `safe_keep_semantics` | safe_mode=1 and keep_unsynchronized=1 are incompatible. safe_mode=1 recalculates map coordinates and synchronizes every step; safe_mode=0 with keep_unsynchronized=1 retains the internal map. | Manifest 15 tested both current-sync and minimal-sync physical-only controls; minimal synchronization did not supply a material causal improvement. |
| `output_and_exact_finish` | exact_finish_time=1 shortens a final step only when needed, restores dt afterward, and integrate synchronizes before returning. | The 0.5, 0.25, and 0.125 day lanes divide the 100-year output cadence exactly; the older 2-day lane did not and received a one-day shortened endpoint step at each output. |
| `archive_semantics` | Archives serialize particles, variations, MEGNO state, WHFast map state, p_jh, kernel, coordinates, correctors, and synchronization flags without first forcing synchronization. | Existing archive hashes remained unchanged during direct restoration and their effective settings match the manifests. |
| `callback_restore` | Function pointers are flagged but not serialized and must be reset after restore. | The production resume path explicitly reattaches the validated compiled callback after archive validation. |
| `gr_potential_contract` | gr_potential is position-only, preserves WHFast splitting, reproduces perihelion precession, has a documented mean-motion error of order GM/(a c^2), and uses the validated -3 G^2 M^2 m/(c^2 r^2) potential. | The custom callback implements the same physical acceleration and potential plus its analytic first-variation Jacobian and central response. |

### Synchronization and outputs

`safe_mode=1` synchronizes and reconstructs coordinates every timestep. Native WHFast variations also need synchronized inertial position, velocity, and acceleration for MEGNO every timestep. With `safe_mode=0, keep_unsynchronized=1`, that MEGNO synchronization is temporary: REBOUND caches and restores its internal Jacobi map. Thus, continuous MEGNO does impose synchronized inertial construction, but it does not inherently require discarding the unsynchronized map.

The Python particle, energy, orbit, MEGNO, and Lyapunov accessors do not independently establish synchronization. In the historical runner that distinction is harmless at scientific samples because every `Simulation.integrate(target, exact_finish_time=1)` synchronizes before returning. Scientific telemetry is therefore read from synchronized particle arrays.

`exact_finish_time=1` changes the step only when an output target is not divisible by the fixed step. A 100-Julian-year cadence is exactly divisible by 1, 0.5, 0.25, and 0.125 day, but not by 2 days. The older 2-day Step 3 lane therefore received a shortened one-day endpoint step at each 100-year target. This is a documented caveat for that older comparison, not a defect affecting the decisive 0.5/0.25/0.125-day evidence in Manifests 13-18.

### Checkpoints and restart

SimulationArchive heartbeat writes the live state without an implicit synchronization. Its binary schema preserves particles, variation configurations, MEGNO accumulators, WHFast coordinates/kernel/correctors, `safe_mode`, `keep_unsynchronized`, `is_synchronized`, recalculation state, and the internal `p_jh` map. Function pointers are deliberately not serialized. The production resume path loads and validates the archive, trims sidecar telemetry to the archive epoch, and then reattaches the validated callback. JSON/CSV sidecars do not mutate simulation state.

Archive files were opened read-only through existing REBOUND restoration paths and hashed before and after inspection:

| Lane | Snapshots | SHA-256 | Result |
|---|---:|---|---|
| `m0_conv_0p25d_1myr_s12345` | 11 | `1aebd60e2671a4b7b1b1dbe6453e48d10fd460daa31e167f4d86b5fe9c7d6def` | unchanged |
| `m0_conv_0p5d_1myr_s12345` | 11 | `3ee131ed27f67797ffda168ead0d1a163291610599b0f3495f9d3b4c75aeb026` | unchanged |
| `m0_diag_phys_current_sync_0p25d_100k` | 11 | `079130971f6c720090ed6247b3895429c978e23e48ba4ce1411f2ab530e02f28` | unchanged |
| `m0_diag_phys_current_sync_0p5d_100k` | 11 | `816bdb9af6553d9d010e7c3fb910e97fd68fe5e8cb309344065a2f037ef1af05` | unchanged |
| `m0_diag_phys_ias15_default_10k` | 11 | `e3ba4e01f330efc707c633d7f88c86125155646ab02991b2d74a057100755432` | unchanged |
| `m0_diag_phys_ias15_eps1e12_10k` | 11 | `de9eb3f9bc4513d77024388fb85972a8489aa644827cfcfb231b6cd38689c8e6` | unchanged |
| `m0_diag_phys_ias15_eps1e13_10k` | 11 | `a79e2957e28f25acd4b508e166c577c18e74da8162034e07babe042ac43382c2` | unchanged |
| `m0_diag_phys_min_sync_0p25d_100k` | 11 | `e78a3b50d10a18b25e08aab9d5d785d939cdbcf6d53bd623b7f44780095d6d16` | unchanged |
| `m0_diag_phys_min_sync_0p5d_100k` | 11 | `da6d3c965d79f771851a357ae18609d978183a6df4a10da84492a64dd59f9263` | unchanged |
| `m0_step3e_tangent_whfast_0125d_1myr` | 11 | `6a3248c560491ca0d8dfd35e84ac669df15161afc7728ae57b0c94b33a68f143` | unchanged |

## Custom GR Callback

The callback remains byte-identical to Manifest 19 (`c764740a...`). It is position-only, sets `force_is_velocity_dependent=0`, applies the validated `gr_potential` acceleration and central response to every real particle, and applies the analytic Jacobian to each full real-particle variation block. It reads no WHFast configuration fields and performs no synchronization or coordinate transform.

REBOUNDx 4.6.1 documents that `gr_potential` preserves the WHFast split and gets perihelion precession right while shifting mean motion by order `GM/(a c^2)`. Its potential is exactly the validated correction `-3 G^2 M_sun^2 m_i/(c^2 r_i^2)`. The historical M0 scientific model intentionally inherits that physical approximation; Step 3f0 neither changes nor revalidates it.

## Literature Alignment

| Year | Primary source | Direct application | Caveat |
|---:|---|---|---|
| 1991 | [Symplectic maps for the n-body problem](https://doi.org/10.1086/115978) | Foundational method for WHFast. | Does not specify REBOUND implementation settings. |
| 2015 | [WHFast: a fast and unbiased implementation of a symplectic Wisdom-Holman integrator for long-term gravitational simulations](https://doi.org/10.1093/mnras/stv1257) | Supports standard-kernel combined-lane design. | Does not imply support for later nonstandard kernels with variations. |
| 2019 | [High-order symplectic integrators for planetary dynamics and their implementation in REBOUND](https://doi.org/10.1093/mnras/stz2503) | Directly explains the physical-WHCKL versus native-tangent capability split. | Accuracy gains are problem- and timestep-dependent and do not establish causation here. |
| 2016 | [Second-order variational equations for N-body simulations](https://doi.org/10.1093/mnras/stw644) | Supports the analytic first-variation telemetry requirement. | The paper's second-order examples emphasize IAS15; exact WHFast constraints come from tagged source. |
| 2003 | [Phase space structure of multi-dimensional systems by means of the mean exponential growth factor of nearby orbits](https://doi.org/10.1016/S0167-2789(03)00103-9) | Defines the scientific tangent diagnostic. | Does not prescribe REBOUND synchronization or kernel settings. |
| 2020 | [REBOUNDx: a library for adding conservative and dissipative forces to otherwise symplectic N-body integrations](https://doi.org/10.1093/mnras/stz2870) | The validated gr_potential callback is position-only. | The project uses a custom analytic tangent callback, not the stock effect object. |
| 2020 | [A repository of vanilla long-term integrations of the Solar System](https://doi.org/10.3847/2515-5172/abd103) | Demonstrates a precedent for a separate canonical physical lane. | Its model, timestep, and scientific target differ from M0. |
| 2022 | [Stepsize errors in the N-body problem: discerning Mercury's true possible long-term orbits](https://doi.org/10.1093/mnras/stab3664) | Motivates successor screening with orbital and secular observables, not energy alone. | Does not diagnose the completed M0 lanes by inspection. |
| 2025 | [On the statistical convergence of N-body simulations of the Solar System](https://doi.org/10.33232/001c.154745) | Limits the interpretation of the one-trajectory Step 3 pointwise failures. | Does not waive M0's preregistered production gates. |
| 2020 | [Fundamental limits from chaos on instability time predictions in compact planetary systems](https://doi.org/10.1093/mnras/stz3402) | Supports separating pointwise reproducibility from ensemble claims. | Compact-system distributions are not a direct M0 threshold calibration. |

The literature and exact source agree on the central architectural point: high-order physical kernels are useful candidates for long-term planetary trajectories, but REBOUND's implemented nonstandard WHFast kernels do not carry native variations/MEGNO. Literature on chaotic Solar-System integrations also distinguishes statistical convergence of ensembles from pointwise agreement of one trajectory. Neither principle weakens Manifest 17 or reinterprets Manifest 18.

## Scope of Prior Results

- Manifest 13's historical `BLOCKED` result is unchanged.
- Manifest 14's `REVERSIBILITY_GATE_PASSED` result is unchanged.
- Manifest 15's `BLOCKED` result and synchronization-control evidence are unchanged.
- Manifest 16's `SYSTEMATIC_WHFAST_STEP_BIAS` primary mechanism and qualified IAS15 phase floor are unchanged.
- Manifest 17 remains `STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED`.
- Manifest 18 remains `STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE` / `TRUE_NONPHASE_NONCONVERGENCE` for the historical combined standard-kernel lane.
- No conclusion here applies automatically to all WHFast configurations or to a WHCKL physical-only lane that has not yet been run under M0.

## Successor Architecture

The evidence supports preregistering a two-lane design for any successor experiment:

1. **Canonical physical lane:** full M0 physics and validated position-only compiled-C GR callback; Jacobi coordinates; sim.integrator=WHCKL (WHFast lazy kernel with corrector 17); `safe_mode=0`; `keep_unsynchronized=1`. This lane owns canonical physical-state, energy, angular-momentum, and secular-frequency claims.
2. **Tangent diagnostic lane:** identical initial physical state, timestep, output epochs, and GR callback; standard WHFast kernel; Jacobi coordinates; native first variations and MEGNO; `safe_mode=0`; `keep_unsynchronized=1`; and corrector order 17. This lane owns tangent direction/norm, MEGNO, and finite-time LCN diagnostics, not the canonical physical trajectory.

The smallest evidence-based screening experiment is a preregistered paired 10-kyr run at one already studied timestep (0.25 day is the natural choice), 100-year scientific cadence, with two fresh lanes: physical WHCKL and standard-kernel tangent/MEGNO. Compare both against the qualified 10-kyr IAS15 phase envelope and compare their shared physical observables, corrected energy, angular momentum, secular frequencies, and callback integrity. This is a recommendation only; Step 3f0 executes no command and does not validate 0.25 or 0.125 day for production.

## Residual Questions

- Whether WHCKL materially improves the M0 secular and state defects requires a controlled integration and cannot be inferred from source inspection.
- Whether corrector order 17 and its cost are worthwhile for the custom GR splitting must be preregistered and screened.
- The physical/tangent lane divergence tolerance and interpretation of tangent statistics must be fixed before observing a successor result.
- Production promotion still requires its own preregistered evidence; no Stage 4 or 10-Myr command is provided.

## Reproducibility

- Starting commit: `35a4d3ae6a717f7d40e4c4db0bd1e78b0c169ce4`
- Manifest 19 preregistration commit: `45d3c996dfaae161aeab79fd098f0d81bc90f886`
- Manifest 19 SHA-256: `28d8c390690be7c1b98cfea1b5e22615926dc149b6e7c88640d8db4e5074b521`
- REBOUND: tag 4.6.0, commit `e3b07aa88dc4b004d82c03da070a89de5b699a2c`, git-archive SHA-256 `63354536ba3f7fb3a0365f6619b8a76b4a85c54d444230fd3efc074489b318f5`
- REBOUNDx: tag 4.6.1, commit `d5a4a2b5d28cbbd167bef3148063603ea2f2e131`, git-archive SHA-256 `1b7f3d44a6acaf224f36616f010f88511c71c4ccfb33cb45baf0389de4aaaa23`
- Matrix: `/home/peacelovephysics/ephemeris/docs/validation/m0-step3f0-whfast-configuration-audit-v1/m0_step3f0_effective_settings_matrix.csv`
- No-step guard calls: 0
