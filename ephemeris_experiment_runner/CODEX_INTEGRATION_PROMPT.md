# Codex integration prompt: safe GR-compatible Benettin worker

Use this prompt from the root of `/home/peacelovephysics/ephemeris` after copying this package into the repository.

```text
Implement a production-quality, GR-compatible two-trajectory Benettin path for the ephemeris stability project and integrate it with the ephemeris-experiment-runner manifests.

Context:
- Native REBOUND MEGNO/Simulation.lyapunov() is trusted for Newtonian runs only.
- REBOUNDx forces receive real particles rather than variational particles, so do not use native REBOUND variational particles for GR claims.
- The accepted Newtonian benchmark is the completed full/full_with_pluto WHFast ladder: regular-looking through 10 Myr, positive by 100-200 Myr, and robust chaotic candidates by 500 Myr.
- Production GR should initially use REBOUNDx gr_potential with WHFast because it is position-dependent, fast, and symplectic. IAS15 + REBOUNDx gr is the short-duration trajectory oracle.

Requirements:
1. Preserve all existing CLI behavior and tests.
2. Either enhance long_term_stability_cli.py or add a focused gr_benettin_cli.py. Reuse existing body construction and unit handling rather than duplicating ephemeris logic.
3. Create two completely independent REBOUND simulations from the same physical initial state:
   - reference trajectory
   - shadow trajectory with the configured small phase-space perturbation
4. Attach the selected physics independently and identically to both simulations. For GR, create a separate REBOUNDx Extras object and force instance for each simulation. Never share a force object between simulations.
5. Support at least:
   - none
   - gr_potential with WHFast
   - gr with IAS15 for short validation
6. Implement periodic Benettin renormalization in a documented scaled phase-space norm. Remove center-of-mass translation and velocity modes before measuring/renormalizing the deviation. Preserve the reference trajectory exactly; only reposition the shadow trajectory during renormalization.
7. Accumulate the finite-time exponent as sum(log(norm_before/norm_target)) divided by elapsed fit time. Track pre-fit and fit-window accumulators separately.
8. Write incremental progress atomically during the run, not only at completion:
   benettin_progress_<tag>.csv
   Columns must include at least:
   - time_years
   - renorm_count
   - separation_norm_before
   - target_separation_norm
   - accumulated_log_growth
   - finite_time_lcn_1_per_year
   - fit_start_years
   - fit_elapsed_years
   - seed
   - step_days
   - integrator
   - gr_model
   - reference_relative_energy_error when meaningful
   - shadow_relative_energy_error when meaningful
   - reference_relative_angular_momentum_error
   - shadow_relative_angular_momentum_error
9. Atomically update run_status_<tag>.json at the same cadence with current simulated time, percent complete, elapsed wall time, recent simulated-years-per-wall-second rate, ETA, latest LCN, checkpoint paths, warning list, and configuration hash.
10. Print one concise progress line at startup and at a configurable interval. Include experiment name, simulated time, percent, elapsed, recent rate, ETA, LCN, and invariant drift. Do not use an animated progress bar when --no-progress-bar is set.
11. Add safe dual-trajectory checkpoints. A checkpoint must preserve:
   - both REBOUND simulations
   - REBOUNDx configuration identity
   - current time
   - target perturbation norm
   - accumulated logarithmic growth
   - fit-window accumulated growth and elapsed time
   - renormalization count
   - RNG seed/state if still relevant
   - exact normalized CLI configuration and a cryptographic configuration hash
12. Resume must refuse to run when any physical/numerical parameter differs from the checkpoint. On valid resume, truncate incremental CSV rows after checkpoint time before appending.
13. Add CLI options with clear help text for status cadence, progress-file cadence, dual-checkpoint cadence/directory, and resume-latest. Keep existing --with-lyapunov --lyapunov-method two_trajectory options compatible if practical.
14. Add tests without launching long integrations:
   - CLI/help/compile tests
   - equal-state GR acceleration symmetry between independently configured reference/shadow simulations
   - two-body Newtonian regular control
   - uninterrupted versus checkpoint/resume equivalence for a short deterministic run
   - output truncation at resume
   - configuration mismatch refusal
   - Mercury GR perihelion-precession smoke against the analytic 1PN value
   - gr_potential WHFast trajectory comparison against gr + IAS15 over a short interval
15. Update the three JSON manifests in ephemeris_experiment_runner/manifests so their CSV progress source prioritizes benettin_progress_<tag>.csv and their long stages use safe dual-checkpoint/resume arguments.
16. Add a machine-readable comparison report that compares Benettin LCN with existing Newtonian native-MEGNO LCN at matching duration/seed/model and reports relative difference, sign agreement, and classification agreement.
17. Do not start any long integrations. Run compile, help, and short smoke/unit tests only.

Scientific gates:
- Two-body controls must not be classified chaotic from a stable positive plateau.
- The 10 Myr full-system control must not show a strong persistent positive plateau inconsistent with the trusted native-MEGNO ladder.
- Newtonian 100/200 Myr Benettin should recover the same broad transition and order of magnitude as native MEGNO before GR stages are approved.
- Any failed gate must leave downstream stages blocked.
```
