# Relativistic Model Hierarchy

## Principle

Each rung is a distinct physical model with a distinct claim. A term isolated from a first-post-Newtonian Hamiltonian is a diagnostic ablation, not an independent competing theory. Coordinate/canonical-gauge-dependent pieces must never be labeled a standalone "special-relativity force."

| Level | Model | Mean motion | Perihelion | Conservation and velocity dependence | WHCKL position-kick compatibility | Tangent requirement | Cost and claim |
|---:|---|---|---|---|---|---|---|
| 1 | Newtonian point masses | Newtonian exact within numerical map | No relativistic advance | Hamiltonian; position-only force | Direct | Newtonian force JVP | Lowest; numerical-method control |
| 2 | Newtonian + `gr_potential` | Wrong by `O(GM/(a*c^2))` | Correct leading orbit-averaged central advance | Conservative position-only potential with central reaction | Direct and symplectic for a position split | Position-force JVP only | Low; initial WHCKL tangent methods target and selected secular baseline |
| 3 | Central-body 1PN equivalent to REBOUNDx `gr` | Correct through retained central 1PN approximation | Correct through retained central 1PN approximation | Hamiltonian approximation but acceleration is velocity dependent; ignores central mass-ratio-order terms | Not a plain position kick; needs an appropriate operator/splitting | Derivatives with respect to position and velocity plus operator derivative | Moderate; single-star Solar-System 1PN validation |
| 4 | Full N-body first-PN/EIH equivalent to `gr_full` | Full retained 1PN model | Full retained 1PN model | Velocity dependent, implicit/iterative implementation, all bodies source terms | Not a position-only WHCKL kick | Full state JVP, including iterative-solve derivative or verified equivalent | High, scaling worse with N; required for complete N-body 1PN claims |
| 5 | Solar J2 | Includes oblateness perturbation | Adds oblateness-driven precession | Conservative position-dependent potential for fixed spin axis | Direct when formulated as a position potential | Analytic position JVP, plus parameter derivatives if fitted | Low; required production physics for high-fidelity Mercury/secular work |
| 6 | Solar spin/Lense-Thirring | Small frame-dragging correction | Adds spin-dependent nodal/apsidal rates | Generally velocity and spin dependent | Requires a separate operator; not a plain position kick | Position, velocity, and spin derivatives | Moderate; later sensitivity study |
| 7 | Nonphysical term ablations | Not a physical standalone prediction | Not a physical standalone prediction | Individual kinetic, gravitational, or mixed 1PN Hamiltonian pieces can depend on coordinate/canonical gauge | Only as an explicitly labeled diagnostic | Derivative of the exact retained ablation operator | Research diagnostic only |

## Authoritative basis

- REBOUNDx 4.6.1 source: `gr_potential.c`, `gr.c`, and `gr_full.c`.
- REBOUNDx effects documentation: `gr_potential` gets apsidal precession right but mean motion wrong at `O(GM/(a*c^2))`; `gr` is the dominant-central-mass 1PN approximation; `gr_full` incorporates first-PN effects from all bodies.
- Tamayo, Rein, Shi, and Hernandez, *REBOUNDx*, MNRAS 491 (2020), DOI `10.1093/mnras/stz2870`, especially Section 5 and Appendix B.

## Hamiltonian decomposition caveat

At central-body 1PN order one may write `H = H_N + H_PN`, and decompose `H_PN` into kinetic, gravitational, and mixed position-momentum terms for operator construction. Only the full retained sum is the physical approximation. Individual pieces can be coordinate or canonical-gauge dependent. Running a piece alone is useful for sign, splitting-error, or sensitivity diagnosis, but it must be labeled `NONPHYSICAL_TERM_ABLATION`.

## Recommended use

### Initial WHCKL tangent-map methods paper

Use Level 1 and Level 2. They isolate the discrete tangent-map contribution, keep every perturbing kick position-only, and exercise the validated arbitrary-force JVP interface. Claims must state the mean-motion limitation and omitted production physics.

### Solar-System secular validation

Use a ladder, not a single replacement: Level 1, Level 2, Level 3, Level 4, then Level 5. Compare secular frequencies, Mercury precession, phase, and conservation within each model's proper Hamiltonian. Level 3 is the minimum central-body 1PN phase-sensitive reference; Level 4 resolves N-body 1PN sensitivity.

### Later physical-sensitivity studies

Add Level 5 as required production physics, then Level 6 where the target observable warrants it. Add the separate Moon and massive asteroids in the nonrelativistic physical-model ladder before claiming ephemeris-grade or fine secular fidelity.

## Derivative architecture implication

The v2 initial `ForceJvp` interface is sufficient for Levels 1, 2, and position-form Level 5. Levels 3, 4, and 6 require a broader operator interface with both position and momentum/velocity derivatives and a splitting proven appropriate for conservative velocity-dependent forces. They must not be forced through the WHCKL position-kick API.
