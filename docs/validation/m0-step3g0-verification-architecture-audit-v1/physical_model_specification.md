# M0 Physical Model Specification

## Scope and claim

`m0_grpot_emb_pluto_v1` is a reproducible Solar-System baseline for numerical-method development. It is not an ephemeris-production model and is not a complete relativistic Solar-System model. Within the tests listed here, its implemented equations are `VERIFIED_WITHIN_DOCUMENTED_MODEL_AND_NUMERICAL_ENVELOPE`.

## State and constants

- Epoch: `2000-01-01T00:00:00+00:00` as passed to Skyfield.
- Ephemeris: `de431_part-2.bsp`.
- Stored frame: Solar-System-barycentric BCRS/ICRF, J2000 equatorial axes.
- State units: metre, kilogram, second.
- `G = 6.67430e-11 m^3 kg^-1 s^-2`.
- `c = 299792458 m s^-1` exactly.
- No post-extraction translation, boost, or center-of-mass recentering is applied.
- Element observers rotate to the J2000 ecliptic and use heliocentric relative states.

The initial-state audit found a total represented mass of `1.991077995694271e30 kg`, center-of-mass position `(229.7982, -17.5512, -23.0837) m`, and center-of-mass velocity `(2.25437e-6, 1.09468e-5, 3.66233e-6) m/s`. These small nonzero values are retained provenance of the DE431 extraction and chosen GM set, not silently removed.

## Bodies and masses

Masses are `GM * 1e9 / G`. The authoritative identity order is:

| Index | Identity | DE431-consistent GM (km^3/s^2) | Convention |
|---:|---|---:|---|
| 0 | Sun | 132712440041.939400 | geometric Sun state, solar GM |
| 1 | Mercury barycenter | 22031.780000 | planet-system barycenter |
| 2 | Venus barycenter | 324858.592000 | planet-system barycenter |
| 3 | Earth barycenter | 403503.235502 | Earth-Moon barycenter and aggregate Earth+Moon GM |
| 4 | Mars barycenter | 42828.375214 | planet-system barycenter |
| 5 | Jupiter barycenter | 126712764.800000 | Jupiter satellite-system barycenter and aggregate GM |
| 6 | Saturn barycenter | 37940585.200000 | Saturn satellite-system barycenter and aggregate GM |
| 7 | Uranus barycenter | 5794548.600000 | Uranus satellite-system barycenter and aggregate GM |
| 8 | Neptune barycenter | 6836527.100580 | Neptune satellite-system barycenter and aggregate GM |
| 9 | Pluto barycenter | 977.000000 | Pluto-system barycenter and aggregate GM |

The model does not mix an Earth geometric-center state with an Earth-Moon aggregate mass: both state and mass refer to the Earth barycenter. The same planet-system convention applies to the other named barycenters.

## Newtonian equations

For positions `x_i`, velocities `v_i`, and masses `m_i`, all bodies are mutually active point masses:

```text
d x_i / dt = v_i
d v_i / dt = G * sum_{j != i} m_j * (x_j - x_i) / |x_j - x_i|^3 + a_i_GRpot
```

The Newtonian energy and angular momentum diagnostics are:

```text
E_N = sum_i 0.5*m_i*|v_i|^2 - sum_{i<j} G*m_i*m_j/r_ij
L   = sum_i m_i * (x_i cross v_i)
```

They are evaluated in the stored inertial frame. The pairwise force obeys total-force closure in exact arithmetic.

## Position-only GR potential

Let body 0 be the Sun, `d_i = x_i - x_0`, `r_i = |d_i|`, and coefficient scale `s=1`. For each non-Sun body:

```text
a_i_GRpot = -6*s*(G*m_0)^2/c^2 * d_i/r_i^4
a_0_GRpot = -sum_{i>0} (m_i/m_0) * a_i_GRpot
```

The solar term is the central reaction required for zero net applied force. It preserves translation, rotation, reflection, and Galilean covariance and prevents secular center-of-mass forcing by this model term.

The corresponding potential and corrected energy are:

```text
U_GRpot = -3*s*G^2*m_0^2/c^2 * sum_{i>0} m_i/r_i^2
E_corrected = E_N + U_GRpot
```

Frozen Step 3f1 state rows give maximum relative corrected-energy drift `3.7719734648363945e-13` in Lane P and `3.184104415205596e-12` in Lane T, versus Newtonian-only values `2.0076108306729636e-9` and `2.0094202603386918e-9` respectively. This is an offline diagnostic result, not a new trajectory.

## First variation

For `delta_d_i = delta_x_i - delta_x_0`:

```text
J_i(d_i) = -6*s*(G*m_0)^2/c^2 * [I/r_i^4 - 4*d_i*d_i^T/r_i^6]
delta_a_i = J_i(d_i) * delta_d_i
delta_a_0 = -sum_{i>0} (m_i/m_0) * delta_a_i
```

The JVP is linear in the variation, covariant under the same Euclidean transformations as the physical force, and does not backreact on physical particles. Decimal 70-digit, complex-step formula, centered finite-difference, protected Python, and compiled C comparisons agree within their documented binary64 envelopes.

## Limits and analytic check

- `s=0`: physical and tangent GR contributions are exactly zero.
- `m_i=0`: body `i` retains test-particle acceleration but contributes no solar reaction.
- One Sun and one planet: the equations reduce to the central `-alpha/r^3` potential correction with momentum-preserving reaction.
- Coincidence is outside the point-mass model. The protected current kernels return zero at coincidence rather than failing; v2 must reject this state explicitly.
- Nonfinite state input is outside the model. The v2 boundary must reject it before kernel execution.

The standard first-order perihelion advance is

```text
Delta_varpi = 6*pi*G*m_0 / [a*(1-e^2)*c^2] per orbit.
```

Using the frozen initial Mercury state gives `a=57909070249.50741 m`, `e=0.2056302512708917`, and `42.98066416845862 arcsec/century`. This verifies the analytic precession limit for the selected position-only potential. REBOUNDx documents that `gr_potential` gets this precession right while its mean motion is wrong by `O(GM/(a*c^2))`; instantaneous phase and osculating elements inherit that model error.

## Conservation and covariance

The model is invariant under common spatial translation, common velocity boost, proper rotation, and reflection. Newtonian pair forces and GR central reaction each close total force. Angular momentum is conserved by the central position-only interactions in exact arithmetic. The corrected Hamiltonian is the relevant energy diagnostic for this selected model.

## Excluded physics and disposition

| Omission | Baseline disposition | Required action before broader physical claim |
|---|---|---|
| Separate Moon | Required production physics | Add Earth and Moon states/masses and validate lunar coupling; not needed for the first WHCKL tangent methods gate. |
| Lunar quadrupole/tides | Required sensitivity experiment | Add only after separate-Moon baseline and timescale-specific claim are defined. |
| Solar J2 | Required production physics for high-fidelity Mercury/secular work | Include in the physical ladder with an analytic JVP. |
| Massive asteroids | Required production physics for ephemeris-grade or fine secular claims | Preregister asteroid set, masses, and ephemeris. |
| Full N-body 1PN/EIH | Required sensitivity experiment, then production physics for complete relativistic claims | Compare `gr_potential`, central `gr`, and `gr_full`. |
| Solar mass evolution | Immaterial for the proposed initial methods-paper baseline; required for multi-Gyr physical evolution | Add as a distinct time-dependent model. |
| Stellar encounters/Galactic tide | Outside initial project scope; required for relevant outer-system or multi-Gyr claims | Add only with an explicit environment model. |
| DE440/DE441 initial conditions | Required sensitivity experiment | Compare frozen model outputs as an initial-condition design study; do not rewrite DE431 history. |

## Permitted claims

The model can support a methods claim about a WHCKL discrete-map tangent implementation with a validated position-only perturbing potential, once future primitive and trajectory gates pass. It cannot by itself support ephemeris accuracy, complete 1PN accuracy, lunar dynamics, or present M0 production qualification.
