# Final physics and experiment plan

## What the Newtonian runs established

The completed 500 Myr runs establish robust finite-time chaos for the current point-mass Newtonian model. They do not establish the asymptotic exponent of the physical Solar System. The current finite-time LCN rose as the tangent direction aligned with the unstable manifold and the orbit spent longer in the chaotic regime. A finite-time LCN should eventually approach a statistically stable plateau; it should not grow linearly forever. MEGNO itself is expected to grow approximately linearly for chaotic motion.

The often quoted 4–5 Myr value is a Lyapunov time, `tau = 1/lambda`, not the elapsed duration at which a computation must saturate. A 5 Myr Lyapunov time corresponds to `lambda ~= 2e-7 /yr`; the present 500 Myr runs near `8.4e-8 /yr` correspond to roughly 12 Myr.

## Historical model hierarchy

1. Early secular studies used equations averaged over fast mean longitudes. They were computationally efficient and exposed the inner-system resonance structure, but they were not direct integrations near close encounters.
2. Sussman and Wisdom directly integrated the planetary system for nearly 100 Myr and reported an approximately 4 Myr divergence time.
3. Ito and Tanikawa used a simpler Newtonian Sun-plus-nine-planets model, generally replacing Earth and Moon by the Earth-Moon barycenter, and explicitly neglected relativity, asteroids, and other small effects.
4. Varadi, Runnegar, and Ghil refined direct integrations by adding general relativity and the finite lunar orbit; one model resolved Earth and Moon as separate bodies. They reported roughly a 4 Myr inner-system Lyapunov time.
5. Laskar and Gastineau's 5 Gyr ensemble included contributions from the Moon and general relativity and focused on long-term instability probabilities.

## Physics priority on the local machine

### Priority 1: 1PN solar relativity

This is the highest-value missing effect because Mercury's relativistic perihelion precession shifts the secular resonance structure. A central-body approximation is physically appropriate at leading order for planets around the Sun.

- Production candidate: REBOUNDx `gr_potential` + WHFast.
- Advantage: fast, position-dependent, symplectic with WHFast, correct perihelion precession.
- Limitation: approximate mean motion.
- Validation oracle: REBOUNDx `gr` + IAS15 over short intervals.

A 100–200 Myr `gr_potential` Benettin run should be reasonable locally after validation. A 500 Myr run is unnecessary unless the finite-time exponent has not stabilized by 200 Myr.

### Priority 2: lunar forcing

The Moon matters secularly. The most efficient production treatment is an averaged finite-lunar-orbit correction, validated against an explicit Earth-Moon model over 1–10 Myr.

- Explicit Earth and Moon are feasible for short tests.
- CPU time, not memory, is the limiting resource.
- A direct Moon model introduces a shorter dynamical timescale and more demanding coordinate/integrator choices, making 100–500 Myr production substantially more expensive.
- If an averaged lunar model reproduces the explicit model's secular frequencies and short Benettin behavior, use the averaged model for long runs.

### Priority 3: cheap sensitivity physics

After GR and lunar forcing:

- solar J2;
- Ceres, Vesta, and Pallas;
- updated mass/initial-condition sensitivity.

These are inexpensive in particle count and useful as sensitivity tests, but they are lower priority than GR and the Moon for the inner-system chaos scale.

### Not worth prioritizing

- Another identical 500 Myr Newtonian run;
- full pairwise 1PN production over hundreds of Myr;
- tides, solar mass loss, or Galactic perturbations for the present Lyapunov-timescale question;
- a 5 Gyr ensemble on the local workstation.

## Gated final sequence

### Phase A — validate the GR-compatible chaos estimator

1. Two-body Newtonian controls, seeds 12345 and 67890.
2. Full-with-Pluto Newtonian 10 Myr false-positive control.
3. Full-with-Pluto Newtonian 100 Myr calibration.
4. Full-with-Pluto Newtonian 200 Myr, seed 12345.
5. Repeat 200 Myr with seed 67890 only after review.

Pass condition: broad agreement with native Newtonian MEGNO in sign, onset, and order of magnitude. Exact equality is not required.

### Phase B — validate GR physics

1. Analytic Mercury perihelion-precession test.
2. Short two-body comparison: WHFast + `gr_potential` versus IAS15 + `gr`.
3. Full-system 1 Myr GR smoke.
4. Full-system 10 Myr GR run at 1 day.
5. Timestep comparison at 0.5 day if the 1-day diagnostics or invariant drift are questionable.

### Phase C — first scientific GR result

1. Full-with-Pluto GR 100 Myr, seed 12345.
2. Review LCN convergence by windows, invariant drift, and secular spectra.
3. Repeat seed 67890 if positive or ambiguous.
4. Run a 0.5-day convergence case only if the 1-day result changes materially from Newtonian or shows numerical warning signs.
5. Extend to 200 Myr only if 100 Myr has not reached a stable finite-time plateau or the seeds disagree materially.

### Phase D — lunar refinement

1. Explicit Earth-Moon versus averaged lunar correction over 1 Myr.
2. Extend comparison to 5–10 Myr if secular-frequency agreement is unclear.
3. Run GR + averaged Moon for 100 Myr, seed 12345.
4. Repeat seed 67890 if needed.
5. Make this the final 200 Myr best-physics experiment if the earlier gates pass.

### Phase E — sensitivity panel

At 50–100 Myr, one seed each:

- best physics baseline;
- + solar J2;
- + major asteroids;
- + both.

This panel answers whether small omitted physics changes the inferred chaos scale more than seed/timestep uncertainty.

## Automation safety policy

- Every stage has a separate output directory.
- Stages run serially.
- A nonzero process exit blocks downstream stages.
- Missing, corrupted, incomplete, duplicated, or nonmonotonic output blocks downstream stages.
- Expensive or physics-changing stages require explicit approval.
- Progress is estimated from the latest archive or incremental Benettin CSV.
- ETA uses the median recent simulated-time rate and is intentionally labeled as an estimate.
- Stalls produce warnings but are not killed automatically.
- Safe long-run resume should be enabled only after both trajectories and the Benettin accumulator are checkpointed together.
