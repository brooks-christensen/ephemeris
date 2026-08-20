# Research plan

Supersedes the manifest 23–28 chain. Written 2026-08-20.

---

## The question

**Baseline:** can this pipeline reproduce Lyapunov timescales that other people have
already measured for the solar system?

**Then:** how do successive physical refinements — 1PN GR, solar oblateness (J₂),
higher-order relativistic terms, and whatever else earns a place — change the chaotic
behaviour of the solar system?

The second half is only meaningful if the first half succeeds. A pipeline that cannot
recover a known number has no standing to report an unknown one.

---

## What is already known, and therefore what we must hit

These are the anchors. They are not aspirations; they are the pass criteria.

| System | Published Lyapunov time | Source |
|---|---|---|
| Inner planets | **≈ 5 Myr** | Laskar 1989, *Nature* 338, 237; still the quoted value in Mogavero, Hoang & Laskar, *Phys. Rev. X* 13, 021018 (2023) |
| Pluto | **≈ 20 Myr** | Applegate et al. 1986; Sussman & Wisdom 1988 |

And for the refinement study, the headline result already exists for GR:

> With general relativity, the probability that Mercury's eccentricity reaches 0.6
> within 5 Gyr is about **1–2 %**. Without it, substantially more high-eccentricity
> orbits appear, through a secular resonance between the perihelia of Mercury and
> Jupiter (g₁ − g₅) that GR detunes.
> — Laskar 2008, *Icarus* 196, 1

**GR is therefore not the first refinement to study. It is the last rung of the
validation ladder.** If the pipeline cannot recover "GR is stabilising, and it acts by
moving g₁ away from g₅," then nothing it says about J₂ or SR is worth reading.

### One design warning, before any compute is spent

The 5 Myr Lyapunov time is *robust*. Modern integrations that include GR still report
about 5 Myr. Meanwhile the stability statistics change dramatically when GR is removed.

Those two facts together say something important: **λ is not where the GR signal lives.**
The Lyapunov time is set by the secular resonance structure of the inner system and is
comparatively insensitive to which refinements are switched on; the effect shows up in
the *secular diffusion* of the fundamental frequencies over Gyr timescales.

If the study measures only λ, the most likely outcome is a null result for every
refinement — and a null result you did not predict is indistinguishable from a broken
pipeline. So the observables must be fixed in advance:

1. λ (and its convergence behaviour) — the instrument check
2. the secular frequencies g₁ and g₅, and the separation g₁ − g₅ — where GR acts
3. diffusion of those frequencies across an ensemble — where the stability effect lives

A refinement that moves (2) and (3) while leaving (1) alone is a *finding*, not a failure.
Say so now, in writing, so it cannot be reinterpreted later.

---

## The validation ladder

Each rung must pass before the next is attempted. Rungs 0–2 run in seconds and would
have caught every defect found in this codebase so far.

**Rung 0 — estimator unit tests, no dynamics.**
Already in place: 34 chaos-estimator tests, 24 MEGNO-convention tests, 9 GR physics
tests. Pure numpy, ~2 s, no REBOUND or kernel needed.

**Rung 1 — integrable two-body. λ = 0 exactly.**
The estimator must return *regular*, not a small positive number. This is the test the
old line-fit estimator failed: it reported a Lyapunov time of 0.35 × run duration,
stable to 2 % across a 20× span of durations. Pass criterion: classification is
`regular` or `ambiguous`, never `chaotic_candidate`, at every duration from 10³ to 10⁶
orbits.

**Rung 2 — a chaotic system with an analytic λ.**
Standard map at large K, where λ → ln(K/2), and/or Hénon–Heiles at E = 1/8. Tests the
estimator against a known exponent without involving the N-body integrator at all.
Pass criterion: recovered λ within 5 % of the analytic value.

**Rung 3 — Pluto. Target ≈ 20 Myr.**
Outer planets + Pluto, ≥ 200 Myr. You already have
`run_full_with_pluto_500myr_megno_queue.sh`, so this rung is within reach today.
Pass criterion: measured Lyapunov time in **10–40 Myr**, stable under dt halving.

**Rung 4 — inner solar system. Target ≈ 5 Myr.**
Full system, ≥ 200 Myr, with GR on (as the literature value assumes).
Pass criterion: measured Lyapunov time in **3–10 Myr**, stable under dt halving.

**Rung 5 — the GR sign test.**
Same configuration, GR off. Confirm g₁ − g₅ moves toward commensurability and secular
diffusion increases. Direction is known; magnitude is the measurement.
Pass criterion: the sign is right. If it is not, stop — the refinement study cannot begin.

The factor-of-two acceptance windows on rungs 3 and 4 are deliberate. Published Lyapunov
times for these systems vary by more than that between authors and methods. A tighter
window would be a claim about precision nobody has earned, and it would tempt tuning.
**These windows are fixed now, before the runs.**

---

## The convergence protocol

No λ is reportable unless all of the following hold. This is code that can return FAIL,
not a paragraph in a document.

1. **Timestep.** λ changes by less than 10 % when dt is halved.
2. **Two estimators agree.** Benettin and MEGNO within 20 %, using the now-measured
   factor of 2.0.
3. **The halving ratio is chaotic.** λ_running(T)/λ_running(T/2) ∈ [0.85, 1.15].
   A value near 0.5 means the estimate is still decaying as ln(t)/t — regular motion,
   not a small exponent.
4. **Energy is conserved.** Relative drift below 10⁻⁷, measured without aliasing to the
   orbital period.
5. **Tangent-vector independence.** At least 5 random initial tangent vectors; report
   the spread, not the mean alone.
6. **Saturation excluded.** The fit window ends before the tangent vector saturates.

A run failing any of these is recorded as a failure. Which requires, first, that the
reporting layer be able to emit a failure at all — see below.

---

## What is being retired, and why

**The manifest 23–28 chain is superseded, not re-litigated.** One archival note, then we
stop referring to it.

The apparatus froze the *implementation* rather than the *analysis plan*, and that
inverted its purpose. Concretely: Manifest 23 pins the SHA-256 of `__init__.py`, which
eagerly imports `nbody`, which makes the Manifest 28 reporting module unrunnable — so
Manifest 28 requires a command its own inherited freeze forbids. When the fix was
written, it could not be applied. **The seal meant to prevent post-hoc tuning was
protecting a bug from being fixed**, which creates pressure not to look too hard. That
is the opposite of the intent.

Also retired: `m0_step3g1d_reporting.py`'s certification path, which transcribes the
manifest's node lists and stamps a literal `"result": "PASS"` without ever invoking
pytest. The 124/124 clean campaign is not evidence. The artifact would be byte-identical
if every test failed.

**Replacement posture:** preregister the analysis plan, not the source tree. One short
manifest fixes systems, durations, observables, acceptance windows, and what gets
reported regardless of outcome. Hash *that*. Code stays freely fixable; the requirement
is that every reported result be regenerated from a single commit, after the fixes land.
(You didn't pick between the options on this — I've taken the recommended one. Say so if
you want it different.)

---

## Sequence

**Now — make failure expressible.** No module can currently emit a failure status. Until
it can, every gate above is decorative. This is the single blocking item.

**Then — rungs 0–2.** Seconds of compute. If the estimator cannot recover ln(K/2) from a
standard map, nothing downstream matters.

**Then — rung 3 (Pluto).** The first real integration, against a target you can check.

**Then — rung 4 (inner system, GR on).** Expect real compute: ≥ 200 Myr at a converged
timestep, run twice for the dt check, times 5 tangent vectors.

**Then — rung 5 (GR off), and only then the refinement study:** J₂ (adopted value matters —
published J₂ ranges over 1.5–2.3 × 10⁻⁷ and the corresponding Mercury precession
contribution is small against GR's 42.98″/century, so characterise the *magnitude* of
each refinement before measuring its effect on chaos), then higher-order terms.

---

## Working agreement

- **Claude Code in WSL** — implementation. Direct file access, runs the real pipeline.
- **This cloud session** — review only. Reads the repo from GitHub and *cannot* push,
  so it cannot quietly fix what it finds. Whoever wrote a thing does not certify it.
- **Codex** — independent verification of anything either of the above produces.
- **Brooks** — merges, and decides anything involving deletion or scope.
