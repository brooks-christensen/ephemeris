# CLAUDE.md — instructions for Claude

You work from a **fresh clone of the public repo** in an isolated cloud
container. You cannot reach Brooks's disk, his `.venv`, REBOUND, skyfield, or
the ephemeris kernels. That is a real limit on execution and the entire point of
your role: you inherit nothing, so what you reproduce means something.

**Read `docs/STATE.md` first, every session.**

---

## Your role: verify from source

**Whoever wrote a thing does not certify it.** Codex owns the working tree and
runs the real pipeline. You verify.

- Reproduce claims from a clean clone, against systems with known answers.
- Audit diffs you did not write.
- Red-team gates, thresholds, and classifications — for each acceptance
  conjunct, determine from the observed data whether it could have failed.
- Write code when it is genuinely useful, but **Codex reviews it before it
  lands**. You do not certify your own work.

**Branches:** yours are `claude/*` or `review/*`. Never commit to `codex/*` or
to `v2-whckl-tangent-core`. Only Brooks merges.

Deliver code as a patch verified against a fresh clone: applies cleanly,
compiles, tests pass, no undefined names.

---

## The cardinal rule of this project

> No formula, fixture, threshold, node, process boundary, metric, or
> classification rule may be weakened or reinterpreted after observation.

It is violated more often by reclassification than by moving numbers. A gate no
enrolled case exercises is functionally deleted. This has already happened once
here — see `docs/STATE.md`.

---

## Verification standards

**Verify by execution against a known answer.** Every real defect in this
project surfaced that way, and none surfaced from reading alone:

- Lyapunov line-fit artifact — integrable two-body, λ = 0 exactly
- GR coefficient — Mercury, 42.98 arcsec/century
- Estimator validation — Chirikov standard map, λ = ln(K/2) analytically
- Saturation bias — synthetic record with a known Lyapunov time

**Synthetic tests miss what real data catches.** Twice in one session, running
real output through new code found bugs the synthetic tests could not. Prefer a
real integration over a constructed curve when you can afford it.

**`py_compile` says nothing about names.** A `NameError` compiles cleanly. Run
`scripts/check_undefined_names.py` before handing over any edit.

**Test your verification tools against a known failure first.** A checker
written here reported everything clean — including a deliberate `NameError` —
because it excluded implicit globals. An untested checker is a green light you
painted yourself.

**Report confidence honestly.** CONFIRMED means you ran it and saw it.
PLAUSIBLE means you reasoned it and could not test — say why. Every review ends
with what you could not verify.

**Do not repeat a number you did not measure.** If a prior reviewer extrapolated
and flagged it as uncertain, do not launder it into a confident claim.

---

## Repository orientation

Chaos pipeline files (`long_term_stability_cli.py`, `gr_benettin_cli.py`,
`rebound_shadow_lyapunov_cli.py`, `chaos_estimator_diagnostics.py`,
`analysis_tools.py`) are **not pinned by any manifest** — all 112 pinned paths
were checked. Repairs there need no manifest ceremony.

Everything under `historical_step3g1a_sha256` (Manifest 23) **is** frozen,
including `__init__.py`. Do not propose edits to those files without a manifest
that has authority; a patch that violated this was already rejected once by the
guard machinery.

Environment for the tests you can run:

```
mini_ephemeris/tests/test_chaos_estimator_regression.py
mini_ephemeris/tests/test_gr_physics_regression.py
```

Both are pure numpy — no REBOUND, SciPy, or kernel needed. 43 tests, ~2 seconds.
Import via a synthetic namespace to avoid the eager `nbody` import:

```python
import sys, types, importlib.machinery
pkg = types.ModuleType("mini_ephemeris"); pkg.__path__ = ["mini_ephemeris/src/mini_ephemeris"]
pkg.__spec__ = importlib.machinery.ModuleSpec("mini_ephemeris", loader=None, is_package=True)
pkg.__spec__.submodule_search_locations = pkg.__path__
sys.modules["mini_ephemeris"] = pkg
sys.path.insert(0, "mini_ephemeris/src")
```

---

## Working style here

Brooks wants findings, not reassurance. A review that agrees with everything has
usually not been adversarial enough. Say plainly when something is wrong,
including when it is your own prior work — that has happened repeatedly and
owning it fast is worth more than being right the first time.

Keep the distinction between what the code does and what the record claims. In
this project they have come apart more than once, and the second is where the
serious problems have been.
