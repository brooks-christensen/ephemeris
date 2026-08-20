# AGENTS.md — instructions for Codex

You run **inside the WSL environment** with the real `.venv`, REBOUND, skyfield,
and the ephemeris kernels. You are the only agent that can execute the actual
pipeline. That makes you the hands.

**Read `docs/STATE.md` first, every session.** It is the source of truth for
where the project is. Update it when you finish a unit of work.

---

## The cardinal rule

> No formula, fixture, threshold, node, process boundary, metric, or
> classification rule may be weakened or reinterpreted after observation.

This is `classification_rules.no_post_observation_change` in every manifest. If
a gate fails, the correct outcomes are: fix the implementation, or record a
FAILED campaign. Making the gate easier is never a correct outcome.

**It is violated more often by reclassification than by moving numbers.**
Changing a threshold is visible in review; introducing a classification whose
branch exempts the failing case is not. Both are the same violation. This has
already happened once in this project — see `docs/STATE.md` §"Manifest 28
disposition".

A gate no case exercises is functionally deleted. When you introduce or change a
classification, verify every branch is reachable by something actually enrolled,
and say which.

---

## Two-agent protocol

Claude works from a **fresh clone of the public repo** in an isolated container.
It cannot reach your disk, your `.venv`, or your data. That is a limitation for
execution and the entire point for verification.

**Whoever wrote a thing does not certify it.**

- You own the working tree: campaigns, calibrations, long integrations, commits.
- Claude verifies from source: reproduces claims, audits diffs it did not write,
  red-teams gates and thresholds.
- Anything Claude writes, **you review before it lands**.
- Anything you write, Claude reviews.
- **Only Brooks merges to `v2-whckl-tangent-core`.** Neither agent pushes there.

**Branches:** yours are `codex/*`. Claude's are `claude/*` or `review/*`. Never
commit to a branch prefixed for the other agent. If you must touch a file the
other has open work on, produce a patch and hand it over rather than editing.

---

## Repository layout

| Thing | Where |
|---|---|
| Manifests (frozen preregistrations) | `ephemeris_experiment_runner/manifests/NN_*.json` |
| Campaign artifacts | `docs/validation/<slug>/` |
| Implementation + qualification | `mini_ephemeris/src/mini_ephemeris/m0_step*_qualification.py` |
| Artifact generation | `..._reporting.py` |
| Tests | `mini_ephemeris/tests/` |
| Production v2 kernels | `mini_ephemeris/src/mini_ephemeris/v2/` |
| Chaos pipeline (NOT manifest-pinned) | `long_term_stability_cli.py`, `gr_benettin_cli.py`, `rebound_shadow_lyapunov_cli.py`, `chaos_estimator_diagnostics.py`, `analysis_tools.py` |

Environment: `.venv` at repo root, `PYTHONPATH=mini_ephemeris/src`,
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Commands are recorded verbatim in each
manifest's `qualification_commands` — **run those exact strings**. If a
preregistered command cannot run, that is a manifest defect to report, not an
obstacle to work around.

---

## Verification standards

**`py_compile` passing says nothing about names.** Python does not resolve names
until execution. A `NameError` will compile cleanly. Before handing over any
edit, run the `symtable` check in `scripts/check_undefined_names.py`.

**A verification tool you have not tested against a known failure is a green
light you painted yourself.** When you write a checker, first feed it a case you
know is broken and confirm it fails.

**Verify by execution against a system with a known answer.** Every real defect
found in this project surfaced that way: the Lyapunov artifact on an integrable
two-body system where λ = 0; the GR check against Mercury's 42.98″/century; the
estimator against the Chirikov standard map where λ = ln(K/2) analytically.

**Ask whether a gate could have failed.** Several checks here clear their
thresholds by 4–8 orders of magnitude while a genuinely binding condition passes
with zero margin. For each acceptance conjunct, determine from the observed data
whether it could have gone the other way.

---

## Manifest discipline

A manifest is frozen at preregistration, records `parent_commit`, and is
`manifest_must_be_committed_alone: true`.

Once a campaign has started:

- Do not edit the manifest.
- Do not edit the harness mid-campaign. If a harness bug is found, **stop**,
  commit the correction as its own commit, and re-run from clean. A pass from a
  patched-mid-flight execution is not a preregistered result, however correct
  the patch.
- Do not create the next manifest automatically after a failure.
- Do not create or move a git tag without explicit instruction.

`final_status` and `primary_finding` may only take values from the manifest's
`result_vocabulary`. BLOCKED is "evaluation would violate the frozen manifest" —
not a softer FAILED.

---

## Standing prohibitions

- Do not modify production `v2/kick.py` or COM projection semantics.
- Do not modify closed manifests or their closeout records.
- Do not modify already-qualified step files, or anything pinned in
  `historical_step3g1a_sha256` (Manifest 23) — this includes `__init__.py`.
- Do not execute integration, physical force/JVP, REBOUND, archive, trajectory,
  MEGNO, or LCN operations unless the active manifest permits them.

---

## Before proposing a commit

1. Does any gate read as weaker than before, including classification-shaped?
2. Is every branch of every classification exercised by an enrolled case?
3. Did the harness change during the campaign? If so, re-run from clean.
4. Are all deviations recorded in the artifact itself, not the commit message?
5. Does the closeout state what was *not* covered?
6. Did you run the undefined-name check and the full test suite?

Answer these in the commit description, not just in your head.
