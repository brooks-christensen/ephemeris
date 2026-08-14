# ADR: M0 v2 Core Architecture

Status: Proposed and implementation-ready for Step 3g1 primitive work. This ADR does not qualify the current runner and does not implement v2.

## Decision

Build a small external WHCKL tangent engine around pure model/force-JVP and operator primitives, reusing only audited REBOUND mathematical utilities where their contracts can be isolated. Keep a version-pinned REBOUND 4.6.0 compatibility adapter for historical replay. Design the engine so a human-authored upstream contribution remains possible after independent validation.

This selects Strategy C for the research core, with Strategy D as a later distribution path. It rejects making an immediate production patch to the protected runtime.

## Four layers

### 1. Immutable physical model

Proposed modules:

```text
mini_ephemeris_v2/model/ids.py
mini_ephemeris_v2/model/units.py
mini_ephemeris_v2/model/config.py
mini_ephemeris_v2/model/initial_conditions.py
mini_ephemeris_v2/model/provenance.py
```

`BodyId` is authoritative and independent of array order. `CompiledLayout` is an immutable checked mapping from IDs to dense indices. Units are explicit SI newtypes at public boundaries; internal arrays carry a `UnitSystemId`. Configurations are frozen dataclasses/structs serialized canonically and content hashed. Ephemeris release, kernel hash, epoch/time scale, GM source, frame, and all exclusions are mandatory provenance.

### 2. Pure force and derivative kernels

```text
mini_ephemeris_v2/kernels/protocol.py
mini_ephemeris_v2/kernels/newtonian.py|.c
mini_ephemeris_v2/kernels/gr_potential.py|.c
mini_ephemeris_v2/kernels/composite.py
```

Public conceptual API:

```text
ForceKernel.evaluate(context, model, x, out_a) -> None
ForceKernel.jvp(context, model, x, dx, out_da) -> None
```

Inputs are finite, shape checked, nonaliased according to declared rules, and carry a layout/unit fingerprint. Coincidence has an explicit policy error. Kernels do not synchronize, mutate integrator settings, output, infer context from mutable globals, or contain MEGNO logic. The C boundary uses length-bearing plain structs, fixed-width schema/API versions, caller-owned buffers, integer error codes, and no hidden allocation in the hot path.

### 3. Integrator and tangent-map adapters

```text
mini_ephemeris_v2/maps/state.py
mini_ephemeris_v2/maps/jacobi.py
mini_ephemeris_v2/maps/kepler.py|.c
mini_ephemeris_v2/maps/kick.py|.c
mini_ephemeris_v2/maps/lazy.py|.c
mini_ephemeris_v2/maps/corrector17.py|.c
mini_ephemeris_v2/maps/whckl.py|.c
mini_ephemeris_v2/maps/checkpoint.py
```

Each primitive owns a physical action and matching JVP action. `CanonicalState` stores Jacobi `(q,p)` and integer tick; `TangentState` stores `(dq,dp)`, accumulated log scale, and rescale ledger. The adapter receives explicit `EvaluationContext` values and counter handles. A live-map owner is single-writer. Observer copies are immutable snapshots and cannot write through to live buffers.

### 4. Offline observers and diagnostics

```text
mini_ephemeris_v2/observe/snapshot.py
mini_ephemeris_v2/observe/energy.py
mini_ephemeris_v2/observe/orbits.py
mini_ephemeris_v2/observe/orientation.py
mini_ephemeris_v2/observe/megno.py
mini_ephemeris_v2/observe/artifacts.py
```

Observers consume synchronized copies or stored rows. They never own the integrator and cannot cause live force/JVP calls. Orientation uses `atan2` primary plus chord cross-check. Energy formulas are model-versioned. MEGNO is introduced only after primitive tangent-map acceptance and consumes an explicit tangent growth stream.

## Deterministic timebase

Time is an integer `Tick`, with immutable rational `seconds_per_tick` and signed integer step. Targets, sample cadence, archive cadence, tangent rescaling, and observer events are integer divisibility contracts. Floating seconds/years are derived output only. The event scheduler rejects fractional targets rather than asking an integrator to finish with a partial step.

## Checkpoint and provenance

Checkpoint v1 contains:

- schema/API version and endianness;
- model/config/layout/unit hashes;
- source/runtime/compiler identities;
- integer tick, step, and next event IDs;
- canonical physical/tangent arrays;
- synchronization and corrector-applied state;
- rescaling and future MEGNO/LCN accumulators;
- per-context counters;
- sample uniqueness ledger;
- content checksum.

Write to a same-directory temporary file, flush/fsync, atomically replace, and fsync the directory where supported. Restart rejects unknown schema, hash mismatch, nonfinite values, duplicate sample identity, or incompatible runtime policy before constructing a live map.

Artifact metadata contains command, source commit/tag, manifest hash, model/config hash, runtime and compiler identities, state schema, start/end UTC, input/output hashes, and counter ledger. Scientific CSV/JSON sidecars are append-free atomic snapshots or collision-safe immutable shards with an index.

## Counter model

Counters are separate and additive:

```text
live_physical_steps
live_force_evaluations
live_jvp_evaluations
corrector_force_evaluations
corrector_jvp_evaluations
diagnostic_copy_evaluations
offline_analysis_evaluations
restart_reconstruction_events
```

Every call receives a context enum. The sum by context must equal total calls exactly. Context is not inferred from call stack, synchronization flag, globals, or output cadence.

## Errors and testing seams

Errors are typed: `InvalidModel`, `NonfiniteState`, `CollisionDomain`, `LayoutMismatch`, `SchemaMismatch`, `FingerprintMismatch`, `CounterMismatch`, and `ObserverMutation`. Pure primitives accept caller-owned scratch buffers so tests can inspect alias and allocation behavior. Every primitive has a deterministic reference implementation and can be replaced by a test mutant.

## Strategy comparison

| Strategy | Correctness risk | Maintainability | Restart | Performance | Testability | Publication/upstream | Historical replay |
|---|---|---|---|---|---|---|---|
| A. Patch/fork REBOUND 4.6.0 | Medium: exact source proximity but invasive state coupling | Low-medium: permanent fork burden | Strong if serialization extended correctly | Highest potential | Medium; internal coupling is large | Good methods value, weak long-term upstream path | Strongest |
| B. Current custom-integrator API | Medium-high until current API/source is audited | Medium-high if API stable | Unknown until specified | Potentially high | Medium | Good if generally reusable | Weaker; environment upgrade required later |
| C. External engine using audited utilities | Lowest initial research risk through small pure units | High within project | Explicit and controllable | High enough; C hot path retained | Highest | Strong methods paper; upstreamable after proof | Adapter preserves 4.6.0 fixtures |
| D. Immediate upstream implementation | High process and review risk before independent proof | Highest after acceptance | Must fit upstream schema | Highest potential | High once designed | Highest community value | Depends on upstream compatibility |

## Consequences

Positive: the research claim becomes testable at primitive boundaries; observer effects and callback counts are explicit; history remains untouched; future velocity-dependent relativity has a clean separate operator path.

Costs: some REBOUND logic must be represented behind stable adapters; exact bitwise equivalence requires locking operation order; a separate checkpoint schema must be maintained; upstream contribution needs independent human review and conformance with repository policy.

## Deferred decisions

- Whether audited REBOUND Kepler/transform C symbols are linked directly or transcribed into a small namespaced compatibility library.
- Whether the final upstream target is REBOUND 4.x or a current custom-integrator API.
- Exact discrete MEGNO estimator and tangent rescaling cadence.
- Production relativistic operator architecture for velocity-dependent 1PN.

These are not blockers for Step 3g1's pure model, timebase, observer ownership, force/JVP, and primitive-map tests.
