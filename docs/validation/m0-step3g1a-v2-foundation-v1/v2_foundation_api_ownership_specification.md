# V2 Foundation API and Ownership Specification

## Evidence boundary

This specification covers representation and interface contracts only. No physical model, force equation, integrator, Jacobi transform, WHCKL map, tangent map, MEGNO/LCN process, or Solar-System trajectory has been dynamically validated in v2.

## Namespace

`mini_ephemeris.v2` is isolated from historical M0. It imports neither REBOUND nor REBOUNDx, performs no import-time registration, and exposes no integration, timestep, archive, or primitive-map entrypoint.

## Immutable model

- `BodyId`: lowercase stable identity, never a display label or inferred index.
- `CompiledLayout`: unique ordered body IDs plus an explicitly present central body; `index_of` is the only dense-index compilation boundary.
- `UnitSystem`: explicit length, time, mass, velocity, momentum, and acceleration declarations.
- `PhysicalModel`: model/schema IDs, layout, positive finite masses and gravitational constant, units, sorted effects, and sorted provenance.
- Canonical model JSON uses sorted keys, compact separators, UTF-8, and exact `float.hex()` strings. SHA-256 covers every material field. This validates an auditable representation, not the physical truth of a model.

## State contracts

- `InertialCartesianState`: immutable `(x,v)` rows in explicit body order.
- `CanonicalJacobiState`: immutable canonical `(q,p)` rows; `p` means momentum, never velocity.
- `CanonicalJacobiTangentState`: immutable `(delta_q,delta_p)` in the identical canonical layout.
- `CartesianPositionTangent`, `CartesianAcceleration`, and `CartesianAccelerationJVP`: distinct force-boundary values.
- Constructors normalize numeric rows to detached tuples, require finite 3-vectors, and retain no writable NumPy or list aliases. No type performs coordinate or velocity/momentum conversion.
- These concrete fields are SI-only and require `unit_system_id=si_v1`; canonical state/tangent compatibility is checked explicitly.

## Force and JVP semantics

`ForceProvider.evaluate(model,state,context)` and `JVPProvider.jvp(model,state,direction,context)` are separate pure semantic protocols. Inputs and results carry exact layout/unit meaning. Identical inputs require deterministic outputs. Providers may not synchronize, observe, mutate inputs, update counters, inspect output history, or use mutable globals. JVP is linear in direction by contract.

The immutable semantic API is not the hot-loop ABI. A future private `evaluate_into` backend may use caller-owned validated buffers without changing these semantics. No physical provider or optimized backend exists in Step 3g1a.

## Deterministic timebase

`ExactSeconds` stores a reduced integer numerator and positive denominator. `MacroTimebase` stores exact epoch, positive interval, and a maximum absolute integer index. `at(index)` derives `epoch + index*interval` directly; no repeated floating addition occurs. `to_binary64()` is the named numerical boundary. This governs macro-step targets, observations, and restart identity, not internal floating stage times.

## Ownership and accounting

Future live map buffers are private and single-writer. `capture_observer_snapshot` copies values into an immutable `ObserverSnapshot`; observers receive only that snapshot and no live handle. Four disjoint accounting domains are typed now: map stage, corrector/synchronization, observer only, and serialization/restart. Events do not own or mutate counters.

## Step 3g0 high findings

- `V2-DIAG-ANGLE-001` carries G0-001: future orientation observers require robust `atan2` primary plus chord cross-check and zero-vector rejection. Observer implementation is deferred.
- `V2-THRESH-SCOPE-001` carries G0-002: threshold compatibility requires exact map, trajectory, tangent seed, normalization, coordinates, rescaling history, timestamps, and comparison class. Step 3g1a implements this applicability type and invents no threshold.

## Serialization boundary

Model and timebase canonical encodings are implemented. Tuple-backed public state and snapshots are serialization-safe values, but a checkpoint/state wire schema is intentionally deferred. Unknown schemas and fingerprints must be rejected before a future live owner exists.

## Exact successor gate

Only isolated inertial/Jacobi coordinate transforms and their canonical tangent maps may begin next. Kepler, kick, lazy kernel, corrector, WHCKL composition, MEGNO/LCN, trajectory, and archive work remain prohibited.
