"""Deterministic compact artifact generation for the Step 3g1a foundation."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = ROOT / "docs/validation/m0-step3g1a-v2-foundation-v1"
STEP3G0_TRACEABILITY = (
    ROOT
    / "docs/validation/m0-step3g0-verification-architecture-audit-v1"
    / "requirements_traceability.csv"
)
MANIFEST22 = ROOT / "ephemeris_experiment_runner/manifests/22_m0_step3g1a_v2_foundation_v1.json"
PREREGISTRATION_COMMIT = "d8947b3ee48250a0576ebf648b857bfa6c401200"


PROTECTED_SOURCES = {
    "mini_ephemeris/src/mini_ephemeris/gr_potential_tangent.py": "9ffe480140dc5f0d7075057f77584ac76d9af192021b035c6a984dc7e7f026ff",
    "mini_ephemeris/src/mini_ephemeris/rebound_gr_tangent_cli.py": "37293b573fca143a29a25855a1d146264d40c22fdae6a3c72821b2420da0fbc5",
    "mini_ephemeris/src/mini_ephemeris/gr_tangent_validation_matrix.py": "1ea8926cbe6519e384b95415f9813bfa2c665efc5fe0afd97c24ea6fc1fe71a9",
    "mini_ephemeris/src/mini_ephemeris/csrc/gr_potential_tangent.c": "c764740a9561fadffc93ca337ab41b51aa67b4e7140ecbf335c0b00b89ab2e23",
}

PROTECTED_MANIFESTS = {
    "ephemeris_experiment_runner/manifests/13_m0_integrator_roundoff_diagnosis_v1.json": "215bc0713aa17e2ff0986077cfadee8b40f520708d352593a6ffc04da9417240",
    "ephemeris_experiment_runner/manifests/14_m0_reversibility_roundoff_gate_v1.json": "6d74db1a0a0a8d96a00295d0a4279d91c7d1811c194546de11f328c612f97188",
    "ephemeris_experiment_runner/manifests/15_m0_integrator_roundoff_diagnosis_continuation_v1.json": "9fc06c7e0ae5811afc644df6f08ff75bfae52172eb5fc41d400c2bef6835aafe",
    "ephemeris_experiment_runner/manifests/16_m0_ias15_phase_reference_v1.json": "3895ead3f3641463d320143e2fa46abb6293d3f9c0461af4cc3f6d90fb591ec6",
    "ephemeris_experiment_runner/manifests/17_m0_step3e_whfast_0125d_convergence_v1.json": "978ab813979ea6c728e113c1f473afabb54cd553d2097dd9d26add8391f5589b",
    "ephemeris_experiment_runner/manifests/18_m0_step3e1_offline_state_diagnosis_v1.json": "088b55fa40cf0ccb7fa50f42d41f017fcdf560d7a8f7a7dfa69678717544021d",
    "ephemeris_experiment_runner/manifests/19_m0_step3f0_whfast_configuration_audit_v1.json": "28d8c390690be7c1b98cfea1b5e22615926dc149b6e7c88640d8db4e5074b521",
    "ephemeris_experiment_runner/manifests/20_m0_step3f1_two_lane_architecture_screen_v1.json": "46b19b54f278f6e174f7aa10d6fa6e2ed68e25394fc7839d1acecbcde601a65c",
    "ephemeris_experiment_runner/manifests/21_m0_step3g0_verification_architecture_audit_v1.json": "b81495b1b561ba10bddb9002912c9fdd03cb37113da29ecd479f03cb1e9614f9",
}

TRACEABILITY_COLUMNS = [
    "requirement_id",
    "category",
    "formal_requirement",
    "risk_addressed",
    "existing_evidence",
    "existing_test",
    "new_step3g0_test",
    "future_v2_test",
    "independent_oracle",
    "pass_criterion",
    "current_disposition",
]

FOUNDATION_ROWS = [
    ("V2-MODEL-001", "immutable_model", "BodyId is authoritative and CompiledLayout explicitly validates unique order and central identity.", "Body identity/index conflation", "Step 3g0 G0-005", "None", "Body-order audit", "test_body_identity_and_explicit_order; test_duplicate_missing_and_unknown_body_rejected", "Typed ID/layout construction", "All identity and swap cases pass", "PASS_STEP3G1A"),
    ("V2-MODEL-002", "immutable_model", "Model values are immutable, finite, unit-declared, provenance-bearing, and defensively copied.", "Mutable or malformed model", "Step 3g0 ADR", "Current fingerprints only", "Input-contract audit", "test_model_is_immutable_and_defensively_copies_inputs; test_invalid_ids_masses_constants_units_and_provenance_rejected", "Independent mutation of source collections", "Stored model remains unchanged and invalid input fails", "PASS_STEP3G1A"),
    ("V2-MODEL-003", "artifact_provenance", "Canonical binary64-hex serialization and SHA-256 cover every material model field independent of process state.", "Nondeterministic or incomplete fingerprint", "Manifest 22", "Historical fingerprint tests", "Canonicalization review", "test_canonical_serialization_ignores_mapping_and_collection_order; test_fingerprint_sensitive_to_every_material_field; test_fresh_process_hash_seed_and_locale_consistency", "Fresh Python processes", "Byte identity and isolated field sensitivity", "PASS_STEP3G1A"),
    ("V2-STATE-001", "state_contracts", "Inertial x-v, canonical Jacobi q-p, and canonical tangent dq-dp are distinct public types with explicit momentum semantics.", "Velocity/momentum ambiguity", "Step 3g0 ADR", "None", "Type-contract review", "test_coordinate_types_are_semantically_distinct", "Distinct immutable dataclasses", "No implicit conversion and semantic fields preserved", "PASS_STEP3G1A"),
    ("V2-STATE-002", "state_contracts", "Public state validates finite SI 3-vectors, units, body layout, and retains no writable alias.", "State corruption or aliasing", "Manifest 22", "None", "Static-state audit", "test_shape_nonfinite_and_unit_validation; test_numpy_inputs_are_detached_and_public_storage_is_immutable; test_canonical_state_tangent_layout_compatibility", "Mutable NumPy fixtures", "Invalid states fail and source mutation is invisible", "PASS_STEP3G1A"),
    ("V2-KERNEL-001", "force_and_jacobian", "Force and JVP interfaces are pure, deterministic, layout/unit checked, and semantically distinct.", "Hidden synchronization or side effects", "Step 3g0 ADR", "Protected kernel matrix", "Protocol specification", "test_force_is_deterministic_and_inputs_remain_bitwise_unchanged; test_layout_and_unit_mismatches_are_rejected_before_provider; test_wrong_output_semantics_are_rejected", "Stateless synthetic linear provider", "Repeated immutable outputs and fail-closed mismatch checks", "PASS_STEP3G1A_SYNTHETIC_ONLY"),
    ("V2-KERNEL-002", "tangent_linearity", "For fixed model/state/context JVP output is linear in direction and zero direction is exact positive zero in the fixture.", "Nonlinear tangent contamination", "Step 3g0 TAN-001", "Protected JVP tests", "Static JVP review", "test_jvp_linearity_and_exact_zero_direction", "Synthetic analytic linear map", "Exact superposition and signed-zero check", "PASS_STEP3G1A_SYNTHETIC_ONLY"),
    ("V2-TIME-001", "deterministic_targeting", "Epoch and positive interval are exact rationals and integer indices derive targets directly within a declared bound.", "Accumulation drift and restart ambiguity", "Step 3g0 TIME-001", "Historical target tests", "Endpoint schedule audit", "test_exact_roundtrip_equality_and_hashing; test_negative_and_very_large_direct_indices; test_invalid_intervals_denominators_indices_and_bounds; test_binary64_boundary_and_accumulation_drift", "fractions.Fraction", "Exact round trip and direct indexing differs from accumulated float", "PASS_STEP3G1A"),
    ("V2-OWN-001", "observer_noninterference", "Observers receive only detached immutable snapshots and no live state handle.", "Observer changes future map", "Step 3g0 G0-006", "Historical restart equality", "Ownership specification", "test_snapshot_detaches_from_mutable_source; test_observer_only_receives_immutable_snapshot", "Mutable source fixtures", "Snapshot bytes remain unchanged after source mutation", "PASS_STEP3G1A_FOUNDATION"),
    ("V2-COUNT-001", "performance_instrumentation", "Map-stage, corrector/synchronization, observer-only, and serialization/restart domains are disjoint typed values.", "Callback-accounting conflation", "Step 3g0 exact reconciliation", "Historical aggregate counters", "Callback schedule model", "test_accounting_domains_are_typed_and_disjoint", "Enum identity", "Four unique domains and untyped values rejected", "PASS_STEP3G1A_FOUNDATION"),
    ("V2-DIAG-ANGLE-001", "finite_diagnostics", "G0-001 requires atan2 primary plus chord cross-check and zero-vector rejection for future orientation observers.", "Ill-conditioned acos gate", "Step 3g0 G0-001 HIGH", "Step 3g0 conditioning suite", "test_direction_estimators_conditioning", "test_high_finding_angle_requirement_is_explicit; test_both_high_findings_have_named_tests_and_requirements", "Manifest and traceability contract", "Requirement and named tests are explicit; observer remains deferred", "PASS_FOUNDATION_CONTRACT_PRIMITIVE_DEFERRED"),
    ("V2-THRESH-SCOPE-001", "threshold_validity", "G0-002 requires exact map/trajectory/tangent/normalization/coordinate/rescaling/time/comparison applicability before threshold use.", "Misapplied implementation threshold", "Step 3g0 G0-002 HIGH", "Step 3g0 provenance matrix", "test_threshold_provenance_consistency", "test_threshold_scope_rejects_mismatched_contexts; test_both_high_findings_have_named_tests_and_requirements", "Typed compatibility context", "Every changed context field is rejected", "PASS_STEP3G1A"),
    ("V2-ISOLATION-001", "software_isolation", "The v2 foundation imports no REBOUND/REBOUNDx and exposes no dynamics, archive, diagnostic-growth, or primitive-map entrypoint.", "Scope leakage or historical coupling", "Manifest 22", "Step 3g0 no-integration guard", "Source-only audit", "test_v2_imports_no_rebound_and_exposes_no_primitive_entrypoint; test_source_has_no_hidden_dynamics_or_dependency_imports; test_public_api_docstrings_and_annotations", "Fresh-process import and AST", "No prohibited imports or executable symbols", "PASS_STEP3G1A"),
]

REVIEW_FINDINGS = [
    {
        "finding_id": "G1A-R001",
        "severity": "MEDIUM",
        "area": "canonical provenance and snapshot metadata",
        "finding": "The first pass coerced arbitrary objects with str(), which could embed identity-dependent representations.",
        "resolution": "Require nonempty trimmed strings before sorting or serialization.",
        "regression_tests": [
            "test_invalid_ids_masses_constants_units_and_provenance_rejected",
            "test_snapshot_detaches_from_mutable_source",
        ],
        "status": "RESOLVED",
    },
    {
        "finding_id": "G1A-R002",
        "severity": "MEDIUM",
        "area": "numeric input validation",
        "finding": "The first pass accepted booleans and coercible numeric strings at selected binary64 boundaries.",
        "resolution": "Require real non-boolean inputs, then finite-check normalized binary64 values.",
        "regression_tests": [
            "test_invalid_ids_masses_constants_units_and_provenance_rejected",
            "test_shape_nonfinite_and_unit_validation",
        ],
        "status": "RESOLVED",
    },
    {
        "finding_id": "G1A-R003",
        "severity": "LOW",
        "area": "synthetic JVP zero fixture",
        "finding": "Multiplication by a negative fixture coefficient emitted negative signed zero for a zero direction.",
        "resolution": "Canonicalize fixture zero results to positive exact zero and inspect float.hex().",
        "regression_tests": ["test_jvp_linearity_and_exact_zero_direction"],
        "status": "RESOLVED",
    },
    {
        "finding_id": "G1A-R004",
        "severity": "MEDIUM",
        "area": "public API annotations",
        "finding": "Custom public constructors omitted explicit None return annotations.",
        "resolution": "Annotate every custom constructor and add an AST-wide public API gate.",
        "regression_tests": ["test_public_api_docstrings_and_annotations"],
        "status": "RESOLVED",
    },
    {
        "finding_id": "G1A-R005",
        "severity": "INFORMATIONAL",
        "area": "future hot-loop allocation",
        "finding": "The immutable semantic API allocates return values and must not become the optimized map-loop ABI.",
        "resolution": "Document a separate future private caller-owned-buffer evaluate_into boundary; do not implement it in Step 3g1a.",
        "regression_tests": ["test_public_api_docstrings_and_annotations"],
        "status": "RESOLVED_BY_SCOPE",
    },
    {
        "finding_id": "G1A-R006",
        "severity": "MEDIUM",
        "area": "state unit semantics",
        "finding": "Explicit SI field names initially accepted arbitrary unit-system identifiers.",
        "resolution": "Require si_v1 for concrete public state types and add a canonical state/tangent compatibility gate.",
        "regression_tests": ["test_shape_nonfinite_and_unit_validation", "test_canonical_state_tangent_layout_compatibility"],
        "status": "RESOLVED",
    },
    {
        "finding_id": "G1A-X001",
        "severity": "HIGH",
        "area": "execution scope and provenance",
        "finding": "The final broad Step 3g0 wildcard command reevaluated protected physical force/JVP functions, contrary to the Step 3g1a static source-only boundary.",
        "resolution": "Not resolvable within this execution history; require a fresh preregistered source-only requalification and do not advance to Step 3g1b.",
        "regression_tests": [],
        "status": "UNRESOLVED_REQUIRES_FRESH_REQUALIFICATION",
    },
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _test_inventory() -> List[Dict[str, Any]]:
    inventory = []
    test_paths = [
        ROOT / "mini_ephemeris/tests/test_v2_foundation.py",
        ROOT / "mini_ephemeris/tests/test_m0_step3g1a_artifacts.py",
    ]
    for path in test_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                    inventory.append(
                        {
                            "id": f"{path.stem}.{node.name}.{method.name}",
                            "class": node.name,
                            "dynamics_executed": False,
                            "source": str(path.relative_to(ROOT)),
                            "synthetic_or_static_only": True,
                        }
                    )
    return sorted(inventory, key=lambda item: item["id"])


def _api_specification() -> str:
    return """# V2 Foundation API and Ownership Specification

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
"""


def _report(test_count: int) -> str:
    return f"""# M0 Step 3g1a V2 Foundation Report

## Result

- Final status: `STEP3G1A_V2_FOUNDATION_INCOMPLETE`
- Primary finding: `V2_FOUNDATION_NOT_READY`
- Verification envelope: not established; `FOUNDATION_ONLY_NO_DYNAMICS_EXECUTED` was not earned
- Branch: `v2-whckl-tangent-core`
- Preregistration commit: `{PREREGISTRATION_COMMIT}`

The isolated foundation implements every Manifest 22 representation and interface contract. It does not qualify dynamics or the physical model.
No physical model, integrator, WHCKL map, tangent map, or Solar-System trajectory has yet been dynamically validated in v2.

## Implemented contracts

- Stable `BodyId`, checked `CompiledLayout`, explicit SI unit declaration, immutable model/provenance, exact canonical binary64 encoding, and SHA-256 fingerprinting.
- Distinct immutable inertial `(x,v)`, canonical Jacobi `(q,p)`, tangent `(delta_q,delta_p)`, acceleration/JVP, and observer-snapshot types.
- Pure force and JVP protocols with explicit context, layout/unit checks, no counter or observer handles, and a separate future hot-loop boundary.
- Exact rational epoch/interval plus bounded integer macro-step index and named binary64 conversion boundary.
- Detached snapshot ownership, four disjoint accounting domains, and exact threshold-applicability typing.

## High findings carried forward

G0-001 is `V2-DIAG-ANGLE-001`: future orientation observers must use `atan2` plus chord and reject zero vectors. G0-002 is `V2-THRESH-SCOPE-001`: thresholds fail compatibility when any map, trajectory, tangent, normalization, coordinate, rescaling, timestamp, or comparison-class field differs. Both have named passing tests and traceability rows.

## Review

The dedicated review found seven items: one high execution-scope deviation, four medium, one low, and one informational. The six implementation findings are resolved or deferred by scope; the high deviation requires fresh source-only requalification.

## Verification

The machine inventory contains {test_count} passing Step 3g1a tests. A broad 19-test Step 3g0 command also passed and confirmed hashes, but it was not a valid source-only allowlist because selected tests reevaluated the protected physical kernels. Fresh-process probes and artifact regeneration otherwise passed, and all protected bytes remain exact.

No trajectory, timestep, IAS15 run, Simulationarchive creation, tag, MEGNO/LCN implementation, Jacobi transform, Kepler drift, kick, lazy kernel, corrector, or WHCKL map was created or executed. Static protected physical force/JVP evaluations did occur in the overbroad inherited test command; no files or dynamics state were changed by them.

## Limitations and remaining risks

- The model contract validates representation, not the scientific correctness of any physical equations.
- Force/JVP evidence is synthetic protocol evidence only; protected physical equations were neither copied nor reevaluated for v2.
- Public immutable return values allocate; a private caller-owned-buffer backend remains a future design gate.
- State/checkpoint wire schemas, canonical transforms, primitive maps, tangent maps, symplecticity, reversibility, and restart execution remain unimplemented.
- The future orientation observer must still implement and test G0-001; Step 3g1a only freezes its contract.

## Successor

Do not begin Step 3g1b. The smallest next action is a new preregistered Step 3g1a source-only requalification from this reviewed code, using an explicit test list that cannot import or invoke protected physical kernels.
"""


def _traceability_bytes() -> bytes:
    with STEP3G0_TRACEABILITY.open(newline="", encoding="utf-8") as source:
        existing = list(csv.DictReader(source))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=TRACEABILITY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(existing)
    for row in FOUNDATION_ROWS:
        writer.writerow(dict(zip(TRACEABILITY_COLUMNS, row)))
    return output.getvalue().encode("utf-8")


def generate_artifacts(destination: Path = DEFAULT_DESTINATION) -> Mapping[str, Any]:
    """Generate every compact Step 3g1a artifact deterministically."""

    destination = Path(destination)
    inventory = _test_inventory()
    test_inventory = {
        "schema_version": 1,
        "kind": "m0_step3g1a_test_inventory",
        "commands": [
            "PYTHONPATH=mini_ephemeris/src .venv/bin/python -m unittest discover -s mini_ephemeris/tests -p test_v2_foundation.py -v",
            "PYTHONPATH=mini_ephemeris/src .venv/bin/python -m unittest discover -s mini_ephemeris/tests -p test_m0_step3g1a_artifacts.py -v",
            "PYTHONPATH=mini_ephemeris/src .venv/bin/python -m unittest discover -s mini_ephemeris/tests -p test_m0_step3g0*.py -v",
        ],
        "safe_allowlist_only": False,
        "execution_deviation": "The executed Step 3g0 wildcard included protected physical force/JVP evaluations and is retained for provenance, not counted as a qualifying source-only gate.",
        "tests": inventory,
    }
    review = {
        "schema_version": 1,
        "kind": "m0_step3g1a_review_findings",
        "findings": REVIEW_FINDINGS,
        "unresolved_material_findings": 1,
    }
    files = {
        "v2_foundation_api_ownership_specification.md": _api_specification().encode("utf-8"),
        "requirements_traceability.csv": _traceability_bytes(),
        "test_inventory.json": _json_bytes(test_inventory),
        "review_findings.json": _json_bytes(review),
        "m0_step3g1a_v2_foundation_report.md": _report(len(inventory)).encode("utf-8"),
    }
    for name, payload in files.items():
        _atomic_write(destination / name, payload)

    protected = {
        path: {
            "expected_sha256": expected,
            "observed_sha256": _sha256(ROOT / path),
        }
        for path, expected in sorted(PROTECTED_SOURCES.items())
    }
    manifests = {
        path: {
            "expected_sha256": expected,
            "observed_sha256": _sha256(ROOT / path),
        }
        for path, expected in sorted(PROTECTED_MANIFESTS.items())
    }
    source_paths = sorted((ROOT / "mini_ephemeris/src/mini_ephemeris/v2").glob("*.py"))
    source_paths.extend(
        [
            ROOT / "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_reporting.py",
            ROOT / "mini_ephemeris/tests/test_v2_foundation.py",
            ROOT / "mini_ephemeris/tests/test_m0_step3g1a_artifacts.py",
            MANIFEST22,
        ]
    )
    summary = {
        "schema_version": 1,
        "kind": "m0_step3g1a_v2_foundation_summary",
        "final_status": "STEP3G1A_V2_FOUNDATION_INCOMPLETE",
        "primary_finding": "V2_FOUNDATION_NOT_READY",
        "verification_envelope": None,
        "branch": "v2-whckl-tangent-core",
        "required_start_commit": "e7b7ace3c4d348a018eeb83872c3c7f03e8ad322",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implemented_contracts": [
            "immutable_model_and_layout",
            "semantic_state_and_tangent_types",
            "pure_force_and_jvp_protocols_synthetic_only",
            "deterministic_integer_timebase",
            "detached_observer_snapshot",
            "typed_accounting_domains",
            "typed_threshold_applicability",
        ],
        "high_findings": {
            "G0-001": "V2-DIAG-ANGLE-001",
            "G0-002": "V2-THRESH-SCOPE-001",
        },
        "test_counts": {
            "step3g1a": len(inventory),
            "overbroad_step3g0_command_executed": 19,
            "qualifying_source_only_step3g0": 0,
            "total_qualifying": len(inventory),
        },
        "review": {
            "finding_count": len(REVIEW_FINDINGS),
            "severity_counts": {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 4, "LOW": 1, "INFORMATIONAL": 1},
            "unresolved_material_findings": 1,
        },
        "integrity": {
            "protected_sources": protected,
            "protected_manifests_13_21": manifests,
            "manifest21_inherited_historical_and_archive_ledger_verified_by_safe_tests": True,
        },
        "forbidden_operations": {
            "dynamics_executed": False,
            "physical_force_or_jvp_experiment": True,
            "protected_physical_force_or_jvp_reevaluated": True,
            "rebound_or_reboundx_imported_by_v2": False,
            "trajectory_or_archive_created": False,
            "ias15_executed": False,
            "megno_or_lcn_implemented": False,
            "whckl_primitive_implemented": False,
            "tag_created_or_moved": False,
        },
        "artifact_inventory_sha256": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(files.items())
        },
        "source_inventory_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "remaining_risks": [
            "The success envelope was not earned because protected physical kernels were reevaluated in an overbroad inherited test command.",
            "No v2 dynamics or physical provider has been validated.",
            "Canonical coordinate transforms and tangent maps remain unimplemented.",
            "The immutable semantic force API is not the future caller-owned-buffer hot-loop ABI.",
            "Checkpoint/restart execution and state wire schemas remain future gates.",
        ],
        "smallest_successor": "Fresh preregistered Step 3g1a source-only requalification; do not begin Step 3g1b.",
    }
    _atomic_write(destination / "m0_step3g1a_v2_foundation_summary.json", _json_bytes(summary))
    return summary


def main() -> int:
    """Regenerate Step 3g1a compact artifacts at their committed destination."""

    generate_artifacts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
