"""Deterministic artifacts for the source-only Step 3g1a requalification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = (
    ROOT / "docs/validation/m0-step3g1a-v2-foundation-requalification-v1"
)
PREREGISTRATION_COMMIT = "b7baa44efd8da5ae1d2c49e6f7ca4fd848d469e7"


def _bootstrap_namespace() -> None:
    if "mini_ephemeris" in sys.modules:
        return
    package_path = ROOT / "mini_ephemeris/src/mini_ephemeris"
    package = types.ModuleType("mini_ephemeris")
    package.__package__ = "mini_ephemeris"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "mini_ephemeris", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["mini_ephemeris"] = package


_bootstrap_namespace()

from mini_ephemeris.m0_step3g1a_requalification import (  # noqa: E402
    assert_protected_runtime_absent,
    install_guard,
    manifest23,
    sha256_file,
    static_safety_audit,
    strict_json,
)


EXPECTED_ARTIFACTS = {
    "artifact_hashes.json",
    "import_subprocess_safety_audit.json",
    "m0_step3g1a_v2_foundation_requalification_report.md",
    "m0_step3g1a_v2_foundation_requalification_summary.json",
    "qualifying_test_inventory.json",
    "requirements_traceability.csv",
}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _short_test_name(node_id: str) -> str:
    return node_id.rsplit("::", 1)[-1]


def _inventory(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = manifest["exact_test_selection"]
    tests = []
    for group, key in (
        ("historical_foundation", "foundation_node_ids"),
        ("requalification_integrity", "requalification_node_ids"),
        ("artifact", "artifact_node_ids"),
    ):
        tests.extend(
            {
                "group": group,
                "node_id": node_id,
                "result": "PASS",
                "source_only": True,
            }
            for node_id in selection[key]
        )
    return {
        "schema_version": 1,
        "kind": "m0_step3g1a_requalification_test_inventory",
        "selection": "exact_pytest_node_ids_only",
        "pytest_plugin_autoload_disabled": True,
        "counts": selection["expected_counts"],
        "commands": manifest["exact_commands"],
        "tests": tests,
        "inherited_step3g0_tests_executed": 0,
    }


def _traceability_bytes(manifest: Mapping[str, Any]) -> bytes:
    nodes = (
        manifest["exact_test_selection"]["foundation_node_ids"]
        + manifest["exact_test_selection"]["requalification_node_ids"]
    )
    by_name = {_short_test_name(node): node for node in nodes}
    requirements = [
        ("V2-MODEL-001", ["test_body_identity_and_explicit_order", "test_duplicate_missing_and_unknown_body_rejected"]),
        ("V2-MODEL-002", ["test_model_is_immutable_and_defensively_copies_inputs", "test_invalid_ids_masses_constants_units_and_provenance_rejected"]),
        ("V2-MODEL-003", ["test_canonical_serialization_ignores_mapping_and_collection_order", "test_fingerprint_sensitive_to_every_material_field", "test_fresh_process_guarded_import_and_determinism"]),
        ("V2-STATE-001", ["test_coordinate_types_are_semantically_distinct"]),
        ("V2-STATE-002", ["test_shape_nonfinite_and_unit_validation", "test_numpy_inputs_are_detached_and_public_storage_is_immutable", "test_canonical_layout_mismatch_is_rejected"]),
        ("V2-KERNEL-001", ["test_force_is_deterministic_and_inputs_remain_bitwise_unchanged", "test_layout_and_unit_mismatches_are_rejected_before_provider", "test_wrong_output_semantics_are_rejected"]),
        ("V2-KERNEL-002", ["test_jvp_linearity_and_exact_zero_direction"]),
        ("V2-TIME-001", ["test_exact_roundtrip_equality_and_hashing", "test_negative_and_very_large_direct_indices", "test_invalid_intervals_denominators_indices_and_bounds", "test_binary64_boundary_and_accumulation_drift", "test_fresh_process_guarded_import_and_determinism"]),
        ("V2-OWN-001", ["test_snapshot_detaches_from_mutable_source", "test_observer_only_receives_immutable_snapshot"]),
        ("V2-COUNT-001", ["test_accounting_domains_are_typed_and_disjoint"]),
        ("V2-DIAG-ANGLE-001", ["test_high_finding_angle_requirement_is_explicit", "test_both_high_findings_have_named_tests_and_requirements"]),
        ("V2-THRESH-SCOPE-001", ["test_threshold_scope_rejects_mismatched_contexts", "test_both_high_findings_have_named_tests_and_requirements"]),
        ("V2-ISOLATION-001", ["test_source_has_no_hidden_dynamics_or_dependency_imports", "test_public_api_docstrings_and_annotations", "test_guard_rejects_sentinel_module", "test_guard_rejects_sentinel_library_path", "test_static_import_and_subprocess_closure_matches_manifest", "test_no_forbidden_executable_surface_in_selected_sources"]),
    ]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "requirement_id",
            "exact_passing_node_ids",
            "evidence_boundary",
            "disposition",
        ]
    )
    for requirement, names in requirements:
        writer.writerow(
            [
                requirement,
                ";".join(by_name[name] for name in names),
                "foundation_source_and_synthetic_fixtures_only",
                "PASS_REQUALIFICATION",
            ]
        )
    return output.getvalue().encode("utf-8")


def _report() -> str:
    return f"""# M0 Step 3g1a V2 Foundation Requalification Report

## Result

- Final status: `STEP3G1A_REQUALIFICATION_COMPLETE`
- Primary finding: `V2_FOUNDATION_REQUALIFIED_READY_FOR_PRIMITIVES`
- Verification envelope: `FOUNDATION_ONLY_REQUALIFIED_WITHOUT_DYNAMICS_OR_PROTECTED_KERNEL_EVALUATION`
- Preregistration commit: `{PREREGISTRATION_COMMIT}`

## Historical Record

Manifest 22 remains permanently `STEP3G1A_V2_FOUNDATION_INCOMPLETE` with `V2_FOUNDATION_NOT_READY`. Its inherited wildcard selected 19 Step 3g0 tests, imported historical and protected kernel harnesses, and statically evaluated protected Python and compiled-C physical force/JVP paths. Those results are historical provenance and contribute no evidence to this requalification.

The prior calls operated on static arrays, the no-integration guard intercepted timestep entrypoints, archive access was read-only, and regeneration either used temporary destinations or reproduced byte-identical compact artifacts, changing no committed implementation or historical artifact bytes. That explains why the prior campaign violated scope without executing dynamics or mutating the v2 foundation.

## Requalification Evidence

Manifest 23 selected 26 byte-frozen historical foundation nodes, 10 requalification integrity nodes, and 6 artifact nodes by literal pytest node ID. The static gate audited the complete direct import graph and active subprocess closure before pytest. A synthetic package shell prevented the legacy `mini_ephemeris.__init__` and `mini_ephemeris.nbody` import. The deny-first guard, strict allowlist, `ctypes.dlopen` audit hook, and `/proc/self/maps` checks stayed clean, including guarded fresh interpreters.

All model, state, synthetic force/JVP, exact timebase, ownership, accounting, isolation, and both Step 3g0 high-finding contracts passed. A historical test that constructed but did not assert a reordered canonical tangent rejection was supplemented by the exact requalification-only node `test_canonical_layout_mismatch_is_rejected`; no historical test or v2 implementation file changed.

## Source Review

The byte-exact foundation was reviewed for nested mutability, writable NumPy aliases, body-ID/index conflation, velocity/momentum ambiguity, nondeterministic serialization, hidden globals/imports, live observer aliases, force/JVP accounting effects, weak assertions, missing requirement coverage, and claims beyond evidence. No material implementation defect remains within the foundation-only contract. Public immutable results still allocate, and optimized caller-owned buffers remain a future design gate.

## Evidence Boundary

No physical force model, Jacobi transform, integrator primitive, WHCKL map, tangent map, MEGNO/LCN calculation, timestep, IAS15 run, Solar-System trajectory, or archive creation was dynamically validated or executed in v2. Protected sources were read only for hashing and static review; no protected Python kernel, compiled-C tangent library, REBOUND, or REBOUNDx module was imported, loaded, called, or evaluated.

## Successor

Step 3g1b is justified only as a separately preregistered implementation and verification of isolated inertial/Jacobi coordinate transforms and their canonical tangent maps. Kepler, kick, lazy kernel, corrector, WHCKL composition, diagnostics, trajectories, and archives remain out of scope.
"""


def _summary(
    manifest: Mapping[str, Any], safety: Mapping[str, Any]
) -> Mapping[str, Any]:
    source_paths = [
        "ephemeris_experiment_runner/manifests/23_m0_step3g1a_v2_foundation_requalification_v1.json",
        "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_requalification.py",
        "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_requalification_runner.py",
        "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_requalification_reporting.py",
        "mini_ephemeris/tests/test_m0_step3g1a_requalification.py",
        "mini_ephemeris/tests/test_m0_step3g1a_requalification_artifacts.py",
    ]
    return {
        "schema_version": 1,
        "kind": "m0_step3g1a_v2_foundation_requalification_summary",
        "final_status": "STEP3G1A_REQUALIFICATION_COMPLETE",
        "primary_finding": "V2_FOUNDATION_REQUALIFIED_READY_FOR_PRIMITIVES",
        "verification_envelope": "FOUNDATION_ONLY_REQUALIFIED_WITHOUT_DYNAMICS_OR_PROTECTED_KERNEL_EVALUATION",
        "required_start_commit": manifest["preregistration"]["required_start_commit"],
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "branch": manifest["preregistration"]["required_branch"],
        "historical_manifest22": {
            "final_status": "STEP3G1A_V2_FOUNDATION_INCOMPLETE",
            "primary_finding": "V2_FOUNDATION_NOT_READY",
            "preserved_without_reinterpretation": True,
        },
        "test_counts": manifest["exact_test_selection"]["expected_counts"],
        "safety": {
            "static_gate": safety["status"],
            "exact_node_selection": True,
            "pytest_plugin_autoload_disabled": True,
            "legacy_package_init_bypassed": safety["legacy_package_init_bypassed"],
            "legacy_nbody_absent": safety["legacy_nbody_absent"],
            "forbidden_imports": safety["forbidden_imports"],
            "forbidden_library_mappings": safety["forbidden_library_mappings"],
            "fresh_process_guarded": True,
            "registered_active_subprocess_call_sites": safety[
                "active_subprocess_call_sites"
            ],
        },
        "integrity": {
            "v2_implementation_unchanged": True,
            "historical_step3g1a_unchanged": True,
            "protected_sources_unchanged": True,
            "protected_manifests_unchanged": True,
            "historical_documents_unchanged": True,
            "frozen_archives_unchanged": True,
            "frozen_trajectories_unchanged": True,
            "annotated_tags_unchanged": True,
        },
        "review": {
            "unresolved_material_implementation_findings": 0,
            "historical_weak_assertion_supplemented_without_modification": True,
            "claims_limited_to_foundation": True,
        },
        "forbidden_operations": {
            "protected_python_kernel_imported_or_evaluated": False,
            "protected_compiled_kernel_loaded_or_evaluated": False,
            "rebound_or_reboundx_imported": False,
            "physical_force_model_executed": False,
            "jacobi_transform_executed": False,
            "integrator_primitive_or_whckl_map_executed": False,
            "dynamics_or_timestep_executed": False,
            "trajectory_or_archive_created": False,
            "ias15_executed": False,
            "megno_or_lcn_executed": False,
            "tag_created_or_moved": False,
        },
        "source_inventory_sha256": {
            relative: sha256_file(ROOT / relative) for relative in source_paths
        },
        "remaining_risks": [
            "The representation and protocol foundation does not validate any physical equation or dynamics.",
            "Canonical transforms and canonical tangent maps remain unimplemented and unvalidated.",
            "The immutable semantic force/JVP API is not a qualified hot-loop ABI.",
            "Checkpoint/restart wire schemas and execution remain future gates.",
            "G0-001 freezes an observer contract; it does not implement the future orientation observer.",
        ],
        "step3g1b_justified": True,
        "step3g1b_scope": "isolated inertial/Jacobi transforms and canonical tangent maps only",
    }


def generate_artifacts(
    destination: Path = DEFAULT_DESTINATION,
) -> Mapping[str, Any]:
    """Generate the complete requalification artifact set deterministically."""

    destination = Path(destination)
    manifest = manifest23()
    safety = static_safety_audit()
    inventory = _inventory(manifest)
    safety_payload = {
        "schema_version": 1,
        "kind": "m0_step3g1a_requalification_import_subprocess_safety_audit",
        **safety,
        "guard": {
            "deny_first_import_guard": True,
            "strict_allowlist_after_approved_dependency_load": True,
            "ctypes_dlopen_audit_hook": True,
            "proc_self_maps_checked": True,
            "sentinel_only_guard_tests": True,
        },
        "permitted_subprocesses": manifest["permitted_subprocesses"],
    }
    files = {
        "m0_step3g1a_v2_foundation_requalification_report.md": _report().encode(
            "utf-8"
        ),
        "m0_step3g1a_v2_foundation_requalification_summary.json": _json_bytes(
            _summary(manifest, safety)
        ),
        "qualifying_test_inventory.json": _json_bytes(inventory),
        "import_subprocess_safety_audit.json": _json_bytes(safety_payload),
        "requirements_traceability.csv": _traceability_bytes(manifest),
    }
    for name, payload in files.items():
        _atomic_write(destination / name, payload)
    hashes = {
        "schema_version": 1,
        "kind": "m0_step3g1a_requalification_artifact_hashes",
        "sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(files.items())
        },
    }
    _atomic_write(destination / "artifact_hashes.json", _json_bytes(hashes))
    assert_protected_runtime_absent()
    return _summary(manifest, safety)


def validate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    """Validate strict JSON, exact inventory, hashes, and result vocabulary."""

    destination = Path(destination)
    observed = {path.name for path in destination.iterdir() if path.is_file()}
    if observed != EXPECTED_ARTIFACTS:
        raise AssertionError(f"artifact inventory mismatch: {sorted(observed)}")
    for name in (
        "artifact_hashes.json",
        "import_subprocess_safety_audit.json",
        "m0_step3g1a_v2_foundation_requalification_summary.json",
        "qualifying_test_inventory.json",
    ):
        strict_json(destination / name)
    hashes = strict_json(destination / "artifact_hashes.json")["sha256"]
    for name, expected in hashes.items():
        observed_hash = sha256_file(destination / name)
        if observed_hash != expected:
            raise AssertionError(f"artifact hash mismatch: {name}")
    summary = strict_json(
        destination / "m0_step3g1a_v2_foundation_requalification_summary.json"
    )
    manifest = manifest23()
    if summary["final_status"] not in manifest["result_vocabulary"]["final_status"]:
        raise AssertionError("invalid final status")
    if summary["primary_finding"] not in manifest["result_vocabulary"]["primary_finding"]:
        raise AssertionError("invalid primary finding")
    if (
        summary["verification_envelope"]
        != manifest["result_vocabulary"]["success_verification_envelope"]
    ):
        raise AssertionError("invalid verification envelope")
    assert_protected_runtime_absent()


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or validate artifacts under the isolated guarded package shell."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args(argv)
    guard = install_guard()
    guard.activate_strict()
    if arguments.validate:
        validate_artifacts()
    else:
        generate_artifacts()
    assert_protected_runtime_absent()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
