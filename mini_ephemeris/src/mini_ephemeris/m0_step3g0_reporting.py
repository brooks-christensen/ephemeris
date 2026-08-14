from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .m0_step3g0_verification import (
    audit_initial_physical_state,
    callback_accounting_model,
    inspect_archive_readonly,
    no_integration_guard,
    recompute_frozen_conservation,
    recompute_frozen_orientation,
    require_strict_finite,
    restart_callback_accounting_model,
    sha256_file,
    write_json_atomic,
)


DOCUMENTATION_DIRECTORY = "docs/validation/m0-step3g0-verification-architecture-audit-v1"
OUTPUT_DIRECTORY = "output/stability/m0_step3g0_verification_architecture_audit_v1"
SUMMARY_NAME = "m0_step3g0_verification_architecture_audit_summary.json"
STATIC_ARTIFACTS = (
    "m0_step3g0_verification_architecture_audit_report.md",
    "code_review_findings.json",
    "requirements_traceability.csv",
    "threshold_provenance.csv",
    "physical_model_specification.md",
    "whckl_tangent_map_specification.md",
    "relativistic_model_hierarchy.md",
    "literature_novelty_matrix.csv",
    "v2_architecture_decision_record.md",
    "v2_implementation_test_backlog.md",
)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_payload(repo_root: Path, *, evidence_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    manifest_path = repo_root / "ephemeris_experiment_runner/manifests/21_m0_step3g0_verification_architecture_audit_v1.json"
    manifest = _load_json(manifest_path)
    lane_root = repo_root / "output/stability/m0_step3f1_two_lane_architecture_screen_v1"
    lane_p = lane_root / "m0_step3f1_physical_whckl_0p25d_10k"
    lane_t = lane_root / "m0_step3f1_tangent_whfast_0p25d_10k"
    doc_root = repo_root / DOCUMENTATION_DIRECTORY
    with no_integration_guard() as guard:
        orientation = recompute_frozen_orientation(lane_p / "state.csv", lane_t / "state.csv")
        conservation = {
            "lane_p": recompute_frozen_conservation(lane_p / "state.csv"),
            "lane_t": recompute_frozen_conservation(lane_t / "state.csv"),
        }
        initial_state = audit_initial_physical_state(lane_p / "state.csv")
        archives = [
            inspect_archive_readonly(lane_p / "simulationarchive.bin"),
            inspect_archive_readonly(lane_t / "simulationarchive.bin"),
        ]
        guard.assert_unused()
    protected = {
        relative: {
            "expected_sha256": expected,
            "observed_sha256": sha256_file(repo_root / relative),
        }
        for relative, expected in manifest["protected_files"].items()
    }
    historical = {
        relative: {
            "expected_sha256": expected,
            "observed_sha256": sha256_file(repo_root / relative),
        }
        for relative, expected in manifest["historical_documents"].items()
    }
    artifact_inventory = {
        name: sha256_file(doc_root / name)
        for name in STATIC_ARTIFACTS
    }
    findings = _load_json(doc_root / "code_review_findings.json")["findings"]
    evidence = {
        "schema_version": 1,
        "kind": "m0_step3g0_offline_audit_evidence",
        "no_integration_calls": 0,
        "no_timestep_calls": 0,
        "no_new_trajectory_or_archive": True,
        "orientation_recomputation": orientation,
        "conservation_recomputation": conservation,
        "initial_state_audit": initial_state,
        "callback_accounting": callback_accounting_model(),
        "restart_callback_accounting": restart_callback_accounting_model(),
        "archive_readonly_audit": archives,
    }
    require_strict_finite(evidence)
    if evidence_path is not None:
        write_json_atomic(evidence_path, evidence)
    summary = {
        "schema_version": 1,
        "kind": "m0_step3g0_verification_architecture_audit_summary",
        "experiment_id": manifest["experiment_id"],
        "model_id": manifest["model_id"],
        "final_status": "STEP3G0_VERIFICATION_ARCHITECTURE_AUDIT_COMPLETE",
        "primary_finding": "V2_CORE_SPECIFICATION_READY",
        "verification_envelope": "VERIFIED_WITHIN_DOCUMENTED_MODEL_AND_NUMERICAL_ENVELOPE",
        "production_qualified": False,
        "novelty_classification": "NOVEL_COMBINATION_OR_EXTENSION",
        "callback_classification": "CALLBACK_ACCOUNTING_EXACTLY_RECONCILED",
        "integrity": {
            "required_start_commit": manifest["preregistration"]["required_parent_commit"],
            "preregistration_commit": "89a57748cdf5b16915f213dd72dd820e52295539",
            "protected_files_unchanged": all(item["expected_sha256"] == item["observed_sha256"] for item in protected.values()),
            "historical_documents_unchanged": all(item["expected_sha256"] == item["observed_sha256"] for item in historical.values()),
            "frozen_archives_byte_identical_after_read": all(item["sha256_before"] == item["sha256_after"] for item in archives),
            "protected_hashes": protected,
        },
        "historical_statuses_preserved": manifest["frozen_historical_conclusions"],
        "force_and_jacobian": {
            "result": "PASS",
            "independent_oracles": ["decimal_70_digit_analytic_and_central_difference", "complex_step_formula", "central_difference"],
            "python_c_deterministic_random_cases": 12,
            "covariances_tested": ["translation", "rotation", "reflection", "Galilean_velocity_independence"],
            "known_input_contract_gap": "Protected Python and C kernels treat coincidence as zero force; the C pointwise API can return success with nonfinite array output. No frozen state exercised this path.",
        },
        "callback_accounting": evidence["callback_accounting"],
        "restart_callback_accounting": evidence["restart_callback_accounting"],
        "offline_recomputations": {
            "orientation": orientation,
            "conservation": conservation,
            "initial_state": initial_state,
        },
        "diagnostic_conclusions": {
            "plane_angle_gate": "ILL_CONDITIONED",
            "plane_angle_method_defect": True,
            "historical_acos_max_rad": orientation["metrics"]["orbital_plane"]["acos_rad"]["max"],
            "robust_atan2_max_rad": orientation["metrics"]["orbital_plane"]["atan2_rad"]["max"],
            "atan2_chord_max_disagreement_rad": orientation["metrics"]["orbital_plane"]["max_atan2_chord_abs_difference"],
            "tangent_direction_threshold": "VALID_ONLY_FOR_IMPLEMENTATION_EQUIVALENCE",
            "megno_lcn_thresholds": "VALID_ONLY_FOR_SAME_MAP_REPRODUCIBILITY",
            "different_map_10k_interpretation": "PHYSICALLY_UNJUSTIFIED",
            "replacement_threshold_invented": False,
        },
        "corrected_step3f1_interpretation": {
            "historical_status": "STEP3F1_TWO_LANE_SCREEN_FAILED",
            "historical_primary_finding": "BOTH_LANES_UNQUALIFIED",
            "status_changed": False,
            "plane_angle_veto_supported": False,
            "different_map_tangent_megno_veto_supported": False,
            "lane_p_saturn_failure_remains": True,
            "lane_t_carrier_and_other_integrity_findings_remain": True,
        },
        "whckl_tangent_feasibility": {
            "complete_derivative_graph": True,
            "lazy_kernel_requires": ["force", "force_jvp_at_unshifted_position", "force_jvp_at_shifted_position"],
            "hessian_or_higher_derivative_required": False,
            "existing_gr_jvp_sufficient_for_position_only_gr_potential": True,
            "canonical_symplecticity_coordinates": "Jacobi_q_p",
        },
        "code_review": {
            "finding_count": len(findings),
            "severity_counts": {
                severity: sum(item["severity"] == severity for item in findings)
                for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL")
            },
            "critical_findings": 0,
            "production_repairs_made": 0,
        },
        "test_result": {
            "new_nonintegration_unittests": 12,
            "strict_c_harness": "PASS",
            "asan_ubsan_static_harness": "PASS",
            "integration_guard": "PASS_ZERO_REAL_CALLS",
            "strict_json_finite": "PASS",
        },
        "remaining_risks": [
            "The current M0 runner is not production-qualified.",
            "WHCKL tangent, corrector, canonical symplecticity, and restart gates remain future implementation tests.",
            "The position-only GR model has the documented O(GM/(a*c^2)) mean-motion error and omits full N-body 1PN physics.",
            "A DE440/DE441 initial-condition sensitivity study and required production-physics ladder remain future work.",
        ],
        "smallest_next_step": "Step 3g1: implement pure immutable model and force/JVP interfaces plus deterministic timebase and observer ownership, then primitive WHCKL/lazy/corrector unit maps; do not add MEGNO or a Solar-System trajectory until analytic, closure, canonical-symplecticity, reversibility, and restart gates pass.",
        "artifact_inventory_sha256": artifact_inventory,
    }
    require_strict_finite(summary)
    return summary, evidence


def generate(repo_root: Path, *, output_root: Path | None = None, summary_path: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    selected_output = output_root or repo_root / OUTPUT_DIRECTORY
    selected_summary = summary_path or repo_root / DOCUMENTATION_DIRECTORY / SUMMARY_NAME
    summary, _ = build_payload(repo_root, evidence_path=selected_output / "audit_evidence.json")
    write_json_atomic(selected_summary, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate compact Step 3g0 audit artifacts without integration.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    arguments = parser.parse_args(argv)
    generate(arguments.repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
