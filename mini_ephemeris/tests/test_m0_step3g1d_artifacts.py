"""Artifact gates for Step 3g1d requalification."""

from __future__ import annotations

import csv
import unittest

from mini_ephemeris.m0_step3g1d_qualification import (
    manifest28,
    manifest28_preregistration_commit,
    sha256_file,
    strict_json,
)
from mini_ephemeris.m0_step3g1d_reporting import (
    DEFAULT_DESTINATION,
    EXPECTED_ARTIFACTS,
    compare_fresh_regeneration,
    validate_artifacts,
)


SUMMARY_NAME = "m0_step3g1d_interaction_kick_requalification_summary.json"
REPORT_NAME = "m0_step3g1d_interaction_kick_requalification_report.md"


class Step3g1dArtifactTests(unittest.TestCase):
    def test_required_artifacts_and_strict_json(self) -> None:
        observed = {
            path.name
            for path in DEFAULT_DESTINATION.iterdir()
            if path.is_file()
        }
        self.assertEqual(observed, EXPECTED_ARTIFACTS)
        validate_artifacts()

    def test_summary_report_manifest_and_specs_agree(self) -> None:
        manifest = manifest28()
        baseline = manifest["baseline"]
        summary = strict_json(DEFAULT_DESTINATION / SUMMARY_NAME)
        report = (DEFAULT_DESTINATION / REPORT_NAME).read_text(
            encoding="utf-8"
        )
        provider_spec = (
            DEFAULT_DESTINATION / "canonical_kick_provider_specification.md"
        ).read_text(encoding="utf-8")
        tangent = (
            DEFAULT_DESTINATION / "tangent_map_derivation.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            summary["verification_envelope"],
            manifest["result_vocabulary"][
                "successful_verification_envelope"
            ],
        )
        self.assertEqual(summary["manifest26_final_status"], "STEP3G1D_BLOCKED")
        self.assertEqual(
            summary["manifest26_blocked_closeout_commit"],
            baseline["manifest26"]["blocked_closeout_commit"],
        )
        self.assertEqual(
            summary["manifest27_final_status"],
            "STEP3G1D_CORRECTIVE_COMPLETION_FAILED",
        )
        self.assertEqual(
            summary["manifest27_failed_campaign_commit"],
            baseline["manifest27"]["failed_campaign_commit"],
        )
        self.assertEqual(
            summary["manifest28_preregistration_commit"],
            manifest28_preregistration_commit(),
        )
        self.assertEqual(
            summary["method_correction_commit"],
            baseline["method_correction"]["commit"],
        )
        self.assertEqual(
            summary["exact_test_counts"],
            manifest["exact_test_selection"]["expected_counts"],
        )
        self.assertEqual(len(summary["process_isolation"]["groups"]), 4)
        self.assertEqual(
            summary["production_kick_sha256"],
            baseline["production_kick_sha256"],
        )
        for value in (
            summary["final_status"],
            summary["primary_finding"],
            summary["verification_envelope"],
        ):
            self.assertIn(value, report)
        self.assertIn("F_q=A^-T f_x", provider_spec)
        self.assertIn("M=[[I,0],[h J_projected,I]]", tangent)
        self.assertIn("AFFINE_EXACT", report)
        self.assertIn("NONLINEAR_SMOOTH", report)
        self.assertIn("No physical force provider", report)
        self.assertIn("Manifest 26 remains permanently", report)
        self.assertIn("Manifest 27 remains", report)
        self.assertIn("test-runner isolation, not production", report)
        self.assertIn(
            "Step 3g1c raw symplectic residual remains an inherited risk",
            report,
        )
        self.assertIn("it was not implemented or started", report)

    def test_exact_test_inventory_and_traceability_are_complete(self) -> None:
        manifest = manifest28()
        inventory = strict_json(
            DEFAULT_DESTINATION / "qualifying_test_inventory.json"
        )
        expected = manifest["exact_test_selection"]
        nodes = [value["node_id"] for value in inventory["tests"]]
        self.assertEqual(len(nodes), expected["expected_counts"]["total"])
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertTrue(
            all(value["result"] == "PASS" for value in inventory["tests"])
        )
        with (
            DEFAULT_DESTINATION / "requirements_traceability.csv"
        ).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        traced = {
            node
            for row in rows
            for node in row["exact_passing_node_ids"].split(";")
        }
        self.assertTrue(
            set(expected["step3g1d_core_node_ids"]).issubset(traced)
        )
        self.assertTrue(
            set(expected["step3g1d_integrity_node_ids"]).issubset(traced)
        )

    def test_all_metric_artifacts_satisfy_frozen_bounds(self) -> None:
        for name in (
            "force_jvp_closure_metrics.json",
            "physical_tangent_oracle_metrics.json",
            "finite_difference_metrics.json",
            "symplecticity_jacobian_symmetry_metrics.json",
            "reversibility_composition_metrics.json",
            "evaluation_accounting.json",
            "negative_control_metrics.json",
        ):
            metrics = strict_json(DEFAULT_DESTINATION / name)
            self.assertTrue(metrics["acceptance"], msg=name)
            self.assertTrue(all(metrics["acceptance"].values()), msg=name)

        finite_difference = strict_json(
            DEFAULT_DESTINATION / "finite_difference_metrics.json"
        )
        providers = {
            provider["kind"]: provider
            for provider in finite_difference["providers"]
        }
        dense = providers["dense"]
        nonlinear = providers["nonlinear"]
        self.assertEqual(dense["derivative_class"], "AFFINE_EXACT")
        for gate_name in ("force_gate", "kick_gate"):
            gate = dense[gate_name]
            self.assertTrue(gate["acceptance"])
            self.assertEqual(gate["early_improvements"], 0)
            self.assertTrue(gate["oracle_pass"])
            self.assertLessEqual(gate["largest_epsilon_error"], 2.0e-7)
            self.assertTrue(gate["roundoff_model"]["consistent"])
        self.assertEqual(nonlinear["derivative_class"], "NONLINEAR_SMOOTH")
        for gate_name in ("force_gate", "kick_gate"):
            gate = nonlinear[gate_name]
            self.assertTrue(gate["acceptance"])
            self.assertGreaterEqual(gate["early_improvements"], 3)
            self.assertLessEqual(gate["minimum"], 2.0e-7)
            self.assertLess(gate["minimum_index"], 9)

        closure = strict_json(
            DEFAULT_DESTINATION / "force_jvp_closure_metrics.json"
        )["com_projection"]
        self.assertTrue(closure["acceptance"])
        self.assertLessEqual(closure["maximum_norm_ratio"], 1.0)
        self.assertLessEqual(closure["maximum_component_ratio"], 1.0)
        self.assertTrue(
            all(
                case["projection_applied"]
                and case["exact_zero_com_output"]
                for case in closure["cases"]
            )
        )

    def test_fresh_process_regeneration_is_byte_identical(self) -> None:
        compare_fresh_regeneration()

    def test_artifact_hash_inventory_is_exact(self) -> None:
        hashes = strict_json(
            DEFAULT_DESTINATION / "artifact_hashes.json"
        )["sha256"]
        self.assertEqual(
            set(hashes), EXPECTED_ARTIFACTS - {"artifact_hashes.json"}
        )
        observed = {
            name: sha256_file(DEFAULT_DESTINATION / name)
            for name in hashes
        }
        self.assertEqual(observed, hashes)


if __name__ == "__main__":
    unittest.main()
