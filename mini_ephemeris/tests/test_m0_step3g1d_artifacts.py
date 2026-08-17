"""Deterministic artifact gates for Step 3g1d."""

from __future__ import annotations

import csv
import unittest

from mini_ephemeris.m0_step3g1d_qualification import (
    manifest26,
    sha256_file,
    strict_json,
)
from mini_ephemeris.m0_step3g1d_reporting import (
    DEFAULT_DESTINATION,
    EXPECTED_ARTIFACTS,
    compare_fresh_regeneration,
    validate_artifacts,
)


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
        manifest = manifest26()
        summary = strict_json(
            DEFAULT_DESTINATION
            / "m0_step3g1d_interaction_kick_tangent_primitive_summary.json"
        )
        report = (
            DEFAULT_DESTINATION
            / "m0_step3g1d_interaction_kick_tangent_primitive_report.md"
        ).read_text(encoding="utf-8")
        provider_spec = (
            DEFAULT_DESTINATION
            / "canonical_kick_provider_specification.md"
        ).read_text(encoding="utf-8")
        tangent = (
            DEFAULT_DESTINATION / "tangent_map_derivation.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            summary["verification_envelope"],
            manifest["result_vocabulary"]["verification_envelope"],
        )
        for value in (
            summary["final_status"],
            summary["primary_finding"],
            summary["verification_envelope"],
        ):
            self.assertIn(value, report)
        self.assertIn("F_q=A^-T f_x", provider_spec)
        self.assertIn("M=[[I,0],[h J_F,I]]", tangent)
        self.assertIn("No physical force provider", report)
        self.assertIn(
            "Step 3g1c raw symplectic residual remains an inherited risk",
            report,
        )

    def test_exact_test_inventory_and_traceability_are_complete(self) -> None:
        manifest = manifest26()
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
            self.assertTrue(
                all(metrics["acceptance"].values()), msg=name
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
