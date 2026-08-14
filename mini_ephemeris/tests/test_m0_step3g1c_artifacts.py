"""Deterministic artifact gates for Step 3g1c."""

from __future__ import annotations

import csv
import unittest

from mini_ephemeris.m0_step3g1c_qualification import manifest25, sha256_file, strict_json
from mini_ephemeris.m0_step3g1c_reporting import (
    DEFAULT_DESTINATION,
    EXPECTED_ARTIFACTS,
    compare_fresh_regeneration,
    validate_artifacts,
)


class Step3g1cArtifactTests(unittest.TestCase):
    def test_required_artifacts_and_strict_json(self) -> None:
        observed = {
            path.name for path in DEFAULT_DESTINATION.iterdir() if path.is_file()
        }
        self.assertEqual(observed, EXPECTED_ARTIFACTS)
        validate_artifacts()

    def test_summary_report_manifest_and_specs_agree(self) -> None:
        manifest = manifest25()
        summary = strict_json(
            DEFAULT_DESTINATION
            / "m0_step3g1c_kepler_drift_tangent_primitive_summary.json"
        )
        report = (
            DEFAULT_DESTINATION
            / "m0_step3g1c_kepler_drift_tangent_primitive_report.md"
        ).read_text(encoding="utf-8")
        hamiltonian = (
            DEFAULT_DESTINATION / "hamiltonian_supported_domain_specification.md"
        ).read_text(encoding="utf-8")
        tangent = (DEFAULT_DESTINATION / "tangent_map_derivation.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            summary["verification_envelope"],
            manifest["result_vocabulary"]["success_verification_envelope"],
        )
        for value in (
            summary["final_status"],
            summary["primary_finding"],
            summary["verification_envelope"],
        ):
            self.assertIn(value, report)
        self.assertIn("P^2/(2*mu_r) - mu_r*mu_g/r", hamiltonian)
        self.assertIn("dq/dt = P/mu_r", hamiltonian)
        self.assertIn("implicit universal-anomaly root", tangent)
        self.assertIn("No N-body dynamics", report)

    def test_exact_test_inventory_and_traceability_are_complete(self) -> None:
        manifest = manifest25()
        inventory = strict_json(DEFAULT_DESTINATION / "qualifying_test_inventory.json")
        expected = manifest["exact_test_selection"]
        nodes = [value["node_id"] for value in inventory["tests"]]
        self.assertEqual(len(nodes), expected["expected_counts"]["total"])
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertTrue(all(value["result"] == "PASS" for value in inventory["tests"]))
        with (DEFAULT_DESTINATION / "requirements_traceability.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))
        traced = {node for row in rows for node in row["exact_passing_node_ids"].split(";")}
        self.assertTrue(set(expected["step3g1c_core_node_ids"]).issubset(traced))
        self.assertTrue(set(expected["step3g1c_integrity_node_ids"][1:]).issubset(traced))

    def test_all_metric_artifacts_satisfy_frozen_bounds(self) -> None:
        for name in (
            "physical_oracle_metrics.json",
            "finite_difference_metrics.json",
            "invariant_metrics.json",
            "reversibility_metrics.json",
            "symplecticity_metrics.json",
            "solver_metrics.json",
        ):
            metrics = strict_json(DEFAULT_DESTINATION / name)
            self.assertTrue(metrics["acceptance"])
            self.assertTrue(all(metrics["acceptance"].values()))

    def test_fresh_process_regeneration_is_byte_identical(self) -> None:
        compare_fresh_regeneration()

    def test_artifact_hash_inventory_is_exact(self) -> None:
        hashes = strict_json(DEFAULT_DESTINATION / "artifact_hashes.json")["sha256"]
        self.assertEqual(set(hashes), EXPECTED_ARTIFACTS - {"artifact_hashes.json"})
        observed = {name: sha256_file(DEFAULT_DESTINATION / name) for name in hashes}
        self.assertEqual(observed, hashes)
