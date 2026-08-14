"""Deterministic artifact gates for the Step 3g1a requalification."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from mini_ephemeris.m0_step3g1a_requalification import (
    ROOT,
    manifest23,
    run_fresh_artifact_probe,
    sha256_file,
    strict_json,
)
from mini_ephemeris.m0_step3g1a_requalification_reporting import (
    DEFAULT_DESTINATION,
    EXPECTED_ARTIFACTS,
)


class RequalificationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = manifest23()
        cls.summary = strict_json(
            DEFAULT_DESTINATION
            / "m0_step3g1a_v2_foundation_requalification_summary.json"
        )

    def test_required_artifacts_and_strict_json(self) -> None:
        self.assertEqual(
            {path.name for path in DEFAULT_DESTINATION.iterdir() if path.is_file()},
            EXPECTED_ARTIFACTS,
        )
        for name in (
            "artifact_hashes.json",
            "import_subprocess_safety_audit.json",
            "m0_step3g1a_v2_foundation_requalification_summary.json",
            "qualifying_test_inventory.json",
        ):
            self.assertIsInstance(strict_json(DEFAULT_DESTINATION / name), dict)

    def test_summary_report_and_manifest_agree(self) -> None:
        report = (
            DEFAULT_DESTINATION
            / "m0_step3g1a_v2_foundation_requalification_report.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            self.summary["final_status"], self.manifest["result_vocabulary"]["final_status"]
        )
        self.assertIn(
            self.summary["primary_finding"],
            self.manifest["result_vocabulary"]["primary_finding"],
        )
        self.assertEqual(
            self.summary["verification_envelope"],
            self.manifest["result_vocabulary"]["success_verification_envelope"],
        )
        for value in (
            self.summary["final_status"],
            self.summary["primary_finding"],
            self.summary["verification_envelope"],
            "STEP3G1A_V2_FOUNDATION_INCOMPLETE",
        ):
            self.assertIn(value, report)
        self.assertTrue(self.summary["step3g1b_justified"])

    def test_exact_test_inventory_and_traceability_are_complete(self) -> None:
        inventory = strict_json(DEFAULT_DESTINATION / "qualifying_test_inventory.json")
        expected_nodes = (
            self.manifest["exact_test_selection"]["foundation_node_ids"]
            + self.manifest["exact_test_selection"]["requalification_node_ids"]
            + self.manifest["exact_test_selection"]["artifact_node_ids"]
        )
        self.assertEqual(
            [item["node_id"] for item in inventory["tests"]], expected_nodes
        )
        self.assertTrue(all(item["result"] == "PASS" for item in inventory["tests"]))
        self.assertEqual(inventory["inherited_step3g0_tests_executed"], 0)
        with (DEFAULT_DESTINATION / "requirements_traceability.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            rows = list(csv.DictReader(source))
        manifest22 = (
            ROOT
            / "ephemeris_experiment_runner/manifests/"
            "22_m0_step3g1a_v2_foundation_v1.json"
        )
        expected_requirements = set(strict_json(manifest22)["implementation_requirements"])
        self.assertEqual({row["requirement_id"] for row in rows}, expected_requirements)
        self.assertTrue(
            all(row["disposition"] == "PASS_REQUALIFICATION" for row in rows)
        )

    def test_import_subprocess_audit_is_closed(self) -> None:
        audit = strict_json(
            DEFAULT_DESTINATION / "import_subprocess_safety_audit.json"
        )
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["selected_node_count"], 42)
        self.assertEqual(audit["forbidden_imports"], [])
        self.assertEqual(audit["forbidden_library_mappings"], [])
        self.assertTrue(audit["legacy_package_init_bypassed"])
        self.assertTrue(audit["legacy_nbody_absent"])
        self.assertEqual(len(audit["active_subprocess_call_sites"]), 3)
        self.assertTrue(all(audit["guard"].values()))

    def test_fresh_process_regeneration_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as first_raw, tempfile.TemporaryDirectory() as second_raw:
            first = Path(first_raw)
            second = Path(second_raw)
            run_fresh_artifact_probe(first)
            run_fresh_artifact_probe(second)
            first_bytes = {
                path.name: path.read_bytes() for path in sorted(first.iterdir())
            }
            second_bytes = {
                path.name: path.read_bytes() for path in sorted(second.iterdir())
            }
            committed_bytes = {
                path.name: path.read_bytes()
                for path in sorted(DEFAULT_DESTINATION.iterdir())
                if path.is_file()
            }
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first_bytes, committed_bytes)

    def test_artifact_hash_inventory_is_exact(self) -> None:
        hashes = strict_json(DEFAULT_DESTINATION / "artifact_hashes.json")["sha256"]
        self.assertEqual(set(hashes), EXPECTED_ARTIFACTS - {"artifact_hashes.json"})
        for name, expected in hashes.items():
            self.assertEqual(sha256_file(DEFAULT_DESTINATION / name), expected, name)


if __name__ == "__main__":
    unittest.main()
