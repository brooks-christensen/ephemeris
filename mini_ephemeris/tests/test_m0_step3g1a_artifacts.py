"""Artifact, integrity, and regeneration gates for Step 3g1a."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mini_ephemeris.m0_step3g1a_reporting import (
    DEFAULT_DESTINATION,
    MANIFEST22,
    PROTECTED_MANIFESTS,
    PROTECTED_SOURCES,
    ROOT,
    STEP3G0_TRACEABILITY,
    generate_artifacts,
)


EXPECTED_ARTIFACTS = {
    "m0_step3g1a_v2_foundation_report.md",
    "m0_step3g1a_v2_foundation_summary.json",
    "requirements_traceability.csv",
    "review_findings.json",
    "test_inventory.json",
    "v2_foundation_api_ownership_specification.md",
}


def strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


class Step3g1aArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate_artifacts(DEFAULT_DESTINATION)
        cls.summary = strict_json(
            DEFAULT_DESTINATION / "m0_step3g1a_v2_foundation_summary.json"
        )

    def test_required_artifacts_and_strict_json(self):
        self.assertEqual(
            {path.name for path in DEFAULT_DESTINATION.iterdir() if path.is_file()},
            EXPECTED_ARTIFACTS,
        )
        for name in (
            "m0_step3g1a_v2_foundation_summary.json",
            "review_findings.json",
            "test_inventory.json",
        ):
            self.assertIsInstance(strict_json(DEFAULT_DESTINATION / name), dict)

    def test_artifact_and_source_hash_inventories_are_exact(self):
        for name, expected in self.summary["artifact_inventory_sha256"].items():
            observed = hashlib.sha256((DEFAULT_DESTINATION / name).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, name)
        for relative, expected in self.summary["source_inventory_sha256"].items():
            observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)

    def test_regeneration_is_byte_identical_in_fresh_destinations(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            generate_artifacts(first)
            generate_artifacts(second)
            self.assertEqual(
                {path.name: path.read_bytes() for path in sorted(first.iterdir())},
                {path.name: path.read_bytes() for path in sorted(second.iterdir())},
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in sorted(first.iterdir())},
                {path.name: path.read_bytes() for path in sorted(DEFAULT_DESTINATION.iterdir())},
            )

    def test_traceability_preserves_step3g0_and_covers_both_high_findings(self):
        with STEP3G0_TRACEABILITY.open(newline="", encoding="utf-8") as source:
            historical = list(csv.DictReader(source))
        with (DEFAULT_DESTINATION / "requirements_traceability.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            updated = list(csv.DictReader(source))
        self.assertEqual(updated[: len(historical)], historical)
        by_id = {row["requirement_id"]: row for row in updated}
        self.assertIn("G0-001", by_id["V2-DIAG-ANGLE-001"]["existing_evidence"])
        self.assertIn("G0-002", by_id["V2-THRESH-SCOPE-001"]["existing_evidence"])
        self.assertEqual(by_id["V2-THRESH-SCOPE-001"]["current_disposition"], "PASS_STEP3G1A")

    def test_manifest_summary_report_and_api_scope_agree(self):
        manifest = strict_json(MANIFEST22)
        report = (DEFAULT_DESTINATION / "m0_step3g1a_v2_foundation_report.md").read_text(
            encoding="utf-8"
        )
        specification = (
            DEFAULT_DESTINATION / "v2_foundation_api_ownership_specification.md"
        ).read_text(encoding="utf-8")
        self.assertIn(self.summary["final_status"], manifest["result_vocabulary"]["final_status"])
        self.assertIn(
            self.summary["primary_finding"], manifest["result_vocabulary"]["primary_finding"]
        )
        self.assertIsNone(self.summary["verification_envelope"])
        for value in (
            self.summary["final_status"],
            self.summary["primary_finding"],
        ):
            self.assertIn(value, report)
        self.assertIn(manifest["result_vocabulary"]["success_verification_envelope"], report)
        self.assertIn("No physical model", specification)
        self.assertIn("No physical model", report)

    def test_protected_sources_and_manifests_remain_exact(self):
        for relative, expected in {**PROTECTED_SOURCES, **PROTECTED_MANIFESTS}.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        self.assertTrue(
            self.summary["integrity"][
                "manifest21_inherited_historical_and_archive_ledger_verified_by_safe_tests"
            ]
        )

    def test_review_resolutions_inventory_and_forbidden_operations(self):
        review = strict_json(DEFAULT_DESTINATION / "review_findings.json")
        inventory = strict_json(DEFAULT_DESTINATION / "test_inventory.json")
        self.assertEqual(review["unresolved_material_findings"], 1)
        by_id = {item["finding_id"]: item for item in review["findings"]}
        self.assertEqual(
            by_id["G1A-X001"]["status"], "UNRESOLVED_REQUIRES_FRESH_REQUALIFICATION"
        )
        self.assertTrue(
            all(item["status"].startswith("RESOLVED") for key, item in by_id.items() if key != "G1A-X001")
        )
        self.assertEqual(len(inventory["tests"]), self.summary["test_counts"]["step3g1a"])
        self.assertTrue(all(not item["dynamics_executed"] for item in inventory["tests"]))
        operations = self.summary["forbidden_operations"]
        self.assertTrue(operations["physical_force_or_jvp_experiment"])
        self.assertTrue(operations["protected_physical_force_or_jvp_reevaluated"])
        self.assertTrue(all(not value for key, value in operations.items() if "force_or_jvp" not in key))


if __name__ == "__main__":
    unittest.main()
