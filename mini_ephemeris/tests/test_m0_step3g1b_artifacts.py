"""Deterministic artifact gates for Step 3g1b."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest

from mini_ephemeris.m0_step3g1b_qualification import manifest24, sha256_file, strict_json
from mini_ephemeris.m0_step3g1b_reporting import (
    DEFAULT_DESTINATION,
    EXPECTED_ARTIFACTS,
    compare_fresh_regeneration,
    validate_artifacts,
)


class Step3g1bArtifactTests(unittest.TestCase):
    def test_required_artifacts_and_strict_json(self) -> None:
        observed = {
            path.name for path in DEFAULT_DESTINATION.iterdir() if path.is_file()
        }
        self.assertEqual(observed, EXPECTED_ARTIFACTS)
        validate_artifacts()

    def test_summary_report_manifest_and_convention_agree(self) -> None:
        manifest = manifest24()
        summary = strict_json(
            DEFAULT_DESTINATION
            / "m0_step3g1b_canonical_jacobi_tangent_primitives_summary.json"
        )
        report = (
            DEFAULT_DESTINATION
            / "m0_step3g1b_canonical_jacobi_tangent_primitives_report.md"
        ).read_text(encoding="utf-8")
        convention = (
            DEFAULT_DESTINATION / "canonical_convention_specification.md"
        ).read_text(encoding="utf-8")
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
        self.assertIn("P=A^(-T)*p", convention)
        self.assertIn("does not perform a velocity-to-momentum conversion", convention)
        self.assertIn("No physical force or JVP was evaluated", report)
        self.assertIn("No dynamical map was implemented", report)
        self.assertIn("No integration or timestep occurred", report)

    def test_exact_test_inventory_and_traceability_are_complete(self) -> None:
        manifest = manifest24()
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
        self.assertTrue(set(expected["step3g1b_core_node_ids"]).issubset(traced))
        self.assertTrue(set(expected["step3g1b_integrity_node_ids"][1:3]).issubset(traced))

    def test_numerical_metrics_satisfy_frozen_bounds(self) -> None:
        metrics = strict_json(DEFAULT_DESTINATION / "numerical_metrics.json")
        self.assertTrue(all(metrics["acceptance"].values()))
        fd = metrics["finite_difference"]
        self.assertEqual(
            [value["epsilon_hex"] for value in fd["values"]],
            manifest24()["finite_difference"]["epsilon_ladder_hex"],
        )
        self.assertLessEqual(
            fd["first_four_minimum_relative_l2_error"], fd["floor_bound"]
        )
        symplectic = metrics["symplecticity"]
        self.assertLessEqual(
            symplectic["forward"]["max_abs"], metrics["bounds"]["symplectic_absolute"]
        )
        self.assertLessEqual(
            symplectic["inverse"]["max_abs"], metrics["bounds"]["symplectic_absolute"]
        )
        self.assertLessEqual(
            max(
                metrics["dense_oracle"]["state"]["max_abs"],
                metrics["dense_oracle"]["tangent"]["max_abs"],
            ),
            metrics["bounds"]["dense_oracle_componentwise"],
        )
        self.assertLessEqual(
            max(metrics["invariance"].values()),
            metrics["bounds"]["invariance_internal"],
        )

    def test_fresh_process_regeneration_is_byte_identical(self) -> None:
        compare_fresh_regeneration()

    def test_artifact_hash_inventory_is_exact(self) -> None:
        hashes = strict_json(DEFAULT_DESTINATION / "artifact_hashes.json")["sha256"]
        self.assertEqual(set(hashes), EXPECTED_ARTIFACTS - {"artifact_hashes.json"})
        observed = {
            name: sha256_file(DEFAULT_DESTINATION / name) for name in hashes
        }
        self.assertEqual(observed, hashes)
