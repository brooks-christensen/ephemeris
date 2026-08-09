from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from mini_ephemeris.m0_step3f0_configuration_audit import (
    FINAL_STATUS,
    LITERATURE,
    PRIMARY_FINDING,
    SOURCE_FINDINGS,
    _build_settings_matrix,
    _manifest_contract,
    _requested_value,
)
from mini_ephemeris.m0_step3f0_configuration_contract import (
    lane_manifest_numbers,
    merge_lane_configuration,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "19_m0_step3f0_whfast_configuration_audit_v1.json"
)
REPORT_ROOT = ROOT / "docs/validation/m0-step3f0-whfast-configuration-audit-v1"
SUMMARY = REPORT_ROOT / "m0_step3f0_whfast_configuration_audit_summary.json"
REPORT = REPORT_ROOT / "m0_step3f0_whfast_configuration_audit_report.md"
MATRIX = REPORT_ROOT / "m0_step3f0_effective_settings_matrix.csv"
MODULE = ROOT / (
    "mini_ephemeris/src/mini_ephemeris/"
    "m0_step3f0_configuration_audit.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Step3f0ConfigurationAuditTests(unittest.TestCase):
    def test_manifest_preregisters_exact_matrix_and_status_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        _manifest_contract(manifest)
        self.assertEqual(manifest["effective_settings_matrix"]["expected_rows"], 777)
        self.assertIn(FINAL_STATUS, manifest["allowed_final_statuses"])
        self.assertIn(PRIMARY_FINDING, manifest["allowed_primary_findings"])
        self.assertEqual(manifest["preregistration"]["primary_finding"], "NOT_EVALUATED")

    def test_command_configuration_recovers_combined_runner_request(self) -> None:
        manifest = {
            "common_configuration": {"gravity": "basic", "N_active": -1}
        }
        lane = {
            "command": [
                "python",
                "-m",
                "mini_ephemeris.rebound_gr_tangent_backend_cli",
                "--step-days",
                "0.5",
                "--duration-years",
                "1000000",
                "--record-every-years",
                "100",
                "--archive-interval-years",
                "100000",
                "--megno-seed",
                "12345",
            ]
        }
        result = merge_lane_configuration(manifest, lane)
        self.assertEqual(result["integrator"], "whfast")
        self.assertEqual(result["step_days"], 0.5)
        self.assertEqual(result["record_every_years"], 100.0)
        self.assertTrue(result["variations"])
        self.assertTrue(result["megno"])
        self.assertEqual(result["exact_finish_time"], 1)

    def test_origin_manifest_is_included_in_matrix_provenance(self) -> None:
        self.assertEqual(
            lane_manifest_numbers({"manifests": [18], "origin_manifest": 10}),
            [10, 18],
        )

    def test_output_divisibility_distinguishes_historical_two_day_lane(self) -> None:
        zero = {"body_order": ["sun"], "masses_kg": [1.0]}
        for step in (1.0, 0.5, 0.25, 0.125):
            self.assertTrue(
                _requested_value(
                    "integration.output_targets_integer_steps",
                    {"step_days": step, "record_every_years": 100},
                    "lane",
                    zero,
                )
            )
        self.assertFalse(
            _requested_value(
                "integration.output_targets_integer_steps",
                {"step_days": 2.0, "record_every_years": 100},
                "lane",
                zero,
            )
        )

    def test_matrix_builder_is_complete_and_unique_without_archive_inference(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        zero = {
            "body_order": ["sun", "planet"],
            "masses_kg": [1.0, 1.0e-6],
            "initial": {},
            "after_gr_attachment": {},
            "after_megno": {},
        }
        config = {
            "integrator": "whfast",
            "step_days": 0.5,
            "record_every_years": 100,
            "archive_interval_years": 1000,
            "coordinates": "jacobi",
            "kernel": "default",
            "corrector": 0,
            "corrector2": 0,
            "safe_mode": 1,
            "keep_unsynchronized": 0,
            "recalculate_coordinates_this_timestep": 0,
            "variations": False,
            "megno": False,
        }
        with mock.patch(
            "mini_ephemeris.m0_step3f0_configuration_audit._lane_configuration",
            return_value=config,
        ):
            rows = _build_settings_matrix(manifest, zero, {})
        self.assertEqual(len(rows), 777)
        self.assertEqual(
            len({(row["lane_id"], row["setting"]) for row in rows}), 777
        )
        self.assertTrue(all(row["confidence"] == "INFERRED" for row in rows))

    def test_source_and_literature_cover_preregistered_questions(self) -> None:
        source_ids = {item["id"] for item in SOURCE_FINDINGS}
        self.assertTrue(
            {
                "variation_capability",
                "megno_per_step_sync",
                "archive_semantics",
                "callback_restore",
                "gr_potential_contract",
            }.issubset(source_ids)
        )
        topics = {item["topic"] for item in LITERATURE}
        self.assertIn("advanced WHFast kernels and correctors", topics)
        self.assertIn("statistical versus pointwise convergence", topics)

    def test_audit_module_has_no_executable_step_call(self) -> None:
        tree = ast.parse(MODULE.read_text())
        forbidden = {"integrate", "step", "steps", "integrate_rebound_streaming"}
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden
        ]
        self.assertEqual(calls, [])

    def test_compact_artifacts_match_manifest_and_matrix_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        summary = json.loads(SUMMARY.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(summary["final_status"], FINAL_STATUS)
        self.assertEqual(summary["primary_finding"], PRIMARY_FINDING)
        self.assertEqual(summary["manifest_sha256"], _sha256(MANIFEST))
        self.assertEqual(summary["execution"]["integration_steps"], 0)
        self.assertEqual(summary["execution"]["force_evaluations"], 0)
        self.assertEqual(summary["no_step_guard"]["blocked_call_count"], 0)
        self.assertFalse(summary["classification"]["material_misconfiguration_confirmed"])
        self.assertTrue(
            summary["classification"]["combined_lane_capability_constraint_confirmed"]
        )
        self.assertEqual(summary["effective_settings_matrix"]["sha256"], _sha256(MATRIX))
        with MATRIX.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), manifest["effective_settings_matrix"]["expected_rows"])
        self.assertEqual(len({(row["lane_id"], row["setting"]) for row in rows}), 777)
        self.assertIn("No material historical setting mismatch", REPORT.read_text())


if __name__ == "__main__":
    unittest.main()
