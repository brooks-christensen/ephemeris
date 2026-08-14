"""Integrity and guard tests unique to the Step 3g1a requalification."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import unittest

from mini_ephemeris.m0_step3g1a_requalification import (
    FORBIDDEN_EXECUTABLE_NAMES,
    ROOT,
    active_guard,
    git_output,
    manifest23,
    reject_forbidden_library_path,
    run_fresh_v2_probe,
    sha256_file,
    static_safety_audit,
    strict_json,
)
from mini_ephemeris.v2 import (
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    CompiledLayout,
    LayoutMismatch,
    require_canonical_tangent_compatible,
)


class RequalificationIntegrityTests(unittest.TestCase):
    def test_guard_rejects_sentinel_module(self) -> None:
        self.assertTrue(active_guard().strict)
        with self.assertRaises(ImportError):
            __import__("requalification_forbidden_sentinel")

    def test_guard_rejects_sentinel_library_path(self) -> None:
        with self.assertRaises(RuntimeError):
            reject_forbidden_library_path(
                "/tmp/requalification_forbidden_library_sentinel.so"
            )

    def test_fresh_process_guarded_import_and_determinism(self) -> None:
        first = run_fresh_v2_probe(1, "C")
        second = run_fresh_v2_probe(8675309, "C.UTF-8")
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertFalse(payload["legacy_nbody"])

    def test_static_import_and_subprocess_closure_matches_manifest(self) -> None:
        audit = static_safety_audit()
        expected = manifest23()["exact_test_selection"]["expected_counts"]
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["selected_node_count"], expected["total"])
        self.assertEqual(audit["forbidden_imports"], [])
        self.assertEqual(audit["forbidden_library_mappings"], [])
        self.assertTrue(audit["legacy_package_init_bypassed"])
        self.assertTrue(audit["legacy_nbody_absent"])
        self.assertEqual(len(audit["active_subprocess_call_sites"]), 3)

    def test_v2_source_hashes_and_inventory_match_manifest(self) -> None:
        manifest = manifest23()
        expected = manifest["v2_implementation_sha256"]
        observed_paths = {
            str(path.relative_to(ROOT))
            for path in (ROOT / "mini_ephemeris/src/mini_ephemeris/v2").iterdir()
            if path.is_file() and path.suffix == ".py"
        }
        self.assertEqual(observed_paths, set(expected))
        for relative, digest in expected.items():
            self.assertEqual(sha256_file(ROOT / relative), digest, relative)

    def test_protected_historical_archive_and_trajectory_hashes_match(self) -> None:
        manifest = manifest23()
        direct_ledgers = (
            manifest["protected_sources"],
            manifest["protected_manifests"],
            manifest["historical_step3g1a_sha256"],
            manifest["inherited_integrity_ledger"]["frozen_archives"],
            manifest["inherited_integrity_ledger"]["frozen_trajectories"],
        )
        for ledger in direct_ledgers:
            for relative, digest in ledger.items():
                self.assertEqual(sha256_file(ROOT / relative), digest, relative)
        manifest21_path = (
            ROOT
            / "ephemeris_experiment_runner/manifests/"
            "21_m0_step3g0_verification_architecture_audit_v1.json"
        )
        manifest21 = strict_json(manifest21_path)
        for key in ("protected_files", "historical_documents", "frozen_archives"):
            for relative, digest in manifest21[key].items():
                self.assertEqual(sha256_file(ROOT / relative), digest, relative)

    def test_annotated_tags_resolve_exactly(self) -> None:
        for name, expected in manifest23()["protected_tags"].items():
            self.assertEqual(git_output(["cat-file", "-t", name]), "tag")
            self.assertEqual(git_output(["rev-parse", name]), expected["tag_object"])
            self.assertEqual(
                git_output(["rev-list", "-n", "1", name]), expected["commit"]
            )

    def test_manifest22_outcome_and_artifacts_remain_historical(self) -> None:
        manifest = manifest23()
        summary_path = (
            ROOT
            / "docs/validation/m0-step3g1a-v2-foundation-v1/"
            "m0_step3g1a_v2_foundation_summary.json"
        )
        report_path = (
            ROOT
            / "docs/validation/m0-step3g1a-v2-foundation-v1/"
            "m0_step3g1a_v2_foundation_report.md"
        )
        summary = strict_json(summary_path)
        report = report_path.read_text(encoding="utf-8")
        historical = manifest["historical_result_preservation"]
        self.assertEqual(summary["final_status"], historical["final_status"])
        self.assertEqual(summary["primary_finding"], historical["primary_finding"])
        self.assertIn(historical["final_status"], report)
        self.assertIn(historical["primary_finding"], report)
        self.assertIsNone(summary["verification_envelope"])

    def test_canonical_layout_mismatch_is_rejected(self) -> None:
        layout = CompiledLayout(("sun", "planet"), "sun")
        reordered = CompiledLayout(("planet", "sun"), "sun")
        state = CanonicalJacobiState(
            layout,
            ((0.0, 0.0, 0.0),) * 2,
            ((1.0, 0.0, 0.0),) * 2,
            "si_v1",
        )
        tangent = CanonicalJacobiTangentState(
            reordered,
            ((0.0, 1.0, 0.0),) * 2,
            ((0.0, 0.0, 1.0),) * 2,
            "si_v1",
        )
        with self.assertRaises(LayoutMismatch):
            require_canonical_tangent_compatible(state, tangent)

    def test_no_forbidden_executable_surface_in_selected_sources(self) -> None:
        manifest = manifest23()
        for relative in manifest["v2_implementation_sha256"]:
            path = ROOT / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names = {
                node.name.lower()
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            self.assertTrue(FORBIDDEN_EXECUTABLE_NAMES.isdisjoint(names), path)
        self.assertNotIn("mini_ephemeris.nbody", sys.modules)


if __name__ == "__main__":
    unittest.main()
