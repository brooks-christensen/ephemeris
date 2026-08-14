"""Integrity and isolation gates for Step 3g1b."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from mini_ephemeris.m0_step3g1b_qualification import (
    PREREGISTRATION_COMMIT,
    ROOT,
    git_output,
    install_guard,
    manifest24,
    reject_forbidden_library_path,
    run_fresh_transform_probe,
    sha256_file,
    static_safety_audit,
    verify_inherited_integrity,
)


class Step3g1bIntegrityTests(unittest.TestCase):
    def test_manifest_and_required_ancestry(self) -> None:
        manifest = manifest24()
        self.assertEqual(git_output(["branch", "--show-current"]), "v2-whckl-tangent-core")
        self.assertEqual(
            git_output(["log", "-1", "--format=%H", "--", manifest["paths"]["manifest"]]),
            PREREGISTRATION_COMMIT,
        )
        committed = git_output(
            ["show", f"{PREREGISTRATION_COMMIT}:{manifest['paths']['manifest']}"]
        ).encode("utf-8") + b"\n"
        self.assertEqual(
            hashlib.sha256(committed).hexdigest(),
            sha256_file(ROOT / manifest["paths"]["manifest"]),
        )
        for commit in manifest["required_ancestry"].values():
            self.assertEqual(
                git_output(["merge-base", "--is-ancestor", commit, "HEAD"]), ""
            )
        self.assertEqual(
            git_output(["merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"]),
            "",
        )

    def test_step3g1a_sources_and_tests_are_byte_exact(self) -> None:
        manifest = manifest24()
        observed = {
            relative: sha256_file(ROOT / relative)
            for relative in manifest["qualified_step3g1a_read_only_sha256"]
        }
        self.assertEqual(observed, manifest["qualified_step3g1a_read_only_sha256"])

    def test_protected_historical_archive_trajectory_and_tag_integrity(self) -> None:
        result = verify_inherited_integrity()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["checked_hashes"], 60)
        self.assertEqual(result["protected_tags"], 2)

    def test_guard_rejects_only_harmless_sentinels(self) -> None:
        guard = install_guard()
        with self.assertRaises(ImportError):
            guard.find_spec("step3g1b_forbidden_sentinel", None)
        with self.assertRaises(RuntimeError):
            reject_forbidden_library_path("/tmp/step3g1b_forbidden_library_sentinel.so")

    def test_static_import_node_and_subprocess_closure(self) -> None:
        result = static_safety_audit()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_node_count"], 53)
        self.assertEqual(result["forbidden_imports"], [])
        self.assertEqual(result["forbidden_library_mappings"], [])
        self.assertTrue(result["legacy_package_init_bypassed"])
        self.assertTrue(result["legacy_nbody_absent"])

    def test_guarded_fresh_process_hash_seed_determinism(self) -> None:
        first = run_fresh_transform_probe(1, "C")
        second = run_fresh_transform_probe(8675309, "C.UTF-8")
        self.assertEqual(first, second)
        self.assertIn('"legacy_nbody":false', first)
