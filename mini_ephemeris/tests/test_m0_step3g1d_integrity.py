"""Integrity and isolation gates for Step 3g1d requalification."""

from __future__ import annotations

import copy
import hashlib
import re
import unittest
from unittest import mock

from mini_ephemeris.m0_step3g1d_qualification import (
    ROOT,
    git_output,
    install_guard,
    manifest28,
    manifest28_preregistration_commit,
    reject_forbidden_library_path,
    run_fresh_kick_probe,
    sha256_file,
    static_safety_audit,
    verify_manifest28_provenance,
    verify_requalification_integrity,
)


class Step3g1dIntegrityTests(unittest.TestCase):
    def test_manifest_commit_branch_and_required_ancestry(self) -> None:
        manifest = manifest28()
        commit = manifest28_preregistration_commit()
        self.assertEqual(
            git_output(["branch", "--show-current"]),
            "v2-whckl-tangent-core",
        )
        manifest_path = manifest["paths"]["manifest"]
        committed = git_output(
            ["show", f"{commit}:{manifest_path}"],
            binary=True,
        )
        self.assertIsInstance(committed, bytes)
        self.assertEqual(
            hashlib.sha256(committed).hexdigest(),
            sha256_file(ROOT / manifest_path),
        )
        self.assertEqual(
            git_output(["rev-parse", f"{commit}^"]),
            manifest["preregistration"]["parent_commit"],
        )
        self.assertEqual(
            git_output(
                [
                    "diff-tree",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit,
                ]
            ),
            manifest_path,
        )
        for required in (
            manifest["baseline"]["manifest27"]["failed_campaign_commit"],
            manifest["baseline"]["method_correction"]["commit"],
            commit,
        ):
            self.assertEqual(
                git_output(
                    ["merge-base", "--is-ancestor", required, "HEAD"]
                ),
                "",
            )

    def test_prior_qualified_and_inherited_integrity_is_exact(self) -> None:
        result = verify_requalification_integrity()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["checked_hashes"], 190)
        self.assertEqual(result["historical_manifests"], 15)
        self.assertEqual(result["manifest26_error_count"], 12)
        self.assertTrue(result["manifest27_failed_result"])
        self.assertEqual(result["protected_tags"], 2)
        self.assertEqual(
            result["method_commit"],
            manifest28()["baseline"]["method_correction"]["commit"],
        )

    def test_guard_rejects_only_harmless_sentinels(self) -> None:
        guard = install_guard()
        with self.assertRaises(ImportError):
            guard.find_spec("step3g1d_forbidden_sentinel", None)
        with self.assertRaises(RuntimeError):
            reject_forbidden_library_path(
                "/tmp/step3g1d_forbidden_library_sentinel.so"
            )

    def test_static_import_node_and_subprocess_closure(self) -> None:
        result = static_safety_audit()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_node_count"], 124)
        self.assertEqual(len(result["active_subprocess_call_sites"]), 4)
        self.assertEqual(result["forbidden_imports"], [])
        self.assertEqual(result["forbidden_library_mappings"], [])
        self.assertTrue(result["legacy_package_init_bypassed"])
        self.assertTrue(result["legacy_nbody_absent"])
        self.assertEqual(
            result["production_kick_sha256"],
            manifest28()["baseline"]["production_kick_sha256"],
        )

    def test_guarded_fresh_process_hash_seed_determinism(self) -> None:
        first = run_fresh_kick_probe(1, "C")
        second = run_fresh_kick_probe(8675309, "C.UTF-8")
        self.assertEqual(first, second)
        self.assertIn('"events":["force","jvp"]', first)
        self.assertIn('"legacy_nbody":false', first)
        self.assertIn('"projection":', first)

    def test_generated_historical_hashes_are_full_exact_and_detect_manifest26_errors(self) -> None:
        result = verify_manifest28_provenance()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["historical_manifests"], 15)
        self.assertEqual(result["manifest26_error_count"], 12)
        self.assertTrue(result["manifest27_failed_result"])

        manifest = manifest28()
        hashes = manifest["generated_integrity"][
            "historical_manifests_13_through_27_sha256"
        ]
        self.assertEqual(len(hashes), 15)
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())
        )

        tampered = copy.deepcopy(manifest)
        name = next(iter(hashes))
        tampered["generated_integrity"][
            "historical_manifests_13_through_27_sha256"
        ][name] = hashes[name][:8]
        with mock.patch(
            "mini_ephemeris.m0_step3g1d_qualification.manifest28",
            return_value=tampered,
        ), self.assertRaises(AssertionError):
            verify_manifest28_provenance()


if __name__ == "__main__":
    unittest.main()
