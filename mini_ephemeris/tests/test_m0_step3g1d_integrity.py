"""Integrity and isolation gates for Step 3g1d corrective completion."""

from __future__ import annotations

import copy
import hashlib
import re
import unittest
from unittest import mock

from mini_ephemeris.m0_step3g1d_qualification import (
    PREREGISTRATION_COMMIT,
    ROOT,
    git_output,
    install_guard,
    manifest27,
    reject_forbidden_library_path,
    run_fresh_kick_probe,
    sha256_file,
    static_safety_audit,
    verify_generated_provenance,
    verify_inherited_integrity,
)


class Step3g1dIntegrityTests(unittest.TestCase):
    def test_manifest_commit_branch_and_required_ancestry(self) -> None:
        manifest = manifest27()
        self.assertEqual(
            git_output(["branch", "--show-current"]),
            "v2-whckl-tangent-core",
        )
        manifest_path = manifest["paths"]["manifest"]
        self.assertEqual(
            git_output(["log", "-1", "--format=%H", "--", manifest_path]),
            PREREGISTRATION_COMMIT,
        )
        committed = (
            git_output(
                ["show", f"{PREREGISTRATION_COMMIT}:{manifest_path}"]
            ).encode("utf-8")
            + b"\n"
        )
        self.assertEqual(
            hashlib.sha256(committed).hexdigest(),
            sha256_file(ROOT / manifest_path),
        )
        blocked = manifest["preregistration"][
            "manifest26_blocked_closeout_commit"
        ]
        self.assertEqual(
            git_output(["rev-parse", f"{PREREGISTRATION_COMMIT}^"]),
            blocked,
        )
        for commit in (
            manifest["preregistration"]["manifest26_preregistration_commit"],
            blocked,
            PREREGISTRATION_COMMIT,
        ):
            self.assertEqual(
                git_output(["merge-base", "--is-ancestor", commit, "HEAD"]),
                "",
            )

    def test_prior_qualified_and_inherited_integrity_is_exact(self) -> None:
        result = verify_inherited_integrity()
        self.assertEqual(result["status"], "PASS")
        self.assertGreaterEqual(result["checked_hashes"], 120)
        self.assertEqual(result["historical_manifests"], 13)
        self.assertEqual(result["manifest26_error_count"], 12)
        self.assertEqual(result["protected_tags"], 2)

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
        self.assertEqual(result["selected_node_count"], 118)
        self.assertEqual(len(result["active_subprocess_call_sites"]), 4)
        self.assertEqual(result["forbidden_imports"], [])
        self.assertEqual(result["forbidden_library_mappings"], [])
        self.assertTrue(result["legacy_package_init_bypassed"])
        self.assertTrue(result["legacy_nbody_absent"])

    def test_guarded_fresh_process_hash_seed_determinism(self) -> None:
        first = run_fresh_kick_probe(1, "C")
        second = run_fresh_kick_probe(8675309, "C.UTF-8")
        self.assertEqual(first, second)
        self.assertIn('"events":["force","jvp"]', first)
        self.assertIn('"legacy_nbody":false', first)
        self.assertIn('"projection":', first)

    def test_generated_historical_hashes_are_full_exact_and_detect_manifest26_errors(self) -> None:
        result = verify_generated_provenance()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["historical_manifests"], 13)
        self.assertEqual(result["manifest26_error_count"], 12)

        manifest = manifest27()
        hashes = manifest["generated_provenance"][
            "manifests_13_through_25_sha256"
        ]
        self.assertEqual(len(hashes), 13)
        self.assertTrue(
            all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values())
        )
        errors = manifest["generated_provenance"]["manifest26_errors"]
        self.assertEqual(
            {int(value["file"].split("_", 1)[0]) for value in errors},
            set(range(13, 25)),
        )

        tampered = copy.deepcopy(manifest)
        name = next(iter(hashes))
        tampered["generated_provenance"][
            "manifests_13_through_25_sha256"
        ][name] = hashes[name][:8]
        with mock.patch(
            "mini_ephemeris.m0_step3g1d_qualification.manifest27",
            return_value=tampered,
        ), self.assertRaises(AssertionError):
            verify_generated_provenance()


if __name__ == "__main__":
    unittest.main()
