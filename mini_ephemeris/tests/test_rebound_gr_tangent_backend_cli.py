from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mini_ephemeris.rebound_gr_tangent_backend_cli import (
    INTENTIONAL_INCOMPLETE_EXIT,
    RunnerSafetyError,
    apply_output_policy,
    build_parser,
    output_paths,
)


class BackendCliSafetyTests(unittest.TestCase):
    def test_collision_skip_and_overwrite_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = output_paths(root, "case", None)
            paths["progress"].write_text("partial")
            with self.assertRaisesRegex(RunnerSafetyError, "will not be overwritten"):
                apply_output_policy(
                    paths,
                    skip_if_complete=False,
                    overwrite_existing_output=False,
                    resume=False,
                )
            with self.assertRaisesRegex(RunnerSafetyError, "valid final summary"):
                apply_output_policy(
                    paths,
                    skip_if_complete=True,
                    overwrite_existing_output=False,
                    resume=False,
                )
            paths["summary"].write_text(
                json.dumps({"complete": True, "status": "COMPLETED"})
            )
            self.assertEqual(
                apply_output_policy(
                    paths,
                    skip_if_complete=True,
                    overwrite_existing_output=False,
                    resume=False,
                ),
                "skip",
            )
            self.assertTrue(paths["progress"].exists())
            self.assertEqual(
                apply_output_policy(
                    paths,
                    skip_if_complete=False,
                    overwrite_existing_output=True,
                    resume=False,
                ),
                "fresh",
            )
            self.assertFalse(paths["progress"].exists())
            self.assertFalse(paths["summary"].exists())

    def test_long_production_duration_requires_explicit_authorization(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--kernel-path",
                "kernel.bsp",
                "--duration-years",
                "100001",
                "--production-duration-approved",
            ]
        )
        self.assertTrue(args.production_duration_approved)
        self.assertEqual(args.duration_years, 100001.0)

    def test_fresh_process_stop_and_resume_has_unique_samples(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        kernel = project_root / "data/de440s.bsp"
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            command = [
                sys.executable,
                "-m",
                "mini_ephemeris.rebound_gr_tangent_backend_cli",
                "--kernel-path",
                str(kernel),
                "--model-scope",
                "two_body_mercury",
                "--duration-years",
                "0.04",
                "--step-days",
                "10",
                "--record-every-years",
                "0.01",
                "--archive-interval-years",
                "0.01",
                "--gr-tangent-backend",
                "c",
                "--output-dir",
                str(output),
                "--tag",
                "restart_smoke",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "mini_ephemeris/src")
            partial = subprocess.run(
                [
                    *command,
                    "--overwrite-existing-output",
                    "--stop-after-years",
                    "0.02",
                ],
                cwd=project_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                partial.returncode,
                INTENTIONAL_INCOMPLETE_EXIT,
                msg=partial.stdout + partial.stderr,
            )
            resumed = subprocess.run(
                [*command, "--resume"],
                cwd=project_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            paths = output_paths(output, "restart_smoke", None)
            summary = json.loads(paths["summary"].read_text())
            self.assertTrue(summary["complete"])
            self.assertTrue(summary["restart"]["resumed"])
            self.assertTrue(
                summary["restart"]["callbacks_increased_after_reattachment"]
            )
            with paths["progress"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            times = [float(row["time_years"]) for row in rows]
            self.assertEqual(times, [0.0, 0.01, 0.02, 0.03, 0.04])
            self.assertEqual(len(times), len(set(times)))


if __name__ == "__main__":
    unittest.main()
