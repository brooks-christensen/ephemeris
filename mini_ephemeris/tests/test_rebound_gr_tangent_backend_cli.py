from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mini_ephemeris.m0_telemetry import STATE_SAMPLE_SCHEMA_VERSION
from mini_ephemeris.rebound_gr_tangent_backend_cli import (
    INTENTIONAL_INCOMPLETE_EXIT,
    RUNNER_SCHEMA_VERSION,
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
                json.dumps(
                    {
                        "complete": True,
                        "status": "COMPLETED",
                        "schema_version": RUNNER_SCHEMA_VERSION,
                        "state_sample_schema_version": STATE_SAMPLE_SCHEMA_VERSION,
                    }
                )
            )
            for name in ("state", "status", "restart", "archive"):
                paths[name].write_text("")
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
            pre_resume_paths = output_paths(output, "restart_smoke", None)
            restart_payload = json.loads(pre_resume_paths["restart"].read_text())
            original_fingerprint = restart_payload["configuration_fingerprint"]
            restart_payload["configuration_fingerprint"] = "0" * 64
            pre_resume_paths["restart"].write_text(json.dumps(restart_payload))
            rejected = subprocess.run(
                [*command, "--resume"],
                cwd=project_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "configuration fingerprint mismatch",
                (rejected.stdout + rejected.stderr).lower(),
            )
            restart_payload["configuration_fingerprint"] = original_fingerprint
            pre_resume_paths["restart"].write_text(json.dumps(restart_payload))
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
            with paths["state"].open(newline="") as handle:
                state_rows = list(csv.DictReader(handle))
            self.assertEqual(len(state_rows), 2 * len(times))
            identities = [
                (int(row["sample_index"]), int(row["body_index"]))
                for row in state_rows
            ]
            self.assertEqual(len(identities), len(set(identities)))
            self.assertEqual(
                [float(row["time_years"]) for row in state_rows],
                [value for value in times for _ in range(2)],
            )
            self.assertEqual(
                {row["configuration_fingerprint"] for row in state_rows},
                {summary["configuration_fingerprint"]},
            )
            self.assertEqual(summary["diagnostics"]["state_rows_written_total"], 10)


if __name__ == "__main__":
    unittest.main()
