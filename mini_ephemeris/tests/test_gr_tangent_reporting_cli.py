
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from mini_ephemeris import gr_tangent_validation_matrix as gm
from mini_ephemeris import rebound_gr_tangent_cli as cli


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def legacy_gr_summary(tag: str) -> dict:
    return {
        "tag": tag,
        "variation_api": {
            "rebound_version": "4.6.0",
            "n_real": 2,
            "n_total_after_variation": 4,
            "n_variation_particles": 2,
            "readback_exact": True,
        },
        "configuration": {"duration_years": 100000.0},
        "production_metadata": {
            "n_real": 10,
            "n_total": 20,
            "n_var": 10,
            "n_var_config": 1,
            "variation_particle_count": 10,
            "variation_particle_ranges": [{"start_index": 10, "end_index_exclusive": 20, "count": 10}],
            "worker_pid": 123,
            "callback_stats": {"callback_invocations": 42},
        },
        "diagnostics": {
            "actual_time_years": 100000.0,
            "classification_hint": "regular_likely",
            "final_megno": 2.0,
            "final_lcn_1_per_year": 1.0e-14,
            "lcn_trends_toward_zero": True,
            "stable_positive_lcn_plateau": False,
            "rows_written": 101,
            "runtime_seconds": 12.5,
            "max_energy_rel_drift": 2.0e-9,
            "max_angular_momentum_rel_drift": 6.0e-12,
            "mercury_perihelion_drift_arcsec_per_century": 564.5,
        },
    }


def passed_payload(stage: str) -> dict:
    payload = {"stage": stage, "passed": True}
    if stage == "existing_1myr_smoke_audit":
        payload.update(
            {
                "row_count": 101,
                "final_time_years": 1000000.0,
                "production_archive_audit": {"snapshot_count": 101, "n_real": 10, "variation_particle_count": 10},
            }
        )
    elif stage == "monitor_process_tree_audit":
        payload.update({"sample_count": 3, "max_descendant_count": 2})
    elif stage == "dynamic_gr_tangent_oracle":
        payload.update(
            {
                "summary_by_group": {
                    "0_1": {"min_relative_norm_error": 1.0e-6, "max_direction_cosine": 0.999999}
                }
            }
        )
    elif stage == "newtonian_zero_limit_100kyr":
        payload.update(
            {
                "max_variation_norm_relative_difference": 1.0e-12,
                "min_variation_direction_cosine": 0.999999999999,
                "max_megno_difference": 1.0e-12,
                "max_lcn_difference": 1.0e-14,
            }
        )
    elif stage == "seed_comparison":
        payload.update(
            {
                "stage": "seed",
                "row": {
                    "left_classification": "regular_likely",
                    "right_classification": "regular_likely",
                    "left_final_megno": 2.0,
                    "right_final_megno": 1.98,
                    "left_final_lcn": 1.0e-14,
                    "right_final_lcn": 2.0e-15,
                },
            }
        )
    elif stage == "timestep_comparison":
        payload.update(
            {
                "stage": "timestep",
                "row": {
                    "left_classification": "regular_likely",
                    "right_classification": "regular_likely",
                    "right_final_lcn": 1.0e-14,
                    "throughput_runtime_ratio_right_over_left": 1.9,
                },
            }
        )
    elif stage == "physical_gr_trajectory_comparison_100kyr":
        payload.update(
            {
                "max_custom_vs_reboundx_scaled_phase_difference": 1.0e-5,
                "max_paired_gr_minus_newtonian_difference": 1.0e-6,
                "max_paired_gr_minus_newtonian_relative_difference": 1.0e-6,
                "warnings": [],
            }
        )
    elif stage == "gr_checkpoint_resume_equivalence_20kyr":
        payload.update(
            {
                "physical_scaled_phase_difference": 1.0e-12,
                "tangent_scaled_phase_difference": 1.0e-10,
                "tangent_direction_cosine": 0.999999999,
                "callback_invocations_after_restart": 10,
            }
        )
    return payload


def write_complete_matrix(root: Path) -> None:
    for spec in gm.EXPECTED_REQUIRED_STAGES:
        path = root / spec["path"]
        if spec["kind"] == "gr_tangent_summary":
            write_json(path, legacy_gr_summary(spec["stage"]))
        else:
            write_json(path, passed_payload(spec["stage"]))


def run_final_report(root: Path) -> int:
    args = argparse.Namespace(output_root=root, kernel_path=root / "missing-kernel.bsp")
    return gm.final_report(args)


class MatrixReportV2Tests(unittest.TestCase):
    def test_final_report_v2_normalizes_legacy_semantics_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_matrix(root)
            self.assertEqual(run_final_report(root), 0)
            summary = json.loads((root / "gr_tangent_validation_matrix_summary_v2.json").read_text())
            self.assertEqual(summary["report_schema_version"], 2)
            self.assertEqual(summary["status"], "READY_FOR_C_PORT")
            self.assertEqual(summary["expected_stage_count"], 11)
            self.assertEqual(summary["observed_stage_count"], 11)
            stage = next(item for item in summary["stage_results"] if item["stage"] == "gr_100kyr_1d_seed12345")
            diagnostics = stage["diagnostics"]
            self.assertIn("max_newtonian_energy_component_rel_change", diagnostics)
            self.assertNotIn("max_energy_rel_drift", diagnostics)
            self.assertIn("mercury_total_apsidal_drift_arcsec_per_century", diagnostics)
            self.assertNotIn("mercury_perihelion_drift_arcsec_per_century", diagnostics)
            self.assertIsNot(stage["variation_api_smoke_metadata"], stage["production_metadata"])
            self.assertEqual(stage["variation_api_smoke_metadata"]["n_real"], 2)
            self.assertEqual(stage["production_metadata"]["n_real"], 10)
            self.assertEqual(stage["production_metadata"]["n_total"], 20)
            self.assertTrue(stage["production_metadata_is_authoritative"])
            energy_def = summary["diagnostic_definitions"]["newtonian_energy_component_rel_change"]
            self.assertFalse(energy_def["includes_custom_gr_potential_energy"])
            self.assertFalse(energy_def["may_be_interpreted_as_total_conserved_energy_error"])
            self.assertEqual(summary["apsidal_drift_definition"]["kind"], "full_system_total")
            self.assertFalse(summary["apsidal_drift_definition"]["is_isolated_gr_excess"])
            rendered = json.dumps(summary)
            self.assertNotIn("max_energy_rel_drift", rendered)
            self.assertNotIn("mercury_perihelion_drift", rendered)

    def test_missing_required_stage_blocks_and_unrelated_json_cannot_satisfy_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_matrix(root)
            (root / "dynamic_gr_tangent_oracle/dynamic_gr_tangent_oracle_summary.json").unlink()
            write_json(root / "unrelated/milestone.json", {"stage": "dynamic_gr_tangent_oracle", "passed": True})
            self.assertEqual(run_final_report(root), 1)
            summary = json.loads((root / "gr_tangent_validation_matrix_summary_v2.json").read_text())
            self.assertEqual(summary["status"], "BLOCKED_MULTIPLE")
            self.assertIn("dynamic_gr_tangent_oracle", summary["missing_stages"])
            self.assertEqual(summary["observed_stage_count"], 10)

    def test_duplicate_or_milestone_json_does_not_create_duplicate_stage_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_matrix(root)
            write_json(root / "milestones/gr_100kyr_1d_seed12345.json", {"stage": "gr_100kyr_1d_seed12345", "passed": False})
            self.assertEqual(run_final_report(root), 0)
            summary = json.loads((root / "gr_tangent_validation_matrix_summary_v2.json").read_text())
            self.assertEqual(len(summary["stage_results"]), 11)
            self.assertEqual(summary["passed_stages"].count("gr_100kyr_1d_seed12345"), 1)
            self.assertEqual(summary["status"], "READY_FOR_C_PORT")

    def test_malformed_required_json_blocks_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_matrix(root)
            bad = root / "monitor_process_tree_audit/monitor_process_tree_audit.json"
            bad.write_text("{not valid json")
            self.assertEqual(run_final_report(root), 1)
            summary = json.loads((root / "gr_tangent_validation_matrix_summary_v2.json").read_text())
            self.assertEqual(summary["status"], "BLOCKED_MULTIPLE")
            self.assertIn("monitor_process_tree_audit", summary["unreadable_stages"])
            stage = next(item for item in summary["stage_results"] if item["stage"] == "monitor_process_tree_audit")
            self.assertEqual(stage["status"], "unreadable")
            self.assertIn("error", stage)

    def test_markdown_report_is_human_readable_and_includes_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_complete_matrix(root)
            self.assertEqual(run_final_report(root), 0)
            report = (root / "gr_tangent_validation_matrix_report_v2.md").read_text()
            self.assertTrue(report.startswith("# GR Tangent Validation Matrix V2"))
            self.assertIn("## Overall Verdict", report)
            self.assertIn("## Required-Stage Results", report)
            self.assertIn("## Diagnostic-Semantics Notes", report)
            self.assertIn("Newtonian energy component", report)
            self.assertIn("full-system Mercury apsidal drift", report)
            self.assertIn("Compiled-C port and C-versus-Python validation only", report)
            self.assertNotIn('"stage_results"', report[:500])

    def test_legacy_alias_helpers_accept_v1_energy_and_variation_keys(self) -> None:
        row = {"energy_rel_drift": "1.25"}
        self.assertEqual(gm.row_first_value(row, "newtonian_energy_component_rel_change", "energy_rel_drift"), "1.25")
        canonical = gm.canonicalize_gr_tangent_summary(legacy_gr_summary("legacy"))
        self.assertIn("max_newtonian_energy_component_rel_change", canonical["diagnostics"])
        self.assertIn("variation_api_smoke_metadata", canonical)
        self.assertEqual(canonical["variation_api_smoke_metadata"]["n_total_after_variation"], 4)


class GrTangentCliSafetyTests(unittest.TestCase):
    def test_skip_if_complete_has_old_safe_skip_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "gr_tangent_summary_done.json", {"completed": True})
            archive = root / "done.bin"
            archive.write_text("archive")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                cli.main([
                    "--kernel-path",
                    str(root / "missing.bsp"),
                    "--output-dir",
                    str(root),
                    "--tag",
                    "done",
                    "--simulationarchive",
                    str(archive),
                    "--skip-if-complete",
                ])
            self.assertIn("--skip-if-complete found summary", stdout.getvalue())
            self.assertTrue(archive.exists())

    def test_resume_is_rejected_with_unambiguous_message(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli.main(["--kernel-path", "missing.bsp", "--resume"])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("True production checkpoint resume is not implemented", stderr.getvalue())
        self.assertIn("--skip-if-complete", stderr.getvalue())

    def test_existing_archive_is_not_deleted_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "run.bin"
            archive.write_text("archive")
            with self.assertRaises(cli.OutputSafetyError):
                cli.apply_existing_output_policy(
                    progress_path=root / "gr_tangent_progress_run.csv",
                    summary_path=root / "gr_tangent_summary_run.json",
                    status_path=root / "gr_tangent_status_run.json",
                    simulationarchive=str(archive),
                    skip_if_complete=False,
                    overwrite_existing_output=False,
                )
            self.assertTrue(archive.exists())

    def test_overwrite_option_removes_only_exact_tagged_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = root / "gr_tangent_progress_run.csv"
            summary = root / "gr_tangent_summary_run.json"
            status = root / "gr_tangent_status_run.json"
            archive = root / "run.bin"
            unrelated = root / "gr_tangent_progress_other.csv"
            for path in [progress, summary, status, archive, unrelated]:
                path.write_text(path.name)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                policy = cli.apply_existing_output_policy(
                    progress_path=progress,
                    summary_path=summary,
                    status_path=status,
                    simulationarchive=str(archive),
                    skip_if_complete=False,
                    overwrite_existing_output=True,
                )
            self.assertEqual(policy["action"], "run")
            self.assertEqual(set(policy["removed_paths"]), {str(progress), str(summary), str(status), str(archive)})
            self.assertIn(str(progress), stdout.getvalue())
            self.assertIn(str(archive), stdout.getvalue())
            self.assertFalse(progress.exists())
            self.assertFalse(summary.exists())
            self.assertFalse(status.exists())
            self.assertFalse(archive.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
