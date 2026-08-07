from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_step3e_convergence import (
    FINAL_STATUSES,
    _finite_json,
    _read_prefix,
    derive_energy_prediction,
    derive_step_accounting,
    energy_prediction_gate,
)
from mini_ephemeris.m0_timestep_convergence import _manifest_configuration
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash, sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT
    / "ephemeris_experiment_runner/manifests/17_m0_step3e_whfast_0125d_convergence_v1.json"
)


class M0Step3EConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text())

    def test_statuses_and_single_lane_are_frozen(self) -> None:
        self.assertEqual(set(self.manifest["decision_rules"]["allowed_statuses"]), FINAL_STATUSES)
        self.assertEqual(self.manifest["execution_policy"]["trajectory_count"], 1)
        self.assertTrue(self.manifest["execution_policy"]["no_benchmark_trajectory"])
        self.assertTrue(self.manifest["execution_policy"]["no_10myr"])
        self.assertEqual(self.manifest["new_lane"]["step_days"], 0.125)
        self.assertEqual(self.manifest["production_candidate_step_days"], 0.25)

    def test_exact_integer_step_accounting(self) -> None:
        accounting = derive_step_accounting(1_000_000, 100, 0.125)
        self.assertEqual(accounting["full_steps"], 2_922_000_000)
        self.assertEqual(accounting["steps_per_scientific_interval"], 292_200)
        self.assertFalse(accounting["fractional_endpoint_step"])
        self.assertEqual(
            self.manifest["endpoint_semantics"]["expected_callback_invocations"],
            accounting["full_steps"],
        )

    def test_new_lane_fingerprint_matches_runner_payload(self) -> None:
        configuration = _manifest_configuration(self.manifest, self.manifest["new_lane"])
        self.assertEqual(
            canonical_hash(configuration),
            self.manifest["new_lane"]["configuration_fingerprint"],
        )

    def test_authoritative_source_hashes_are_unchanged(self) -> None:
        for item in self.manifest["source_artifacts"].values():
            path = Path(item["path"])
            if not path.is_absolute():
                path = ROOT / path
            self.assertEqual(sha256_file(path), item["sha256"])

    def test_energy_prediction_is_rederived_from_frozen_evidence(self) -> None:
        source = json.loads(
            (
                ROOT
                / "docs/validation/m0-integrator-roundoff-diagnosis-continuation-v1/m0_integrator_roundoff_diagnosis_continuation_summary.json"
            ).read_text()
        )
        expected = self.manifest["energy_prediction"]
        actual = derive_energy_prediction(
            source, expected["components"]["ias15_energy_floor"]
        )
        self.assertAlmostEqual(actual["q_low"], expected["predicted_q_interval"][0], places=33)
        self.assertAlmostEqual(actual["q_high"], expected["predicted_q_interval"][1], places=33)
        self.assertAlmostEqual(
            actual["slope_center"],
            expected["predicted_slope_per_year"]["center"],
            places=28,
        )
        self.assertAlmostEqual(
            actual["endpoint_high"],
            expected["predicted_1myr_endpoint_envelope"][1],
            places=22,
        )

    def test_expected_doubled_slope_passes_mechanism_aware_gate(self) -> None:
        times = np.arange(0.0, 1_000_000.0 + 100.0, 100.0)
        slope = self.manifest["energy_prediction"]["predicted_slope_per_year"]["center"]
        history = slope * times
        block = {"fitted_slope_per_year": slope}
        statistics = {
            "fitted_slope_per_year": slope,
            "fitted_change_over_history": slope * 1_000_000.0,
            "max_abs": float(np.max(np.abs(history))),
            "energy_change_per_step": slope * 0.125 / 365.25,
            "blocks": [block] * 10,
            "same_sign_block_count": 10,
        }
        result = energy_prediction_gate(self.manifest, times, history, statistics)
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["slope_interval"])
        self.assertTrue(result["checks"]["history_envelope"])

    def test_nonpredicted_energy_behavior_fails(self) -> None:
        times = np.arange(0.0, 1_000_000.0 + 100.0, 100.0)
        slope = 3.0e-15
        history = slope * times
        statistics = {
            "fitted_slope_per_year": slope,
            "fitted_change_over_history": slope * 1_000_000.0,
            "max_abs": float(np.max(np.abs(history))),
            "energy_change_per_step": slope * 0.125 / 365.25,
            "blocks": [{"fitted_slope_per_year": slope}] * 10,
            "same_sign_block_count": 10,
        }
        result = energy_prediction_gate(self.manifest, times, history, statistics)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["slope_interval"])

    def test_prefix_parser_uses_progress_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            progress = root / "progress.csv"
            state = root / "state.csv"
            fields = ["time_years", "callback_invocations", "nonfinite_result_count", "configuration_fingerprint"]
            lines = [",".join(fields)]
            for index in range(101):
                lines.append(f"{100 * index},{292200 * index},0,fingerprint")
            progress.write_text(chr(10).join(lines) + chr(10))
            state.write_text("header" + chr(10) + ("row" + chr(10)) * 1010)
            payload = _read_prefix(progress, state)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["sample_index"], 100)
            self.assertEqual(payload["time_years"], 10000.0)
            self.assertEqual(payload["callback_invocations"], 29220000)
            self.assertEqual(payload["state_rows"], 1010)

    def test_strict_json_rejects_nonfinite_values(self) -> None:
        _finite_json({"finite": [0.0, 1.0]})
        with self.assertRaisesRegex(Exception, "Nonfinite JSON value"):
            _finite_json({"bad": math.nan})


if __name__ == "__main__":
    unittest.main()
