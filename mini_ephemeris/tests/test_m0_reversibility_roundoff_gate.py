from __future__ import annotations

import json
from pathlib import Path
import unittest

from mini_ephemeris.m0_reversibility_roundoff_gate import (
    FINAL_STATUSES,
    absolute_return_checks,
    diagnostic_ratios,
    final_status,
)
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash


class M0ReversibilityRoundoffGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.manifest_14 = json.loads(
            (
                cls.root
                / "ephemeris_experiment_runner/manifests/14_m0_reversibility_roundoff_gate_v1.json"
            ).read_text()
        )
        cls.manifest_13 = json.loads(
            (
                cls.root
                / "ephemeris_experiment_runner/manifests/13_m0_integrator_roundoff_diagnosis_v1.json"
            ).read_text()
        )

    def test_manifest_reuses_all_four_frozen_configurations_exactly(self) -> None:
        definition = self.manifest_13["method_validation"]["reversibility_two_body"]
        originals = {case["id"]: case for case in definition["cases"]}
        self.assertEqual(len(self.manifest_14["cases"]), 4)
        for case in self.manifest_14["cases"]:
            original = originals[case["source_case_id"]]
            copied = {
                key: case["configuration"][key]
                for key in original["configuration"]
            }
            self.assertEqual(copied, original["configuration"])
            self.assertEqual(
                canonical_hash({**definition["common_configuration"], **copied}),
                original["configuration_fingerprint"],
            )
            self.assertEqual(
                case["scientific_configuration_fingerprint"],
                original["configuration_fingerprint"],
            )

    def test_run_fingerprints_are_unique_and_reproducible(self) -> None:
        fingerprints = set()
        for case in self.manifest_14["cases"]:
            identity = {
                "schema_version": 1,
                "experiment_id": self.manifest_14["experiment_id"],
                "run_id": case["run_id"],
                "scientific_configuration_fingerprint": case[
                    "scientific_configuration_fingerprint"
                ],
            }
            self.assertEqual(canonical_hash(identity), case["run_fingerprint"])
            fingerprints.add(case["run_fingerprint"])
        self.assertEqual(len(fingerprints), 4)

    def test_ratio_is_diagnostic_only_and_status_vocabulary_is_exact(self) -> None:
        self.assertEqual(
            set(self.manifest_14["gate_definition"]["final_statuses"]),
            FINAL_STATUSES,
        )
        self.assertTrue(
            self.manifest_14["gate_definition"]["ratio_is_diagnostic_only"]
        )
        self.assertTrue(
            self.manifest_14["gate_definition"]["ratio_must_not_determine_validity"]
        )
        basis = self.manifest_14["mathematical_basis"]
        self.assertIn("not a truncation-order convergence estimate", basis["interpretation"])

    @staticmethod
    def _summaries(*, failed_source: str | None = None) -> dict[str, dict]:
        summaries = {}
        for mode in ("current_sync", "min_sync"):
            for step, rms in (("0p5d", 1e-30), ("0p25d", 1e-6)):
                source = f"two_body_{mode}_{step}"
                summaries[source] = {
                    "status": "COMPLETED",
                    "source_case_id": source,
                    "case_passed": source != failed_source,
                    "metrics": {"global_scaled_rms": rms},
                }
        return summaries

    def test_large_fine_coarse_ratio_cannot_fail_an_absolute_pass(self) -> None:
        summaries = self._summaries()
        ratios = diagnostic_ratios(summaries)
        self.assertGreater(ratios["min_sync"]["fine_over_coarse"], 4.0)
        self.assertFalse(ratios["min_sync"]["affects_validity"])
        self.assertEqual(
            final_status(summaries),
            (
                "REVERSIBILITY_GATE_PASSED",
                {"current_sync": True, "min_sync": True},
            ),
        )

    def test_absolute_failure_and_incomplete_matrix_are_distinct(self) -> None:
        failed = self._summaries(failed_source="two_body_min_sync_0p25d")
        self.assertEqual(final_status(failed)[0], "REVERSIBILITY_GATE_FAILED")
        failed.pop("two_body_current_sync_0p5d")
        self.assertEqual(final_status(failed)[0], "BLOCKED")

    def test_uniform_absolute_limits_cover_every_requested_metric(self) -> None:
        metrics = {
            "global_scaled_rms": 1e-9,
            "corrected_energy_relative_difference": -2e-9,
            "angular_momentum_vector_relative_difference": 3e-9,
            "center_of_mass_position_error_m": 10.0,
            "center_of_mass_velocity_error_m_per_s": 1e-6,
            "per_body": {
                "sun": {
                    "scaled_rms": 1e-9,
                    "position_error_m": 1.0,
                    "velocity_error_m_per_s": 1e-7,
                },
                "mercury": {
                    "scaled_rms": 2e-9,
                    "position_error_m": 2.0,
                    "velocity_error_m_per_s": 2e-7,
                },
            },
        }
        checks = absolute_return_checks(
            metrics, self.manifest_14["absolute_limits"]
        )
        self.assertTrue(all(item["passed"] for item in checks.values()))
        self.assertEqual(len(checks), 11)

    def test_historical_blocked_result_is_preserved(self) -> None:
        provenance = self.manifest_14["provenance"]
        self.assertEqual(provenance["historical_manifest_13_primary_mechanism"], "BLOCKED")
        self.assertEqual(provenance["historical_manifest_13_step3_status"], "BLOCKED")
        self.assertTrue(provenance["historical_result_is_immutable"])


if __name__ == "__main__":
    unittest.main()
