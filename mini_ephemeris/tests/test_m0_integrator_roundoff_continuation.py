from __future__ import annotations

import json
from pathlib import Path
import unittest

from mini_ephemeris.m0_integrator_roundoff_diagnosis import (
    _audit_continuation_provenance,
    _require_reversibility_gate,
    _reversibility_timestep_diagnostics,
)
from mini_ephemeris.long_term_stability_cli import stability_body_list
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash


class M0IntegratorRoundoffContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.path_13 = (
            cls.root
            / "ephemeris_experiment_runner/manifests/13_m0_integrator_roundoff_diagnosis_v1.json"
        )
        cls.path_15 = (
            cls.root
            / "ephemeris_experiment_runner/manifests/15_m0_integrator_roundoff_diagnosis_continuation_v1.json"
        )
        cls.manifest_13 = json.loads(cls.path_13.read_text())
        cls.manifest_15 = json.loads(cls.path_15.read_text())

    def test_every_locked_manifest_13_section_is_exact(self) -> None:
        locked = self.manifest_15["continuation_provenance"][
            "locked_sections_imported_unchanged"
        ]
        self.assertEqual(len(locked), 11)
        for section in locked:
            self.assertEqual(
                self.manifest_15[section], self.manifest_13[section], section
            )

    def test_all_lane_and_benchmark_fingerprints_remain_exact(self) -> None:
        common = self.manifest_15["common_configuration"]
        self.assertEqual(len(self.manifest_15["lanes"]), 10)
        for lane in self.manifest_15["lanes"]:
            self.assertEqual(
                canonical_hash({**common, **lane["configuration"]}),
                lane["configuration_fingerprint"],
                lane["id"],
            )
        benchmark = self.manifest_15["method_validation"]["ias15_benchmark"]
        for case in benchmark["cases"]:
            self.assertEqual(
                canonical_hash({**common, **case["configuration"]}),
                case["configuration_fingerprint"],
                case["id"],
            )

    def test_frozen_body_order_matches_the_canonical_tuple(self) -> None:
        canonical = stability_body_list("full_with_pluto", include_pluto=True)
        self.assertIsInstance(canonical, tuple)
        self.assertEqual(
            list(canonical), self.manifest_15["common_configuration"]["body_names"]
        )

    def test_manifest_14_is_the_only_method_gate_supersession(self) -> None:
        gate = self.manifest_15["continuation_method_gate"]
        self.assertEqual(gate["required_status"], "REVERSIBILITY_GATE_PASSED")
        self.assertTrue(gate["absolute_roundoff_gate_controls_validity"])
        self.assertTrue(gate["fine_coarse_return_error_ratios_are_diagnostic_only"])
        self.assertFalse(gate["fine_coarse_return_error_ratios_affect_validity"])
        self.assertTrue(gate["does_not_change_scientific_or_causal_thresholds"])

    def test_continuation_provenance_audit_passes(self) -> None:
        evidence = _audit_continuation_provenance(
            self.manifest_15, self.root
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(
            evidence["source_manifest_14_status"], "REVERSIBILITY_GATE_PASSED"
        )
        self.assertTrue(evidence["fine_coarse_ratios_diagnostic_only"])

    def test_reversibility_lanes_use_manifest_14_gate(self) -> None:
        evidence = _require_reversibility_gate(self.path_15, self.manifest_15)
        self.assertEqual(evidence["source"], "manifest 14")
        self.assertEqual(evidence["status"], "REVERSIBILITY_GATE_PASSED")
        self.assertTrue(evidence["fine_coarse_ratios_diagnostic_only"])

    def test_reversibility_timestep_ratios_are_diagnostic_only(self) -> None:
        metrics = {
            "global_scaled_rms": 2.0,
            "corrected_energy_relative_difference": -4.0,
            "angular_momentum_vector_relative_difference": 8.0,
            "center_of_mass_position_error_m": 16.0,
            "center_of_mass_velocity_error_m_per_s": 32.0,
        }
        lanes = {}
        for mode in ("current_sync", "min_sync"):
            lanes[f"m0_diag_reversibility_{mode}_0p5d_10k"] = {
                "metrics": metrics
            }
            lanes[f"m0_diag_reversibility_{mode}_0p25d_10k"] = {
                "metrics": {name: value / 2.0 for name, value in metrics.items()}
            }
        diagnostics = _reversibility_timestep_diagnostics(lanes)
        for item in diagnostics.values():
            self.assertTrue(item["diagnostic_only"])
            self.assertFalse(item["affects_validity"])
            for metric in item["metrics"].values():
                self.assertEqual(metric["fine_over_coarse_ratio"], 0.5)
                self.assertEqual(metric["apparent_order"], 1.0)

    def test_execution_matrix_is_serial_and_bounded(self) -> None:
        execution = self.manifest_15["continuation_execution"]
        commands = execution["preregistered_commands_in_order"]
        self.assertTrue(execution["serial"])
        self.assertEqual(len(commands), 13)
        self.assertTrue(commands[0].endswith(" audit"))
        self.assertTrue(commands[1].endswith(" benchmark-ias15"))
        self.assertTrue(commands[-1].endswith(" analyze"))
        self.assertFalse(any("validate-reversibility" in command for command in commands))
        self.assertTrue(execution["no_0p125_day_1myr_lane"])
        self.assertTrue(execution["no_stage4"])
        self.assertTrue(execution["no_10myr"])

    def test_historical_statuses_remain_immutable(self) -> None:
        provenance = self.manifest_15["continuation_provenance"]
        self.assertEqual(provenance["historical_manifest_13_primary_mechanism"], "BLOCKED")
        self.assertEqual(provenance["historical_manifest_13_step3_status"], "BLOCKED")
        self.assertTrue(provenance["historical_manifest_13_result_immutable"])


if __name__ == "__main__":
    unittest.main()
