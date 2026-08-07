from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_integrator_roundoff_diagnosis import (
    PHYSICAL_STATE_FIELDS,
    PROGRESS_FIELDS,
    _material_reduction,
    _synchronize_for_direction_reversal,
    _systematic_signature,
    _two_body_state,
    classify_mechanism,
    control_history_analysis,
)
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash
from mini_ephemeris.stability_diagnostics import center_of_mass_position_velocity


class M0IntegratorRoundoffDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.manifest = json.loads(
            (
                cls.root
                / "ephemeris_experiment_runner/manifests/13_m0_integrator_roundoff_diagnosis_v1.json"
            ).read_text()
        )

    def test_all_preregistered_fingerprints_match_expanded_configurations(self) -> None:
        common = self.manifest["common_configuration"]
        for lane in self.manifest["lanes"]:
            self.assertEqual(
                canonical_hash({**common, **lane["configuration"]}),
                lane["configuration_fingerprint"],
                lane["id"],
            )
        for group in ("reversibility_two_body", "ias15_benchmark"):
            definition = self.manifest["method_validation"][group]
            case_common = definition.get("common_configuration", common)
            for case in definition["cases"]:
                self.assertEqual(
                    canonical_hash({**case_common, **case["configuration"]}),
                    case["configuration_fingerprint"],
                    case["id"],
                )

    def test_manifest_bounds_every_decisive_lane(self) -> None:
        self.assertTrue(self.manifest["frozen_before_new_integrations"])
        self.assertTrue(self.manifest["execution_policy"]["no_0p125d_1myr_lane"])
        self.assertTrue(self.manifest["execution_policy"]["no_10myr_run"])
        for lane in self.manifest["lanes"]:
            config = lane["configuration"]
            duration = config.get("duration_years", config.get("forward_years"))
            self.assertLessEqual(duration, 100000.0, lane["id"])
            self.assertFalse(config["variations"], lane["id"])
            self.assertFalse(config["megno"], lane["id"])

    def test_control_history_uses_ten_equal_blocks(self) -> None:
        times = np.arange(1001, dtype=np.float64) * 100.0
        values = 2.5e-16 * times
        result = control_history_analysis(times, values, 0.5)
        self.assertEqual(len(result["blocks"]), 10)
        self.assertEqual(result["same_sign_block_count"], 10)
        self.assertAlmostEqual(result["fitted_slope_per_year"], 2.5e-16, places=28)
        self.assertAlmostEqual(
            result["energy_change_per_step"], 2.5e-16 * 0.5 / 365.25, places=30
        )

    def test_material_reduction_and_systematic_signature_are_threshold_exact(self) -> None:
        baseline = {
            "fitted_slope_per_year": 4.0,
            "max_abs": 8.0,
            "rms": 6.0,
            "p99_abs": 7.0,
        }
        candidate = {
            "fitted_slope_per_year": 1.0,
            "max_abs": 2.0,
            "rms": 1.5,
            "p99_abs": 1.75,
        }
        self.assertTrue(_material_reduction(baseline, candidate, True)["passed"])
        self.assertFalse(_material_reduction(baseline, candidate, False)["passed"])

        threshold = self.manifest["comparison_definitions"]["systematic_per_step_signature"]
        coarse = {"fitted_slope_per_year": 2.0, "same_sign_block_count": 10}
        fine = {"fitted_slope_per_year": 4.0, "same_sign_block_count": 10}
        self.assertTrue(_systematic_signature(coarse, fine, threshold)["passed"])

    def test_classification_prioritizes_integrity_and_independent_integrator(self) -> None:
        evidence = {
            "integrity_passed": True,
            "ias15_tolerance_converged": True,
            "reversibility_valid": True,
            "ias15_force_problem": False,
            "ias15_reproduces": False,
            "current_reproduces_full_both": True,
            "min_material_reduction_both": True,
            "current_material_reduction_both": False,
            "current_min_compatible_both": False,
            "full_systematic_signature": True,
            "current_systematic_signature": True,
            "any_sync_material_reduction": True,
            "random_walk_rules_passed": False,
            "bounded_rules_passed": False,
        }
        self.assertEqual(
            classify_mechanism(evidence),
            ("SYNCHRONIZATION_RECALCULATION_BIAS", "STEP3_INTEGRATOR_CONFIGURATION_CHANGE_REQUIRED"),
        )
        evidence["ias15_force_problem"] = True
        self.assertEqual(
            classify_mechanism(evidence),
            ("CORRECTED_INVARIANT_OR_FORCE_PROBLEM", "STEP3_FORCE_INVARIANT_PROBLEM"),
        )
        evidence["integrity_passed"] = False
        self.assertEqual(classify_mechanism(evidence), ("BLOCKED", "BLOCKED"))

    def test_scientific_schemas_cover_state_energy_and_callback_integrity(self) -> None:
        for field in (
            "configuration_fingerprint",
            "sample_index",
            "time_seconds",
            "mass_kg",
            "x_m",
            "vx_m_per_s",
        ):
            self.assertIn(field, PHYSICAL_STATE_FIELDS)
        for field in (
            "newtonian_energy_j",
            "gr_potential_energy_j",
            "corrected_energy_j",
            "angular_momentum_norm_kg_m2_s",
            "callback_invocations",
            "nonfinite_result_count",
        ):
            self.assertIn(field, PROGRESS_FIELDS)

    def test_two_body_validation_state_is_barycentric(self) -> None:
        definition = self.manifest["method_validation"]["reversibility_two_body"]
        config = {
            **definition["common_configuration"],
            **definition["cases"][0]["configuration"],
        }
        state = _two_body_state(config)
        position, velocity = center_of_mass_position_velocity(state)
        np.testing.assert_allclose(position, 0.0, atol=1e-9)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
        self.assertGreater(state.velocities[1, 1] - state.velocities[0, 1], 0.0)

    def test_direction_reversal_synchronizes_internal_state_then_restores_mode(self) -> None:
        class Whfast:
            keep_unsynchronized = 1

        class Simulation:
            ri_whfast = Whfast()
            observed = None

            def synchronize(self) -> None:
                self.observed = self.ri_whfast.keep_unsynchronized

        simulation = Simulation()
        _synchronize_for_direction_reversal(simulation)
        self.assertEqual(simulation.observed, 0)
        self.assertEqual(simulation.ri_whfast.keep_unsynchronized, 1)


if __name__ == "__main__":
    unittest.main()
