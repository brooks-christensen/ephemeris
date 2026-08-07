from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from mini_ephemeris.m0_timestep_convergence import INNER_BODIES, _manifest_configuration
from mini_ephemeris.m0_timestep_convergence_0p25 import (
    ORDERED_RUN_IDS,
    _energy_pattern,
    evaluate_candidate_criteria,
)
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash


class M0TimestepConvergence0p25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.manifest = json.loads(
            (
                root
                / "ephemeris_experiment_runner/manifests/11_m0_timestep_convergence_0p25_v1.json"
            ).read_text()
        )
        cls.step3_manifest = json.loads(
            (
                root
                / "ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json"
            ).read_text()
        )

    def test_manifest_inherits_definitions_and_thresholds_exactly(self) -> None:
        self.assertEqual(
            self.manifest["comparison_definitions"],
            self.step3_manifest["comparison_definitions"],
        )
        self.assertEqual(self.manifest["thresholds"], self.step3_manifest["thresholds"])
        self.assertEqual(
            self.manifest["shared_configuration"],
            self.step3_manifest["shared_configuration"],
        )
        self.assertEqual(
            self.manifest["endpoint_semantics"],
            self.step3_manifest["endpoint_semantics"],
        )

    def test_decisive_fingerprint_and_expected_counts_are_frozen(self) -> None:
        run = self.manifest["decisive_run"]
        self.assertEqual(run["step_days"], 0.25)
        self.assertEqual(
            canonical_hash(_manifest_configuration(self.manifest, run)),
            "3e79729659677339dd5a4cd64c9f8d217af3d97a863786b9388b6a5b8b42533c",
        )
        endpoint = self.manifest["endpoint_semantics"]
        self.assertEqual(endpoint["expected_scientific_samples_per_run"], 10001)
        self.assertEqual(endpoint["expected_state_rows_per_run"], 100010)
        self.assertEqual(
            len(self.manifest["artifact_inventory_before_launch"]["expected_new_artifacts"]),
            7,
        )

    def _inputs(self) -> tuple:
        per_body_coarse = {body: {"rms": 2.0} for body in INNER_BODIES}
        per_body_fine = {body: {"rms": 0.5} for body in INNER_BODIES}
        elements = {
            body: {
                "eccentricity_max_abs_difference": 1.0e-8,
                "semimajor_axis_max_relative_difference": 1.0e-9,
            }
            for body in INNER_BODIES
        }
        coarse = {
            "physical": {"global_scaled_rms": 2.0, "per_body": per_body_coarse},
            "tangent": {"direction_discrepancy_rms": 1.0e-2},
        }
        fine = {
            "physical": {"global_scaled_rms": 0.5, "per_body": per_body_fine},
            "tangent": {
                "final_direction_cosine": 0.99999,
                "direction_discrepancy_rms": 1.0e-3,
            },
            "orbital_elements": {"per_body": elements},
            "megno": {"final_abs_difference": 1.0e-5, "history_rms_difference": 1.0e-5},
            "lcn": {"final_accumulated_abs_difference": 1.0e-5},
        }
        runs = {
            run_id: SimpleNamespace(integrity={"passed": True})
            for run_id in ORDERED_RUN_IDS
        }
        perihelion = {
            run_id: {"mean_rate_arcsec_per_century": 500.0 + index * 1.0e-4}
            for index, run_id in enumerate(ORDERED_RUN_IDS)
        }
        energy = {
            run_id: {
                "max_abs": value,
                "rms": value / 2.0,
                "p99_abs": value * 0.9,
                "fitted_change_over_1myr": value * 0.1,
            }
            for run_id, value in zip(ORDERED_RUN_IDS, (4.0e-9, 2.0e-9, 1.0e-9))
        }
        angular = {run_id: {"max_abs": 1.0e-10} for run_id in ORDERED_RUN_IDS}
        return runs, coarse, fine, perihelion, energy, angular

    def test_candidate_status_logic_passes_registered_synthetic_case(self) -> None:
        criteria = evaluate_candidate_criteria(self.manifest, *self._inputs())
        self.assertTrue(all(result["passed"] for result in criteria.values()))

    def test_further_energy_worsening_fails_without_roundoff_waiver(self) -> None:
        runs, coarse, fine, perihelion, energy, angular = self._inputs()
        energy[ORDERED_RUN_IDS[2]] = {
            "max_abs": 3.0e-9,
            "rms": 1.5e-9,
            "p99_abs": 2.7e-9,
            "fitted_change_over_1myr": 1.0e-10,
        }
        criteria = evaluate_candidate_criteria(
            self.manifest, runs, coarse, fine, perihelion, energy, angular
        )
        self.assertFalse(criteria["corrected_energy"]["passed"])
        self.assertFalse(
            criteria["corrected_energy"]["metrics_nonincreasing"]["max_abs"]
        )
        pattern = _energy_pattern(criteria, energy)
        self.assertTrue(pattern["physical_convergence_with_further_energy_worsening"])


if __name__ == "__main__":
    unittest.main()
