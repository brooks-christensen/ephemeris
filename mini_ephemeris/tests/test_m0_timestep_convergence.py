from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from mini_ephemeris.m0_timestep_convergence import (
    INNER_BODIES,
    RunData,
    _linear_fit,
    _manifest_configuration,
    _pair_physical,
    _pair_tangent,
    _record_targets,
    _series_metrics,
    canonical_hash,
    evaluate_criteria,
)


def synthetic_run(
    run_id: str,
    positions: np.ndarray,
    variation_positions: np.ndarray,
) -> RunData:
    sample_count, body_count, _ = positions.shape
    zeros = np.zeros_like(positions)
    return RunData(
        run_id=run_id,
        step_days=1.0,
        body_names=tuple(f"body-{index}" for index in range(body_count)),
        times=np.arange(sample_count, dtype=float) * 100.0,
        masses=np.ones(body_count),
        positions=positions,
        velocities=zeros,
        variation_positions=variation_positions,
        variation_velocities=zeros,
        progress={},
        summary={},
        integrity={"passed": True},
        inventory=[],
    )


class M0TimestepConvergenceTests(unittest.TestCase):
    def test_scaled_physical_and_tangent_metrics(self) -> None:
        baseline_positions = np.zeros((3, 2, 3))
        coarse_positions = baseline_positions.copy()
        coarse_positions[:, 1, 0] = 2.0
        fine_positions = baseline_positions.copy()
        fine_positions[:, 1, 0] = 0.5
        tangent = np.ones((3, 2, 3))
        tangent_coarse = tangent.copy()
        tangent_coarse[:, 1, 1] += 0.1
        tangent_fine = tangent.copy()
        tangent_fine[:, 1, 1] += 0.01

        baseline = synthetic_run("baseline", baseline_positions, tangent)
        coarse = synthetic_run("coarse", coarse_positions, tangent_coarse)
        fine = synthetic_run("fine", fine_positions, tangent_fine)
        coarse_physical = _pair_physical(coarse, baseline)
        fine_physical = _pair_physical(fine, baseline)
        self.assertAlmostEqual(
            fine_physical["global_scaled_rms"]
            / coarse_physical["global_scaled_rms"],
            0.25,
        )
        coarse_tangent = _pair_tangent(coarse, baseline)
        fine_tangent = _pair_tangent(fine, baseline)
        self.assertGreaterEqual(fine_tangent["final_direction_cosine"], 0.9999)
        self.assertLess(
            fine_tangent["direction_discrepancy_rms"],
            coarse_tangent["direction_discrepancy_rms"],
        )

    def test_centered_fit_and_series_worst_epoch(self) -> None:
        times = np.array([0.0, 100.0, 200.0, 300.0])
        values = 2.5e-9 * times - 7.0
        slope, intercept = _linear_fit(times, values)
        np.testing.assert_allclose(slope, 2.5e-9, rtol=5.0e-10, atol=0.0)
        self.assertAlmostEqual(intercept, -7.0, places=13)
        metrics = _series_metrics(times, np.array([0.0, 1.0e-10, -4.0e-10, 2.0e-10]))
        self.assertEqual(metrics["max_abs"], 4.0e-10)
        self.assertEqual(metrics["max_abs_worst_epoch_years"], 200.0)

    def test_manifest_endpoint_and_fingerprints_are_frozen(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (
                project_root
                / "ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json"
            ).read_text()
        )
        targets = _record_targets(1_000_000.0, 100.0)
        self.assertEqual((len(targets), targets[0], targets[-1]), (10001, 0.0, 1_000_000.0))
        self.assertEqual(len(targets) * 10, 100010)
        for run in manifest["runs"]:
            self.assertEqual(
                canonical_hash(_manifest_configuration(manifest, run)),
                run["configuration_fingerprint"],
            )

    def test_registered_status_logic(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        manifest = json.loads(
            (
                project_root
                / "ephemeris_experiment_runner/manifests/10_m0_timestep_convergence_v1.json"
            ).read_text()
        )
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
        run_ids = [run["id"] for run in manifest["runs"]]
        runs = {run_id: SimpleNamespace(integrity={"passed": True}) for run_id in run_ids}
        perihelion = {
            run_id: {"mean_rate_arcsec_per_century": 500.0 + index * 1.0e-4}
            for index, run_id in enumerate(run_ids)
        }
        energy = {
            run_id: {
                "max_abs": value,
                "rms": value / 2.0,
                "p99_abs": value * 0.9,
                "fitted_change_over_1myr": value * 0.1,
            }
            for run_id, value in zip(run_ids, (4.0e-9, 2.0e-9, 1.0e-9))
        }
        angular = {run_id: {"max_abs": 1.0e-10} for run_id in run_ids}
        criteria = evaluate_criteria(
            manifest, runs, coarse, fine, perihelion, energy, angular
        )
        self.assertTrue(all(result["passed"] for result in criteria.values()))
        fine["physical"]["global_scaled_rms"] = 1.1
        criteria = evaluate_criteria(
            manifest, runs, coarse, fine, perihelion, energy, angular
        )
        self.assertFalse(criteria["physical_state"]["passed"])


if __name__ == "__main__":
    unittest.main()
