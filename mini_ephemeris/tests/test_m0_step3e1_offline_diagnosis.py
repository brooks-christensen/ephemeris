from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mini_ephemeris.m0_step3e1_offline_diagnosis import (
    BODY_NAMES,
    FINAL_STATUSES,
    PRIMARY_CLASSIFICATIONS,
    _alignment,
    _atomic_csv,
    _ecliptic_vectors,
    _element_pair,
    _entity_view,
    _equatorial_vectors,
    _finite_json,
    _metric_bundle,
    _project_rtn,
    _quantiles,
    _scaled_defect,
    _solve_eccentric_anomaly,
    _three_consecutive,
    _window_slices,
    _wrap,
)
from mini_ephemeris.orbital_elements import AU_M, JULIAN_YEAR_S


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "ephemeris_experiment_runner/manifests/"
    "18_m0_step3e1_offline_state_diagnosis_v1.json"
)


class Step3e1OfflineDiagnosisTests(unittest.TestCase):
    def test_manifest_statuses_and_unobserved_result_are_frozen(self) -> None:
        payload = json.loads(MANIFEST.read_text())
        self.assertEqual(set(payload["allowed_final_statuses"]), FINAL_STATUSES)
        self.assertEqual(
            set(payload["allowed_primary_classifications"]),
            PRIMARY_CLASSIFICATIONS,
        )
        self.assertEqual(
            payload["new_diagnostics_at_preregistration"],
            {
                "calculated": False,
                "final_status_calculated": False,
                "primary_classification": "NOT_EVALUATED",
            },
        )

    def test_manifest_is_offline_only_and_has_no_run_command(self) -> None:
        payload = json.loads(MANIFEST.read_text())
        self.assertTrue(payload["constraints"]["offline_only"])
        self.assertEqual(payload["analysis_scope"]["integration_count"], 0)
        self.assertIn(
            "launch_resume_or_regenerate_any_trajectory",
            payload["constraints"]["forbidden"],
        )

    def test_manifest_expected_counts(self) -> None:
        payload = json.loads(MANIFEST.read_text())
        self.assertEqual(len(payload["stored_lanes"]), 3)
        self.assertEqual(
            sum(len(lane["artifact_inventory"]) for lane in payload["stored_lanes"]),
            21,
        )
        self.assertEqual(
            sum(
                len(inventory)
                for inventory in payload["ias15_evidence_artifacts"].values()
            ),
            15,
        )
        self.assertEqual(
            payload["derived_output_contract"]["uncommitted_compact_tables"],
            {
                "cumulative_metrics.csv": 110,
                "epoch_body_metrics.csv": 100000,
                "orbital_window_metrics.csv": 90,
                "phase_stripped_window_metrics.csv": 220,
                "window_metrics.csv": 110,
            },
        )

    def test_fixed_windows_cover_each_noninitial_epoch_once(self) -> None:
        selected = np.zeros(10000, dtype=np.int64)
        for selection in _window_slices():
            selected[selection] += 1
        np.testing.assert_array_equal(selected, np.ones(10000, dtype=np.int64))

    def test_scaled_defect_uses_manifest_17_scales(self) -> None:
        left_r = np.zeros((1, 1, 3))
        right_r = np.zeros_like(left_r)
        left_v = np.zeros_like(left_r)
        right_v = np.zeros_like(left_r)
        left_r[0, 0, 0] = AU_M
        left_v[0, 0, 1] = AU_M / JULIAN_YEAR_S
        result = _scaled_defect(left_r, left_v, right_r, right_v)
        np.testing.assert_array_equal(
            result,
            np.array([[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]]),
        )

    def test_pairwise_closure_is_componentwise(self) -> None:
        rng = np.random.default_rng(123)
        state05 = rng.normal(size=(5, 2, 6))
        state025 = rng.normal(size=(5, 2, 6))
        state0125 = rng.normal(size=(5, 2, 6))
        coarse = state05 - state025
        fine = state025 - state0125
        np.testing.assert_allclose(coarse + fine, state05 - state0125, atol=1e-15)

    def test_metric_bundle_matches_six_component_rms(self) -> None:
        values = np.ones((4, 2, 6), dtype=np.float64)
        result = _metric_bundle(values)
        self.assertEqual(result["scaled_rms"], 1.0)
        self.assertEqual(result["position_scaled_rms"], 1.0)
        self.assertEqual(result["velocity_scaled_rms"], 1.0)

    def test_entity_view(self) -> None:
        values = np.arange(3 * 2 * 6).reshape(3, 2, 6)
        self.assertIs(_entity_view(values, None), values)
        np.testing.assert_array_equal(_entity_view(values, 1), values[:, 1:2])

    def test_quantiles_are_deterministic_linear(self) -> None:
        result = _quantiles(np.array([0.0, 1.0, 2.0, 3.0]))
        self.assertEqual(result["median"], 1.5)
        self.assertEqual(result["maximum"], 3.0)
        self.assertAlmostEqual(result["p90"], 2.7)

    def test_alignment_identifies_second_order(self) -> None:
        coarse = np.array([1.0, 2.0, 3.0])
        fine = 0.25 * coarse
        result = _alignment(coarse, fine, 1e-16)
        self.assertEqual(result["order_status"], "IDENTIFIABLE")
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertAlmostEqual(result["projection"], 0.25)
        self.assertAlmostEqual(result["order"], 2.0)
        self.assertAlmostEqual(result["orthogonal_residual_fraction"], 0.0)

    def test_alignment_rejects_negative_projection(self) -> None:
        result = _alignment(np.array([1.0, 0.0]), np.array([-0.5, 0.0]), 1e-16)
        self.assertEqual(result["order_status"], "ORDER_NOT_IDENTIFIABLE")
        self.assertIsNone(result["order"])

    def test_alignment_rejects_near_floor(self) -> None:
        result = _alignment(np.array([1e-15]), np.array([1e-15]), 1e-12)
        self.assertEqual(
            result["order_status"], "ORDER_NOT_IDENTIFIABLE_NEAR_FLOOR"
        )
        self.assertIsNone(result["cosine"])

    def test_wrap_and_unwrap_convention(self) -> None:
        values = np.array([0.0, math.pi + 0.1, -math.pi - 0.1])
        wrapped = _wrap(values)
        self.assertAlmostEqual(wrapped[0], 0.0)
        self.assertAlmostEqual(wrapped[1], -math.pi + 0.1)
        self.assertAlmostEqual(wrapped[2], math.pi - 0.1)
        history = np.unwrap(np.array([3.0, -3.0]), period=2.0 * math.pi)
        self.assertGreater(history[1], history[0])

    def test_eccentric_anomaly_solver(self) -> None:
        mean = np.array([[0.2, 2.0], [4.0, 6.0]])
        eccentricity = np.array([[0.1, 0.01], [0.2, 0.05]])
        anomaly = _solve_eccentric_anomaly(mean, eccentricity)
        residual = anomaly - eccentricity * np.sin(anomaly) - mean
        np.testing.assert_allclose(residual, 0.0, atol=2e-15)

    def test_ecliptic_rotation_round_trip(self) -> None:
        rng = np.random.default_rng(456)
        values = rng.normal(size=(8, 3))
        np.testing.assert_allclose(
            _equatorial_vectors(_ecliptic_vectors(values)),
            values,
            rtol=0.0,
            atol=5e-16,
        )

    def test_rtn_projection_reconstructs_identity_basis(self) -> None:
        position = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
        velocity = position + 0.5
        basis = np.broadcast_to(np.eye(3), (2, 3, 3, 3)).copy()
        projected, checks = _project_rtn(position, velocity, basis)
        np.testing.assert_array_equal(projected[..., :3], position)
        np.testing.assert_array_equal(projected[..., 3:], velocity)
        self.assertEqual(checks["position_reconstruction_relative"], 0.0)
        self.assertEqual(checks["velocity_reconstruction_relative"], 0.0)

    def test_element_pair_wraps_classical_angles_and_builds_nonsingular(self) -> None:
        shape = (2, 1)
        left = {
            "a": np.full(shape, 2.0),
            "e": np.full(shape, 0.1),
            "i": np.full(shape, 0.2),
            "Omega": np.full(shape, math.pi - 0.1),
            "omega": np.full(shape, math.pi - 0.2),
            "varpi": np.full(shape, math.pi - 0.3),
            "M": np.full(shape, math.pi - 0.4),
            "lambda": np.full(shape, math.pi - 0.5),
            "hhat": np.broadcast_to(np.array([0.0, 0.0, 1.0]), (2, 1, 3)),
            "evec": np.broadcast_to(np.array([0.1, 0.0, 0.0]), (2, 1, 3)),
            "ecc_k": np.full(shape, 0.1),
            "ecc_h": np.zeros(shape),
            "inc_p": np.zeros(shape),
            "inc_q": np.full(shape, 0.1),
        }
        right = {
            key: np.array(value, copy=True) for key, value in left.items()
        }
        right["Omega"][:] = -math.pi + 0.1
        result = _element_pair(left, right)
        np.testing.assert_allclose(result["Omega"], -0.2)
        np.testing.assert_allclose(result["nonphase_combined"], 0.0)

    def test_three_consecutive_respects_start_index(self) -> None:
        self.assertFalse(_three_consecutive([True, True, True, False, False], 2))
        self.assertTrue(
            _three_consecutive([False, False, True, True, True, False], 2)
        )

    def test_finite_json_rejects_nan(self) -> None:
        with self.assertRaises(Exception):
            _finite_json({"bad": math.nan})

    def test_atomic_csv_has_stable_schema_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            _atomic_csv(path, [{"a": 1, "b": 2}, {"a": 3, "b": None}])
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], "a,b")
            self.assertEqual(len(lines), 3)

    def test_body_contract_has_ten_real_particles(self) -> None:
        self.assertEqual(len(BODY_NAMES), 10)
        self.assertEqual(BODY_NAMES[0], "sun")
        self.assertEqual(BODY_NAMES[-1], "pluto barycenter")


if __name__ == "__main__":
    unittest.main()
