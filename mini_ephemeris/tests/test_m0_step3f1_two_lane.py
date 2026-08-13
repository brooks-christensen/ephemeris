from __future__ import annotations

import copy
import math
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_step3f1_analysis import _finite, _raw_detail
from mini_ephemeris.m0_step3f1_closeout import _sync_raw_threshold
from mini_ephemeris.m0_step3f1_contract import (
    BODY_NAMES,
    DEFAULT_MANIFEST,
    Step3f1Error,
    canonical_hash,
    lane_payload,
    load_json,
    validate_manifest,
)
from mini_ephemeris.m0_step3f1_runner import _build_lane, _diagnostic_snapshot
from mini_ephemeris.m0_timestep_convergence import RunData


class Step3f1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_json(DEFAULT_MANIFEST, "Manifest 20")

    def test_manifest_and_fingerprints(self) -> None:
        validate_manifest(self.manifest)
        for lane in ("P", "T"):
            self.assertEqual(
                canonical_hash(lane_payload(self.manifest, lane)),
                self.manifest["lane_contracts"][lane]["configuration_fingerprint"],
            )

    def test_duration_step_and_lane_guards_fail_closed(self) -> None:
        mutations = (
            ("duration_years", 10001),
            ("step_days", 0.125),
            ("total_steps", 14610001),
            ("scientific_samples", 102),
        )
        for field, value in mutations:
            changed = copy.deepcopy(self.manifest)
            changed["common_physical_contract"][field] = value
            with self.subTest(field=field), self.assertRaises(Step3f1Error):
                validate_manifest(changed)
        changed = copy.deepcopy(self.manifest)
        changed["lane_contracts"]["unauthorized"] = changed["lane_contracts"]["P"]
        with self.assertRaises(Step3f1Error):
            validate_manifest(changed)

    def test_callback_accounting_is_exact(self) -> None:
        common = self.manifest["common_physical_contract"]
        self.assertEqual(
            self.manifest["lane_contracts"]["P"]["expected_callback_invocations"],
            2 * common["total_steps"] + 32 + 32 * (common["scientific_samples"] - 1),
        )
        self.assertEqual(
            self.manifest["lane_contracts"]["T"]["expected_callback_invocations"],
            common["total_steps"],
        )

    def test_zero_step_lane_layout_and_copy_nonmutation(self) -> None:
        initial_hashes = []
        for lane in ("P", "T"):
            _, sim, backend, _, _, construction = _build_lane(self.manifest, lane)
            before = np.asarray([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles])
            snapshot = _diagnostic_snapshot(sim)
            after = np.asarray([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles])
            self.assertTrue(np.array_equal(before, after))
            self.assertEqual(int(sim.steps_done), 0)
            self.assertEqual(int(sim.ri_whfast.is_synchronized), 1)
            self.assertEqual(int(snapshot.N_var), 0 if lane == "P" else 10)
            self.assertEqual(backend.stats(sim)["callback_invocations"], 0)
            initial_hashes.append(construction["initial_real_sha256"])
        self.assertEqual(len(set(initial_hashes)), 1)

    def test_strict_finite_guard(self) -> None:
        _finite({"value": 1.0, "nested": [2.0]})
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(Step3f1Error):
                _finite({"value": value})

    def test_sync_threshold_is_frozen_scale(self) -> None:
        threshold = _sync_raw_threshold(self.manifest)
        self.assertEqual(
            threshold["global_scaled_rms_max"],
            self.manifest["screen_thresholds"]["tangent_sync_control"]["physical_global_scaled_rms_max"],
        )
        for name in BODY_NAMES:
            expected = max(
                10.0 * self.manifest["reference_contract"]["ias15_per_body_scaled_rms_envelope"][name],
                0.1 * self.manifest["reference_contract"]["historical_tangent_0p25_vs_ias15_per_body_scaled_rms_10k"][name],
            )
            self.assertEqual(threshold["per_body_scaled_rms_max"][name], expected)

    def test_scaled_state_aggregation_weights_every_component_equally(self) -> None:
        times = np.asarray([0.0, 100.0])
        masses = np.ones(10)
        positions = np.zeros((2, 10, 3))
        velocities = np.zeros((2, 10, 3))
        left = RunData("left", 0.25, BODY_NAMES, times, masses, positions.copy(), velocities.copy(), np.zeros_like(positions), np.zeros_like(velocities), {}, {}, {}, [])
        right = RunData("right", 0.25, BODY_NAMES, times, masses, positions.copy(), velocities.copy(), np.zeros_like(positions), np.zeros_like(velocities), {}, {}, {}, [])
        left.positions[1, 1, 0] = 149597870700.0
        detail = _raw_detail(left, right)
        self.assertAlmostEqual(detail["global_scaled_rms"], math.sqrt(1.0 / 120.0))
        self.assertEqual(detail["per_body"]["mercury barycenter"]["squared_error_contribution"], 1.0)


if __name__ == "__main__":
    unittest.main()
