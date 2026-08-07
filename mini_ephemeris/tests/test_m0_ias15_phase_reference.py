from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_ias15_phase_reference import (
    REFERENCE_STATUSES,
    _envelope,
    pair_diagnostics,
)
from mini_ephemeris.m0_integrator_roundoff_diagnosis import classify_mechanism
from mini_ephemeris.nbody import NBodyState
from mini_ephemeris.rebound_gr_tangent_backend_cli import canonical_hash, sha256_file


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT
    / "ephemeris_experiment_runner/manifests/16_m0_ias15_phase_reference_v1.json"
)


class M0IAS15PhaseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text())

    def test_manifest_statuses_and_fingerprints_are_frozen(self) -> None:
        self.assertEqual(
            set(self.manifest["qualification_rules"]["reference_statuses"]),
            REFERENCE_STATUSES,
        )
        common = self.manifest["common_configuration"]
        for definition in (self.manifest["benchmark"], self.manifest["new_lane"]):
            self.assertEqual(
                canonical_hash({**common, **definition["configuration"]}),
                definition["configuration_fingerprint"],
            )
        self.assertEqual(self.manifest["new_lane"]["configuration"]["epsilon"], 1e-9)

    def test_immutable_source_manifest_hashes(self) -> None:
        for label in ("manifest_13", "manifest_14", "manifest_15"):
            source = self.manifest["source_artifacts"][label]
            self.assertEqual(sha256_file(Path(source["path"])), source["sha256"])

    def test_rtn_detects_transverse_phase_offset(self) -> None:
        masses = np.asarray([1.9884098713264225e30, 3.3009873694619664e23])
        times = np.asarray([0.0, 1.0, 2.0, 3.0])
        right = []
        left = []
        for time_years in times:
            positions = np.asarray([[0.0, 0.0, 0.0], [5.79e10, 0.0, 0.0]])
            velocities = np.asarray([[0.0, 0.0, 0.0], [0.0, 47_000.0, 0.0]])
            right.append(NBodyState(positions=positions, velocities=velocities, masses=masses))
            shifted = positions.copy()
            shifted[1, 1] += time_years * 10.0
            left.append(NBodyState(positions=shifted, velocities=velocities, masses=masses))
        payload = pair_diagnostics(times, left, right, ["sun", "mercury"])
        self.assertAlmostEqual(payload["global_transverse_position_variance_fraction"], 1.0)
        self.assertAlmostEqual(
            payload["bodies"]["mercury"]["rtn"]["position_m"]["transverse"]["maximum_abs"],
            30.0,
        )

    def test_componentwise_envelope(self) -> None:
        masses = np.asarray([1.0, 1.0])
        zero = NBodyState(np.zeros((2, 3)), np.zeros((2, 3)), masses)
        x = NBodyState(
            np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.zeros((2, 3)),
            masses,
        )
        y = NBodyState(
            np.asarray([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
            np.zeros((2, 3)),
            masses,
        )
        lanes = [
            {"lane_id": "a", "times": np.asarray([0.0]), "states": [zero]},
            {"lane_id": "b", "times": np.asarray([0.0]), "states": [x]},
            {"lane_id": "c", "times": np.asarray([0.0]), "states": [y]},
        ]
        payload = _envelope(lanes, ["sun", "planet"])
        self.assertEqual(payload["worst_component"]["component"], "y")
        self.assertEqual(payload["worst_component"]["body"], "planet")

    def test_only_ias15_gate_supersession_unblocks_frozen_classifier(self) -> None:
        source = json.loads(
            Path(self.manifest["source_artifacts"]["manifest_15_summary"]["path"]).read_text()
        )
        evidence = dict(source["classification_evidence"])
        self.assertFalse(evidence["ias15_tolerance_converged"])
        evidence["ias15_tolerance_converged"] = True
        self.assertEqual(
            classify_mechanism(evidence),
            ("SYSTEMATIC_WHFAST_STEP_BIAS", "STEP3_NUMERICAL_FLOOR_CHARACTERIZED"),
        )


if __name__ == "__main__":
    unittest.main()
