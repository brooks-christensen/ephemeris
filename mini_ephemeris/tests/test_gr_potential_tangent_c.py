from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import rebound

from mini_ephemeris.gr_potential_tangent import (
    gr_potential_accelerations_and_tangent,
)
from mini_ephemeris.gr_potential_tangent_c import (
    CBackendCompatibilityError,
    build_c_backend,
    default_artifact_path,
    load_c_backend,
)
from mini_ephemeris.nbody import G_SI


class GrPotentialTangentCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_c_backend(force=True)
        cls.backend = load_c_backend()

    def test_abi_and_direct_callback_execution(self) -> None:
        sim = rebound.Simulation()
        sim.G = 1.0
        sim.add(m=1.0)
        sim.add(m=1.0e-3, x=1.0, vy=1.0)
        sim.add_variation().particles[1].x = 1.0
        self.backend.attach(sim, c_m_per_s=100.0)
        proof = self.backend.hot_path_proof(sim)
        self.assertTrue(proof["addresses_match"])
        self.assertFalse(proof["python_callback_in_force_path"])
        self.assertFalse(hasattr(sim, "_afp"))
        before = self.backend.stats(sim)
        sim.integrate(0.01)
        after = self.backend.stats(sim)
        self.assertGreater(after["callback_invocations"], before["callback_invocations"])
        self.assertGreater(after["real_gr_accel_norm_count"], 0)
        self.assertGreater(after["tangent_gr_accel_norm_count"], 0)
        self.assertEqual(after["nonfinite_result_count"], 0)

    def test_pointwise_matches_python_and_zero_scale_is_exact(self) -> None:
        positions = np.array(
            [
                [1.0e8, -2.0e8, 3.0e8],
                [1.8e10, 2.0e9, -5.0e9],
                [-7.0e9, 4.0e8, 3.0e9],
            ]
        )
        masses = np.array([1.9884987698e30, 3.3011e23, 4.8673848426e24])
        delta = np.arange(9, dtype=float).reshape(3, 3) - 4.0
        py_acc, py_tangent = gr_potential_accelerations_and_tangent(
            positions, masses, delta, gravitational_constant=G_SI
        )
        c_acc, c_tangent = self.backend.pointwise(
            positions, masses, delta, gravitational_constant=G_SI
        )
        np.testing.assert_allclose(c_acc, py_acc, rtol=1.0e-14, atol=1.0e-22)
        np.testing.assert_allclose(c_tangent, py_tangent, rtol=1.0e-13, atol=1.0e-30)
        zero_acc, zero_tangent = self.backend.pointwise(
            positions,
            masses,
            delta,
            gravitational_constant=G_SI,
            coefficient_scale=0.0,
        )
        self.assertTrue(np.array_equal(zero_acc, np.zeros_like(zero_acc)))
        self.assertTrue(np.array_equal(zero_tangent, np.zeros_like(zero_tangent)))

    def test_stale_artifact_metadata_is_rejected(self) -> None:
        artifact = default_artifact_path()
        metadata = artifact.parent / "build_metadata.json"
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw)
            copied_artifact = destination / artifact.name
            shutil.copy2(artifact, copied_artifact)
            payload = json.loads(metadata.read_text())
            payload["source_sha256"] = "0" * 64
            (destination / "build_metadata.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                CBackendCompatibilityError, "stale or incompatible"
            ):
                load_c_backend(artifact_path=copied_artifact)


if __name__ == "__main__":
    unittest.main()
