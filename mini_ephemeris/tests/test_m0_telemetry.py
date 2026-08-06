from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import rebound

from mini_ephemeris.gr_potential_tangent import C_M_PER_S
from mini_ephemeris.m0_telemetry import (
    STATE_SAMPLE_FIELDS,
    STATE_SAMPLE_SCHEMA_VERSION,
    TelemetrySchemaError,
    gr_potential_energy,
    read_state_samples,
    state_sample_rows,
)
from mini_ephemeris.nbody import G_SI, NBodyState
from mini_ephemeris.rebound_gr_tangent_backend_cli import output_paths


class M0TelemetryTests(unittest.TestCase):
    def test_gr_potential_energy_matches_analytic_two_body_expression(self) -> None:
        central_mass = 1.7e30
        planet_mass = 4.2e24
        radius = 8.3e10
        scale = 0.75
        translation = np.array([2.0e8, -3.0e8, 5.0e8])
        state = NBodyState(
            positions=np.array([translation, translation + [radius, 0.0, 0.0]]),
            velocities=np.zeros((2, 3)),
            masses=np.array([central_mass, planet_mass]),
        )
        expected = (
            -3.0
            * scale
            * G_SI**2
            * central_mass**2
            * planet_mass
            / (C_M_PER_S**2 * radius**2)
        )
        actual = gr_potential_energy(state, coefficient_scale=scale)
        np.testing.assert_allclose(actual, expected, rtol=3.0e-16, atol=0.0)
        self.assertEqual(gr_potential_energy(state, coefficient_scale=0.0), 0.0)

        radial_force = -6.0 * scale * G_SI**2 * central_mass**2 * planet_mass / (
            C_M_PER_S**2 * radius**3
        )
        self.assertAlmostEqual(radial_force, 2.0 * expected / radius, places=15)

    def test_state_schema_round_trip_and_fingerprint_rejection(self) -> None:
        sim = rebound.Simulation()
        sim.G = G_SI
        sim.add(m=1.0e30, x=1.0, vx=2.0)
        sim.add(m=2.0e24, x=3.0, vy=4.0)
        variation = sim.add_variation()
        variation.particles[1].x = 5.0
        rows = state_sample_rows(
            sim,
            ("sun", "planet"),
            sample_index=0,
            configuration_fingerprint="fingerprint",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["variation_x_m"], 5.0)
        self.assertEqual(rows[0]["state_sample_schema_version"], STATE_SAMPLE_SCHEMA_VERSION)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "state.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=STATE_SAMPLE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            loaded = read_state_samples(
                path,
                body_names=("sun", "planet"),
                configuration_fingerprint="fingerprint",
            )
            self.assertEqual(len(loaded), 2)
            with self.assertRaisesRegex(
                TelemetrySchemaError, "configuration fingerprint mismatch"
            ):
                read_state_samples(
                    path,
                    body_names=("sun", "planet"),
                    configuration_fingerprint="different",
                )

    def test_isolated_two_body_corrected_energy_is_better_conserved(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        kernel = project_root / "data/de440s.bsp"
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            command = [
                sys.executable,
                "-m",
                "mini_ephemeris.rebound_gr_tangent_backend_cli",
                "--kernel-path",
                str(kernel),
                "--model-id",
                "m0_grpot_emb_pluto_v1",
                "--model-scope",
                "two_body_mercury",
                "--duration-years",
                "0.04",
                "--step-days",
                "10",
                "--record-every-years",
                "0.01",
                "--archive-interval-years",
                "0.01",
                "--gr-tangent-backend",
                "c",
                "--output-dir",
                str(output),
                "--tag",
                "energy_smoke",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(project_root / "mini_ephemeris/src")
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode, 0, msg=completed.stdout + completed.stderr
            )
            paths = output_paths(output, "energy_smoke", None)
            summary = json.loads(paths["summary"].read_text())
            diagnostics = summary["diagnostics"]
            self.assertTrue(diagnostics["corrected_energy_better_conserved"])
            self.assertGreater(diagnostics["corrected_energy_improvement_factor"], 10.0)
            self.assertLess(
                diagnostics["max_corrected_energy_rel_change"],
                diagnostics["max_newtonian_energy_component_rel_change"],
            )
            with paths["state"].open(newline="") as handle:
                state_rows = list(csv.DictReader(handle))
            self.assertEqual(len(state_rows), 10)
            identities = {
                (int(row["sample_index"]), int(row["body_index"]))
                for row in state_rows
            }
            self.assertEqual(len(identities), len(state_rows))


if __name__ == "__main__":
    unittest.main()
