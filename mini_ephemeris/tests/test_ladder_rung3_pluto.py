"""Focused non-trajectory tests for the Pluto rung-3 harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ladder_rung3_pluto.py"
SPEC = importlib.util.spec_from_file_location("ladder_rung3_pluto", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
RUNG3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNG3)


class CircularSpan(unittest.TestCase):
    def test_libration_centered_on_180_does_not_look_like_circulation(self) -> None:
        span, center = RUNG3.minimum_circular_span_degrees(
            [150.0, 170.0, 190.0, 210.0]
        )
        self.assertAlmostEqual(span, 60.0)
        self.assertAlmostEqual(center, 180.0)

    def test_arc_crossing_zero_is_measured_circularly(self) -> None:
        span, center = RUNG3.minimum_circular_span_degrees([350.0, 10.0])
        self.assertAlmostEqual(span, 20.0)
        self.assertAlmostEqual(center, 0.0)

    def test_invalid_angles_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RUNG3.minimum_circular_span_degrees([])
        with self.assertRaises(ValueError):
            RUNG3.minimum_circular_span_degrees([0.0, float("nan")])


class ObservationTiming(unittest.TestCase):
    def test_targets_are_irregular_monotone_and_anchor_comparison_epochs(self) -> None:
        targets = RUNG3.observation_targets(4.0e8, 2000)
        self.assertTrue(np.all(np.diff(targets) > 0.0))
        self.assertEqual(targets[-1], 4.0e8)
        self.assertEqual(targets[999], 2.0e8)
        intervals = np.diff(targets)
        self.assertGreater(float(np.ptp(intervals)), 1.0)

    def test_too_few_samples_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RUNG3.observation_targets(1.0, 19)


class EphemerisInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.particles, cls.provenance = RUNG3.load_ephemeris_initial_conditions()

    def test_de440s_identity_and_fixed_j2000_state(self) -> None:
        self.assertEqual(
            self.provenance["kernel_sha256"],
            "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2",
        )
        self.assertEqual(
            self.provenance["physical_configuration_fingerprint"],
            "7a06fd788272a0b9f430340b4a75e42627c153cd7970b0d54d37b0ddbd6fb1f3",
        )
        self.assertEqual(self.provenance["epoch_tt_jd"], 2_451_545.0)
        self.assertEqual(len(self.particles), 6)
        np.testing.assert_allclose(
            self.particles[-1]["position_au"],
            [-9.882489409088551, -27.981515457543086, -5.754614330514002],
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_terrestrial_masses_and_states_are_folded_by_gm(self) -> None:
        source = {
            item["name"]: item
            for item in self.provenance["source_states"]
        }
        expected_gm = sum(
            source[name]["gm_km3_s2"]
            for name in RUNG3.CENTRAL_COMPONENTS
        )
        weighted_position = sum(
            source[name]["gm_km3_s2"]
            * np.asarray(source[name]["position_au"])
            for name in RUNG3.CENTRAL_COMPONENTS
        ) / expected_gm
        central = self.particles[0]
        self.assertAlmostEqual(central["gm_km3_s2"], expected_gm)
        np.testing.assert_allclose(
            central["position_au"],
            weighted_position,
            rtol=0.0,
            atol=1.0e-16,
        )

    def test_build_has_six_real_particles_and_deterministic_variation(self) -> None:
        first, first_provenance = RUNG3.build(0.4, megno_seed=12345)
        second, second_provenance = RUNG3.build(0.4, megno_seed=12345)
        self.assertEqual(first.N_real, 6)
        self.assertEqual(first.N, 12)
        self.assertEqual(
            first_provenance["physical_configuration_fingerprint"],
            second_provenance["physical_configuration_fingerprint"],
        )
        first_variation = first.particles[first.N_real]
        second_variation = second.particles[second.N_real]
        self.assertEqual(first_variation.x, second_variation.x)
        self.assertEqual(first_variation.vz, second_variation.vz)


class FrozenComparisonRules(unittest.TestCase):
    @staticmethod
    def _summary(dt: float, scale: float = 1.0) -> dict:
        seeds = list(RUNG3.DEFAULT_TANGENT_SEEDS)
        seed_summaries = []
        for seed in seeds:
            seed_summaries.append({
                "seed": seed,
                "lambda_benettin_1_per_year": 5.0e-8 * scale,
                "lambda_megno_1_per_year": 5.1e-8 * scale,
            })
        return {
            "dt_years": dt,
            "duration_years": 2.0e8,
            "seeds": seeds,
            "seed_summaries": seed_summaries,
        }

    def _write_coarse(self, directory: str, fingerprint: str = "same") -> Path:
        path = Path(directory) / "coarse.json"
        payload = {
            "evidence": {
                "convergence_checkpoint_summary": self._summary(0.4),
                "initial_condition_provenance": {
                    "physical_configuration_fingerprint": fingerprint,
                },
            },
        }
        path.write_text(json.dumps(payload))
        return path

    def test_both_estimators_must_pass_the_ten_percent_gate_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coarse = self._write_coarse(directory)
            provenance = {"physical_configuration_fingerprint": "same"}
            passing = RUNG3.load_timestep_comparison(
                coarse,
                self._summary(0.2, 1.05),
                provenance,
            )
            failing = RUNG3.load_timestep_comparison(
                coarse,
                self._summary(0.2, 1.25),
                provenance,
            )
        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])

    def test_incompatible_physical_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coarse = self._write_coarse(directory, fingerprint="old")
            with self.assertRaises(ValueError):
                RUNG3.load_timestep_comparison(
                    coarse,
                    self._summary(0.2),
                    {"physical_configuration_fingerprint": "new"},
                )

    def test_seed_parser_rejects_duplicates_and_out_of_range_values(self) -> None:
        self.assertEqual(RUNG3.parse_seed_list("1,2,3"), (1, 2, 3))
        for value in ("1,1", "-1", "4294967296", ""):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    RUNG3.parse_seed_list(value)


class GateReachability(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[float, dict, list[dict], dict, dict]:
        seed_summaries = [
            {
                "halving_ratio": 1.0,
                "max_relative_energy_drift": 1.0e-10,
                "estimator_relative_disagreement": 0.01,
            }
            for _ in RUNG3.DEFAULT_TANGENT_SEEDS
        ]
        summary = {
            "seed_count": 5,
            "all_classified_chaotic": True,
            "all_saturation_excluded": True,
        }
        return (
            2.0e8,
            summary,
            seed_summaries,
            {"librating": True},
            {"passed": True},
        )

    def test_all_preregistered_conditions_can_pass(self) -> None:
        conditions = RUNG3.rung_conditions(*self._inputs())
        self.assertTrue(all(passed for _, passed in conditions))

    def test_every_condition_has_an_exercised_failure_branch(self) -> None:
        cases = (
            ("duration", "duration is at least 200 Myr"),
            ("seed_count", "at least 5 independent tangent seeds"),
            ("classification", "every seed is classified chaotic"),
            ("halving", "every seed has halving ratio in [0.85, 1.15]"),
            ("energy", "every seed has energy drift < 1e-09"),
            (
                "estimators",
                "every seed has Benettin/MEGNO disagreement < 0.2",
            ),
            (
                "saturation",
                "linear variational saturation is excluded for every seed",
            ),
            ("resonance", "Pluto is protected by the 3:2 resonance"),
            (
                "timestep",
                "same-duration dt-halving changes both estimators by < 10% per seed",
            ),
        )
        for case, expected_label in cases:
            with self.subTest(case=case):
                years, summary, seeds, resonance, comparison = self._inputs()
                if case == "duration":
                    years = 2.0e8 - 1.0
                elif case == "seed_count":
                    summary["seed_count"] = 4
                elif case == "classification":
                    summary["all_classified_chaotic"] = False
                elif case == "halving":
                    seeds[0]["halving_ratio"] = 0.5
                elif case == "energy":
                    seeds[0]["max_relative_energy_drift"] = 1.0e-8
                elif case == "estimators":
                    seeds[0]["estimator_relative_disagreement"] = 0.25
                elif case == "saturation":
                    summary["all_saturation_excluded"] = False
                elif case == "resonance":
                    resonance["librating"] = False
                elif case == "timestep":
                    comparison["passed"] = False
                conditions = RUNG3.rung_conditions(
                    years,
                    summary,
                    seeds,
                    resonance,
                    comparison,
                )
                failed = [label for label, passed in conditions if not passed]
                self.assertEqual(failed, [expected_label])


class CollisionSafety(unittest.TestCase):
    def test_json_writer_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            RUNG3.write_json_atomic(path, {"status": "first"})
            self.assertEqual(json.loads(path.read_text()), {"status": "first"})
            with self.assertRaises(FileExistsError):
                RUNG3.write_json_atomic(path, {"status": "second"})


if __name__ == "__main__":
    unittest.main()
