from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mini_ephemeris import gr_benettin_cli as gb
from mini_ephemeris.ephem import BODY_MASSES
from mini_ephemeris.long_term_stability_cli import build_rebound_simulation, rebound_state_from_sim
from mini_ephemeris.nbody import G_SI, NBodyState
from mini_ephemeris.orbital_elements import AU_M, DAY_S, JULIAN_YEAR_S, seconds_to_years


def _require_rebound():
    rebound = gb.optional_import_module("rebound")
    if rebound is None:
        raise unittest.SkipTest("REBOUND is not installed")
    return rebound


def _kernel_path() -> Path:
    path = Path("/home/peacelovephysics/ephemeris/data/de431_part-2.bsp")
    if not path.exists():
        raise unittest.SkipTest(f"DE431 test kernel not found: {path}")
    return path


def _two_body_state() -> NBodyState:
    m_sun = BODY_MASSES["sun"]
    m_jupiter = BODY_MASSES["jupiter barycenter"]
    masses = np.array([m_sun, m_jupiter], dtype=float)
    radius = 5.2 * AU_M
    speed = math.sqrt(G_SI * float(np.sum(masses)) / radius)
    positions = np.array(
        [
            [-m_jupiter / np.sum(masses) * radius, 0.0, 0.0],
            [m_sun / np.sum(masses) * radius, 0.0, 0.0],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [0.0, -m_jupiter / np.sum(masses) * speed, 0.0],
            [0.0, m_sun / np.sum(masses) * speed, 0.0],
        ],
        dtype=float,
    )
    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def _sim_from_state(state: NBodyState):
    rebound = _require_rebound()
    return build_rebound_simulation(
        rebound,
        state,
        integrator="whfast",
        step_s=16.0 * DAY_S,
        ias15_epsilon=1.0e-10,
    )


class BenettinRenormalizationTests(unittest.TestCase):
    def test_renormalization_preserves_reference_and_direction(self) -> None:
        state = _two_body_state()
        ref = _sim_from_state(state)
        shadow = _sim_from_state(state)
        delta_pos = np.zeros_like(state.positions)
        delta_vel = np.zeros_like(state.velocities)
        delta_pos[1, 0] = 1000.0
        delta_pos, delta_vel = gb.remove_com_modes(delta_pos, delta_vel, state.masses)
        target_norm = gb.scaled_norm(delta_pos, delta_vel)
        gb.apply_state_to_sim(
            shadow,
            NBodyState(state.positions + 10.0 * delta_pos, state.velocities + 10.0 * delta_vel, state.masses),
        )
        gb.synchronize_simulation(ref)
        ref_before = rebound_state_from_sim(ref, state.masses)
        pre_delta_pos, pre_delta_vel = gb.deviation_between_sims(ref, shadow, state.masses)
        pre_vector = gb.scaled_vector(pre_delta_pos, pre_delta_vel)

        diag = gb.renormalize_shadow(ref, shadow, state.masses, target_norm)

        gb.synchronize_simulation(ref)
        ref_after = rebound_state_from_sim(ref, state.masses)
        post_delta_pos, post_delta_vel = gb.deviation_between_sims(ref, shadow, state.masses)
        post_vector = gb.scaled_vector(post_delta_pos, post_delta_vel)
        self.assertTrue(np.array_equal(ref_before.positions, ref_after.positions))
        self.assertTrue(np.array_equal(ref_before.velocities, ref_after.velocities))
        post_norm = gb.scaled_norm(post_delta_pos, post_delta_vel)
        self.assertTrue(
            diag["post_renorm_relative_norm_error"] < 1.0e-10
            or abs(post_norm - target_norm) < 1.0e-14
        )
        self.assertGreater(gb.direction_cosine(pre_vector, post_vector), 1.0 - 1.0e-12)

    def test_checkpoint_resume_restores_state_growth_and_direction(self) -> None:
        rebound = _require_rebound()
        state = _two_body_state()
        ref = _sim_from_state(state)
        shadow = _sim_from_state(state)
        delta_pos = np.zeros_like(state.positions)
        delta_vel = np.zeros_like(state.velocities)
        delta_pos[1, 1] = 10.0
        delta_pos, delta_vel = gb.remove_com_modes(delta_pos, delta_vel, state.masses)
        target_norm = gb.scaled_norm(delta_pos, delta_vel)
        gb.apply_state_to_sim(
            shadow,
            NBodyState(state.positions + delta_pos, state.velocities + delta_vel, state.masses),
        )
        ref.integrate(0.25 * JULIAN_YEAR_S, exact_finish_time=1)
        shadow.integrate(0.25 * JULIAN_YEAR_S, exact_finish_time=1)
        gb.renormalize_shadow(ref, shadow, state.masses, target_norm)
        direction = gb.current_scaled_deviation_direction(ref, shadow, state.masses)
        payload = {
            "tag": "unit",
            "config_hash": "abc123",
            "current_time_years": seconds_to_years(float(ref.t)),
            "target_separation_norm": target_norm,
            "accumulated_log_growth": 1.25,
            "fit_accumulated_log_growth": 0.5,
            "fit_elapsed_years": 0.25,
            "renorm_count": 1,
            "rng_state_repr": gb.encode_random_state(__import__("random").Random(7)),
            "perturbation_vectors_m": {"jupiter barycenter": [0.0, 10.0, 0.0]},
            "scaled_deviation_direction": direction,
        }
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = gb.write_checkpoint(Path(tmp), "unit", ref, shadow, payload, keep=2)
            loaded_payload, loaded_ref, loaded_shadow = gb.load_checkpoint(checkpoint, "abc123", rebound)
            gb.validate_checkpoint_direction(loaded_payload, loaded_ref, loaded_shadow, state.masses)
            self.assertEqual(loaded_payload["renorm_count"], payload["renorm_count"])
            self.assertEqual(loaded_payload["accumulated_log_growth"], payload["accumulated_log_growth"])
            self.assertAlmostEqual(float(loaded_ref.t), float(ref.t), delta=1.0e-9)
            self.assertAlmostEqual(float(loaded_shadow.t), float(shadow.t), delta=1.0e-9)


class BenettinCliResumeTests(unittest.TestCase):
    def test_uninterrupted_and_resumed_short_runs_agree(self) -> None:
        kernel = _kernel_path()
        base_args = [
            "--kernel-path",
            str(kernel),
            "--start-date",
            "2000-01-01",
            "--step-days",
            "16",
            "--record-every-years",
            "1",
            "--model-scope",
            "two_body_jupiter",
            "--integrator",
            "whfast",
            "--gr-model",
            "none",
            "--perturb-body",
            "jupiter",
            "--perturbation-m",
            "1",
            "--renorm-years",
            "1",
            "--fit-start-years",
            "0",
            "--seed",
            "123",
            "--no-progress-bar",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_dir = root / "full"
            split_dir = root / "split"
            gb.main([*base_args, "--duration-years", "2", "--output-dir", str(full_dir), "--tag", "full"])
            gb.main(
                [
                    *base_args,
                    "--duration-years",
                    "1",
                    "--output-dir",
                    str(split_dir),
                    "--tag",
                    "split",
                    "--checkpoint-every-years",
                    "1",
                    "--checkpoint-dir",
                    str(split_dir / "checkpoints"),
                ]
            )
            gb.main(
                [
                    *base_args,
                    "--duration-years",
                    "2",
                    "--output-dir",
                    str(split_dir),
                    "--tag",
                    "split",
                    "--checkpoint-every-years",
                    "1",
                    "--checkpoint-dir",
                    str(split_dir / "checkpoints"),
                    "--resume-latest",
                ]
            )
            full = json.loads((full_dir / "benettin_summary_full.json").read_text())
            split = json.loads((split_dir / "benettin_summary_split.json").read_text())
            self.assertAlmostEqual(full["actual_time_years"], split["actual_time_years"], delta=1.0e-12)
            self.assertEqual(full["renorm_count"], split["renorm_count"])
            self.assertAlmostEqual(
                full["accumulated_log_growth"],
                split["accumulated_log_growth"],
                delta=1.0e-10,
            )


if __name__ == "__main__":
    unittest.main()
