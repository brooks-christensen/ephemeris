from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_step3e1_offline_diagnosis import (
    BODY_NAMES,
    _ias_overlap_summary,
    _ols_slope,
    _pointwise_order_summary,
    _unwrapped_element_rates,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "ephemeris_experiment_runner/manifests/"
    "18_m0_step3e1_offline_state_diagnosis_v1.json"
)


class Step3e1OfflineReportingTests(unittest.TestCase):
    def test_ols_slope_preserves_sign_and_scale(self) -> None:
        times = np.arange(10, dtype=np.float64)
        self.assertAlmostEqual(_ols_slope(times, 3.5 * times - 2.0), 3.5)
        self.assertAlmostEqual(_ols_slope(times, -0.25 * times + 9.0), -0.25)

    def test_pointwise_order_summary_excludes_no_well_conditioned_points(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        times = np.arange(1, 10001, dtype=np.float64) * 100.0
        coarse = np.ones((10000, 10, 6), dtype=np.float64) * 1.0e-4
        fine = 0.25 * coarse
        roundtrip = {
            "global_scaled_floor": 1.0e-14,
            "maximum_per_body_scaled_rms": {
                name: 1.0e-14 for name in BODY_NAMES
            },
        }
        result = _pointwise_order_summary(
            manifest, times, coarse, fine, roundtrip
        )
        for entity in ("full system", *BODY_NAMES):
            for window in result[entity]["windows"]:
                self.assertEqual(window["identifiable_fraction"], 1.0)
                self.assertAlmostEqual(window["median"], 2.0)

    def test_ias_overlap_is_limited_to_first_one_hundred_rows(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        coarse = np.ones((10000, 10, 6), dtype=np.float64) * 2.0e-6
        fine = np.ones_like(coarse) * 1.0e-6
        coarse[100:] = 2.0
        fine[100:] = 1.0
        phase = {
            method: {"coarse": 0.1 * coarse, "fine": 0.1 * fine}
            for method in ("mean_anomaly", "mean_longitude")
        }
        result = _ias_overlap_summary(manifest, coarse, fine, phase)
        self.assertEqual(result["scope_years"], 10000)
        self.assertFalse(result["extrapolated_beyond_scope"])
        self.assertAlmostEqual(
            result["entities"]["full system"]["coarse_scaled_rms"], 2.0e-6
        )
        self.assertAlmostEqual(
            result["entities"]["full system"]["fine_scaled_rms"], 1.0e-6
        )

    def test_unwrapped_angle_rate_recovers_boundary_crossing(self) -> None:
        times = np.arange(100, dtype=np.float64)
        base = np.mod(0.2 * times, 2.0 * np.pi)[:, None]
        elements = {
            "0p5": {
                field: base + offset
                for field, offset in zip(
                    ("Omega", "omega", "varpi", "M", "lambda"),
                    (0.0, 0.1, 0.2, 0.3, 0.4),
                )
            },
            "0p25": {
                field: np.repeat(values, 9, axis=1)
                for field, values in {
                    field: base + offset
                    for field, offset in zip(
                        ("Omega", "omega", "varpi", "M", "lambda"),
                        (0.0, 0.1, 0.2, 0.3, 0.4),
                    )
                }.items()
            },
            "0p125": {},
        }
        elements["0p5"] = {
            field: np.repeat(values, 9, axis=1)
            for field, values in elements["0p5"].items()
        }
        elements["0p125"] = {
            field: np.array(values, copy=True)
            for field, values in elements["0p25"].items()
        }
        result = _unwrapped_element_rates(times, elements)
        self.assertAlmostEqual(
            result["mercury barycenter"]["lambda"][
                "lane_slopes_rad_per_year"
            ]["0p25"],
            0.2,
        )
        self.assertEqual(
            result["mercury barycenter"]["lambda"]["fine_slope_defect"], 0.0
        )

    def test_report_contract_has_seven_named_figures(self) -> None:
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(len(manifest["figures"]), 7)
        self.assertEqual(
            set(manifest["figures"]),
            {
                "global_per_body_rms_contributions.png",
                "windowed_convergence_ratios.png",
                "mercury_rtn_defects.png",
                "venus_orientation_conditioning.png",
                "uranus_phase_orientation.png",
                "phase_stripped_state_defects.png",
                "richardson_alignment_order.png",
            },
        )


if __name__ == "__main__":
    unittest.main()
