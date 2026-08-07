from __future__ import annotations

from decimal import Decimal, localcontext
import json
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.m0_energy_precision_diagnosis import (
    LaneResult,
    _artifact_inventory,
    _canonical_float_token,
    classify_diagnosis,
    compensated_energy,
    decimal_energy,
    decimal_statistics,
    float64_energy,
)


def decimal_row(
    mass: str,
    x: str,
    y: str = "0",
    z: str = "0",
    vx: str = "0",
    vy: str = "0",
    vz: str = "0",
) -> dict[str, str]:
    return {
        "mass_kg": mass,
        "x_m": x,
        "y_m": y,
        "z_m": z,
        "vx_m_per_s": vx,
        "vy_m_per_s": vy,
        "vz_m_per_s": vz,
    }


class M0EnergyPrecisionDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[2]
        cls.manifest = json.loads(
            (
                root
                / "ephemeris_experiment_runner/manifests/12_m0_energy_precision_diagnosis_v1.json"
            ).read_text()
        )

    def test_two_body_energy_formulas_agree_analytically(self) -> None:
        masses = np.asarray([2.0, 3.0])
        positions = np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        velocities = np.zeros((2, 3))
        expected_pair = -1.5
        expected_gr = -0.5625
        expected_corrected = expected_pair + expected_gr
        float_result = float64_energy(
            masses,
            positions,
            velocities,
            gravitational_constant=1.0,
            speed_of_light=2.0,
            coefficient_scale=1.0,
        )
        compensated_result = compensated_energy(
            masses,
            positions,
            velocities,
            gravitational_constant=1.0,
            speed_of_light=2.0,
            coefficient_scale=1.0,
        )
        with localcontext() as context:
            context.prec = 60
            decimal_result = decimal_energy(
                [decimal_row("2", "0"), decimal_row("3", "4")],
                gravitational_constant=Decimal(1),
                speed_of_light=Decimal(2),
                coefficient_scale=Decimal(1),
            )
        for result in (float_result, compensated_result):
            self.assertEqual(result["kinetic"], 0.0)
            self.assertEqual(result["pair_potential"], expected_pair)
            self.assertEqual(result["gr_potential"], expected_gr)
            self.assertEqual(result["corrected"], expected_corrected)
        self.assertEqual(decimal_result["pair_potential"], Decimal("-1.5"))
        self.assertEqual(decimal_result["gr_potential"], Decimal("-0.5625"))
        self.assertEqual(decimal_result["corrected"], Decimal("-2.0625"))

    def test_decimal_reference_never_rounds_through_float(self) -> None:
        token = "9007199254740993"
        self.assertNotEqual(Decimal(token), Decimal.from_float(float(token)))
        self.assertTrue(_canonical_float_token("1.2345678901234567"))
        self.assertFalse(_canonical_float_token("1.2300"))

    def test_historical_artifact_inventory_formats_are_flattened(self) -> None:
        first = {"path": "one", "sha256": "1", "size_bytes": 1}
        second = {"path": "two", "sha256": "2", "size_bytes": 2}
        self.assertEqual(_artifact_inventory({"artifact_inventory": [first]}, "list"), [first])
        self.assertEqual(
            _artifact_inventory(
                {"artifact_inventory": {"new": [first], "reused": [second]}},
                "groups",
            ),
            [first, second],
        )

    def test_decimal_statistics_use_registered_percentile_and_centered_fit(self) -> None:
        times = [Decimal(index) for index in range(10001)]
        values = [Decimal(index) * Decimal("1e-12") for index in range(10001)]
        with localcontext() as context:
            context.prec = 60
            result = decimal_statistics(times, values)
        self.assertEqual(result["max_abs"], Decimal("1.0000e-8"))
        self.assertEqual(result["max_abs_worst_epoch_years"], Decimal(10000))
        self.assertEqual(result["p99_abs"], Decimal("9.900e-9"))
        self.assertEqual(result["fitted_trend_per_year"], Decimal("1e-12"))
        self.assertEqual(result["fitted_change_over_1myr"], Decimal("1e-6"))

    def _lane_result(
        self,
        lane_id: str,
        *,
        maximum: str,
        rms: str,
        p99: str,
        trend: str,
        telemetry_passed: bool = True,
        agreement_passed: bool = True,
    ) -> LaneResult:
        decimal_stats = {
            "max_abs": Decimal(maximum),
            "rms": Decimal(rms),
            "p99_abs": Decimal(p99),
            "fitted_change_over_1myr": Decimal(trend),
        }
        compensated = {
            key: float(value) for key, value in decimal_stats.items()
        }
        return LaneResult(
            lane_id=lane_id,
            summary={
                "telemetry_reproduction": {"passed": telemetry_passed},
                "method_agreement": {"passed": agreement_passed},
            },
            statistics_internal={
                "decimal": decimal_stats,
                "compensated": compensated,
            },
            timeseries_inventory={},
        )

    def _classify(
        self,
        coarse: LaneResult,
        fine: LaneResult,
    ) -> str:
        lane_ids = [lane["id"] for lane in self.manifest["input_lanes"]]
        diagnosis, _ = classify_diagnosis(
            self.manifest,
            {lane_ids[0]: coarse, lane_ids[1]: fine},
        )
        return diagnosis

    def test_classification_rules_are_mutually_decisive(self) -> None:
        coarse_id, fine_id = [lane["id"] for lane in self.manifest["input_lanes"]]
        coarse = self._lane_result(
            coarse_id,
            maximum="1e-10",
            rms="8e-11",
            p99="9e-11",
            trend="5e-11",
        )
        confirmed = self._lane_result(
            fine_id,
            maximum="3e-10",
            rms="2.8e-10",
            p99="2.9e-10",
            trend="3e-10",
        )
        self.assertEqual(self._classify(coarse, confirmed), "ENERGY_DRIFT_CONFIRMED")

        precision_coarse = self._lane_result(
            coarse_id,
            maximum="3e-10",
            rms="2e-10",
            p99="2.5e-10",
            trend="5e-11",
        )
        precision_fine = self._lane_result(
            fine_id,
            maximum="1e-10",
            rms="8e-11",
            p99="9e-11",
            trend="4e-11",
        )
        self.assertEqual(
            self._classify(precision_coarse, precision_fine),
            "ENERGY_TELEMETRY_PRECISION_LIMITED",
        )

        inconclusive = self._lane_result(
            fine_id,
            maximum="3e-10",
            rms="2.8e-10",
            p99="2.9e-10",
            trend="3e-10",
            agreement_passed=False,
        )
        self.assertEqual(self._classify(coarse, inconclusive), "INCONCLUSIVE")

        blocked = self._lane_result(
            fine_id,
            maximum="3e-10",
            rms="2.8e-10",
            p99="2.9e-10",
            trend="3e-10",
            telemetry_passed=False,
        )
        self.assertEqual(self._classify(coarse, blocked), "BLOCKED")

    def test_manifest_preserves_historical_thresholds_and_input_hashes(self) -> None:
        self.assertEqual(
            self.manifest["statistics"]["unchanged_trend_rule"],
            "abs(fitted_change_over_1myr) <= max(0.25*max_abs,1e-10) for every compared lane.",
        )
        self.assertEqual(
            self.manifest["numeric_tolerances"]["operation_count_budget"], 1024
        )
        self.assertEqual(
            self.manifest["authoritative_constants"]["decimal_precision_digits"], 60
        )
        self.assertEqual(len(self.manifest["input_lanes"]), 2)
        self.assertTrue(
            all(len(lane["state_sha256"]) == 64 for lane in self.manifest["input_lanes"])
        )


if __name__ == "__main__":
    unittest.main()
