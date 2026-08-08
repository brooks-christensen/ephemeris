from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import unittest

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = REPO_ROOT / "docs/validation/m0-step3e1-offline-state-diagnosis-v1"
OUTPUT_ROOT = REPO_ROOT / "output/stability/m0_step3e1_offline_state_diagnosis_v1"
SUMMARY_PATH = DOC_ROOT / "m0_step3e1_offline_state_diagnosis_summary.json"
REPORT_PATH = DOC_ROOT / "m0_step3e1_offline_state_diagnosis_report.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


class Step3e1ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = _load_strict_json(SUMMARY_PATH)

    def test_status_and_report_are_consistent(self) -> None:
        self.assertEqual(
            self.summary["final_status"], "STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE"
        )
        self.assertEqual(
            self.summary["primary_classification"], "TRUE_NONPHASE_NONCONVERGENCE"
        )
        self.assertFalse(self.summary["trajectory_or_ias15_executed"])
        self.assertFalse(self.summary["production_timestep_validated"])
        self.assertFalse(self.summary["stage4_authorized"])

        report = REPORT_PATH.read_text(encoding="utf-8")
        required_text = (
            "STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE",
            "TRUE_NONPHASE_NONCONVERGENCE",
            "does not retroactively validate 0.25 day",
            "0.551548145246",
            "1.07191551941",
            "8.30345456837e-08",
            "0.854837076986",
            "Global position-component RMS",
            "### Cumulative ratios",
            "Mercury RTN fine/coarse ratios",
            "Full-history Richardson alignment",
        )
        cumulative = self.summary["physical_state"]["cumulative_key_ratios"]
        self.assertEqual(set(cumulative), {
            "full system",
            "mercury barycenter",
            "venus barycenter",
            "uranus barycenter",
        })
        for rows in cumulative.values():
            self.assertEqual(len(rows), 10)
            self.assertEqual(
                [row["endpoint_years"] for row in rows],
                list(range(100_000, 1_000_001, 100_000)),
            )
        for text in required_text:
            with self.subTest(text=text):
                self.assertIn(text, report)

    def test_table_inventory_hashes_rows_keys_and_finite_values(self) -> None:
        expected_keys = {
            "epoch_body_metrics.csv": ("time_years", "body"),
            "window_metrics.csv": ("endpoint_years", "entity"),
            "cumulative_metrics.csv": ("endpoint_years", "entity"),
            "phase_stripped_window_metrics.csv": (
                "method",
                "endpoint_years",
                "entity",
            ),
            "orbital_window_metrics.csv": ("endpoint_years", "body"),
        }
        text_fields = {
            "body",
            "entity",
            "kind",
            "method",
            "order_status",
            "nonphase_order_status",
        }
        inventory = {
            Path(entry["path"]).name: entry
            for entry in self.summary["derived_outputs"]["tables_and_audit"]
        }
        expected_rows = self.summary["derived_outputs"]["row_counts"]

        for filename, key_fields in expected_keys.items():
            with self.subTest(filename=filename):
                path = OUTPUT_ROOT / filename
                entry = inventory[filename]
                self.assertEqual(path.stat().st_size, entry["size_bytes"])
                self.assertEqual(_sha256(path), entry["sha256"])

                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected_rows[filename])
                keys = {tuple(row[field] for field in key_fields) for row in rows}
                self.assertEqual(len(keys), len(rows))

                for row in rows:
                    for field, value in row.items():
                        if field in text_fields or value == "":
                            continue
                        self.assertTrue(math.isfinite(float(value)), (filename, field, value))

        audit_path = OUTPUT_ROOT / "offline_audit.json"
        audit_entry = inventory[audit_path.name]
        self.assertEqual(_sha256(audit_path), audit_entry["sha256"])
        self.assertEqual(audit_path.stat().st_size, audit_entry["size_bytes"])
        _load_strict_json(audit_path)

    def test_figure_inventory_hashes_and_pixels(self) -> None:
        figures = self.summary["derived_outputs"]["figures"]
        self.assertEqual(len(figures), 7)
        for entry in figures:
            filename = Path(entry["path"]).name
            with self.subTest(filename=filename):
                path = DOC_ROOT / "figures" / filename
                self.assertEqual(path.stat().st_size, entry["size_bytes"])
                self.assertEqual(_sha256(path), entry["sha256"])
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    self.assertGreaterEqual(image.width, 600)
                    self.assertGreaterEqual(image.height, 400)
                    extrema = image.convert("RGB").getextrema()
                self.assertTrue(any(low < high for low, high in extrema), extrema)


if __name__ == "__main__":
    unittest.main()
