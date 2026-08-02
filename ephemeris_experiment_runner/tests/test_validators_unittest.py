from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ephemeris_experiments.validators import csv_integrity


class CsvIntegrityAliasTests(unittest.TestCase):
    def test_accepts_legacy_energy_column_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.csv"
            path.write_text("time_years,energy_rel_drift\n0,1e-9\n5,2e-9\n10,3e-9\n")
            result = csv_integrity({
                "kind": "csv_integrity",
                "pattern": str(path),
                "target_years": 10,
                "finite_columns": ["newtonian_energy_component_rel_change"],
                "finite_column_aliases": {
                    "newtonian_energy_component_rel_change": ["energy_rel_drift"],
                },
            })
            self.assertTrue(result.passed, result.detail)


if __name__ == "__main__":
    unittest.main()
