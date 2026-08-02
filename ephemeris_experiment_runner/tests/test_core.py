from __future__ import annotations

import json
from pathlib import Path

from ephemeris_experiments.manifest import load_manifest
from ephemeris_experiments.progress import ProgressSample, estimate_rate_and_eta
from ephemeris_experiments.validators import csv_integrity


def test_manifest_loads(tmp_path: Path):
    manifest = {
        "experiment_id": "x",
        "title": "X",
        "description": "test",
        "variables": {"ROOT": str(tmp_path)},
        "state_dir": "${ROOT}/state",
        "stages": [
            {
                "id": "s",
                "command": ["python", "-V"],
                "cwd": "${ROOT}",
                "output_dir": "${ROOT}/out"
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = load_manifest(path)
    assert loaded.stages[0].cwd == str(tmp_path)


def test_eta_uses_recent_rate():
    samples = [
        ProgressSample(0.0, 0.0, "x", 0.0),
        ProgressSample(10.0, 100.0, "x", 10.0),
        ProgressSample(20.0, 200.0, "x", 20.0),
    ]
    rate, eta = estimate_rate_and_eta(samples, 500.0)
    assert rate == 10.0
    assert eta == 30.0


def test_csv_integrity(tmp_path: Path):
    path = tmp_path / "progress.csv"
    path.write_text("time_years,value\n0,1\n5,2\n10,3\n")
    result = csv_integrity({
        "kind": "csv_integrity",
        "pattern": str(path),
        "target_years": 10,
        "finite_columns": ["value"]
    })
    assert result.passed, result.detail
