from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Sequence

import rebound

from .m0_step3f1_contract import DEFAULT_MANIFEST, JULIAN_YEAR_S, PROGRESS_FIELDS, STATE_FIELDS, lane_paths, lane_payload, load_json, require, sha256_file, validate_manifest
from .m0_step3f1_runner import _event, _settings, audit
from .rebound_gr_tangent_backend_cli import atomic_write_json


def recover(manifest_path: Path) -> dict[str, object]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    audit_payload = audit(manifest_path)
    lane = manifest["lane_contracts"]["P"]
    paths = lane_paths(manifest, "P")
    status = load_json(paths["status"], "Lane P failed status")
    require(status == {"failure": "Callback accounting mismatch.", "lane": "P", "manifest_sha256": sha256_file(manifest_path), "schema_version": 1, "state": "FAILED"}, "Lane P is not the exact recoverable callback-accounting failure.")
    with paths["progress"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        progress = list(reader)
        require(list(reader.fieldnames or ()) == PROGRESS_FIELDS, "Lane P progress schema changed.")
    with paths["state"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        state_rows = list(reader)
        require(list(reader.fieldnames or ()) == STATE_FIELDS, "Lane P state schema changed.")
    require(len(progress) == 101 and len(state_rows) == 1010, "Lane P raw trajectory is incomplete.")
    require(int(progress[-1]["steps_done"]) == 14610000 and float(progress[-1]["time_years"]) == 10000.0, "Lane P endpoint is incomplete.")
    require(all(int(row["nonfinite_result_count"]) == 0 for row in progress), "Lane P contains a nonfinite callback result.")
    observed = int(progress[-1]["callback_invocations"])
    expected = lane["expected_callback_invocations"]
    reconciled = 2 * 14610000 + 32 + 64 * 100
    require(observed == reconciled and observed - expected == 3200, "Lane P callback mismatch is not the exact diagnostic-copy pattern.")
    archive = rebound.Simulationarchive(str(paths["archive"]))
    require(len(archive) == 11, "Lane P archive count changed.")
    require([float(archive[index].t) / JULIAN_YEAR_S for index in range(11)] == [1000.0 * index for index in range(11)], "Lane P archive times changed.")
    event_lines = paths["events"].read_text(encoding="utf-8").splitlines()
    require(len(event_lines) == 2 and "START lane=P" in event_lines[0] and "FAILED lane=P error=Callback accounting mismatch." in event_lines[1], "Lane P event history changed.")
    started = dt.datetime.fromisoformat(event_lines[0].split(" ", 1)[0])
    failed = dt.datetime.fromisoformat(event_lines[1].split(" ", 1)[0])
    runtime = (failed - started).total_seconds()
    zero_step = load_json(Path(manifest["paths"]["output_root"]) / "zero_step_audit.json", "zero-step audit")
    atomic_write_json(paths["status"], {
        "schema_version": 1, "state": "RECOVERED_COMPLETE_WITH_CALLBACK_ACCOUNTING_MISMATCH",
        "lane": "P", "lane_id": lane["id"], "configuration_fingerprint": lane["configuration_fingerprint"],
        "manifest_sha256": sha256_file(manifest_path), "samples": 101, "state_rows": 1010,
        "steps": 14610000, "time_years": 10000.0, "callback_invocations": observed,
        "expected_callback_invocations": expected, "nonfinite_result_count": 0,
    })
    _event(paths["events"], f"RECOVERED_COMPLETE lane=P observed_callbacks={observed} preregistered_callbacks={expected} integrity_pass=false")
    inventory = {name: {"path": str(paths[name]), "size_bytes": paths[name].stat().st_size, "sha256": sha256_file(paths[name])} for name in ("progress", "state", "archive", "status", "events")}
    final_archive = archive[-1]
    summary: dict[str, object] = {
        "schema_version": 1, "status": "RECOVERED_COMPLETE_WITH_CALLBACK_ACCOUNTING_MISMATCH",
        "lane": "P", "lane_id": lane["id"], "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "configuration": lane_payload(manifest, "P"),
        "configuration_fingerprint": lane["configuration_fingerprint"], "construction": zero_step["lanes"]["P"],
        "final_settings": _settings(final_archive), "runtime_seconds": runtime,
        "throughput_years_per_wall_second": 10000.0 / runtime, "scientific_samples": 101,
        "state_rows": 1010, "steps": 14610000, "archive_snapshots": 11,
        "archive_times_years": [1000.0 * index for index in range(11)],
        "callback_stats": {"callback_invocations": observed, "nonfinite_result_count": 0},
        "hot_path": zero_step["lanes"]["P"]["hot_path"], "command": sys.argv,
        "preregistered_callback_accounting": {
            "passed": False, "expected": expected, "observed": observed, "difference": observed - expected,
            "observed_formula": "2*14,610,000 physical lazy-kernel evaluations + 32 first-map corrector evaluations + 64*100 positive-time output/copy corrector evaluations",
            "interpretation": "The complete finite trajectory is retained, but the preregistered callback-accounting integrity gate remains failed.",
        },
        "provenance": {"recovered_without_integration": True, "audit_runtime": audit_payload["runtime"]},
        "artifact_inventory": inventory,
    }
    atomic_write_json(paths["summary"], summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(json.dumps(recover(args.manifest.resolve()), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
