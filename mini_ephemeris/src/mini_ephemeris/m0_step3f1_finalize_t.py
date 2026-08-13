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


def finalize(manifest_path: Path) -> dict[str, object]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    audit_payload = audit(manifest_path)
    lane = manifest["lane_contracts"]["T"]
    paths = lane_paths(manifest, "T")
    status = load_json(paths["status"], "Lane T failed closeout status")
    require(status == {"failure": "name 'expected_callbacks' is not defined", "lane": "T", "manifest_sha256": sha256_file(manifest_path), "schema_version": 1, "state": "FAILED"}, "Lane T is not the exact recoverable closeout failure.")
    with paths["progress"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        progress = list(reader)
        require(list(reader.fieldnames or ()) == PROGRESS_FIELDS, "Lane T progress schema changed.")
    with paths["state"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        state_rows = list(reader)
        require(list(reader.fieldnames or ()) == STATE_FIELDS, "Lane T state schema changed.")
    require(len(progress) == 101 and len(state_rows) == 1010, "Lane T raw trajectory is incomplete.")
    require(int(progress[-1]["steps_done"]) == 14610000 and float(progress[-1]["time_years"]) == 10000.0, "Lane T endpoint is incomplete.")
    require(int(progress[-1]["callback_invocations"]) == lane["expected_callback_invocations"], "Lane T callback accounting changed.")
    require(all(int(row["nonfinite_result_count"]) == 0 for row in progress), "Lane T contains a nonfinite callback result.")
    require(all(row["megno"] != "" and row["lcn_1_per_year"] != "" for row in progress), "Lane T chaos telemetry is incomplete.")
    require(all(row["variation_config_index"] == "0" for row in state_rows), "Lane T variation identity changed.")
    archive = rebound.Simulationarchive(str(paths["archive"]))
    require(len(archive) == 11, "Lane T archive count changed.")
    require([float(archive[index].t) / JULIAN_YEAR_S for index in range(11)] == [1000.0 * index for index in range(11)], "Lane T archive times changed.")
    event_lines = paths["events"].read_text(encoding="utf-8").splitlines()
    require(len(event_lines) == 2 and "START lane=T" in event_lines[0] and "FAILED lane=T error=name 'expected_callbacks' is not defined" in event_lines[1], "Lane T event history changed.")
    started = dt.datetime.fromisoformat(event_lines[0].split(" ", 1)[0])
    failed = dt.datetime.fromisoformat(event_lines[1].split(" ", 1)[0])
    runtime = (failed - started).total_seconds()
    zero_step = load_json(Path(manifest["paths"]["output_root"]) / "zero_step_audit.json", "zero-step audit")
    atomic_write_json(paths["status"], {
        "schema_version": 1, "state": "COMPLETED", "lane": "T", "lane_id": lane["id"],
        "configuration_fingerprint": lane["configuration_fingerprint"], "manifest_sha256": sha256_file(manifest_path),
        "samples": 101, "state_rows": 1010, "steps": 14610000, "time_years": 10000.0,
        "callback_invocations": lane["expected_callback_invocations"], "nonfinite_result_count": 0,
        "closeout_recovered_without_integration": True,
    })
    _event(paths["events"], "RECOVERED_COMPLETE lane=T integration_assertions_passed=true rerun=false")
    inventory = {name: {"path": str(paths[name]), "size_bytes": paths[name].stat().st_size, "sha256": sha256_file(paths[name])} for name in ("progress", "state", "archive", "status", "events")}
    final_archive = archive[-1]
    summary: dict[str, object] = {
        "schema_version": 1, "status": "COMPLETED", "lane": "T", "lane_id": lane["id"],
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "configuration": lane_payload(manifest, "T"), "configuration_fingerprint": lane["configuration_fingerprint"],
        "construction": zero_step["lanes"]["T"], "final_settings": _settings(final_archive),
        "runtime_seconds": runtime, "throughput_years_per_wall_second": 10000.0 / runtime,
        "scientific_samples": 101, "state_rows": 1010, "steps": 14610000,
        "archive_snapshots": 11, "archive_times_years": [1000.0 * index for index in range(11)],
        "callback_stats": {"callback_invocations": lane["expected_callback_invocations"], "nonfinite_result_count": 0},
        "hot_path": zero_step["lanes"]["T"]["hot_path"], "command": sys.argv,
        "provenance": {"closeout_recovered_without_integration": True, "audit_runtime": audit_payload["runtime"]},
        "artifact_inventory": inventory,
    }
    atomic_write_json(paths["summary"], summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.manifest.resolve()), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
