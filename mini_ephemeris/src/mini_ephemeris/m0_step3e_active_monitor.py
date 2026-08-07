from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import shutil
import time
from typing import Sequence

from .m0_step3e_convergence import _atomic_json
from .m0_timestep_convergence import _load_json, _require
from .rebound_gr_tangent_backend_cli import output_paths


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def monitor(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Step 3e manifest")
    root = Path(manifest["paths"]["output_root"])
    operation_root = root / "operations"
    lane = manifest["new_lane"]
    paths = output_paths(Path(lane["output_dir"]), lane["id"], None)
    launch = _load_json(operation_root / "launch_record.json", "launch record")
    status = _load_json(paths["status"], "live status")
    pid = int(status["worker_pid"])
    _require(status["state"] == "RUNNING" and _pid_alive(pid), "The authenticated Step 3e child is not running.")

    with paths["progress"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) >= 101, "The live lane has not reached the 10-kyr prefix.")
    prefix_row = rows[100]
    launch_utc = dt.datetime.fromisoformat(launch["created_utc"])
    prefix_utc = dt.datetime.fromisoformat(prefix_row["wall_time_utc"])
    elapsed = (prefix_utc - launch_utc).total_seconds()
    projected = elapsed * 100.0
    state_rows_at_prefix = 101 * len(manifest["shared_configuration"]["body_names"])
    gate = manifest["operational_gate"]
    prefix_gate = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reconstructed_after_launcher_parser_correction": True,
        "reconstruction": "Prefix row wall_time_utc minus the persisted pre-Popen launch-record UTC; this is conservative because the launch record precedes child creation.",
        "worker_pid": pid,
        "sample_index": 100,
        "time_years": float(prefix_row["time_years"]),
        "progress_rows_at_prefix": 101,
        "state_rows_at_prefix": state_rows_at_prefix,
        "callback_invocations": int(prefix_row["callback_invocations"]),
        "nonfinite_result_count": int(prefix_row["nonfinite_result_count"]),
        "configuration_fingerprint": prefix_row["configuration_fingerprint"],
        "elapsed_seconds": elapsed,
        "projected_runtime_seconds": projected,
        "projected_throughput_years_per_second": 1_000_000.0 / projected,
        "maximum_elapsed_seconds": gate["prefix"]["maximum_elapsed_seconds"],
        "available_disk_bytes": shutil.disk_usage(Path(manifest["paths"]["project_root"])).free,
    }
    prefix_gate["passed"] = (
        prefix_gate["time_years"] == gate["prefix"]["minimum_completed_years"]
        and prefix_gate["callback_invocations"] == gate["prefix"]["minimum_completed_steps"]
        and prefix_gate["nonfinite_result_count"] == 0
        and prefix_gate["configuration_fingerprint"] == lane["configuration_fingerprint"]
        and prefix_gate["state_rows_at_prefix"] == gate["prefix"]["expected_state_rows_including_t0"]
        and elapsed <= gate["prefix"]["maximum_elapsed_seconds"]
        and prefix_gate["available_disk_bytes"] >= gate["disk_required_with_atomic_and_safety_allowance_bytes"]
    )
    _atomic_json(operation_root / "prefix_gate.json", prefix_gate)
    _require(prefix_gate["passed"], "The reconstructed in-lane prefix gate failed.")
    print(f"[step3e-monitor] prefix PASS at {elapsed:.6f} s; adopting live PID {pid}.", flush=True)

    last_reported_bucket = -1
    while _pid_alive(pid):
        try:
            status = _load_json(paths["status"], "live status")
            bucket = int(float(status["percent_complete"]) // 5)
            if bucket > last_reported_bucket:
                print(
                    f"[step3e-monitor] {float(status['percent_complete']):.2f}% "
                    f"({float(status['time_years']):.0f} years), "
                    f"callbacks={int(status['callback_stats']['callback_invocations'])}.",
                    flush=True,
                )
                last_reported_bucket = bucket
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(30.0)

    summary = _load_json(paths["summary"], "completed lane summary")
    _require(summary.get("complete") is True and summary.get("status") == "COMPLETED", "Adopted lane did not complete cleanly.")
    completion = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "adopted_live_pid": pid,
        "same_trajectory": True,
        "complete": True,
        "prefix_gate": prefix_gate,
    }
    _atomic_json(operation_root / "process_completion.json", completion)
    print("[step3e-monitor] authorized lane completed.", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Adopt and monitor the live Step 3e child after an operational parser correction.")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    raise SystemExit(monitor(args.manifest))


if __name__ == "__main__":
    main()
