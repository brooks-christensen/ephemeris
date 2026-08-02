from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .progress import process_tree_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit process-tree CPU/RSS monitoring with a CPU-using child.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="monitor_process_tree_audit")
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "monitor_process_tree_audit.json"

    code = (
        "import subprocess, sys, time, os\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "\"import time; end=time.time()+4; x=0\\nwhile time.time()<end:\\n x+=1\"])\n"
        "print('child_pid', child.pid, flush=True)\n"
        "time.sleep(4.5)\n"
        "child.wait()\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    samples = []
    try:
        start = time.monotonic()
        while process.poll() is None:
            metrics = process_tree_metrics(process.pid)
            sample = {
                "runner_pid": os.getpid(),
                "direct_child_pid": metrics.direct_child_pid,
                "selected_integration_worker_pid": metrics.worker_pid,
                "descendant_pids": metrics.descendant_pids or [],
                "aggregate_cpu_percent": metrics.cpu_percent,
                "aggregate_rss_bytes": metrics.rss_bytes,
                "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_time_monotonic_seconds": time.monotonic(),
            }
            samples.append(sample)
            time.sleep(0.25)
            if time.monotonic() - start > 10:
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    cpu_nonzero = any((sample.get("aggregate_cpu_percent") or 0.0) > 0.0 for sample in samples)
    rss_present = any((sample.get("aggregate_rss_bytes") or 0) > 0 for sample in samples)
    descendants_present = any(sample.get("descendant_pids") for sample in samples)
    payload = {
        "passed": bool(cpu_nonzero and rss_present and descendants_present),
        "sample_count": len(samples),
        "cpu_nonzero": cpu_nonzero,
        "rss_present": rss_present,
        "descendants_present": descendants_present,
        "samples": samples,
        "requirements": [
            "actual child/descendant process is detected",
            "aggregate CPU is nonzero while worker is active",
            "aggregate RSS is recorded",
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[monitor-audit] wrote {output_path}")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
