from __future__ import annotations

import json
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .models import ExperimentSpec, StageSpec
from .progress import (
    estimate_rate_and_eta,
    human_bytes,
    human_duration,
    load_progress_history,
    process_metrics,
    process_tree_metrics,
    sample_progress,
    write_progress_history,
)
from .state import load_approvals, load_state, save_state
from .validators import run_gates


class RunnerError(RuntimeError):
    pass


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _print_stage_header(index: int, total: int, stage: StageSpec, command: list[str]) -> None:
    print("\n" + "=" * 88, flush=True)
    print(f"STAGE {index}/{total}: {stage.title}", flush=True)
    print(f"id: {stage.stage_id}", flush=True)
    print(f"objective: {stage.objective}", flush=True)
    if stage.target_years is not None:
        print(f"target simulated duration: {stage.target_years:,.0f} Julian years", flush=True)
    print(f"working directory: {stage.cwd}", flush=True)
    print(f"output directory: {stage.output_dir}", flush=True)
    print("command:", flush=True)
    print("  " + shlex.join(command), flush=True)
    print("=" * 88, flush=True)


def _tee_output(pipe: Any, log_handle: Any, output_queue: queue.Queue[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            output_queue.put(line)
            print(line, end="", flush=True)
    finally:
        pipe.close()


def _can_resume(stage: StageSpec) -> bool:
    if not stage.resume_args or not stage.resume_probe_globs:
        return False
    import glob

    return any(glob.glob(pattern) for pattern in stage.resume_probe_globs)


def _stage_state(state: dict[str, Any], stage_id: str) -> dict[str, Any]:
    return state.setdefault("stages", {}).setdefault(stage_id, {})


def run_experiment(spec: ExperimentSpec, resume: bool = False, dry_run: bool = False) -> int:
    state_dir = Path(spec.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    approvals_path = state_dir / "approvals.json"
    lock_path = state_dir / "runner.lock"

    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RunnerError(
            f"Runner lock already exists: {lock_path}. Remove it only after confirming no ephem-exp process is active."
        ) from exc
    os.write(lock_fd, f"pid={os.getpid()} started={_utc_now()}\n".encode())
    os.close(lock_fd)

    state = load_state(state_path)
    approvals = load_approvals(approvals_path)
    try:
        print(f"EXPERIMENT: {spec.title}", flush=True)
        print(spec.description, flush=True)
        print(f"manifest: {spec.source_path}", flush=True)
        print(f"state directory: {state_dir}", flush=True)
        print("Safety policy: stages run serially; any nonzero exit or failed gate blocks all downstream stages.", flush=True)

        passed = {sid for sid, data in state.get("stages", {}).items() if data.get("status") == "PASSED"}
        total = len(spec.stages)
        for index, stage in enumerate(spec.stages, start=1):
            info = _stage_state(state, stage.stage_id)
            if info.get("status") == "PASSED" and resume:
                print(f"[skip] {stage.stage_id} already passed", flush=True)
                passed.add(stage.stage_id)
                continue
            unmet = [dep for dep in stage.depends_on if dep not in passed]
            if unmet:
                info.update(status="BLOCKED", blocked_by=unmet, updated_utc=_utc_now())
                save_state(state_path, state)
                print(f"[blocked] {stage.stage_id}: unmet dependencies {unmet}", flush=True)
                return 2
            if stage.approval_required_before and stage.stage_id not in approvals:
                info.update(status="AWAITING_APPROVAL", updated_utc=_utc_now())
                save_state(state_path, state)
                print(
                    f"[approval required] Review prior outputs, then run:\n"
                    f"  ephem-exp approve {shlex.quote(str(spec.source_path))} {shlex.quote(stage.stage_id)}\n"
                    f"Then resume with:\n"
                    f"  ephem-exp run {shlex.quote(str(spec.source_path))} --resume",
                    flush=True,
                )
                return 3

            command = list(stage.command)
            resumed = False
            if resume and _can_resume(stage):
                command.extend(stage.resume_args)
                resumed = True
            _print_stage_header(index, total, stage, command)
            if dry_run:
                passed.add(stage.stage_id)
                continue

            output_dir = Path(stage.output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            stage_state_dir = state_dir / stage.stage_id
            stage_state_dir.mkdir(parents=True, exist_ok=True)
            history_path = stage_state_dir / "progress_history.jsonl"
            samples = load_progress_history(history_path)
            log_path = stage_state_dir / f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
            env = dict(os.environ)
            env.update(stage.env)
            env.setdefault("PYTHONUNBUFFERED", "1")

            info.update(
                status="RUNNING",
                started_utc=_utc_now(),
                command=command,
                cwd=stage.cwd,
                output_dir=str(output_dir),
                log_path=str(log_path),
                resumed=resumed,
                pid=None,
            )
            save_state(state_path, state)

            with log_path.open("a", buffering=1) as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=stage.cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                info["pid"] = process.pid
                save_state(state_path, state)
                output_queue: queue.Queue[str] = queue.Queue()
                tee = threading.Thread(
                    target=_tee_output,
                    args=(process.stdout, log_handle, output_queue),
                    daemon=True,
                )
                tee.start()
                started_wall = time.time()
                next_update = started_wall
                last_sim_years: float | None = samples[-1].sim_years if samples else None
                last_progress_wall = samples[-1].wall_time if samples else started_wall
                warned_stall = False

                try:
                    while process.poll() is None:
                        now = time.time()
                        if now >= next_update:
                            sample = sample_progress(stage)
                            if sample is not None:
                                if last_sim_years is None or sample.sim_years > last_sim_years:
                                    last_progress_wall = now
                                    warned_stall = False
                                last_sim_years = sample.sim_years
                                samples.append(sample)
                                write_progress_history(history_path, sample)
                            rate, eta = estimate_rate_and_eta(samples, stage.target_years)
                            metrics = process_tree_metrics(process.pid)
                            if sample is not None and stage.target_years:
                                pct = max(0.0, min(100.0, 100.0 * sample.sim_years / stage.target_years))
                                rate_myr_hour = rate * 3600 / 1e6 if rate else None
                                print(
                                    f"[progress] {stage.stage_id}: {pct:6.2f}% | "
                                    f"{sample.sim_years/1e6:,.3f}/{stage.target_years/1e6:,.3f} Myr | "
                                    f"elapsed {human_duration(now-started_wall)} | "
                                    f"rate {rate_myr_hour:.3f} Myr/h | " if rate_myr_hour is not None else
                                    f"[progress] {stage.stage_id}: {pct:6.2f}% | "
                                    f"{sample.sim_years/1e6:,.3f}/{stage.target_years/1e6:,.3f} Myr | "
                                    f"elapsed {human_duration(now-started_wall)} | rate warming up | ",
                                    end="",
                                    flush=True,
                                )
                                print(
                                    f"ETA {human_duration(eta)} | CPU {metrics.cpu_percent if metrics.cpu_percent is not None else 'unknown'}% | "
                                    f"RSS {human_bytes(metrics.rss_bytes)} | source {Path(sample.source).name}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[progress] {stage.stage_id}: waiting for first checkpoint/output | "
                                    f"elapsed {human_duration(now-started_wall)} | CPU {metrics.cpu_percent if metrics.cpu_percent is not None else 'unknown'}% | "
                                    f"RSS {human_bytes(metrics.rss_bytes)}",
                                    flush=True,
                                )
                            if now - last_progress_wall > stage.stall_warning_seconds and not warned_stall:
                                print(
                                    f"[warning] No simulated-time advance detected for {human_duration(now-last_progress_wall)}. "
                                    "The process is not killed automatically; inspect CPU and output files before intervening.",
                                    flush=True,
                                )
                                warned_stall = True
                            info.update(
                                updated_utc=_utc_now(),
                                latest_sim_years=sample.sim_years if sample else last_sim_years,
                                percent=(100.0 * sample.sim_years / stage.target_years) if sample and stage.target_years else None,
                                eta_seconds=eta,
                                cpu_percent=metrics.cpu_percent,
                                rss_bytes=metrics.rss_bytes,
                                runner_pid=metrics.runner_pid,
                                direct_child_pid=metrics.direct_child_pid,
                                selected_worker_pid=metrics.worker_pid,
                                descendant_pids=metrics.descendant_pids,
                            )
                            save_state(state_path, state)
                            next_update = now + stage.status_interval_seconds
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n[interrupt] Forwarding SIGINT to the stage process...", flush=True)
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=30)
                    raise
                finally:
                    tee.join(timeout=10)

                return_code = process.returncode

            info.update(return_code=return_code, finished_utc=_utc_now(), updated_utc=_utc_now())
            if return_code != 0:
                info["status"] = "FAILED"
                info["failure_reason"] = f"process exit code {return_code}"
                save_state(state_path, state)
                print(f"[failed] {stage.stage_id}: exit code {return_code}. Downstream stages will not run.", flush=True)
                return 1

            results = run_gates(stage.gates, stage.cwd)
            info["gate_results"] = [result.__dict__ for result in results]
            for result in results:
                print(f"[gate {'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}", flush=True)
            if any(not result.passed for result in results):
                info["status"] = "FAILED"
                info["failure_reason"] = "post-run gate failure"
                save_state(state_path, state)
                print(f"[failed] {stage.stage_id}: validation gate failed. Downstream stages blocked.", flush=True)
                return 1

            info["status"] = "PASSED"
            save_state(state_path, state)
            passed.add(stage.stage_id)
            print(f"[passed] {stage.stage_id}", flush=True)

        print("\nAll enabled stages completed and passed their gates.", flush=True)
        return 0
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
