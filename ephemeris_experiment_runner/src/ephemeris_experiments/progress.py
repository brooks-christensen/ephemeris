from __future__ import annotations

import csv
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import ProgressSource, StageSpec

JULIAN_YEAR_SECONDS = 365.25 * 24 * 3600


@dataclass
class ProgressSample:
    wall_time: float
    sim_years: float
    source: str
    source_mtime: float


@dataclass
class ProcessMetrics:
    cpu_percent: float | None
    rss_bytes: int | None
    elapsed: str | None
    runner_pid: int | None = None
    direct_child_pid: int | None = None
    worker_pid: int | None = None
    descendant_pids: list[int] | None = None


def _newest_match(patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(pattern))
    files = [p for p in matches if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _read_last_csv_time(path: Path, columns: tuple[str, ...]) -> float | None:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        chosen = next((name for name in columns if name in (reader.fieldnames or [])), None)
        if chosen is None:
            return None
        last: float | None = None
        for row in reader:
            raw = row.get(chosen)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                last = value
        return last


def _read_archive_time(path: Path, target_years: float | None, unit: str) -> float | None:
    try:
        import rebound  # type: ignore
    except ImportError:
        return None

    with tempfile.TemporaryDirectory(prefix="ephem_archive_inspect_") as tmp_dir:
        copied = Path(tmp_dir) / path.name
        try:
            shutil.copy2(path, copied)
            archive = rebound.Simulationarchive(str(copied))
            if len(archive) == 0:
                return None
            raw_t = float(archive[-1].t)
        except Exception:
            return None

    if unit == "seconds":
        return raw_t / JULIAN_YEAR_SECONDS
    if unit == "years":
        return raw_t
    if target_years and raw_t > max(target_years * 100.0, 1e12):
        return raw_t / JULIAN_YEAR_SECONDS
    return raw_t


def sample_progress(stage: StageSpec) -> ProgressSample | None:
    for source in stage.progress_sources:
        path = _newest_match(source.path_globs)
        if path is None:
            continue
        sim_years: float | None
        if source.kind == "archive":
            sim_years = _read_archive_time(path, stage.target_years, source.archive_time_unit)
        elif source.kind == "csv":
            sim_years = _read_last_csv_time(path, source.time_columns)
        else:
            continue
        if sim_years is not None and math.isfinite(sim_years):
            return ProgressSample(
                wall_time=time.time(),
                sim_years=sim_years,
                source=str(path),
                source_mtime=path.stat().st_mtime,
            )
    return None


def estimate_rate_and_eta(
    samples: list[ProgressSample], target_years: float | None
) -> tuple[float | None, float | None]:
    if len(samples) < 2:
        return None, None
    recent = samples[-12:]
    rates: list[float] = []
    for left, right in zip(recent, recent[1:]):
        dt = right.wall_time - left.wall_time
        dy = right.sim_years - left.sim_years
        if dt > 0 and dy > 0:
            rates.append(dy / dt)
    if not rates:
        return None, None
    rate = statistics.median(rates)
    if rate <= 0 or target_years is None:
        return rate, None
    remaining = max(0.0, target_years - recent[-1].sim_years)
    return rate, remaining / rate


def _process_table() -> dict[int, tuple[int, str]]:
    table: dict[int, tuple[int, str]] = {}
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            stat = (proc_path / "stat").read_text()
        except Exception:
            continue
        close = stat.rfind(")")
        if close < 0:
            continue
        try:
            pid = int(stat.split(" ", 1)[0])
            comm = stat[stat.find("(") + 1 : close]
            rest = stat[close + 2 :].split()
            ppid = int(rest[1])
        except Exception:
            continue
        table[pid] = (ppid, comm)
    return table


def descendant_pids(pid: int) -> list[int]:
    table = _process_table()
    children: dict[int, list[int]] = {}
    for child_pid, (ppid, _comm) in table.items():
        children.setdefault(ppid, []).append(child_pid)
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop(0)
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def process_tree_metrics(pid: int) -> ProcessMetrics:
    descendants = descendant_pids(pid)
    all_pids = [pid, *descendants]
    if not all_pids:
        return ProcessMetrics(None, None, None, pid, None, None, [])
    try:
        output = subprocess.check_output(
            ["ps", "-o", "pid=,ppid=,%cpu=,rss=,etime=,comm=", "-p", ",".join(str(p) for p in all_pids)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ProcessMetrics(None, None, None, pid, descendants[0] if descendants else None, None, descendants)
    cpu_total = 0.0
    rss_total = 0
    elapsed = None
    rows = []
    for line in output.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 5:
            continue
        try:
            row_pid = int(parts[0])
            row_ppid = int(parts[1])
            cpu = float(parts[2])
            rss = int(parts[3]) * 1024
        except Exception:
            continue
        comm = parts[5] if len(parts) > 5 else ""
        rows.append((row_pid, row_ppid, cpu, rss, comm))
        cpu_total += cpu
        rss_total += rss
        if row_pid == pid:
            elapsed = parts[4]
    direct_children = [row_pid for row_pid, row_ppid, *_ in rows if row_ppid == pid]
    worker_candidates = [
        row for row in rows if row[0] != pid and row[4] not in {"bash", "sh", "env", "timeout"}
    ]
    if not worker_candidates:
        worker_candidates = [row for row in rows if row[0] != pid]
    worker_pid = max(worker_candidates, key=lambda row: (row[2], row[3]))[0] if worker_candidates else pid
    return ProcessMetrics(
        cpu_total,
        rss_total,
        elapsed,
        runner_pid=pid,
        direct_child_pid=direct_children[0] if direct_children else None,
        worker_pid=worker_pid,
        descendant_pids=descendants,
    )


def process_metrics(pid: int) -> ProcessMetrics:
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu=,rss=,etime="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not output:
            return ProcessMetrics(None, None, None)
        parts = output.split(maxsplit=2)
        cpu = float(parts[0]) if parts else None
        rss = int(parts[1]) * 1024 if len(parts) > 1 else None
        elapsed = parts[2] if len(parts) > 2 else None
        return ProcessMetrics(cpu, rss, elapsed)
    except Exception:
        return ProcessMetrics(None, None, None)


def human_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} PiB"


def write_progress_history(path: Path, sample: ProgressSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(sample.__dict__, sort_keys=True) + "\n")


def load_progress_history(path: Path) -> list[ProgressSample]:
    if not path.exists():
        return []
    samples: list[ProgressSample] = []
    for line in path.read_text().splitlines():
        try:
            data = json.loads(line)
            samples.append(ProgressSample(**data))
        except Exception:
            continue
    return samples[-100:]
