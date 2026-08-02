from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProgressSource:
    kind: str
    path_globs: tuple[str, ...]
    time_columns: tuple[str, ...] = ("time_years", "t_years", "time_yr")
    archive_time_unit: str = "auto"


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    title: str
    objective: str
    command: tuple[str, ...]
    cwd: str
    output_dir: str
    target_years: float | None = None
    progress_sources: tuple[ProgressSource, ...] = ()
    status_interval_seconds: int = 300
    stall_warning_seconds: int = 7200
    depends_on: tuple[str, ...] = ()
    approval_required_before: bool = False
    resume_args: tuple[str, ...] = ()
    resume_probe_globs: tuple[str, ...] = ()
    gates: tuple[dict[str, Any], ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    description: str
    state_dir: str
    variables: dict[str, str]
    stages: tuple[StageSpec, ...]
    source_path: Path
