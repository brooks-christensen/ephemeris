from __future__ import annotations

import json
import os
import string
from pathlib import Path
from typing import Any

from .models import ExperimentSpec, ProgressSource, StageSpec


class ManifestError(ValueError):
    pass


def _expand_text(value: str, variables: dict[str, str]) -> str:
    merged = dict(os.environ)
    merged.update(variables)
    current = value
    for _ in range(10):
        expanded = string.Template(current).safe_substitute(merged)
        expanded = os.path.expanduser(expanded)
        if expanded == current:
            return expanded
        current = expanded
    return current


def _expand_obj(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_text(value, variables)
    if isinstance(value, list):
        return [_expand_obj(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_obj(item, variables) for key, item in value.items()}
    return value


def load_manifest(path: str | Path) -> ExperimentSpec:
    source = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"Manifest not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON in {source}: {exc}") from exc

    required = ["experiment_id", "title", "description", "state_dir", "stages"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ManifestError(f"Missing manifest keys: {', '.join(missing)}")

    variables = {str(k): str(v) for k, v in raw.get("variables", {}).items()}
    variables.setdefault("MANIFEST_DIR", str(source.parent))
    # Resolve variables against each other before expanding the full document.
    for _ in range(10):
        updated = {key: _expand_text(value, variables) for key, value in variables.items()}
        if updated == variables:
            break
        variables = updated

    expanded = _expand_obj(raw, variables)
    seen: set[str] = set()
    stages: list[StageSpec] = []
    for item in expanded["stages"]:
        stage_id = str(item["id"])
        if stage_id in seen:
            raise ManifestError(f"Duplicate stage id: {stage_id}")
        seen.add(stage_id)
        command = item.get("command")
        if not isinstance(command, list) or not command:
            raise ManifestError(f"Stage {stage_id} must have a non-empty command list")
        progress_sources = []
        for source_item in item.get("progress_sources", []):
            path_globs = source_item.get("path_globs") or [source_item.get("path_glob")]
            path_globs = tuple(str(p) for p in path_globs if p)
            if not path_globs:
                raise ManifestError(f"Stage {stage_id} has a progress source without paths")
            progress_sources.append(
                ProgressSource(
                    kind=str(source_item.get("kind", "csv")),
                    path_globs=path_globs,
                    time_columns=tuple(source_item.get("time_columns", ["time_years", "t_years", "time_yr"])),
                    archive_time_unit=str(source_item.get("archive_time_unit", "auto")),
                )
            )
        stages.append(
            StageSpec(
                stage_id=stage_id,
                title=str(item.get("title", stage_id)),
                objective=str(item.get("objective", "")),
                command=tuple(str(part) for part in command),
                cwd=str(item.get("cwd", variables.get("PROJECT_ROOT", source.parent))),
                output_dir=str(item.get("output_dir", variables.get("OUTPUT_ROOT", source.parent))),
                target_years=float(item["target_years"]) if item.get("target_years") is not None else None,
                progress_sources=tuple(progress_sources),
                status_interval_seconds=int(item.get("status_interval_seconds", 300)),
                stall_warning_seconds=int(item.get("stall_warning_seconds", 7200)),
                depends_on=tuple(str(dep) for dep in item.get("depends_on", [])),
                approval_required_before=bool(item.get("approval_required_before", False)),
                resume_args=tuple(str(part) for part in item.get("resume_args", [])),
                resume_probe_globs=tuple(str(part) for part in item.get("resume_probe_globs", [])),
                gates=tuple(item.get("gates", [])),
                env={str(k): str(v) for k, v in item.get("env", {}).items()},
            )
        )

    unknown_dependencies = {
        dep for stage in stages for dep in stage.depends_on if dep not in seen
    }
    if unknown_dependencies:
        raise ManifestError(
            "Unknown stage dependencies: " + ", ".join(sorted(unknown_dependencies))
        )

    return ExperimentSpec(
        experiment_id=str(expanded["experiment_id"]),
        title=str(expanded["title"]),
        description=str(expanded["description"]),
        state_dir=str(expanded["state_dir"]),
        variables={str(k): str(v) for k, v in expanded.get("variables", {}).items()},
        stages=tuple(stages),
        source_path=source,
    )
