from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
from pathlib import Path
from typing import Any


BENCHMARK_FIELDS = [
    "tag",
    "model_scope",
    "duration_years",
    "step_days",
    "n_steps",
    "runtime_seconds",
    "runtime_minutes",
    "steps_per_second",
    "max_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "max_com_velocity_drift",
    "finite_time_lambda_1_per_year",
    "classification",
    "main_warnings",
    "summary_path",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return value


def _finite_or_blank(value: Any) -> float | str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _string_or_blank(value: Any) -> str:
    return "" if value is None else str(value)


def _resolve_input_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        expanded = [Path(match) for match in glob.glob(item)]
        if not expanded:
            expanded = [Path(item)]
        for path in expanded:
            if path.is_dir():
                paths.extend(sorted(path.glob("summary_*.json")))
            elif path.is_file():
                paths.append(path)
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return list(unique.keys())


def _load_linked_json(summary: dict[str, Any], key: str) -> dict[str, Any] | None:
    section = summary.get(key)
    if not isinstance(section, dict):
        return None
    path_text = section.get("summary_path")
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _extract_warnings(summary: dict[str, Any], lyapunov_summary: dict[str, Any] | None) -> str:
    warnings: list[str] = []
    for source in (summary, lyapunov_summary or {}):
        values = source.get("warnings")
        if isinstance(values, list):
            warnings.extend(str(value) for value in values if value)
        caveats = source.get("caveats")
        if isinstance(caveats, list):
            warnings.extend(str(value) for value in caveats if value)
    return "; ".join(warnings[:4])


def _row_from_summary(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    config = summary.get("configuration", {})
    if not isinstance(config, dict):
        config = {}
    counts = summary.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    runtime = summary.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    extrema = summary.get("diagnostic_extrema_over_records", {})
    if not isinstance(extrema, dict):
        extrema = {}

    tag = config.get("tag")
    if not tag:
        tag = path.stem.removeprefix("summary_").removeprefix("lyapunov_summary_")
    duration = config.get("duration_years_actual", config.get("duration_years"))
    step_days = config.get("step_days")
    n_steps = counts.get("n_steps_or_nominal_max_steps")
    runtime_seconds = runtime.get("wall_clock_seconds")
    try:
        steps_per_second = float(n_steps) / float(runtime_seconds)
    except (TypeError, ValueError, ZeroDivisionError):
        steps_per_second = ""

    lyapunov_summary = _load_linked_json(summary, "lyapunov")
    finite_lambda: Any = ""
    if lyapunov_summary is not None:
        final = lyapunov_summary.get("final_running_estimate", {})
        fit = lyapunov_summary.get("fit", {})
        if isinstance(final, dict):
            finite_lambda = final.get("lambda_1_per_year")
        if (finite_lambda is None or finite_lambda == "") and isinstance(fit, dict):
            finite_lambda = fit.get("lambda_1_per_year")
    elif summary.get("diagnostic", "").startswith("finite-time"):
        final = summary.get("final_running_estimate", {})
        fit = summary.get("fit", {})
        if isinstance(final, dict):
            finite_lambda = final.get("lambda_1_per_year")
        if (finite_lambda is None or finite_lambda == "") and isinstance(fit, dict):
            finite_lambda = fit.get("lambda_1_per_year")

    classification = summary.get("classification", "")
    if not classification and isinstance(summary.get("duration_scaling"), dict):
        classification = summary["duration_scaling"].get("classification", "")

    return {
        "tag": _string_or_blank(tag),
        "model_scope": _string_or_blank(config.get("model_scope")),
        "duration_years": _finite_or_blank(duration),
        "step_days": _finite_or_blank(step_days),
        "n_steps": _finite_or_blank(n_steps),
        "runtime_seconds": _finite_or_blank(runtime_seconds),
        "runtime_minutes": _finite_or_blank(runtime.get("wall_clock_minutes")),
        "steps_per_second": _finite_or_blank(steps_per_second),
        "max_energy_rel_drift": _finite_or_blank(extrema.get("max_abs_energy_rel_drift")),
        "max_angular_momentum_rel_drift": _finite_or_blank(
            extrema.get("max_angular_momentum_rel_drift")
        ),
        "max_com_velocity_drift": _finite_or_blank(
            extrema.get("max_com_velocity_drift_au_per_year")
        ),
        "finite_time_lambda_1_per_year": _finite_or_blank(finite_lambda),
        "classification": _string_or_blank(classification),
        "main_warnings": _extract_warnings(summary, lyapunov_summary),
        "summary_path": str(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize stability-mode runtime and conservation diagnostics."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON files, globs, or directories containing summary_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for benchmark_summary_<timestamp>.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output CSV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_paths = _resolve_input_paths(args.inputs)
    if not input_paths:
        parser.error("No summary JSON files found.")

    rows = [_row_from_summary(path) for path in input_paths]
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"benchmark_summary_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
