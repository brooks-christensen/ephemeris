from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return value


def _resolve_inputs(inputs: list[str]) -> tuple[list[Path], list[Path]]:
    summary_paths: list[Path] = []
    scaling_paths: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)] or [Path(item)]
        for path in matches:
            if path.is_dir():
                summary_paths.extend(sorted(path.glob("summary_*.json")))
                scaling_paths.extend(sorted(path.glob("*duration_scaling_summary.csv")))
            elif path.suffix.lower() == ".json":
                summary_paths.append(path)
            elif path.suffix.lower() == ".csv" and "scaling_summary" in path.name:
                scaling_paths.append(path)
    unique_summaries = list(dict.fromkeys(path.resolve() for path in summary_paths))
    unique_scaling = list(dict.fromkeys(path.resolve() for path in scaling_paths))
    return unique_summaries, unique_scaling


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}g}"


def _load_optional_json(path_text: Any) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _max_elements(orbital_elements_path: Path) -> dict[str, dict[str, float]]:
    maxima: dict[str, dict[str, float]] = {}
    if not orbital_elements_path.exists():
        return maxima
    with orbital_elements_path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if not body:
                continue
            bucket = maxima.setdefault(
                body,
                {"max_e": 0.0, "max_i_deg": 0.0, "max_a_au": -math.inf, "min_a_au": math.inf},
            )
            for key, column in (("max_e", "e"), ("max_i_deg", "i_deg")):
                try:
                    bucket[key] = max(bucket[key], abs(float(row[column])))
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                a_au = float(row["a_au"])
            except (KeyError, TypeError, ValueError):
                continue
            bucket["max_a_au"] = max(bucket["max_a_au"], a_au)
            bucket["min_a_au"] = min(bucket["min_a_au"], a_au)
    return maxima


def _read_scaling_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                row = dict(row)
                row["_source"] = str(path)
                rows.append(row)
    return rows


def _recommendation(
    summary: dict[str, Any],
    scaling_rows: list[dict[str, str]],
) -> str:
    config = summary.get("configuration", {})
    extrema = summary.get("diagnostic_extrema_over_records", {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(extrema, dict):
        extrema = {}
    model_scope = str(config.get("model_scope", ""))
    try:
        energy = float(extrema.get("max_abs_energy_rel_drift", math.nan))
    except (TypeError, ValueError):
        energy = math.nan
    if math.isfinite(energy) and energy > 1.0e-4:
        return "Repeat with a smaller timestep before interpreting nonlinear diagnostics."
    if scaling_rows:
        classifications = {row.get("classification", "") for row in scaling_rows}
        if "chaotic_candidate" in classifications:
            return "Extend duration and repeat across timestep, perturbation, and renormalization interval before interpretation."
        if "ambiguous" in classifications:
            return "Add longer duration points and a timestep ladder to resolve the ambiguous duration scaling."
        if classifications == {"near_integrable_likely"}:
            return "Use a longer duration or richer model scope only after confirming conservation at the chosen timestep."
    if model_scope == "inner":
        return "Run the inner duration-scaling ladder before interpreting finite-time Lyapunov, FLI, or MEGNO-lite values."
    if model_scope == "full":
        return "Use this as a broad survey baseline; run inner Mercury-sensitive duration scaling for chaos interpretation."
    return "Run the appropriate two-body or inner duration-scaling validation before interpreting chaos indicators."


def _summary_section(
    path: Path,
    summary: dict[str, Any],
    scaling_rows: list[dict[str, str]],
) -> list[str]:
    config = summary.get("configuration", {})
    counts = summary.get("counts", {})
    runtime = summary.get("runtime", {})
    extrema = summary.get("diagnostic_extrema_over_records", {})
    outputs = summary.get("outputs", {})
    if not isinstance(config, dict):
        config = {}
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(extrema, dict):
        extrema = {}
    if not isinstance(outputs, dict):
        outputs = {}

    tag = str(config.get("tag") or path.stem.removeprefix("summary_"))
    lines = [f"## Run: `{tag}`", ""]
    lines.extend(
        [
            "### Model Configuration",
            "",
            f"- Model scope: `{config.get('model_scope', 'n/a')}`",
            f"- GR model: `{config.get('gr_model', 'n/a')}`",
            f"- Integrator: `{config.get('integrator', 'n/a')}`",
            f"- Duration: {_fmt(config.get('duration_years_actual', config.get('duration_years_requested')))} years",
            f"- Step: {_fmt(config.get('step_days'))} days",
            f"- Records: {_fmt(counts.get('n_records'), 0)}",
            f"- Steps: {_fmt(counts.get('n_steps_or_nominal_max_steps'), 0)}",
            f"- Runtime: {_fmt(runtime.get('wall_clock_seconds'))} s ({_fmt(runtime.get('wall_clock_minutes'))} min)",
            "",
            "### Conservation Diagnostics",
            "",
            f"- Max relative energy drift: {_fmt(extrema.get('max_abs_energy_rel_drift'))}",
            f"- Max relative angular momentum drift: {_fmt(extrema.get('max_angular_momentum_rel_drift'))}",
            f"- Max COM velocity drift: {_fmt(extrema.get('max_com_velocity_drift_au_per_year'))} AU/year",
            "",
        ]
    )

    min_separations = summary.get("min_separations")
    if isinstance(min_separations, list) and min_separations:
        sorted_pairs = sorted(
            min_separations,
            key=lambda row: float(row.get("min_separation_au", math.inf)),
        )
        lines.extend(["### Minimum Separations", ""])
        for row in sorted_pairs[:8]:
            lines.append(
                "- "
                f"{row.get('body_i')} - {row.get('body_j')}: "
                f"{_fmt(row.get('min_separation_au'))} AU at "
                f"{_fmt(row.get('time_years'))} years"
            )
        lines.append("")

    orbital_path_text = outputs.get("orbital_elements")
    if orbital_path_text:
        maxima = _max_elements(Path(str(orbital_path_text)))
        if maxima:
            lines.extend(["### Orbital Element Extrema", ""])
            lines.append("| Body | max e | max i deg | a range AU |")
            lines.append("| --- | ---: | ---: | ---: |")
            for body, values in sorted(maxima.items()):
                a_min = values.get("min_a_au")
                a_max = values.get("max_a_au")
                a_range = (
                    f"{_fmt(a_min)} - {_fmt(a_max)}"
                    if math.isfinite(float(a_min)) and math.isfinite(float(a_max))
                    else "n/a"
                )
                lines.append(
                    f"| {body} | {_fmt(values.get('max_e'))} | "
                    f"{_fmt(values.get('max_i_deg'))} | {a_range} |"
                )
            lines.append("")

    lyapunov_section = summary.get("lyapunov")
    if isinstance(lyapunov_section, dict) and lyapunov_section.get("enabled"):
        lyapunov_summary = _load_optional_json(lyapunov_section.get("summary_path"))
        lines.extend(["### Finite-Time Tangent Diagnostics", ""])
        lines.append(
            "These are finite-time diagnostics, not an asymptotic Lyapunov exponent."
        )
        if lyapunov_summary:
            fit = lyapunov_summary.get("fit", {})
            final = lyapunov_summary.get("final_running_estimate", {})
            if isinstance(fit, dict):
                lines.append(f"- Fit lambda: {_fmt(fit.get('lambda_1_per_year'))} 1/year")
                lines.append(f"- Fit finite-time scale: {_fmt(fit.get('lyapunov_time_years'))} years")
            if isinstance(final, dict):
                lines.append(
                    f"- Final running lambda: {_fmt(final.get('lambda_1_per_year'))} 1/year"
                )
            warnings = lyapunov_summary.get("warnings")
            if isinstance(warnings, list) and warnings:
                lines.append("- Main warning: " + str(warnings[0]))
        lines.append("")

    fli_section = summary.get("fli_megno_lite")
    if isinstance(fli_section, dict) and fli_section.get("enabled"):
        lines.extend(["### FLI/MEGNO-lite", ""])
        lines.append(
            "- FLI-lite and MEGNO-lite are finite-time tangent-growth indicators."
        )
        if "final_fli" in fli_section:
            lines.append(f"- Final FLI-lite: {_fmt(fli_section.get('final_fli'))}")
            lines.append(f"- Final MEGNO-lite: {_fmt(fli_section.get('final_megno_lite'))}")
        lines.append("")

    if scaling_rows:
        lines.extend(["### Duration-Scaling Classification", ""])
        counts: dict[str, int] = {}
        for row in scaling_rows:
            classification = row.get("classification", "unknown")
            counts[classification] = counts.get(classification, 0) + 1
        for classification, count in sorted(counts.items()):
            lines.append(f"- `{classification}`: {count}")
        examples = scaling_rows[:5]
        for row in examples:
            lines.append(
                "- "
                f"{row.get('model_scope', 'scope')} step={row.get('step_days')} "
                f"renorm={row.get('renorm_years')} perturb={row.get('perturbation_m')} "
                f"body={row.get('lyapunov_body', row.get('body', 'n/a'))}: "
                f"`{row.get('classification')}`"
            )
        lines.append("")

    lines.extend(["### Recommended Next Run", "", _recommendation(summary, scaling_rows), ""])
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a conservative Markdown scientific summary for stability outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Output directories, summary JSON files, or duration scaling summary CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for stability_report_<tag_or_timestamp>.md.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional report tag used in the output filename.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit Markdown output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary_paths, scaling_paths = _resolve_inputs(args.inputs)
    if not summary_paths and not scaling_paths:
        parser.error("No summary JSON or duration scaling summary CSV files found.")

    scaling_rows = _read_scaling_rows(scaling_paths)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_tag = args.tag or (
        summary_paths[0].stem.removeprefix("summary_") if len(summary_paths) == 1 else timestamp
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path(args.output_dir) / f"stability_report_{report_tag}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Stability Scientific Summary",
        "",
        "This report uses finite-time language intentionally. It does not turn "
        "a short diagnostic into an asymptotic Solar System Lyapunov claim.",
        "",
    ]
    if not summary_paths and scaling_rows:
        lines.extend(["## Duration-Scaling Inputs", ""])
        for path in scaling_paths:
            lines.append(f"- {path}")
        lines.append("")
        lines.extend(["### Classification Counts", ""])
        counts: dict[str, int] = {}
        for row in scaling_rows:
            classification = row.get("classification", "unknown")
            counts[classification] = counts.get(classification, 0) + 1
        for classification, count in sorted(counts.items()):
            lines.append(f"- `{classification}`: {count}")
        lines.append("")
    for path in summary_paths:
        summary = _read_json(path)
        lines.extend(_summary_section(path, summary, scaling_rows))

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")
    print(f"Summary files: {len(summary_paths)}")
    print(f"Scaling rows: {len(scaling_rows)}")


if __name__ == "__main__":
    main()
