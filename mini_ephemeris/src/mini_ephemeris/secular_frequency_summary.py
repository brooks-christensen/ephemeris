from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .long_term_stability_cli import (
    FREQUENCY_MAP_FIELDS,
    _csv_float,
    dominant_frequency_fft_lite,
    plot_frequency_drift,
)


KNOWN_BODIES = {
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "earth": "earth barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto": "pluto barycenter",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize secular FFT-lite/NAFF-lite frequency drift from orbital_elements CSVs."
    )
    parser.add_argument("--orbital-elements", type=Path, required=True)
    parser.add_argument("--bodies", default="mercury,venus,earth,mars,jupiter,saturn")
    parser.add_argument("--window-years", type=float, required=True)
    parser.add_argument("--step-years", type=float, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--frequency-min-samples", type=int, default=8)
    return parser


def parse_bodies(text: str, available: set[str]) -> list[str]:
    if text.strip().lower() == "all":
        return sorted(available)
    selected: list[str] = []
    for token in text.split(","):
        key = token.strip().lower()
        if not key:
            continue
        body = KNOWN_BODIES.get(key, token.strip())
        if body in available:
            selected.append(body)
    return selected


def load_series(path: Path) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if not body:
                continue
            values = series.setdefault(
                body,
                {"time": [], "e": [], "varpi_rad": [], "i_rad": [], "Omega_rad": []},
            )
            try:
                values["time"].append(float(row["time_years"]))
                values["e"].append(float(row["e"]))
                values["varpi_rad"].append(math.radians(float(row["varpi_deg"])))
                values["i_rad"].append(math.radians(float(row["i_deg"])))
                values["Omega_rad"].append(math.radians(float(row["Omega_deg"])))
            except (KeyError, TypeError, ValueError):
                continue
    return series


def compute_rows(
    series: dict[str, dict[str, list[float]]],
    selected_bodies: list[str],
    *,
    window_years: float,
    step_years: float,
    min_samples: int,
) -> list[dict]:
    rows: list[dict] = []
    previous_frequency: dict[tuple[str, str], float] = {}
    for body in selected_bodies:
        values = series.get(body)
        if not values:
            rows.append(
                {
                    "body": body,
                    "variable": "eccentricity_complex",
                    "window_start_years": "",
                    "window_end_years": "",
                    "dominant_frequency_rad_per_year": "",
                    "dominant_period_years": "",
                    "amplitude": "",
                    "estimated_frequency_drift_rad_per_year": "",
                    "n_samples": 0,
                    "warning": "no orbital-element samples for selected body",
                }
            )
            continue
        times = np.array(values["time"], dtype=float)
        start = float(np.min(times))
        max_time = float(np.max(times))
        while start + window_years <= max_time + 1.0e-9:
            end = start + window_years
            mask = (times >= start - 1.0e-9) & (times <= end + 1.0e-9)
            n_samples = int(np.count_nonzero(mask))
            for variable in ("eccentricity_complex", "inclination_complex"):
                warning = ""
                if n_samples < min_samples:
                    frequency = math.nan
                    amplitude = math.nan
                    warning = f"window has {n_samples} samples; requires at least {min_samples}"
                else:
                    if variable == "eccentricity_complex":
                        complex_values = np.array(values["e"], dtype=float)[mask] * np.exp(
                            1j * np.array(values["varpi_rad"], dtype=float)[mask]
                        )
                    else:
                        complex_values = np.sin(0.5 * np.array(values["i_rad"], dtype=float)[mask]) * np.exp(
                            1j * np.array(values["Omega_rad"], dtype=float)[mask]
                        )
                    frequency, amplitude, warning = dominant_frequency_fft_lite(times[mask], complex_values)
                previous = previous_frequency.get((body, variable))
                drift = math.nan if previous is None or not math.isfinite(frequency) else frequency - previous
                if math.isfinite(frequency):
                    previous_frequency[(body, variable)] = frequency
                rows.append(
                    {
                        "body": body,
                        "variable": variable,
                        "window_start_years": start,
                        "window_end_years": end,
                        "dominant_frequency_rad_per_year": _csv_float(frequency),
                        "dominant_period_years": (
                            _csv_float(2.0 * math.pi / abs(frequency))
                            if math.isfinite(frequency) and frequency != 0.0
                            else ""
                        ),
                        "amplitude": _csv_float(amplitude),
                        "estimated_frequency_drift_rad_per_year": _csv_float(drift),
                        "n_samples": n_samples,
                        "warning": warning,
                    }
                )
            start += step_years
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window_years <= 0.0 or args.step_years <= 0.0:
        parser.error("--window-years and --step-years must be positive.")
    if args.frequency_min_samples < 4:
        parser.error("--frequency-min-samples must be at least 4.")

    series = load_series(args.orbital_elements)
    selected = parse_bodies(args.bodies, set(series))
    if not selected:
        parser.error("No selected bodies were found in the orbital-elements file.")

    prefix = args.output_prefix
    if prefix is None:
        stem = args.orbital_elements.stem.replace("orbital_elements_", "")
        prefix = args.orbital_elements.with_name(f"secular_frequency_summary_{stem}")
    output_dir = prefix.parent
    tag = prefix.name.replace("secular_frequency_summary_", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")

    rows = compute_rows(
        series,
        selected,
        window_years=args.window_years,
        step_years=args.step_years,
        min_samples=args.frequency_min_samples,
    )
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FREQUENCY_MAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = plot_frequency_drift(rows, output_dir=output_dir, tag=tag)
    warnings = sorted({str(row["warning"]) for row in rows if row.get("warning")})
    summary = {
        "diagnostic": "FFT-lite / NAFF-lite secular frequency summary",
        "not_full_laskar_naff": True,
        "orbital_elements": str(args.orbital_elements),
        "bodies": selected,
        "window_years": args.window_years,
        "step_years": args.step_years,
        "row_count": len(rows),
        "warnings": warnings,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote csv: {csv_path}")
    print(f"wrote json: {json_path}")
    for path in plot_paths:
        print(f"wrote plot: {path}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
