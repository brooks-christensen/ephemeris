from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

KNOWN_BODIES = {
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "earth": "earth barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track top secular FFT-lite peaks from orbital-elements CSVs."
    )
    parser.add_argument("--orbital-elements", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bodies", default="mercury,venus,earth,mars,jupiter,saturn")
    parser.add_argument("--window-years", type=float, default=20_000_000.0)
    parser.add_argument("--step-years", type=float, default=5_000_000.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-period-years", type=float, default=None)
    parser.add_argument("--max-period-years", type=float, default=None)
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


def body_slug(body: str) -> str:
    return body.replace(" barycenter", "").replace(" ", "_")


def load_series(path: Path) -> dict[str, dict[str, list[float]]]:
    series: dict[str, dict[str, list[float]]] = {}
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if not body:
                continue
            bucket = series.setdefault(
                body,
                {"time": [], "e": [], "varpi_rad": [], "i_rad": [], "Omega_rad": []},
            )
            try:
                bucket["time"].append(float(row["time_years"]))
                bucket["e"].append(float(row["e"]))
                bucket["varpi_rad"].append(math.radians(float(row["varpi_deg"])))
                bucket["i_rad"].append(math.radians(float(row["i_deg"])))
                bucket["Omega_rad"].append(math.radians(float(row["Omega_deg"])))
            except (KeyError, TypeError, ValueError):
                continue
    for values in series.values():
        order = np.argsort(np.array(values["time"], dtype=float))
        for key in list(values):
            arr = np.array(values[key], dtype=float)
            values[key] = list(arr[order])
    return series


def _quadratic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset):
        return 0.0
    return max(-0.5, min(0.5, offset))


def spectrum_peaks(
    times: np.ndarray,
    complex_values: np.ndarray,
    *,
    top_k: int,
    min_period_years: float | None,
    max_period_years: float | None,
) -> tuple[list[dict[str, float | int | str]], str]:
    n_samples = int(complex_values.size)
    if n_samples < 4:
        return [], f"fewer than four samples ({n_samples})"
    dt_values = np.diff(times)
    median_dt = float(np.median(dt_values))
    if median_dt <= 0.0 or not math.isfinite(median_dt):
        return [], "invalid sample cadence"
    warnings: list[str] = []
    if np.max(np.abs(dt_values - median_dt)) > 1.0e-6 * max(1.0, abs(median_dt)):
        warnings.append("nonuniform sample cadence; FFT-lite assumes near-uniform sampling")

    centered = complex_values - np.mean(complex_values)
    window = np.hanning(n_samples)
    if float(np.sum(window)) == 0.0:
        window = np.ones(n_samples)

    n_fft = 1
    target = max(256, n_samples * 4)
    while n_fft < target:
        n_fft <<= 1
    spectrum = np.fft.fft(centered * window, n=n_fft)
    freqs = np.fft.fftfreq(n_fft, d=median_dt) * 2.0 * math.pi
    positive = freqs > 0.0
    freqs = freqs[positive]
    mags = np.abs(spectrum[positive])
    if freqs.size < 3 or not np.any(np.isfinite(mags)):
        return [], "no finite positive-frequency bins"
    mags[0] = 0.0

    if min_period_years is not None:
        mags = np.where(freqs >= 2.0 * math.pi / max(min_period_years, 1.0e-300), mags, 0.0)
    if max_period_years is not None:
        mags = np.where(freqs <= 2.0 * math.pi / max(max_period_years, 1.0e-300), mags, 0.0)

    peak_candidates: list[int] = []
    for index in range(1, len(mags) - 1):
        if mags[index] > mags[index - 1] and mags[index] >= mags[index + 1]:
            peak_candidates.append(index)
    if not peak_candidates:
        peak_candidates = list(np.argsort(mags)[::-1][: max(top_k, 1)])
    else:
        peak_candidates.sort(key=lambda index: float(mags[index]), reverse=True)

    if not np.any(mags > 0.0):
        return [], "no nonzero FFT-lite peak"

    peak_rows: list[dict[str, float | int | str]] = []
    peak_max = float(np.max(mags))
    chosen: list[int] = []
    for index in peak_candidates:
        if len(chosen) >= top_k:
            break
        if any(abs(index - previous) <= 1 for previous in chosen):
            continue
        chosen.append(index)
    if len(chosen) < top_k:
        for index in np.argsort(mags)[::-1]:
            if len(chosen) >= top_k:
                break
            if index in chosen or any(abs(index - previous) <= 1 for previous in chosen):
                continue
            chosen.append(int(index))

    for rank, index in enumerate(chosen[:top_k], start=1):
        frequency = float(freqs[index])
        if 0 < index < len(mags) - 1:
            spacing = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
            offset = _quadratic_peak_offset(float(mags[index - 1]), float(mags[index]), float(mags[index + 1]))
            frequency += offset * spacing
        amplitude = float(mags[index] / max(1.0, np.sum(window)))
        relative_amplitude = amplitude / peak_max if peak_max > 0.0 else math.nan
        period_years = 2.0 * math.pi / abs(frequency) if frequency != 0.0 else math.nan
        warning = "; ".join(warnings)
        if min_period_years is not None and math.isfinite(period_years) and period_years < min_period_years:
            warning = "; ".join(filter(None, [warning, f"period shorter than min-period-years {min_period_years:g}"]))
        if max_period_years is not None and math.isfinite(period_years) and period_years > max_period_years:
            warning = "; ".join(filter(None, [warning, f"period longer than max-period-years {max_period_years:g}"]))
        peak_rows.append(
            {
                "peak_rank": rank,
                "frequency_rad_per_year": frequency,
                "period_years": period_years,
                "amplitude": amplitude,
                "relative_amplitude": relative_amplitude,
                "warning": warning,
            }
        )
    return peak_rows, "; ".join(warnings)


def plot_mode_tracks(rows: list[dict[str, object]], output_dir: Path, tag: str) -> list[Path]:
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[Path] = []
    for body in sorted({str(row["body"]) for row in rows}):
        body_rows = [row for row in rows if row["body"] == body]
        if not body_rows:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for ax, variable in zip(axes, ("eccentricity_complex", "inclination_complex"), strict=False):
            subset = [row for row in body_rows if row["variable"] == variable]
            if not subset:
                ax.set_visible(False)
                continue
            midpoints = sorted({0.5 * (float(row["window_start_years"]) + float(row["window_end_years"])) for row in subset})
            for peak_rank in range(1, min(4, max(int(row["peak_rank"]) for row in subset) + 1)):
                peak_rows = [
                    row for row in subset if int(row["peak_rank"]) == peak_rank
                ]
                if not peak_rows:
                    continue
                x_values = [
                    0.5 * (float(row["window_start_years"]) + float(row["window_end_years"]))
                    for row in peak_rows
                ]
                y_values = [float(row["frequency_rad_per_year"]) for row in peak_rows]
                ax.plot(x_values, y_values, marker="o", linewidth=1.0, markersize=3, label=f"peak {peak_rank}")
            ax.set_ylabel("frequency [rad/year]")
            ax.set_title(f"{body} - {variable}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        axes[-1].set_xlabel("window midpoint [years]")
        fig.tight_layout()
        path = output_dir / f"secular_mode_peaks_{tag}_{body_slug(body)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window_years <= 0.0 or args.step_years <= 0.0:
        parser.error("--window-years and --step-years must be positive.")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in args.tag) or "modes"

    series = load_series(args.orbital_elements)
    selected = parse_bodies(args.bodies, set(series))
    if not selected:
        parser.error("No selected bodies were found in the orbital-elements file.")

    rows: list[dict[str, object]] = []
    warnings: set[str] = set()
    for body in selected:
        values = series.get(body)
        if not values:
            warnings.add(f"no orbital-element samples for {body}")
            continue
        times = np.array(values["time"], dtype=float)
        max_time = float(np.max(times))
        start = float(np.min(times))
        while start + args.window_years <= max_time + 1.0e-9:
            end = start + args.window_years
            mask = (times >= start - 1.0e-9) & (times <= end + 1.0e-9)
            n_samples = int(np.count_nonzero(mask))
            window_warning = ""
            if n_samples < 4:
                window_warning = f"window has {n_samples} samples"
            for variable in ("eccentricity_complex", "inclination_complex"):
                if n_samples < 4:
                    peaks: list[dict[str, float | int | str]] = []
                    warning = window_warning
                else:
                    if variable == "eccentricity_complex":
                        complex_values = np.array(values["e"], dtype=float)[mask] * np.exp(
                            1j * np.array(values["varpi_rad"], dtype=float)[mask]
                        )
                    else:
                        complex_values = np.sin(0.5 * np.array(values["i_rad"], dtype=float)[mask]) * np.exp(
                            1j * np.array(values["Omega_rad"], dtype=float)[mask]
                        )
                    peaks, warning = spectrum_peaks(
                        times[mask],
                        complex_values,
                        top_k=args.top_k,
                        min_period_years=args.min_period_years,
                        max_period_years=args.max_period_years,
                    )
                if warning:
                    warnings.add(warning)
                if not peaks:
                    rows.append(
                        {
                            "body": body,
                            "variable": variable,
                            "window_start_years": start,
                            "window_end_years": end,
                            "peak_rank": "",
                            "frequency_rad_per_year": "",
                            "period_years": "",
                            "amplitude": "",
                            "relative_amplitude": "",
                            "warning": warning or "no peaks",
                        }
                    )
                    continue
                for peak in peaks:
                    rows.append(
                        {
                            "body": body,
                            "variable": variable,
                            "window_start_years": start,
                            "window_end_years": end,
                            "peak_rank": peak["peak_rank"],
                            "frequency_rad_per_year": peak["frequency_rad_per_year"],
                            "period_years": peak["period_years"],
                            "amplitude": peak["amplitude"],
                            "relative_amplitude": peak["relative_amplitude"],
                            "warning": peak["warning"],
                        }
                    )
            start += args.step_years

    csv_path = output_dir / f"secular_mode_peaks_{tag}.csv"
    json_path = output_dir / f"secular_mode_peaks_{tag}.json"
    md_path = output_dir / f"secular_mode_peaks_{tag}.md"
    fieldnames = [
        "body",
        "variable",
        "window_start_years",
        "window_end_years",
        "peak_rank",
        "frequency_rad_per_year",
        "period_years",
        "amplitude",
        "relative_amplitude",
        "warning",
    ]
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = plot_mode_tracks(rows, output_dir, tag)
    ranked_preview = sorted(
        [row for row in rows if row.get("peak_rank") in (1, "1") and row.get("frequency_rad_per_year") not in ("", None)],
        key=lambda row: float(row["relative_amplitude"]),
        reverse=True,
    )
    summary = {
        "diagnostic": "secular mode tracker",
        "not_full_laskar_naff": True,
        "orbital_elements": str(args.orbital_elements),
        "bodies": selected,
        "window_years": args.window_years,
        "step_years": args.step_years,
        "top_k": args.top_k,
        "min_period_years": args.min_period_years,
        "max_period_years": args.max_period_years,
        "row_count": len(rows),
        "warnings": sorted({warning for warning in warnings if warning}),
        "top_preview": ranked_preview[:12],
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "plots": [str(path) for path in plot_paths],
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md_lines = [
        f"# Secular Mode Tracker: {tag}",
        "",
        "FFT-lite / NAFF-lite multi-peak secular tracking. This is diagnostic only, not full Laskar NAFF.",
        "",
        "| body | variable | window | peak rank | frequency [rad/year] | period [years] | rel amp | warning |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranked_preview[:20]:
        md_lines.append(
            f"| {row['body']} | {row['variable']} | {float(row['window_start_years']):.3g}:{float(row['window_end_years']):.3g} | "
            f"{row['peak_rank']} | {float(row['frequency_rad_per_year']):.6g} | {float(row['period_years']):.6g} | "
            f"{float(row['relative_amplitude']):.4g} | {row.get('warning', '')} |"
        )
    md_lines.extend(
        [
            "",
            "Use peak-rank comparisons across windows to spot mode switching, not just a single dominant frequency.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"wrote csv: {csv_path}")
    print(f"wrote json: {json_path}")
    print(f"wrote markdown: {md_path}")
    for path in plot_paths:
        print(f"wrote plot: {path}")
    for warning in sorted({warning for warning in warnings if warning}):
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
