from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Newtonian and GR secular mode-tracker outputs."
    )
    parser.add_argument("--newtonian-peaks", type=Path, required=True)
    parser.add_argument("--gr-peaks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--frequency-tolerance-rad-per-year", type=float, default=1.0e-4)
    return parser


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "secular_mode_comparison"


def load_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if b"\0" in raw:
        raise SystemExit(f"CSV contains NUL bytes: {path}")
    with path.open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    return rows


def f(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, float, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, float, float], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("body", ""),
            row.get("variable", ""),
            f(row.get("window_start_years")),
            f(row.get("window_end_years")),
        )
        grouped.setdefault(key, []).append(row)
    for key in list(grouped):
        grouped[key].sort(key=lambda row: int(f(row.get("peak_rank"))) if row.get("peak_rank") not in ("", None) else 10**9)
    return grouped


def closest_peak(source_freq: float, target_rows: list[dict[str, str]]) -> dict[str, object]:
    best: dict[str, object] | None = None
    for row in target_rows:
        target_freq = f(row.get("frequency_rad_per_year"))
        if not math.isfinite(source_freq) or not math.isfinite(target_freq):
            continue
        diff = abs(target_freq - source_freq)
        candidate = {
            "peak_rank": int(f(row.get("peak_rank"))),
            "frequency_rad_per_year": target_freq,
            "period_years": f(row.get("period_years")),
            "amplitude": f(row.get("amplitude")),
            "relative_amplitude": f(row.get("relative_amplitude")),
            "diff": diff,
        }
        if best is None or diff < float(best["diff"]):
            best = candidate
    return best or {
        "peak_rank": "",
        "frequency_rad_per_year": math.nan,
        "period_years": math.nan,
        "amplitude": math.nan,
        "relative_amplitude": math.nan,
        "diff": math.nan,
    }


def classify_window(
    *,
    n_dom_rank: int,
    g_dom_rank: int,
    n_to_g_rank: int,
    g_to_n_rank: int,
    n_shift: float,
    g_shift: float,
    tolerance: float,
) -> str:
    if not math.isfinite(n_shift) or not math.isfinite(g_shift):
        return "ambiguous"
    if n_to_g_rank == 1 and g_to_n_rank == 1:
        if abs(n_shift) <= tolerance and abs(g_shift) <= tolerance:
            return "matched_persistent_mode"
        return "persistent_mode_shift"
    if n_dom_rank != 1 or g_dom_rank != 1 or n_to_g_rank != 1 or g_to_n_rank != 1:
        return "dominant_peak_switching"
    if abs(n_shift) > tolerance or abs(g_shift) > tolerance:
        return "ambiguous"
    return "matched_persistent_mode"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)

    n_rows = load_rows(args.newtonian_peaks)
    g_rows = load_rows(args.gr_peaks)
    n_grouped = group_rows(n_rows)
    g_grouped = group_rows(g_rows)

    common_keys = sorted(set(n_grouped) & set(g_grouped))
    if not common_keys:
        raise SystemExit("No matching body/variable/window groups were found between the two peak files.")

    detail_rows: list[dict[str, object]] = []
    summary_groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    tolerance = float(args.frequency_tolerance_rad_per_year)
    for key in common_keys:
        body, variable, start, end = key
        n_peaks = n_grouped[key]
        g_peaks = g_grouped[key]
        n_dom = n_peaks[0]
        g_dom = g_peaks[0]
        n_dom_freq = f(n_dom.get("frequency_rad_per_year"))
        g_dom_freq = f(g_dom.get("frequency_rad_per_year"))
        n_to_g = closest_peak(n_dom_freq, g_peaks)
        g_to_n = closest_peak(g_dom_freq, n_peaks)
        status = classify_window(
            n_dom_rank=1,
            g_dom_rank=1,
            n_to_g_rank=int(n_to_g["peak_rank"]) if n_to_g["peak_rank"] != "" else 10**9,
            g_to_n_rank=int(g_to_n["peak_rank"]) if g_to_n["peak_rank"] != "" else 10**9,
            n_shift=float(g_dom_freq - n_dom_freq) if math.isfinite(n_dom_freq) and math.isfinite(g_dom_freq) else math.nan,
            g_shift=float(n_dom_freq - g_dom_freq) if math.isfinite(n_dom_freq) and math.isfinite(g_dom_freq) else math.nan,
            tolerance=tolerance,
        )
        detail = {
            "body": body,
            "variable": variable,
            "window_start_years": start,
            "window_end_years": end,
            "newtonian_dominant_frequency_rad_per_year": n_dom_freq,
            "newtonian_dominant_peak_rank": int(f(n_dom.get("peak_rank"))),
            "newtonian_dominant_relative_amplitude": f(n_dom.get("relative_amplitude")),
            "gr_dominant_frequency_rad_per_year": g_dom_freq,
            "gr_dominant_peak_rank": int(f(g_dom.get("peak_rank"))),
            "gr_dominant_relative_amplitude": f(g_dom.get("relative_amplitude")),
            "dominant_frequency_shift_rad_per_year": g_dom_freq - n_dom_freq if math.isfinite(n_dom_freq) and math.isfinite(g_dom_freq) else math.nan,
            "newtonian_to_gr_best_match_rank": int(n_to_g["peak_rank"]) if n_to_g["peak_rank"] != "" else "",
            "newtonian_to_gr_best_match_frequency_rad_per_year": n_to_g["frequency_rad_per_year"],
            "newtonian_to_gr_best_match_diff_rad_per_year": n_to_g["diff"],
            "gr_to_newtonian_best_match_rank": int(g_to_n["peak_rank"]) if g_to_n["peak_rank"] != "" else "",
            "gr_to_newtonian_best_match_frequency_rad_per_year": g_to_n["frequency_rad_per_year"],
            "gr_to_newtonian_best_match_diff_rad_per_year": g_to_n["diff"],
            "comparison_classification": status,
            "warning": "",
        }
        if status == "dominant_peak_switching":
            detail["warning"] = "dominant peaks do not match cleanly; likely mode switching"
        elif status == "ambiguous":
            detail["warning"] = "mode matching ambiguous in this window"
        detail_rows.append(detail)
        summary_groups.setdefault((body, variable), []).append(detail)

    summary_rows: list[dict[str, object]] = []
    for (body, variable), rows in sorted(summary_groups.items()):
        matched = [row for row in rows if row["comparison_classification"] == "matched_persistent_mode"]
        shifted = [row for row in rows if row["comparison_classification"] == "persistent_mode_shift"]
        switched = [row for row in rows if row["comparison_classification"] == "dominant_peak_switching"]
        ambiguous = [row for row in rows if row["comparison_classification"] == "ambiguous"]
        shifts = [abs(float(row["dominant_frequency_shift_rad_per_year"])) for row in rows if math.isfinite(float(row["dominant_frequency_shift_rad_per_year"]))]
        median_shift = float(np.median(shifts)) if shifts else math.nan
        max_shift = float(np.max(shifts)) if shifts else math.nan
        if len(switched) >= max(1, len(rows) // 4):
            classification = "dominant_peak_switching"
            reason = "Dominant peak rank changes in a substantial fraction of windows."
        elif len(matched) + len(shifted) >= max(1, len(rows) // 2):
            classification = "persistent_mode_shift"
            reason = "Top peaks match in most windows; GR acts more like a mode shift than switching."
        else:
            classification = "ambiguous"
            reason = "Peak identity is mixed across windows."
        summary_rows.append(
            {
                "body": body,
                "variable": variable,
                "window_count": len(rows),
                "matched_persistent_mode_count": len(matched),
                "persistent_mode_shift_count": len(shifted),
                "dominant_peak_switching_count": len(switched),
                "ambiguous_window_count": len(ambiguous),
                "median_abs_frequency_shift_rad_per_year": median_shift,
                "max_abs_frequency_shift_rad_per_year": max_shift,
                "classification": classification,
                "reason": reason,
            }
        )

    csv_path = output_dir / f"secular_mode_comparison_{tag}.csv"
    json_path = output_dir / f"secular_mode_comparison_{tag}.json"
    md_path = output_dir / f"secular_mode_comparison_{tag}.md"
    fieldnames = [
        "body",
        "variable",
        "window_count",
        "matched_persistent_mode_count",
        "persistent_mode_shift_count",
        "dominant_peak_switching_count",
        "ambiguous_window_count",
        "median_abs_frequency_shift_rad_per_year",
        "max_abs_frequency_shift_rad_per_year",
        "classification",
        "reason",
    ]
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "diagnostic": "secular mode comparison",
        "not_full_laskar_naff": True,
        "newtonian_peaks": str(args.newtonian_peaks),
        "gr_peaks": str(args.gr_peaks),
        "frequency_tolerance_rad_per_year": tolerance,
        "detail_row_count": len(detail_rows),
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    md_lines = [
        f"# Secular Mode Comparison: {tag}",
        "",
        "FFT-lite / NAFF-lite comparison of multi-peak secular modes. This is diagnostic only, not full Laskar NAFF.",
        "",
        "| body | variable | class | median |max| shift | matched | switching | ambiguous | reason |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['body']} | {row['variable']} | {row['classification']} | "
            f"{float(row['median_abs_frequency_shift_rad_per_year']):.6g} | {float(row['max_abs_frequency_shift_rad_per_year']):.6g} | "
            f"{float(row['window_count'])} | {float(row['matched_persistent_mode_count'])} | "
            f"{float(row['dominant_peak_switching_count'])} | {float(row['ambiguous_window_count'])} | {row['reason']} |"
        )
    md_lines.extend(
        [
            "",
            "Interpretation: if top peaks swap ranks across windows, treat the GR-vs-Newtonian difference as mode switching rather than a clean secular detuning.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"wrote csv: {csv_path}")
    print(f"wrote json: {json_path}")
    print(f"wrote markdown: {md_path}")
    for row in summary_rows:
        if row["body"] == "venus barycenter" and row["variable"] == "inclination_complex":
            print(
                "venus_inclination_classification:",
                row["classification"],
                row["reason"],
            )


if __name__ == "__main__":
    main()
