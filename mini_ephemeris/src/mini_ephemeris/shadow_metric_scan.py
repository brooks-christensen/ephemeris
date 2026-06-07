from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .shadow_fit_diagnostics import (
    classify_window,
    f,
    linear_fit,
    parse_windows,
    saturation_applies_to_metric,
)


BODY_SLUGS = {
    "mercury": "mercury_barycenter",
    "venus": "venus_barycenter",
    "earth": "earth_barycenter",
    "mars": "mars_barycenter",
    "jupiter": "jupiter_barycenter",
    "saturn": "saturn_barycenter",
    "uranus": "uranus_barycenter",
    "neptune": "neptune_barycenter",
    "pluto": "pluto_barycenter",
}

METRIC_SUFFIXES = (
    "_delta_a_au",
    "_delta_e",
    "_delta_i",
    "_delta_Omega_wrapped",
    "_delta_varpi_wrapped",
    "_delta_lambda_wrapped",
    "_eccentricity_vector_separation",
    "_inclination_vector_separation",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan shadow orbital-element divergence metrics for finite-time exponential-vs-shear fits."
    )
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--windows",
        default="1e6:1e7,1e6:2e7,2e6:2e7,5e6:3e7,1e7:4e7,1e6:5e7",
        help="Comma-separated start:end year windows.",
    )
    parser.add_argument("--bodies", default="all", help="Comma-separated body list or all.")
    parser.add_argument("--saturation-threshold-au", type=float, default=0.1)
    return parser


def sanitize_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "shadow_metric_scan"


def selected_body_slugs(text: str) -> set[str]:
    tokens = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not tokens or "all" in tokens:
        return set(BODY_SLUGS.values())
    unknown = [token for token in tokens if token not in BODY_SLUGS]
    if unknown:
        raise SystemExit(f"Unknown body name(s): {', '.join(unknown)}")
    return {BODY_SLUGS[token] for token in tokens}


def discover_metrics(fieldnames: list[str], body_slugs: set[str]) -> list[str]:
    metrics = ["raw_position_separation_au"] if "raw_position_separation_au" in fieldnames else []
    for field in fieldnames:
        for slug in body_slugs:
            if field.startswith(f"{slug}_") and field.endswith(METRIC_SUFFIXES):
                metrics.append(field)
                break
    return sorted(dict.fromkeys(metrics))


def metric_body(metric: str) -> str:
    for body, slug in BODY_SLUGS.items():
        if metric.startswith(f"{slug}_"):
            return body
    if metric == "raw_position_separation_au":
        return "all_cartesian"
    return ""


def mercury_priority(metric: str) -> int:
    return 1 if metric.startswith("mercury_barycenter_") else 0


def lyapunov_time_stability(lambdas: list[float]) -> tuple[float, float, float]:
    if len(lambdas) < 2:
        return math.nan, math.nan, 0.0
    lyap_times = [1.0 / value for value in lambdas if value > 0.0 and math.isfinite(value)]
    if len(lyap_times) < 2:
        return math.nan, math.nan, 0.0
    mean = float(np.mean(lyap_times))
    std = float(np.std(lyap_times))
    if mean <= 0.0 or not math.isfinite(mean):
        return mean, std, 0.0
    rel_std = abs(std / mean) if math.isfinite(std) else math.nan
    if not math.isfinite(rel_std):
        return mean, rel_std, 0.0
    return mean, rel_std, 1.0 / (1.0 + rel_std)


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    raw = path.read_bytes()
    if b"\0" in raw:
        raise SystemExit(f"Shadow CSV contains NUL bytes: {path}")
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        raise SystemExit(f"Shadow CSV has no header: {path}")
    return reader.fieldnames, list(reader)


def metric_samples(rows: list[dict[str, str]], metric: str) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []
    for row in rows:
        time_years = f(row.get("time_years"))
        value = f(row.get(metric))
        magnitude = abs(value)
        if math.isfinite(time_years) and math.isfinite(value) and magnitude > 0.0:
            samples.append(
                {
                    "time_years": time_years,
                    "metric_value": value,
                    "metric_magnitude": magnitude,
                    "log_metric": math.log(max(magnitude, 1.0e-300)),
                }
            )
    return samples


def evaluate_metric(
    samples: list[dict[str, float]],
    metric: str,
    windows: list[tuple[float, float]],
    saturation_threshold_au: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    saturation_metric = saturation_applies_to_metric(metric)
    for start, end in windows:
        window_samples = [sample for sample in samples if start <= sample["time_years"] <= end]
        sample_count = len(window_samples)
        window_label = f"{start:.6g}:{end:.6g}"
        if sample_count < 3:
            results.append(
                {
                    "metric": metric,
                    "body": metric_body(metric),
                    "window_label": window_label,
                    "window_start_years": start,
                    "window_end_years": end,
                    "lambda_1_per_year": math.nan,
                    "lyapunov_time_years": math.nan,
                    "r_squared_exponential": math.nan,
                    "r_squared_powerlaw": math.nan,
                    "powerlaw_exponent": math.nan,
                    "number_of_samples": sample_count,
                    "exceeded_saturation_threshold": False,
                    "classification": "insufficient_data",
                }
            )
            continue
        xs = np.array([sample["time_years"] for sample in window_samples], dtype=float)
        ys = np.array([sample["log_metric"] for sample in window_samples], dtype=float)
        exp_slope, _, exp_r2 = linear_fit(xs, ys)

        positive_samples = [sample for sample in window_samples if sample["time_years"] > 0.0]
        power_slope = math.nan
        power_r2 = math.nan
        if len(positive_samples) >= 3:
            power_xs = np.log(np.array([sample["time_years"] for sample in positive_samples], dtype=float))
            power_ys = np.array([sample["log_metric"] for sample in positive_samples], dtype=float)
            power_slope, _, power_r2 = linear_fit(power_xs, power_ys)

        saturated = (
            any(sample["metric_magnitude"] >= saturation_threshold_au for sample in window_samples)
            if saturation_metric
            else False
        )
        classification = classify_window(
            sample_count=sample_count,
            end_sep_au=float(window_samples[-1]["metric_magnitude"]),
            saturated=saturated,
            exp_r2=exp_r2,
            power_r2=power_r2,
            lambda_1_per_year=exp_slope,
        )
        results.append(
            {
                "metric": metric,
                "body": metric_body(metric),
                "window_label": window_label,
                "window_start_years": start,
                "window_end_years": end,
                "lambda_1_per_year": exp_slope,
                "lyapunov_time_years": (1.0 / exp_slope) if exp_slope > 0.0 else math.nan,
                "r_squared_exponential": exp_r2,
                "r_squared_powerlaw": power_r2,
                "r2_exponential_minus_powerlaw": exp_r2 - power_r2 if math.isfinite(power_r2) else math.nan,
                "powerlaw_exponent": power_slope,
                "number_of_samples": sample_count,
                "exceeded_saturation_threshold": saturated,
                "classification": classification,
            }
        )
    return results


def aggregate_metric(metric: str, window_results: list[dict[str, object]]) -> dict[str, object]:
    valid = [
        row for row in window_results
        if math.isfinite(float(row["r_squared_exponential"]))
        and math.isfinite(float(row["r_squared_powerlaw"]))
    ]
    exp_candidates = [row for row in valid if row["classification"] == "exponential_candidate"]
    best_exp = max(
        valid,
        key=lambda row: (
            float(row["r_squared_exponential"]),
            float(row.get("r2_exponential_minus_powerlaw", math.nan)),
        ),
        default=None,
    )
    best_by_margin = max(
        valid,
        key=lambda row: (
            float(row.get("r2_exponential_minus_powerlaw", math.nan)),
            float(row["r_squared_exponential"]),
        ),
        default=None,
    )
    lambdas = [
        float(row["lambda_1_per_year"])
        for row in valid
        if math.isfinite(float(row["lambda_1_per_year"])) and float(row["lambda_1_per_year"]) > 0.0
    ]
    lambda_mean = float(np.mean(lambdas)) if lambdas else math.nan
    lambda_std = float(np.std(lambdas)) if lambdas else math.nan
    lambda_relative_std = abs(lambda_std / lambda_mean) if lambdas and lambda_mean != 0.0 else math.nan
    lyap_time_mean, lyap_time_relative_std, lyap_time_stability_score = lyapunov_time_stability(lambdas)
    best_row = best_exp or best_by_margin
    if exp_candidates:
        recommendation = "exponential_candidate_windows_exist; still finite-time and needs duration/timestep/seed checks"
    elif best_row and float(best_row.get("r2_exponential_minus_powerlaw", math.nan)) <= 0.0:
        recommendation = "powerlaw_or_shear_competes_or_wins"
    else:
        recommendation = "ambiguous_or_insufficient"
    best_exp_r2 = float(best_row["r_squared_exponential"]) if best_row else math.nan
    best_margin = float(best_by_margin.get("r2_exponential_minus_powerlaw", math.nan)) if best_by_margin else math.nan
    priority = mercury_priority(metric)
    return {
        "metric": metric,
        "body": metric_body(metric),
        "valid_window_count": len(valid),
        "exponential_candidate_window_count": len(exp_candidates),
        "saturated_or_ambiguous_window_count": sum(
            1 for row in window_results if row["classification"] == "saturated_or_ambiguous"
        ),
        "best_window": best_row["window_label"] if best_row else "",
        "best_classification": best_row["classification"] if best_row else "insufficient_data",
        "best_lambda_1_per_year": best_row["lambda_1_per_year"] if best_row else math.nan,
        "best_lyapunov_time_years": best_row["lyapunov_time_years"] if best_row else math.nan,
        "best_exponential_r_squared": best_row["r_squared_exponential"] if best_row else math.nan,
        "best_powerlaw_r_squared": best_row["r_squared_powerlaw"] if best_row else math.nan,
        "best_exp_minus_power_r_squared": best_row.get("r2_exponential_minus_powerlaw", math.nan) if best_row else math.nan,
        "lambda_mean": lambda_mean,
        "lambda_std": lambda_std,
        "lambda_relative_std": lambda_relative_std,
        "lyapunov_time_mean_years": lyap_time_mean,
        "lyapunov_time_relative_std": lyap_time_relative_std,
        "lyapunov_time_stability_score": lyap_time_stability_score,
        "mercury_priority": priority,
        "ranking_exp_r_squared": best_exp_r2,
        "ranking_exp_minus_power_r_squared": best_margin,
        "recommendation": recommendation,
    }


def write_markdown(path: Path, aggregates: list[dict[str, object]], tag: str) -> None:
    lines = [
        f"# Shadow Metric Scan: {tag}",
        "",
        "Finite-time orbital/secular shadow divergence scan. This is not an asymptotic Lyapunov exponent.",
        "",
        "## Best Metric Table",
        "",
        "| rank | metric | body | exp r^2 | exp-power r^2 | lyap-time rel std | mercury priority | recommendation |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(aggregates[:12], start=1):
        lines.append(
            "| {rank} | {metric} | {body} | {exp:.4g} | {margin:.4g} | {stability:.4g} | {priority} | {rec} |".format(
                rank=rank,
                metric=row["metric"],
                body=row["body"],
                exp=float(row["ranking_exp_r_squared"]),
                margin=float(row["ranking_exp_minus_power_r_squared"]),
                stability=float(row["lyapunov_time_relative_std"]),
                priority=int(row["mercury_priority"]),
                rec=row["recommendation"],
            )
        )
    lines.extend(
        [
            "",
            "Ranking criteria: higher exp r^2, higher exp-power r^2 margin, more stable Lyapunov time across windows (lower relative std), and Mercury-specific priority as a final tie-break.",
            "",
            "Interpretation rule of thumb: an exponential-looking metric is only a candidate if it beats a power-law/shear fit in unsaturated windows and survives duration, timestep, seed, perturbation, and metric checks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    windows = parse_windows(args.windows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = sanitize_tag(args.tag)

    fieldnames, rows = load_rows(args.shadow_csv)
    metrics = discover_metrics(fieldnames, selected_body_slugs(args.bodies))
    if not metrics:
        raise SystemExit("No matching shadow metrics found in the CSV header.")

    detail_results: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    for metric in metrics:
        samples = metric_samples(rows, metric)
        window_results = evaluate_metric(samples, metric, windows, args.saturation_threshold_au)
        detail_results.extend(window_results)
        aggregates.append(aggregate_metric(metric, window_results))

    aggregates.sort(
        key=lambda row: (
            float(row["ranking_exp_r_squared"]) if math.isfinite(float(row["ranking_exp_r_squared"])) else -math.inf,
            float(row["ranking_exp_minus_power_r_squared"]) if math.isfinite(float(row["ranking_exp_minus_power_r_squared"])) else -math.inf,
            -float(row["lyapunov_time_relative_std"]) if math.isfinite(float(row["lyapunov_time_relative_std"])) else -math.inf,
            int(row["mercury_priority"]),
            int(row["valid_window_count"]),
        ),
        reverse=True,
    )

    csv_out = output_dir / f"shadow_metric_scan_{tag}.csv"
    json_out = output_dir / f"shadow_metric_scan_{tag}.json"
    md_out = output_dir / f"shadow_metric_scan_{tag}.md"
    fieldnames_out = [
        "metric",
        "body",
        "valid_window_count",
        "exponential_candidate_window_count",
        "saturated_or_ambiguous_window_count",
        "best_window",
        "best_classification",
        "best_lambda_1_per_year",
        "best_lyapunov_time_years",
        "best_exponential_r_squared",
        "best_powerlaw_r_squared",
        "best_exp_minus_power_r_squared",
        "lambda_mean",
        "lambda_std",
        "lambda_relative_std",
        "lyapunov_time_mean_years",
        "lyapunov_time_relative_std",
        "lyapunov_time_stability_score",
        "mercury_priority",
        "ranking_exp_r_squared",
        "ranking_exp_minus_power_r_squared",
        "recommendation",
    ]
    with csv_out.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(aggregates)

    payload = {
        "diagnostic": "shadow metric scan",
        "finite_time_only": True,
        "not_asymptotic_lyapunov_exponent": True,
        "shadow_csv": str(args.shadow_csv),
        "bodies": args.bodies,
        "windows": [f"{start:.12g}:{end:.12g}" for start, end in windows],
        "ranking_criteria": [
            "ranking_exp_r_squared (higher is better)",
            "ranking_exp_minus_power_r_squared (higher is better)",
            "lyapunov_time_relative_std (lower is better)",
            "mercury_priority (higher as final tie-break)",
        ],
        "metrics_scanned": metrics,
        "ranked_metrics": aggregates,
        "window_results": detail_results,
        "outputs": {
            "csv": str(csv_out),
            "json": str(json_out),
            "markdown": str(md_out),
        },
    }
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_out, aggregates, tag)

    print(f"wrote csv: {csv_out}")
    print(f"wrote json: {json_out}")
    print(f"wrote markdown: {md_out}")
    if aggregates:
        print(f"top_metric: {aggregates[0]['metric']} ({aggregates[0]['recommendation']})")


if __name__ == "__main__":
    main()
