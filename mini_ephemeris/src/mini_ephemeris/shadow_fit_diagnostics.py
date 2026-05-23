from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare exponential and power-law fits for shadow-separation windows."
    )
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        default="raw_position_separation_au",
        help="Shadow CSV column to fit. Values are fit using log(abs(metric)).",
    )
    parser.add_argument(
        "--windows",
        default="1e6:1e7,1e6:2e7,2e6:2e7,5e6:3e7,1e7:4e7,1e6:5e7",
        help="Comma-separated start:end year windows.",
    )
    parser.add_argument("--saturation-threshold-au", type=float, default=0.1)
    return parser


def parse_windows(text: str) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        start_text, end_text = item.split(":", 1)
        start = float(start_text)
        end = float(end_text)
        if end <= start:
            raise ValueError(f"window end must exceed start: {item}")
        windows.append((start, end))
    if not windows:
        raise ValueError("no windows parsed")
    return windows


def f(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    coeff = np.polyfit(xs, ys, 1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = slope * xs + intercept
    ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
    ss_res = float(np.sum((ys - pred) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r_squared


def classify_window(
    *,
    sample_count: int,
    end_sep_au: float,
    saturated: bool,
    exp_r2: float,
    power_r2: float,
    lambda_1_per_year: float,
) -> str:
    if sample_count < 3:
        return "insufficient_data"
    if saturated:
        return "saturated_or_ambiguous"
    if math.isfinite(exp_r2) and math.isfinite(power_r2):
        if exp_r2 >= 0.85 and exp_r2 >= power_r2 + 0.03 and lambda_1_per_year > 0.0:
            return "exponential_candidate"
        if power_r2 >= 0.85 and power_r2 >= exp_r2 - 0.01:
            return "powerlaw_or_shear_candidate"
    return "saturated_or_ambiguous"


def saturation_applies_to_metric(metric: str) -> bool:
    return metric == "raw_position_separation_au" or (metric.startswith("sep_") and metric.endswith("_au"))


def plot_window_comparison(rows: list[dict[str, object]], path: Path, metric: str) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(row["window_label"]) for row in rows]
    x_values = np.arange(len(rows))
    exp_r2 = [float(row["r_squared_exponential"]) for row in rows]
    power_r2 = [float(row["r_squared_powerlaw"]) for row in rows]
    lambdas = [float(row["lambda_1_per_year"]) if math.isfinite(float(row["lambda_1_per_year"])) else math.nan for row in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(x_values, exp_r2, marker="o", label="exp r^2")
    ax1.plot(x_values, power_r2, marker="s", label="power-law r^2")
    ax1.set_ylabel("fit quality")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x_values, lambdas, marker="o", color="tab:red")
    ax2.set_ylabel("lambda [1/year]")
    ax2.set_xlabel("fit window")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_values, labels, rotation=35, ha="right")
    fig.suptitle(f"Shadow fit comparison: {metric}", y=0.995)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    windows = parse_windows(args.windows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in args.tag)
    csv_out = output_dir / f"shadow_fit_diagnostics_{tag}.csv"
    json_out = output_dir / f"shadow_fit_diagnostics_{tag}.json"
    plot_out = output_dir / f"shadow_fit_window_comparison_{tag}.png"

    metric = args.metric
    saturation_metric = saturation_applies_to_metric(metric)
    samples: list[dict[str, float]] = []
    with args.shadow_csv.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise SystemExit(f"Metric column '{metric}' not found in {args.shadow_csv}. Available columns: {available}")
        for row in reader:
            time_years = f(row.get("time_years"))
            metric_value = f(row.get(metric))
            metric_magnitude = abs(metric_value)
            if (
                math.isfinite(time_years)
                and math.isfinite(metric_value)
                and metric_magnitude > 0.0
            ):
                samples.append(
                    {
                        "time_years": time_years,
                        "metric_value": metric_value,
                        "metric_magnitude": metric_magnitude,
                        "log_metric": math.log(max(metric_magnitude, 1.0e-300)),
                    }
                )
    if not samples:
        raise SystemExit(f"No finite nonzero rows were found for metric '{metric}'.")

    rows: list[dict[str, object]] = []
    for start, end in windows:
        window_samples = [sample for sample in samples if start <= sample["time_years"] <= end]
        sample_count = len(window_samples)
        window_label = f"{start:.6g}:{end:.6g}"
        if sample_count < 3:
            rows.append(
                {
                    "window_label": window_label,
                    "window_start_years": start,
                    "window_end_years": end,
                    "lambda_1_per_year": math.nan,
                    "lyapunov_time_years": math.nan,
                    "r_squared_exponential": math.nan,
                    "r_squared_powerlaw": math.nan,
                    "number_of_samples": sample_count,
                    "metric": metric,
                    "start_metric_value": math.nan,
                    "end_metric_value": math.nan,
                    "start_metric_magnitude": math.nan,
                    "end_metric_magnitude": math.nan,
                    "start_separation_au": math.nan,
                    "end_separation_au": math.nan,
                    "exceeded_saturation_threshold": False,
                    "classification": "insufficient_data",
                    "recommendation": "Window has too few samples for reliable finite-time fitting.",
                }
            )
            continue

        xs = np.array([sample["time_years"] for sample in window_samples], dtype=float)
        ys = np.array([sample["log_metric"] for sample in window_samples], dtype=float)
        exp_slope, _, exp_r2 = linear_fit(xs, ys)

        positive_samples = [sample for sample in window_samples if sample["time_years"] > 0.0]
        power_r2 = math.nan
        power_slope = math.nan
        if len(positive_samples) >= 3:
            power_xs = np.log(np.array([sample["time_years"] for sample in positive_samples], dtype=float))
            power_ys = np.array([sample["log_metric"] for sample in positive_samples], dtype=float)
            power_slope, _, power_r2 = linear_fit(power_xs, power_ys)

        start_metric_value = float(window_samples[0]["metric_value"])
        end_metric_value = float(window_samples[-1]["metric_value"])
        start_metric_magnitude = float(window_samples[0]["metric_magnitude"])
        end_metric_magnitude = float(window_samples[-1]["metric_magnitude"])
        saturated = (
            any(sample["metric_magnitude"] >= args.saturation_threshold_au for sample in window_samples)
            if saturation_metric
            else False
        )
        classification = classify_window(
            sample_count=sample_count,
            end_sep_au=end_metric_magnitude,
            saturated=saturated,
            exp_r2=exp_r2,
            power_r2=power_r2,
            lambda_1_per_year=exp_slope,
        )
        if classification == "exponential_candidate":
            recommendation = "Finite-time exponential fit looks stronger than the power-law/shear fit in this unsaturated window."
        elif classification == "powerlaw_or_shear_candidate":
            recommendation = "Power-law/shear fit competes with or exceeds the exponential fit; do not treat this window as a robust Lyapunov estimate."
        elif classification == "saturated_or_ambiguous":
            recommendation = "Late samples may be saturated or the window is mixed; avoid using it for a final Lyapunov-time claim."
        else:
            recommendation = "Window is too sparse for a useful finite-time comparison."

        rows.append(
            {
                "window_label": window_label,
                "window_start_years": start,
                "window_end_years": end,
                "lambda_1_per_year": exp_slope,
                "lyapunov_time_years": (1.0 / exp_slope) if exp_slope > 0.0 else math.nan,
                "r_squared_exponential": exp_r2,
                "r_squared_powerlaw": power_r2,
                "powerlaw_exponent": power_slope,
                "number_of_samples": sample_count,
                "metric": metric,
                "start_metric_value": start_metric_value,
                "end_metric_value": end_metric_value,
                "start_metric_magnitude": start_metric_magnitude,
                "end_metric_magnitude": end_metric_magnitude,
                "start_separation_au": start_metric_magnitude,
                "end_separation_au": end_metric_magnitude,
                "exceeded_saturation_threshold": saturated,
                "classification": classification,
                "recommendation": recommendation,
            }
        )

    fieldnames = [
        "window_label",
        "window_start_years",
        "window_end_years",
        "lambda_1_per_year",
        "lyapunov_time_years",
        "r_squared_exponential",
        "r_squared_powerlaw",
        "powerlaw_exponent",
        "number_of_samples",
        "metric",
        "start_metric_value",
        "end_metric_value",
        "start_metric_magnitude",
        "end_metric_magnitude",
        "start_separation_au",
        "end_separation_au",
        "exceeded_saturation_threshold",
        "classification",
        "recommendation",
    ]
    with csv_out.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exp_candidates = [
        row for row in rows
        if row["classification"] == "exponential_candidate"
    ]
    power_candidates = [
        row for row in rows
        if row["classification"] == "powerlaw_or_shear_candidate"
    ]
    saturated = [
        row for row in rows
        if row["classification"] == "saturated_or_ambiguous"
    ]
    best_exp = max(exp_candidates, key=lambda row: float(row["r_squared_exponential"]), default=None)
    best_power = max(power_candidates, key=lambda row: float(row["r_squared_powerlaw"]), default=None)
    if best_exp is not None and (
        best_power is None
        or float(best_exp["r_squared_exponential"]) >= float(best_power["r_squared_powerlaw"]) + 0.03
    ):
        interpretation = (
            "Some unsaturated windows look more exponential than power-law, but this remains a finite-time shadow divergence result, not an asymptotic Lyapunov exponent."
        )
    elif best_power is not None:
        interpretation = (
            "Power-law/shear fits compete with or exceed exponential fits in the tested windows, so phase shear remains a serious alternative interpretation."
        )
    else:
        interpretation = (
            "No robust unsaturated exponential window emerged from the requested comparisons."
        )

    summary = {
        "diagnostic": "shadow fit diagnostics",
        "finite_time_only": True,
        "not_asymptotic_lyapunov_exponent": True,
        "shadow_csv": str(args.shadow_csv),
        "metric": metric,
        "saturation_threshold_au": args.saturation_threshold_au,
        "saturation_threshold_applied": saturation_metric,
        "windows": rows,
        "best_exponential_candidate": best_exp,
        "best_powerlaw_or_shear_candidate": best_power,
        "saturated_or_ambiguous_windows": [
            row["window_label"] for row in saturated
        ],
        "recommended_interpretation": interpretation,
        "outputs": {
            "csv": str(csv_out),
            "json": str(json_out),
            "plot": str(plot_out),
        },
    }
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_window_comparison(rows, plot_out, metric)

    print(f"wrote csv: {csv_out}")
    print(f"wrote json: {json_out}")
    print(f"wrote plot: {plot_out}")
    print(f"recommended_interpretation: {interpretation}")


if __name__ == "__main__":
    main()
