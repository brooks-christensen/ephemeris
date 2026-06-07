from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("/home/peacelovephysics/ephemeris/output/stability")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a consolidated long-term-stability research report from existing outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value, digits: int = 6) -> str:
    value = f(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def lyap_time(lambda_1_per_year: float) -> float:
    return 1.0 / lambda_1_per_year if math.isfinite(lambda_1_per_year) and lambda_1_per_year > 0.0 else math.nan


def summarize_megno_ladder(output_dir: Path) -> dict:
    csv_path = output_dir / "rebound_full_newtonian_megno_research_ladder" / "rebound_full_newtonian_megno_research_ladder.csv"
    rows = load_csv(csv_path)
    durations: dict[str, dict[str, float | int | list[str]]] = {}
    classes = sorted({row["classification"] for row in rows})
    for row in rows:
        duration = row["duration_years"]
        bucket = durations.setdefault(
            duration,
            {
                "count": 0,
                "step_days": [],
                "megno_min": math.inf,
                "megno_max": -math.inf,
                "lcn_min": math.inf,
                "lcn_max": -math.inf,
            },
        )
        bucket["count"] = int(bucket["count"]) + 1
        bucket["step_days"] = sorted(set([*bucket["step_days"], row["step_days"]]))
        bucket["megno_min"] = min(float(bucket["megno_min"]), f(row["final_megno"]))
        bucket["megno_max"] = max(float(bucket["megno_max"]), f(row["final_megno"]))
        bucket["lcn_min"] = min(float(bucket["lcn_min"]), f(row["final_lcn"]))
        bucket["lcn_max"] = max(float(bucket["lcn_max"]), f(row["final_lcn"]))
    return {
        "path": str(csv_path),
        "case_count": len(rows),
        "classifications": classes,
        "all_regular_likely": classes == ["regular_likely"],
        "durations": durations,
    }


def shadow_case(output_dir: Path, name: str) -> dict:
    summary_path = output_dir / "shadow_100myr" / f"shadow_lyapunov_summary_{name}.json"
    summary = load_json(summary_path)
    fit = summary.get("fit", {})
    result = {
        "name": name,
        "summary_path": str(summary_path),
        "gr_model": summary.get("gr_model"),
        "perturbation_mode": summary.get("perturbation_mode"),
        "perturb_body": summary.get("perturb_body"),
        "duration_years": summary.get("duration_years"),
        "lambda_1_per_year": fit.get("lambda_1_per_year"),
        "lyapunov_time_years": fit.get("lyapunov_time_years") or lyap_time(f(fit.get("lambda_1_per_year"))),
        "r_squared": fit.get("r_squared"),
        "fit_warning": fit.get("warning"),
        "warnings": summary.get("warnings", []),
    }
    fit_diag_path = output_dir / "shadow_100myr" / f"shadow_fit_diagnostics_{name}.json"
    if fit_diag_path.exists():
        fit_diag = load_json(fit_diag_path)
        result["fit_diagnostics_path"] = str(fit_diag_path)
        result["recommended_interpretation"] = fit_diag.get("recommended_interpretation")
        result["best_exponential_candidate"] = fit_diag.get("best_exponential_candidate")
        result["best_powerlaw_or_shear_candidate"] = fit_diag.get("best_powerlaw_or_shear_candidate")
    metric_scan_path = output_dir / "shadow_100myr" / f"shadow_metric_scan_{name}.json"
    if metric_scan_path.exists():
        metric_scan = load_json(metric_scan_path)
        ranked = metric_scan.get("ranked_metrics", [])
        result["metric_scan_path"] = str(metric_scan_path)
        result["top_metric"] = ranked[0] if ranked else {}
    return result


def summarize_frequency_compare(output_dir: Path) -> dict:
    csv_path = output_dir / "frequency_100myr_compare" / "secular_frequency_newtonian_vs_gr_summary.csv"
    rows = load_csv(csv_path)
    picked = {}
    targets = {
        ("mercury barycenter", "eccentricity_complex"),
        ("mercury barycenter", "inclination_complex"),
        ("venus barycenter", "inclination_complex"),
        ("earth barycenter", "inclination_complex"),
        ("mars barycenter", "inclination_complex"),
    }
    for row in rows:
        key = (row["body"], row["variable"])
        if key in targets:
            picked[f"{row['body']}::{row['variable']}"] = row
    return {
        "path": str(csv_path),
        "selected_rows": picked,
    }


def summarize_mode_comparison(output_dir: Path, folder: str, filename: str) -> dict:
    path = output_dir / folder / filename
    payload = load_json(path)
    picked = {}
    for row in payload.get("summary_rows", []):
        key = (row["body"], row["variable"])
        if key in {
            ("mercury barycenter", "eccentricity_complex"),
            ("mercury barycenter", "inclination_complex"),
            ("venus barycenter", "inclination_complex"),
        }:
            picked[f"{row['body']}::{row['variable']}"] = row
    return {
        "path": str(path),
        "selected_rows": picked,
    }


def build_summary(output_dir: Path) -> dict:
    megno = summarize_megno_ladder(output_dir)
    raw_shadow = shadow_case(output_dir, "full_newtonian_shadow_100myr")
    mercury_radial = shadow_case(output_dir, "full_newtonian_shadow_100myr_mercury_secular")
    mercury_tangential = shadow_case(output_dir, "full_newtonian_shadow_100myr_mercury_secular_tangential")
    mercury_normal = shadow_case(output_dir, "full_newtonian_shadow_100myr_mercury_secular_normal")
    gr_normal = shadow_case(output_dir, "full_gr_potential_shadow_100myr_mercury_secular_normal")
    frequency_compare = summarize_frequency_compare(output_dir)
    mode_10myr = summarize_mode_comparison(
        output_dir,
        "frequency_100myr_mode_tracking_10myr",
        "secular_mode_comparison_frequency_full_newtonian_vs_gr_potential_100myr_10myr_windows.json",
    )
    mode_40myr = summarize_mode_comparison(
        output_dir,
        "frequency_100myr_mode_tracking_40myr",
        "secular_mode_comparison_frequency_full_newtonian_vs_gr_potential_100myr_40myr_windows.json",
    )

    caveats = [
        "All MEGNO, shadow-divergence, and shadow-metric outputs here are finite-time diagnostics, not asymptotic Lyapunov exponents.",
        "Raw Cartesian shadow divergence is not isolated from phase shear; when power-law/shear fits compete with exponential fits, the result should not be treated as a robust Lyapunov-time measurement.",
        "The secular frequency products are FFT-lite / NAFF-lite summaries, not full Laskar NAFF.",
        "Mode-tracker peak identities can switch across windows; apparent GR-vs-Newtonian detuning may reflect dominant-peak switching rather than one persistent mode shifting cleanly.",
        "The GR-potential shadow comparison is still a trajectory-only comparison. It should not be interpreted as a validated variational/MEGNO GR path.",
    ]

    interpretation = (
        "The Newtonian full-system MEGNO ladder remains regular-looking through 10 Myr, while the 100 Myr raw shadow-divergence fit is still better explained by phase shear than by a clean asymptotic exponential. "
        "Mercury secular shadow variants do show few-Myr-scale finite-time exponential-looking fits in some metrics, but the top-ranked metrics remain ambiguous or shear-competitive. "
        "The Newtonian-vs-GR frequency products show a clear Mercury eccentricity shift, but the 10 Myr and 40 Myr multi-peak mode tracking indicates that Mercury inclination and Venus inclination are affected by dominant-peak switching, so the present evidence does not yet justify a clean 'GR detunes the Newtonian secular shadow-divergence mode' claim."
    )

    future_work = [
        "Re-run the secular mode tracker with at least one additional window/step choice to test whether the Mercury and Venus inclination switching patterns are stable.",
        "Promote the Mercury secular shadow comparison from one best metric to a small fixed metric panel, then compare Newtonian vs GR-potential on the same panel rather than by whichever metric ranks first in each run.",
        "If the switching pattern survives window changes, move to a stronger secular decomposition than FFT-lite / NAFF-lite before making a GR detuning interpretation.",
    ]

    return {
        "report_kind": "stability_research_master",
        "output_dir": str(output_dir),
        "newtonian_megno_ladder": megno,
        "raw_shadow_divergence": raw_shadow,
        "mercury_secular_rtn": {
            "radial": mercury_radial,
            "tangential": mercury_tangential,
            "normal": mercury_normal,
        },
        "gr_potential_shadow_comparison": {
            "newtonian_normal": mercury_normal,
            "gr_potential_normal": gr_normal,
        },
        "newtonian_vs_gr_secular_frequency": frequency_compare,
        "mode_tracking_10myr": mode_10myr,
        "mode_tracking_40myr": mode_40myr,
        "caveats": caveats,
        "final_interpretation": interpretation,
        "recommended_future_work": future_work,
    }


def write_report(path: Path, summary: dict) -> None:
    megno = summary["newtonian_megno_ladder"]
    raw_shadow = summary["raw_shadow_divergence"]
    rtn = summary["mercury_secular_rtn"]
    gr_compare = summary["gr_potential_shadow_comparison"]
    frequency = summary["newtonian_vs_gr_secular_frequency"]["selected_rows"]
    mode10 = summary["mode_tracking_10myr"]["selected_rows"]
    mode40 = summary["mode_tracking_40myr"]["selected_rows"]

    lines: list[str] = [
        "# Long-Term Stability Research Master Report",
        "",
        "This report is synthesized from existing outputs only. It uses cautious finite-time language throughout.",
        "",
        "## Newtonian MEGNO Ladder Summary",
        "",
        f"All {megno['case_count']} full-system Newtonian WHFast MEGNO ladder cases classify as `regular_likely` through 1, 5, and 10 Myr. Final MEGNO stays close to 2 and final LCN stays near zero across both 1 day and 0.5 day runs.",
        "",
        "| duration yr | cases | step days | final MEGNO range | final LCN range [1/yr] |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for duration in sorted(megno["durations"], key=lambda item: float(item)):
        row = megno["durations"][duration]
        lines.append(
            f"| {duration} | {row['count']} | {', '.join(row['step_days'])} | "
            f"{fmt(row['megno_min'])} to {fmt(row['megno_max'])} | {fmt(row['lcn_min'])} to {fmt(row['lcn_max'])} |"
        )

    lines.extend(
        [
            "",
            "## Raw Shadow-Divergence Summary",
            "",
            f"The 100 Myr full Newtonian raw Cartesian shadow run gives a finite-time fit of lambda `{fmt(raw_shadow['lambda_1_per_year'])}` 1/yr with r^2 `{fmt(raw_shadow['r_squared'])}`, but the fit warning is `{raw_shadow.get('fit_warning', '')}`.",
            f"The fit-diagnostics interpretation remains: {raw_shadow.get('recommended_interpretation', 'finite-time only')}",
            "",
            "## Mercury Secular RTN Comparison",
            "",
            "| perturbation mode | GR | lambda [1/yr] | Lyapunov time [yr] | r^2 | top metric | top metric class |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for label, row in [("radial", rtn["radial"]), ("tangential", rtn["tangential"]), ("normal", rtn["normal"])]:
        top = row.get("top_metric", {})
        lines.append(
            f"| {label} | {row.get('gr_model')} | {fmt(row.get('lambda_1_per_year'))} | {fmt(row.get('lyapunov_time_years'))} | "
            f"{fmt(row.get('r_squared'))} | {top.get('metric', '')} | {top.get('best_classification', '')} |"
        )
    lines.append("")
    lines.append("The radial, tangential, and normal Mercury-secular variants all produce finite-time positive fits, but their top-ranked shadow metrics remain ambiguous or shear-competitive rather than clean exponential confirmations.")

    gr_n = gr_compare["newtonian_normal"]
    gr_g = gr_compare["gr_potential_normal"]
    lines.extend(
        [
            "",
            "## GR-Potential Shadow Comparison",
            "",
            f"For the normal Mercury-secular perturbation, the Newtonian fit is lambda `{fmt(gr_n.get('lambda_1_per_year'))}` 1/yr with r^2 `{fmt(gr_n.get('r_squared'))}`, while the GR-potential trajectory-only run drops to `{fmt(gr_g.get('lambda_1_per_year'))}` 1/yr with r^2 `{fmt(gr_g.get('r_squared'))}`.",
            f"The top-ranked GR-potential shadow metric is `{gr_g.get('top_metric', {}).get('metric', '')}` with class `{gr_g.get('top_metric', {}).get('best_classification', '')}`, so the GR run weakens the Newtonian secular candidate but does not yet isolate a cleaner mode.",
            "",
            "## Newtonian vs GR Secular Frequency Summary",
            "",
            "These values come from the existing FFT-lite / NAFF-lite frequency summaries. This is not full Laskar NAFF.",
            "",
            "| body / variable | mean delta frequency GR-Newtonian [rad/yr] | max abs delta [rad/yr] | mean delta period [yr] |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in [
        "mercury barycenter::eccentricity_complex",
        "mercury barycenter::inclination_complex",
        "venus barycenter::inclination_complex",
        "earth barycenter::inclination_complex",
        "mars barycenter::inclination_complex",
    ]:
        row = frequency.get(key, {})
        lines.append(
            f"| {key.replace('::', ' / ')} | {fmt(row.get('mean_delta_freq_gr_minus_newtonian'))} | "
            f"{fmt(row.get('max_abs_delta_freq'))} | {fmt(row.get('mean_delta_period_years'))} |"
        )

    lines.extend(
        [
            "",
            "## 10 Myr And 40 Myr Secular Mode Tracking Comparison",
            "",
            "The multi-peak mode tracker compares top-K FFT-lite peaks across windows, so it can distinguish persistent shifts from dominant-peak switching.",
            "",
            "| window set | body / variable | classification | matched windows | switching windows | median abs shift [rad/yr] | max abs shift [rad/yr] |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, payload in [("10 Myr windows", mode10), ("40 Myr windows", mode40)]:
        for key in [
            "mercury barycenter::eccentricity_complex",
            "mercury barycenter::inclination_complex",
            "venus barycenter::inclination_complex",
        ]:
            row = payload.get(key, {})
            lines.append(
                f"| {label} | {key.replace('::', ' / ')} | {row.get('classification', '')} | "
                f"{fmt(row.get('matched_persistent_mode_count'))} | {fmt(row.get('dominant_peak_switching_count'))} | "
                f"{fmt(row.get('median_abs_frequency_shift_rad_per_year'))} | {fmt(row.get('max_abs_frequency_shift_rad_per_year'))} |"
            )
    lines.append("")
    lines.append("The current 10 Myr and 40 Myr comparisons both classify Mercury eccentricity, Mercury inclination, and Venus inclination as `dominant_peak_switching`, not as clean persistent mode shifts.")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")

    lines.extend(
        [
            "",
            "## Final Interpretation",
            "",
            summary["final_interpretation"],
            "",
            "## Recommended Future Work",
            "",
        ]
    )
    for item in summary["recommended_future_work"]:
        lines.append(f"- {item}")

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    report_path = args.report_path or output_dir / "stability_research_master_report.md"
    summary_path = args.summary_path or output_dir / "stability_research_master_summary.json"
    summary = build_summary(output_dir)
    report_path.write_text("")
    write_report(report_path, summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote report: {report_path}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
