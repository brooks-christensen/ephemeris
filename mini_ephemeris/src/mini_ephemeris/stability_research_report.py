from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a cautious Markdown report for stability research runs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, default=None)
    parser.add_argument("--include-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-megno", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default=None)
    return parser


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value) -> str:
    value = f(value)
    return f"{value:.6g}" if math.isfinite(value) else ""


def collect_summaries(root: Path, tag: str | None) -> list[Path]:
    paths = sorted(root.glob("summary_*.json"))
    if tag:
        paths = [path for path in paths if tag in path.name]
    return paths


def collect_shadow_summaries(root: Path, tag: str | None) -> list[Path]:
    paths = sorted(root.glob("shadow_lyapunov_summary_*.json"))
    if tag:
        paths = [path for path in paths if tag in path.name]
    return paths


def orbital_extrema(path: Path) -> dict[str, dict[str, float]]:
    extrema: dict[str, dict[str, float]] = {}
    if not path.exists():
        return extrema
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if not body:
                continue
            target = extrema.setdefault(
                body,
                {"max_e": -math.inf, "max_i_deg": -math.inf, "max_a_au": -math.inf, "min_a_au": math.inf},
            )
            e = f(row.get("e"))
            inc = f(row.get("i_deg"))
            a = f(row.get("a_au"))
            if math.isfinite(e):
                target["max_e"] = max(target["max_e"], e)
            if math.isfinite(inc):
                target["max_i_deg"] = max(target["max_i_deg"], inc)
            if math.isfinite(a):
                target["max_a_au"] = max(target["max_a_au"], a)
                target["min_a_au"] = min(target["min_a_au"], a)
    return extrema


def min_separation(summary: dict) -> float:
    values = []
    for row in summary.get("min_separations", []):
        value = f(row.get("min_separation_au"))
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else math.nan


def lyapunov_time_from_lambda(value) -> float:
    lambda_1_per_year = f(value)
    if math.isfinite(lambda_1_per_year) and lambda_1_per_year > 0.0:
        return 1.0 / lambda_1_per_year
    return math.nan


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = [args.output_dir]
    if args.batch_dir is not None and args.batch_dir != args.output_dir:
        roots.append(args.batch_dir)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.tag or timestamp
    report_path = args.output_dir / f"stability_research_report_{label}.md"

    summary_paths: list[Path] = []
    shadow_paths: list[Path] = []
    for root in roots:
        summary_paths.extend(collect_summaries(root, args.tag))
        shadow_paths.extend(collect_shadow_summaries(root, args.tag))
    summary_paths = sorted(set(summary_paths))
    shadow_paths = sorted(set(shadow_paths))

    lines: list[str] = []
    lines.append(f"# Stability Research Report: {label}")
    lines.append("")
    lines.append("This report uses cautious finite-time language. Regular-looking behavior over a run duration is not a proof of long-term stability.")
    lines.append("")
    lines.append("## Model And Backend Summary")
    if not summary_paths and not shadow_paths:
        lines.append("")
        lines.append("No standard or shadow summaries were found for the requested inputs.")
    for path in summary_paths:
        summary = load_json(path)
        config = summary.get("configuration", {})
        runtime = summary.get("runtime", {})
        lines.append("")
        lines.append(f"- `{path.name}`: backend `{config.get('backend')}`, integrator `{config.get('active_integrator')}`, GR `{config.get('active_gr_model')}`, model `{config.get('model_scope')}`, duration `{fmt(config.get('duration_years_actual'))}` yr, runtime `{fmt(runtime.get('wall_clock_seconds'))}` s.")
    for path in shadow_paths:
        shadow = load_json(path)
        fit = shadow.get("fit", {})
        lines.append("")
        lines.append(
            f"- `{path.name}`: finite-time shadow divergence, model `{shadow.get('model_scope')}`, integrator `{shadow.get('integrator')}`, "
            f"GR `{shadow.get('gr_model')}`, duration `{fmt(shadow.get('duration_years'))}` yr, "
            f"lambda `{fmt(fit.get('lambda_1_per_year'))}` 1/yr, runtime `{fmt(shadow.get('runtime_seconds'))}` s."
        )

    lines.append("")
    lines.append("## Run Table")
    lines.append("")
    lines.append("| tag | backend | integrator | GR | duration yr | step d | max dE/E | max dL/L | min sep AU |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for path in summary_paths:
        summary = load_json(path)
        config = summary.get("configuration", {})
        extrema = summary.get("diagnostic_extrema_over_records", {})
        lines.append(
            f"| {config.get('tag', path.stem)} | {config.get('backend', '')} | {config.get('active_integrator', '')} | "
            f"{config.get('active_gr_model', '')} | {fmt(config.get('duration_years_actual'))} | "
            f"{fmt(config.get('step_days'))} | {fmt(extrema.get('max_abs_energy_rel_drift'))} | "
            f"{fmt(extrema.get('max_angular_momentum_rel_drift'))} | {fmt(min_separation(summary))} |"
        )
    for path in shadow_paths:
        shadow = load_json(path)
        fit = shadow.get("fit", {})
        lines.append(
            f"| {path.stem.replace('shadow_lyapunov_summary_', '')} | shadow | {shadow.get('integrator', '')} | {shadow.get('gr_model', '')} | "
            f"{fmt(shadow.get('duration_years'))} | {fmt(shadow.get('step_days'))} |  |  |  |"
        )

    megno_rows: list[str] = []
    if args.include_megno:
        for path in summary_paths:
            summary = load_json(path)
            config = summary.get("configuration", {})
            megno = summary.get("rebound_megno", {})
            if not megno.get("enabled"):
                continue
            caveat = "GR MEGNO not validated in current REBOUNDx path" if config.get("active_gr_model") != "none" else "finite-time diagnostic"
            megno_rows.append(
                f"| {config.get('tag', path.stem)} | {fmt(megno.get('final_megno'))} | "
                f"{fmt(megno.get('estimated_lyapunov_if_available'))} | {megno.get('classification_hint', '')} | {caveat} |"
            )
    if megno_rows:
        lines.append("")
        lines.append("## MEGNO / LCN Summary")
        lines.append("")
        lines.append("These are finite-time diagnostics, not asymptotic Lyapunov exponents.")
        lines.append("")
        lines.append("| tag | final MEGNO | final LCN 1/yr | classification | caveat |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        lines.extend(megno_rows)

    if shadow_paths:
        lines.append("")
        lines.append("## Shadow-Trajectory Divergence")
        lines.append("")
        lines.append("These are finite-time shadow divergence results, not asymptotic Lyapunov exponents. Late samples may be saturated, and alternate perturbations should be checked before claiming a robust Lyapunov time.")
        lines.append("")
        lines.append("| tag | duration yr | model | integrator | GR | perturbation | fit window yr | lambda 1/yr | Lyapunov time yr | r2 | checkpoints | warnings |")
        lines.append("| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
        for path in shadow_paths:
            shadow = load_json(path)
            fit = shadow.get("fit", {})
            perturb = f"{shadow.get('perturb_body')} {shadow.get('perturbation_m')} m {shadow.get('perturbation_mode')}"
            window = f"{fmt(shadow.get('fit_start_years'))}-{fmt(shadow.get('fit_end_years'))}"
            lyap_time = fit.get("lyapunov_time_years")
            if not math.isfinite(f(lyap_time)):
                lyap_time = lyapunov_time_from_lambda(fit.get("lambda_1_per_year"))
            lines.append(
                f"| {path.stem.replace('shadow_lyapunov_summary_', '')} | {fmt(shadow.get('duration_years'))} | {shadow.get('model_scope', '')} | "
                f"{shadow.get('integrator', '')} | {shadow.get('gr_model', '')} | {perturb} | {window} | "
                f"{fmt(fit.get('lambda_1_per_year'))} | {fmt(lyap_time)} | {fmt(fit.get('r_squared'))} | "
                f"{fmt(shadow.get('number_of_checkpoints_written'))} | {'; '.join(shadow.get('warnings', []))} |"
            )
            lines.append(
                f"Checkpointing details for `{path.stem.replace('shadow_lyapunov_summary_', '')}`: "
                f"enabled `{shadow.get('checkpointing_enabled')}`, cadence `{fmt(shadow.get('checkpoint_every_years'))}` yr, "
                f"resumed `{shadow.get('resumed_from_checkpoint')}`, resumed-from-time `{fmt(shadow.get('resumed_from_time_years'))}` yr, "
                f"latest checkpoint `{shadow.get('latest_checkpoint')}`, output rows `{fmt(shadow.get('output_rows'))}`, "
                f"runtime `{fmt(shadow.get('runtime_seconds'))}` s."
            )
            diagnostics_json = path.with_name(
                path.name.replace("shadow_lyapunov_summary_", "shadow_fit_diagnostics_")
            )
            diagnostics = load_json(diagnostics_json)
            if diagnostics:
                best_exp = diagnostics.get("best_exponential_candidate")
                best_power = diagnostics.get("best_powerlaw_or_shear_candidate")
                saturated = diagnostics.get("saturated_or_ambiguous_windows", [])
                lines.append("")
                lines.append(f"Fit diagnostics for `{path.stem.replace('shadow_lyapunov_summary_', '')}`:")
                if best_exp:
                    lines.append(
                        f"- Best exponential candidate window: `{best_exp.get('window_label')}` with r^2 `{fmt(best_exp.get('r_squared_exponential'))}` and lambda `{fmt(best_exp.get('lambda_1_per_year'))}` 1/yr."
                    )
                if best_power:
                    lines.append(
                        f"- Competing power-law/shear window: `{best_power.get('window_label')}` with power-law r^2 `{fmt(best_power.get('r_squared_powerlaw'))}`."
                    )
                if saturated:
                    lines.append(
                        f"- Saturated or ambiguous windows: `{', '.join(str(item) for item in saturated)}`."
                    )
                lines.append(
                    f"- Recommended interpretation: {diagnostics.get('recommended_interpretation', 'finite-time shadow divergence only').rstrip('.')}."
                )

    lines.append("")
    lines.append("## Eccentricity / Inclination Extrema")
    for path in summary_paths:
        summary = load_json(path)
        config = summary.get("configuration", {})
        elements_path = Path(summary.get("outputs", {}).get("orbital_elements", ""))
        extrema = orbital_extrema(elements_path)
        if not extrema:
            continue
        lines.append("")
        lines.append(f"### {config.get('tag', path.stem)}")
        lines.append("")
        lines.append("| body | max e | max i deg | min a AU | max a AU |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for body, values in sorted(extrema.items()):
            lines.append(
                f"| {body} | {fmt(values.get('max_e'))} | {fmt(values.get('max_i_deg'))} | "
                f"{fmt(values.get('min_a_au'))} | {fmt(values.get('max_a_au'))} |"
            )

    if args.include_frequency:
        freq_paths = []
        for root in roots:
            freq_paths.extend(sorted(root.glob("secular_frequency_summary_*.json")))
            freq_paths.extend(sorted(root.glob("frequency_map_*.csv")))
        if args.tag:
            freq_paths = [path for path in freq_paths if args.tag in path.name]
        lines.append("")
        lines.append("## Secular Frequency Summary")
        if freq_paths:
            lines.append("")
            for path in sorted(set(freq_paths)):
                lines.append(f"- `{path}`")
        else:
            lines.append("")
            lines.append("No secular frequency summaries were found.")

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Reduced model: Earth-Moon barycenter, point masses, no asteroids.")
    lines.append("- Regular-looking over this duration is not a proof of long-term stability.")
    lines.append("- MEGNO/LCN entries are finite-time diagnostics.")
    lines.append("- GR MEGNO is not validated in the current REBOUNDx path because REBOUNDx does not evolve variational particles self-consistently there.")
    lines.append("- Frequency summaries are FFT-lite / NAFF-lite, not full Laskar NAFF.")
    lines.append("")
    lines.append("## Recommended Next Experiment")
    lines.append("")
    lines.append("Run the smallest missing duration/timestep/seed comparison that would test whether the current regular-looking result is reproducible before increasing duration or adding ensembles.")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
