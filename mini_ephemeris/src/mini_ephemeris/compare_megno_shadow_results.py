from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {"value": data}


def _read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _finite_or_none(value: Any) -> float | None:
    value_f = _safe_float(value)
    return value_f if math.isfinite(value_f) else None


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 3:
        return math.nan, math.nan
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx <= 0.0:
        return math.nan, math.nan
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ss_xx
    intercept = y_mean - slope * x_mean
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return slope, r2


def megno_slope_fallback_from_csv(csv_path: Path, duration_years: float) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    samples: list[tuple[float, float]] = []
    for row in _read_csv_rows(csv_path):
        t = _safe_float(row.get("time_years"))
        y = _safe_float(row.get("megno"))
        if math.isfinite(t) and math.isfinite(y):
            samples.append((t, y))
    estimates: list[dict[str, Any]] = []
    for start in (0.0, 100_000_000.0, 200_000_000.0, 300_000_000.0):
        if duration_years <= start:
            continue
        selected = [(t, y) for t, y in samples if start <= t <= duration_years]
        if len(selected) < 3:
            continue
        slope, r2 = _linear_fit([t for t, _ in selected], [y for _, y in selected])
        if not math.isfinite(slope):
            continue
        estimates.append(
            {
                "window_start_years": start,
                "window_end_years": duration_years,
                "n_samples": len(selected),
                "megno_slope_per_year": slope,
                "lyapunov_proxy_1_per_year": max(0.0, slope) * 0.5,
                "r_squared": _finite_or_none(r2),
                "warning": "LCN accessor unavailable after resume; MEGNO slope fallback used.",
            }
        )
    return estimates


def last_finite_lcn_from_csv(csv_path: Path) -> tuple[float | None, float | None]:
    if not csv_path.exists():
        return None, None
    last_lcn: float | None = None
    last_time: float | None = None
    for row in _read_csv_rows(csv_path):
        time_years = _safe_float(row.get("time_years"))
        lcn = _safe_float(row.get("finite_time_lyapunov_estimate"))
        if math.isfinite(time_years) and math.isfinite(lcn):
            last_lcn = lcn
            last_time = time_years
    return last_lcn, last_time


def classification_with_megno_fallback(
    *,
    final_megno: Any,
    final_lcn: Any,
    duration_years: Any,
    existing: Any,
    slope_estimates: list[dict[str, Any]],
) -> str:
    final_megno_f = _safe_float(final_megno)
    final_lcn_f = _safe_float(final_lcn)
    duration_f = _safe_float(duration_years)
    if math.isfinite(final_lcn_f):
        return str(existing or "")
    best_proxy = max(
        (
            _safe_float(row.get("lyapunov_proxy_1_per_year"))
            for row in slope_estimates
        ),
        default=math.nan,
    )
    if (
        math.isfinite(final_megno_f)
        and final_megno_f > 10.0
        and math.isfinite(best_proxy)
        and best_proxy > 0.0
        and math.isfinite(duration_f)
        and best_proxy * duration_f > 1.0
    ):
        return "chaotic_candidate"
    return str(existing or "")


def collect_sources(output_dir: Path) -> dict[str, list[Path]]:
    return {
        "megno_summaries": sorted(output_dir.rglob("megno_summary_*.json")),
        "megno_ladders": sorted(output_dir.rglob("rebound_full_newtonian_megno_research_ladder.csv")),
        "shadow_summaries": sorted(output_dir.rglob("shadow_lyapunov_summary_*.json")),
        "shadow_metric_scans": sorted(output_dir.rglob("shadow_metric_scan_*.csv")),
    }


def summarize_megno(sources: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sources["megno_summaries"]:
        data = _read_json(path)
        duration_years = data.get("duration_years")
        csv_path_text = data.get("outputs", {}).get("megno_csv")
        csv_path = Path(csv_path_text) if csv_path_text else path.with_name(path.name.replace("megno_summary_", "megno_").replace(".json", ".csv"))
        slope_estimates = data.get("megno_slope_window_estimates") or []
        if not slope_estimates and _safe_float(duration_years) >= 100_000_000.0:
            slope_estimates = megno_slope_fallback_from_csv(csv_path, _safe_float(duration_years))
        last_lcn = data.get("last_finite_lcn")
        last_lcn_time = data.get("last_finite_lcn_time_years")
        if last_lcn is None or last_lcn_time is None:
            last_lcn, last_lcn_time = last_finite_lcn_from_csv(csv_path)
        final_lcn = data.get("estimated_lyapunov_if_available")
        classification = classification_with_megno_fallback(
            final_megno=data.get("final_megno"),
            final_lcn=final_lcn,
            duration_years=duration_years,
            existing=data.get("classification_hint"),
            slope_estimates=slope_estimates,
        )
        rows.append(
            {
                "path": str(path),
                "duration_years": duration_years,
                "step_days": data.get("timestep_days"),
                "integrator": data.get("integrator"),
                "gr_model": data.get("gr_model"),
                "final_megno": data.get("final_megno"),
                "final_lcn": final_lcn,
                "last_finite_lcn": last_lcn,
                "last_finite_lcn_time_years": last_lcn_time,
                "megno_slope_window_estimates": slope_estimates,
                "classification": classification,
                "classification_note": (
                    "LCN accessor unavailable after resume; MEGNO slope fallback used."
                    if classification != data.get("classification_hint")
                    else ""
                ),
            }
        )
    for path in sources["megno_ladders"]:
        for row in _read_csv_rows(path):
            rows.append(
                {
                    "path": str(path),
                    "duration_years": row.get("duration_years"),
                    "step_days": row.get("step_days"),
                    "integrator": row.get("integrator"),
                    "gr_model": row.get("gr_model"),
                    "final_megno": row.get("final_megno"),
                    "final_lcn": row.get("final_lcn"),
                    "classification": row.get("classification"),
                }
            )
    return sorted(rows, key=lambda row: (_safe_float(row.get("duration_years")), str(row.get("path"))))


def summarize_shadow(sources: dict[str, list[Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sources["shadow_summaries"]:
        data = _read_json(path)
        fit = data.get("fit") if isinstance(data.get("fit"), dict) else data
        rows.append(
            {
                "path": str(path),
                "duration_years": data.get("duration_years")
                or data.get("configuration", {}).get("duration_years"),
                "model_scope": data.get("model_scope") or data.get("configuration", {}).get("model_scope"),
                "gr_model": data.get("gr_model") or data.get("configuration", {}).get("gr_model"),
                "perturb_body": data.get("perturb_body") or data.get("perturbation", {}).get("body"),
                "lambda_1_per_year": fit.get("lambda_1_per_year"),
                "lyapunov_time_years": fit.get("lyapunov_time_years"),
                "r_squared": fit.get("r_squared"),
                "warnings": data.get("warnings"),
            }
        )
    return rows


def summarize_metric_scans(sources: dict[str, list[Path]], max_rows: int = 12) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sources["shadow_metric_scans"]:
        for row in _read_csv_rows(path):
            score = _safe_float(row.get("best_exponential_r_squared")) - _safe_float(
                row.get("best_powerlaw_r_squared")
            )
            metric = row.get("metric", "")
            mercury_bonus = 0.05 if metric.startswith("mercury") else 0.0
            candidates.append(
                {
                    "path": str(path),
                    "metric": metric,
                    "classification": row.get("classification") or row.get("recommendation"),
                    "lambda_1_per_year": row.get("best_lambda_1_per_year")
                    or row.get("lambda_1_per_year"),
                    "lyapunov_time_years": row.get("best_lyapunov_time_years")
                    or row.get("lyapunov_time_years"),
                    "exp_r2_minus_powerlaw_r2": score,
                    "rank_score": score + mercury_bonus,
                }
            )
    candidates.sort(key=lambda row: _safe_float(row.get("rank_score")), reverse=True)
    return candidates[:max_rows]


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# MEGNO and Shadow-Divergence Comparison",
        "",
        "This is a finite-time research comparison, not a Solar System Lyapunov-time claim.",
        "REBOUND-native Newtonian MEGNO/LCN is the primary variational diagnostic; shadow and secular metrics are interpretation aids.",
        "The literature-scale target of a few Myr is included only as context.",
        "",
        "## REBOUND MEGNO/LCN",
    ]
    megno_rows = summary["megno"]
    if megno_rows:
        lines.extend(
            [
                "| duration yr | step d | integrator | GR | MEGNO | LCN 1/yr | classification | note |",
                "|---:|---:|---|---|---:|---:|---|---|",
            ]
        )
        for row in megno_rows[-20:]:
            lines.append(
                f"| {row.get('duration_years', '')} | {row.get('step_days', '')} | "
                f"{row.get('integrator', '')} | {row.get('gr_model', '')} | "
                f"{row.get('final_megno', '')} | {row.get('final_lcn', '')} | "
                f"{row.get('classification', '')} | {row.get('classification_note', '')} |"
            )
    else:
        lines.append("No MEGNO summaries or ladder CSVs were found.")

    lines.extend(["", "## Shadow Divergence"])
    shadow_rows = summary["shadow"]
    if shadow_rows:
        lines.extend(
            [
                "| duration yr | scope | GR | perturb body | lambda 1/yr | time yr | R2 |",
                "|---:|---|---|---|---:|---:|---:|",
            ]
        )
        for row in shadow_rows[-20:]:
            lines.append(
                f"| {row.get('duration_years', '')} | {row.get('model_scope', '')} | "
                f"{row.get('gr_model', '')} | {row.get('perturb_body', '')} | "
                f"{row.get('lambda_1_per_year', '')} | {row.get('lyapunov_time_years', '')} | "
                f"{row.get('r_squared', '')} |"
            )
    else:
        lines.append("No shadow Lyapunov summaries were found.")

    lines.extend(["", "## Secular Shadow Metric Scan"])
    metric_rows = summary["shadow_metric_scan_best"]
    if metric_rows:
        lines.extend(
            [
                "| metric | classification | lambda 1/yr | time yr | exp R2 - powerlaw R2 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in metric_rows:
            lines.append(
                f"| {row.get('metric', '')} | {row.get('classification', '')} | "
                f"{row.get('lambda_1_per_year', '')} | {row.get('lyapunov_time_years', '')} | "
                f"{row.get('exp_r2_minus_powerlaw_r2', ''):.6g} |"
            )
    else:
        lines.append("No shadow metric scan CSVs were found.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MEGNO/LCN results near MEGNO=2 and LCN near zero are regular-looking over the sampled duration, not a proof of long-term stability.",
            "- Raw Cartesian shadow divergence can be dominated by phase shear or saturation.",
            "- Mercury secular inclination/nodal shadow metrics are useful candidates for follow-up, but they still require perturbation, timestep, duration, and seed checks.",
            "- GR MEGNO through the current REBOUNDx path remains unvalidated; use Newtonian REBOUND-native MEGNO first.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare finite-time REBOUND MEGNO/LCN and shadow-divergence outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="/home/peacelovephysics/ephemeris/output/stability",
        help="Directory to scan recursively for existing stability outputs.",
    )
    parser.add_argument("--tag", default=None, help="Tag for output files.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    tag = args.tag or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    sources = collect_sources(output_dir)
    summary = {
        "created_utc": dt.datetime.utcnow().isoformat() + "Z",
        "output_dir": str(output_dir),
        "literature_context": "Classic Solar System chaos estimates are often a few Myr, but this report does not claim reproduction.",
        "source_counts": {key: len(value) for key, value in sources.items()},
        "latest_sources": {key: str(_latest(value)) if _latest(value) else None for key, value in sources.items()},
        "megno": summarize_megno(sources),
        "shadow": summarize_shadow(sources),
        "shadow_metric_scan_best": summarize_metric_scans(sources),
        "caveats": [
            "All diagnostics are finite-time.",
            "Shadow/secular metrics are interpretation aids, not primary Lyapunov proof.",
            "GR MEGNO via REBOUNDx is not validated in the current workflow.",
        ],
    }
    json_path = output_dir / f"megno_shadow_comparison_{tag}.json"
    md_path = output_dir / f"megno_shadow_comparison_{tag}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    md_path.write_text(build_report(summary))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
