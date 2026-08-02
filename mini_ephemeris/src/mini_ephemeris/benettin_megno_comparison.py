from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


def f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def classification_from_lcn(lcn: float, duration_years: float) -> str:
    if not math.isfinite(lcn):
        return "ambiguous"
    if lcn > 0.0 and math.isfinite(duration_years) and lcn * duration_years > 1.0:
        return "chaotic_candidate"
    return "regular_likely"


def key_from_config(config: dict[str, Any], duration: float, seed: Any) -> tuple[str, float, int | None]:
    seed_int = None
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        pass
    return str(config.get("model_scope", "")), float(duration), seed_int


def collect_benettin(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("benettin_summary_*.json")):
        data = load_json(path)
        config = data.get("configuration", {})
        duration = f(data.get("duration_years") or config.get("duration_years"))
        lcn = f(data.get("finite_time_lcn_1_per_year"))
        rows.append(
            {
                "path": str(path),
                "model_scope": config.get("model_scope"),
                "duration_years": duration,
                "seed": config.get("seed"),
                "gr_model": config.get("gr_model"),
                "integrator": config.get("integrator"),
                "lcn_1_per_year": lcn,
                "classification": data.get("classification_hint")
                or classification_from_lcn(lcn, duration),
            }
        )
    return rows


def collect_megno(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("megno_summary_*.json")):
        data = load_json(path)
        duration = f(data.get("duration_years"))
        lcn = f(
            data.get("estimated_lyapunov_if_available")
            or data.get("last_finite_lcn")
            or data.get("best_megno_slope_fallback_1_per_year")
        )
        rows.append(
            {
                "path": str(path),
                "model_scope": data.get("model_scope") or "full",
                "duration_years": duration,
                "seed": data.get("seed") or data.get("configuration", {}).get("seed"),
                "gr_model": data.get("gr_model"),
                "integrator": data.get("integrator"),
                "lcn_1_per_year": lcn,
                "classification": data.get("classification_hint")
                or classification_from_lcn(lcn, duration),
            }
        )
    return rows


def compare(output_dir: Path) -> list[dict[str, Any]]:
    megno = collect_megno(output_dir)
    megno_by_key = {
        key_from_config(row, row["duration_years"], row.get("seed")): row
        for row in megno
        if math.isfinite(row["duration_years"])
    }
    rows = []
    for benettin in collect_benettin(output_dir):
        key = key_from_config(benettin, benettin["duration_years"], benettin.get("seed"))
        native = megno_by_key.get(key)
        b_lcn = f(benettin.get("lcn_1_per_year"))
        n_lcn = f(native.get("lcn_1_per_year")) if native else math.nan
        rel_diff = (
            abs(b_lcn - n_lcn) / max(abs(n_lcn), 1.0e-30)
            if math.isfinite(b_lcn) and math.isfinite(n_lcn)
            else math.nan
        )
        rows.append(
            {
                "model_scope": benettin.get("model_scope"),
                "duration_years": benettin.get("duration_years"),
                "seed": benettin.get("seed"),
                "benettin_lcn_1_per_year": b_lcn,
                "native_megno_lcn_1_per_year": n_lcn,
                "relative_difference": rel_diff if math.isfinite(rel_diff) else None,
                "sign_agreement": (
                    (b_lcn >= 0.0) == (n_lcn >= 0.0)
                    if math.isfinite(b_lcn) and math.isfinite(n_lcn)
                    else None
                ),
                "classification_agreement": (
                    benettin.get("classification") == native.get("classification")
                    if native
                    else None
                ),
                "benettin_classification": benettin.get("classification"),
                "native_megno_classification": native.get("classification") if native else None,
                "benettin_summary": benettin.get("path"),
                "native_megno_summary": native.get("path") if native else None,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare finite-time Benettin LCN outputs with native Newtonian MEGNO summaries."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/home/peacelovephysics/ephemeris/output/stability"))
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)
    tag = args.tag or dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    rows = compare(args.output_dir)
    json_path = args.output_dir / f"benettin_megno_comparison_{tag}.json"
    csv_path = args.output_dir / f"benettin_megno_comparison_{tag}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"created_utc": dt.datetime.utcnow().isoformat() + "Z", "rows": rows}, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "model_scope",
            "duration_years",
            "seed",
            "benettin_lcn_1_per_year",
            "native_megno_lcn_1_per_year",
            "relative_difference",
            "sign_agreement",
            "classification_agreement",
            "benettin_classification",
            "native_megno_classification",
            "benettin_summary",
            "native_megno_summary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
