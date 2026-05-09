from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import Bounds, dual_annealing, minimize
from skyfield.api import load

from .advanced_integrators import (
    acceleration_newtonian,
    acceleration_newtonian_earth_j2,
    acceleration_newtonian_gr_sun,
    acceleration_newtonian_gr_sun_earth_j2,
    integrate_dop853_with_accel,
    make_acceleration_with_earth_moon_empirical_term,
)
from .american_ephemeris import (
    circular_angle_diff_deg,
    ecliptic_lon_lat_from_icrf_vectors_m,
    jpl_geometric_and_apparent_ecliptic,
)
from .ephem import (
    EphemerisConfig,
    apply_lunar_velocity_correction,
    initial_state_solar_system_barycentric_time,
    solar_system_body_list_earth_moon,
)
from .lunar_calibration import LunarCalibration, save_lunar_calibration_profile


DAY_S = 86400.0
YEAR_S = 365.25 * DAY_S

PARAMETER_NAMES = (
    "dv_r_mm_s",
    "dv_t_mm_s",
    "dv_h_mm_s",
    "a_r_1e_15_m_s2",
    "a_t_1e_15_m_s2",
    "a_h_1e_15_m_s2",
)

PARAMETER_UNITS = {
    "dv_r_mm_s": "mm/s",
    "dv_t_mm_s": "mm/s",
    "dv_h_mm_s": "mm/s",
    "a_r_1e_15_m_s2": "1e-15 m/s^2",
    "a_t_1e_15_m_s2": "1e-15 m/s^2",
    "a_h_1e_15_m_s2": "1e-15 m/s^2",
}

OPTIMIZER_METHOD_LABELS = {
    "powell": "Powell",
    "nelder-mead": "Nelder-Mead",
    "dual-annealing": "Dual Annealing",
}


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def daily_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise ValueError("end date must be >= start date")
    n = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(n + 1)]


def make_tt_times_for_dates(ts, dates: list[dt.date]):
    years = np.array([d.year for d in dates], dtype=int)
    months = np.array([d.month for d in dates], dtype=int)
    days = np.array([d.day for d in dates], dtype=int)
    return ts.tt(years, months, days, 0, 0, 0)


def rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def choose_base_acceleration(args, bodies: tuple[str, ...]):
    sun_index = bodies.index("sun")
    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")

    if args.gr_model == "none":
        if args.earth_j2:
            return acceleration_newtonian_earth_j2, {
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        return acceleration_newtonian, {}

    if args.gr_model == "sun":
        if args.earth_j2:
            return acceleration_newtonian_gr_sun_earth_j2, {
                "sun_index": sun_index,
                "earth_index": earth_index,
                "moon_index": moon_index,
            }
        return acceleration_newtonian_gr_sun, {"sun_index": sun_index}

    raise ValueError(f"Unsupported gr_model: {args.gr_model!r}")


def polynomial_diagnostics(
    years: np.ndarray,
    lon_err_arcsec: np.ndarray,
) -> dict[str, float]:
    linear_coeff = np.polyfit(years, lon_err_arcsec, deg=1)
    linear_trend = np.polyval(linear_coeff, years)
    linear_detrended = lon_err_arcsec - linear_trend

    quadratic_coeff = np.polyfit(years, lon_err_arcsec, deg=2)
    quadratic_trend = np.polyval(quadratic_coeff, years)
    quadratic_detrended = lon_err_arcsec - quadratic_trend

    return {
        "linear_slope_arcsec_per_year": float(linear_coeff[0]),
        "linear_intercept_arcsec": float(linear_coeff[1]),
        "linear_detrended_rms_arcsec": rms(linear_detrended),
        "linear_detrended_peak_abs_arcsec": float(np.max(np.abs(linear_detrended))),
        "quadratic_coeff_arcsec_per_year2": float(quadratic_coeff[0]),
        "quadratic_linear_coeff_arcsec_per_year": float(quadratic_coeff[1]),
        "quadratic_intercept_arcsec": float(quadratic_coeff[2]),
        "quadratic_detrended_rms_arcsec": rms(quadratic_detrended),
        "quadratic_detrended_peak_abs_arcsec": float(np.max(np.abs(quadratic_detrended))),
    }


def objective_from_row(
    row: dict[str, float],
    objective: str,
    *,
    lat_weight: float,
    lat_peak_weight: float,
) -> float:
    if objective == "lon_rms":
        return row["lon_rms_arcsec"]
    if objective == "lon_peak":
        return row["lon_peak_abs_arcsec"]
    if objective == "lon_rms_plus_peak":
        return row["lon_rms_arcsec"] + row["lon_peak_abs_arcsec"]
    if objective == "lon_peak_plus_lat_rms":
        return row["lon_peak_abs_arcsec"] + lat_weight * row["lat_rms_arcsec"]
    if objective == "lon_rms_plus_peak_plus_lat_rms":
        return (
            row["lon_rms_arcsec"]
            + row["lon_peak_abs_arcsec"]
            + lat_weight * row["lat_rms_arcsec"]
        )
    if objective == "lon_rms_plus_peak_plus_lat_rms_plus_lat_peak":
        return (
            row["lon_rms_arcsec"]
            + row["lon_peak_abs_arcsec"]
            + lat_weight * row["lat_rms_arcsec"]
            + lat_peak_weight * row["lat_peak_abs_arcsec"]
        )
    if objective == "lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms":
        return (
            row["lon_peak_abs_arcsec"]
            + 0.5 * row["lon_rms_arcsec"]
            + 0.25 * row["linear_detrended_peak_abs_arcsec"]
            + 0.25 * row["quadratic_detrended_peak_abs_arcsec"]
            + lat_weight * row["lat_rms_arcsec"]
        )
    if objective == "slope_abs":
        return abs(row["linear_slope_arcsec_per_year"])
    raise ValueError(f"Unsupported objective: {objective!r}")


def expand_active_parameters(
    active_params: np.ndarray,
    *,
    template: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Expand an optimizer vector over active parameters to the full vector."""
    full_params = np.asarray(template, dtype=float).copy()
    full_params[active_mask] = np.asarray(active_params, dtype=float)
    return full_params


def evaluate_params(
    *,
    params: np.ndarray,
    base_state,
    earth_index: int,
    moon_index: int,
    times,
    offsets_s: np.ndarray,
    years_since_start: np.ndarray,
    date_strings: list[str],
    jpl_moon: dict[str, np.ndarray],
    base_accel_func,
    base_accel_kwargs: dict,
    args,
    keep_series: bool = False,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    (
        dv_r_mm_s,
        dv_t_mm_s,
        dv_h_mm_s,
        a_r_1e_15_m_s2,
        a_t_1e_15_m_s2,
        a_h_1e_15_m_s2,
    ) = [
        float(x) for x in params
    ]

    state0 = apply_lunar_velocity_correction(
        base_state,
        earth_index=earth_index,
        moon_index=moon_index,
        dv_r_m_s=dv_r_mm_s * 1.0e-3,
        dv_t_m_s=dv_t_mm_s * 1.0e-3,
        dv_h_m_s=dv_h_mm_s * 1.0e-3,
        preserve_emb_momentum=not args.no_preserve_emb_momentum,
    )

    accel_func = base_accel_func
    accel_kwargs = dict(base_accel_kwargs)

    if (
        a_r_1e_15_m_s2 != 0.0
        or a_t_1e_15_m_s2 != 0.0
        or a_h_1e_15_m_s2 != 0.0
    ):
        accel_func = make_acceleration_with_earth_moon_empirical_term(
            base_accel_func,
            earth_index=earth_index,
            moon_index=moon_index,
            a_r_m_s2=a_r_1e_15_m_s2 * 1.0e-15,
            a_t_m_s2=a_t_1e_15_m_s2 * 1.0e-15,
            a_h_m_s2=a_h_1e_15_m_s2 * 1.0e-15,
            base_accel_kwargs=base_accel_kwargs,
        )
        accel_kwargs = {}

    t_end = float(offsets_s[-1])
    times_s, positions_m, _ = integrate_dop853_with_accel(
        state0=state0,
        t_span=(0.0, t_end),
        dt=DAY_S,
        accel_func=accel_func,
        accel_kwargs=accel_kwargs,
        record_every=1,
        rtol=args.rtol,
        atol=args.atol,
        chunk_duration=args.chunk_years * YEAR_S,
        show_progress=args.progress_bar,
        max_step=args.max_step_days * DAY_S if args.max_step_days is not None else None,
    )

    sample_indices = np.rint(offsets_s / DAY_S).astype(int)
    max_alignment_error_s = float(np.max(np.abs(times_s[sample_indices] - offsets_s)))
    if max_alignment_error_s > 1e-5:
        raise RuntimeError(
            f"Model output is not aligned with requested TT midnights. "
            f"Max alignment error = {max_alignment_error_s} s"
        )

    model_positions = positions_m[sample_indices]
    model_geo_vec = model_positions[:, moon_index, :] - model_positions[:, earth_index, :]

    model_lon, model_lat, model_distance_km = ecliptic_lon_lat_from_icrf_vectors_m(
        model_geo_vec,
        times,
    )

    lon_err_arcsec = circular_angle_diff_deg(
        model_lon,
        jpl_moon["geom_lon_deg"],
    ) * 3600.0
    lat_err_arcsec = (model_lat - jpl_moon["geom_lat_deg"]) * 3600.0
    dist_err_km = model_distance_km - jpl_moon["geom_distance_km"]

    row = {
        "dv_r_mm_s": dv_r_mm_s,
        "dv_t_mm_s": dv_t_mm_s,
        "dv_h_mm_s": dv_h_mm_s,
        "a_r_1e_15_m_s2": a_r_1e_15_m_s2,
        "a_t_1e_15_m_s2": a_t_1e_15_m_s2,
        "a_h_1e_15_m_s2": a_h_1e_15_m_s2,
        "lon_rms_arcsec": rms(lon_err_arcsec),
        "lon_mean_arcsec": float(np.mean(lon_err_arcsec)),
        "lon_max_arcsec": float(np.max(lon_err_arcsec)),
        "lon_min_arcsec": float(np.min(lon_err_arcsec)),
        "lon_peak_abs_arcsec": float(np.max(np.abs(lon_err_arcsec))),
        "lat_rms_arcsec": rms(lat_err_arcsec),
        "lat_mean_arcsec": float(np.mean(lat_err_arcsec)),
        "lat_max_arcsec": float(np.max(lat_err_arcsec)),
        "lat_min_arcsec": float(np.min(lat_err_arcsec)),
        "lat_peak_abs_arcsec": float(np.max(np.abs(lat_err_arcsec))),
        "dist_rms_km": rms(dist_err_km),
        "dist_mean_km": float(np.mean(dist_err_km)),
        "dist_max_km": float(np.max(dist_err_km)),
        "dist_min_km": float(np.min(dist_err_km)),
        "dist_peak_abs_km": float(np.max(np.abs(dist_err_km))),
        **polynomial_diagnostics(years_since_start, lon_err_arcsec),
    }

    if not keep_series:
        return row, None

    series = {
        "date": date_strings,
        "model_lon_deg": model_lon,
        "jpl_geom_lon_deg": jpl_moon["geom_lon_deg"],
        "lon_error_arcsec": lon_err_arcsec,
        "model_lat_deg": model_lat,
        "jpl_geom_lat_deg": jpl_moon["geom_lat_deg"],
        "lat_error_arcsec": lat_err_arcsec,
        "model_distance_km": model_distance_km,
        "jpl_geom_distance_km": jpl_moon["geom_distance_km"],
        "distance_error_km": dist_err_km,
    }
    return row, series


def write_rows_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def reset_incremental_csv(path: str | Path) -> None:
    """Start a fresh per-trial journal at the normal trial CSV path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")


def append_row_csv(
    row: dict[str, Any],
    path: str | Path,
    fieldnames: list[str],
) -> None:
    """Append one trial row immediately so interrupted optimizer runs leave a record."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = out.stat().st_size == 0 if out.exists() else True

    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_best_residual_csv(series: dict[str, Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    n = len(series["date"])
    for i in range(n):
        rows.append(
            {
                "date": series["date"][i],
                "model_lon_deg": float(series["model_lon_deg"][i]),
                "jpl_geom_lon_deg": float(series["jpl_geom_lon_deg"][i]),
                "lon_error_arcsec": float(series["lon_error_arcsec"][i]),
                "model_lat_deg": float(series["model_lat_deg"][i]),
                "jpl_geom_lat_deg": float(series["jpl_geom_lat_deg"][i]),
                "lat_error_arcsec": float(series["lat_error_arcsec"][i]),
                "model_distance_km": float(series["model_distance_km"][i]),
                "jpl_geom_distance_km": float(series["jpl_geom_distance_km"][i]),
                "distance_error_km": float(series["distance_error_km"][i]),
            }
        )
    write_rows_csv(rows, path)


def plot_residual(
    *,
    dates: list[str],
    values: np.ndarray,
    output_path: str | Path,
    title: str,
    ylabel: str,
    y_limit_arcsec: float | None,
    target_peak_arcsec: float | None,
) -> None:
    plot_dates = [dt.date.fromisoformat(d) for d in dates]
    values = np.asarray(values, dtype=float)

    peak_abs = float(np.max(np.abs(values)))
    if y_limit_arcsec is None:
        y_limit_arcsec = max(1.0, 1.15 * peak_abs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(plot_dates, values, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(y_limit_arcsec, linestyle=":", linewidth=0.8)
    plt.axhline(-y_limit_arcsec, linestyle=":", linewidth=0.8)
    if target_peak_arcsec is not None:
        plt.axhline(target_peak_arcsec, linestyle="-.", linewidth=0.8)
        plt.axhline(-target_peak_arcsec, linestyle="-.", linewidth=0.8)
    plt.ylim(-y_limit_arcsec, y_limit_arcsec)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def default_sidecar_path(output_path: str | Path, suffix: str) -> str:
    out = Path(output_path)
    return str(out.with_name(f"{out.stem}{suffix}"))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def build_nelder_mead_initial_simplex(
    *,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    parameter_names: tuple[str, ...],
    step_fraction: float = 0.10,
) -> np.ndarray:
    """
    Build a custom bounded initial simplex scaled to each parameter's range.

    The first vertex is the requested initial point. Each additional vertex
    moves one parameter by ``step_fraction`` of its allowed range, choosing the
    positive direction when possible and the negative direction near an upper
    bound.
    """
    x0 = np.asarray(x0, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    n_params = len(x0)
    simplex = np.empty((n_params + 1, n_params), dtype=float)
    simplex[0] = x0

    spans = upper - lower
    for i in range(n_params):
        if spans[i] <= 0.0:
            raise ValueError(
                "Nelder-Mead requires nonzero bounds for each parameter so "
                f"the initial simplex is non-degenerate; {parameter_names[i]} "
                f"has bounds [{lower[i]}, {upper[i]}]."
            )

        step = step_fraction * spans[i]
        vertex = x0.copy()

        if x0[i] + step <= upper[i]:
            vertex[i] = x0[i] + step
        elif x0[i] - step >= lower[i]:
            vertex[i] = x0[i] - step
        else:
            vertex[i] = lower[i] + 0.5 * spans[i]

        if vertex[i] == x0[i]:
            raise ValueError(
                f"Could not build a distinct Nelder-Mead simplex vertex for "
                f"{parameter_names[i]}."
            )

        simplex[i + 1] = vertex

    return simplex


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit empirical lunar corrections: initial radial, tangential, and "
            "out-of-plane velocity plus optional radial, tangential, and "
            "out-of-plane basis accelerations."
        )
    )
    parser.add_argument("--kernel-path", required=True)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default="2050-12-31")
    parser.add_argument("--output", required=True, help="Trial metrics CSV.")
    parser.add_argument("--summary-output", default=None, help="Summary JSON path.")
    parser.add_argument("--best-residual-output", default=None, help="Best-run Moon residual CSV.")
    parser.add_argument("--moon-lon-plot", default=None, help="Best-run Moon longitude residual plot.")
    parser.add_argument("--moon-lat-plot", default=None, help="Best-run Moon latitude residual plot.")

    parser.add_argument("--gr-model", default="sun", choices=["none", "sun"])
    parser.add_argument("--earth-j2", action="store_true")
    parser.add_argument("--include-pluto", action="store_true")
    parser.add_argument("--no-preserve-emb-momentum", action="store_true")

    parser.add_argument("--chunk-years", type=float, default=1.0)
    parser.add_argument("--max-step-days", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-15)
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show DOP853 chunk progress bars for each trial.",
    )

    parser.add_argument("--initial-dv-r-mm-s", type=float, default=0.0)
    parser.add_argument("--initial-dv-t-mm-s", type=float, default=0.039220792423)
    parser.add_argument("--initial-dv-h-mm-s", type=float, default=0.0)
    parser.add_argument("--initial-ar-1e-15", type=float, default=0.0)
    parser.add_argument("--initial-at-1e-15", type=float, default=4.744123111671)
    parser.add_argument("--initial-ah-1e-15", type=float, default=0.0)

    parser.add_argument("--dv-r-min-mm-s", type=float, default=-0.05)
    parser.add_argument("--dv-r-max-mm-s", type=float, default=0.05)
    parser.add_argument("--dv-t-min-mm-s", type=float, default=0.038)
    parser.add_argument("--dv-t-max-mm-s", type=float, default=0.041)
    parser.add_argument("--dv-h-min-mm-s", type=float, default=-0.05)
    parser.add_argument("--dv-h-max-mm-s", type=float, default=0.05)
    parser.add_argument("--ar-min-1e-15", type=float, default=0.0)
    parser.add_argument("--ar-max-1e-15", type=float, default=0.0)
    parser.add_argument("--at-min-1e-15", type=float, default=4.0)
    parser.add_argument("--at-max-1e-15", type=float, default=5.5)
    parser.add_argument("--ah-min-1e-15", type=float, default=0.0)
    parser.add_argument("--ah-max-1e-15", type=float, default=0.0)

    parser.add_argument(
        "--objective",
        choices=[
            "lon_rms",
            "lon_peak",
            "lon_rms_plus_peak",
            "lon_peak_plus_lat_rms",
            "lon_rms_plus_peak_plus_lat_rms",
            "lon_rms_plus_peak_plus_lat_rms_plus_lat_peak",
            "lon_peak_plus_half_lon_rms_plus_trend_peaks_plus_lat_rms",
            "slope_abs",
        ],
        default="lon_rms_plus_peak_plus_lat_rms",
    )
    parser.add_argument("--lat-weight", type=float, default=0.5)
    parser.add_argument("--lat-peak-weight", type=float, default=0.1)
    parser.add_argument("--target-lon-peak-arcsec", type=float, default=0.5)
    parser.add_argument(
        "--method",
        choices=["powell", "nelder-mead", "dual-annealing"],
        default="powell",
        help=(
            "Optimizer method for non-grid runs. Powell is the default for "
            "backward compatibility. Nelder-Mead uses a custom bounded initial "
            "simplex. Dual Annealing uses the configured parameter bounds."
        ),
    )
    parser.add_argument("--opt-maxiter", type=int, default=40)
    parser.add_argument("--opt-xtol", type=float, default=1e-5)
    parser.add_argument("--opt-ftol", type=float, default=1e-5)
    parser.add_argument("--anneal-maxiter", type=int, default=100)
    parser.add_argument("--anneal-initial-temp", type=float, default=5230.0)
    parser.add_argument("--anneal-seed", type=int, default=None)
    parser.add_argument(
        "--anneal-local-powell",
        action="store_true",
        help="For --method dual-annealing, finish each annealing cycle with a bounded Powell local minimizer.",
    )

    parser.add_argument("--grid-only", action="store_true")
    parser.add_argument("--grid-dv-r-count", type=int, default=3)
    parser.add_argument("--grid-dv-t-count", type=int, default=5)
    parser.add_argument("--grid-dv-h-count", type=int, default=3)
    parser.add_argument("--grid-ar-count", type=int, default=1)
    parser.add_argument("--grid-at-count", type=int, default=5)
    parser.add_argument("--grid-ah-count", type=int, default=1)

    parser.add_argument("--quiet-trials", action="store_true")
    parser.add_argument("--skip-best-artifacts", action="store_true")
    parser.add_argument("--moon-lon-ylim-arcsec", type=float, default=None)
    parser.add_argument("--moon-lat-ylim-arcsec", type=float, default=None)

    parser.add_argument(
        "--save-calibration-file",
        default=None,
        help="Optional JSON file to update with the best-fit lunar calibration profile.",
    )
    parser.add_argument(
        "--save-calibration-name",
        default=None,
        help="Profile name to save when --save-calibration-file is used.",
    )
    parser.add_argument(
        "--save-calibration-description",
        default="",
        help="Optional description for the saved lunar calibration profile.",
    )
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    summary_output = args.summary_output or default_sidecar_path(args.output, "_summary.json")
    best_residual_output = args.best_residual_output or default_sidecar_path(
        args.output,
        "_best_moon_residuals.csv",
    )
    moon_lon_plot = args.moon_lon_plot or default_sidecar_path(
        args.output,
        "_best_moon_longitude_residual.png",
    )
    moon_lat_plot = args.moon_lat_plot or default_sidecar_path(
        args.output,
        "_best_moon_latitude_residual.png",
    )

    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    dates_dt = daily_dates(start, end)
    date_strings = [d.isoformat() for d in dates_dt]

    ts = load.timescale()
    times = make_tt_times_for_dates(ts, dates_dt)
    t0 = ts.tt(2000, 1, 1, 0, 0, 0)

    offsets_s = (times.tt - t0.tt) * DAY_S
    if np.any(offsets_s < -1e-9):
        raise ValueError("This script assumes requested dates are >= 2000-01-01.")

    years_since_start = (times.tt - times.tt[0]) / 365.25

    bodies = tuple(solar_system_body_list_earth_moon())
    if args.include_pluto and "pluto barycenter" not in bodies:
        bodies = bodies + ("pluto barycenter",)

    config = EphemerisConfig(kernel_path=args.kernel_path)
    base_state = initial_state_solar_system_barycentric_time(
        t0,
        bodies=bodies,
        config=config,
        verbose=True,
    )

    earth_index = bodies.index("earth")
    moon_index = bodies.index("moon")
    base_accel_func, base_accel_kwargs = choose_base_acceleration(args, bodies)

    x0 = np.array(
        [
            args.initial_dv_r_mm_s,
            args.initial_dv_t_mm_s,
            args.initial_dv_h_mm_s,
            args.initial_ar_1e_15,
            args.initial_at_1e_15,
            args.initial_ah_1e_15,
        ],
        dtype=float,
    )
    lower = np.array(
        [
            args.dv_r_min_mm_s,
            args.dv_t_min_mm_s,
            args.dv_h_min_mm_s,
            args.ar_min_1e_15,
            args.at_min_1e_15,
            args.ah_min_1e_15,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            args.dv_r_max_mm_s,
            args.dv_t_max_mm_s,
            args.dv_h_max_mm_s,
            args.ar_max_1e_15,
            args.at_max_1e_15,
            args.ah_max_1e_15,
        ],
        dtype=float,
    )
    if np.any(upper < lower):
        raise ValueError(f"Parameter upper bounds must be >= lower bounds. bounds=({lower}, {upper})")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError(f"Initial parameters must lie inside bounds. x0={x0}, bounds=({lower}, {upper})")

    # Equal min/max bounds mark a parameter as fixed. This keeps historical
    # four-parameter runs compatible while allowing the v4 six-parameter search
    # to open the radial and out-of-plane acceleration dimensions explicitly.
    active_mask = upper > lower
    active_names = tuple(
        name for name, is_active in zip(PARAMETER_NAMES, active_mask) if is_active
    )
    if not np.any(active_mask) and not args.grid_only:
        raise ValueError("At least one parameter bound must be open for optimizer runs.")

    optimizer_mode = "grid" if args.grid_only else args.method
    optimizer_label = "Grid" if args.grid_only else OPTIMIZER_METHOD_LABELS[args.method]

    print("[Empirical Lunar Velocity + Acceleration Fit]")
    print("  Target convention")
    print("    comparison     : Moon geocentric ecliptic-of-date longitude vs JPL geometry")
    print("    book range     : American Ephemeris 2000-01-01 through 2050-12-31")
    print("    target peak    : %.6f arcsec" % args.target_lon_peak_arcsec)
    print("  Experiment")
    print(f"    start_date     : {args.start_date}")
    print(f"    end_date       : {args.end_date}")
    print(f"    sample_days    : {len(dates_dt)}")
    print(f"    bodies         : {bodies}")
    print(f"    gr_model       : {args.gr_model}")
    print(f"    earth_j2       : {args.earth_j2}")
    print(f"    preserve_emb   : {not args.no_preserve_emb_momentum}")
    print(f"    chunk_years    : {args.chunk_years}")
    print(f"    max_step_days  : {args.max_step_days}")
    print(f"    rtol / atol    : {args.rtol} / {args.atol}")
    print("  Parameters")
    for name, initial, lo, hi in zip(PARAMETER_NAMES, x0, lower, upper):
        status = "active" if hi > lo else "fixed"
        print(
            f"    {name:17s}: initial={initial:.12f}, "
            f"bounds=[{lo:.12f}, {hi:.12f}], {status}"
        )
    print("  Optimizer")
    print(f"    mode           : {optimizer_label}")
    print(f"    objective      : {args.objective}")
    print(f"    lat_weight     : {args.lat_weight}")
    print(f"    lat_peak_weight: {args.lat_peak_weight}")
    print(f"    maxiter        : {args.opt_maxiter}")
    print(f"    xtol / ftol    : {args.opt_xtol} / {args.opt_ftol}")
    if not args.grid_only and args.method == "dual-annealing":
        print(f"    anneal_maxiter : {args.anneal_maxiter}")
        print(f"    initial_temp   : {args.anneal_initial_temp}")
        print(f"    anneal_seed    : {args.anneal_seed}")
        print(f"    local_powell   : {args.anneal_local_powell}")
    print("  Outputs")
    print(f"    trial_csv      : {args.output}")
    print("    trial_journal  : append one completed trial row immediately")
    print(f"    summary_json   : {summary_output}")
    if not args.skip_best_artifacts:
        print(f"    residual_csv   : {best_residual_output}")
        print(f"    lon_plot       : {moon_lon_plot}")
        print(f"    lat_plot       : {moon_lat_plot}")
    print()

    reset_incremental_csv(args.output)

    print("Precomputing JPL Moon geometric reference...")
    jpl_moon = jpl_geometric_and_apparent_ecliptic(
        kernel_path=args.kernel_path,
        times=times,
        bodies=["moon"],
    )["moon"]

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    trial_csv_fieldnames: list[str] | None = None

    def eval_cached(x: np.ndarray) -> dict[str, Any]:
        nonlocal trial_csv_fieldnames

        x = np.asarray(x, dtype=float)
        key = tuple(round(float(v), 12) for v in x)
        if key not in cache:
            trial_no = len(rows) + 1
            trial_start = dt.datetime.now(dt.timezone.utc)
            trial_start_perf = time.perf_counter()
            if not args.quiet_trials:
                print(
                    "Trial %04d: dv_r=%.12f, dv_t=%.12f, dv_h=%.12f mm/s, "
                    "a_r=%.12f, a_t=%.12f, a_h=%.12f x 1e-15 m/s^2"
                    % (trial_no, x[0], x[1], x[2], x[3], x[4], x[5])
                )

            trial_metadata = {
                "trial": trial_no,
                "trial_status": "success",
                "trial_error": "",
                "trial_start_timestamp_utc": trial_start.isoformat(),
                "optimizer_mode": optimizer_mode,
                "optimizer_method": "grid" if args.grid_only else args.method,
                "optimizer_method_label": optimizer_label,
                "anneal_local_powell": bool(args.anneal_local_powell)
                if args.method == "dual-annealing"
                else False,
                "objective": args.objective,
                "lat_weight": args.lat_weight,
                "lat_peak_weight": args.lat_peak_weight,
                "target_lon_peak_arcsec": args.target_lon_peak_arcsec,
            }

            try:
                metrics_row, _ = evaluate_params(
                    params=x,
                    base_state=base_state,
                    earth_index=earth_index,
                    moon_index=moon_index,
                    times=times,
                    offsets_s=offsets_s,
                    years_since_start=years_since_start,
                    date_strings=date_strings,
                    jpl_moon=jpl_moon,
                    base_accel_func=base_accel_func,
                    base_accel_kwargs=base_accel_kwargs,
                    args=args,
                )
            except Exception as exc:
                failure_row = {
                    **trial_metadata,
                    "trial_status": "error",
                    "trial_error": repr(exc),
                    "trial_end_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "trial_runtime_s": time.perf_counter() - trial_start_perf,
                    **dict(zip(PARAMETER_NAMES, [float(v) for v in x])),
                }
                if trial_csv_fieldnames is None:
                    trial_csv_fieldnames = list(failure_row.keys())
                append_row_csv(failure_row, args.output, trial_csv_fieldnames)
                raise

            objective_value = objective_from_row(
                metrics_row,
                args.objective,
                lat_weight=args.lat_weight,
                lat_peak_weight=args.lat_peak_weight,
            )
            row = {
                **trial_metadata,
                "trial_end_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "trial_runtime_s": time.perf_counter() - trial_start_perf,
                **metrics_row,
                "objective_value": objective_value,
                "target_met": metrics_row["lon_peak_abs_arcsec"]
                <= args.target_lon_peak_arcsec,
            }

            cache[key] = row
            rows.append(row)
            if trial_csv_fieldnames is None:
                trial_csv_fieldnames = list(row.keys())
            append_row_csv(row, args.output, trial_csv_fieldnames)

            if not args.quiet_trials:
                print(
                    "  objective=%.9f, lon_rms=%.6f, lon_peak=%.6f, "
                    "lon_mean=%.6f, lon_max=%.6f, lon_min=%.6f, "
                    "lat_rms=%.6f, lat_peak=%.6f, dist_rms=%.6f, "
                    "runtime=%.2fs"
                    % (
                        row["objective_value"],
                        row["lon_rms_arcsec"],
                        row["lon_peak_abs_arcsec"],
                        row["lon_mean_arcsec"],
                        row["lon_max_arcsec"],
                        row["lon_min_arcsec"],
                        row["lat_rms_arcsec"],
                        row["lat_peak_abs_arcsec"],
                        row["dist_rms_km"],
                        row["trial_runtime_s"],
                    )
                )

        return cache[key]

    def scalar_objective(x: np.ndarray) -> float:
        row = eval_cached(np.asarray(x, dtype=float))
        return row["objective_value"]

    initial_simplex = None
    opt_x0 = x0[active_mask]
    opt_lower = lower[active_mask]
    opt_upper = upper[active_mask]

    def scalar_objective_active(active_params: np.ndarray) -> float:
        full_params = expand_active_parameters(
            active_params,
            template=x0,
            active_mask=active_mask,
        )
        return scalar_objective(full_params)

    if args.grid_only:
        values = [
            np.linspace(args.dv_r_min_mm_s, args.dv_r_max_mm_s, args.grid_dv_r_count),
            np.linspace(args.dv_t_min_mm_s, args.dv_t_max_mm_s, args.grid_dv_t_count),
            np.linspace(args.dv_h_min_mm_s, args.dv_h_max_mm_s, args.grid_dv_h_count),
            np.linspace(args.ar_min_1e_15, args.ar_max_1e_15, args.grid_ar_count),
            np.linspace(args.at_min_1e_15, args.at_max_1e_15, args.grid_at_count),
            np.linspace(args.ah_min_1e_15, args.ah_max_1e_15, args.grid_ah_count),
        ]
        total_trials = int(np.prod([len(v) for v in values]))
        print()
        print("[Grid scan]")
        print(f"  total_trials   : {total_trials}")
        print()
        for dv_r in values[0]:
            for dv_t in values[1]:
                for dv_h in values[2]:
                    for a_r in values[3]:
                        for a_t in values[4]:
                            for a_h in values[5]:
                                eval_cached(
                                    np.array(
                                        [dv_r, dv_t, dv_h, a_r, a_t, a_h],
                                        dtype=float,
                                    )
                                )

        best_row = min(rows, key=lambda r: r["objective_value"])

        class GridResult:
            success = True
            message = "Grid scan complete."
            nfev = total_trials
            nit = 0
            x = np.array([best_row[name] for name in PARAMETER_NAMES], dtype=float)
            fun = best_row["objective_value"]

        result = GridResult()
    else:
        if args.method == "powell":
            result = minimize(
                scalar_objective_active,
                x0=opt_x0,
                method="Powell",
                bounds=Bounds(opt_lower, opt_upper),
                options={
                    "maxiter": args.opt_maxiter,
                    "xtol": args.opt_xtol,
                    "ftol": args.opt_ftol,
                    "disp": True,
                },
            )
        elif args.method == "nelder-mead":
            initial_simplex = build_nelder_mead_initial_simplex(
                x0=opt_x0,
                lower=opt_lower,
                upper=opt_upper,
                parameter_names=active_names,
            )

            print()
            print("[Nelder-Mead initial simplex]")
            for vertex_index, vertex in enumerate(initial_simplex):
                formatted = ", ".join(
                    f"{name}={value:.12f}"
                    for name, value in zip(active_names, vertex)
                )
                print(f"  vertex {vertex_index}: {formatted}")
            print()

            def bounded_scalar_objective(x: np.ndarray) -> float:
                x = np.asarray(x, dtype=float)
                lower_violation = np.maximum(opt_lower - x, 0.0)
                upper_violation = np.maximum(x - opt_upper, 0.0)
                violation = lower_violation + upper_violation
                if np.any(violation > 0.0):
                    return 1.0e9 + float(np.sum(violation * violation))
                return scalar_objective_active(x)

            result = minimize(
                bounded_scalar_objective,
                x0=opt_x0,
                method="Nelder-Mead",
                bounds=Bounds(opt_lower, opt_upper),
                options={
                    "maxiter": args.opt_maxiter,
                    "xatol": args.opt_xtol,
                    "fatol": args.opt_ftol,
                    "initial_simplex": initial_simplex,
                    "disp": True,
                },
            )
        elif args.method == "dual-annealing":
            anneal_bounds = list(zip(opt_lower, opt_upper))
            minimizer_kwargs = None
            if args.anneal_local_powell:
                minimizer_kwargs = {
                    "method": "Powell",
                    "bounds": Bounds(opt_lower, opt_upper),
                    "options": {
                        "maxiter": args.opt_maxiter,
                        "xtol": args.opt_xtol,
                        "ftol": args.opt_ftol,
                        "disp": False,
                    },
                }

            result = dual_annealing(
                scalar_objective_active,
                bounds=anneal_bounds,
                maxiter=args.anneal_maxiter,
                initial_temp=args.anneal_initial_temp,
                seed=args.anneal_seed,
                no_local_search=not args.anneal_local_powell,
                minimizer_kwargs=minimizer_kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer method: {args.method!r}")

        best_params_from_optimizer = expand_active_parameters(
            np.clip(result.x, opt_lower, opt_upper),
            template=x0,
            active_mask=active_mask,
        )
        best_row = eval_cached(best_params_from_optimizer)

    best_params = np.array([best_row[name] for name in PARAMETER_NAMES], dtype=float)
    best_series = None
    if not args.skip_best_artifacts:
        print()
        print("[Best-run artifact pass]")
        print("  Reintegrating best parameter set for residual CSV and plots.")
        best_row, best_series = evaluate_params(
            params=best_params,
            base_state=base_state,
            earth_index=earth_index,
            moon_index=moon_index,
            times=times,
            offsets_s=offsets_s,
            years_since_start=years_since_start,
            date_strings=date_strings,
            jpl_moon=jpl_moon,
            base_accel_func=base_accel_func,
            base_accel_kwargs=base_accel_kwargs,
            args=args,
            keep_series=True,
        )
        best_row["objective"] = args.objective
        best_row["lat_weight"] = args.lat_weight
        best_row["lat_peak_weight"] = args.lat_peak_weight
        best_row["objective_value"] = objective_from_row(
            best_row,
            args.objective,
            lat_weight=args.lat_weight,
            lat_peak_weight=args.lat_peak_weight,
        )
        best_row["target_lon_peak_arcsec"] = args.target_lon_peak_arcsec
        best_row["target_met"] = best_row["lon_peak_abs_arcsec"] <= args.target_lon_peak_arcsec

        write_best_residual_csv(best_series, best_residual_output)
        plot_residual(
            dates=best_series["date"],
            values=best_series["lon_error_arcsec"],
            output_path=moon_lon_plot,
            title="Moon Longitude Residual - Empirical Lunar Calibration",
            ylabel="Longitude error [arcsec]",
            y_limit_arcsec=args.moon_lon_ylim_arcsec,
            target_peak_arcsec=args.target_lon_peak_arcsec,
        )
        plot_residual(
            dates=best_series["date"],
            values=best_series["lat_error_arcsec"],
            output_path=moon_lat_plot,
            title="Moon Latitude Residual - Empirical Lunar Calibration",
            ylabel="Latitude error [arcsec]",
            y_limit_arcsec=args.moon_lat_ylim_arcsec,
            target_peak_arcsec=None,
        )

    print()
    print("Optimizer result")
    print(f"  success                       : {result.success}")
    print(f"  message                       : {result.message}")
    print(f"  method                        : {'grid' if args.grid_only else OPTIMIZER_METHOD_LABELS[args.method]}")
    print(f"  objective                     : {args.objective}")
    print(f"  lat_weight                    : {args.lat_weight:.9f}")
    print(f"  lat_peak_weight               : {args.lat_peak_weight:.9f}")
    print(f"  target_lon_peak_arcsec        : {args.target_lon_peak_arcsec:.9f}")
    print(f"  target_met                    : {best_row['target_met']}")
    print(f"  dv_r_mm_s                     : {best_row['dv_r_mm_s']:.12f}")
    print(f"  dv_t_mm_s                     : {best_row['dv_t_mm_s']:.12f}")
    print(f"  dv_h_mm_s                     : {best_row['dv_h_mm_s']:.12f}")
    print(f"  a_r_1e_15_m_s2                : {best_row['a_r_1e_15_m_s2']:.12f}")
    print(f"  a_t_1e_15_m_s2                : {best_row['a_t_1e_15_m_s2']:.12f}")
    print(f"  a_h_1e_15_m_s2                : {best_row['a_h_1e_15_m_s2']:.12f}")
    print(f"  objective_value               : {best_row['objective_value']:.9f}")
    print(f"  lon_rms_arcsec                : {best_row['lon_rms_arcsec']:.9f}")
    print(f"  lon_peak_abs_arcsec           : {best_row['lon_peak_abs_arcsec']:.9f}")
    print(f"  lon_mean_arcsec               : {best_row['lon_mean_arcsec']:.9f}")
    print(f"  lon_max_arcsec                : {best_row['lon_max_arcsec']:.9f}")
    print(f"  lon_min_arcsec                : {best_row['lon_min_arcsec']:.9f}")
    print(f"  linear_slope_arcsec_per_year  : {best_row['linear_slope_arcsec_per_year']:.9f}")
    print(f"  linear_detrended_rms_arcsec   : {best_row['linear_detrended_rms_arcsec']:.9f}")
    print(f"  quadratic_coeff_arcsec_yr2    : {best_row['quadratic_coeff_arcsec_per_year2']:.12f}")
    print(f"  quadratic_detrended_rms       : {best_row['quadratic_detrended_rms_arcsec']:.9f}")
    print(f"  lat_rms_arcsec                : {best_row['lat_rms_arcsec']:.9f}")
    print(f"  lat_peak_abs_arcsec           : {best_row['lat_peak_abs_arcsec']:.9f}")
    print(f"  dist_rms_km                   : {best_row['dist_rms_km']:.9f}")
    print(f"Wrote trial CSV                 : {args.output}")

    if args.save_calibration_file is not None:
        if args.save_calibration_name is None:
            raise ValueError(
                "--save-calibration-name is required when --save-calibration-file is used."
            )

        profile = LunarCalibration(
            name=args.save_calibration_name,
            moon_dv_r_mm_s=best_row["dv_r_mm_s"],
            moon_dv_t_mm_s=best_row["dv_t_mm_s"],
            moon_dv_h_mm_s=best_row["dv_h_mm_s"],
            moon_a_r_1e_15_m_s2=best_row["a_r_1e_15_m_s2"],
            moon_a_t_1e_15_m_s2=best_row["a_t_1e_15_m_s2"],
            moon_a_h_1e_15_m_s2=best_row["a_h_1e_15_m_s2"],
            description=args.save_calibration_description,
            fit_start_date=args.start_date,
            fit_end_date=args.end_date,
            validation_start_date=args.start_date,
            validation_end_date=args.end_date,
            objective=args.objective,
            lat_weight=args.lat_weight,
            lat_peak_weight=args.lat_peak_weight,
            model_notes=(
                "Newtonian N-body + optional Sun 1PN GR + optional Earth J2 + "
                "empirical 3D lunar initial velocity correction + empirical "
                "lunar radial/tangential/out-of-plane acceleration. Short-range "
                "ephemeris matching profile; do not use for long-term stability "
                "studies."
            ),
            lon_rms_arcsec=best_row["lon_rms_arcsec"],
            lon_peak_abs_arcsec=best_row["lon_peak_abs_arcsec"],
            lat_rms_arcsec=best_row["lat_rms_arcsec"],
            dist_rms_km=best_row["dist_rms_km"],
        )

        save_lunar_calibration_profile(profile, args.save_calibration_file)
        print()
        print(f"Saved lunar calibration profile : {args.save_calibration_name}")
        print(f"Calibration file                : {args.save_calibration_file}")

    summary = {
        "experiment": {
            "script": "mini_ephemeris.fit_lunar_velocity_3d_and_accel",
            "start_date": args.start_date,
            "end_date": args.end_date,
            "sample_days": len(dates_dt),
            "kernel_path": args.kernel_path,
            "bodies": bodies,
            "gr_model": args.gr_model,
            "earth_j2": args.earth_j2,
            "preserve_emb_momentum": not args.no_preserve_emb_momentum,
            "chunk_years": args.chunk_years,
            "max_step_days": args.max_step_days,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "parameters": {
            "names": PARAMETER_NAMES,
            "initial": dict(zip(PARAMETER_NAMES, x0)),
            "lower_bounds": dict(zip(PARAMETER_NAMES, lower)),
            "upper_bounds": dict(zip(PARAMETER_NAMES, upper)),
            "active": dict(zip(PARAMETER_NAMES, active_mask)),
            "units": PARAMETER_UNITS,
        },
        "optimizer": {
            "mode": "grid" if args.grid_only else OPTIMIZER_METHOD_LABELS[args.method],
            "method": None if args.grid_only else args.method,
            "objective": args.objective,
            "lat_weight": args.lat_weight,
            "lat_peak_weight": args.lat_peak_weight,
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": getattr(result, "nfev", None),
            "nit": getattr(result, "nit", None),
            "maxiter": args.opt_maxiter,
            "xtol": args.opt_xtol,
            "ftol": args.opt_ftol,
            "anneal_maxiter": args.anneal_maxiter,
            "anneal_initial_temp": args.anneal_initial_temp,
            "anneal_seed": args.anneal_seed,
            "anneal_local_powell": args.anneal_local_powell,
            "active_parameter_names": active_names,
            "initial_simplex": None if initial_simplex is None else initial_simplex,
            "target_lon_peak_arcsec": args.target_lon_peak_arcsec,
        },
        "best": best_row,
        "outputs": {
            "trial_csv": args.output,
            "summary_json": summary_output,
            "best_residual_csv": None if args.skip_best_artifacts else best_residual_output,
            "moon_longitude_plot": None if args.skip_best_artifacts else moon_lon_plot,
            "moon_latitude_plot": None if args.skip_best_artifacts else moon_lat_plot,
            "saved_calibration_file": args.save_calibration_file,
            "saved_calibration_name": args.save_calibration_name,
        },
    }

    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(summary_output).open("w") as f:
        json.dump(json_ready(summary), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote summary JSON              : {summary_output}")

    if not args.skip_best_artifacts:
        print(f"Wrote best residual CSV         : {best_residual_output}")
        print(f"Wrote Moon longitude plot       : {moon_lon_plot}")
        print(f"Wrote Moon latitude plot        : {moon_lat_plot}")


if __name__ == "__main__":
    main()
