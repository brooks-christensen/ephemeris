from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CSV_PATH = Path(
    "/home/peacelovephysics/ephemeris/output/model_vs_jpl_american_ephemeris_2000_2050_daily.csv"
)
DEFAULT_OUT_DIR = Path("/home/peacelovephysics/ephemeris/output")


def rms(x) -> float:
    arr = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def load_moon_residuals(csv_path: Path):
    dates = []
    lon_err = []
    lat_err = []
    dist_err = []

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["body"] != "Moon":
                continue
            dates.append(dt.date.fromisoformat(row["date"]))
            lon_err.append(float(row["lon_error_arcsec"]))
            lat_err.append(float(row["lat_error_arcsec"]))
            dist_err.append(float(row["distance_error_km"]))

    if not dates:
        raise ValueError(f"No Moon rows found in {csv_path}")

    return (
        dates,
        np.asarray(lon_err, dtype=float),
        np.asarray(lat_err, dtype=float),
        np.asarray(dist_err, dtype=float),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Moon longitude residual trends from a model-vs-JPL CSV."
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to model-vs-JPL American Ephemeris comparison CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for residual diagnostic plots.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag used in output plot filenames.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or csv_path.stem

    dates, lon_err, lat_err, dist_err = load_moon_residuals(csv_path)

    t_days = np.asarray([(d - dates[0]).days for d in dates], dtype=float)
    t_years = t_days / 365.25

    p1 = np.polyfit(t_years, lon_err, deg=1)
    p2 = np.polyfit(t_years, lon_err, deg=2)

    trend1 = np.polyval(p1, t_years)
    trend2 = np.polyval(p2, t_years)

    detrended1 = lon_err - trend1
    detrended2 = lon_err - trend2

    print("Moon longitude residual trend diagnostics")
    print(f"CSV path                : {csv_path}")
    print(f"Samples                 : {len(dates)}")
    print(f"Date range              : {dates[0]} to {dates[-1]}")
    print(f"Raw mean                : {float(np.mean(lon_err)):.6f} arcsec")
    print(f"Raw RMS                 : {rms(lon_err):.6f} arcsec")
    print(f"Raw peak abs            : {float(np.max(np.abs(lon_err))):.6f} arcsec")
    print(f"Raw max                 : {float(np.max(lon_err)):.6f} arcsec")
    print(f"Raw min                 : {float(np.min(lon_err)):.6f} arcsec")
    print()
    print(f"Linear fit              : lon_err ≈ {p1[0]:.9f} arcsec/yr * t + {p1[1]:.9f}")
    print(f"Linear detrended RMS    : {rms(detrended1):.6f} arcsec")
    print(f"Linear detrended peak   : {float(np.max(np.abs(detrended1))):.6f} arcsec")
    print()
    print(
        "Quadratic fit           : "
        f"lon_err ≈ {p2[0]:.12f} arcsec/yr^2 * t^2 "
        f"+ {p2[1]:.9f} arcsec/yr * t + {p2[2]:.9f}"
    )
    print(f"Quadratic detrended RMS : {rms(detrended2):.6f} arcsec")
    print(f"Quadratic detrended peak: {float(np.max(np.abs(detrended2))):.6f} arcsec")
    print()
    print(f"Latitude RMS            : {rms(lat_err):.6f} arcsec")
    print(f"Distance RMS            : {rms(dist_err):.6f} km")

    # Raw + linear + quadratic fit
    plt.figure(figsize=(12, 5))
    plt.plot(dates, lon_err, linewidth=0.8, label="Moon longitude residual")
    plt.plot(dates, trend1, linewidth=1.2, label="linear fit")
    plt.plot(dates, trend2, linewidth=1.2, label="quadratic fit")
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.title("Moon longitude residual with trend fits")
    plt.xlabel("Date")
    plt.ylabel("Longitude error [arcsec]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_moon_lon_residual_trend_fits.png", dpi=200)
    plt.close()

    # Linear detrended residual
    plt.figure(figsize=(12, 5))
    plt.plot(dates, detrended1, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(1.0, linestyle=":", linewidth=0.8)
    plt.axhline(-1.0, linestyle=":", linewidth=0.8)
    plt.title("Moon longitude residual after removing linear trend")
    plt.xlabel("Date")
    plt.ylabel("Detrended longitude error [arcsec]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_moon_lon_residual_detrended_linear.png", dpi=200)
    plt.close()

    # Quadratic detrended residual
    plt.figure(figsize=(12, 5))
    plt.plot(dates, detrended2, linewidth=0.8)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axhline(1.0, linestyle=":", linewidth=0.8)
    plt.axhline(-1.0, linestyle=":", linewidth=0.8)
    plt.title("Moon longitude residual after removing quadratic trend")
    plt.xlabel("Date")
    plt.ylabel("Detrended longitude error [arcsec]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_moon_lon_residual_detrended_quadratic.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()