from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = Path("/home/peacelovephysics/ephemeris/output/model_vs_jpl_american_ephemeris_2000_2050_daily.csv")
OUT_DIR = Path("/home/peacelovephysics/ephemeris/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


dates = []
lon_err = []
lat_err = []
dist_err = []

with CSV_PATH.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["body"] != "Moon":
            continue
        dates.append(dt.date.fromisoformat(row["date"]))
        lon_err.append(float(row["lon_error_arcsec"]))
        lat_err.append(float(row["lat_error_arcsec"]))
        dist_err.append(float(row["distance_error_km"]))

lon_err = np.asarray(lon_err)
lat_err = np.asarray(lat_err)
dist_err = np.asarray(dist_err)

t_days = np.asarray([(d - dates[0]).days for d in dates], dtype=float)
t_years = t_days / 365.25

# Linear and quadratic fits
p1 = np.polyfit(t_years, lon_err, deg=1)
p2 = np.polyfit(t_years, lon_err, deg=2)

trend1 = np.polyval(p1, t_years)
trend2 = np.polyval(p2, t_years)

detrended1 = lon_err - trend1
detrended2 = lon_err - trend2

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

print("Moon longitude residual trend diagnostics")
print(f"Raw RMS                 : {rms(lon_err):.3f} arcsec")
print(f"Linear fit              : lon_err ≈ {p1[0]:.6f} arcsec/yr * t + {p1[1]:.6f}")
print(f"Linear detrended RMS    : {rms(detrended1):.3f} arcsec")
print(f"Quadratic fit           : lon_err ≈ {p2[0]:.9f} arcsec/yr^2 * t^2 + {p2[1]:.6f} arcsec/yr * t + {p2[2]:.6f}")
print(f"Quadratic detrended RMS : {rms(detrended2):.3f} arcsec")
print(f"Raw peak abs            : {float(np.max(np.abs(lon_err))):.3f} arcsec")
print(f"Linear detrended peak   : {float(np.max(np.abs(detrended1))):.3f} arcsec")
print(f"Quadratic detrended peak: {float(np.max(np.abs(detrended2))):.3f} arcsec")

# Raw + linear fit
plt.figure(figsize=(12, 5))
plt.plot(dates, lon_err, linewidth=0.8, label="Moon longitude residual")
plt.plot(dates, trend1, linewidth=1.2, label="linear fit")
plt.axhline(0.0, linestyle="--", linewidth=0.8)
plt.title("Moon longitude residual with linear trend")
plt.xlabel("Date")
plt.ylabel("Longitude error [arcsec]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "moon_lon_residual_linear_fit.png", dpi=200)
plt.close()

# Detrended residual
plt.figure(figsize=(12, 5))
plt.plot(dates, detrended1, linewidth=0.8)
plt.axhline(0.0, linestyle="--", linewidth=0.8)
plt.axhline(10.0, linestyle=":", linewidth=0.8)
plt.axhline(-10.0, linestyle=":", linewidth=0.8)
plt.title("Moon longitude residual after removing linear trend")
plt.xlabel("Date")
plt.ylabel("Detrended longitude error [arcsec]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "moon_lon_residual_detrended_linear.png", dpi=200)
plt.close()