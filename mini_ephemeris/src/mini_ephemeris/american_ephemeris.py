from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Iterable

import numpy as np
from skyfield.api import load
from skyfield.framelib import ecliptic_frame


ZODIAC_SIGNS = [
    ("Aries", "Ari", "♈"),
    ("Taurus", "Tau", "♉"),
    ("Gemini", "Gem", "♊"),
    ("Cancer", "Can", "♋"),
    ("Leo", "Leo", "♌"),
    ("Virgo", "Vir", "♍"),
    ("Libra", "Lib", "♎"),
    ("Scorpio", "Sco", "♏"),
    ("Sagittarius", "Sag", "♐"),
    ("Capricorn", "Cap", "♑"),
    ("Aquarius", "Aqu", "♒"),
    ("Pisces", "Pis", "♓"),
]


BOOK_BODY_NAMES = {
    "sun": "Sun",
    "moon": "Moon",
    "mercury barycenter": "Mercury",
    "venus barycenter": "Venus",
    "mars barycenter": "Mars",
    "jupiter barycenter": "Jupiter",
    "saturn barycenter": "Saturn",
    "uranus barycenter": "Uranus",
    "neptune barycenter": "Neptune",
    "pluto barycenter": "Pluto",
    "earth": "Earth",
}


def wrap360(deg: float | np.ndarray) -> float | np.ndarray:
    return np.mod(deg, 360.0)


def circular_angle_diff_deg(model_deg: np.ndarray, truth_deg: np.ndarray) -> np.ndarray:
    """
    Signed angular difference model - truth in degrees, wrapped to [-180, 180).
    """
    return (model_deg - truth_deg + 180.0) % 360.0 - 180.0


def longitude_to_zodiac_parts(lon_deg: float) -> tuple[int, int, int, float]:
    """
    Convert 0..360 longitude into sign index, degree-in-sign, arcminute, arcsecond.
    """
    lon_deg = float(wrap360(lon_deg))
    sign_index = int(lon_deg // 30.0)
    within = lon_deg - 30.0 * sign_index
    degree = int(within)
    minutes_float = (within - degree) * 60.0
    minute = int(minutes_float)
    second = (minutes_float - minute) * 60.0
    return sign_index, degree, minute, second


def format_zodiac_lon(
    lon_deg: float,
    precision: str = "auto",
    body: str | None = None,
) -> str:
    """
    Format longitude in a book-like zodiac style.

    Sun and Moon are usually printed to nearest arcsecond.
    Other planets are usually printed to nearest tenth of an arcminute.
    """
    body_lower = (body or "").lower()
    if precision == "auto":
        if body_lower in {"sun", "moon"}:
            precision = "arcsec"
        else:
            precision = "tenth_arcmin"

    sign_index, degree, minute, second = longitude_to_zodiac_parts(lon_deg)
    sign = ZODIAC_SIGNS[sign_index][1]

    if precision == "arcsec":
        sec_round = int(round(second))
        deg = degree
        minu = minute
        if sec_round == 60:
            sec_round = 0
            minu += 1
        if minu == 60:
            minu = 0
            deg += 1
        if deg == 30:
            deg = 0
            sign_index = (sign_index + 1) % 12
            sign = ZODIAC_SIGNS[sign_index][1]
        return f"{deg:02d} {sign} {minu:02d}' {sec_round:02d}\""

    if precision == "tenth_arcmin":
        total_minutes = (float(degree) * 60.0) + minute + second / 60.0
        total_tenths = int(round(total_minutes * 10.0))
        deg = total_tenths // 600
        rem_tenths = total_tenths - deg * 600
        minu = rem_tenths / 10.0
        if deg == 30:
            deg = 0
            sign_index = (sign_index + 1) % 12
            sign = ZODIAC_SIGNS[sign_index][1]
        return f"{deg:02d} {sign} {minu:04.1f}'"

    raise ValueError(f"Unknown precision: {precision!r}")


def make_tt_midnight_times(ts, year: int, month: int):
    """
    Build Skyfield Time array for every ET/TT midnight in a month.
    """
    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    import datetime as dt

    start = dt.date(year, month, 1)
    stop = dt.date(next_year, next_month, 1)
    n_days = (stop - start).days

    days = np.arange(1, n_days + 1, dtype=int)

    # The book says this edition is based on ET. TT is the modern practical
    # timescale to use for this kind of ephemeris-style comparison.
    t = ts.tt(year, month, days, 0, 0, 0)
    return days, t


def jpl_apparent_geocentric_ecliptic_longitudes(
    kernel_path: str,
    year: int,
    month: int,
    bodies: Iterable[str],
) -> list[dict]:
    """
    Generate apparent geocentric tropical/ecliptic-of-date longitudes from JPL/Skyfield.
    This is the table that should be closest to The American Ephemeris.
    """
    ts = load.timescale()
    eph = load(kernel_path)
    days, t = make_tt_midnight_times(ts, year, month)

    earth = eph["earth"]

    rows: list[dict] = []

    for body in bodies:
        target = eph[body]

        apparent = earth.at(t).observe(target).apparent()
        lat, lon, distance = apparent.frame_latlon(ecliptic_frame)

        lon_deg = wrap360(lon.degrees)
        lat_deg = lat.degrees
        dist_au = distance.au

        for i, day in enumerate(days):
            rows.append(
                {
                    "date": f"{year:04d}-{month:02d}-{int(day):02d}",
                    "day": int(day),
                    "body_key": body,
                    "body": BOOK_BODY_NAMES.get(body, body),
                    "jpl_lon_deg": float(lon_deg[i]),
                    "jpl_lat_deg": float(lat_deg[i]),
                    "jpl_distance_au": float(dist_au[i]),
                    "jpl_zodiac": format_zodiac_lon(float(lon_deg[i]), body=BOOK_BODY_NAMES.get(body, body)),
                }
            )

    return rows


def write_rows_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_jpl_book_style_csv(
    kernel_path: str,
    year: int,
    month: int,
    output_path: str,
) -> None:
    bodies = [
        "sun",
        "moon",
        "mercury barycenter",
        "venus barycenter",
        "mars barycenter",
        "jupiter barycenter",
        "saturn barycenter",
        "uranus barycenter",
        "neptune barycenter",
        "pluto barycenter",
    ]

    rows = jpl_apparent_geocentric_ecliptic_longitudes(
        kernel_path=kernel_path,
        year=year,
        month=month,
        bodies=bodies,
    )
    write_rows_csv(rows, output_path)


AU_M = 149_597_870_700.0


def ecliptic_lon_lat_from_icrf_vectors_m(vectors_m: np.ndarray, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert ICRF/J2000-ish vectors to longitude/latitude in the true
    ecliptic/equinox-of-date frame using Skyfield's ecliptic_frame.

    Returns lon_deg, lat_deg, distance_km.
    """
    vectors_m = np.asarray(vectors_m, dtype=float)

    lon_deg = np.empty(len(vectors_m), dtype=float)
    lat_deg = np.empty(len(vectors_m), dtype=float)
    distance_km = np.empty(len(vectors_m), dtype=float)

    for i, vec_m in enumerate(vectors_m):
        vec_au = vec_m / AU_M
        R = ecliptic_frame.rotation_at(times[i])
        v = R @ vec_au

        x, y, z = v
        r_xy = math.hypot(x, y)
        r = math.sqrt(x * x + y * y + z * z)

        lon_deg[i] = wrap360(math.degrees(math.atan2(y, x)))
        lat_deg[i] = math.degrees(math.atan2(z, r_xy))
        distance_km[i] = r * AU_M / 1e3

    return lon_deg, lat_deg, distance_km


def jpl_geometric_and_apparent_ecliptic(
    kernel_path: str,
    times,
    bodies: Iterable[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute both geometric and apparent JPL geocentric ecliptic-of-date
    quantities.

    geometric: same-time geocentric vector, useful for isolating dynamics
    apparent: book-like Skyfield apparent position, useful for matching the printed table
    """
    eph = load(kernel_path)
    earth = eph["earth"]

    earth_geom_m = earth.at(times).position.km.T * 1e3

    out: dict[str, dict[str, np.ndarray]] = {}

    for body in bodies:
        if body == "earth":
            continue

        target = eph[body]

        target_geom_m = target.at(times).position.km.T * 1e3
        geo_geom_m = target_geom_m - earth_geom_m

        geom_lon, geom_lat, geom_dist_km = ecliptic_lon_lat_from_icrf_vectors_m(
            geo_geom_m,
            times,
        )

        apparent = earth.at(times).observe(target).apparent()
        app_lat, app_lon, app_dist = apparent.frame_latlon(ecliptic_frame)

        out[body] = {
            "geom_lon_deg": wrap360(geom_lon),
            "geom_lat_deg": geom_lat,
            "geom_distance_km": geom_dist_km,
            "app_lon_deg": wrap360(app_lon.degrees),
            "app_lat_deg": app_lat.degrees,
            "app_distance_km": app_dist.km,
        }

    return out


def build_model_vs_jpl_ephemeris_rows(
    *,
    dates: list[str],
    times,
    model_positions_m: np.ndarray,
    bodies: tuple[str, ...],
    kernel_path: str,
) -> list[dict]:
    """
    Build comparison rows between integrated model positions and JPL/Skyfield.

    The important diagnostic is model geometric vs JPL geometric longitude.
    The book-like zodiac column uses the JPL apparent-geometric offset as an
    output-convention correction, so we can isolate model dynamics error.
    """
    body_to_index = {name: i for i, name in enumerate(bodies)}

    if "earth" not in body_to_index:
        raise ValueError("This comparison requires an explicit 'earth' body.")

    earth_idx = body_to_index["earth"]
    earth_model = model_positions_m[:, earth_idx, :]

    output_bodies = [b for b in bodies if b != "earth"]
    jpl = jpl_geometric_and_apparent_ecliptic(
        kernel_path=kernel_path,
        times=times,
        bodies=output_bodies,
    )

    rows: list[dict] = []

    for body in output_bodies:
        idx = body_to_index[body]
        body_name = BOOK_BODY_NAMES.get(body, body)

        model_geo_vec = model_positions_m[:, idx, :] - earth_model
        model_lon, model_lat, model_distance_km = ecliptic_lon_lat_from_icrf_vectors_m(
            model_geo_vec,
            times,
        )

        j = jpl[body]

        # Dynamics-only longitude residual.
        dyn_lon_err_arcsec = circular_angle_diff_deg(
            model_lon,
            j["geom_lon_deg"],
        ) * 3600.0

        lat_err_arcsec = (model_lat - j["geom_lat_deg"]) * 3600.0
        distance_err_km = model_distance_km - j["geom_distance_km"]

        # Apparent/book output correction inferred from JPL.
        # This is not yet a standalone apparent-position model; it is a
        # comparison tool that lets the model be judged in book coordinates
        # while isolating the dynamics error.
        app_minus_geom_deg = circular_angle_diff_deg(
            j["app_lon_deg"],
            j["geom_lon_deg"],
        )

        model_booklike_lon = wrap360(model_lon + app_minus_geom_deg)

        booklike_lon_err_arcsec = circular_angle_diff_deg(
            model_booklike_lon,
            j["app_lon_deg"],
        ) * 3600.0

        for k, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "body_key": body,
                    "body": body_name,
                    "model_zodiac_booklike": format_zodiac_lon(
                        float(model_booklike_lon[k]),
                        body=body_name,
                    ),
                    "jpl_zodiac": format_zodiac_lon(
                        float(j["app_lon_deg"][k]),
                        body=body_name,
                    ),
                    "lon_error_arcsec": float(booklike_lon_err_arcsec[k]),
                    "dyn_lon_error_arcsec": float(dyn_lon_err_arcsec[k]),
                    "lat_error_arcsec": float(lat_err_arcsec[k]),
                    "distance_error_km": float(distance_err_km[k]),
                    "model_lon_deg": float(model_booklike_lon[k]),
                    "jpl_app_lon_deg": float(j["app_lon_deg"][k]),
                    "model_geom_lon_deg": float(model_lon[k]),
                    "jpl_geom_lon_deg": float(j["geom_lon_deg"][k]),
                    "model_lat_deg": float(model_lat[k]),
                    "jpl_geom_lat_deg": float(j["geom_lat_deg"][k]),
                    "model_distance_km": float(model_distance_km[k]),
                    "jpl_geom_distance_km": float(j["geom_distance_km"][k]),
                }
            )

    return rows