from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skyfield.api import load

from .nbody import NBodyState, G_SI

def _gm_km3_s2_to_mass_kg(gm_km3_s2: float) -> float:
    """Convert GM in km^3/s^2 to mass in kg using the package G."""
    return gm_km3_s2 * 1.0e9 / G_SI

# # Approximate masses (kg)
# M_SUN = 1.98847e30
# M_MERCURY = 3.3011e23
# M_VENUS = 4.8675e24
# M_EARTH = 5.9722e24
# M_MARS = 6.4171e23
# M_JUPITER = 1.89813e27
# M_SATURN = 5.6834e26
# M_URANUS = 8.6810e25
# M_NEPTUNE = 1.02413e26
# M_MOON = 7.34767309e22

# BODY_MASSES = {
#     "sun": M_SUN,
#     "mercury barycenter": M_MERCURY,
#     "venus barycenter": M_VENUS,
#     "earth barycenter": M_EARTH + M_MOON,
#     "mars barycenter": M_MARS,
#     "jupiter barycenter": M_JUPITER,
#     "saturn barycenter": M_SATURN,
#     "uranus barycenter": M_URANUS,
#     "neptune barycenter": M_NEPTUNE,
# }

# DE431-consistent GM values (km^3 / s^2), from the kernel metadata/comments.
GM_SUN_KM3_S2 = 132_712_440_041.939400
GM_MERCURY_KM3_S2 = 22_031.780000
GM_VENUS_KM3_S2 = 324_858.592000
GM_EARTH_KM3_S2 = 398_600.435436
GM_MARS_KM3_S2 = 42_828.375214
GM_JUPITER_KM3_S2 = 126_712_764.800000
GM_SATURN_KM3_S2 = 37_940_585.200000
GM_URANUS_KM3_S2 = 5_794_548.600000
GM_NEPTUNE_KM3_S2 = 6_836_527.100580
GM_PLUTO_KM3_S2 = 977.000000
GM_MOON_KM3_S2 = 4_902.800066
GM_EARTH_MOON_BARYCENTER_KM3_S2 = 403_503.235502

M_SUN = _gm_km3_s2_to_mass_kg(GM_SUN_KM3_S2)
M_MERCURY = _gm_km3_s2_to_mass_kg(GM_MERCURY_KM3_S2)
M_VENUS = _gm_km3_s2_to_mass_kg(GM_VENUS_KM3_S2)
M_EARTH = _gm_km3_s2_to_mass_kg(GM_EARTH_KM3_S2)
M_MARS = _gm_km3_s2_to_mass_kg(GM_MARS_KM3_S2)
M_JUPITER = _gm_km3_s2_to_mass_kg(GM_JUPITER_KM3_S2)
M_SATURN = _gm_km3_s2_to_mass_kg(GM_SATURN_KM3_S2)
M_URANUS = _gm_km3_s2_to_mass_kg(GM_URANUS_KM3_S2)
M_NEPTUNE = _gm_km3_s2_to_mass_kg(GM_NEPTUNE_KM3_S2)
M_PLUTO = _gm_km3_s2_to_mass_kg(GM_PLUTO_KM3_S2)
M_MOON = _gm_km3_s2_to_mass_kg(GM_MOON_KM3_S2)
M_EARTH_MOON_BARYCENTER = _gm_km3_s2_to_mass_kg(GM_EARTH_MOON_BARYCENTER_KM3_S2)

BODY_GM_KM3_S2 = {
    "sun": GM_SUN_KM3_S2,
    "mercury barycenter": GM_MERCURY_KM3_S2,
    "venus barycenter": GM_VENUS_KM3_S2,
    "earth barycenter": GM_EARTH_MOON_BARYCENTER_KM3_S2,
    "earth": GM_EARTH_KM3_S2,
    "moon": GM_MOON_KM3_S2,
    "mars barycenter": GM_MARS_KM3_S2,
    "jupiter barycenter": GM_JUPITER_KM3_S2,
    "saturn barycenter": GM_SATURN_KM3_S2,
    "uranus barycenter": GM_URANUS_KM3_S2,
    "neptune barycenter": GM_NEPTUNE_KM3_S2,
    "pluto barycenter": GM_PLUTO_KM3_S2,
}

BODY_MASSES = {
    name: _gm_km3_s2_to_mass_kg(gm_km3_s2)
    for name, gm_km3_s2 in BODY_GM_KM3_S2.items()
}


@dataclass
class EphemerisConfig:
    kernel_path: str = "de431_part-2.bsp"


def load_kernel(config: EphemerisConfig = EphemerisConfig()):
    """Load the JPL kernel and return (timescale, ephemeris)."""
    print(f"[Ephem] Loading kernel from {config.kernel_path} ...", flush=True)
    ts = load.timescale()
    eph = load(config.kernel_path)
    print("[Ephem] Kernel loaded.", flush=True)
    return ts, eph


def _to_timescale(ts, t0: dt.datetime, offsets_s: np.ndarray):
    """
    Convert an array of offsets in seconds since t0 into a Skyfield Time object.

    Uses Julian-date arithmetic instead of datetime+timedelta so we can safely
    handle very long integrations (e.g., 10^4 years) without overflowing
    Python's datetime range.
    """
    offsets_s = np.asarray(offsets_s, dtype=float)

    # Base time as a Skyfield Time object
    t0_sf = ts.utc(t0)
    jd0 = t0_sf.tt  # TT Julian date (scalar float or 0-dim array)

    # Advance in days using the offsets in seconds
    jd = jd0 + offsets_s / 86400.0

    # Build a new Time object from the resulting Julian dates
    return ts.tt_jd(jd)


def initial_state_sun_earth(t0: dt.datetime,
                            config: EphemerisConfig = EphemerisConfig()
                            ) -> NBodyState:
    ts, eph = load_kernel(config)
    t = ts.utc(t0)

    sun = eph["sun"]
    earth = eph["earth"]

    rel = earth.at(t) - sun.at(t)
    r_earth = rel.position.km * 1e3
    v_earth = rel.velocity.km_per_s * 1e3

    positions = np.stack([
        np.zeros(3),
        r_earth,
    ], axis=0)

    velocities = np.stack([
        np.zeros(3),
        v_earth,
    ], axis=0)

    masses = np.array([M_SUN, M_EARTH], dtype=float)
    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def truth_positions_sun_earth(times_s: np.ndarray,
                              t0: dt.datetime,
                              config: EphemerisConfig = EphemerisConfig()
                              ) -> np.ndarray:
    ts, eph = load_kernel(config)
    sun = eph["sun"]
    earth = eph["earth"]

    t = _to_timescale(ts, t0, times_s)
    rel = earth.at(t) - sun.at(t)
    r_earth = rel.position.km.T * 1e3
    return r_earth


def initial_state_sun_earth_jupiter(t0: dt.datetime,
                                    config: EphemerisConfig = EphemerisConfig()
                                    ) -> NBodyState:
    ts, eph = load_kernel(config)
    t = ts.utc(t0)

    sun = eph["sun"]
    earth = eph["earth"]
    jupiter = eph["jupiter barycenter"]

    rel_e = earth.at(t) - sun.at(t)
    rel_j = jupiter.at(t) - sun.at(t)

    r_e = rel_e.position.km * 1e3
    v_e = rel_e.velocity.km_per_s * 1e3
    r_j = rel_j.position.km * 1e3
    v_j = rel_j.velocity.km_per_s * 1e3

    positions = np.stack([
        np.zeros(3),
        r_e,
        r_j,
    ], axis=0)

    velocities = np.stack([
        np.zeros(3),
        v_e,
        v_j,
    ], axis=0)

    masses = np.array([M_SUN, M_EARTH, M_JUPITER], dtype=float)
    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def truth_positions_sun_earth_jupiter(times_s: np.ndarray,
                                      t0: dt.datetime,
                                      config: EphemerisConfig = EphemerisConfig()
                                      ) -> np.ndarray:
    ts, eph = load_kernel(config)
    sun = eph["sun"]
    earth = eph["earth"]
    jupiter = eph["jupiter barycenter"]

    t = _to_timescale(ts, t0, times_s)

    rel_e = earth.at(t) - sun.at(t)
    rel_j = jupiter.at(t) - sun.at(t)

    r_e = rel_e.position.km.T * 1e3
    r_j = rel_j.position.km.T * 1e3

    M = len(times_s)
    positions = np.zeros((M, 3, 3), dtype=float)
    positions[:, 1, :] = r_e
    positions[:, 2, :] = r_j
    return positions


def initial_state_sun_earth_moon(t0: dt.datetime,
                                 config: EphemerisConfig = EphemerisConfig()
                                 ) -> NBodyState:
    ts, eph = load_kernel(config)
    t = ts.utc(t0)

    sun = eph["sun"]
    earth = eph["earth"]
    moon = eph["moon"]

    rel_e = earth.at(t) - sun.at(t)
    rel_m = moon.at(t) - sun.at(t)

    r_e = rel_e.position.km * 1e3
    v_e = rel_e.velocity.km_per_s * 1e3
    r_m = rel_m.position.km * 1e3
    v_m = rel_m.velocity.km_per_s * 1e3

    positions = np.stack([
        np.zeros(3),
        r_e,
        r_m,
    ], axis=0)

    velocities = np.stack([
        np.zeros(3),
        v_e,
        v_m,
    ], axis=0)

    masses = np.array([M_SUN, M_EARTH, M_MOON], dtype=float)
    return NBodyState(positions=positions, velocities=velocities, masses=masses)


def truth_positions_sun_earth_moon(times_s: np.ndarray,
                                   t0: dt.datetime,
                                   config: EphemerisConfig = EphemerisConfig()
                                   ) -> np.ndarray:
    ts, eph = load_kernel(config)
    sun = eph["sun"]
    earth = eph["earth"]
    moon = eph["moon"]

    t = _to_timescale(ts, t0, times_s)

    rel_e = earth.at(t) - sun.at(t)
    rel_m = moon.at(t) - sun.at(t)

    r_e = rel_e.position.km.T * 1e3
    r_m = rel_m.position.km.T * 1e3

    M = len(times_s)
    positions = np.zeros((M, 3, 3), dtype=float)
    positions[:, 1, :] = r_e
    positions[:, 2, :] = r_m
    return positions


SOLAR_SYSTEM_BODIES_DEFAULT: tuple[str, ...] = (
    "sun",
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
    "jupiter barycenter",
    "saturn barycenter",
    "uranus barycenter",
    "neptune barycenter",
)


SOLAR_SYSTEM_BODIES_WITH_PLUTO: tuple[str, ...] = (
    "sun",
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
    "jupiter barycenter",
    "saturn barycenter",
    "uranus barycenter",
    "neptune barycenter",
    "pluto barycenter",
)


SOLAR_SYSTEM_BODIES_EARTH_MOON = (
    "sun",
    "mercury barycenter",
    "venus barycenter",
    "earth",
    "moon",
    "mars barycenter",
    "jupiter barycenter",
    "saturn barycenter",
    "uranus barycenter",
    "neptune barycenter",
)


def solar_system_body_list(include_pluto: bool = False) -> tuple[str, ...]:
    return SOLAR_SYSTEM_BODIES_WITH_PLUTO if include_pluto else SOLAR_SYSTEM_BODIES_DEFAULT


def solar_system_body_list_earth_moon() -> tuple[str, ...]:
    return SOLAR_SYSTEM_BODIES_EARTH_MOON


def initial_state_solar_system_barycentric(
    t0: dt.datetime,
    bodies: Sequence[str] = SOLAR_SYSTEM_BODIES_DEFAULT,
    config: EphemerisConfig = EphemerisConfig(),
) -> NBodyState:
    print("[Ephem] Building initial barycentric state...", flush=True)
    ts, eph = load_kernel(config)
    t = ts.utc(t0)

    positions = []
    velocities = []
    masses = []

    for name in bodies:
        r = eph[name].at(t).position.km * 1e3
        v = eph[name].at(t).velocity.km_per_s * 1e3
        positions.append(r)
        velocities.append(v)
        if name not in BODY_MASSES:
            raise KeyError(f"No mass known for body {name!r}.")
        masses.append(BODY_MASSES[name])

    print(f"[Ephem] Initial state constructed for bodies: {bodies}", flush=True)

    return NBodyState(
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )


def initial_state_solar_system_barycentric_time(
    t,
    bodies: tuple[str, ...],
    config: EphemerisConfig | None = None,
    verbose: bool = False,
):
    """
    Build initial barycentric state from a Skyfield Time object.

    This is useful for American Ephemeris-style comparisons because the
    book is based on ET/TT-style midnight, not UTC midnight.
    """
    from .nbody import NBodyState

    if config is None:
        config = EphemerisConfig()

    ts, eph = load_kernel(config)

    positions = []
    velocities = []
    masses = []

    for name in bodies:
        if name not in BODY_MASSES:
            raise KeyError(f"No GM/mass known for body {name!r}")

        vec = eph[name].at(t)
        positions.append(vec.position.km * 1e3)
        velocities.append(vec.velocity.km_per_s * 1e3)
        masses.append(BODY_MASSES[name])

    if verbose:
        print(f"[Ephem] Initial state constructed from Skyfield Time for bodies: {bodies}")

    return NBodyState(
        positions=np.asarray(positions, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )


def truth_positions_solar_system_barycentric(
    times_s: np.ndarray,
    t0: dt.datetime,
    bodies: Sequence[str] = SOLAR_SYSTEM_BODIES_DEFAULT,
    config: EphemerisConfig = EphemerisConfig(),
) -> np.ndarray:
    ts, eph = load_kernel(config)
    t = _to_timescale(ts, t0, times_s)
    M = len(times_s)
    N = len(bodies)
    positions = np.zeros((M, N, 3), dtype=float)

    for j, name in enumerate(bodies):
        r = eph[name].at(t).position.km.T * 1e3
        positions[:, j, :] = r

    return positions


def _unit_vector(vec: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Return vec / |vec| with a useful error if the norm is degenerate."""
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Cannot normalize degenerate {name}.")
    return vec / norm


def lunar_geocentric_basis(
    state: NBodyState,
    earth_index: int,
    moon_index: int,
) -> dict[str, np.ndarray]:
    """
    Return the instantaneous geocentric lunar orbital basis at the state epoch.

    The basis is built from the Moon's position and velocity relative to Earth:

        r_hat : Earth -> Moon radial direction
        h_hat : orbital angular-momentum direction
        t_hat : prograde transverse/tangential direction in the orbital plane

    Positive t_hat points in the direction of the Moon's instantaneous
    geocentric transverse motion.
    """
    r_geo = state.positions[moon_index] - state.positions[earth_index]
    v_geo = state.velocities[moon_index] - state.velocities[earth_index]

    r_hat = _unit_vector(r_geo, name="geocentric lunar radius vector")
    h_hat = _unit_vector(
        np.cross(r_geo, v_geo),
        name="geocentric lunar angular momentum",
    )
    t_hat = _unit_vector(
        np.cross(h_hat, r_hat),
        name="geocentric lunar tangential direction",
    )

    return {
        "r_hat": r_hat,
        "t_hat": t_hat,
        "h_hat": h_hat,
        "r_geo_m": r_geo,
        "v_geo_m_s": v_geo,
    }


def apply_lunar_tangential_velocity_correction(
    state: NBodyState,
    earth_index: int,
    moon_index: int,
    dv_t_m_s: float,
    *,
    preserve_emb_momentum: bool = True,
) -> NBodyState:
    """
    Apply a small lunar geocentric tangential velocity correction.

    Parameters
    ----------
    state
        Input NBodyState. It is not modified in-place.
    earth_index, moon_index
        Indices of the explicit Earth and Moon entries in ``state``.
    dv_t_m_s
        Desired change in the Moon's relative geocentric tangential velocity,
        in m/s. Positive is prograde; negative slows the Moon along-track.
    preserve_emb_momentum
        If True, split the relative velocity correction between Earth and Moon
        so that Earth-Moon barycenter linear momentum is unchanged:

            v_moon  += m_earth / (m_earth + m_moon) * dv_rel
            v_earth -= m_moon  / (m_earth + m_moon) * dv_rel

        If False, only the Moon velocity is changed.

    Returns
    -------
    NBodyState
        A corrected copy of the input state.
    """
    return apply_lunar_velocity_correction(
        state,
        earth_index=earth_index,
        moon_index=moon_index,
        dv_r_m_s=0.0,
        dv_t_m_s=dv_t_m_s,
        dv_h_m_s=0.0,
        preserve_emb_momentum=preserve_emb_momentum,
    )


def apply_lunar_velocity_correction(
    state: NBodyState,
    earth_index: int,
    moon_index: int,
    *,
    dv_r_m_s: float = 0.0,
    dv_t_m_s: float = 0.0,
    dv_h_m_s: float = 0.0,
    preserve_emb_momentum: bool = True,
) -> NBodyState:
    """
    Apply a small 3D correction to the Moon's initial geocentric velocity.

    The correction is expressed in the instantaneous lunar orbital basis:

        dv_r : radial, positive from Earth toward Moon
        dv_t : tangential, positive prograde in the orbital plane
        dv_h : out-of-plane, positive along geocentric angular momentum

    If ``preserve_emb_momentum`` is true, the relative velocity change is split
    between Earth and Moon so the Earth-Moon barycenter momentum is unchanged.
    """
    corrected = state.copy()
    basis = lunar_geocentric_basis(corrected, earth_index, moon_index)
    dv_rel = (
        float(dv_r_m_s) * basis["r_hat"]
        + float(dv_t_m_s) * basis["t_hat"]
        + float(dv_h_m_s) * basis["h_hat"]
    )

    if preserve_emb_momentum:
        m_earth = float(corrected.masses[earth_index])
        m_moon = float(corrected.masses[moon_index])
        m_total = m_earth + m_moon

        corrected.velocities[moon_index] += (m_earth / m_total) * dv_rel
        corrected.velocities[earth_index] -= (m_moon / m_total) * dv_rel
    else:
        corrected.velocities[moon_index] += dv_rel

    return corrected
