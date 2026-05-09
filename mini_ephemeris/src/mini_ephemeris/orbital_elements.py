from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .nbody import G_SI, NBodyState


AU_M = 149_597_870_700.0
DAY_S = 86_400.0
JULIAN_YEAR_S = 365.25 * DAY_S
RAD_TO_DEG = 180.0 / math.pi
ARCSEC_PER_RAD = 206_264.80624709636
J2000_MEAN_OBLIQUITY_RAD = math.radians(23.43929111111111)

_TWO_PI = 2.0 * math.pi
_EPS = 1.0e-14


def seconds_to_years(seconds: float) -> float:
    return float(seconds) / JULIAN_YEAR_S


def meters_to_au(meters: float) -> float:
    return float(meters) / AU_M


def radians_to_degrees(angle_rad: float) -> float:
    if math.isnan(angle_rad):
        return math.nan
    return math.degrees(angle_rad)


def _wrap_2pi(angle_rad: float) -> float:
    if math.isnan(angle_rad):
        return math.nan
    return angle_rad % _TWO_PI


def _safe_acos(value: float) -> float:
    return math.acos(max(-1.0, min(1.0, value)))


def equatorial_to_ecliptic_j2000(vector: np.ndarray) -> np.ndarray:
    """Rotate an ICRF/J2000-equatorial vector into the J2000 ecliptic plane."""
    x, y, z = np.asarray(vector, dtype=float)
    cos_eps = math.cos(J2000_MEAN_OBLIQUITY_RAD)
    sin_eps = math.sin(J2000_MEAN_OBLIQUITY_RAD)
    return np.array(
        [
            x,
            cos_eps * y + sin_eps * z,
            -sin_eps * y + cos_eps * z,
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class OsculatingElements:
    """Heliocentric two-body osculating elements for one body."""

    body_name: str
    reference_plane: str
    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    longitude_ascending_node_rad: float
    argument_perihelion_rad: float
    longitude_perihelion_rad: float
    true_anomaly_rad: float
    mean_anomaly_rad: float
    mean_longitude_rad: float
    perihelion_m: float
    aphelion_m: float
    specific_energy_j_kg: float

    def as_output_row(self, time_years: float) -> dict[str, float | str]:
        return {
            "time_years": time_years,
            "body": self.body_name,
            "reference_plane": self.reference_plane,
            "a_au": meters_to_au(self.semi_major_axis_m),
            "e": self.eccentricity,
            "i_deg": radians_to_degrees(self.inclination_rad),
            "Omega_deg": radians_to_degrees(self.longitude_ascending_node_rad),
            "omega_deg": radians_to_degrees(self.argument_perihelion_rad),
            "varpi_deg": radians_to_degrees(self.longitude_perihelion_rad),
            "true_anomaly_deg": radians_to_degrees(self.true_anomaly_rad),
            "mean_anomaly_deg": radians_to_degrees(self.mean_anomaly_rad),
            "mean_longitude_deg": radians_to_degrees(self.mean_longitude_rad),
            "perihelion_au": meters_to_au(self.perihelion_m),
            "aphelion_au": meters_to_au(self.aphelion_m),
            "specific_energy_j_kg": self.specific_energy_j_kg,
        }


def heliocentric_osculating_elements(
    body_name: str,
    body_position_m: np.ndarray,
    body_velocity_m_s: np.ndarray,
    sun_position_m: np.ndarray,
    sun_velocity_m_s: np.ndarray,
    sun_mass_kg: float,
    body_mass_kg: float,
    *,
    G: float = G_SI,
    reference_plane: str = "ecliptic_j2000",
) -> OsculatingElements:
    """
    Compute heliocentric osculating elements from inertial Cartesian state.

    The elements use the instantaneous Sun-body two-body problem with
    mu = G * (M_sun + M_body). By default, angles are measured in the
    J2000 ecliptic plane, which is the natural diagnostic frame for
    long-term planetary stability outputs.
    """
    r_vec = np.asarray(body_position_m, dtype=float) - np.asarray(sun_position_m, dtype=float)
    v_vec = np.asarray(body_velocity_m_s, dtype=float) - np.asarray(sun_velocity_m_s, dtype=float)

    if reference_plane == "ecliptic_j2000":
        r_vec = equatorial_to_ecliptic_j2000(r_vec)
        v_vec = equatorial_to_ecliptic_j2000(v_vec)
    elif reference_plane == "input_xy":
        pass
    else:
        raise ValueError(
            "reference_plane must be 'ecliptic_j2000' or 'input_xy'."
        )

    r = float(np.linalg.norm(r_vec))
    v2 = float(np.dot(v_vec, v_vec))
    mu = float(G * (sun_mass_kg + body_mass_kg))

    if r <= 0.0 or mu <= 0.0:
        raise ValueError("Cannot compute orbital elements for a degenerate Sun-body state.")

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    if h <= 0.0:
        raise ValueError("Cannot compute orbital elements for a zero-angular-momentum state.")
    h_hat = h_vec / h

    k_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    n_vec = np.cross(k_hat, h_vec)
    n = float(np.linalg.norm(n_vec))

    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r
    e = float(np.linalg.norm(e_vec))

    specific_energy = 0.5 * v2 - mu / r
    if abs(specific_energy) > 0.0:
        a = -mu / (2.0 * specific_energy)
    else:
        a = math.inf

    inclination = _safe_acos(float(h_vec[2] / h))

    if n > _EPS:
        Omega = _wrap_2pi(math.atan2(float(n_vec[1]), float(n_vec[0])))
    else:
        Omega = 0.0

    if n > _EPS and e > _EPS:
        cos_omega = float(np.dot(n_vec, e_vec) / (n * e))
        sin_omega = float(np.dot(np.cross(n_vec, e_vec), h_hat) / (n * e))
        omega = _wrap_2pi(math.atan2(sin_omega, cos_omega))
    elif e > _EPS:
        # For a nearly reference-plane orbit the node is poorly defined, but
        # the perihelion longitude remains useful.
        omega = _wrap_2pi(math.atan2(float(e_vec[1]), float(e_vec[0])) - Omega)
    else:
        omega = 0.0

    varpi = _wrap_2pi(Omega + omega) if e > _EPS else 0.0

    if e > _EPS:
        cos_true = float(np.dot(e_vec, r_vec) / (e * r))
        sin_true = float(np.dot(np.cross(e_vec, r_vec), h_hat) / (e * r))
        true_anomaly = _wrap_2pi(math.atan2(sin_true, cos_true))
    elif n > _EPS:
        cos_u = float(np.dot(n_vec, r_vec) / (n * r))
        sin_u = float(np.dot(np.cross(n_vec, r_vec), h_hat) / (n * r))
        true_anomaly = _wrap_2pi(math.atan2(sin_u, cos_u))
    else:
        true_anomaly = _wrap_2pi(math.atan2(float(r_vec[1]), float(r_vec[0])))

    if e < 1.0 - _EPS:
        if e > _EPS:
            half_true = 0.5 * true_anomaly
            eccentric_anomaly = 2.0 * math.atan2(
                math.sqrt(max(0.0, 1.0 - e)) * math.sin(half_true),
                math.sqrt(1.0 + e) * math.cos(half_true),
            )
            mean_anomaly = _wrap_2pi(eccentric_anomaly - e * math.sin(eccentric_anomaly))
            mean_longitude = _wrap_2pi(varpi + mean_anomaly)
        else:
            mean_anomaly = true_anomaly
            mean_longitude = true_anomaly
    else:
        mean_anomaly = math.nan
        mean_longitude = math.nan

    if math.isfinite(a):
        perihelion = a * (1.0 - e)
        aphelion = a * (1.0 + e) if e < 1.0 else math.nan
    else:
        perihelion = math.nan
        aphelion = math.nan

    return OsculatingElements(
        body_name=body_name,
        reference_plane=reference_plane,
        semi_major_axis_m=float(a),
        eccentricity=e,
        inclination_rad=inclination,
        longitude_ascending_node_rad=Omega,
        argument_perihelion_rad=omega,
        longitude_perihelion_rad=varpi,
        true_anomaly_rad=true_anomaly,
        mean_anomaly_rad=mean_anomaly,
        mean_longitude_rad=mean_longitude,
        perihelion_m=float(perihelion),
        aphelion_m=float(aphelion),
        specific_energy_j_kg=float(specific_energy),
    )


def heliocentric_elements_for_state(
    state: NBodyState,
    body_names: Sequence[str],
    *,
    sun_index: int = 0,
    G: float = G_SI,
    reference_plane: str = "ecliptic_j2000",
) -> list[OsculatingElements]:
    """Return heliocentric osculating elements for all non-Sun bodies."""
    elements: list[OsculatingElements] = []
    sun_position = state.positions[sun_index]
    sun_velocity = state.velocities[sun_index]
    sun_mass = float(state.masses[sun_index])

    for index, name in enumerate(body_names):
        if index == sun_index:
            continue
        elements.append(
            heliocentric_osculating_elements(
                body_name=name,
                body_position_m=state.positions[index],
                body_velocity_m_s=state.velocities[index],
                sun_position_m=sun_position,
                sun_velocity_m_s=sun_velocity,
                sun_mass_kg=sun_mass,
                body_mass_kg=float(state.masses[index]),
                G=G,
                reference_plane=reference_plane,
            )
        )

    return elements
