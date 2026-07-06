from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import TextIO
import warnings as warning_lib

import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from .advanced_integrators import (
    acceleration_newtonian,
    acceleration_newtonian_gr_sun,
    pack_state,
    rhs_solve_ivp,
    unpack_state,
    velocity_verlet_step_generic,
)
from .chaos_diagnostics import (
    cosine_between_scaled_deltas,
    delta_arrays_from_scaled_delta_vector,
    lyapunov_time_years,
    make_radial_perturbed_state,
    renormalize_to_scaled_norm,
    scaled_phase_space_component_diagnostics,
    scaled_phase_space_delta_vector,
    scaled_phase_space_delta_vector_from_arrays,
    tangent_acceleration_newtonian,
)
from .ephem import (
    EphemerisConfig,
    initial_state_solar_system_barycentric,
    solar_system_body_list,
)
from .nbody import G_SI, NBodyState
from .orbital_elements import (
    ARCSEC_PER_RAD,
    AU_M,
    DAY_S,
    JULIAN_YEAR_S,
    equatorial_to_ecliptic_j2000,
    heliocentric_elements_for_state,
    seconds_to_years,
)
from .stability_diagnostics import (
    InvariantReference,
    PairwiseMinimumTracker,
    invariant_diagnostics_row,
    invariant_reference,
    total_angular_momentum_vector,
    total_newtonian_energy,
)


MODE_DESCRIPTION = "stability mode: physical reduced model, no empirical lunar calibration"

EMPIRICAL_LUNAR_FLAGS = {
    "--earth-j2",
    "--lunar-calibration-file",
    "--lunar-calibration-profile",
    "--moon-dv-r-mm-s",
    "--moon-dv-t-mm-s",
    "--moon-dv-h-mm-s",
    "--moon-a-r-1e-15-m-s2",
    "--moon-a-t-1e-15-m-s2",
    "--moon-a-h-1e-15-m-s2",
    "--moon-lon-plot",
    "--moon-lat-plot",
    "--moon-lon-ylim-arcsec",
    "--moon-lat-ylim-arcsec",
    "--no-preserve-emb-momentum",
}

STABILITY_TIMESERIES_FIELDS = [
    "time_years",
    "body",
    "x_au",
    "y_au",
    "z_au",
    "vx_au_per_year",
    "vy_au_per_year",
    "vz_au_per_year",
    "heliocentric_x_au",
    "heliocentric_y_au",
    "heliocentric_z_au",
    "heliocentric_vx_au_per_year",
    "heliocentric_vy_au_per_year",
    "heliocentric_vz_au_per_year",
    "heliocentric_r_au",
    "heliocentric_speed_au_per_year",
]

ORBITAL_ELEMENT_FIELDS = [
    "time_years",
    "body",
    "reference_plane",
    "a_au",
    "e",
    "i_deg",
    "Omega_deg",
    "omega_deg",
    "varpi_deg",
    "true_anomaly_deg",
    "mean_anomaly_deg",
    "mean_longitude_deg",
    "perihelion_au",
    "aphelion_au",
    "specific_energy_j_kg",
]

INVARIANT_FIELDS = [
    "time_years",
    "energy_j",
    "energy_abs_drift_j",
    "energy_rel_drift",
    "angular_momentum_norm_kg_m2_s",
    "angular_momentum_abs_drift_kg_m2_s",
    "angular_momentum_rel_drift",
    "angular_momentum_direction_drift_arcsec",
    "com_x_au",
    "com_y_au",
    "com_z_au",
    "com_vx_au_per_year",
    "com_vy_au_per_year",
    "com_vz_au_per_year",
    "com_position_drift_au",
    "com_velocity_drift_au_per_year",
]

MIN_SEPARATION_FIELDS = [
    "body_i",
    "body_j",
    "min_separation_au",
    "min_separation_km",
    "time_years",
]

LYAPUNOV_FIELDS = [
    "time_years",
    "separation_norm",
    "pre_renorm_separation_norm",
    "post_renorm_separation_norm",
    "target_norm",
    "growth_factor",
    "log_growth_increment",
    "cumulative_log_growth",
    "local_lambda_1_per_year",
    "running_lambda_1_per_year",
    "lyapunov_time_years",
    "max_position_separation_m",
    "max_velocity_separation_m_s",
    "dominant_body_in_norm",
    "dominant_component_type",
    "renorm_interval_years_actual",
    "cosine_with_previous_delta_direction",
    "cosine_with_initial_delta_direction",
    "direction_reset_suspected",
]

NO_RENORM_SEPARATION_FIELDS = [
    "time_years",
    "separation_norm",
    "log_separation_norm",
    "position_separation_m",
    "velocity_separation_m_s",
]

POINCARE_FIELDS = [
    "time_years",
    "body",
    "plane",
    "direction",
    "x_au",
    "y_au",
    "z_au",
    "vx_au_per_year",
    "vy_au_per_year",
    "vz_au_per_year",
    "r_au",
    "speed_au_per_year",
]

FREQUENCY_MAP_FIELDS = [
    "body",
    "variable",
    "window_start_years",
    "window_end_years",
    "dominant_frequency_rad_per_year",
    "dominant_period_years",
    "amplitude",
    "estimated_frequency_drift_rad_per_year",
    "n_samples",
    "warning",
]

FLI_MEGNO_FIELDS = [
    "time_years",
    "tangent_norm",
    "log_tangent_norm",
    "finite_time_lambda_1_per_year",
    "fli",
    "megno_lite",
    "running_megno_lite_slope",
    "classification_hint",
    "warning",
]

MEGNO_FIELDS = [
    "time_years",
    "megno",
    "mean_megno",
    "finite_time_lyapunov_estimate",
    "backend",
    "integrator",
    "gr_model",
    "warnings",
]

TWO_BODY_VALIDATION_FIELDS = [
    "duration_years",
    "step_days",
    "n_steps",
    "max_energy_rel_drift",
    "final_energy_rel_drift",
    "max_angular_momentum_rel_drift",
    "final_angular_momentum_rel_drift",
    "max_a_drift_au",
    "final_a_drift_au",
    "max_e_drift",
    "final_e_drift",
    "estimated_perihelion_drift_arcsec_per_century",
    "kepler_period_years_initial",
    "measured_period_years",
    "runtime_seconds",
]

LYAPUNOV_BODY_NAME_MAP = {
    "mercury": "mercury barycenter",
    "venus": "venus barycenter",
    "earth": "earth barycenter",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
}

PLANET_BODY_NAME_MAP = {
    **LYAPUNOV_BODY_NAME_MAP,
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
    "pluto": "pluto barycenter",
}

MODEL_SCOPES = {
    "two_body_mercury": (
        "sun",
        "mercury barycenter",
    ),
    "two_body_jupiter": (
        "sun",
        "jupiter barycenter",
    ),
    "two_body_saturn": (
        "sun",
        "saturn barycenter",
    ),
    "inner": (
        "sun",
        "mercury barycenter",
        "venus barycenter",
        "earth barycenter",
        "mars barycenter",
    ),
}

TWO_BODY_MODEL_SCOPES = {
    "two_body_mercury": "mercury barycenter",
    "two_body_jupiter": "jupiter barycenter",
    "two_body_saturn": "saturn barycenter",
}

LYAPUNOV_ALL_BODY_NAMES = (
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
)


@dataclass
class CsvOutputs:
    stability_timeseries: csv.DictWriter
    orbital_elements: csv.DictWriter
    invariants: csv.DictWriter
    files: tuple[TextIO, ...]
    paths: dict[str, Path]

    def flush(self) -> None:
        for file_obj in self.files:
            file_obj.flush()

    def close(self) -> None:
        for file_obj in self.files:
            file_obj.close()


@dataclass
class LyapunovConfig:
    body_choice: str
    body_indices: tuple[int, ...]
    body_names: tuple[str, ...]
    model_scope: str
    gr_model: str
    perturbation_m: float
    target_norm: float
    renorm_years: float
    fit_start_years: float
    fit_end_years: float
    seed: int | None
    norm_name: str
    method: str
    no_renorm: bool
    debug: bool
    displacement_m_by_body: dict[str, float]
    sun_position_compensation_m: tuple[float, float, float]


@dataclass
class LyapunovOutputs:
    writer: csv.DictWriter
    file: TextIO
    csv_path: Path
    summary_path: Path
    plot_path: Path
    no_renorm_path: Path | None = None

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@dataclass
class ReboundMegnoOutputs:
    writer: csv.DictWriter
    file: TextIO
    csv_path: Path
    summary_path: Path
    plot_path: Path

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@dataclass
class ReboundMegnoSample:
    time_years: float
    megno: float
    mean_megno: float
    finite_time_lyapunov_estimate: float
    warnings: str = ""


@dataclass
class ReboundMegnoResult:
    samples: list[ReboundMegnoSample]
    final_megno: float
    final_mean_megno: float
    estimated_lyapunov_if_available: float
    final_lcn_raw: float
    last_finite_lcn: float | None
    last_finite_lcn_time_years: float | None
    megno_slope_window_estimates: list[dict[str, float | int | str | None]]
    classification_hint: str
    caveats: list[str]
    lcn_available: bool


@dataclass
class ReboundResumeInfo:
    requested: str | None
    archive_path: str | None
    resumed_from_time_years: float
    megno_state_validated: bool
    duplicate_rows_removed: dict[str, int]
    warnings: list[str]


@dataclass
class LyapunovSample:
    time_years: float
    separation_norm: float
    pre_renorm_separation_norm: float
    post_renorm_separation_norm: float
    target_norm: float
    growth_factor: float
    log_growth_increment: float
    cumulative_log_growth: float
    local_lambda_1_per_year: float
    running_lambda_1_per_year: float
    lyapunov_time_years: float
    max_position_separation_m: float
    max_velocity_separation_m_s: float
    dominant_body_in_norm: str
    dominant_component_type: str
    renorm_interval_years_actual: float
    cosine_with_previous_delta_direction: float
    cosine_with_initial_delta_direction: float
    direction_reset_suspected: bool


@dataclass
class LyapunovResult:
    samples: list[LyapunovSample]
    config: LyapunovConfig
    fit: dict[str, float | int | None]
    warnings: list[str]
    debug_warnings: list[str]
    final_running_lambda_1_per_year: float | None
    final_running_lyapunov_time_years: float | None


@dataclass
class PoincareConfig:
    body_choice: str
    body_name: str
    body_index: int
    plane: str
    direction: str


@dataclass
class PoincareOutputs:
    writer: csv.DictWriter
    file: TextIO
    csv_path: Path
    plot_path: Path

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@dataclass(frozen=True)
class PoincareSample:
    time_years: float
    body: str
    plane: str
    direction: str
    x_au: float
    y_au: float
    z_au: float
    vx_au_per_year: float
    vy_au_per_year: float
    vz_au_per_year: float
    r_au: float
    speed_au_per_year: float


@dataclass(frozen=True)
class FliMegnoSample:
    time_years: float
    tangent_norm: float
    log_tangent_norm: float
    finite_time_lambda_1_per_year: float
    fli: float
    megno_lite: float
    running_megno_lite_slope: float
    classification_hint: str
    warning: str


@dataclass
class CheckpointData:
    path: Path | None
    time_s: float
    n_records: int
    state: NBodyState
    reference_energy_j: float
    reference_angular_momentum: np.ndarray
    reference_angular_momentum_norm: float
    reference_com_position_m: np.ndarray
    reference_com_velocity_m_s: np.ndarray
    min_distance_m: np.ndarray
    min_time_s: np.ndarray
    extrema: dict[str, float]
    lyapunov_cumulative_log_growth: float
    lyapunov_last_renorm_s: float
    lyapunov_next_renorm_s: float
    lyapunov_previous_sample_norm: float
    lyapunov_delta_positions: np.ndarray | None
    lyapunov_delta_velocities: np.ndarray | None
    lyapunov_initial_delta_vector: np.ndarray | None
    lyapunov_previous_delta_vector: np.ndarray | None
    lyapunov_samples: list[LyapunovSample]
    rng_state: dict | None
    config_hash: str


@dataclass
class RunningLinearFit:
    """Tiny online least-squares accumulator for one diagnostic slope."""

    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_xy += x * y

    def slope(self) -> float:
        denominator = self.n * self.sum_xx - self.sum_x * self.sum_x
        if self.n < 2 or denominator == 0.0:
            return math.nan
        return (self.n * self.sum_xy - self.sum_x * self.sum_y) / denominator


@dataclass
class TwoBodyValidationTracker:
    """Per-step numerical diagnostics for a Sun+planet validation problem."""

    body_name: str
    body_index: int
    sun_index: int
    energy0_j: float
    angular_momentum0: np.ndarray
    angular_momentum0_norm: float
    a0_m: float
    e0: float
    varpi0_rad: float
    previous_varpi_rad: float
    unwrapped_varpi_rad: float
    mean_anomaly0_rad: float
    previous_mean_anomaly_rad: float
    unwrapped_mean_anomaly_rad: float
    kepler_period_years_initial: float
    varpi_fit: RunningLinearFit
    max_energy_rel_drift: float = 0.0
    final_energy_rel_drift: float = 0.0
    max_angular_momentum_rel_drift: float = 0.0
    final_angular_momentum_rel_drift: float = 0.0
    max_a_drift_au: float = 0.0
    final_a_drift_au: float = 0.0
    max_e_drift: float = 0.0
    final_e_drift: float = 0.0
    last_time_years: float = 0.0

    @classmethod
    def create(
        cls,
        state: NBodyState,
        body_names: tuple[str, ...],
        *,
        sun_index: int,
    ) -> "TwoBodyValidationTracker":
        if len(body_names) != 2 or body_names[sun_index] != "sun":
            raise ValueError("Two-body validation requires Sun + one planetary barycenter.")
        body_index = 1 - sun_index
        body_name = body_names[body_index]
        elements = heliocentric_elements_for_state(
            state,
            body_names,
            sun_index=sun_index,
        )[0]
        energy0 = total_newtonian_energy(state, G=G_SI)
        angular0 = total_angular_momentum_vector(state)
        angular0_norm = float(np.linalg.norm(angular0))
        mu = G_SI * (
            float(state.masses[sun_index]) + float(state.masses[body_index])
        )
        period_s = 2.0 * math.pi * math.sqrt(
            elements.semi_major_axis_m**3 / mu
        )
        tracker = cls(
            body_name=body_name,
            body_index=body_index,
            sun_index=sun_index,
            energy0_j=energy0,
            angular_momentum0=angular0,
            angular_momentum0_norm=angular0_norm,
            a0_m=elements.semi_major_axis_m,
            e0=elements.eccentricity,
            varpi0_rad=elements.longitude_perihelion_rad,
            previous_varpi_rad=elements.longitude_perihelion_rad,
            unwrapped_varpi_rad=elements.longitude_perihelion_rad,
            mean_anomaly0_rad=elements.mean_anomaly_rad,
            previous_mean_anomaly_rad=elements.mean_anomaly_rad,
            unwrapped_mean_anomaly_rad=elements.mean_anomaly_rad,
            kepler_period_years_initial=seconds_to_years(period_s),
            varpi_fit=RunningLinearFit(),
        )
        tracker.varpi_fit.add(0.0, 0.0)
        return tracker

    @staticmethod
    def _wrapped_delta(current_rad: float, previous_rad: float) -> float:
        if not (math.isfinite(current_rad) and math.isfinite(previous_rad)):
            return math.nan
        return (current_rad - previous_rad + math.pi) % (2.0 * math.pi) - math.pi

    def update(
        self,
        time_s: float,
        state: NBodyState,
        body_names: tuple[str, ...],
        *,
        update_orbital_elements: bool = True,
    ) -> None:
        time_years = seconds_to_years(time_s)
        self.last_time_years = time_years

        energy = total_newtonian_energy(state, G=G_SI)
        energy_scale = abs(self.energy0_j) if self.energy0_j != 0.0 else 1.0
        energy_rel = (energy - self.energy0_j) / energy_scale
        self.final_energy_rel_drift = energy_rel
        self.max_energy_rel_drift = max(self.max_energy_rel_drift, abs(energy_rel))

        angular = total_angular_momentum_vector(state)
        angular_scale = self.angular_momentum0_norm if self.angular_momentum0_norm != 0.0 else 1.0
        angular_rel = float(np.linalg.norm(angular - self.angular_momentum0)) / angular_scale
        self.final_angular_momentum_rel_drift = angular_rel
        self.max_angular_momentum_rel_drift = max(
            self.max_angular_momentum_rel_drift,
            abs(angular_rel),
        )

        if not update_orbital_elements:
            return

        elements = heliocentric_elements_for_state(
            state,
            body_names,
            sun_index=self.sun_index,
        )[0]
        a_drift_au = (elements.semi_major_axis_m - self.a0_m) / AU_M
        e_drift = elements.eccentricity - self.e0
        self.final_a_drift_au = a_drift_au
        self.final_e_drift = e_drift
        self.max_a_drift_au = max(self.max_a_drift_au, abs(a_drift_au))
        self.max_e_drift = max(self.max_e_drift, abs(e_drift))

        varpi_delta = self._wrapped_delta(
            elements.longitude_perihelion_rad,
            self.previous_varpi_rad,
        )
        if math.isfinite(varpi_delta):
            self.unwrapped_varpi_rad += varpi_delta
            self.previous_varpi_rad = elements.longitude_perihelion_rad
            self.varpi_fit.add(
                time_years,
                self.unwrapped_varpi_rad - self.varpi0_rad,
            )

        mean_delta = self._wrapped_delta(
            elements.mean_anomaly_rad,
            self.previous_mean_anomaly_rad,
        )
        if math.isfinite(mean_delta):
            self.unwrapped_mean_anomaly_rad += mean_delta
            self.previous_mean_anomaly_rad = elements.mean_anomaly_rad

    def result(
        self,
        *,
        duration_s: float,
        step_days: float,
        n_steps: int,
        runtime_seconds: float = math.nan,
    ) -> "TwoBodyValidationResult":
        elapsed_years = seconds_to_years(duration_s)
        mean_cycles = (
            (self.unwrapped_mean_anomaly_rad - self.mean_anomaly0_rad)
            / (2.0 * math.pi)
        )
        measured_period_years = (
            elapsed_years / mean_cycles
            if mean_cycles > 0.0 and math.isfinite(mean_cycles)
            else math.nan
        )
        perihelion_drift = self.varpi_fit.slope() * ARCSEC_PER_RAD * 100.0
        return TwoBodyValidationResult(
            duration_years=elapsed_years,
            step_days=step_days,
            n_steps=n_steps,
            max_energy_rel_drift=self.max_energy_rel_drift,
            final_energy_rel_drift=self.final_energy_rel_drift,
            max_angular_momentum_rel_drift=self.max_angular_momentum_rel_drift,
            final_angular_momentum_rel_drift=self.final_angular_momentum_rel_drift,
            max_a_drift_au=self.max_a_drift_au,
            final_a_drift_au=self.final_a_drift_au,
            max_e_drift=self.max_e_drift,
            final_e_drift=self.final_e_drift,
            estimated_perihelion_drift_arcsec_per_century=perihelion_drift,
            kepler_period_years_initial=self.kepler_period_years_initial,
            measured_period_years=measured_period_years,
            runtime_seconds=runtime_seconds,
        )


@dataclass(frozen=True)
class TwoBodyValidationResult:
    duration_years: float
    step_days: float
    n_steps: int
    max_energy_rel_drift: float
    final_energy_rel_drift: float
    max_angular_momentum_rel_drift: float
    final_angular_momentum_rel_drift: float
    max_a_drift_au: float
    final_a_drift_au: float
    max_e_drift: float
    final_e_drift: float
    estimated_perihelion_drift_arcsec_per_century: float
    kepler_period_years_initial: float
    measured_period_years: float
    runtime_seconds: float

    def as_row(self) -> dict[str, float]:
        return {
            "duration_years": self.duration_years,
            "step_days": self.step_days,
            "n_steps": self.n_steps,
            "max_energy_rel_drift": self.max_energy_rel_drift,
            "final_energy_rel_drift": self.final_energy_rel_drift,
            "max_angular_momentum_rel_drift": self.max_angular_momentum_rel_drift,
            "final_angular_momentum_rel_drift": self.final_angular_momentum_rel_drift,
            "max_a_drift_au": self.max_a_drift_au,
            "final_a_drift_au": self.final_a_drift_au,
            "max_e_drift": self.max_e_drift,
            "final_e_drift": self.final_e_drift,
            "estimated_perihelion_drift_arcsec_per_century": self.estimated_perihelion_drift_arcsec_per_century,
            "kepler_period_years_initial": self.kepler_period_years_initial,
            "measured_period_years": self.measured_period_years,
            "runtime_seconds": self.runtime_seconds,
        }


@dataclass
class IntegrationResult:
    final_state: NBodyState
    actual_duration_s: float
    n_steps: int
    n_records: int
    min_tracker: PairwiseMinimumTracker
    extrema: dict[str, float]
    min_separation_sampling: str
    lyapunov_result: LyapunovResult | None = None
    two_body_validation: TwoBodyValidationResult | None = None
    poincare_samples: list[PoincareSample] | None = None
    runtime_warnings: list[str] | None = None
    rebound_megno_result: ReboundMegnoResult | None = None
    rebound_resume_info: ReboundResumeInfo | None = None


def parse_start_datetime(text: str) -> dt.datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        if "T" in value or " " in value:
            parsed = dt.datetime.fromisoformat(value)
        else:
            parsed_date = dt.date.fromisoformat(value)
            parsed = dt.datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=dt.timezone.utc,
            )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected ISO date or datetime, for example 2000-01-01."
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def sanitize_tag(tag: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in tag.strip()
    )
    return cleaned or "stability"


def default_lyapunov_fit_start_years(duration_years: float, renorm_years: float) -> float:
    return min(float(renorm_years), max(0.0, 0.2 * float(duration_years)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run long-term Solar System stability experiments using the "
            "physical reduced barycenter model."
        )
    )
    parser.add_argument(
        "--kernel-path",
        default="de431_part-2.bsp",
        help="Path to a JPL BSP kernel used only for the initial state.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_start_datetime,
        default=parse_start_datetime("2000-01-01"),
        help="UTC ISO start date or datetime for the initial state.",
    )
    parser.add_argument(
        "--duration-years",
        type=float,
        default=1000.0,
        help="Integration duration in Julian years.",
    )
    parser.add_argument(
        "--step-days",
        type=float,
        default=4.0,
        help=(
            "Fixed leapfrog timestep in days, or DOP853 maximum internal step "
            "for validation runs."
        ),
    )
    parser.add_argument(
        "--record-every-years",
        type=float,
        default=1.0,
        help="CSV output cadence in Julian years.",
    )
    parser.add_argument(
        "--include-pluto",
        action="store_true",
        help="Include Pluto barycenter in the reduced model.",
    )
    parser.add_argument(
        "--model-scope",
        choices=["full", "inner", "two_body_mercury", "two_body_jupiter", "two_body_saturn"],
        default="full",
        help=(
            "Reduced model scope. two_body_jupiter and two_body_saturn are "
            "clean near-integrable validation cases; two_body_mercury is a "
            "hard inner-planet stress test."
        ),
    )
    parser.add_argument(
        "--gr-model",
        choices=["none", "sun"],
        default="none",
        help="Relativity model: none or Sun-centered 1PN GR.",
    )
    parser.add_argument(
        "--integrator",
        choices=["leapfrog", "dop853"],
        default="leapfrog",
        help="Integrator. Leapfrog is the default long-term stability integrator.",
    )
    parser.add_argument(
        "--backend",
        choices=["inhouse", "rebound"],
        default="inhouse",
        help="Production backend. inhouse preserves the existing leapfrog/DOP853 path.",
    )
    parser.add_argument(
        "--rebound-integrator",
        choices=["whfast", "ias15"],
        default="whfast",
        help="REBOUND integrator used when --backend rebound.",
    )
    parser.add_argument(
        "--rebound-gr-model",
        choices=["none", "gr", "gr_potential", "gr_full"],
        default="none",
        help="REBOUNDx GR force used when --backend rebound. Not used for in-house runs.",
    )
    parser.add_argument(
        "--rebound-ias15-epsilon",
        type=float,
        default=1.0e-10,
        help="IAS15 epsilon when --backend rebound --rebound-integrator ias15.",
    )
    parser.add_argument(
        "--rebound-simulationarchive",
        default=None,
        help="Optional REBOUND SimulationArchive path for production backend snapshots.",
    )
    parser.add_argument(
        "--rebound-archive-interval-years",
        type=float,
        default=None,
        help="Cadence for REBOUND SimulationArchive snapshots in Julian years.",
    )
    parser.add_argument(
        "--rebound-resume",
        default=None,
        help=(
            "Resume a REBOUND backend run from a SimulationArchive. Use 'latest' "
            "to load the latest snapshot from --rebound-simulationarchive, or pass "
            "a specific archive path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="../output",
        help="Directory for stability CSV and JSON outputs.",
    )
    parser.add_argument(
        "--tag",
        default="stability",
        help="Tag inserted into output file names.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    parser.add_argument(
        "--checkpoint-every-years",
        type=float,
        default=None,
        help="Write in-house leapfrog restart checkpoints at this cadence in Julian years.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for checkpoint_*.npz files. Defaults to <output-dir>/checkpoints_<tag>.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from a checkpoint .npz file, or from the latest checkpoint_*.npz in a directory.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Keep only the last N checkpoint files in the checkpoint directory.",
    )
    parser.add_argument(
        "--with-lyapunov",
        action="store_true",
        help="Run a finite-time Benettin-style maximum Lyapunov estimate.",
    )
    parser.add_argument(
        "--lyapunov-body",
        choices=["mercury", "venus", "earth", "mars", "jupiter", "saturn", "all"],
        default="mercury",
        help="Body or inner-planet set to perturb radially for the Lyapunov estimate.",
    )
    parser.add_argument(
        "--lyapunov-perturbation-m",
        type=float,
        default=1.0,
        help="Initial radial perturbation scale in meters.",
    )
    parser.add_argument(
        "--lyapunov-renorm-years",
        type=float,
        default=10.0,
        help="Benettin renormalization interval in Julian years.",
    )
    parser.add_argument(
        "--lyapunov-fit-start-years",
        type=float,
        default=None,
        help=(
            "Start of the linear fit window in Julian years. Defaults to one "
            "renormalization interval, capped at 20 percent of duration."
        ),
    )
    parser.add_argument(
        "--lyapunov-fit-end-years",
        type=float,
        default=None,
        help="End of the linear fit window in Julian years. Defaults to the run duration.",
    )
    parser.add_argument(
        "--lyapunov-seed",
        type=int,
        default=42,
        help="Random seed used when --lyapunov-body all is selected.",
    )
    parser.add_argument(
        "--lyapunov-norm",
        choices=["scaled_phase_space"],
        default="scaled_phase_space",
        help="Phase-space norm for Lyapunov renormalization.",
    )
    parser.add_argument(
        "--lyapunov-method",
        choices=["tangent", "two_trajectory"],
        default="tangent",
        help=(
            "Lyapunov propagation method. tangent integrates the Newtonian "
            "variational equations through the leapfrog map; two_trajectory "
            "keeps the older paired-trajectory finite-difference debug mode."
        ),
    )
    parser.add_argument(
        "--lyapunov-debug",
        action="store_true",
        help="Emit additional Lyapunov validation warnings and console diagnostics.",
    )
    parser.add_argument(
        "--lyapunov-no-renorm",
        action="store_true",
        help="Do not rescale the perturbed trajectory; record raw separation growth.",
    )
    parser.add_argument(
        "--with-poincare",
        action="store_true",
        help="Record exploratory heliocentric Poincare-style section crossings.",
    )
    parser.add_argument(
        "--poincare-body",
        choices=["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"],
        default="mercury",
        help="Planetary barycenter used for Poincare-style section crossings.",
    )
    parser.add_argument(
        "--poincare-plane",
        choices=["z", "y"],
        default="z",
        help="Heliocentric ecliptic coordinate plane used for section crossings.",
    )
    parser.add_argument(
        "--poincare-direction",
        choices=["positive", "negative", "both"],
        default="positive",
        help="Crossing direction for the selected Poincare-style section.",
    )
    parser.add_argument(
        "--with-frequency-map",
        action="store_true",
        help="Run optional NAFF-lite/FFT-lite frequency-map analysis from recorded orbital elements.",
    )
    parser.add_argument(
        "--frequency-window-years",
        type=float,
        default=1000.0,
        help="Sliding FFT-lite window length in Julian years.",
    )
    parser.add_argument(
        "--frequency-step-years",
        type=float,
        default=500.0,
        help="Step between FFT-lite windows in Julian years.",
    )
    parser.add_argument(
        "--frequency-bodies",
        default="all",
        help="Comma-separated body list for frequency analysis, or all.",
    )
    parser.add_argument(
        "--frequency-min-samples",
        type=int,
        default=32,
        help="Minimum orbital-element samples required inside each frequency window.",
    )
    parser.add_argument(
        "--with-fli",
        action="store_true",
        help="Write finite-time FLI-lite tangent-growth indicators.",
    )
    parser.add_argument(
        "--with-megno-lite",
        action="store_true",
        help="Write finite-time MEGNO-lite tangent-growth indicators.",
    )
    parser.add_argument(
        "--with-megno",
        action="store_true",
        help="Run REBOUND-native finite-time MEGNO diagnostics. Requires --backend rebound.",
    )
    parser.add_argument(
        "--megno-record-every-years",
        type=float,
        default=None,
        help="Cadence for REBOUND-native MEGNO output. Defaults to --record-every-years.",
    )
    parser.add_argument(
        "--megno-duration-scaling-mode",
        action="store_true",
        help="Annotate MEGNO outputs as part of a duration-scaling validation workflow.",
    )
    parser.add_argument(
        "--rebound-chaos-method",
        choices=["megno"],
        default="megno",
        help="REBOUND-native chaos diagnostic method.",
    )
    parser.add_argument(
        "--with-rebound-lyapunov",
        action="store_true",
        help="Record REBOUND-native Lyapunov characteristic number when available after MEGNO init.",
    )
    parser.add_argument(
        "--megno-seed",
        type=int,
        default=12345,
        help="Seed passed to REBOUND init_megno for reproducible variational initial conditions.",
    )
    parser.add_argument(
        "--fli-method",
        choices=["tangent"],
        default="tangent",
        help="FLI-lite propagation method. Only Newtonian tangent propagation is supported.",
    )
    parser.add_argument(
        "--fli-record-every-renorm",
        action="store_true",
        help="Record FLI-lite samples at every Lyapunov renormalization interval.",
    )
    parser.add_argument(
        "--megno-record-every-renorm",
        action="store_true",
        help="Record MEGNO-lite samples at every Lyapunov renormalization interval.",
    )
    return parser


def reject_empirical_lunar_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> None:
    rejected: list[str] = []
    for token in argv:
        flag = token.split("=", 1)[0]
        if (
            flag in EMPIRICAL_LUNAR_FLAGS
            or flag.startswith("--moon-dv-")
            or flag.startswith("--moon-a-")
            or flag.startswith("--lunar-calibration")
        ):
            rejected.append(flag)

    if rejected:
        unique = ", ".join(sorted(set(rejected)))
        parser.error(
            "Empirical lunar calibration and explicit Earth-Moon tuning flags "
            f"are not accepted in stability mode: {unique}. Use the American "
            "Ephemeris CLIs for fitted short-range reproduction."
        )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.duration_years <= 0.0:
        parser.error("--duration-years must be positive.")
    if args.step_days <= 0.0:
        parser.error("--step-days must be positive.")
    if args.record_every_years <= 0.0:
        parser.error("--record-every-years must be positive.")
    if args.megno_record_every_years is not None and args.megno_record_every_years <= 0.0:
        parser.error("--megno-record-every-years must be positive.")
    if args.rebound_ias15_epsilon <= 0.0:
        parser.error("--rebound-ias15-epsilon must be positive.")
    if args.rebound_archive_interval_years is not None and args.rebound_archive_interval_years <= 0.0:
        parser.error("--rebound-archive-interval-years must be positive.")
    if args.backend == "inhouse" and (
        args.rebound_simulationarchive is not None
        or args.rebound_archive_interval_years is not None
        or args.rebound_resume is not None
    ):
        parser.error("REBOUND SimulationArchive options require --backend rebound.")
    if args.backend == "rebound":
        if args.with_lyapunov or args.with_fli or args.with_megno_lite:
            parser.error(
                "Tangent Lyapunov, FLI-lite, and MEGNO-lite are not implemented for --backend rebound. "
                "Use --backend inhouse for those diagnostics."
            )
        if args.with_poincare:
            parser.error(
                "Poincare section streaming is not implemented for --backend rebound yet. "
                "Run trajectory/orbital-element outputs first or use --backend inhouse."
            )
        if args.resume_from_checkpoint is not None:
            parser.error("--resume-from-checkpoint is currently an in-house checkpoint option.")
        if args.checkpoint_every_years is not None:
            parser.error("--checkpoint-every-years is currently an in-house checkpoint option.")
        if args.gr_model != "none":
            parser.error("Use --rebound-gr-model for --backend rebound; keep --gr-model none.")
        if args.rebound_simulationarchive is not None and args.rebound_archive_interval_years is None:
            parser.error("--rebound-simulationarchive requires --rebound-archive-interval-years.")
        if args.rebound_resume is not None and args.rebound_simulationarchive is None:
            parser.error("--rebound-resume requires --rebound-simulationarchive.")
        if args.rebound_resume is not None and args.rebound_archive_interval_years is None:
            parser.error("--rebound-resume requires --rebound-archive-interval-years.")
    if args.with_megno and args.backend != "rebound":
        parser.error("--with-megno is REBOUND-native and requires --backend rebound.")
    if args.with_rebound_lyapunov and not args.with_megno:
        parser.error("--with-rebound-lyapunov requires --with-megno.")
    if args.checkpoint_every_years is not None:
        if args.integrator != "leapfrog":
            parser.error("--checkpoint-every-years currently supports --integrator leapfrog only.")
        if args.checkpoint_every_years <= 0.0:
            parser.error("--checkpoint-every-years must be positive.")
    if args.resume_from_checkpoint is not None and args.integrator != "leapfrog":
        parser.error("--resume-from-checkpoint currently supports --integrator leapfrog only.")
    if args.keep_checkpoints < 1:
        parser.error("--keep-checkpoints must be at least 1.")
    if args.model_scope in TWO_BODY_MODEL_SCOPES and args.gr_model != "none":
        parser.error("Two-body validation scopes require --gr-model none.")
    if args.with_poincare and args.integrator != "leapfrog":
        parser.error("--with-poincare currently requires --integrator leapfrog.")
    if args.with_frequency_map:
        if args.frequency_window_years <= 0.0:
            parser.error("--frequency-window-years must be positive.")
        if args.frequency_step_years <= 0.0:
            parser.error("--frequency-step-years must be positive.")
        if args.frequency_min_samples < 4:
            parser.error("--frequency-min-samples must be at least 4.")
    tangent_diagnostics = args.with_lyapunov or args.with_fli or args.with_megno_lite
    if args.with_fli or args.with_megno_lite:
        if args.fli_method != "tangent":
            parser.error("--fli-method currently supports tangent only.")
        if args.lyapunov_method != "tangent":
            parser.error("--with-fli/--with-megno-lite require --lyapunov-method tangent.")
    if tangent_diagnostics:
        if args.integrator != "leapfrog":
            parser.error("Tangent chaos diagnostics currently require --integrator leapfrog.")
        if args.lyapunov_method == "tangent" and args.gr_model != "none":
            parser.error(
                "Validated tangent Lyapunov/FLI/MEGNO-lite diagnostics currently support --gr-model none only."
            )
        if args.lyapunov_perturbation_m <= 0.0:
            parser.error("--lyapunov-perturbation-m must be positive.")
        if args.lyapunov_renorm_years <= 0.0:
            parser.error("--lyapunov-renorm-years must be positive.")
        fit_start = (
            args.lyapunov_fit_start_years
            if args.lyapunov_fit_start_years is not None
            else default_lyapunov_fit_start_years(
                args.duration_years,
                args.lyapunov_renorm_years,
            )
        )
        fit_end = (
            args.lyapunov_fit_end_years
            if args.lyapunov_fit_end_years is not None
            else args.duration_years
        )
        if fit_start < 0.0:
            parser.error("--lyapunov-fit-start-years must be non-negative.")
        if fit_end <= fit_start:
            parser.error("--lyapunov-fit-end-years must be greater than the fit start.")
        if fit_start > args.duration_years:
            parser.error("--lyapunov-fit-start-years must be inside the integration span.")
        if fit_end > args.duration_years:
            parser.error("--lyapunov-fit-end-years must not exceed --duration-years.")


def output_paths(
    output_dir: Path,
    tag: str,
    *,
    with_lyapunov: bool = False,
    lyapunov_no_renorm: bool = False,
    with_poincare: bool = False,
    poincare_body: str = "mercury",
    with_frequency_map: bool = False,
    with_fli_megno: bool = False,
    with_megno: bool = False,
    model_scope: str = "full",
) -> dict[str, Path]:
    paths = {
        "stability_timeseries": output_dir / f"stability_timeseries_{tag}.csv",
        "orbital_elements": output_dir / f"orbital_elements_{tag}.csv",
        "invariants": output_dir / f"invariants_{tag}.csv",
        "min_separations": output_dir / f"min_separations_{tag}.csv",
        "summary": output_dir / f"summary_{tag}.json",
    }
    if with_lyapunov:
        paths.update(
            {
                "lyapunov": output_dir / f"lyapunov_{tag}.csv",
                "lyapunov_summary": output_dir / f"lyapunov_summary_{tag}.json",
                "lyapunov_growth_plot": output_dir / f"lyapunov_growth_{tag}.png",
            }
        )
        if lyapunov_no_renorm:
            paths["no_renorm_separation"] = output_dir / f"no_renorm_separation_{tag}.csv"
    if with_poincare:
        paths["poincare"] = output_dir / f"poincare_{tag}_{poincare_body}.csv"
        paths["poincare_plot"] = output_dir / f"poincare_{tag}_{poincare_body}.png"
    if with_frequency_map:
        paths["frequency_map"] = output_dir / f"frequency_map_{tag}.csv"
    if with_fli_megno:
        paths["fli_megno"] = output_dir / f"fli_megno_{tag}.csv"
        paths["fli_megno_summary"] = output_dir / f"fli_megno_summary_{tag}.json"
    if with_megno:
        paths["megno"] = output_dir / f"megno_{tag}.csv"
        paths["megno_summary"] = output_dir / f"megno_summary_{tag}.json"
        paths["megno_growth_plot"] = output_dir / f"megno_growth_{tag}.png"
    if model_scope in TWO_BODY_MODEL_SCOPES:
        paths["two_body_validation"] = output_dir / f"two_body_validation_{tag}.csv"
    return paths


def _csv_needs_header(path: Path, *, append: bool) -> bool:
    return (not append) or (not path.exists()) or path.stat().st_size == 0


def open_csv_outputs(paths: dict[str, Path], *, append: bool = False) -> CsvOutputs:
    mode = "a" if append else "w"
    stability_needs_header = _csv_needs_header(paths["stability_timeseries"], append=append)
    elements_needs_header = _csv_needs_header(paths["orbital_elements"], append=append)
    invariants_needs_header = _csv_needs_header(paths["invariants"], append=append)
    stability_file = paths["stability_timeseries"].open(mode, newline="")
    elements_file = paths["orbital_elements"].open(mode, newline="")
    invariants_file = paths["invariants"].open(mode, newline="")

    stability_writer = csv.DictWriter(
        stability_file,
        fieldnames=STABILITY_TIMESERIES_FIELDS,
    )
    elements_writer = csv.DictWriter(
        elements_file,
        fieldnames=ORBITAL_ELEMENT_FIELDS,
    )
    invariants_writer = csv.DictWriter(
        invariants_file,
        fieldnames=INVARIANT_FIELDS,
    )

    if stability_needs_header:
        stability_writer.writeheader()
    if elements_needs_header:
        elements_writer.writeheader()
    if invariants_needs_header:
        invariants_writer.writeheader()

    return CsvOutputs(
        stability_timeseries=stability_writer,
        orbital_elements=elements_writer,
        invariants=invariants_writer,
        files=(stability_file, elements_file, invariants_file),
        paths=paths,
    )


def open_lyapunov_outputs(paths: dict[str, Path], *, append: bool = False) -> LyapunovOutputs:
    file_obj = paths["lyapunov"].open("a" if append else "w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=LYAPUNOV_FIELDS)
    if not append:
        writer.writeheader()
    return LyapunovOutputs(
        writer=writer,
        file=file_obj,
        csv_path=paths["lyapunov"],
        summary_path=paths["lyapunov_summary"],
        plot_path=paths["lyapunov_growth_plot"],
        no_renorm_path=paths.get("no_renorm_separation"),
    )


def open_poincare_outputs(paths: dict[str, Path]) -> PoincareOutputs:
    file_obj = paths["poincare"].open("w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=POINCARE_FIELDS)
    writer.writeheader()
    return PoincareOutputs(
        writer=writer,
        file=file_obj,
        csv_path=paths["poincare"],
        plot_path=paths["poincare_plot"],
    )


def open_rebound_megno_outputs(paths: dict[str, Path], *, append: bool = False) -> ReboundMegnoOutputs:
    needs_header = _csv_needs_header(paths["megno"], append=append)
    file_obj = paths["megno"].open("a" if append else "w", newline="")
    writer = csv.DictWriter(file_obj, fieldnames=MEGNO_FIELDS)
    if needs_header:
        writer.writeheader()
    return ReboundMegnoOutputs(
        writer=writer,
        file=file_obj,
        csv_path=paths["megno"],
        summary_path=paths["megno_summary"],
        plot_path=paths["megno_growth_plot"],
    )


def select_acceleration_model(
    gr_model: str,
    *,
    sun_index: int,
):
    if gr_model == "none":
        return acceleration_newtonian, {}
    if gr_model == "sun":
        return acceleration_newtonian_gr_sun, {"sun_index": sun_index}
    raise ValueError(f"Unsupported gr_model: {gr_model!r}")


def optional_import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def rebound_state_from_sim(sim, masses: np.ndarray) -> NBodyState:
    n_real = int(getattr(sim, "N_real", len(masses)) or len(masses))
    positions = []
    velocities = []
    for particle in sim.particles[:n_real]:
        positions.append([particle.x, particle.y, particle.z])
        velocities.append([particle.vx, particle.vy, particle.vz])
    return NBodyState(
        positions=np.array(positions, dtype=float),
        velocities=np.array(velocities, dtype=float),
        masses=masses.copy(),
    )


def build_rebound_simulation(
    rebound,
    state: NBodyState,
    *,
    integrator: str,
    step_s: float,
    ias15_epsilon: float,
):
    sim = rebound.Simulation()
    sim.G = G_SI
    for position, velocity, mass in zip(state.positions, state.velocities, state.masses):
        sim.add(
            m=float(mass),
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            vz=float(velocity[2]),
        )
    sim.integrator = integrator
    if integrator == "whfast":
        sim.dt = step_s
    elif integrator == "ias15" and hasattr(sim, "ri_ias15"):
        sim.ri_ias15.epsilon = ias15_epsilon
    return sim


def add_reboundx_gr_force(sim, gr_model: str) -> str:
    if gr_model == "none":
        return "none"
    reboundx = optional_import_module("reboundx")
    if reboundx is None:
        raise RuntimeError(
            "reboundx is not installed. Use --rebound-gr-model none or install reboundx."
        )
    rebx = reboundx.Extras(sim)
    force = rebx.load_force(gr_model)
    rebx.add_force(force)
    force.params["c"] = 299_792_458.0
    return gr_model


def configure_rebound_simulationarchive(
    sim,
    path: Path,
    *,
    interval_s: float,
    delete_existing: bool = True,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(sim, "save_to_file"):
        sim.save_to_file(str(path), interval=interval_s, delete_file=delete_existing)
        return "save_to_file"
    if hasattr(sim, "automateSimulationArchive"):
        sim.automateSimulationArchive(str(path), interval=interval_s, deletefile=delete_existing)
        return "automateSimulationArchive"
    raise RuntimeError(
        "This REBOUND version does not expose save_to_file or automateSimulationArchive."
    )


def load_rebound_archive_snapshot(rebound, archive_path: Path):
    if not archive_path.exists():
        raise RuntimeError(f"REBOUND SimulationArchive does not exist: {archive_path}")
    if archive_path.stat().st_size == 0:
        raise RuntimeError(f"REBOUND SimulationArchive is empty: {archive_path}")
    errors: list[str] = []
    try:
        sim = rebound.Simulation(str(archive_path))
        if getattr(sim, "N", 0) > 0:
            return sim
        errors.append("Simulation(path): loaded zero particles")
    except Exception as exc:  # pragma: no cover - depends on REBOUND internals.
        errors.append(f"Simulation(path): {exc}")
    loaders = (
        getattr(rebound.Simulation, "from_file", None),
        getattr(rebound.Simulation, "from_simulationarchive", None),
    )
    for loader in loaders:
        if loader is None:
            continue
        try:
            sim = loader(str(archive_path))
            if getattr(sim, "N", 0) > 0:
                return sim
            errors.append(f"{getattr(loader, '__name__', 'loader')}: loaded zero particles")
        except Exception as exc:  # pragma: no cover - depends on REBOUND internals.
            errors.append(f"{getattr(loader, '__name__', 'loader')}: {exc}")
    joined = "; ".join(errors) if errors else "no archive loader was available"
    raise RuntimeError(f"Could not load REBOUND SimulationArchive {archive_path}: {joined}")


def _csv_has_nul(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                return False
            if b"\x00" in chunk:
                return True


def truncate_csv_after_time(
    path: Path,
    *,
    time_column: str,
    max_time_years: float,
    tolerance_years: float = 1.0e-9,
) -> int:
    if not path.exists():
        return 0
    if _csv_has_nul(path):
        raise RuntimeError(f"NUL bytes detected in CSV output: {path}")
    with path.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = reader.fieldnames
        if not fieldnames or time_column not in fieldnames:
            raise RuntimeError(f"CSV lacks required {time_column!r} column: {path}")
        kept: list[dict[str, str]] = []
        removed = 0
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise RuntimeError(f"Malformed CSV row {row_number} in {path}")
            try:
                time_years = float(row[time_column])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Could not parse {time_column!r} on row {row_number} in {path}"
                ) from exc
            if time_years <= max_time_years + tolerance_years:
                kept.append(row)
            else:
                removed += 1
    if removed:
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    return removed


def count_csv_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    if _csv_has_nul(path):
        raise RuntimeError(f"NUL bytes detected in CSV output: {path}")
    with path.open(newline="") as file_obj:
        return sum(1 for _ in csv.DictReader(file_obj))


def extrema_from_invariants_csv(path: Path) -> dict[str, float]:
    extrema = initial_extrema()
    if not path.exists() or path.stat().st_size == 0:
        return extrema
    if _csv_has_nul(path):
        raise RuntimeError(f"NUL bytes detected in CSV output: {path}")
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            update_extrema(extrema, row)
    return extrema


def min_tracker_from_csv(path: Path, body_names: tuple[str, ...]) -> PairwiseMinimumTracker:
    tracker = PairwiseMinimumTracker.create(body_names)
    if not path.exists() or path.stat().st_size == 0:
        return tracker
    if _csv_has_nul(path):
        raise RuntimeError(f"NUL bytes detected in CSV output: {path}")
    pair_lookup = {
        (body_names[i], body_names[j]): pair_index
        for pair_index, (i, j) in enumerate(tracker.pair_indices)
    }
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            key = (row.get("body_i", ""), row.get("body_j", ""))
            pair_index = pair_lookup.get(key)
            if pair_index is None:
                continue
            try:
                tracker.min_distance_m[pair_index] = float(row["min_separation_au"]) * AU_M
                tracker.min_time_s[pair_index] = float(row["time_years"]) * JULIAN_YEAR_S
            except (KeyError, TypeError, ValueError):
                continue
    return tracker


def read_rebound_megno_samples(path: Path) -> list[ReboundMegnoSample]:
    samples: list[ReboundMegnoSample] = []
    if not path.exists() or path.stat().st_size == 0:
        return samples
    if _csv_has_nul(path):
        raise RuntimeError(f"NUL bytes detected in CSV output: {path}")
    with path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            try:
                samples.append(
                    ReboundMegnoSample(
                        time_years=float(row["time_years"]),
                        megno=float(row["megno"]) if row.get("megno") not in {"", None} else math.nan,
                        mean_megno=(
                            float(row["mean_megno"])
                            if row.get("mean_megno") not in {"", None}
                            else math.nan
                        ),
                        finite_time_lyapunov_estimate=(
                            float(row["finite_time_lyapunov_estimate"])
                            if row.get("finite_time_lyapunov_estimate") not in {"", None}
                            else math.nan
                        ),
                        warnings=row.get("warnings", ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return samples


def resolve_rebound_resume_archive(args: argparse.Namespace) -> Path | None:
    requested = getattr(args, "rebound_resume", None)
    if requested is None:
        return None
    if requested == "latest":
        return Path(args.rebound_simulationarchive)
    return Path(requested)


def stability_body_list(model_scope: str, *, include_pluto: bool = False) -> tuple[str, ...]:
    if model_scope == "full":
        return tuple(solar_system_body_list(include_pluto=include_pluto))
    if model_scope in MODEL_SCOPES:
        return MODEL_SCOPES[model_scope]
    raise ValueError(f"Unsupported model_scope: {model_scope!r}")


def resolve_lyapunov_body_selection(
    body_choice: str,
    body_names: tuple[str, ...],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if body_choice == "all":
        selected_names = tuple(
            name
            for name in LYAPUNOV_ALL_BODY_NAMES
            if name in body_names
        )
        if len(selected_names) == 0:
            raise ValueError("--lyapunov-body all found no supported inner bodies in this model scope.")
    else:
        selected_names = (LYAPUNOV_BODY_NAME_MAP[body_choice],)

    missing = [name for name in selected_names if name not in body_names]
    if missing:
        raise ValueError(f"Lyapunov body selection is missing from model: {missing}")

    return (
        tuple(body_names.index(name) for name in selected_names),
        selected_names,
    )


def resolve_planet_body_choice(
    body_choice: str,
    body_names: tuple[str, ...],
    *,
    option_name: str,
) -> tuple[int, str]:
    body_name = PLANET_BODY_NAME_MAP[body_choice]
    if body_name not in body_names:
        raise ValueError(
            f"{option_name}={body_choice!r} is not present in model_scope; available bodies are {body_names}."
        )
    return body_names.index(body_name), body_name


def build_poincare_config(
    args: argparse.Namespace,
    body_names: tuple[str, ...],
) -> PoincareConfig:
    body_index, body_name = resolve_planet_body_choice(
        args.poincare_body,
        body_names,
        option_name="--poincare-body",
    )
    return PoincareConfig(
        body_choice=args.poincare_body,
        body_name=body_name,
        body_index=body_index,
        plane=args.poincare_plane,
        direction=args.poincare_direction,
    )


def parse_frequency_body_choices(
    text: str,
    body_names: tuple[str, ...],
) -> tuple[str, ...]:
    raw = text.strip().lower()
    if raw == "all":
        return tuple(name for name in body_names if name != "sun")
    selected: list[str] = []
    for token in raw.split(","):
        choice = token.strip()
        if not choice:
            continue
        if choice not in PLANET_BODY_NAME_MAP:
            raise ValueError(
                f"Unsupported --frequency-bodies entry {choice!r}; use comma-separated planet names or all."
            )
        body_name = PLANET_BODY_NAME_MAP[choice]
        if body_name not in body_names:
            raise ValueError(
                f"--frequency-bodies entry {choice!r} is not present in model_scope."
            )
        selected.append(body_name)
    if not selected:
        raise ValueError("--frequency-bodies resolved to an empty body list.")
    return tuple(dict.fromkeys(selected))


def build_lyapunov_config(
    args: argparse.Namespace,
    state0: NBodyState,
    body_names: tuple[str, ...],
    *,
    sun_index: int,
) -> tuple[LyapunovConfig, NBodyState]:
    body_choice = args.lyapunov_body
    if (
        body_choice == "mercury"
        and "mercury barycenter" not in body_names
        and args.model_scope in TWO_BODY_MODEL_SCOPES
    ):
        validation_body_name = TWO_BODY_MODEL_SCOPES[args.model_scope]
        reverse_map = {value: key for key, value in LYAPUNOV_BODY_NAME_MAP.items()}
        body_choice = reverse_map[validation_body_name]
    body_indices, selected_names = resolve_lyapunov_body_selection(
        body_choice,
        body_names,
    )
    perturbation = make_radial_perturbed_state(
        state0,
        body_indices=body_indices,
        body_names=selected_names,
        sun_index=sun_index,
        perturbation_m=args.lyapunov_perturbation_m,
        seed=args.lyapunov_seed,
        preserve_barycenter=True,
    )
    fit_start = (
        args.lyapunov_fit_start_years
        if args.lyapunov_fit_start_years is not None
        else default_lyapunov_fit_start_years(
            args.duration_years,
            args.lyapunov_renorm_years,
        )
    )
    fit_end = (
        args.lyapunov_fit_end_years
        if args.lyapunov_fit_end_years is not None
        else args.duration_years
    )

    config = LyapunovConfig(
        body_choice=body_choice,
        body_indices=body_indices,
        body_names=selected_names,
        model_scope=args.model_scope,
        gr_model=args.gr_model,
        perturbation_m=args.lyapunov_perturbation_m,
        target_norm=perturbation.target_norm,
        renorm_years=args.lyapunov_renorm_years,
        fit_start_years=float(fit_start),
        fit_end_years=float(fit_end),
        seed=args.lyapunov_seed,
        norm_name=args.lyapunov_norm,
        method=args.lyapunov_method,
        no_renorm=args.lyapunov_no_renorm,
        debug=args.lyapunov_debug,
        displacement_m_by_body=perturbation.displacement_m_by_body,
        sun_position_compensation_m=perturbation.sun_position_compensation_m,
    )
    return config, perturbation.state


def _finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value_float = float(value)
    if not math.isfinite(value_float):
        return None
    return value_float


def _csv_float(value: float) -> float | str:
    return value if math.isfinite(value) else ""


def stability_config_hash(args: argparse.Namespace, body_names: tuple[str, ...]) -> str:
    payload = {
        "kernel_path": str(args.kernel_path),
        "start_date": args.start_date.isoformat(),
        "model_scope": args.model_scope,
        "body_names": body_names,
        "step_days": float(args.step_days),
        "record_every_years": float(args.record_every_years),
        "gr_model": args.gr_model,
        "integrator": args.integrator,
        "with_lyapunov": bool(args.with_lyapunov),
        "lyapunov_method": getattr(args, "lyapunov_method", None),
        "lyapunov_body": getattr(args, "lyapunov_body", None),
        "lyapunov_perturbation_m": getattr(args, "lyapunov_perturbation_m", None),
        "lyapunov_renorm_years": getattr(args, "lyapunov_renorm_years", None),
        "with_fli": bool(getattr(args, "with_fli", False)),
        "with_megno_lite": bool(getattr(args, "with_megno_lite", False)),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_with_delta(
    reference: NBodyState,
    delta_positions_m: np.ndarray,
    delta_velocities_m_s: np.ndarray,
) -> NBodyState:
    """Build a temporary state equal to reference plus a tangent delta."""
    return NBodyState(
        positions=reference.positions + delta_positions_m,
        velocities=reference.velocities + delta_velocities_m_s,
        masses=reference.masses,
    )


def write_lyapunov_sample(
    outputs: LyapunovOutputs,
    sample: LyapunovSample,
) -> None:
    outputs.writer.writerow(
        {
            "time_years": sample.time_years,
            "separation_norm": sample.separation_norm,
            "pre_renorm_separation_norm": sample.pre_renorm_separation_norm,
            "post_renorm_separation_norm": sample.post_renorm_separation_norm,
            "target_norm": sample.target_norm,
            "growth_factor": sample.growth_factor,
            "log_growth_increment": sample.log_growth_increment,
            "cumulative_log_growth": sample.cumulative_log_growth,
            "local_lambda_1_per_year": sample.local_lambda_1_per_year,
            "running_lambda_1_per_year": sample.running_lambda_1_per_year,
            "lyapunov_time_years": _csv_float(sample.lyapunov_time_years),
            "max_position_separation_m": sample.max_position_separation_m,
            "max_velocity_separation_m_s": sample.max_velocity_separation_m_s,
            "dominant_body_in_norm": sample.dominant_body_in_norm,
            "dominant_component_type": sample.dominant_component_type,
            "renorm_interval_years_actual": sample.renorm_interval_years_actual,
            "cosine_with_previous_delta_direction": _csv_float(
                sample.cosine_with_previous_delta_direction
            ),
            "cosine_with_initial_delta_direction": _csv_float(
                sample.cosine_with_initial_delta_direction
            ),
            "direction_reset_suspected": int(sample.direction_reset_suspected),
        }
    )


def fit_lyapunov_growth(
    samples: list[LyapunovSample],
    *,
    fit_start_years: float,
    fit_end_years: float,
) -> dict[str, float | int | None]:
    selected = [
        sample
        for sample in samples
        if fit_start_years <= sample.time_years <= fit_end_years
    ]
    if len(selected) < 2:
        return {
            "fit_start_years": fit_start_years,
            "fit_end_years": fit_end_years,
            "n_points": len(selected),
            "lambda_1_per_year": None,
            "lyapunov_time_years": None,
            "lyapunov_time_myr": None,
            "intercept": None,
            "r_squared": None,
        }

    times = np.array([sample.time_years for sample in selected], dtype=float)
    growth = np.array([sample.cumulative_log_growth for sample in selected], dtype=float)
    slope, intercept = np.polyfit(times, growth, deg=1)
    fitted = slope * times + intercept
    residual = growth - fitted
    ss_res = float(np.sum(residual * residual))
    centered = growth - float(np.mean(growth))
    ss_tot = float(np.sum(centered * centered))
    r_squared = math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    time_years = lyapunov_time_years(float(slope))

    return {
        "fit_start_years": fit_start_years,
        "fit_end_years": fit_end_years,
        "n_points": len(selected),
        "lambda_1_per_year": float(slope),
        "lyapunov_time_years": time_years,
        "lyapunov_time_myr": time_years / 1.0e6 if math.isfinite(time_years) else None,
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def _linear_fit_xy(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float | int | None]:
    if x_values.size < 2 or y_values.size < 2:
        return {
            "n_points": int(x_values.size),
            "slope": None,
            "intercept": None,
            "r_squared": None,
        }
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    fitted = slope * x_values + intercept
    residual = y_values - fitted
    ss_res = float(np.sum(residual * residual))
    centered = y_values - float(np.mean(y_values))
    ss_tot = float(np.sum(centered * centered))
    r_squared = math.nan if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "n_points": int(x_values.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
    }


def fit_no_renorm_separation(
    samples: list[LyapunovSample],
    *,
    fit_start_years: float,
    fit_end_years: float,
) -> dict[str, dict[str, float | int | None]]:
    """Fit raw separation growth for --lyapunov-no-renorm debug runs."""
    selected = [
        sample
        for sample in samples
        if (
            fit_start_years <= sample.time_years <= fit_end_years
            and sample.time_years > 0.0
            and sample.separation_norm > 0.0
            and math.isfinite(sample.separation_norm)
        )
    ]
    times = np.array([sample.time_years for sample in selected], dtype=float)
    log_separations = np.array(
        [math.log(sample.separation_norm) for sample in selected],
        dtype=float,
    )
    log_times = np.log(times) if times.size else np.array([], dtype=float)
    return {
        "log_separation_vs_time": _linear_fit_xy(times, log_separations),
        "log_separation_vs_log_time": _linear_fit_xy(log_times, log_separations),
    }


def write_no_renorm_separation_csv(path: Path, samples: list[LyapunovSample]) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=NO_RENORM_SEPARATION_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "time_years": sample.time_years,
                    "separation_norm": sample.separation_norm,
                    "log_separation_norm": (
                        math.log(sample.separation_norm)
                        if sample.separation_norm > 0.0
                        else ""
                    ),
                    "position_separation_m": sample.max_position_separation_m,
                    "velocity_separation_m_s": sample.max_velocity_separation_m_s,
                }
            )


def write_two_body_validation_csv(
    path: Path,
    result: TwoBodyValidationResult,
    *,
    runtime_seconds: float,
) -> None:
    row = result.as_row()
    row["runtime_seconds"] = runtime_seconds
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=TWO_BODY_VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def lyapunov_samples_to_json(samples: list[LyapunovSample]) -> str:
    return json.dumps([sample.__dict__ for sample in samples], separators=(",", ":"))


def lyapunov_samples_from_json(text: str) -> list[LyapunovSample]:
    if not text:
        return []
    values = json.loads(text)
    if not isinstance(values, list):
        return []
    return [LyapunovSample(**value) for value in values if isinstance(value, dict)]


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.glob("checkpoint_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint_*.npz files found in {path}.")
        return candidates[-1]
    return path


def load_checkpoint(path: Path) -> CheckpointData:
    checkpoint_path = resolve_checkpoint_path(path)
    with np.load(checkpoint_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        state = NBodyState(
            positions=np.array(data["positions_m"], dtype=float),
            velocities=np.array(data["velocities_m_s"], dtype=float),
            masses=np.array(data["masses_kg"], dtype=float),
        )
        lyapunov_delta_positions = (
            np.array(data["lyapunov_delta_positions_m"], dtype=float)
            if bool(metadata.get("has_lyapunov_delta_positions"))
            else None
        )
        lyapunov_delta_velocities = (
            np.array(data["lyapunov_delta_velocities_m_s"], dtype=float)
            if bool(metadata.get("has_lyapunov_delta_velocities"))
            else None
        )
        lyapunov_initial_delta_vector = (
            np.array(data["lyapunov_initial_delta_vector"], dtype=float)
            if bool(metadata.get("has_lyapunov_initial_delta_vector"))
            else None
        )
        lyapunov_previous_delta_vector = (
            np.array(data["lyapunov_previous_delta_vector"], dtype=float)
            if bool(metadata.get("has_lyapunov_previous_delta_vector"))
            else None
        )
        return CheckpointData(
            path=checkpoint_path,
            time_s=float(metadata["time_s"]),
            n_records=int(metadata.get("n_records", 0)),
            state=state,
            reference_energy_j=float(metadata["reference_energy_j"]),
            reference_angular_momentum=np.array(data["reference_angular_momentum"], dtype=float),
            reference_angular_momentum_norm=float(metadata["reference_angular_momentum_norm"]),
            reference_com_position_m=np.array(data["reference_com_position_m"], dtype=float),
            reference_com_velocity_m_s=np.array(data["reference_com_velocity_m_s"], dtype=float),
            min_distance_m=np.array(data["min_distance_m"], dtype=float),
            min_time_s=np.array(data["min_time_s"], dtype=float),
            extrema=dict(metadata.get("extrema", {})),
            lyapunov_cumulative_log_growth=float(metadata.get("lyapunov_cumulative_log_growth", 0.0)),
            lyapunov_last_renorm_s=float(metadata.get("lyapunov_last_renorm_s", 0.0)),
            lyapunov_next_renorm_s=float(metadata.get("lyapunov_next_renorm_s", math.inf)),
            lyapunov_previous_sample_norm=float(metadata.get("lyapunov_previous_sample_norm", math.nan)),
            lyapunov_delta_positions=lyapunov_delta_positions,
            lyapunov_delta_velocities=lyapunov_delta_velocities,
            lyapunov_initial_delta_vector=lyapunov_initial_delta_vector,
            lyapunov_previous_delta_vector=lyapunov_previous_delta_vector,
            lyapunov_samples=lyapunov_samples_from_json(metadata.get("lyapunov_samples", "[]")),
            rng_state=metadata.get("rng_state"),
            config_hash=str(metadata.get("config_hash", "")),
        )


def write_checkpoint_atomic(
    path: Path,
    *,
    time_s: float,
    n_records: int,
    state: NBodyState,
    reference: InvariantReference,
    min_tracker: PairwiseMinimumTracker,
    extrema: dict[str, float],
    config_hash: str,
    lyapunov_cumulative_log_growth: float = 0.0,
    lyapunov_last_renorm_s: float = 0.0,
    lyapunov_next_renorm_s: float = math.inf,
    lyapunov_previous_sample_norm: float = math.nan,
    lyapunov_delta_positions: np.ndarray | None = None,
    lyapunov_delta_velocities: np.ndarray | None = None,
    lyapunov_initial_delta_vector: np.ndarray | None = None,
    lyapunov_previous_delta_vector: np.ndarray | None = None,
    lyapunov_samples: list[LyapunovSample] | None = None,
    rng_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "time_s": float(time_s),
        "time_years": seconds_to_years(time_s),
        "n_records": int(n_records),
        "reference_energy_j": reference.energy_j,
        "reference_angular_momentum_norm": reference.angular_momentum_norm,
        "extrema": extrema,
        "config_hash": config_hash,
        "lyapunov_cumulative_log_growth": lyapunov_cumulative_log_growth,
        "lyapunov_last_renorm_s": lyapunov_last_renorm_s,
        "lyapunov_next_renorm_s": lyapunov_next_renorm_s,
        "lyapunov_previous_sample_norm": lyapunov_previous_sample_norm,
        "lyapunov_samples": lyapunov_samples_to_json(lyapunov_samples or []),
        "rng_state": rng_state,
        "has_lyapunov_delta_positions": lyapunov_delta_positions is not None,
        "has_lyapunov_delta_velocities": lyapunov_delta_velocities is not None,
        "has_lyapunov_initial_delta_vector": lyapunov_initial_delta_vector is not None,
        "has_lyapunov_previous_delta_vector": lyapunov_previous_delta_vector is not None,
    }
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        metadata=np.array(json.dumps(metadata)),
        positions_m=state.positions,
        velocities_m_s=state.velocities,
        masses_kg=state.masses,
        reference_angular_momentum=reference.angular_momentum,
        reference_com_position_m=reference.com_position_m,
        reference_com_velocity_m_s=reference.com_velocity_m_s,
        min_distance_m=min_tracker.min_distance_m,
        min_time_s=min_tracker.min_time_s,
        lyapunov_delta_positions_m=(
            lyapunov_delta_positions if lyapunov_delta_positions is not None else np.empty((0, 3))
        ),
        lyapunov_delta_velocities_m_s=(
            lyapunov_delta_velocities if lyapunov_delta_velocities is not None else np.empty((0, 3))
        ),
        lyapunov_initial_delta_vector=(
            lyapunov_initial_delta_vector if lyapunov_initial_delta_vector is not None else np.empty(0)
        ),
        lyapunov_previous_delta_vector=(
            lyapunov_previous_delta_vector if lyapunov_previous_delta_vector is not None else np.empty(0)
        ),
    )
    with tmp_path.open("rb") as file_obj:
        os.fsync(file_obj.fileno())
    os.replace(tmp_path, path)


def prune_checkpoints(checkpoint_dir: Path, *, keep: int) -> None:
    if keep <= 0:
        return
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.npz"))
    for old_path in candidates[:-keep]:
        old_path.unlink(missing_ok=True)


def checkpoint_filename(tag: str, time_s: float) -> str:
    return f"checkpoint_{tag}_{seconds_to_years(time_s):012.6f}yr.npz"


def heliocentric_ecliptic_state_vectors(
    state: NBodyState,
    *,
    body_index: int,
    sun_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return heliocentric J2000 ecliptic position and velocity vectors."""
    position = state.positions[body_index] - state.positions[sun_index]
    velocity = state.velocities[body_index] - state.velocities[sun_index]
    return (
        equatorial_to_ecliptic_j2000(position),
        equatorial_to_ecliptic_j2000(velocity),
    )


def detect_poincare_crossing(
    previous_state: NBodyState,
    current_state: NBodyState,
    *,
    previous_time_s: float,
    current_time_s: float,
    config: PoincareConfig,
    sun_index: int,
) -> PoincareSample | None:
    prev_pos, prev_vel = heliocentric_ecliptic_state_vectors(
        previous_state,
        body_index=config.body_index,
        sun_index=sun_index,
    )
    curr_pos, curr_vel = heliocentric_ecliptic_state_vectors(
        current_state,
        body_index=config.body_index,
        sun_index=sun_index,
    )
    axis = 2 if config.plane == "z" else 1
    prev_value = float(prev_pos[axis])
    curr_value = float(curr_pos[axis])
    delta_value = curr_value - prev_value
    if delta_value == 0.0 or not math.isfinite(delta_value):
        return None

    direction = "positive" if delta_value > 0.0 else "negative"
    if config.direction != "both" and direction != config.direction:
        return None
    if direction == "positive":
        crossed = prev_value < 0.0 <= curr_value
    else:
        crossed = prev_value > 0.0 >= curr_value
    if not crossed:
        return None

    fraction = -prev_value / delta_value
    if fraction < 0.0 or fraction > 1.0 or not math.isfinite(fraction):
        return None

    position = prev_pos + fraction * (curr_pos - prev_pos)
    velocity = prev_vel + fraction * (curr_vel - prev_vel)
    time_s = previous_time_s + fraction * (current_time_s - previous_time_s)
    r_au = float(np.linalg.norm(position)) / AU_M
    speed_au_year = float(np.linalg.norm(velocity)) * JULIAN_YEAR_S / AU_M
    return PoincareSample(
        time_years=seconds_to_years(time_s),
        body=config.body_name,
        plane=config.plane,
        direction=direction,
        x_au=float(position[0] / AU_M),
        y_au=float(position[1] / AU_M),
        z_au=float(position[2] / AU_M),
        vx_au_per_year=float(velocity[0] * JULIAN_YEAR_S / AU_M),
        vy_au_per_year=float(velocity[1] * JULIAN_YEAR_S / AU_M),
        vz_au_per_year=float(velocity[2] * JULIAN_YEAR_S / AU_M),
        r_au=r_au,
        speed_au_per_year=speed_au_year,
    )


def write_poincare_sample(
    outputs: PoincareOutputs,
    sample: PoincareSample,
) -> None:
    outputs.writer.writerow(
        {
            "time_years": sample.time_years,
            "body": sample.body,
            "plane": sample.plane,
            "direction": sample.direction,
            "x_au": sample.x_au,
            "y_au": sample.y_au,
            "z_au": sample.z_au,
            "vx_au_per_year": sample.vx_au_per_year,
            "vy_au_per_year": sample.vy_au_per_year,
            "vz_au_per_year": sample.vz_au_per_year,
            "r_au": sample.r_au,
            "speed_au_per_year": sample.speed_au_per_year,
        }
    )


def classify_megno_result(
    *,
    final_megno: float,
    estimated_lyapunov_per_year: float,
    duration_years: float,
    model_scope: str,
    fallback_megno_slope_per_year: float | None = None,
) -> str:
    if not math.isfinite(final_megno):
        return "ambiguous"
    if final_megno > 10.0:
        if (
            math.isfinite(estimated_lyapunov_per_year)
            and estimated_lyapunov_per_year > 0.0
            and estimated_lyapunov_per_year * duration_years > 1.0
        ):
            return "chaotic_candidate"
        if (
            fallback_megno_slope_per_year is not None
            and math.isfinite(fallback_megno_slope_per_year)
            and fallback_megno_slope_per_year > 0.0
            and fallback_megno_slope_per_year * duration_years > 1.0
        ):
            return "chaotic_candidate"
        return "ambiguous"
    if abs(final_megno - 2.0) <= 3.0:
        return "regular_likely"
    if model_scope in TWO_BODY_MODEL_SCOPES and final_megno < 8.0:
        return "regular_likely"
    return "ambiguous"


def _linear_fit_xy(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    if len(xs) < 2:
        return math.nan, math.nan, math.nan
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    ss_xx = float(np.sum((x_arr - x_mean) ** 2))
    if ss_xx <= 0.0:
        return math.nan, math.nan, math.nan
    slope = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / ss_xx)
    intercept = y_mean - slope * x_mean
    predicted = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - predicted) ** 2))
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    return slope, intercept, r_squared


def last_finite_lcn_sample(
    samples: list[ReboundMegnoSample],
) -> tuple[float | None, float | None]:
    for sample in reversed(samples):
        value = sample.finite_time_lyapunov_estimate
        if math.isfinite(value):
            return value, sample.time_years
    return None, None


def megno_slope_window_estimates(
    samples: list[ReboundMegnoSample],
    *,
    duration_years: float,
) -> list[dict[str, float | int | str | None]]:
    windows = [0.0, 100_000_000.0, 200_000_000.0, 300_000_000.0]
    finite = [
        sample
        for sample in samples
        if math.isfinite(sample.time_years) and math.isfinite(sample.megno)
    ]
    results: list[dict[str, float | int | str | None]] = []
    for start_years in windows:
        if duration_years <= start_years:
            continue
        selected = [
            sample
            for sample in finite
            if start_years <= sample.time_years <= duration_years
        ]
        if len(selected) < 3:
            results.append(
                {
                    "window_start_years": start_years,
                    "window_end_years": duration_years,
                    "n_samples": len(selected),
                    "megno_slope_per_year": None,
                    "megno_intercept": None,
                    "r_squared": None,
                    "delta_megno": None,
                    "lyapunov_proxy_1_per_year": None,
                    "warning": "insufficient finite MEGNO samples",
                }
            )
            continue
        xs = [sample.time_years for sample in selected]
        ys = [sample.megno for sample in selected]
        slope, intercept, r_squared = _linear_fit_xy(xs, ys)
        results.append(
            {
                "window_start_years": start_years,
                "window_end_years": duration_years,
                "n_samples": len(selected),
                "megno_slope_per_year": _finite_or_none(slope),
                "megno_intercept": _finite_or_none(intercept),
                "r_squared": _finite_or_none(r_squared),
                "delta_megno": _finite_or_none(ys[-1] - ys[0]),
                "lyapunov_proxy_1_per_year": (
                    _finite_or_none(max(0.0, slope) * 0.5)
                    if math.isfinite(slope)
                    else None
                ),
                "warning": (
                    "MEGNO slope fallback; not a direct Simulation.lyapunov() accessor value"
                    if math.isfinite(slope)
                    else "fit failed"
                ),
            }
        )
    return results


def best_megno_slope_fallback(
    estimates: list[dict[str, float | int | str | None]],
) -> float | None:
    finite: list[float] = []
    for row in estimates:
        value = row.get("lyapunov_proxy_1_per_year")
        if value is None:
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            finite.append(value_f)
    if not finite:
        return None
    return max(finite)


def write_rebound_megno_sample(
    outputs: ReboundMegnoOutputs,
    sample: ReboundMegnoSample,
    *,
    backend: str,
    integrator: str,
    gr_model: str,
) -> None:
    outputs.writer.writerow(
        {
            "time_years": sample.time_years,
            "megno": _csv_float(sample.megno),
            "mean_megno": _csv_float(sample.mean_megno),
            "finite_time_lyapunov_estimate": _csv_float(sample.finite_time_lyapunov_estimate),
            "backend": backend,
            "integrator": integrator,
            "gr_model": gr_model,
            "warnings": sample.warnings,
        }
    )


def plot_megno_growth(samples: list[ReboundMegnoSample], path: Path) -> None:
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = [sample.time_years for sample in samples]
    values = [sample.megno for sample in samples]
    lyapunov_values = [sample.finite_time_lyapunov_estimate for sample in samples]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, values, label="MEGNO")
    ax.axhline(2.0, color="0.4", linestyle="--", linewidth=1.0, label="regular reference Y=2")
    ax.set_xlabel("time [years]")
    ax.set_ylabel("MEGNO")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if any(math.isfinite(value) for value in lyapunov_values):
        ax2 = ax.twinx()
        ax2.plot(times, lyapunov_values, color="tab:red", alpha=0.5, label="LCN [1/year]")
        ax2.set_ylabel("finite-time LCN [1/year]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rebound_megno_summary_dict(
    *,
    args: argparse.Namespace,
    tag: str,
    result: ReboundMegnoResult,
    outputs: ReboundMegnoOutputs,
    runtime_s: float,
    resume_info: ReboundResumeInfo | None = None,
) -> dict:
    return {
        "mode": MODE_DESCRIPTION,
        "diagnostic": "REBOUND-native finite-time MEGNO",
        "final_megno": _finite_or_none(result.final_megno),
        "final_mean_megno": _finite_or_none(result.final_mean_megno),
        "final_lcn_raw": _finite_or_none(result.final_lcn_raw),
        "last_finite_lcn": _finite_or_none(result.last_finite_lcn),
        "last_finite_lcn_time_years": _finite_or_none(result.last_finite_lcn_time_years),
        "megno_slope_window_estimates": result.megno_slope_window_estimates,
        "estimated_lyapunov_if_available": _finite_or_none(result.estimated_lyapunov_if_available),
        "duration_years": args.duration_years,
        "timestep_days": args.step_days,
        "backend": args.backend,
        "integrator": args.rebound_integrator,
        "gr_model": args.rebound_gr_model,
        "classification_hint": result.classification_hint,
        "lcn_available": result.lcn_available,
        "caveats": result.caveats,
        "rebound_resume": (
            {
                "requested": resume_info.requested,
                "archive_path": resume_info.archive_path,
                "resumed_from_time_years": resume_info.resumed_from_time_years,
                "megno_state_validated": resume_info.megno_state_validated,
                "duplicate_rows_removed": resume_info.duplicate_rows_removed,
                "warnings": resume_info.warnings,
            }
            if resume_info is not None
            else {
                "requested": getattr(args, "rebound_resume", None),
                "resumed_from_time_years": None,
            }
        ),
        "runtime_seconds": runtime_s,
        "outputs": {
            "megno_csv": str(outputs.csv_path),
            "megno_summary": str(outputs.summary_path),
            "megno_growth_plot": str(outputs.plot_path),
        },
    }


def plot_poincare_samples(samples: list[PoincareSample], path: Path) -> None:
    if not samples:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        [sample.x_au for sample in samples],
        [sample.vx_au_per_year for sample in samples],
        s=8,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_xlabel("heliocentric ecliptic x [AU]")
    ax.set_ylabel("heliocentric ecliptic vx [AU/year]")
    ax.set_title("Exploratory Poincare-style section")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return math.nan
    if value > 700.0:
        return math.inf
    if value < -745.0:
        return 0.0
    return math.exp(value)


def build_fli_megno_samples(
    lyapunov_result: LyapunovResult,
    *,
    model_scope: str,
) -> list[FliMegnoSample]:
    samples: list[FliMegnoSample] = []
    megno_integral = 0.0
    previous_time = 0.0
    megno_fit = RunningLinearFit()
    for sample in lyapunov_result.samples:
        time_years = sample.time_years
        if time_years <= 0.0:
            continue
        # Use growth relative to the initial/renormalized target norm. The
        # absolute scaled phase-space norm is arbitrary because the requested
        # perturbation size sets its zero point.
        log_tangent_norm = sample.cumulative_log_growth
        tangent_norm = _safe_exp(log_tangent_norm)
        finite_lambda = sample.cumulative_log_growth / time_years
        local_lambda = sample.local_lambda_1_per_year
        if math.isfinite(local_lambda):
            megno_integral += local_lambda * 0.5 * (
                time_years * time_years - previous_time * previous_time
            )
        megno_lite = 2.0 * megno_integral / time_years
        megno_fit.add(time_years, megno_lite)
        warning = ""
        if abs(sample.log_growth_increment) > 1.0:
            warning = "large local tangent growth increment; check renormalization interval"
        if model_scope in TWO_BODY_MODEL_SCOPES:
            classification_hint = "two_body_expected_near_integrable_likely"
            if finite_lambda > 1.0e-2:
                warning = (
                    "strong finite-time growth in a two-body validation scope; "
                    "indicator is failing validation unless duration scaling trends to zero"
                )
        else:
            classification_hint = "finite_time_indicator_requires_duration_scaling"
        samples.append(
            FliMegnoSample(
                time_years=time_years,
                tangent_norm=tangent_norm,
                log_tangent_norm=log_tangent_norm,
                finite_time_lambda_1_per_year=finite_lambda,
                fli=log_tangent_norm,
                megno_lite=megno_lite,
                running_megno_lite_slope=megno_fit.slope(),
                classification_hint=classification_hint,
                warning=warning,
            )
        )
        previous_time = time_years
    return samples


def write_fli_megno_outputs(
    *,
    csv_path: Path,
    summary_path: Path,
    samples: list[FliMegnoSample],
    lyapunov_result: LyapunovResult,
    args: argparse.Namespace,
    runtime_s: float,
) -> dict:
    with csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FLI_MEGNO_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "time_years": sample.time_years,
                    "tangent_norm": _csv_float(sample.tangent_norm),
                    "log_tangent_norm": _csv_float(sample.log_tangent_norm),
                    "finite_time_lambda_1_per_year": _csv_float(
                        sample.finite_time_lambda_1_per_year
                    ),
                    "fli": _csv_float(sample.fli),
                    "megno_lite": _csv_float(sample.megno_lite),
                    "running_megno_lite_slope": _csv_float(
                        sample.running_megno_lite_slope
                    ),
                    "classification_hint": sample.classification_hint,
                    "warning": sample.warning,
                }
            )

    warnings = [
        (
            "FLI-lite and MEGNO-lite are finite-time tangent-growth indicators. "
            "They are exploratory diagnostics and are not proof of asymptotic chaos without duration scaling."
        )
    ]
    if args.duration_years < 100_000.0:
        warnings.append(
            "Duration is short compared with expected Myr-scale Solar System chaos times; treat this as a workflow smoke test."
        )
    if args.model_scope in TWO_BODY_MODEL_SCOPES:
        warnings.append(
            "This model scope is a validated near-integrable two-body case; strong FLI/MEGNO-lite chaos hints indicate a diagnostic failure unless duration scaling trends to zero."
        )
    if args.gr_model != "none":
        warnings.append(
            "Validated tangent FLI/MEGNO-lite paths should use gr_model=none unless a new GR tangent model is documented."
        )
    warnings.extend(sorted({sample.warning for sample in samples if sample.warning}))

    final = samples[-1] if samples else None
    summary = {
        "mode": MODE_DESCRIPTION,
        "diagnostic": "finite-time FLI-lite / MEGNO-lite tangent-growth indicators",
        "final_fli": _finite_or_none(final.fli if final is not None else None),
        "final_megno_lite": _finite_or_none(final.megno_lite if final is not None else None),
        "final_finite_time_lambda": _finite_or_none(
            final.finite_time_lambda_1_per_year if final is not None else None
        ),
        "duration_years": args.duration_years,
        "timestep_days": args.step_days,
        "renorm_years": lyapunov_result.config.renorm_years,
        "method": "tangent",
        "sample_count": len(samples),
        "caveats": warnings,
        "runtime": {
            "wall_clock_seconds": runtime_s,
            "wall_clock_minutes": runtime_s / 60.0,
        },
        "outputs": {
            "fli_megno_csv": str(csv_path),
            "fli_megno_summary": str(summary_path),
        },
    }
    write_summary(summary_path, summary)
    return summary


def _quadratic_peak_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    if not math.isfinite(offset):
        return 0.0
    return max(-0.5, min(0.5, offset))


def dominant_frequency_fft_lite(
    times_years: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, str]:
    """Return dominant angular frequency, amplitude, and a warning string."""
    n_samples = int(values.size)
    if n_samples < 4:
        return math.nan, math.nan, "fewer than four samples"
    dt_values = np.diff(times_years)
    median_dt = float(np.median(dt_values))
    if median_dt <= 0.0 or not math.isfinite(median_dt):
        return math.nan, math.nan, "invalid sample cadence"
    cadence_warning = ""
    if np.max(np.abs(dt_values - median_dt)) > 1.0e-6 * max(1.0, abs(median_dt)):
        cadence_warning = "nonuniform sample cadence; FFT-lite assumes near-uniform sampling"

    centered = values - np.mean(values)
    window = np.hanning(n_samples)
    if float(np.sum(window)) == 0.0:
        window = np.ones(n_samples)
    spectrum = np.fft.fft(centered * window)
    freqs = np.fft.fftfreq(n_samples, d=median_dt) * 2.0 * math.pi
    magnitudes = np.abs(spectrum)
    magnitudes[0] = 0.0
    if not np.any(np.isfinite(magnitudes)) or float(np.max(magnitudes)) == 0.0:
        return math.nan, 0.0, "no nonzero FFT-lite peak"
    peak_index = int(np.nanargmax(magnitudes))
    frequency = float(freqs[peak_index])
    amplitude = float(magnitudes[peak_index] / max(1.0, np.sum(window)))

    if 0 < peak_index < n_samples - 1:
        # Avoid the positive/negative Nyquist seam where adjacent bins are not
        # contiguous in physical frequency.
        if abs(freqs[peak_index + 1] - freqs[peak_index]) < 1.5 * abs(freqs[1] - freqs[0]):
            offset = _quadratic_peak_offset(
                float(magnitudes[peak_index - 1]),
                float(magnitudes[peak_index]),
                float(magnitudes[peak_index + 1]),
            )
            frequency += offset * float(freqs[1] - freqs[0])
    return frequency, amplitude, cadence_warning


def run_frequency_map_analysis(
    *,
    orbital_elements_path: Path,
    output_csv_path: Path,
    output_dir: Path,
    tag: str,
    body_names: tuple[str, ...],
    args: argparse.Namespace,
) -> dict:
    selected_bodies = parse_frequency_body_choices(args.frequency_bodies, body_names)
    series: dict[str, dict[str, list[float]]] = {
        body: {
            "time": [],
            "e": [],
            "varpi_rad": [],
            "i_rad": [],
            "Omega_rad": [],
        }
        for body in selected_bodies
    }
    with orbital_elements_path.open(newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            body = row.get("body", "")
            if body not in series:
                continue
            try:
                series[body]["time"].append(float(row["time_years"]))
                series[body]["e"].append(float(row["e"]))
                series[body]["varpi_rad"].append(math.radians(float(row["varpi_deg"])))
                series[body]["i_rad"].append(math.radians(float(row["i_deg"])))
                series[body]["Omega_rad"].append(math.radians(float(row["Omega_deg"])))
            except (KeyError, TypeError, ValueError):
                continue

    rows: list[dict[str, float | int | str]] = []
    previous_frequency: dict[tuple[str, str], tuple[float, float]] = {}
    window = float(args.frequency_window_years)
    step = float(args.frequency_step_years)
    for body, values in series.items():
        times = np.array(values["time"], dtype=float)
        if times.size == 0:
            rows.append(
                {
                    "body": body,
                    "variable": "eccentricity_complex",
                    "window_start_years": "",
                    "window_end_years": "",
                    "dominant_frequency_rad_per_year": "",
                    "dominant_period_years": "",
                    "amplitude": "",
                    "estimated_frequency_drift_rad_per_year": "",
                    "n_samples": 0,
                    "warning": "no orbital-element samples for selected body",
                }
            )
            continue
        start = float(np.min(times))
        max_time = float(np.max(times))
        while start + window <= max_time + 1.0e-9:
            end = start + window
            mask = (times >= start - 1.0e-9) & (times <= end + 1.0e-9)
            n_samples = int(np.count_nonzero(mask))
            midpoint = 0.5 * (start + end)
            for variable in ("eccentricity_complex", "inclination_complex"):
                warning = ""
                if n_samples < args.frequency_min_samples:
                    frequency = math.nan
                    amplitude = math.nan
                    warning = (
                        f"window has {n_samples} samples; requires at least "
                        f"{args.frequency_min_samples}"
                    )
                else:
                    if variable == "eccentricity_complex":
                        complex_values = np.array(values["e"], dtype=float)[mask] * np.exp(
                            1j * np.array(values["varpi_rad"], dtype=float)[mask]
                        )
                    else:
                        complex_values = np.sin(
                            0.5 * np.array(values["i_rad"], dtype=float)[mask]
                        ) * np.exp(1j * np.array(values["Omega_rad"], dtype=float)[mask])
                    frequency, amplitude, warning = dominant_frequency_fft_lite(
                        times[mask],
                        complex_values,
                    )
                previous = previous_frequency.get((body, variable))
                if previous is None or not math.isfinite(frequency):
                    drift = math.nan
                else:
                    drift = frequency - previous[1]
                if math.isfinite(frequency):
                    previous_frequency[(body, variable)] = (midpoint, frequency)
                rows.append(
                    {
                        "body": body,
                        "variable": variable,
                        "window_start_years": start,
                        "window_end_years": end,
                        "dominant_frequency_rad_per_year": _csv_float(frequency),
                        "dominant_period_years": (
                            _csv_float(2.0 * math.pi / abs(frequency))
                            if math.isfinite(frequency) and frequency != 0.0
                            else ""
                        ),
                        "amplitude": _csv_float(amplitude),
                        "estimated_frequency_drift_rad_per_year": _csv_float(drift),
                        "n_samples": n_samples,
                        "warning": warning,
                    }
                )
            start += step

    with output_csv_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=FREQUENCY_MAP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    plot_paths = plot_frequency_drift(rows, output_dir=output_dir, tag=tag)
    warnings = sorted({str(row["warning"]) for row in rows if row.get("warning")})
    return {
        "enabled": True,
        "diagnostic": "NAFF-lite / FFT-lite frequency map",
        "caution": (
            "This is not full Laskar NAFF. Frequency drift across windows is an "
            "exploratory secular-diffusion diagnostic and must be checked against "
            "timestep, sampling cadence, window length, and force model."
        ),
        "bodies": selected_bodies,
        "window_years": args.frequency_window_years,
        "step_years": args.frequency_step_years,
        "min_samples": args.frequency_min_samples,
        "row_count": len(rows),
        "warnings": warnings,
        "outputs": {
            "frequency_map_csv": str(output_csv_path),
            "frequency_drift_plots": [str(path) for path in plot_paths],
        },
    }


def plot_frequency_drift(
    rows: list[dict[str, float | int | str]],
    *,
    output_dir: Path,
    tag: str,
) -> list[Path]:
    finite_rows = [
        row for row in rows
        if row.get("dominant_frequency_rad_per_year") not in ("", None)
    ]
    if not finite_rows:
        return []

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for body in sorted({str(row["body"]) for row in finite_rows}):
        body_rows = [row for row in finite_rows if row["body"] == body]
        if not body_rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for variable in ("eccentricity_complex", "inclination_complex"):
            subset = [row for row in body_rows if row["variable"] == variable]
            if not subset:
                continue
            x_values = [
                0.5 * (float(row["window_start_years"]) + float(row["window_end_years"]))
                for row in subset
            ]
            y_values = [float(row["dominant_frequency_rad_per_year"]) for row in subset]
            ax.plot(x_values, y_values, marker="o", linewidth=1.0, markersize=3, label=variable)
        ax.set_xlabel("window midpoint [years]")
        ax.set_ylabel("dominant frequency [rad/year]")
        ax.set_title(f"FFT-lite secular frequency drift: {body}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        slug = body.replace(" ", "_")
        path = output_dir / f"frequency_drift_{tag}_{slug}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def lyapunov_warnings(
    *,
    duration_years: float,
    step_days: float,
    renorm_years: float,
    fit: dict[str, float | int | None],
    samples: list[LyapunovSample],
    model_scope: str,
    gr_model: str,
    invariant_extrema: dict[str, float],
    no_renorm: bool,
) -> list[str]:
    warnings = [
        (
            "This is a finite-time Lyapunov estimate, not an asymptotic Lyapunov "
            "exponent. It depends on timestep, force model, perturbation norm, "
            "perturbation target, renormalization interval, fit window, and total "
            "duration. Run duration-scaling and timestep convergence checks before "
            "interpreting the value."
        )
    ]

    step_years = step_days / 365.25
    if duration_years < 100_000.0:
        warnings.append(
            "Integration duration is short relative to the expected few-Myr inner "
            "Solar System Lyapunov time; treat this as a smoke or numerical test."
        )
    warnings.append(
        "A single Lyapunov run cannot detect sensitivity to perturbation size or "
        "renormalization interval; run the validation ladders before interpreting the estimate."
    )
    if duration_years < 10.0 * renorm_years:
        warnings.append(
            "Integration duration spans fewer than ten renormalization intervals."
        )
    if renorm_years < 5.0 * step_years:
        warnings.append(
            "Renormalization interval is very short relative to the timestep."
        )
    if renorm_years > 0.25 * duration_years:
        warnings.append(
            "Renormalization interval is long relative to the integration duration."
        )

    lambda_time = fit.get("lyapunov_time_years")
    if isinstance(lambda_time, (float, int)) and math.isfinite(float(lambda_time)):
        lambda_time_float = float(lambda_time)
        if lambda_time_float <= step_years:
            warnings.append(
                "Fitted Lyapunov time is shorter than the timestep scale; the estimate is not resolved."
            )
        if lambda_time_float > 10.0 * duration_years:
            warnings.append(
                "Fitted Lyapunov time is more than ten times longer than the integration duration."
            )
    else:
        warnings.append(
            "Fit did not produce a positive finite Lyapunov time."
        )

    if int(fit.get("n_points") or 0) < 5:
        warnings.append(
            "Fit window contains fewer than five renormalization samples."
        )
    if samples:
        max_log_increment = max(abs(sample.log_growth_increment) for sample in samples)
        max_growth_factor = max(sample.growth_factor for sample in samples)
        max_position = max(sample.max_position_separation_m for sample in samples)
        max_post_rel_error = max(
            abs(sample.post_renorm_separation_norm - sample.target_norm)
            / sample.target_norm
            for sample in samples
            if sample.target_norm > 0.0 and math.isfinite(sample.post_renorm_separation_norm)
        )
        direction_reset_count = sum(
            1 for sample in samples if sample.direction_reset_suspected
        )
        if max_log_increment > 1.0:
            warnings.append(
                "At least one local log growth increment exceeds 1 per renormalization interval; "
                "the perturbation may be leaving the local tangent regime or the interval may be too long."
            )
        if max_growth_factor > 10.0:
            warnings.append(
                "At least one pre-renormalization growth factor exceeds 10; test smaller renormalization intervals."
            )
        if max_position > 1.0e9:
            warnings.append(
                "Maximum position separation before renormalization exceeds 1e9 m; this is likely outside the local linear regime."
            )
        if no_renorm:
            warnings.append(
                "--lyapunov-no-renorm is enabled; post_renorm_separation_norm records the raw separation, not a rescaled norm."
            )
        elif max_post_rel_error > 1.0e-6:
            warnings.append(
                "Post-renormalization norm differs from target_norm by more than 1e-6 relative."
            )
        if direction_reset_count > 0:
            warnings.append(
                "Lyapunov direction reset diagnostic was triggered; verify that renormalization preserves the evolved tangent direction."
            )

    if isinstance(lambda_time, (float, int)) and math.isfinite(float(lambda_time)):
        if float(lambda_time) < 1000.0 and model_scope in {"full", "inner", *TWO_BODY_MODEL_SCOPES.keys()}:
            warnings.append(
                "Fitted Lyapunov time is below 1000 years for this Solar System reduced model; "
                "this is physically suspicious and should be treated as a numerical/methodological failure until debugged."
            )
        if model_scope in TWO_BODY_MODEL_SCOPES and float(lambda_time) < 1.0e6:
            warnings.append(
                "Two-body Sun+planet validation produced a short positive Lyapunov time; "
                "a near-integrable test should not show strong chaos."
            )

    if gr_model == "sun":
        warnings.append(
            "gr_model='sun' uses a Sun-centered 1PN acceleration applied to planets through the acceleration callback. "
            "In the current approximation it is not symplectic and is not momentum-conserving because the equal-and-opposite Sun reaction is not applied."
        )
        if invariant_extrema.get("max_angular_momentum_rel_drift", 0.0) > 1.0e-8:
            warnings.append(
                "Angular momentum drift is large with gr_model='sun'; compare against gr_model='none'."
            )
        if invariant_extrema.get("max_com_velocity_drift_au_per_year", 0.0) > 1.0e-10:
            warnings.append(
                "Center-of-mass velocity drift is large with gr_model='sun'; compare against gr_model='none'."
            )
    return warnings


def plot_lyapunov_growth(
    samples: list[LyapunovSample],
    fit: dict[str, float | int | None],
    path: Path,
) -> None:
    if len(samples) == 0:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.array([sample.time_years for sample in samples], dtype=float)
    growth = np.array([sample.cumulative_log_growth for sample in samples], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, growth, linewidth=1.0, label="accumulated log growth")

    slope = fit.get("lambda_1_per_year")
    intercept = fit.get("intercept")
    fit_start = fit.get("fit_start_years")
    fit_end = fit.get("fit_end_years")
    if (
        isinstance(slope, (float, int))
        and isinstance(intercept, (float, int))
        and isinstance(fit_start, (float, int))
        and isinstance(fit_end, (float, int))
        and math.isfinite(float(slope))
        and math.isfinite(float(intercept))
    ):
        fit_times = np.linspace(float(fit_start), float(fit_end), 200)
        ax.plot(
            fit_times,
            float(slope) * fit_times + float(intercept),
            linestyle="--",
            linewidth=1.0,
            label="linear fit",
        )
        ax.axvspan(float(fit_start), float(fit_end), alpha=0.12, color="gray")

    ax.set_xlabel("time [Julian years]")
    ax.set_ylabel("accumulated log growth")
    ax.set_title("Finite-time Benettin Lyapunov growth diagnostic")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def lyapunov_summary_dict(
    *,
    args: argparse.Namespace,
    tag: str,
    body_names: tuple[str, ...],
    result: LyapunovResult,
    outputs: LyapunovOutputs,
    runtime_s: float,
) -> dict:
    fit = {
        key: _finite_or_none(value)
        for key, value in result.fit.items()
    }
    final_time = result.final_running_lyapunov_time_years
    if result.samples:
        max_log_increment = max(abs(sample.log_growth_increment) for sample in result.samples)
        max_growth_factor = max(sample.growth_factor for sample in result.samples)
        max_position = max(sample.max_position_separation_m for sample in result.samples)
        max_velocity = max(sample.max_velocity_separation_m_s for sample in result.samples)
        max_post_rel_error = max(
            abs(sample.post_renorm_separation_norm - sample.target_norm)
            / sample.target_norm
            for sample in result.samples
            if sample.target_norm > 0.0 and math.isfinite(sample.post_renorm_separation_norm)
        )
        direction_reset_count = sum(
            1 for sample in result.samples if sample.direction_reset_suspected
        )
        finite_cos_prev = [
            sample.cosine_with_previous_delta_direction
            for sample in result.samples
            if math.isfinite(sample.cosine_with_previous_delta_direction)
        ]
        finite_cos_initial = [
            sample.cosine_with_initial_delta_direction
            for sample in result.samples
            if math.isfinite(sample.cosine_with_initial_delta_direction)
        ]
    else:
        max_log_increment = None
        max_growth_factor = None
        max_position = None
        max_velocity = None
        max_post_rel_error = None
        direction_reset_count = 0
        finite_cos_prev = []
        finite_cos_initial = []
    no_renorm_fits = (
        fit_no_renorm_separation(
            result.samples,
            fit_start_years=result.config.fit_start_years,
            fit_end_years=result.config.fit_end_years,
        )
        if result.config.no_renorm
        else None
    )
    return {
        "mode": MODE_DESCRIPTION,
        "diagnostic": "finite-time Benettin-style maximum Lyapunov estimate",
        "caution": (
            "This is a finite-time numerical estimate dependent on timestep, force model, "
            "perturbation norm, perturbation target, renormalization interval, "
            "fit window, and total duration. It is not an asymptotic Lyapunov "
            "exponent unless duration scaling supports a nonzero plateau."
        ),
        "interpretation_boundary": (
            "For two-body validation cases, a positive finite-time slope should "
            "be classified with duration-scaling tests. Near-integrable Kepler "
            "phase shear can produce a clean positive finite-time slope that "
            "trends toward zero as the integration duration grows."
        ),
        "recommended_timestep_convergence_steps_days": [8.0, 4.0, 2.0],
        "scientific_boundary": {
            "uses_earth_moon_barycenter": "earth barycenter" in body_names,
            "uses_explicit_earth_and_moon": False,
            "uses_empirical_lunar_calibration": False,
            "uses_american_ephemeris_apparent_geocentric_tropical_output": False,
        },
        "configuration": {
            "kernel_path": args.kernel_path,
            "start_date_utc": args.start_date.isoformat(),
            "duration_years": args.duration_years,
            "step_days": args.step_days,
            "gr_model": args.gr_model,
            "integrator": args.integrator,
            "model_scope": args.model_scope,
            "tag": tag,
            "body_names": body_names,
            "lyapunov_body": result.config.body_choice,
            "lyapunov_body_names": result.config.body_names,
            "lyapunov_perturbation_m": result.config.perturbation_m,
            "lyapunov_target_norm": result.config.target_norm,
            "lyapunov_norm": result.config.norm_name,
            "lyapunov_norm_units": "positions in AU, velocities in AU/year",
            "lyapunov_method": result.config.method,
            "lyapunov_no_renorm": result.config.no_renorm,
            "lyapunov_debug": result.config.debug,
            "lyapunov_renorm_years": result.config.renorm_years,
            "lyapunov_fit_start_years": result.config.fit_start_years,
            "lyapunov_fit_end_years": result.config.fit_end_years,
            "lyapunov_seed": result.config.seed,
            "barycenter_compensation": "selected radial displacement compensated by a Sun shift",
            "displacement_m_by_body": result.config.displacement_m_by_body,
            "sun_position_compensation_m": result.config.sun_position_compensation_m,
        },
        "fit": fit,
        "final_running_estimate": {
            "lambda_1_per_year": _finite_or_none(result.final_running_lambda_1_per_year),
            "lyapunov_time_years": _finite_or_none(final_time),
            "lyapunov_time_myr": (
                _finite_or_none(final_time / 1.0e6)
                if final_time is not None and math.isfinite(final_time)
                else None
            ),
        },
        "sample_count": len(result.samples),
        "renormalization_diagnostics": {
            "max_abs_log_growth_increment": _finite_or_none(max_log_increment),
            "max_growth_factor": _finite_or_none(max_growth_factor),
            "max_position_separation_m": _finite_or_none(max_position),
            "max_velocity_separation_m_s": _finite_or_none(max_velocity),
            "max_post_renorm_relative_error": _finite_or_none(max_post_rel_error),
            "direction_reset_suspected_count": direction_reset_count,
            "cosine_with_previous_delta_direction_min": (
                _finite_or_none(min(finite_cos_prev)) if finite_cos_prev else None
            ),
            "cosine_with_previous_delta_direction_max": (
                _finite_or_none(max(finite_cos_prev)) if finite_cos_prev else None
            ),
            "cosine_with_initial_delta_direction_min": (
                _finite_or_none(min(finite_cos_initial)) if finite_cos_initial else None
            ),
            "cosine_with_initial_delta_direction_max": (
                _finite_or_none(max(finite_cos_initial)) if finite_cos_initial else None
            ),
        },
        "no_renormalization_fits": no_renorm_fits,
        "warnings": result.warnings,
        "debug_warnings": result.debug_warnings,
        "runtime": {
            "wall_clock_seconds": runtime_s,
            "wall_clock_minutes": runtime_s / 60.0,
        },
        "outputs": {
            "lyapunov_csv": str(outputs.csv_path),
            "lyapunov_summary": str(outputs.summary_path),
            "lyapunov_growth_plot": str(outputs.plot_path),
            "no_renorm_separation_csv": (
                str(outputs.no_renorm_path) if outputs.no_renorm_path is not None else None
            ),
        },
    }


def update_extrema(extrema: dict[str, float], invariant_row: dict[str, float]) -> None:
    extrema["max_abs_energy_abs_drift_j"] = max(
        extrema["max_abs_energy_abs_drift_j"],
        abs(float(invariant_row["energy_abs_drift_j"])),
    )
    extrema["max_abs_energy_rel_drift"] = max(
        extrema["max_abs_energy_rel_drift"],
        abs(float(invariant_row["energy_rel_drift"])),
    )
    extrema["max_angular_momentum_abs_drift_kg_m2_s"] = max(
        extrema["max_angular_momentum_abs_drift_kg_m2_s"],
        abs(float(invariant_row["angular_momentum_abs_drift_kg_m2_s"])),
    )
    extrema["max_angular_momentum_rel_drift"] = max(
        extrema["max_angular_momentum_rel_drift"],
        abs(float(invariant_row["angular_momentum_rel_drift"])),
    )
    extrema["max_com_position_drift_au"] = max(
        extrema["max_com_position_drift_au"],
        abs(float(invariant_row["com_position_drift_au"])),
    )
    extrema["max_com_velocity_drift_au_per_year"] = max(
        extrema["max_com_velocity_drift_au_per_year"],
        abs(float(invariant_row["com_velocity_drift_au_per_year"])),
    )


def write_snapshot(
    time_s: float,
    state: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    reference,
    extrema: dict[str, float],
    *,
    sun_index: int,
) -> dict[str, float]:
    time_years = seconds_to_years(time_s)
    sun_position = state.positions[sun_index]
    sun_velocity = state.velocities[sun_index]

    for index, name in enumerate(body_names):
        position = state.positions[index]
        velocity = state.velocities[index]
        heliocentric_position = position - sun_position
        heliocentric_velocity = velocity - sun_velocity

        outputs.stability_timeseries.writerow(
            {
                "time_years": time_years,
                "body": name,
                "x_au": position[0] / AU_M,
                "y_au": position[1] / AU_M,
                "z_au": position[2] / AU_M,
                "vx_au_per_year": velocity[0] * JULIAN_YEAR_S / AU_M,
                "vy_au_per_year": velocity[1] * JULIAN_YEAR_S / AU_M,
                "vz_au_per_year": velocity[2] * JULIAN_YEAR_S / AU_M,
                "heliocentric_x_au": heliocentric_position[0] / AU_M,
                "heliocentric_y_au": heliocentric_position[1] / AU_M,
                "heliocentric_z_au": heliocentric_position[2] / AU_M,
                "heliocentric_vx_au_per_year": (
                    heliocentric_velocity[0] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_vy_au_per_year": (
                    heliocentric_velocity[1] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_vz_au_per_year": (
                    heliocentric_velocity[2] * JULIAN_YEAR_S / AU_M
                ),
                "heliocentric_r_au": float(np.linalg.norm(heliocentric_position)) / AU_M,
                "heliocentric_speed_au_per_year": (
                    float(np.linalg.norm(heliocentric_velocity)) * JULIAN_YEAR_S / AU_M
                ),
            }
        )

    for elements in heliocentric_elements_for_state(
        state,
        body_names,
        sun_index=sun_index,
    ):
        outputs.orbital_elements.writerow(elements.as_output_row(time_years))

    invariant_row = invariant_diagnostics_row(time_s, state, reference)
    outputs.invariants.writerow(invariant_row)
    update_extrema(extrema, invariant_row)
    return invariant_row


def initial_extrema() -> dict[str, float]:
    return {
        "max_abs_energy_abs_drift_j": 0.0,
        "max_abs_energy_rel_drift": 0.0,
        "max_angular_momentum_abs_drift_kg_m2_s": 0.0,
        "max_angular_momentum_rel_drift": 0.0,
        "max_com_position_drift_au": 0.0,
        "max_com_velocity_drift_au_per_year": 0.0,
    }


def integrate_rebound_streaming(
    state0: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    *,
    duration_s: float,
    dt_s: float,
    record_interval_s: float,
    show_progress: bool,
    sun_index: int,
    rebound_integrator: str,
    rebound_gr_model: str,
    rebound_ias15_epsilon: float,
    simulationarchive_path: Path | None = None,
    simulationarchive_interval_s: float | None = None,
    megno_outputs: ReboundMegnoOutputs | None = None,
    megno_record_interval_s: float | None = None,
    with_rebound_lyapunov: bool = False,
    megno_seed: int | None = None,
    megno_duration_scaling_mode: bool = False,
    model_scope: str = "full",
    rebound_resume_path: Path | None = None,
    rebound_resume_requested: str | None = None,
    resume_duplicate_rows_removed: dict[str, int] | None = None,
    resume_min_tracker: PairwiseMinimumTracker | None = None,
    resume_extrema: dict[str, float] | None = None,
    resume_n_records: int = 0,
    resume_megno_samples: list[ReboundMegnoSample] | None = None,
) -> IntegrationResult:
    rebound = optional_import_module("rebound")
    if rebound is None:
        raise RuntimeError("REBOUND is not installed. Install rebound or use --backend inhouse.")

    runtime_warnings: list[str] = []
    rebound_resume_info: ReboundResumeInfo | None = None
    resumed_from_time_s = 0.0
    megno_state_validated = False

    if rebound_resume_path is not None:
        sim = load_rebound_archive_snapshot(rebound, rebound_resume_path)
        resumed_from_time_s = float(sim.t)
        if resumed_from_time_s <= 0.0:
            raise RuntimeError(
                f"REBOUND resume archive did not contain a nonzero-time snapshot: {rebound_resume_path}"
            )
        if resumed_from_time_s >= duration_s - 1.0e-6:
            raise RuntimeError(
                "REBOUND resume archive is already at or beyond the requested final duration. "
                "Use the completed outputs or request a longer --duration-years."
            )
        if int(getattr(sim, "N_real", len(body_names)) or len(body_names)) != len(body_names):
            raise RuntimeError(
                "Loaded REBOUND archive has an unexpected real-particle count; refusing resume."
            )
        runtime_warnings.append(
            "REBOUND resume loaded SimulationArchive snapshot at "
            f"t={seconds_to_years(resumed_from_time_s):.12g} years: {rebound_resume_path}"
        )
        if megno_outputs is not None and rebound_gr_model != "none":
            raise RuntimeError(
                "REBOUNDx GR MEGNO resume is not validated in this workflow. "
                "Use Newtonian REBOUND MEGNO for checkpoint-safe LCN runs."
            )
    else:
        sim = build_rebound_simulation(
            rebound,
            state0,
            integrator=rebound_integrator,
            step_s=dt_s,
            ias15_epsilon=rebound_ias15_epsilon,
        )

    if rebound_integrator == "whfast" and rebound_gr_model in {"gr", "gr_full"}:
        runtime_warnings.append(
            "REBOUNDx velocity-dependent GR with WHFast requires operator-style validation; "
            "prefer gr_potential with WHFast until that path is implemented."
        )
    if rebound_gr_model != "none":
        runtime_warnings.append(
            "Invariant energy diagnostics remain Newtonian bookkeeping; GR production runs "
            "should be interpreted with perihelion and conservation validation."
        )
    if rebound_resume_path is None:
        try:
            add_reboundx_gr_force(sim, rebound_gr_model)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
    elif rebound_gr_model != "none":
        runtime_warnings.append(
            "REBOUNDx force state after SimulationArchive resume is not independently "
            "validated here; use Newtonian MEGNO for production Lyapunov diagnostics."
        )

    megno_samples: list[ReboundMegnoSample] = list(resume_megno_samples or [])
    megno_caveats: list[str] = []
    lcn_available = False
    if megno_outputs is not None:
        if rebound_resume_path is None and (
            not hasattr(sim, "init_megno") or not hasattr(sim, "megno")
        ):
            raise RuntimeError(
                "Installed REBOUND does not expose init_megno()/megno(); "
                "cannot run --with-megno."
            )
        if rebound_resume_path is not None and not hasattr(sim, "megno"):
            raise RuntimeError(
                "Loaded REBOUND archive does not expose megno(); cannot safely resume MEGNO."
            )
        if with_rebound_lyapunov and not hasattr(sim, "lyapunov"):
            raise RuntimeError(
                "Installed REBOUND does not expose lyapunov() after MEGNO init; "
                "omit --with-rebound-lyapunov."
            )
        if rebound_resume_path is None:
            try:
                sim.init_megno(seed=megno_seed)
            except Exception as exc:
                raise RuntimeError(
                    "REBOUND MEGNO initialization failed for this backend/integrator combination: "
                    f"{exc}"
                ) from exc
        else:
            try:
                float(sim.megno())
                if with_rebound_lyapunov:
                    float(sim.lyapunov())
                megno_state_validated = True
            except Exception as exc:
                raise RuntimeError(
                    "Loaded REBOUND SimulationArchive did not preserve usable MEGNO/LCN state; "
                    "cannot safely resume this run."
                ) from exc
        lcn_available = bool(with_rebound_lyapunov and hasattr(sim, "lyapunov"))
        megno_caveats.append(
            "REBOUND-native MEGNO is a finite-time variational chaos diagnostic; "
            "duration scaling and timestep/integrator comparison are required before "
            "interpreting any Solar System chaos time."
        )
        megno_caveats.append(
            "REBOUND 4.6.0 exposes megno() and lyapunov(); no separate mean_megno "
            "Python accessor was found, so mean_megno is left blank."
        )
        if rebound_gr_model != "none":
            megno_caveats.append(
                "REBOUNDx GR MEGNO diagnostics are exploratory here; validate against "
                "Newtonian two-body regular cases and IAS15 comparison before interpretation."
            )
        if megno_duration_scaling_mode:
            megno_caveats.append(
                "This run is marked as a duration-scaling sample; individual finite-time "
                "MEGNO/LCN values are not asymptotic Lyapunov exponents."
            )

    if simulationarchive_path is not None and simulationarchive_interval_s is not None:
        archive_method = configure_rebound_simulationarchive(
            sim,
            simulationarchive_path,
            interval_s=simulationarchive_interval_s,
            delete_existing=rebound_resume_path is None,
        )
        runtime_warnings.append(
            f"REBOUND SimulationArchive enabled via {archive_method}: {simulationarchive_path}"
        )

    reference = invariant_reference(state0, G=G_SI)
    if rebound_resume_path is not None:
        final_state = rebound_state_from_sim(sim, state0.masses)
        min_tracker = resume_min_tracker or PairwiseMinimumTracker.create(body_names)
        if resume_min_tracker is None:
            min_tracker.update(resumed_from_time_s, final_state.positions)
        extrema = dict(resume_extrema or initial_extrema())
        n_records = int(resume_n_records)
        if n_records <= 0:
            write_snapshot(
                resumed_from_time_s,
                final_state,
                body_names,
                outputs,
                reference,
                extrema,
                sun_index=sun_index,
            )
            n_records = 1
    else:
        final_state = state0
        min_tracker = PairwiseMinimumTracker.create(body_names)
        min_tracker.update(0.0, state0.positions)
        extrema = initial_extrema()
        write_snapshot(0.0, state0, body_names, outputs, reference, extrema, sun_index=sun_index)
        n_records = 1
    two_body_tracker = (
        TwoBodyValidationTracker.create(state0, body_names, sun_index=sun_index)
        if len(body_names) == 2 and body_names[sun_index] == "sun"
        else None
    )

    def add_scheduled_events(
        schedule: dict[float, set[str]],
        event_name: str,
        interval_s: float,
        start_s: float,
    ) -> None:
        next_t = (math.floor(start_s / interval_s) + 1) * interval_s
        while next_t <= start_s + 1.0e-6:
            next_t += interval_s
        while next_t < duration_s - 1.0e-9:
            schedule.setdefault(round(next_t, 6), set()).add(event_name)
            next_t += interval_s
        schedule.setdefault(round(duration_s, 6), set()).add(event_name)

    event_schedule: dict[float, set[str]] = {}
    add_scheduled_events(event_schedule, "record", record_interval_s, resumed_from_time_s)
    if megno_outputs is not None:
        add_scheduled_events(
            event_schedule,
            "megno",
            megno_record_interval_s if megno_record_interval_s is not None else record_interval_s,
            resumed_from_time_s,
        )
        initial_megno_warning = ""
        try:
            initial_megno = float(sim.megno())
        except Exception as exc:
            initial_megno = math.nan
            initial_megno_warning = f"initial MEGNO unavailable: {exc}"
        initial_lcn = math.nan
        if lcn_available:
            try:
                initial_lcn = float(sim.lyapunov()) * JULIAN_YEAR_S
            except Exception as exc:
                initial_megno_warning = "; ".join(
                    filter(
                        None,
                        (
                            initial_megno_warning,
                            f"initial LCN unavailable: {exc}",
                        ),
                    )
                )
        initial_sample = ReboundMegnoSample(
            time_years=seconds_to_years(resumed_from_time_s),
            megno=initial_megno,
            mean_megno=math.nan,
            finite_time_lyapunov_estimate=initial_lcn,
            warnings=initial_megno_warning,
        )
        if (
            not megno_samples
            or abs(megno_samples[-1].time_years - initial_sample.time_years) > 1.0e-9
        ):
            megno_samples.append(initial_sample)
            write_rebound_megno_sample(
                megno_outputs,
                initial_sample,
                backend="rebound",
                integrator=rebound_integrator,
                gr_model=rebound_gr_model,
            )

    event_times = sorted(event_schedule)

    progress = tqdm(
        total=seconds_to_years(max(0.0, duration_s - resumed_from_time_s)),
        desc=f"REBOUND {rebound_integrator}",
        disable=not show_progress,
    )
    current_t = resumed_from_time_s
    try:
        for target_t in event_times:
            previous_t = current_t
            captured_warning_messages: list[str] = []
            with warning_lib.catch_warnings(record=True) as caught:
                warning_lib.simplefilter("always")
                sim.integrate(target_t, exact_finish_time=1)
                captured_warning_messages = [
                    str(item.message)
                    for item in caught
                    if str(item.message) not in runtime_warnings
                ]
            for message in captured_warning_messages:
                if message not in runtime_warnings:
                    runtime_warnings.append(message)
            current_t = float(sim.t)
            final_state = rebound_state_from_sim(sim, state0.masses)
            min_tracker.update(current_t, final_state.positions)
            events = event_schedule.get(round(target_t, 6), set())
            if "record" in events:
                write_snapshot(
                    current_t,
                    final_state,
                    body_names,
                    outputs,
                    reference,
                    extrema,
                    sun_index=sun_index,
                )
                n_records += 1
                if two_body_tracker is not None:
                    two_body_tracker.update(current_t, final_state, body_names)
            if "megno" in events and megno_outputs is not None:
                sample_warning = ""
                try:
                    megno_value = float(sim.megno())
                except Exception as exc:
                    megno_value = math.nan
                    sample_warning = f"MEGNO unavailable: {exc}"
                lcn_value = math.nan
                if lcn_available:
                    try:
                        lcn_value = float(sim.lyapunov()) * JULIAN_YEAR_S
                    except Exception as exc:
                        sample_warning = "; ".join(
                            filter(None, (sample_warning, f"LCN unavailable: {exc}"))
                        )
                sample = ReboundMegnoSample(
                    time_years=seconds_to_years(current_t),
                    megno=megno_value,
                    mean_megno=math.nan,
                    finite_time_lyapunov_estimate=lcn_value,
                    warnings=sample_warning,
                )
                megno_samples.append(sample)
                write_rebound_megno_sample(
                    megno_outputs,
                    sample,
                    backend="rebound",
                    integrator=rebound_integrator,
                    gr_model=rebound_gr_model,
                )
            if n_records % 50 == 0:
                outputs.flush()
            if megno_outputs is not None and len(megno_samples) % 50 == 0:
                megno_outputs.flush()
            progress.update(seconds_to_years(current_t - previous_t))
    finally:
        progress.close()

    if rebound_integrator == "whfast":
        n_steps = int(math.ceil(duration_s / dt_s))
    else:
        n_steps = len(event_times)

    final_megno = megno_samples[-1].megno if megno_samples else math.nan
    final_mean_megno = megno_samples[-1].mean_megno if megno_samples else math.nan
    final_lcn = (
        megno_samples[-1].finite_time_lyapunov_estimate
        if megno_samples
        else math.nan
    )
    last_finite_lcn, last_finite_lcn_time = last_finite_lcn_sample(megno_samples)
    slope_estimates = megno_slope_window_estimates(
        megno_samples,
        duration_years=seconds_to_years(current_t),
    )
    slope_fallback = best_megno_slope_fallback(slope_estimates)
    if (
        megno_outputs is not None
        and math.isfinite(final_megno)
        and not math.isfinite(final_lcn)
        and slope_fallback is not None
    ):
        megno_caveats.append("LCN accessor unavailable after resume; MEGNO slope fallback used.")
    rebound_megno_classification = classify_megno_result(
        final_megno=final_megno,
        estimated_lyapunov_per_year=final_lcn,
        duration_years=seconds_to_years(current_t),
        model_scope=model_scope,
        fallback_megno_slope_per_year=slope_fallback,
    )
    if megno_outputs is not None and rebound_gr_model != "none":
        rebound_megno_classification = "ambiguous"
        megno_caveats.append(
            "Classification is held at ambiguous for REBOUNDx GR MEGNO runs because "
            "the installed REBOUNDx path warns that variational particles are not "
            "evolved self-consistently by REBOUNDx."
        )
    rebound_megno_result = (
        ReboundMegnoResult(
            samples=megno_samples,
            final_megno=final_megno,
            final_mean_megno=final_mean_megno,
            estimated_lyapunov_if_available=final_lcn,
            final_lcn_raw=final_lcn,
            last_finite_lcn=last_finite_lcn,
            last_finite_lcn_time_years=last_finite_lcn_time,
            megno_slope_window_estimates=slope_estimates,
            classification_hint=rebound_megno_classification,
            caveats=megno_caveats,
            lcn_available=lcn_available,
        )
        if megno_outputs is not None
        else None
    )
    if rebound_resume_path is not None:
        rebound_resume_info = ReboundResumeInfo(
            requested=rebound_resume_requested,
            archive_path=str(rebound_resume_path),
            resumed_from_time_years=seconds_to_years(resumed_from_time_s),
            megno_state_validated=megno_state_validated,
            duplicate_rows_removed=dict(resume_duplicate_rows_removed or {}),
            warnings=[
                message
                for message in runtime_warnings
                if "resume" in message.lower() or "SimulationArchive" in message
            ],
        )

    return IntegrationResult(
        final_state=final_state,
        actual_duration_s=current_t,
        n_steps=n_steps,
        n_records=n_records,
        min_tracker=min_tracker,
        extrema=extrema,
        min_separation_sampling=(
            "recorded REBOUND output samples"
            if megno_outputs is None
            else "recorded REBOUND output and MEGNO event samples"
        ),
        lyapunov_result=None,
        two_body_validation=(
            two_body_tracker.result(
                duration_s=current_t,
                step_days=dt_s / DAY_S,
                n_steps=n_steps,
            )
            if two_body_tracker is not None
            else None
        ),
        runtime_warnings=runtime_warnings,
        rebound_megno_result=rebound_megno_result,
        rebound_resume_info=rebound_resume_info,
    )


def integrate_leapfrog_streaming(
    state0: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    *,
    duration_s: float,
    dt_s: float,
    record_interval_s: float,
    accel_func,
    accel_kwargs: dict,
    show_progress: bool,
    sun_index: int,
    lyapunov_config: LyapunovConfig | None = None,
    lyapunov_outputs: LyapunovOutputs | None = None,
    lyapunov_initial_state: NBodyState | None = None,
    poincare_config: PoincareConfig | None = None,
    poincare_outputs: PoincareOutputs | None = None,
    checkpoint_data: CheckpointData | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_s: float | None = None,
    keep_checkpoints: int = 3,
    checkpoint_tag: str = "stability",
    config_hash: str = "",
) -> IntegrationResult:
    start_time_s = checkpoint_data.time_s if checkpoint_data is not None else 0.0
    total_steps = int(math.ceil(max(0.0, duration_s - start_time_s) / dt_s))
    record_every_steps = max(1, int(round(record_interval_s / dt_s)))

    state = checkpoint_data.state.copy() if checkpoint_data is not None else state0.copy()
    acceleration = accel_func(state, G=G_SI, **accel_kwargs)
    if checkpoint_data is not None:
        reference = InvariantReference(
            energy_j=checkpoint_data.reference_energy_j,
            angular_momentum=checkpoint_data.reference_angular_momentum,
            angular_momentum_norm=checkpoint_data.reference_angular_momentum_norm,
            com_position_m=checkpoint_data.reference_com_position_m,
            com_velocity_m_s=checkpoint_data.reference_com_velocity_m_s,
        )
        min_tracker = PairwiseMinimumTracker.create(body_names)
        min_tracker.min_distance_m = checkpoint_data.min_distance_m.copy()
        min_tracker.min_time_s = checkpoint_data.min_time_s.copy()
        extrema = dict(checkpoint_data.extrema)
        n_records = checkpoint_data.n_records
        time_s = checkpoint_data.time_s
    else:
        reference = invariant_reference(state, G=G_SI)
        min_tracker = PairwiseMinimumTracker.create(body_names)
        min_tracker.update(0.0, state.positions)
        extrema = initial_extrema()
        write_snapshot(0.0, state, body_names, outputs, reference, extrema, sun_index=sun_index)
        n_records = 1
        time_s = 0.0

    two_body_tracker = (
        TwoBodyValidationTracker.create(state, body_names, sun_index=sun_index)
        if len(body_names) == 2 and body_names[sun_index] == "sun"
        else None
    )

    final_invariant_row = None
    lyapunov_samples: list[LyapunovSample] = (
        list(checkpoint_data.lyapunov_samples) if checkpoint_data is not None else []
    )
    lyapunov_cumulative_log_growth = (
        checkpoint_data.lyapunov_cumulative_log_growth if checkpoint_data is not None else 0.0
    )
    lyapunov_last_renorm_s = (
        checkpoint_data.lyapunov_last_renorm_s if checkpoint_data is not None else 0.0
    )
    lyapunov_next_renorm_s = (
        checkpoint_data.lyapunov_next_renorm_s if checkpoint_data is not None else math.inf
    )
    lyapunov_previous_sample_norm = (
        checkpoint_data.lyapunov_previous_sample_norm if checkpoint_data is not None else math.nan
    )
    lyapunov_state: NBodyState | None = None
    lyapunov_acceleration: np.ndarray | None = None
    lyapunov_delta_positions: np.ndarray | None = None
    lyapunov_delta_velocities: np.ndarray | None = None
    lyapunov_tangent_acceleration: np.ndarray | None = None
    lyapunov_runtime_warnings: list[str] = []
    lyapunov_initial_delta_vector: np.ndarray | None = (
        checkpoint_data.lyapunov_initial_delta_vector.copy()
        if checkpoint_data is not None and checkpoint_data.lyapunov_initial_delta_vector is not None
        else None
    )
    lyapunov_previous_delta_vector: np.ndarray | None = (
        checkpoint_data.lyapunov_previous_delta_vector.copy()
        if checkpoint_data is not None and checkpoint_data.lyapunov_previous_delta_vector is not None
        else None
    )
    poincare_samples: list[PoincareSample] = []

    if lyapunov_config is not None:
        if checkpoint_data is not None and checkpoint_data.lyapunov_delta_positions is not None:
            lyapunov_delta_positions = checkpoint_data.lyapunov_delta_positions.copy()
            lyapunov_delta_velocities = (
                checkpoint_data.lyapunov_delta_velocities.copy()
                if checkpoint_data.lyapunov_delta_velocities is not None
                else np.zeros_like(lyapunov_delta_positions)
            )
            lyapunov_tangent_acceleration = tangent_acceleration_newtonian(
                state,
                lyapunov_delta_positions,
                G=G_SI,
            )
            lyapunov_state = state_with_delta(
                state,
                lyapunov_delta_positions,
                lyapunov_delta_velocities,
            )
        elif lyapunov_initial_state is None:
            raise ValueError("Lyapunov/tangent diagnostics require an initial perturbed state.")
        elif lyapunov_config.method == "tangent":
            lyapunov_delta_positions = lyapunov_initial_state.positions - state.positions
            lyapunov_delta_velocities = lyapunov_initial_state.velocities - state.velocities
            lyapunov_tangent_acceleration = tangent_acceleration_newtonian(
                state,
                lyapunov_delta_positions,
                G=G_SI,
            )
            lyapunov_state = state_with_delta(
                state,
                lyapunov_delta_positions,
                lyapunov_delta_velocities,
            )
        else:
            lyapunov_state = lyapunov_initial_state.copy()
            lyapunov_acceleration = accel_func(lyapunov_state, G=G_SI, **accel_kwargs)
        if checkpoint_data is None:
            lyapunov_next_renorm_s = lyapunov_config.renorm_years * JULIAN_YEAR_S
            lyapunov_previous_sample_norm = lyapunov_config.target_norm
        if lyapunov_config.method == "tangent" and lyapunov_initial_delta_vector is None:
            lyapunov_initial_delta_vector = scaled_phase_space_delta_vector_from_arrays(
                lyapunov_delta_positions,
                lyapunov_delta_velocities,
            )
        elif lyapunov_initial_delta_vector is None:
            lyapunov_initial_delta_vector = scaled_phase_space_delta_vector(
                state,
                lyapunov_state,
            )
        if lyapunov_previous_delta_vector is None and lyapunov_initial_delta_vector is not None:
            lyapunov_previous_delta_vector = lyapunov_initial_delta_vector.copy()

    next_checkpoint_s = math.inf
    if checkpoint_every_s is not None and checkpoint_dir is not None:
        next_checkpoint_s = (
            checkpoint_every_s
            if time_s <= 0.0
            else (math.floor(time_s / checkpoint_every_s) + 1.0) * checkpoint_every_s
        )

    progress = tqdm(
        total=total_steps,
        desc="leapfrog steps",
        disable=not show_progress,
    )

    try:
        for step_index in range(1, total_steps + 1):
            previous_state_for_poincare = state.copy() if poincare_config is not None else None
            previous_time_s = time_s
            step_dt = min(dt_s, duration_s - time_s)
            tangent_delta_vel_half = None
            tangent_delta_pos_new = None
            if (
                lyapunov_config is not None
                and lyapunov_config.method == "tangent"
                and lyapunov_delta_positions is not None
                and lyapunov_delta_velocities is not None
                and lyapunov_tangent_acceleration is not None
            ):
                tangent_delta_vel_half = (
                    lyapunov_delta_velocities
                    + 0.5 * step_dt * lyapunov_tangent_acceleration
                )
                tangent_delta_pos_new = (
                    lyapunov_delta_positions
                    + step_dt * tangent_delta_vel_half
                )
            state, acceleration = velocity_verlet_step_generic(
                state,
                acceleration,
                step_dt,
                accel_func,
                accel_kwargs=accel_kwargs,
                G=G_SI,
            )
            if (
                lyapunov_config is not None
                and lyapunov_config.method == "tangent"
                and tangent_delta_pos_new is not None
                and tangent_delta_vel_half is not None
            ):
                lyapunov_tangent_acceleration = tangent_acceleration_newtonian(
                    state,
                    tangent_delta_pos_new,
                    G=G_SI,
                )
                lyapunov_delta_velocities = (
                    tangent_delta_vel_half
                    + 0.5 * step_dt * lyapunov_tangent_acceleration
                )
                lyapunov_delta_positions = tangent_delta_pos_new
                lyapunov_state = state_with_delta(
                    state,
                    lyapunov_delta_positions,
                    lyapunov_delta_velocities,
                )
            elif lyapunov_state is not None and lyapunov_acceleration is not None:
                lyapunov_state, lyapunov_acceleration = velocity_verlet_step_generic(
                    lyapunov_state,
                    lyapunov_acceleration,
                    step_dt,
                    accel_func,
                    accel_kwargs=accel_kwargs,
                    G=G_SI,
                )
            time_s += step_dt
            min_tracker.update(time_s, state.positions)
            if (
                poincare_config is not None
                and poincare_outputs is not None
                and previous_state_for_poincare is not None
            ):
                poincare_sample = detect_poincare_crossing(
                    previous_state_for_poincare,
                    state,
                    previous_time_s=previous_time_s,
                    current_time_s=time_s,
                    config=poincare_config,
                    sun_index=sun_index,
                )
                if poincare_sample is not None:
                    poincare_samples.append(poincare_sample)
                    write_poincare_sample(poincare_outputs, poincare_sample)
                    if len(poincare_samples) % 100 == 0:
                        poincare_outputs.flush()
            if two_body_tracker is not None:
                two_body_tracker.update(
                    time_s,
                    state,
                    body_names,
                    update_orbital_elements=(
                        lyapunov_config is None
                        or step_index % record_every_steps == 0
                        or step_index == total_steps
                    ),
                )

            if (
                lyapunov_config is not None
                and lyapunov_state is not None
                and (
                    time_s + 1.0e-9 >= lyapunov_next_renorm_s
                    or step_index == total_steps
                )
            ):
                component_diag = scaled_phase_space_component_diagnostics(
                    state,
                    lyapunov_state,
                    body_names=body_names,
                )
                if (
                    lyapunov_config.method == "tangent"
                    and lyapunov_delta_positions is not None
                    and lyapunov_delta_velocities is not None
                ):
                    current_delta_vector = scaled_phase_space_delta_vector_from_arrays(
                        lyapunov_delta_positions,
                        lyapunov_delta_velocities,
                    )
                    separation_norm = float(np.linalg.norm(current_delta_vector))
                else:
                    current_delta_vector = scaled_phase_space_delta_vector(
                        state,
                        lyapunov_state,
                    )
                    separation_norm = component_diag.separation_norm
                if separation_norm <= 0.0 or not math.isfinite(separation_norm):
                    lyapunov_runtime_warnings.append(
                        "Degenerate Lyapunov separation encountered; perturbation may be below floating-point resolution for this state and timestep."
                    )
                    lyapunov_state = None
                    lyapunov_acceleration = None
                    lyapunov_delta_positions = None
                    lyapunov_delta_velocities = None
                    lyapunov_tangent_acceleration = None
                    continue

                interval_years = seconds_to_years(time_s - lyapunov_last_renorm_s)
                cosine_with_previous = (
                    cosine_between_scaled_deltas(
                        current_delta_vector,
                        lyapunov_previous_delta_vector,
                    )
                    if lyapunov_previous_delta_vector is not None
                    else math.nan
                )
                cosine_with_initial = (
                    cosine_between_scaled_deltas(
                        current_delta_vector,
                        lyapunov_initial_delta_vector,
                    )
                    if lyapunov_initial_delta_vector is not None
                    else math.nan
                )
                direction_reset_suspected = (
                    not lyapunov_config.no_renorm
                    and len(lyapunov_samples) > 0
                    and math.isfinite(cosine_with_initial)
                    and cosine_with_initial > 0.999999
                    and (
                        not math.isfinite(cosine_with_previous)
                        or cosine_with_previous < 0.999
                    )
                )
                if lyapunov_config.no_renorm:
                    growth_factor = separation_norm / lyapunov_previous_sample_norm
                    log_growth = math.log(growth_factor)
                    lyapunov_cumulative_log_growth = math.log(
                        separation_norm / lyapunov_config.target_norm
                    )
                    post_renorm_norm = separation_norm
                else:
                    growth_factor = separation_norm / lyapunov_config.target_norm
                    log_growth = math.log(growth_factor)
                    lyapunov_cumulative_log_growth += log_growth
                    if lyapunov_config.method == "tangent":
                        renormalized_scaled_delta = (
                            current_delta_vector
                            * lyapunov_config.target_norm
                            / separation_norm
                        )
                        try:
                            (
                                renormalized_delta_positions,
                                renormalized_delta_velocities,
                            ) = delta_arrays_from_scaled_delta_vector(
                                state,
                                renormalized_scaled_delta,
                            )
                        except ValueError as exc:
                            lyapunov_runtime_warnings.append(
                                f"Tangent renormalization failed: {exc}"
                            )
                            lyapunov_state = None
                            lyapunov_delta_positions = None
                            lyapunov_delta_velocities = None
                            lyapunov_tangent_acceleration = None
                            continue
                        post_renorm_norm = float(np.linalg.norm(renormalized_scaled_delta))
                        renormalized = None
                    else:
                        try:
                            renormalized = renormalize_to_scaled_norm(
                                state,
                                lyapunov_state,
                                target_norm=lyapunov_config.target_norm,
                                sun_index=sun_index,
                                preserve_barycenter=True,
                            )
                        except ValueError as exc:
                            lyapunov_runtime_warnings.append(
                                f"Renormalization failed: {exc}"
                            )
                            lyapunov_state = None
                            lyapunov_acceleration = None
                            continue
                        post_renorm_norm = renormalized.separation_norm_after

                local_lambda = log_growth / interval_years if interval_years > 0.0 else math.nan
                elapsed_years = seconds_to_years(time_s)
                running_lambda = (
                    lyapunov_cumulative_log_growth / elapsed_years
                    if elapsed_years > 0.0
                    else math.nan
                )
                sample = LyapunovSample(
                    time_years=elapsed_years,
                    separation_norm=separation_norm,
                    pre_renorm_separation_norm=separation_norm,
                    post_renorm_separation_norm=post_renorm_norm,
                    target_norm=lyapunov_config.target_norm,
                    growth_factor=growth_factor,
                    log_growth_increment=log_growth,
                    cumulative_log_growth=lyapunov_cumulative_log_growth,
                    local_lambda_1_per_year=local_lambda,
                    running_lambda_1_per_year=running_lambda,
                    lyapunov_time_years=lyapunov_time_years(running_lambda),
                    max_position_separation_m=component_diag.max_position_separation_m,
                    max_velocity_separation_m_s=component_diag.max_velocity_separation_m_s,
                    dominant_body_in_norm=component_diag.dominant_body_name,
                    dominant_component_type=component_diag.dominant_component_type,
                    renorm_interval_years_actual=interval_years,
                    cosine_with_previous_delta_direction=cosine_with_previous,
                    cosine_with_initial_delta_direction=cosine_with_initial,
                    direction_reset_suspected=direction_reset_suspected,
                )
                lyapunov_samples.append(sample)
                if lyapunov_outputs is not None:
                    write_lyapunov_sample(lyapunov_outputs, sample)

                if step_index != total_steps:
                    if not lyapunov_config.no_renorm:
                        if lyapunov_config.method == "tangent":
                            lyapunov_delta_positions = renormalized_delta_positions
                            lyapunov_delta_velocities = renormalized_delta_velocities
                            lyapunov_state = state_with_delta(
                                state,
                                lyapunov_delta_positions,
                                lyapunov_delta_velocities,
                            )
                            lyapunov_tangent_acceleration = tangent_acceleration_newtonian(
                                state,
                                lyapunov_delta_positions,
                                G=G_SI,
                            )
                            lyapunov_previous_delta_vector = renormalized_scaled_delta.copy()
                        else:
                            lyapunov_state = renormalized.state
                            lyapunov_acceleration = accel_func(
                                lyapunov_state,
                                G=G_SI,
                                **accel_kwargs,
                            )
                            lyapunov_previous_delta_vector = scaled_phase_space_delta_vector(
                                state,
                                lyapunov_state,
                            )
                    else:
                        lyapunov_previous_delta_vector = current_delta_vector.copy()
                    lyapunov_previous_sample_norm = (
                        lyapunov_config.target_norm
                        if not lyapunov_config.no_renorm
                        else separation_norm
                    )
                    lyapunov_last_renorm_s = time_s
                    while lyapunov_next_renorm_s <= time_s + 1.0e-9:
                        lyapunov_next_renorm_s += lyapunov_config.renorm_years * JULIAN_YEAR_S

                if len(lyapunov_samples) % 50 == 0 and lyapunov_outputs is not None:
                    lyapunov_outputs.flush()

            if step_index % record_every_steps == 0 or step_index == total_steps:
                final_invariant_row = write_snapshot(
                    time_s,
                    state,
                    body_names,
                    outputs,
                    reference,
                    extrema,
                    sun_index=sun_index,
                )
                n_records += 1
                if n_records % 50 == 0:
                    outputs.flush()

            if (
                checkpoint_dir is not None
                and checkpoint_every_s is not None
                and (time_s + 1.0e-9 >= next_checkpoint_s or step_index == total_steps)
            ):
                outputs.flush()
                if lyapunov_outputs is not None:
                    lyapunov_outputs.flush()
                checkpoint_path = checkpoint_dir / checkpoint_filename(checkpoint_tag, time_s)
                write_checkpoint_atomic(
                    checkpoint_path,
                    time_s=time_s,
                    n_records=n_records,
                    state=state,
                    reference=reference,
                    min_tracker=min_tracker,
                    extrema=extrema,
                    config_hash=config_hash,
                    lyapunov_cumulative_log_growth=lyapunov_cumulative_log_growth,
                    lyapunov_last_renorm_s=lyapunov_last_renorm_s,
                    lyapunov_next_renorm_s=lyapunov_next_renorm_s,
                    lyapunov_previous_sample_norm=lyapunov_previous_sample_norm,
                    lyapunov_delta_positions=lyapunov_delta_positions,
                    lyapunov_delta_velocities=lyapunov_delta_velocities,
                    lyapunov_initial_delta_vector=lyapunov_initial_delta_vector,
                    lyapunov_previous_delta_vector=lyapunov_previous_delta_vector,
                    lyapunov_samples=lyapunov_samples,
                    rng_state=None,
                )
                prune_checkpoints(checkpoint_dir, keep=keep_checkpoints)
                while next_checkpoint_s <= time_s + 1.0e-9:
                    next_checkpoint_s += checkpoint_every_s

            progress.update(1)
    finally:
        progress.close()

    if final_invariant_row is None:
        final_invariant_row = invariant_diagnostics_row(time_s, state, reference)
        update_extrema(extrema, final_invariant_row)

    lyapunov_result = None
    if lyapunov_config is not None:
        fit = fit_lyapunov_growth(
            lyapunov_samples,
            fit_start_years=lyapunov_config.fit_start_years,
            fit_end_years=lyapunov_config.fit_end_years,
        )
        warnings = lyapunov_warnings(
            duration_years=seconds_to_years(duration_s),
            step_days=dt_s / DAY_S,
            renorm_years=lyapunov_config.renorm_years,
            fit=fit,
            samples=lyapunov_samples,
            model_scope=lyapunov_config.model_scope,
            gr_model=lyapunov_config.gr_model,
            invariant_extrema=extrema,
            no_renorm=lyapunov_config.no_renorm,
        )
        warnings = lyapunov_runtime_warnings + warnings
        final_running_lambda = (
            lyapunov_samples[-1].running_lambda_1_per_year
            if lyapunov_samples
            else None
        )
        final_running_time = (
            lyapunov_samples[-1].lyapunov_time_years
            if lyapunov_samples
            else None
        )
        lyapunov_result = LyapunovResult(
            samples=lyapunov_samples,
            config=lyapunov_config,
            fit=fit,
            warnings=warnings,
            debug_warnings=warnings,
            final_running_lambda_1_per_year=final_running_lambda,
            final_running_lyapunov_time_years=final_running_time,
        )

    return IntegrationResult(
        final_state=state,
        actual_duration_s=time_s,
        n_steps=total_steps,
        n_records=n_records,
        min_tracker=min_tracker,
        extrema=extrema,
        min_separation_sampling="each leapfrog integration step",
        lyapunov_result=lyapunov_result,
        two_body_validation=(
            two_body_tracker.result(
                duration_s=time_s,
                step_days=dt_s / DAY_S,
                n_steps=total_steps,
            )
            if two_body_tracker is not None
            else None
        ),
        poincare_samples=poincare_samples,
    )


def integrate_dop853_streaming(
    state0: NBodyState,
    body_names: tuple[str, ...],
    outputs: CsvOutputs,
    *,
    duration_s: float,
    max_step_s: float,
    record_interval_s: float,
    accel_func,
    accel_kwargs: dict,
    show_progress: bool,
    sun_index: int,
) -> IntegrationResult:
    state = state0.copy()
    y_current = pack_state(state)
    reference = invariant_reference(state, G=G_SI)
    min_tracker = PairwiseMinimumTracker.create(body_names)
    min_tracker.update(0.0, state.positions)

    extrema = initial_extrema()
    write_snapshot(0.0, state, body_names, outputs, reference, extrema, sun_index=sun_index)

    n_records = 1
    current_t = 0.0
    next_record_t = min(record_interval_s, duration_s)

    progress = tqdm(
        total=seconds_to_years(duration_s),
        desc="DOP853 years",
        disable=not show_progress,
    )

    try:
        while current_t < duration_s - 1.0e-9:
            target_t = min(next_record_t, duration_s)
            if target_t <= current_t:
                target_t = min(current_t + record_interval_s, duration_s)

            sol = solve_ivp(
                fun=lambda t, y: rhs_solve_ivp(
                    t,
                    y,
                    masses=state0.masses,
                    accel_func=accel_func,
                    accel_kwargs=accel_kwargs,
                    G=G_SI,
                ),
                t_span=(current_t, target_t),
                y0=y_current,
                method="DOP853",
                t_eval=[target_t],
                rtol=1.0e-12,
                atol=1.0e-15,
                vectorized=False,
                max_step=max_step_s,
            )

            if not sol.success:
                raise RuntimeError(f"DOP853 integration failed: {sol.message}")

            previous_t = current_t
            current_t = float(sol.t[-1])
            y_current = sol.y[:, -1].copy()
            state = unpack_state(y_current, state0.masses)
            min_tracker.update(current_t, state.positions)

            write_snapshot(
                current_t,
                state,
                body_names,
                outputs,
                reference,
                extrema,
                sun_index=sun_index,
            )
            n_records += 1
            if n_records % 20 == 0:
                outputs.flush()

            progress.update(seconds_to_years(current_t - previous_t))
            next_record_t += record_interval_s
    finally:
        progress.close()

    nominal_steps = int(math.ceil(duration_s / max_step_s))
    return IntegrationResult(
        final_state=state,
        actual_duration_s=current_t,
        n_steps=nominal_steps,
        n_records=n_records,
        min_tracker=min_tracker,
        extrema=extrema,
        min_separation_sampling="recorded DOP853 output samples",
        lyapunov_result=None,
    )


def write_min_separations(path: Path, tracker: PairwiseMinimumTracker) -> None:
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=MIN_SEPARATION_FIELDS)
        writer.writeheader()
        for row in tracker.rows():
            writer.writerow(row)


def make_summary(
    *,
    args: argparse.Namespace,
    tag: str,
    body_names: tuple[str, ...],
    paths: dict[str, Path],
    result: IntegrationResult,
    runtime_s: float,
) -> dict:
    two_body_row = (
        result.two_body_validation.as_row()
        if result.two_body_validation is not None
        else None
    )
    if two_body_row is not None:
        two_body_row["runtime_seconds"] = runtime_s
    return {
        "mode": MODE_DESCRIPTION,
        "scientific_boundary": {
            "uses_earth_moon_barycenter": "earth barycenter" in body_names,
            "uses_explicit_earth_and_moon": False,
            "uses_empirical_lunar_calibration": False,
            "uses_american_ephemeris_apparent_geocentric_tropical_output": False,
        },
        "configuration": {
            "kernel_path": args.kernel_path,
            "start_date_utc": args.start_date.isoformat(),
            "duration_years_requested": args.duration_years,
            "duration_years_actual": seconds_to_years(result.actual_duration_s),
            "step_days": args.step_days,
            "record_every_years": args.record_every_years,
            "backend": getattr(args, "backend", "inhouse"),
            "include_pluto": args.include_pluto,
            "gr_model": args.gr_model,
            "integrator": args.integrator,
            "active_integrator": (
                args.rebound_integrator
                if getattr(args, "backend", "inhouse") == "rebound"
                else args.integrator
            ),
            "active_gr_model": (
                args.rebound_gr_model
                if getattr(args, "backend", "inhouse") == "rebound"
                else args.gr_model
            ),
            "rebound_integrator": getattr(args, "rebound_integrator", None),
            "rebound_gr_model": getattr(args, "rebound_gr_model", None),
            "rebound_ias15_epsilon": getattr(args, "rebound_ias15_epsilon", None),
            "rebound_simulationarchive": getattr(args, "rebound_simulationarchive", None),
            "rebound_archive_interval_years": getattr(args, "rebound_archive_interval_years", None),
            "rebound_resume": getattr(args, "rebound_resume", None),
            "model_scope": args.model_scope,
            "tag": tag,
            "body_names": body_names,
            "orbital_elements_reference_plane": "ecliptic_j2000",
            "checkpoint_every_years": getattr(args, "checkpoint_every_years", None),
            "checkpoint_dir": getattr(args, "checkpoint_dir", None),
            "resumed_from_checkpoint": getattr(args, "resume_from_checkpoint", None),
        },
        "integrator_notes": {
            "leapfrog_default_for_long_term": True,
            "dop853_role": "short validation/comparison runs",
            "gr_leapfrog_symplectic_note": (
                "Sun 1PN GR is included through the acceleration callback; "
                "with gr_model='sun' the leapfrog update is not exactly symplectic."
                if args.integrator == "leapfrog" and args.gr_model == "sun"
                else None
            ),
        },
        "counts": {
            "n_steps_or_nominal_max_steps": result.n_steps,
            "n_records": result.n_records,
        },
        "runtime": {
            "wall_clock_seconds": runtime_s,
            "wall_clock_minutes": runtime_s / 60.0,
        },
        "warnings": list(result.runtime_warnings or []),
        "rebound_resume": (
            {
                "requested": result.rebound_resume_info.requested,
                "archive_path": result.rebound_resume_info.archive_path,
                "resumed_from_time_years": result.rebound_resume_info.resumed_from_time_years,
                "megno_state_validated": result.rebound_resume_info.megno_state_validated,
                "duplicate_rows_removed": result.rebound_resume_info.duplicate_rows_removed,
                "warnings": result.rebound_resume_info.warnings,
            }
            if result.rebound_resume_info is not None
            else {
                "requested": getattr(args, "rebound_resume", None),
                "resumed_from_time_years": None,
            }
        ),
        "diagnostic_extrema_over_records": result.extrema,
        "min_separation_sampling": result.min_separation_sampling,
        "min_separations": result.min_tracker.rows(),
        "lyapunov": (
            {
                "enabled": True,
                "summary_path": str(paths.get("lyapunov_summary")),
                "growth_plot_path": str(paths.get("lyapunov_growth_plot")),
                "csv_path": str(paths.get("lyapunov")),
                "warning": (
                    "Lyapunov estimates require timestep convergence checks "
                    "and depend on timestep, force model, norm, perturbation, "
                    "renormalization interval, and fit window."
                ),
            }
            if result.lyapunov_result is not None and "lyapunov" in paths
            else {"enabled": False}
        ),
        "poincare": (
            {
                "enabled": True,
                "scope": "exploratory heliocentric ecliptic section crossings",
                "caution": (
                    "Poincare sections are most rigorous for lower-dimensional systems; "
                    "for the full Solar System these are visual diagnostics, not proof of chaos."
                ),
                "csv_path": str(paths.get("poincare")),
                "plot_path": str(paths.get("poincare_plot")),
                "crossing_count": len(result.poincare_samples or []),
            }
            if "poincare" in paths
            else {"enabled": False}
        ),
        "frequency_map": (
            {
                "enabled": True,
                "csv_path": str(paths.get("frequency_map")),
                "caution": "NAFF-lite / FFT-lite post-processing; not full Laskar NAFF.",
            }
            if "frequency_map" in paths
            else {"enabled": False}
        ),
        "fli_megno_lite": (
            {
                "enabled": True,
                "csv_path": str(paths.get("fli_megno")),
                "summary_path": str(paths.get("fli_megno_summary")),
                "caution": (
                    "Finite-time tangent-growth indicators only; duration scaling is required before interpretation."
                ),
            }
            if "fli_megno" in paths
            else {"enabled": False}
        ),
        "rebound_megno": (
            {
                "enabled": True,
                "csv_path": str(paths.get("megno")),
                "summary_path": str(paths.get("megno_summary")),
                "growth_plot_path": str(paths.get("megno_growth_plot")),
                "final_megno": _finite_or_none(result.rebound_megno_result.final_megno),
                "final_mean_megno": _finite_or_none(
                    result.rebound_megno_result.final_mean_megno
                ),
                "final_lcn_raw": _finite_or_none(
                    result.rebound_megno_result.final_lcn_raw
                ),
                "last_finite_lcn": _finite_or_none(
                    result.rebound_megno_result.last_finite_lcn
                ),
                "last_finite_lcn_time_years": _finite_or_none(
                    result.rebound_megno_result.last_finite_lcn_time_years
                ),
                "megno_slope_window_estimates": (
                    result.rebound_megno_result.megno_slope_window_estimates
                ),
                "estimated_lyapunov_if_available": _finite_or_none(
                    result.rebound_megno_result.estimated_lyapunov_if_available
                ),
                "classification_hint": result.rebound_megno_result.classification_hint,
                "caveats": result.rebound_megno_result.caveats,
                "caution": (
                    "REBOUND-native MEGNO/LCN values are finite-time diagnostics here; "
                    "duration scaling and timestep/integrator comparison are required "
                    "before interpreting an asymptotic Lyapunov exponent."
                ),
            }
            if result.rebound_megno_result is not None and "megno" in paths
            else {"enabled": False}
        ),
        "two_body_validation": (
            {
                "enabled": True,
                "scope": "Sun + one planetary barycenter, Newtonian near-integrable validation",
                "csv_path": str(paths.get("two_body_validation")),
                "diagnostics": two_body_row,
            }
            if result.two_body_validation is not None
            else {"enabled": False}
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    reject_empirical_lunar_args(parser, list(argv))
    args = parser.parse_args(argv)
    validate_args(parser, args)

    tag = sanitize_tag(args.tag)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(
        output_dir,
        tag,
        with_lyapunov=args.with_lyapunov,
        lyapunov_no_renorm=args.lyapunov_no_renorm,
        with_poincare=args.with_poincare,
        poincare_body=args.poincare_body,
        with_frequency_map=args.with_frequency_map,
        with_fli_megno=args.with_fli or args.with_megno_lite,
        with_megno=args.with_megno,
        model_scope=args.model_scope,
    )

    bodies = stability_body_list(args.model_scope, include_pluto=args.include_pluto)
    if "earth" in bodies or "moon" in bodies:
        raise RuntimeError("Stability mode must not use explicit Earth or Moon bodies.")
    if args.model_scope not in TWO_BODY_MODEL_SCOPES and "earth barycenter" not in bodies:
        raise RuntimeError("Stability mode requires the Earth-Moon barycenter.")

    sun_index = bodies.index("sun")
    config_hash = stability_config_hash(args, bodies)
    checkpoint_data = None
    rebound_resume_path = resolve_rebound_resume_archive(args)
    rebound_resume_removed: dict[str, int] = {}
    rebound_resume_n_records = 0
    rebound_resume_extrema: dict[str, float] | None = None
    rebound_resume_min_tracker: PairwiseMinimumTracker | None = None
    rebound_resume_megno_samples: list[ReboundMegnoSample] = []
    if args.resume_from_checkpoint is not None:
        try:
            checkpoint_data = load_checkpoint(Path(args.resume_from_checkpoint))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error(f"Could not load checkpoint: {exc}")
        if checkpoint_data.config_hash and checkpoint_data.config_hash != config_hash:
            parser.error(
                "Checkpoint config hash does not match current run configuration. "
                "Use matching model, timestep, record cadence, force model, and tangent diagnostic settings."
            )
        for required_key in ("stability_timeseries", "orbital_elements", "invariants"):
            if not paths[required_key].exists():
                parser.error(
                    f"--resume-from-checkpoint requires existing CSV output for append: {paths[required_key]}"
                )
        if args.with_lyapunov and not paths["lyapunov"].exists():
            parser.error(
                f"--resume-from-checkpoint requires existing Lyapunov CSV output for append: {paths['lyapunov']}"
            )
    if rebound_resume_path is not None:
        rebound_module = optional_import_module("rebound")
        if rebound_module is None:
            parser.error("REBOUND is not installed; cannot use --rebound-resume.")
        try:
            probe_sim = load_rebound_archive_snapshot(rebound_module, rebound_resume_path)
            rebound_resume_time_s = float(probe_sim.t)
        except RuntimeError as exc:
            parser.error(str(exc))
        if rebound_resume_time_s <= 0.0:
            parser.error(
                "--rebound-resume requested, but the archive latest snapshot is not at nonzero time."
            )
        if rebound_resume_time_s >= args.duration_years * JULIAN_YEAR_S - 1.0e-6:
            parser.error(
                "--rebound-resume archive is already at or beyond --duration-years; "
                "request a longer duration or use the completed outputs."
            )
        try:
            for key in ("stability_timeseries", "orbital_elements", "invariants"):
                rebound_resume_removed[key] = truncate_csv_after_time(
                    paths[key],
                    time_column="time_years",
                    max_time_years=seconds_to_years(rebound_resume_time_s),
                )
            if args.with_megno and "megno" in paths:
                rebound_resume_removed["megno"] = truncate_csv_after_time(
                    paths["megno"],
                    time_column="time_years",
                    max_time_years=seconds_to_years(rebound_resume_time_s),
                )
            rebound_resume_n_records = count_csv_data_rows(paths["stability_timeseries"])
            rebound_resume_extrema = extrema_from_invariants_csv(paths["invariants"])
            rebound_resume_min_tracker = min_tracker_from_csv(paths["min_separations"], bodies)
            if args.with_megno and "megno" in paths:
                rebound_resume_megno_samples = read_rebound_megno_samples(paths["megno"])
        except RuntimeError as exc:
            parser.error(f"Could not prepare CSV outputs for REBOUND resume: {exc}")
    try:
        poincare_config = build_poincare_config(args, bodies) if args.with_poincare else None
        if args.with_frequency_map:
            parse_frequency_body_choices(args.frequency_bodies, bodies)
    except ValueError as exc:
        parser.error(str(exc))

    accel_func, accel_kwargs = select_acceleration_model(
        args.gr_model,
        sun_index=sun_index,
    )

    print(f"[Stability] {MODE_DESCRIPTION}.", flush=True)
    print(
        "[Stability] American Ephemeris apparent/geocentric/tropical machinery is not used.",
        flush=True,
    )
    if "earth barycenter" in bodies:
        print("[Stability] Earth-Moon barycenter is included; explicit Earth and Moon are omitted.")
    else:
        print("[Stability] Validation scope omits Earth-Moon barycenter; explicit Earth and Moon are still omitted.")
    print(f"[Stability] bodies: {bodies}")
    print(f"[Stability] model_scope: {args.model_scope}")
    if args.backend == "inhouse":
        print(f"[Stability] integrator: {args.integrator}")
        print(f"[Stability] gr_model: {args.gr_model}")
    else:
        print(f"[Stability] in-house integrator flag ignored for REBOUND backend: {args.integrator}")
    if checkpoint_data is not None:
        print(
            f"[Stability] Resuming from checkpoint {checkpoint_data.path} "
            f"at t={seconds_to_years(checkpoint_data.time_s):g} years.",
            flush=True,
        )
    if rebound_resume_path is not None:
        removed_total = sum(rebound_resume_removed.values())
        print(
            f"[Stability] REBOUND resume requested from {rebound_resume_path}; "
            f"prepared CSV outputs and removed {removed_total} duplicate/future rows.",
            flush=True,
        )

    if args.integrator == "dop853":
        print(
            "[Stability] DOP853 selected for short validation/comparison runs; "
            "leapfrog remains the default long-term stability integrator.",
            flush=True,
        )
    if args.integrator == "leapfrog" and args.gr_model == "sun":
        print(
            "[Stability] Sun 1PN GR is included through the acceleration callback; "
            "the leapfrog method is no longer exactly symplectic.",
            flush=True,
        )
        print(
            "[Stability] Current Sun 1PN GR approximation applies the perturbation "
            "to planets without an equal-and-opposite Sun reaction, so it is not "
            "momentum-conserving. Use gr_model=none for conservation baselines.",
            flush=True,
        )

    config = EphemerisConfig(kernel_path=args.kernel_path)
    state0 = initial_state_solar_system_barycentric(
        args.start_date,
        bodies=bodies,
        config=config,
    )

    lyapunov_config = None
    lyapunov_initial_state = None
    tangent_diagnostics = args.with_lyapunov or args.with_fli or args.with_megno_lite
    if tangent_diagnostics:
        lyapunov_config, lyapunov_initial_state = build_lyapunov_config(
            args,
            state0,
            bodies,
            sun_index=sun_index,
        )
        print(
            "[Stability] Tangent diagnostics enabled: finite-time Benettin-style renormalized propagation.",
            flush=True,
        )
        print(f"[Stability] Lyapunov method: {lyapunov_config.method}")
        print(
            "[Stability] Finite-time Lyapunov estimates require duration-scaling "
            "and timestep convergence checks before interpretation.",
            flush=True,
        )
        print(f"[Stability] Lyapunov perturbation target: {lyapunov_config.body_names}")
        print(f"[Stability] Lyapunov target norm: {lyapunov_config.target_norm:.6e}")
        print(f"[Stability] Lyapunov renormalization: {lyapunov_config.renorm_years:g} years")
        if lyapunov_config.no_renorm:
            print("[Stability] Lyapunov no-renormalization debug mode is enabled.")
        print(
            "[Stability] Lyapunov fit window: "
            f"{lyapunov_config.fit_start_years:g} to {lyapunov_config.fit_end_years:g} years"
        )
    if args.with_poincare and poincare_config is not None:
        print(
            "[Stability] Poincare-style section enabled: exploratory heliocentric "
            f"{poincare_config.plane}=0 crossings for {poincare_config.body_name}.",
            flush=True,
        )
    if args.with_frequency_map:
        print(
            "[Stability] Frequency map enabled: NAFF-lite / FFT-lite post-processing of orbital elements.",
            flush=True,
        )
    if args.with_fli or args.with_megno_lite:
        print(
            "[Stability] FLI/MEGNO-lite enabled: finite-time tangent-growth indicators only.",
            flush=True,
        )
    if args.with_megno:
        print(
            "[Stability] REBOUND-native MEGNO enabled: finite-time variational diagnostic.",
            flush=True,
        )
        if args.rebound_chaos_method == "megno":
            print(
                "[Stability] REBOUND-native LCN recording enabled via Simulation.lyapunov() "
                "for --rebound-chaos-method megno.",
                flush=True,
            )
        elif args.with_rebound_lyapunov:
            print("[Stability] REBOUND-native LCN output requested via Simulation.lyapunov().")
        else:
            print("[Stability] REBOUND-native LCN recording disabled.")

    duration_s = args.duration_years * JULIAN_YEAR_S
    step_s = args.step_days * DAY_S
    record_interval_s = args.record_every_years * JULIAN_YEAR_S

    print(f"[Stability] duration: {args.duration_years:g} Julian years")
    print(f"[Stability] step: {args.step_days:g} days")
    print(f"[Stability] record cadence: {args.record_every_years:g} Julian years")
    print(f"[Stability] backend: {args.backend}")
    if args.backend == "rebound":
        print(f"[Stability] REBOUND integrator: {args.rebound_integrator}")
        print(f"[Stability] REBOUND GR model: {args.rebound_gr_model}")
        if args.rebound_integrator == "ias15":
            print(f"[Stability] REBOUND IAS15 epsilon: {args.rebound_ias15_epsilon:g}")
        if args.rebound_simulationarchive is not None:
            print(f"[Stability] REBOUND SimulationArchive: {args.rebound_simulationarchive}")
    print(f"[Stability] output directory: {output_dir}")

    append_outputs = checkpoint_data is not None or rebound_resume_path is not None
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir is not None
        else output_dir / f"checkpoints_{tag}"
    )
    checkpoint_every_s = (
        args.checkpoint_every_years * JULIAN_YEAR_S
        if args.checkpoint_every_years is not None
        else None
    )
    if checkpoint_every_s is not None:
        print(f"[Stability] checkpoint directory: {checkpoint_dir}")
        print(f"[Stability] checkpoint cadence: {args.checkpoint_every_years:g} years")

    outputs = open_csv_outputs(paths, append=append_outputs)
    lyapunov_outputs = open_lyapunov_outputs(paths, append=append_outputs) if args.with_lyapunov else None
    poincare_outputs = open_poincare_outputs(paths) if args.with_poincare else None
    megno_outputs = open_rebound_megno_outputs(paths, append=append_outputs) if args.with_megno else None
    effective_with_rebound_lyapunov = bool(
        args.with_rebound_lyapunov
        or (args.with_megno and args.rebound_chaos_method == "megno")
    )
    start_wall = time.perf_counter()
    try:
        if args.backend == "rebound":
            result = integrate_rebound_streaming(
                state0,
                bodies,
                outputs,
                duration_s=duration_s,
                dt_s=step_s,
                record_interval_s=record_interval_s,
                show_progress=not args.no_progress_bar,
                sun_index=sun_index,
                rebound_integrator=args.rebound_integrator,
                rebound_gr_model=args.rebound_gr_model,
                rebound_ias15_epsilon=args.rebound_ias15_epsilon,
                simulationarchive_path=(
                    Path(args.rebound_simulationarchive)
                    if args.rebound_simulationarchive is not None
                    else None
                ),
                simulationarchive_interval_s=(
                    args.rebound_archive_interval_years * JULIAN_YEAR_S
                    if args.rebound_archive_interval_years is not None
                    else None
                ),
                megno_outputs=megno_outputs,
                megno_record_interval_s=(
                    args.megno_record_every_years * JULIAN_YEAR_S
                    if args.megno_record_every_years is not None
                    else record_interval_s
                ),
                with_rebound_lyapunov=effective_with_rebound_lyapunov,
                megno_seed=args.megno_seed,
                megno_duration_scaling_mode=args.megno_duration_scaling_mode,
                model_scope=args.model_scope,
                rebound_resume_path=rebound_resume_path,
                rebound_resume_requested=args.rebound_resume,
                resume_duplicate_rows_removed=rebound_resume_removed,
                resume_min_tracker=rebound_resume_min_tracker,
                resume_extrema=rebound_resume_extrema,
                resume_n_records=rebound_resume_n_records,
                resume_megno_samples=rebound_resume_megno_samples,
            )
        elif args.integrator == "leapfrog":
            result = integrate_leapfrog_streaming(
                state0,
                bodies,
                outputs,
                duration_s=duration_s,
                dt_s=step_s,
                record_interval_s=record_interval_s,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                show_progress=not args.no_progress_bar,
                sun_index=sun_index,
                lyapunov_config=lyapunov_config,
                lyapunov_outputs=lyapunov_outputs,
                lyapunov_initial_state=lyapunov_initial_state,
                poincare_config=poincare_config,
                poincare_outputs=poincare_outputs,
                checkpoint_data=checkpoint_data,
                checkpoint_dir=checkpoint_dir if checkpoint_every_s is not None else None,
                checkpoint_every_s=checkpoint_every_s,
                keep_checkpoints=args.keep_checkpoints,
                checkpoint_tag=tag,
                config_hash=config_hash,
            )
        elif args.integrator == "dop853":
            result = integrate_dop853_streaming(
                state0,
                bodies,
                outputs,
                duration_s=duration_s,
                max_step_s=step_s,
                record_interval_s=record_interval_s,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                show_progress=not args.no_progress_bar,
                sun_index=sun_index,
            )
        else:
            raise ValueError(f"Unsupported integrator: {args.integrator!r}")

        outputs.flush()
        if lyapunov_outputs is not None:
            lyapunov_outputs.flush()
        if poincare_outputs is not None:
            poincare_outputs.flush()
        if megno_outputs is not None:
            megno_outputs.flush()
    finally:
        outputs.close()
        if lyapunov_outputs is not None:
            lyapunov_outputs.close()
        if poincare_outputs is not None:
            poincare_outputs.close()
        if megno_outputs is not None:
            megno_outputs.close()

    runtime_s = time.perf_counter() - start_wall
    write_min_separations(paths["min_separations"], result.min_tracker)
    if result.two_body_validation is not None and "two_body_validation" in paths:
        write_two_body_validation_csv(
            paths["two_body_validation"],
            result.two_body_validation,
            runtime_seconds=runtime_s,
        )

    if result.lyapunov_result is not None and lyapunov_outputs is not None:
        if (
            result.lyapunov_result.config.no_renorm
            and lyapunov_outputs.no_renorm_path is not None
        ):
            write_no_renorm_separation_csv(
                lyapunov_outputs.no_renorm_path,
                result.lyapunov_result.samples,
            )
        plot_lyapunov_growth(
            result.lyapunov_result.samples,
            result.lyapunov_result.fit,
            lyapunov_outputs.plot_path,
        )
        lyapunov_summary = lyapunov_summary_dict(
            args=args,
            tag=tag,
            body_names=bodies,
            result=result.lyapunov_result,
            outputs=lyapunov_outputs,
            runtime_s=runtime_s,
        )
        write_summary(lyapunov_outputs.summary_path, lyapunov_summary)

    if result.rebound_megno_result is not None and megno_outputs is not None:
        plot_megno_growth(result.rebound_megno_result.samples, megno_outputs.plot_path)
        megno_summary = rebound_megno_summary_dict(
            args=args,
            tag=tag,
            result=result.rebound_megno_result,
            outputs=megno_outputs,
            runtime_s=runtime_s,
            resume_info=result.rebound_resume_info,
        )
        write_summary(megno_outputs.summary_path, megno_summary)

    if result.poincare_samples is not None and args.with_poincare and poincare_outputs is not None:
        plot_poincare_samples(result.poincare_samples, poincare_outputs.plot_path)

    fli_megno_summary = None
    if (args.with_fli or args.with_megno_lite) and result.lyapunov_result is not None:
        fli_megno_samples = build_fli_megno_samples(
            result.lyapunov_result,
            model_scope=args.model_scope,
        )
        fli_megno_summary = write_fli_megno_outputs(
            csv_path=paths["fli_megno"],
            summary_path=paths["fli_megno_summary"],
            samples=fli_megno_samples,
            lyapunov_result=result.lyapunov_result,
            args=args,
            runtime_s=runtime_s,
        )

    frequency_summary = None
    if args.with_frequency_map:
        frequency_summary = run_frequency_map_analysis(
            orbital_elements_path=paths["orbital_elements"],
            output_csv_path=paths["frequency_map"],
            output_dir=output_dir,
            tag=tag,
            body_names=bodies,
            args=args,
        )

    summary = make_summary(
        args=args,
        tag=tag,
        body_names=bodies,
        paths=paths,
        result=result,
        runtime_s=runtime_s,
    )
    if frequency_summary is not None:
        summary["frequency_map"] = frequency_summary
    if fli_megno_summary is not None:
        summary["fli_megno_lite"]["final_fli"] = fli_megno_summary.get("final_fli")
        summary["fli_megno_lite"]["final_megno_lite"] = fli_megno_summary.get("final_megno_lite")
        summary["fli_megno_lite"]["final_finite_time_lambda"] = fli_megno_summary.get(
            "final_finite_time_lambda"
        )
    write_summary(paths["summary"], summary)

    print("[Stability] complete.")
    print(f"[Stability] wall-clock runtime: {runtime_s:.3f} s")
    print(f"[Stability] records written: {result.n_records}")
    if result.lyapunov_result is not None:
        fit_lambda = result.lyapunov_result.fit.get("lambda_1_per_year")
        fit_time = result.lyapunov_result.fit.get("lyapunov_time_years")
        if isinstance(fit_lambda, (float, int)) and isinstance(fit_time, (float, int)):
            print(
                "[Stability] Lyapunov fit: "
                f"lambda={float(fit_lambda):.6e} 1/year, "
                f"finite-time={float(fit_time):.6e} years"
            )
        print("[Stability] Tangent diagnostic warnings:")
        for warning in result.lyapunov_result.warnings:
            print(f"  - {warning}")
    if result.rebound_megno_result is not None:
        print(
            "[Stability] REBOUND MEGNO final: "
            f"Y={result.rebound_megno_result.final_megno:.6e}, "
            f"LCN={result.rebound_megno_result.estimated_lyapunov_if_available:.6e} 1/year, "
            f"classification={result.rebound_megno_result.classification_hint}"
        )
    if args.with_poincare and result.poincare_samples is not None:
        print(f"[Stability] Poincare crossings: {len(result.poincare_samples)}")
    if frequency_summary is not None:
        print(f"[Stability] Frequency-map rows: {frequency_summary['row_count']}")
    if fli_megno_summary is not None:
        print(
            "[Stability] FLI/MEGNO-lite final: "
            f"FLI={fli_megno_summary.get('final_fli')}, "
            f"MEGNO-lite={fli_megno_summary.get('final_megno_lite')}"
        )
    if result.runtime_warnings:
        print("[Stability] Runtime/backend warnings:")
        for warning in result.runtime_warnings:
            print(f"  - {warning}")
    for key, path in paths.items():
        print(f"[Stability] wrote {key}: {path}")


if __name__ == "__main__":
    main()
