from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .m0_energy_precision_diagnosis import float64_energy
from .m0_step3e1_offline_diagnosis import _element_pair, _element_series, _reconstruct, _scaled_defect, _wrap
from .m0_step3e_convergence import _naff_lite
from .m0_step3f1_analysis import METRIC_FIELDS, _artifact, _atomic_csv, _atomic_text, _finite, _historical_tangent, _ias15, _new_lane
from .m0_step3f1_contract import AU_M, BODY_NAMES, JULIAN_YEAR_S, VELOCITY_SCALE, load_json, require, sha256_file, validate_manifest
from .m0_step3f1_runner import audit
from .m0_timestep_convergence import RunData, _compute_elements, _pair_tangent
from .nbody import G_SI, NBodyState
from .orbital_elements import ARCSEC_PER_RAD
from .rebound_gr_tangent_backend_cli import atomic_write_json
from .stability_diagnostics import total_angular_momentum_vector


def _rtn(left: RunData, right: RunData) -> dict[str, Any]:
    output = {}
    for body, name in enumerate(BODY_NAMES[1:], start=1):
        r = right.positions[:, body] - right.positions[:, 0]
        v = right.velocities[:, body] - right.velocities[:, 0]
        radial = r / np.linalg.norm(r, axis=1)[:, None]
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        transverse = np.cross(normal, radial)
        basis = np.stack((radial, transverse, normal), axis=2)
        dr = (left.positions[:, body] - left.positions[:, 0]) - r
        dv = (left.velocities[:, body] - left.velocities[:, 0]) - v
        pr = np.einsum("ni,nij->nj", dr, basis)
        pv = np.einsum("ni,nij->nj", dv, basis)
        output[name] = {
            "position_rms_m": {axis: float(np.sqrt(np.mean(pr[:, i] ** 2))) for i, axis in enumerate(("R", "T", "N"))},
            "velocity_rms_m_per_s": {axis: float(np.sqrt(np.mean(pv[:, i] ** 2))) for i, axis in enumerate(("R", "T", "N"))},
            "transverse_position_squared_fraction": float(np.sum(pr[:, 1] ** 2) / max(float(np.sum(pr**2)), 1.0e-300)),
        }
    return output


def _phase_and_orbit(left: RunData, right: RunData) -> dict[str, Any]:
    left_elements = _element_series(left)
    right_elements = _element_series(right)
    difference = _element_pair(left_elements, right_elements)
    per_body = {}
    for body, name in enumerate(BODY_NAMES[1:]):
        per_body[name] = {
            "semimajor_axis_relative_max": float(np.max(np.abs(difference["a_relative"][:, body]))),
            "eccentricity_absolute_max": float(np.max(np.abs(difference["e"][:, body]))),
            "eccentricity_vector_norm_max": float(np.max(np.linalg.norm(difference["evec"][:, body], axis=1))),
            "inclination_component_norm_max": float(np.max(np.linalg.norm(difference["inc_components"][:, body], axis=1))),
            "angular_momentum_direction_rad_max": float(np.max(difference["plane_angle"][:, body])),
            "apsidal_direction_rad_max": float(np.max(difference["peri_direction_angle"][:, body])),
        }
    reconstructions = {}
    for method in ("mean_anomaly", "mean_longitude"):
        if method == "mean_anomaly":
            left_phase = right_elements["M"]
            right_phase = right_elements["M"]
        else:
            left_phase = _wrap(right_elements["lambda"] - left_elements["Omega"] - left_elements["omega"])
            right_phase = _wrap(right_elements["lambda"] - right_elements["Omega"] - right_elements["omega"])
        left_position, left_velocity = _reconstruct(left, left_elements, left_phase)
        right_position, right_velocity = _reconstruct(right, right_elements, right_phase)
        scaled = _scaled_defect(left_position, left_velocity, right_position, right_velocity)
        reconstructions[method] = {
            "global_scaled_rms": float(np.sqrt(np.mean(scaled**2))),
            "per_body_scaled_rms": {name: float(np.sqrt(np.mean(scaled[:, body] ** 2))) for body, name in enumerate(BODY_NAMES)},
        }
    keys = {
        "semimajor_axis_relative_max": np.abs(difference["a_relative"]),
        "eccentricity_absolute_max": np.abs(difference["e"]),
        "eccentricity_vector_norm_max": np.linalg.norm(difference["evec"], axis=2),
        "inclination_component_norm_max": np.linalg.norm(difference["inc_components"], axis=2),
        "angular_momentum_direction_rad_max": difference["plane_angle"],
    }
    thresholds = {name: 1.0e-8 for name in keys}
    failures = {name: [] for name in keys}
    windows = []
    for window in range(10):
        selection = slice(window * 10, (window + 1) * 10)
        row = {"start_years_exclusive": window * 1000, "end_years_inclusive": (window + 1) * 1000}
        for key, values in keys.items():
            maximum = float(np.max(values[selection]))
            row[key] = maximum
            failures[key].append(maximum > thresholds[key])
        windows.append(row)
    persistent = []
    for key, flags in failures.items():
        for start in range(4, 8):
            if all(flags[start : start + 3]):
                persistent.append({"metric": key, "first_three_window_endpoint_years": (start + 3) * 1000})
                break
    return {"per_body": per_body, "phase_aligned": reconstructions, "rtn": _rtn(left, right), "windows": windows, "persistent_nonphase_failures": persistent}


def _perihelion(run: RunData) -> dict[str, float]:
    values = np.unwrap(_compute_elements(run)["mercury barycenter"]["varpi_rad"])
    times = run.times
    centered = times - float(np.mean(times))
    denominator = float(np.dot(centered, centered))
    slope = float(np.dot(centered, values - np.mean(values)) / denominator)
    intercept = float(np.mean(values) - slope * np.mean(times))
    residual = values - (intercept + slope * times)
    standard_error = math.sqrt(float(np.dot(residual, residual)) / (len(times) - 2) / denominator)
    scale = ARCSEC_PER_RAD * 100.0
    return {
        "rate_arcsec_per_century": slope * scale,
        "standard_error_arcsec_per_century": standard_error * scale,
        "confidence_95_half_width_arcsec_per_century": 1.984217 * standard_error * scale,
        "fit_residual_rms_arcsec": float(np.sqrt(np.mean(residual**2)) * ARCSEC_PER_RAD),
    }


def _frequencies(run: RunData) -> dict[str, Any]:
    elements = _element_series(run)
    times = run.times[1:]
    output = {}
    for body, name in enumerate(BODY_NAMES[1:]):
        eccentricity = elements["e"][:, body] * np.exp(1j * elements["varpi"][:, body])
        inclination = np.sin(0.5 * elements["i"][:, body]) * np.exp(1j * elements["Omega"][:, body])
        output[name] = {"eccentricity_mode": _naff_lite(times, eccentricity), "inclination_mode": _naff_lite(times, inclination)}
    return output


def _series(times: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    centered = times - float(np.mean(times))
    slope = float(np.dot(centered, values - np.mean(values)) / np.dot(centered, centered))
    absolute = np.abs(values)
    worst = int(np.argmax(absolute))
    windows = []
    for index in range(10):
        selection = (times > index * 1000.0) & (times <= (index + 1) * 1000.0)
        windows.append({"end_years": (index + 1) * 1000, "max_abs": float(np.max(absolute[selection])), "rms": float(np.sqrt(np.mean(values[selection] ** 2)))})
    return {
        "max_abs": float(absolute[worst]), "worst_epoch_years": float(times[worst]),
        "rms": float(np.sqrt(np.mean(values**2))), "p99_abs": float(np.quantile(absolute, 0.99, method="linear")),
        "endpoint": float(values[-1]), "fitted_trend_per_year": slope,
        "fitted_change_over_10k": slope * 10000.0, "windows": windows,
    }


def _conservation(run: RunData) -> dict[str, Any]:
    corrected = []
    angular = []
    for sample in range(len(run.times)):
        state = NBodyState(run.positions[sample], run.velocities[sample], run.masses)
        energy = float64_energy(state.masses, state.positions, state.velocities, gravitational_constant=G_SI, speed_of_light=299792458.0, coefficient_scale=1.0)
        corrected.append(energy["corrected"])
        angular.append(float(np.linalg.norm(total_angular_momentum_vector(state))))
    corrected_array = np.asarray(corrected)
    angular_array = np.asarray(angular)
    energy_drift = (corrected_array - corrected_array[0]) / abs(corrected_array[0])
    angular_drift = (angular_array - angular_array[0]) / abs(angular_array[0])
    return {
        "energy": _series(run.times, energy_drift), "angular_momentum": _series(run.times, angular_drift),
        "energy_history": energy_drift, "angular_history": angular_drift,
        "telemetry_energy_max_abs_difference": float(np.max(np.abs(energy_drift - run.progress["corrected_energy_rel_change"]))),
        "telemetry_angular_max_abs_difference": float(np.max(np.abs(angular_drift - run.progress["angular_momentum_rel_change"]))),
    }


def _tangent(new: RunData, old: RunData) -> dict[str, Any]:
    pair = _pair_tangent(new, old)
    new_vector = np.concatenate((new.variation_positions / AU_M, new.variation_velocities / VELOCITY_SCALE), axis=2).reshape(101, -1)
    old_vector = np.concatenate((old.variation_positions / AU_M, old.variation_velocities / VELOCITY_SCALE), axis=2).reshape(101, -1)
    new_norm = np.linalg.norm(new_vector, axis=1)
    old_norm = np.linalg.norm(old_vector, axis=1)
    log_difference = np.abs(np.log(new_norm) - np.log(old_norm))
    megno_difference = new.progress["megno"] - old.progress["megno"]
    accumulated_lcn_difference = (new.progress["lcn_1_per_year"] - old.progress["lcn_1_per_year"]) * 10000.0
    centered = new.times - np.mean(new.times)
    tangent_slope = float(np.dot(centered, np.log(new_norm) - np.mean(np.log(new_norm))) / np.dot(centered, centered))
    return {
        **pair, "tangent_log_norm_difference_max": float(np.max(log_difference)),
        "tangent_log_norm_difference_final": float(log_difference[-1]),
        "new_tangent_log_norm_fitted_growth_per_year": tangent_slope,
        "new_tangent_norm": new_norm, "old_tangent_norm": old_norm,
        "final_megno": float(new.progress["megno"][-1]), "final_megno_difference": float(abs(megno_difference[-1])),
        "megno_history_rms_difference": float(np.sqrt(np.mean(megno_difference**2))),
        "final_lcn_1_per_year": float(new.progress["lcn_1_per_year"][-1]),
        "final_accumulated_lcn_difference": float(abs(accumulated_lcn_difference[-1])),
        "lcn_history_accumulated_rms_difference": float(np.sqrt(np.mean(accumulated_lcn_difference**2))),
    }


def _raw_gate(detail: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    checks = {"global": detail["global_scaled_rms"] <= threshold["global_scaled_rms_max"]}
    for name in BODY_NAMES:
        checks[name] = detail["per_body"][name]["scaled_rms"] <= threshold["per_body_scaled_rms_max"][name]
    return {"passed": all(checks.values()), "checks": checks}


def _orbit_gate(detail: dict[str, Any], threshold: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for name, values in detail["per_body"].items():
        for metric in ("semimajor_axis_relative_max", "eccentricity_absolute_max", "eccentricity_vector_norm_max", "inclination_component_norm_max", "angular_momentum_direction_rad_max"):
            checks[f"{name}:{metric}"] = values[metric] <= threshold[metric]
    checks["persistent_nonphase_failures"] = len(detail["persistent_nonphase_failures"]) <= threshold["persistent_nonphase_failures_allowed"]
    return {"passed": all(checks.values()), "checks": checks}
