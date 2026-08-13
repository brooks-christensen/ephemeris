from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .m0_step3f1_analysis import METRIC_FIELDS, _artifact, _atomic_csv, _atomic_text, _finite, _historical_tangent, _ias15, _new_lane, _raw_detail
from .m0_step3f1_contract import BODY_NAMES, load_json, require, sha256_file, validate_manifest
from .m0_step3f1_metrics import _conservation, _frequencies, _orbit_gate, _perihelion, _phase_and_orbit, _raw_gate, _tangent
from .m0_step3f1_runner import audit
from .rebound_gr_tangent_backend_cli import atomic_write_json


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _native(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(child) for child in value]
    return value


def _without_arrays(value: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in names}


def _sync_raw_threshold(manifest: dict[str, Any]) -> dict[str, Any]:
    reference = manifest["reference_contract"]
    return {
        "global_scaled_rms_max": manifest["screen_thresholds"]["tangent_sync_control"]["physical_global_scaled_rms_max"],
        "per_body_scaled_rms_max": {
            name: max(
                10.0 * reference["ias15_per_body_scaled_rms_envelope"][name],
                0.1 * reference["historical_tangent_0p25_vs_ias15_per_body_scaled_rms_10k"][name],
            )
            for name in BODY_NAMES
        },
    }


def _conservation_gate(manifest: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    threshold = manifest["screen_thresholds"]["conservation"]
    p = values["P"]
    t = values["T"]
    energy_difference = np.abs(p["energy_history"] - t["energy_history"])
    angular_difference = np.abs(p["angular_history"] - t["angular_history"])
    checks = {}
    for lane, item in (("P", p), ("T", t)):
        energy = item["energy"]
        angular = item["angular_momentum"]
        trend_limit = max(0.25 * energy["max_abs"], 1.0e-10)
        checks[f"{lane}:energy_max"] = energy["max_abs"] <= threshold["corrected_energy_max_abs_per_lane"]
        checks[f"{lane}:energy_trend"] = abs(energy["fitted_change_over_10k"]) <= trend_limit
        checks[f"{lane}:angular_max"] = angular["max_abs"] <= threshold["angular_momentum_rel_drift_max_per_lane"]
        checks[f"{lane}:energy_telemetry_recomputed"] = item["telemetry_energy_max_abs_difference"] == 0.0
        checks[f"{lane}:angular_telemetry_recomputed"] = item["telemetry_angular_max_abs_difference"] == 0.0
    checks["pair_energy_history"] = float(np.max(energy_difference)) <= threshold["pair_corrected_energy_history_max_abs_difference"]
    checks["pair_angular_history"] = float(np.max(angular_difference)) <= threshold["pair_angular_history_max_abs_difference"]
    return {
        "passed": all(checks.values()), "checks": checks,
        "pair_energy_history_max_abs_difference": float(np.max(energy_difference)),
        "pair_energy_worst_epoch_years": float(np.argmax(energy_difference) * 100.0),
        "pair_angular_history_max_abs_difference": float(np.max(angular_difference)),
        "pair_angular_worst_epoch_years": float(np.argmax(angular_difference) * 100.0),
    }


def _tangent_gate(manifest: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    threshold = manifest["screen_thresholds"]["tangent_sync_control"]
    checks = {
        "final_direction_cosine": values["final_direction_cosine"] >= threshold["final_direction_cosine_min"],
        "direction_discrepancy_rms": values["direction_discrepancy_rms"] <= threshold["direction_discrepancy_rms_max"],
        "tangent_log_norm_difference": values["tangent_log_norm_difference_max"] <= threshold["tangent_log_norm_difference_max"],
        "final_megno_difference": values["final_megno_difference"] <= threshold["final_megno_difference_max"],
        "megno_history_rms_difference": values["megno_history_rms_difference"] <= threshold["megno_history_rms_difference_max"],
        "final_accumulated_lcn_difference": values["final_accumulated_lcn_difference"] <= threshold["final_accumulated_lcn_difference_max"],
        "lcn_history_rms_difference": values["lcn_history_accumulated_rms_difference"] <= threshold["lcn_history_rms_difference_max"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = payload["thresholds"]
    for comparison, detail in payload["physical"]["raw"].items():
        if comparison == "P_vs_IAS15":
            gate_threshold = thresholds["lane_p_vs_ias15"]
        elif comparison == "T_vs_old_T":
            gate_threshold = thresholds["tangent_sync_raw"]
        else:
            gate_threshold = thresholds["lane_t_carrier_vs_ias15_and_lane_p"]
        rows.append({"comparison": comparison, "category": "scaled_state", "metric": "global_scaled_rms", "body": "all", "value": detail["global_scaled_rms"], "units": "scaled", "threshold": gate_threshold["global_scaled_rms_max"], "passed": detail["global_scaled_rms"] <= gate_threshold["global_scaled_rms_max"], "worst_epoch_years": ""})
        for body in BODY_NAMES:
            value = detail["per_body"][body]["scaled_rms"]
            threshold = gate_threshold["per_body_scaled_rms_max"][body]
            rows.append({"comparison": comparison, "category": "scaled_state", "metric": "per_body_scaled_rms", "body": body, "value": value, "units": "scaled", "threshold": threshold, "passed": value <= threshold, "worst_epoch_years": detail["per_body"][body]["worst_position_epoch_years"]})
    orbit_threshold = thresholds["all_physical_pairs"]
    for comparison, detail in payload["physical"]["orbital"].items():
        for body, values in detail["per_body"].items():
            for metric in ("semimajor_axis_relative_max", "eccentricity_absolute_max", "eccentricity_vector_norm_max", "inclination_component_norm_max", "angular_momentum_direction_rad_max"):
                value = values[metric]
                threshold = orbit_threshold[metric]
                rows.append({"comparison": comparison, "category": "coordinate_free", "metric": metric, "body": body, "value": value, "units": "relative_or_rad", "threshold": threshold, "passed": value <= threshold, "worst_epoch_years": ""})
        difference = payload["physical"]["mercury_perihelion_pair_difference"][comparison]
        threshold = orbit_threshold["mercury_perihelion_rate_difference_arcsec_per_century_max"]
        rows.append({"comparison": comparison, "category": "secular", "metric": "mercury_perihelion_rate_difference", "body": "mercury barycenter", "value": difference, "units": "arcsec/century", "threshold": threshold, "passed": difference <= threshold, "worst_epoch_years": ""})
    conservation = payload["conservation"]
    for lane in ("P", "T"):
        for category, metric in (("energy", "max_abs"), ("energy", "fitted_change_over_10k"), ("angular_momentum", "max_abs")):
            rows.append({"comparison": lane, "category": "conservation", "metric": f"{category}_{metric}", "body": "all", "value": conservation[lane][category][metric], "units": "relative", "threshold": "see_manifest20", "passed": payload["gates"]["conservation"]["checks"].get(f"{lane}:energy_max" if category == "energy" and metric == "max_abs" else f"{lane}:energy_trend" if category == "energy" else f"{lane}:angular_max"), "worst_epoch_years": conservation[lane][category].get("worst_epoch_years", "")})
    tangent = payload["tangent"]
    tangent_threshold = thresholds["tangent_sync_control"]
    for metric, threshold_key in (("final_direction_cosine", "final_direction_cosine_min"), ("direction_discrepancy_rms", "direction_discrepancy_rms_max"), ("tangent_log_norm_difference_max", "tangent_log_norm_difference_max"), ("final_megno_difference", "final_megno_difference_max"), ("megno_history_rms_difference", "megno_history_rms_difference_max"), ("final_accumulated_lcn_difference", "final_accumulated_lcn_difference_max"), ("lcn_history_accumulated_rms_difference", "lcn_history_rms_difference_max")):
        rows.append({"comparison": "T_vs_old_T", "category": "tangent", "metric": metric, "body": "all", "value": tangent[metric], "units": "dimensionless", "threshold": tangent_threshold[threshold_key], "passed": payload["gates"]["tangent"]["checks"]["lcn_history_rms_difference" if metric == "lcn_history_accumulated_rms_difference" else metric.replace("_max", "") if metric == "tangent_log_norm_difference_max" else metric], "worst_epoch_years": tangent.get("minimum_direction_cosine_epoch_years", "")})
    return rows


def _configuration_rows(manifest: dict[str, Any], runs: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("lane", "run_id", "responsibility", "integrator", "coordinates", "kernel", "corrector", "corrector2", "safe_mode", "keep_unsynchronized", "variations", "megno", "megno_seed", "step_days", "duration_years", "total_steps", "samples", "state_rows", "archives", "callbacks", "runtime_seconds", "configuration_fingerprint")
    rows = []
    for key in ("P", "T"):
        lane = manifest["lane_contracts"][key]
        summary = runs[key].summary
        rows.append(dict(zip(fields, (
            key, lane["id"], lane["responsibility"], "whfast", "jacobi", lane["kernel"], lane["corrector"], lane["corrector2"], lane["safe_mode"], lane["keep_unsynchronized"], lane["variations"], lane["megno"], lane["megno_seed"], 0.25, 10000, 14610000, 101, 1010, 11, lane["expected_callback_invocations"], summary["runtime_seconds"], lane["configuration_fingerprint"],
        ))))
    return rows


def _figures(manifest: dict[str, Any], runs: dict[str, Any], raw_arrays: dict[str, Any], conservation_arrays: dict[str, Any], tangent_arrays: dict[str, Any], orbital: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(manifest["paths"]["figure_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"Software": "mini_ephemeris Step 3f1"}
    colors = {"P": "#0072B2", "T": "#D55E00", "reference": "#009E73", "neutral": "#333333"}

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.semilogy(runs["P"].times, np.maximum(raw_arrays["P_vs_IAS15"], 1e-18), label="Lane P vs IAS15", color=colors["P"])
    ax.semilogy(runs["T"].times, np.maximum(raw_arrays["T_vs_IAS15"], 1e-18), label="Lane T vs IAS15", color=colors["T"])
    ax.set(xlabel="Time (years)", ylabel="Scaled state RMS", title="Physical carrier defects")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.savefig(directory / "physical_defects_vs_ias15.png", dpi=150, metadata=metadata); plt.close(fig)

    names = [name.split()[0].title() for name in BODY_NAMES[1:]]
    x = np.arange(9)
    p_values = [orbital["P_vs_IAS15"]["per_body"][name]["eccentricity_vector_norm_max"] for name in BODY_NAMES[1:]]
    t_values = [orbital["T_vs_P"]["per_body"][name]["eccentricity_vector_norm_max"] for name in BODY_NAMES[1:]]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.semilogy(x - 0.18, np.maximum(p_values, 1e-18), "o", label="P vs IAS15 e-vector", color=colors["P"])
    ax.semilogy(x + 0.18, np.maximum(t_values, 1e-18), "s", label="T vs P e-vector", color=colors["T"])
    ax.set_xticks(x, names, rotation=35, ha="right"); ax.set(ylabel="Maximum norm", title="Coordinate-free discrepancies")
    ax.grid(True, axis="y", alpha=0.25); ax.legend()
    fig.savefig(directory / "coordinate_free_phase_stripped_by_body.png", dpi=150, metadata=metadata); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for key in ("P", "T"):
        axes[0].plot(runs[key].times, conservation_arrays[key]["energy_history"], label=key, color=colors[key])
        axes[1].plot(runs[key].times, conservation_arrays[key]["angular_history"], label=key, color=colors[key])
    axes[0].set(ylabel="Corrected energy relative drift", title="Conservation histories"); axes[1].set(xlabel="Time (years)", ylabel="Angular momentum relative drift")
    for ax in axes: ax.grid(True, alpha=0.25); ax.legend()
    fig.savefig(directory / "energy_angular_momentum_histories.png", dpi=150, metadata=metadata); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(runs["T"].times, runs["T"].progress["megno"], color=colors["T"]); axes[0].plot(runs["T"].times, runs["old_T"].progress["megno"], color=colors["neutral"], alpha=0.7)
    axes[1].plot(runs["T"].times, runs["T"].progress["lcn_1_per_year"], color=colors["T"]); axes[1].plot(runs["T"].times, runs["old_T"].progress["lcn_1_per_year"], color=colors["neutral"], alpha=0.7)
    axes[2].plot(runs["T"].times, np.log(tangent_arrays["new_tangent_norm"]), color=colors["T"]); axes[2].plot(runs["T"].times, np.log(tangent_arrays["old_tangent_norm"]), color=colors["neutral"], alpha=0.7)
    axes[0].set(ylabel="MEGNO", title="Tangent and chaos continuity"); axes[1].set(ylabel="LCN (1/year)"); axes[2].set(xlabel="Time (years)", ylabel="log scaled tangent norm")
    for ax in axes: ax.grid(True, alpha=0.25)
    fig.savefig(directory / "tangent_megno_lcn_histories.png", dpi=150, metadata=metadata); plt.close(fig)

    output = []
    for name in manifest["figures"]:
        path = directory / name
        pixels = plt.imread(path)
        require(pixels.size > 0 and float(np.std(pixels)) > 0.01, f"Blank Step 3f1 figure: {name}")
        output.append(_artifact(path))
    return output
