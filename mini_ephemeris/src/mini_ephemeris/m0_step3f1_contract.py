from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "20_m0_step3f1_two_lane_architecture_screen_v1.json"
)
DAY_S = 86400.0
JULIAN_YEAR_S = 31557600.0
AU_M = 149597870700.0
VELOCITY_SCALE = AU_M / JULIAN_YEAR_S

BODY_NAMES = (
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

PROGRESS_FIELDS = [
    "schema_version",
    "configuration_fingerprint",
    "lane_id",
    "artifact_identity",
    "sample_index",
    "step_count",
    "target_step_count",
    "time_seconds",
    "time_years",
    "dt_seconds",
    "dt_last_done_seconds",
    "steps_done",
    "integrator",
    "coordinates",
    "kernel",
    "corrector",
    "corrector2",
    "safe_mode",
    "keep_unsynchronized",
    "recalculate_coordinates_this_timestep",
    "live_is_synchronized_before_sample",
    "live_is_synchronized_after_sample",
    "diagnostic_copy_used",
    "n_real",
    "n_var",
    "n_var_config",
    "megno",
    "lcn_1_per_year",
    "tangent_scaled_norm",
    "tangent_log_scaled_norm",
    "newtonian_energy_j",
    "gr_potential_energy_j",
    "corrected_energy_j",
    "corrected_energy_rel_change",
    "angular_momentum_x_kg_m2_s",
    "angular_momentum_y_kg_m2_s",
    "angular_momentum_z_kg_m2_s",
    "angular_momentum_norm_kg_m2_s",
    "angular_momentum_rel_change",
    "callback_invocations",
    "nonfinite_result_count",
]

STATE_FIELDS = [
    "schema_version",
    "configuration_fingerprint",
    "lane_id",
    "artifact_identity",
    "sample_index",
    "step_count",
    "time_seconds",
    "time_years",
    "body_index",
    "body_name",
    "mass_kg",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_per_s",
    "vy_m_per_s",
    "vz_m_per_s",
    "variation_config_index",
    "variation_x_m",
    "variation_y_m",
    "variation_z_m",
    "variation_vx_m_per_s",
    "variation_vy_m_per_s",
    "variation_vz_m_per_s",
]


class Step3f1Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Step3f1Error(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise Step3f1Error(f"Unreadable {label} {path}: {exc}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def lane_payload(manifest: dict[str, Any], lane_key: str) -> dict[str, Any]:
    common = manifest["common_physical_contract"]
    lane = manifest["lane_contracts"][lane_key]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "telemetry_schema_version": manifest["telemetry_contract"]["schema_version"],
        "model_id": manifest["model_id"],
        "model_scope": common["model_scope"],
        "body_names": common["body_names"],
        "start_date": common["start_date"],
        "initial_conditions_sha256": common["initial_conditions_sha256"],
        "kernel_sha256": common["kernel_sha256"],
        "c_source_sha256": manifest["runtime_identity"]["callback_source_sha256"],
        "c_artifact_sha256": manifest["runtime_identity"]["callback_library_sha256"],
        "integrator": common["integrator"],
        "coordinates": common["coordinates"],
        "step_days": common["step_days"],
        "step_seconds": common["step_seconds"],
        "duration_years": common["duration_years"],
        "total_steps": common["total_steps"],
        "scientific_cadence_years": common["scientific_cadence_years"],
        "steps_per_sample": common["steps_per_scientific_sample"],
        "archive_cadence_years": common["archive_cadence_years"],
        "steps_per_archive": common["steps_per_archive"],
        "exact_finish_time": common["exact_finish_time"],
        "integer_targets": not common["fractional_endpoint_steps"],
        "corrector2": lane["corrector2"],
        "safe_mode": lane["safe_mode"],
        "keep_unsynchronized": lane["keep_unsynchronized"],
        "recalculate_coordinates_this_timestep": lane[
            "recalculate_coordinates_this_timestep"
        ],
        "gr_scale": common["gr_scale"],
        "include_central_response": common["include_central_response"],
        "force_is_velocity_dependent": common["force_is_velocity_dependent"],
        "diagnostics_source": "nonmutating synchronized particle representation on a simulation copy",
        "random_perturbation": common["random_perturbation"],
        "lane_id": lane["id"],
        "responsibility": lane["responsibility"].replace(
            " for architecture screening only", "_screen"
        ).replace("native first-variation, MEGNO, and finite-time LCN diagnostics", "tangent_megno_diagnostic_screen").replace(
            "candidate canonical physical trajectory", "canonical_physical_candidate"
        ).replace(" ", "_"),
        "kernel": lane["kernel"],
        "corrector": lane["corrector"],
        "variations": lane["variations"],
        "megno": lane["megno"],
        "lcn": lane["lcn"],
        "megno_seed": lane["megno_seed"],
        "callback_mode": lane["callback_mode"],
    }
    if lane_key == "T":
        payload["variation_order"] = lane["variation_order"]
        payload["variation_testparticle"] = lane["variation_testparticle"]
    return payload


def validate_manifest(manifest: dict[str, Any]) -> None:
    statuses = manifest["allowed_final_statuses"]
    findings = manifest["allowed_primary_findings"]
    require(
        statuses
        == [
            "STEP3F1_TWO_LANE_SCREEN_PASSED",
            "STEP3F1_TWO_LANE_SCREEN_FAILED",
            "STEP3F1_TWO_LANE_SCREEN_INCONCLUSIVE",
            "BLOCKED",
        ],
        "Manifest 20 final statuses changed.",
    )
    require(
        findings
        == [
            "TWO_LANE_ARCHITECTURE_SUPPORTED",
            "PHYSICAL_WHCKL_LANE_UNQUALIFIED",
            "TANGENT_LANE_UNQUALIFIED",
            "BOTH_LANES_UNQUALIFIED",
            "MIXED_OR_INCONCLUSIVE",
            "NOT_EVALUATED",
        ],
        "Manifest 20 primary findings changed.",
    )
    common = manifest["common_physical_contract"]
    require(common["duration_years"] == 10000, "Duration guard changed.")
    require(common["step_days"] == 0.25, "Timestep guard changed.")
    require(common["total_steps"] == 14610000, "Step count changed.")
    require(common["steps_per_scientific_sample"] == 146100, "Sample step count changed.")
    require(common["steps_per_archive"] == 1461000, "Archive step count changed.")
    require(common["scientific_samples"] == 101, "Sample count changed.")
    require(common["state_rows"] == 1010, "State row count changed.")
    require(set(manifest["lane_contracts"]) == {"P", "T"}, "Authorized lanes changed.")
    for lane_key in ("P", "T"):
        lane = manifest["lane_contracts"][lane_key]
        require(
            canonical_hash(lane_payload(manifest, lane_key))
            == lane["configuration_fingerprint"],
            f"Lane {lane_key} configuration fingerprint mismatch.",
        )
    require(manifest["lane_contracts"]["P"]["kernel"] == "lazy", "Lane P kernel changed.")
    require(manifest["lane_contracts"]["P"]["corrector"] == 17, "Lane P corrector changed.")
    require(manifest["lane_contracts"]["P"]["variations"] is False, "Lane P variations changed.")
    require(manifest["lane_contracts"]["T"]["kernel"] == "default", "Lane T kernel changed.")
    require(manifest["lane_contracts"]["T"]["corrector"] == 0, "Lane T corrector changed.")
    require(manifest["lane_contracts"]["T"]["variations"] is True, "Lane T variations changed.")
    for lane in manifest["lane_contracts"].values():
        require(lane["safe_mode"] == 0, "safe_mode must remain zero.")
        require(lane["keep_unsynchronized"] == 1, "keep_unsynchronized must remain one.")


def lane_paths(manifest: dict[str, Any], lane_key: str) -> dict[str, Path]:
    lane_id = manifest["lane_contracts"][lane_key]["id"]
    directory = Path(manifest["paths"]["output_root"]) / lane_id
    return {
        "directory": directory,
        "progress": directory / "progress.csv",
        "progress_partial": directory / "progress.csv.partial",
        "state": directory / "state.csv",
        "state_partial": directory / "state.csv.partial",
        "archive": directory / "simulationarchive.bin",
        "status": directory / "status.json",
        "summary": directory / "summary.json",
        "events": directory / "events.log",
        "restart_check": directory / "restart_check.json",
    }


def artifact_identity(manifest: dict[str, Any], lane_key: str) -> str:
    return f"{manifest['experiment_id']}:{manifest['lane_contracts'][lane_key]['id']}:v1"
