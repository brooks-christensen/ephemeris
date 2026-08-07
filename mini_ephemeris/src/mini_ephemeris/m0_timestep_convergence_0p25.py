from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import shutil
from typing import Any

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import load_c_backend
from .long_term_stability_cli import parse_start_datetime, stability_body_list
from .m0_timestep_convergence import (
    INNER_BODIES,
    ConvergenceError,
    RunData,
    _atomic_write_json,
    _atomic_write_text,
    _git,
    _load_json,
    _load_run,
    _manifest_configuration,
    _pair_angular,
    _pair_elements,
    _pair_lcn,
    _pair_physical,
    _pair_scalar,
    _pair_tangent,
    _perihelion_rate,
    _require,
    _series_metrics,
)
from .rebound_gr_tangent_backend_cli import (
    _config_payload,
    _record_targets,
    canonical_hash,
    initial_condition_hash,
    output_paths,
    sha256_file,
)


DEFAULT_MANIFEST = Path(
    "ephemeris_experiment_runner/manifests/11_m0_timestep_convergence_0p25_v1.json"
)
FINAL_STATUSES = {
    "M0_0P5DAY_CONVERGED",
    "M0_0P5DAY_NOT_CONVERGED",
    "BLOCKED",
}
ORDERED_RUN_IDS = (
    "m0_conv_1d_1myr_s12345",
    "m0_conv_0p5d_1myr_s12345",
    "m0_conv_0p25d_1myr_s12345",
)


def _step3_context(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    step3_manifest_path = Path(manifest["paths"]["step3_manifest"])
    step3_summary_path = Path(manifest["paths"]["step3_summary"])
    _require(
        sha256_file(step3_manifest_path) == manifest["provenance"]["step3_manifest_sha256"],
        "Step 3 manifest hash changed.",
    )
    _require(
        sha256_file(step3_summary_path) == manifest["provenance"]["step3_summary_sha256"],
        "Step 3 summary hash changed.",
    )
    step3_manifest = _load_json(step3_manifest_path, "Step 3 manifest")
    step3_summary = _load_json(step3_summary_path, "Step 3 summary")
    _require(
        step3_summary.get("final_status") == manifest["provenance"]["step3_status"],
        "Step 3 status changed.",
    )
    _require(
        manifest["thresholds"] == step3_manifest["thresholds"],
        "Step 3b thresholds differ from manifest 10.",
    )
    _require(
        manifest["comparison_definitions"] == step3_manifest["comparison_definitions"],
        "Step 3b comparison definitions differ from manifest 10.",
    )
    _require(
        manifest["shared_configuration"] == step3_manifest["shared_configuration"],
        "Step 3b shared configuration differs from manifest 10.",
    )
    _require(
        manifest["endpoint_semantics"] == step3_manifest["endpoint_semantics"],
        "Step 3b endpoint semantics differ from manifest 10.",
    )
    return step3_manifest_path, step3_manifest, step3_summary


def _verify_protected(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["paths"]["project_root"])
    verified = []
    for relative, expected in manifest["protected_files"].items():
        path = root / relative
        actual = sha256_file(path)
        _require(actual == expected, f"Protected file hash mismatch: {relative}")
        verified.append(
            {"path": str(path), "expected_sha256": expected, "actual_sha256": actual}
        )
    return verified


def _verify_step3_inventory(step3_summary: dict[str, Any]) -> list[dict[str, Any]]:
    inventory = step3_summary.get("artifact_inventory", [])
    _require(len(inventory) == 21, "Step 3 artifact inventory count changed.")
    seen: set[str] = set()
    for item in inventory:
        path = Path(item["path"])
        _require(str(path) not in seen, f"Duplicate Step 3 inventory path: {path}")
        seen.add(str(path))
        _require(path.is_file(), f"Missing Step 3 artifact: {path}")
        _require(path.stat().st_size == item["size_bytes"], f"Step 3 size mismatch: {path}")
        _require(sha256_file(path) == item["sha256"], f"Step 3 hash mismatch: {path}")
    return inventory


def _existing_run(
    step3_manifest_path: Path,
    step3_manifest: dict[str, Any],
    run_id: str,
) -> RunData:
    run = next(item for item in step3_manifest["runs"] if item["id"] == run_id)
    return _load_run(step3_manifest_path, step3_manifest, run)


def _expected_new_artifacts(manifest: dict[str, Any]) -> list[dict[str, str]]:
    run = manifest["decisive_run"]
    paths = output_paths(Path(run["output_dir"]), run["id"], None)
    actual = [
        {"role": role, "path": str(paths[role])}
        for role in ("progress", "state", "status", "summary", "restart", "archive")
    ]
    actual.append({"role": "log", "path": run["log_path"]})
    expected = manifest["artifact_inventory_before_launch"]["expected_new_artifacts"]
    _require(actual == expected, "Expected Step 3b artifact paths do not match the runner.")
    return actual


def _preflight_payload(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(manifest["paths"]["project_root"])
    _require(Path.cwd().resolve() == root.resolve(), "Preflight must run at project root.")
    _require(manifest.get("frozen_before_decisive_run") is True, "Manifest is not frozen.")
    _require(
        _git("rev-parse", "HEAD") == manifest["provenance"]["closed_step3_commit"],
        "Step 3b must launch from the closed Step 3 commit.",
    )
    tag = manifest["provenance"]["validated_c_annotated_tag"]
    _require(_git("cat-file", "-t", tag) == "tag", "Compiled-C milestone is not annotated.")
    _require(
        _git("rev-parse", tag + "^{commit}")
        == manifest["provenance"]["validated_c_baseline_commit"],
        "Compiled-C milestone resolves to the wrong commit.",
    )
    protected = _verify_protected(manifest)
    step3_manifest_path, step3_manifest, step3_summary = _step3_context(manifest)
    existing_inventory = _verify_step3_inventory(step3_summary)

    endpoint = manifest["endpoint_semantics"]
    targets = _record_targets(
        float(endpoint["duration_years"]),
        float(endpoint["scientific_cadence_years"]),
    )
    _require(
        len(targets) == endpoint["expected_scientific_samples_per_run"],
        "Scientific sample-count expectation is incompatible with the runner.",
    )
    _require(targets[0] == 0.0 and targets[-1] == 1_000_000.0, "Endpoint mismatch.")
    _require(
        len(targets) * endpoint["real_body_count"]
        == endpoint["expected_state_rows_per_run"],
        "State-row expectation is incompatible with the runner.",
    )

    existing_runs = {
        run_id: _existing_run(step3_manifest_path, step3_manifest, run_id).integrity
        for run_id in ORDERED_RUN_IDS[:2]
    }
    new_artifacts = _expected_new_artifacts(manifest)
    run = manifest["decisive_run"]
    _require(not Path(run["output_dir"]).exists(), "Decisive output directory already exists.")
    for item in new_artifacts:
        _require(not Path(item["path"]).exists(), f"Decisive artifact already exists: {item['path']}")

    shared = manifest["shared_configuration"]
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    start_date = parse_start_datetime(shared["start_date"])
    kernel = Path(manifest["paths"]["kernel"])
    state0 = initial_state_solar_system_barycentric(
        start_date,
        bodies=bodies,
        config=EphemerisConfig(kernel_path=str(kernel)),
    )
    backend = load_c_backend()
    actual_hashes = {
        "kernel_sha256": sha256_file(kernel),
        "initial_conditions_sha256": initial_condition_hash(state0, bodies),
        "artifact_sha256": backend.build_metadata["artifact_sha256"],
        "c_source_sha256": backend.build_metadata["source_sha256"],
    }
    _require(actual_hashes["kernel_sha256"] == shared["kernel_sha256"], "Kernel hash mismatch.")
    _require(
        actual_hashes["initial_conditions_sha256"] == shared["initial_conditions_sha256"],
        "Initial-condition hash mismatch.",
    )
    _require(
        actual_hashes["artifact_sha256"] == shared["c_artifact_sha256"],
        "Compiled-C artifact hash mismatch.",
    )
    _require(
        actual_hashes["c_source_sha256"] == shared["c_source_sha256"],
        "Compiled-C source hash mismatch.",
    )
    runner_args = argparse.Namespace(
        model_id=manifest["model_id"],
        gr_tangent_backend=shared["backend"],
        start_date=start_date,
        model_scope=shared["model_scope"],
        duration_years=float(shared["duration_years"]),
        step_days=float(run["step_days"]),
        record_every_years=float(shared["record_every_years"]),
        archive_interval_years=float(shared["archive_interval_years"]),
        megno_seed=int(shared["megno_seed"]),
        gr_scale=float(shared["gr_scale"]),
        no_central_response=not bool(shared["include_central_response"]),
    )
    actual_configuration = _config_payload(runner_args, list(bodies), actual_hashes)
    _require(
        actual_configuration == _manifest_configuration(manifest, run),
        "Runner configuration does not match the preregistration.",
    )
    fingerprint = canonical_hash(actual_configuration)
    _require(
        fingerprint == run["configuration_fingerprint"],
        "Decisive configuration fingerprint mismatch.",
    )
    disk = shutil.disk_usage(root)
    _require(
        disk.free >= manifest["estimates"]["required_with_safety_factor_bytes"],
        "Insufficient disk space for the decisive run.",
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protected_files": protected,
        "step3_artifact_inventory": existing_inventory,
        "existing_run_integrity": existing_runs,
        "expected_new_artifacts": new_artifacts,
        "configuration_fingerprint": fingerprint,
        "actual_hashes": actual_hashes,
        "endpoint_semantics": {
            "scientific_samples": len(targets),
            "state_rows": len(targets) * len(bodies),
            "first_time_years": targets[0],
            "last_time_years": targets[-1],
        },
        "estimates": manifest["estimates"],
        "available_disk_bytes_at_preflight": disk.free,
    }


def preflight(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Step 3b manifest")
    payload = _preflight_payload(manifest_path, manifest)
    output = (
        Path(manifest["paths"]["output_root"])
        / "preflight/m0_timestep_convergence_0p25_preflight.json"
    )
    _atomic_write_json(output, payload)
    print(f"[m0-convergence-0p25] preflight PASS: {output}")
    return 0


def evaluate_candidate_criteria(
    manifest: dict[str, Any],
    runs: dict[str, RunData],
    coarse: dict[str, Any],
    fine: dict[str, Any],
    perihelion: dict[str, dict[str, float]],
    energy: dict[str, dict[str, float]],
    angular: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    thresholds = manifest["thresholds"]
    coarse_physical = coarse["physical"]
    fine_physical = fine["physical"]
    global_ratio = fine_physical["global_scaled_rms"] / max(
        coarse_physical["global_scaled_rms"], 1.0e-300
    )
    physical_inner = {
        body: fine_physical["per_body"][body]["rms"]
        < coarse_physical["per_body"][body]["rms"]
        for body in INNER_BODIES
    }
    physical_pass = (
        fine_physical["global_scaled_rms"] < coarse_physical["global_scaled_rms"]
        and all(physical_inner.values())
        and global_ratio
        <= thresholds["global_scaled_physical_rms_fine_over_coarse_max"]
    )

    fine_elements = fine["orbital_elements"]["per_body"]
    eccentricity_limits = {
        body: (
            thresholds["mercury_eccentricity_history_max_abs"]
            if body == "mercury barycenter"
            else thresholds["inner_planet_eccentricity_history_max_abs"]
        )
        for body in INNER_BODIES
    }
    eccentricity_checks = {
        body: fine_elements[body]["eccentricity_max_abs_difference"] <= limit
        for body, limit in eccentricity_limits.items()
    }
    semimajor_checks = {
        body: fine_elements[body]["semimajor_axis_max_relative_difference"]
        <= thresholds["inner_planet_semimajor_axis_history_max_relative"]
        for body in INNER_BODIES
    }
    peri_difference = abs(
        perihelion[ORDERED_RUN_IDS[1]]["mean_rate_arcsec_per_century"]
        - perihelion[ORDERED_RUN_IDS[2]]["mean_rate_arcsec_per_century"]
    )
    tangent_pass = (
        fine["tangent"]["final_direction_cosine"]
        >= thresholds["final_tangent_direction_cosine_min"]
        and fine["tangent"]["direction_discrepancy_rms"]
        < coarse["tangent"]["direction_discrepancy_rms"]
    )
    megno_pass = (
        fine["megno"]["final_abs_difference"] <= thresholds["final_megno_difference_max"]
        and fine["megno"]["history_rms_difference"]
        <= thresholds["megno_history_rms_difference_max"]
    )
    lcn_pass = (
        fine["lcn"]["final_accumulated_abs_difference"]
        <= thresholds["final_lcn_accumulated_difference_max"]
    )

    floor = thresholds["corrected_energy_roundoff_floor"]
    all_maxima_at_floor = all(energy[name]["max_abs"] <= floor for name in ORDERED_RUN_IDS)
    energy_reductions = {
        metric: all(
            energy[right][metric] <= energy[left][metric]
            for left, right in zip(ORDERED_RUN_IDS, ORDERED_RUN_IDS[1:])
        )
        for metric in thresholds["corrected_energy_metrics_nonincreasing_with_step_reduction"]
    }
    if all_maxima_at_floor:
        energy_reductions["max_abs"] = True
    energy_bounds = {
        name: energy[name]["max_abs"] <= thresholds["corrected_energy_max_abs_per_run"]
        for name in ORDERED_RUN_IDS
    }
    energy_trends = {
        name: abs(energy[name]["fitted_change_over_1myr"])
        <= max(0.25 * energy[name]["max_abs"], 1.0e-10)
        for name in ORDERED_RUN_IDS
    }
    energy_pass = (
        all(energy_bounds.values())
        and all(energy_trends.values())
        and all(energy_reductions.values())
    )
    angular_checks = {
        name: angular[name]["max_abs"]
        <= thresholds["angular_momentum_rel_drift_max_per_run"]
        for name in ORDERED_RUN_IDS
    }
    return {
        "physical_state": {
            "passed": physical_pass,
            "fine_over_coarse_global_rms_ratio": global_ratio,
            "inner_planet_fine_strictly_smaller": physical_inner,
        },
        "mercury_perihelion_rate": {
            "passed": peri_difference
            <= thresholds["mercury_perihelion_rate_difference_arcsec_per_century_max"],
            "fine_pair_abs_difference_arcsec_per_century": peri_difference,
        },
        "eccentricity_history": {
            "passed": all(eccentricity_checks.values()),
            "per_body": eccentricity_checks,
            "limits": eccentricity_limits,
        },
        "semimajor_axis_history": {
            "passed": all(semimajor_checks.values()),
            "per_body": semimajor_checks,
        },
        "tangent": {"passed": tangent_pass},
        "megno": {"passed": megno_pass},
        "lcn": {"passed": lcn_pass},
        "corrected_energy": {
            "passed": energy_pass,
            "per_run_bounded": energy_bounds,
            "per_run_trend_bounded": energy_trends,
            "metrics_nonincreasing": energy_reductions,
            "all_maxima_at_roundoff_floor": all_maxima_at_floor,
        },
        "angular_momentum": {
            "passed": all(angular_checks.values()),
            "per_run": angular_checks,
        },
        "run_integrity": {
            "passed": all(run.integrity["passed"] for run in runs.values())
        },
    }


def _energy_reduction_evidence(
    manifest: dict[str, Any], energy: dict[str, dict[str, float]]
) -> dict[str, Any]:
    output = {}
    for metric in manifest["thresholds"][
        "corrected_energy_metrics_nonincreasing_with_step_reduction"
    ]:
        transitions = []
        for left, right in zip(ORDERED_RUN_IDS, ORDERED_RUN_IDS[1:]):
            transitions.append(
                {
                    "coarse_run": left,
                    "fine_run": right,
                    "coarse_value": energy[left][metric],
                    "fine_value": energy[right][metric],
                    "fine_over_coarse_ratio": energy[right][metric]
                    / max(energy[left][metric], 1.0e-300),
                    "passed": energy[right][metric] <= energy[left][metric],
                    "worst_epoch_years": (
                        energy[right]["max_abs_worst_epoch_years"]
                        if metric == "max_abs"
                        else None
                    ),
                }
            )
        output[metric] = {
            "passed": all(item["passed"] for item in transitions),
            "transitions": transitions,
            "worst_transition": max(
                transitions, key=lambda item: item["fine_over_coarse_ratio"]
            ),
        }
    return output


def _threshold_evidence(
    manifest: dict[str, Any],
    runs: dict[str, RunData],
    coarse: dict[str, Any],
    fine: dict[str, Any],
    perihelion: dict[str, dict[str, float]],
    energy: dict[str, dict[str, float]],
    angular: dict[str, dict[str, float]],
) -> dict[str, Any]:
    thresholds = manifest["thresholds"]
    fine_elements = fine["orbital_elements"]["per_body"]
    physical_ratios = {
        body: fine["physical"]["per_body"][body]["rms"]
        / max(coarse["physical"]["per_body"][body]["rms"], 1.0e-300)
        for body in INNER_BODIES
    }
    physical_body = max(physical_ratios, key=physical_ratios.get)
    eccentricity_limits = {
        body: (
            thresholds["mercury_eccentricity_history_max_abs"]
            if body == "mercury barycenter"
            else thresholds["inner_planet_eccentricity_history_max_abs"]
        )
        for body in INNER_BODIES
    }
    eccentricity_body = max(
        INNER_BODIES,
        key=lambda body: fine_elements[body]["eccentricity_max_abs_difference"]
        / eccentricity_limits[body],
    )
    semimajor_body = max(
        INNER_BODIES,
        key=lambda body: fine_elements[body]["semimajor_axis_max_relative_difference"],
    )
    peri_difference = abs(
        perihelion[ORDERED_RUN_IDS[1]]["mean_rate_arcsec_per_century"]
        - perihelion[ORDERED_RUN_IDS[2]]["mean_rate_arcsec_per_century"]
    )
    energy_run = max(ORDERED_RUN_IDS, key=lambda name: energy[name]["max_abs"])
    angular_run = max(ORDERED_RUN_IDS, key=lambda name: angular[name]["max_abs"])
    energy_reductions = _energy_reduction_evidence(manifest, energy)
    return {
        "physical_state": {
            "global_fine_over_coarse_ratio": fine["physical"]["global_scaled_rms"]
            / max(coarse["physical"]["global_scaled_rms"], 1.0e-300),
            "limit": thresholds["global_scaled_physical_rms_fine_over_coarse_max"],
            "worst_inner_body": physical_body,
            "worst_inner_fine_over_coarse_ratio": physical_ratios[physical_body],
            "worst_epoch_years": fine["physical"]["per_body"][physical_body][
                "worst_epoch_years"
            ],
            "per_inner_body": {
                body: {
                    "coarse_rms": coarse["physical"]["per_body"][body]["rms"],
                    "fine_rms": fine["physical"]["per_body"][body]["rms"],
                    "fine_over_coarse_ratio": physical_ratios[body],
                    "fine_worst_epoch_years": fine["physical"]["per_body"][body][
                        "worst_epoch_years"
                    ],
                }
                for body in INNER_BODIES
            },
        },
        "mercury_perihelion_rate": {
            "body": "mercury barycenter",
            "value_arcsec_per_century": peri_difference,
            "limit_arcsec_per_century": thresholds[
                "mercury_perihelion_rate_difference_arcsec_per_century_max"
            ],
            "fit_interval_years": [0.0, 1_000_000.0],
            "worst_epoch_years": None,
        },
        "eccentricity_history": {
            "worst_body": eccentricity_body,
            "value": fine_elements[eccentricity_body]["eccentricity_max_abs_difference"],
            "limit": eccentricity_limits[eccentricity_body],
            "worst_epoch_years": fine_elements[eccentricity_body][
                "eccentricity_worst_epoch_years"
            ],
            "mercury": {
                "value": fine_elements["mercury barycenter"][
                    "eccentricity_max_abs_difference"
                ],
                "limit": thresholds["mercury_eccentricity_history_max_abs"],
                "worst_epoch_years": fine_elements["mercury barycenter"][
                    "eccentricity_worst_epoch_years"
                ],
            },
            "per_inner_body": {
                body: {
                    "value": fine_elements[body]["eccentricity_max_abs_difference"],
                    "limit": eccentricity_limits[body],
                    "worst_epoch_years": fine_elements[body][
                        "eccentricity_worst_epoch_years"
                    ],
                }
                for body in INNER_BODIES
            },
        },
        "semimajor_axis_history": {
            "worst_body": semimajor_body,
            "value": fine_elements[semimajor_body][
                "semimajor_axis_max_relative_difference"
            ],
            "limit": thresholds["inner_planet_semimajor_axis_history_max_relative"],
            "worst_epoch_years": fine_elements[semimajor_body][
                "semimajor_axis_worst_epoch_years"
            ],
            "per_inner_body": {
                body: {
                    "value": fine_elements[body][
                        "semimajor_axis_max_relative_difference"
                    ],
                    "limit": thresholds[
                        "inner_planet_semimajor_axis_history_max_relative"
                    ],
                    "worst_epoch_years": fine_elements[body][
                        "semimajor_axis_worst_epoch_years"
                    ],
                }
                for body in INNER_BODIES
            },
        },
        "tangent": {
            "body": "full-system first variation",
            "final_direction_cosine": fine["tangent"]["final_direction_cosine"],
            "minimum": thresholds["final_tangent_direction_cosine_min"],
            "fine_direction_rms": fine["tangent"]["direction_discrepancy_rms"],
            "coarse_direction_rms": coarse["tangent"]["direction_discrepancy_rms"],
            "worst_epoch_years": fine["tangent"][
                "minimum_direction_cosine_epoch_years"
            ],
        },
        "megno": {
            "final_abs_difference": fine["megno"]["final_abs_difference"],
            "final_limit": thresholds["final_megno_difference_max"],
            "history_rms_difference": fine["megno"]["history_rms_difference"],
            "history_rms_limit": thresholds["megno_history_rms_difference_max"],
            "worst_epoch_years": fine["megno"]["worst_epoch_years"],
        },
        "lcn": {
            "final_accumulated_abs_difference": fine["lcn"][
                "final_accumulated_abs_difference"
            ],
            "limit": thresholds["final_lcn_accumulated_difference_max"],
            "worst_epoch_years": 1_000_000.0,
        },
        "corrected_energy": {
            "per_run": {
                name: {
                    **energy[name],
                    "max_abs_limit": thresholds["corrected_energy_max_abs_per_run"],
                    "trend_limit": max(0.25 * energy[name]["max_abs"], 1.0e-10),
                }
                for name in ORDERED_RUN_IDS
            },
            "worst_run_by_max_abs": energy_run,
            "worst_max_abs": energy[energy_run]["max_abs"],
            "worst_epoch_years": energy[energy_run]["max_abs_worst_epoch_years"],
            "roundoff_floor": thresholds["corrected_energy_roundoff_floor"],
            "nonincreasing_with_step_reduction": energy_reductions,
        },
        "angular_momentum": {
            "per_run": angular,
            "worst_run": angular_run,
            "value": angular[angular_run]["max_abs"],
            "limit": thresholds["angular_momentum_rel_drift_max_per_run"],
            "worst_epoch_years": angular[angular_run]["max_abs_worst_epoch_years"],
            "accepted_100k_1d_scale": thresholds[
                "accepted_100k_1d_angular_momentum_rel_drift"
            ],
        },
        "callback_nonfinite": {
            "per_run": {
                name: {
                    "callback_invocations": runs[name].integrity["callback_invocations"],
                    "nonfinite_result_count": runs[name].integrity[
                        "nonfinite_result_count"
                    ],
                }
                for name in ORDERED_RUN_IDS
            },
            "limit": thresholds["nonfinite_callback_results"],
            "worst_epoch_years": None,
        },
    }


def _comparison(left: RunData, right: RunData) -> dict[str, Any]:
    return {
        "physical": _pair_physical(left, right),
        "tangent": _pair_tangent(left, right),
        "orbital_elements": _pair_elements(left, right),
        "megno": _pair_scalar(left, right, "megno"),
        "lcn": _pair_lcn(left, right),
        "angular_momentum": _pair_angular(left, right),
    }


def _energy_pattern(
    criteria: dict[str, dict[str, Any]],
    energy: dict[str, dict[str, float]],
) -> dict[str, Any]:
    candidate = ORDERED_RUN_IDS[1]
    reference = ORDERED_RUN_IDS[2]
    worsening = {
        metric: energy[reference][metric] > energy[candidate][metric]
        for metric in ("max_abs", "rms", "p99_abs")
    }
    physical_continues = criteria["physical_state"]["passed"]
    return {
        "physical_convergence_continues": physical_continues,
        "new_0p25_energy_worsens_relative_to_0p5": worsening,
        "physical_convergence_with_further_energy_worsening": physical_continues
        and any(worsening.values()),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['final_status']}",
        "",
        "One fresh 0.25-day 1 Myr lane was compared with the unchanged 1-day and 0.5-day Step 3 artifacts.",
        "",
        "## Runs",
        "",
        "| Run | Step | Runtime | Throughput | Samples | State rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id, run in payload.get("runs", {}).items():
        lines.append(
            f"| {run_id} | {run['step_days']:g} d | {run['runtime_seconds']:.3f} s | "
            f"{run['throughput_years_per_second']:.3f} yr/s | "
            f"{run['scientific_samples']} | {run['state_rows']} |"
        )
    lines.extend(["", "## Criteria", "", "| Criterion | Result |", "| --- | ---: |"])
    for name, result in payload.get("criteria", {}).items():
        lines.append(f"| {name} | {'PASS' if result.get('passed') else 'FAIL'} |")
    if payload.get("comparisons"):
        coarse = payload["comparisons"]["coarse_1d_vs_0p5d"]
        fine = payload["comparisons"]["fine_0p5d_vs_0p25d"]
        ratio = payload["criteria"]["physical_state"]["fine_over_coarse_global_rms_ratio"]
        peri = payload["criteria"]["mercury_perihelion_rate"][
            "fine_pair_abs_difference_arcsec_per_century"
        ]
        lines.extend(
            [
                "",
                "## Key Metrics",
                "",
                f"- Global scaled physical RMS: coarse `{coarse['physical']['global_scaled_rms']:.12g}`, "
                f"fine `{fine['physical']['global_scaled_rms']:.12g}`, ratio `{ratio:.12g}`.",
                f"- Mercury mean-perihelion-rate fine-pair difference: `{peri:.12g}` arcsec/century.",
                f"- Final tangent cosine: `{fine['tangent']['final_direction_cosine']:.12g}`; "
                f"direction RMS coarse/fine `{coarse['tangent']['direction_discrepancy_rms']:.12g}` / "
                f"`{fine['tangent']['direction_discrepancy_rms']:.12g}`.",
                f"- Final MEGNO difference: `{fine['megno']['final_abs_difference']:.12g}`.",
                f"- Final accumulated LCN difference: `{fine['lcn']['final_accumulated_abs_difference']:.12g}`.",
            ]
        )
        semimajor = payload["threshold_evidence"]["semimajor_axis_history"]
        lines.append(
            f"- Worst fine-pair semimajor-axis difference: {semimajor['worst_body']} "
            f"`{semimajor['value']:.12g}` at `{semimajor['worst_epoch_years']:.12g}` years "
            f"(limit `{semimajor['limit']:.12g}`)."
        )
    if payload.get("corrected_energy"):
        lines.extend(
            [
                "",
                "## Corrected Energy",
                "",
                "| Run | Maximum | RMS | P99 | Fitted change over 1 Myr |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for run_id in ORDERED_RUN_IDS:
            item = payload["corrected_energy"][run_id]
            lines.append(
                f"| {run_id} | {item['max_abs']:.12g} | {item['rms']:.12g} | "
                f"{item['p99_abs']:.12g} | {item['fitted_change_over_1myr']:.12g} |"
            )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Manifest SHA-256: `{payload.get('manifest_sha256')}`.",
            f"- Reused Step 3 artifacts unchanged: `{payload.get('step3_artifacts_unchanged')}`.",
            f"- Protected files unchanged: `{payload.get('protected_files_unchanged')}`.",
            "- No prior convergence lane or 10 Myr integration was launched.",
            "",
        ]
    )
    if payload.get("failures"):
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in payload["failures"])
        lines.append("")
    if payload.get("next_action"):
        lines.extend(["## Next Action", "", payload["next_action"], ""])
    return "\n".join(lines)


def _write_reports(manifest: dict[str, Any], payload: dict[str, Any]) -> None:
    _require(payload["final_status"] in FINAL_STATUSES, "Invalid Step 3b final status.")
    json_path = Path(manifest["paths"]["report_json"])
    markdown_path = Path(manifest["paths"]["report_markdown"])
    _atomic_write_json(json_path, payload)
    _atomic_write_text(markdown_path, _markdown_report(payload))
    print(f"[m0-convergence-0p25] wrote {json_path}")
    print(f"[m0-convergence-0p25] wrote {markdown_path}")


def analyze(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Step 3b manifest")
    manifest_hash = sha256_file(manifest_path)
    protected_unchanged = True
    try:
        _verify_protected(manifest)
    except Exception:
        protected_unchanged = False
    base_payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "model_id": manifest["model_id"],
        "production_candidate_step_days": manifest["production_candidate_step_days"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protected_files_unchanged": protected_unchanged,
        "step3_artifacts_unchanged": False,
        "thresholds": manifest["thresholds"],
        "comparison_definitions": manifest["comparison_definitions"],
        "failures": [],
    }
    try:
        _require(protected_unchanged, "Protected file hash changed during Step 3b.")
        step3_manifest_path, step3_manifest, step3_summary = _step3_context(manifest)
        existing_inventory = _verify_step3_inventory(step3_summary)
        base_payload["step3_artifacts_unchanged"] = True
        runs = {
            ORDERED_RUN_IDS[0]: _existing_run(
                step3_manifest_path, step3_manifest, ORDERED_RUN_IDS[0]
            ),
            ORDERED_RUN_IDS[1]: _existing_run(
                step3_manifest_path, step3_manifest, ORDERED_RUN_IDS[1]
            ),
            ORDERED_RUN_IDS[2]: _load_run(
                manifest_path, manifest, manifest["decisive_run"]
            ),
        }
        coarse = _comparison(runs[ORDERED_RUN_IDS[0]], runs[ORDERED_RUN_IDS[1]])
        fine = _comparison(runs[ORDERED_RUN_IDS[1]], runs[ORDERED_RUN_IDS[2]])
        perihelion = {name: _perihelion_rate(run) for name, run in runs.items()}
        energy = {
            name: _series_metrics(run.times, run.progress["corrected_energy_rel_change"])
            for name, run in runs.items()
        }
        angular = {
            name: _series_metrics(run.times, run.progress["angular_momentum_rel_drift"])
            for name, run in runs.items()
        }
        criteria = evaluate_candidate_criteria(
            manifest, runs, coarse, fine, perihelion, energy, angular
        )
        evidence = _threshold_evidence(
            manifest, runs, coarse, fine, perihelion, energy, angular
        )
        energy_pattern = _energy_pattern(criteria, energy)
        all_passed = all(result.get("passed") is True for result in criteria.values())
        final_status = (
            "M0_0P5DAY_CONVERGED" if all_passed else "M0_0P5DAY_NOT_CONVERGED"
        )
        failures = [
            name for name, result in criteria.items() if result.get("passed") is not True
        ]
        if final_status == "M0_0P5DAY_CONVERGED":
            next_action = manifest["next_action_rules"]["all_criteria_pass"]
        elif energy_pattern["physical_convergence_with_further_energy_worsening"]:
            next_action = manifest["next_action_rules"][
                "energy_worsens_while_physical_convergence_continues"
            ]
        else:
            next_action = manifest["next_action_rules"]["other_numeric_failure"]
        new_inventory = runs[ORDERED_RUN_IDS[2]].inventory
        base_payload.update(
            final_status=final_status,
            failures=failures,
            runs={name: run.integrity for name, run in runs.items()},
            comparisons={
                "coarse_1d_vs_0p5d": coarse,
                "fine_0p5d_vs_0p25d": fine,
            },
            mercury_perihelion=perihelion,
            corrected_energy=energy,
            angular_momentum=angular,
            criteria=criteria,
            threshold_evidence=evidence,
            energy_pattern=energy_pattern,
            next_action=next_action,
            artifact_inventory={
                "reused_step3": existing_inventory,
                "new_0p25d": new_inventory,
            },
        )
    except Exception as exc:
        base_payload.update(
            final_status="BLOCKED",
            failures=[str(exc)],
            criteria={},
            next_action="Resolve the recorded artifact or integrity failure without rerunning a prior lane.",
        )
    _write_reports(manifest, base_payload)
    return 0 if base_payload["final_status"] == "M0_0P5DAY_CONVERGED" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and analyze the M0 0.25-day lane.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("analyze")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            raise SystemExit(preflight(args.manifest))
        raise SystemExit(analyze(args.manifest))
    except ConvergenceError as exc:
        raise SystemExit(f"m0 convergence 0.25-day error: {exc}") from exc


if __name__ == "__main__":
    main()
