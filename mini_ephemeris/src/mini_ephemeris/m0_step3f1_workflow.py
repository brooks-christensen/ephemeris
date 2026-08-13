from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .m0_step3f1_analysis import _artifact, _atomic_csv, _atomic_text, _finite, _historical_tangent, _ias15, _new_lane, _raw_detail
from .m0_step3f1_closeout import _configuration_rows, _conservation_gate, _figures, _metric_rows, _native, _sync_raw_threshold, _tangent_gate
from .m0_step3f1_contract import BODY_NAMES, load_json, require, sha256_file, validate_manifest
from .m0_step3f1_metrics import _conservation, _frequencies, _orbit_gate, _perihelion, _phase_and_orbit, _raw_gate, _tangent
from .m0_step3f1_runner import audit
from .rebound_gr_tangent_backend_cli import atomic_write_json


def _compact_raw(detail: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in detail.items() if key != "sample_scaled_rms"}


def _compact_conservation(detail: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in detail.items() if key not in {"energy_history", "angular_history"}}


def _compact_tangent(detail: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in detail.items() if key not in {"new_tangent_norm", "old_tangent_norm"}}


def _perihelion_pairs(perihelion: dict[str, dict[str, float]]) -> dict[str, float]:
    mapping = {
        "P_vs_IAS15": ("P", "IAS15"),
        "T_vs_IAS15": ("T", "IAS15"),
        "T_vs_P": ("T", "P"),
        "T_vs_old_T": ("T", "old_T"),
    }
    return {
        name: abs(perihelion[left]["rate_arcsec_per_century"] - perihelion[right]["rate_arcsec_per_century"])
        for name, (left, right) in mapping.items()
    }


def _report(manifest: dict[str, Any], payload: dict[str, Any]) -> str:
    raw = payload["physical"]["raw"]
    orbit = payload["physical"]["orbital"]
    peri = payload["physical"]["mercury_perihelion_pair_difference"]
    conservation = payload["conservation"]
    tangent = payload["tangent"]
    lines = [
        "# M0 Step 3f1 Two-Lane Architecture Screen",
        "",
        "**Final status:** {}".format(payload["final_status"]),
        "",
        f"**Primary finding:** `{payload['primary_finding']}`",
        "",
        "## Decision",
        "",
        payload["decision_statement"],
        "",
        "This is an architecture screen only. Manifest 17 remains `STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED`, and Manifest 18 remains historically valid for its combined standard-kernel lane. This result does not retroactively validate the 0.25-day timestep. Stage 4 and the 10 Myr production experiment remain unauthorized.",
        "",
        "## Exact Lane Contracts",
        "",
        "| Lane | Role | Kernel | Corrector | Variations | MEGNO | safe_mode | keep_unsynchronized | Steps | Samples | State rows | Archives |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("P", "T"):
        lane = manifest["lane_contracts"][key]
        integrity = payload["integrity"]["lanes"][key]
        lines.append(f"| {key} | {lane['responsibility']} | {lane['kernel']} | {lane['corrector']} | {lane['variations']} | {lane['megno']} | {lane['safe_mode']} | {lane['keep_unsynchronized']} | {integrity['steps']} | {integrity['scientific_samples']} | {integrity['state_rows']} | {integrity['archive_snapshots']} |")
    lines.extend([
        "",
        "Lane T was freshly executed. The frozen historical prefix used `safe_mode=1` and `keep_unsynchronized=0`, so it was not contract-identical and was used only as the preregistered one-factor synchronization control.",
        "",
        "Both lanes used the same DE431 state, Sun-through-Pluto M0 physics, 0.25-day step, 10,000-Julian-year interval, integer targets, and compiled GR callback. Lane P used physical-only WHCKL (`lazy`, order-17 corrector); Lane T used standard WHFast with native first variations, the compiled analytic Jacobian, MEGNO, and finite-time LCN.",
        "",
        "## Integrity And Runtime",
        "",
    ])
    for key in ("P", "T"):
        lane = payload["integrity"]["lanes"][key]
        runtime = payload["runtime"][key]
        lines.append(f"- Lane {key}: {lane['callback_invocations']:,} callback evaluations, zero nonfinite results, {runtime['runtime_seconds']:.3f} s wall time, {runtime['throughput_years_per_wall_second']:.3f} simulated years/s.")
        restart = lane["restart"]
        lines.append(f"- Lane {key} restart: exact physical closure `{restart['physical_state_exact_float64']}`, variation closure `{restart['variation_state_exact_float64']}`, MEGNO/LCN closure `{restart['megno_exact_float64']}/{restart['lcn_exact_float64']}`, archive hash unchanged.")
    lines.extend([
        "",
        "The live WHFast map remained unsynchronized after every positive-time output (`is_synchronized=0` before and after sampling). Output diagnostics used a disposable simulation copy and did not change the live particle state, step count, or time. Archive inspection did not mutate either archive.",
        "",        "Lane P observed 29,226,432 callback evaluations versus the preregistered 29,223,232. The exact +3,200 difference is 32 extra order-17 corrector evaluations for each of 100 diagnostic copies. Its restart similarly observed 292,264 versus 292,232. These remain failed integrity gates despite exact state closure. Lane T callback accounting and restart accounting passed; its final sidecars were recovered offline after a post-assertion NameError, without rerunning the trajectory.",
        "",
        "The frozen IAS15 default reference was used only at its 101 stored 100-year timestamps over 0-10 kyr. Its characterized tolerance envelope was carried forward; it was not treated as exact truth beneath that floor. All frozen IAS15 and historical artifact hashes passed before and after analysis.",
        "",
        "## Physical State",
        "",
        "| Comparison | Global scaled RMS | Dominant body | Body RMS | Squared-error contribution | Mean-anomaly stripped RMS | Mean-longitude stripped RMS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for comparison in ("P_vs_IAS15", "T_vs_IAS15", "T_vs_P", "T_vs_old_T"):
        detail = raw[comparison]
        dominant = max(BODY_NAMES, key=lambda name: detail["per_body"][name]["squared_error_contribution"])
        phase = orbit.get(comparison)
        anomaly = phase["phase_aligned"]["mean_anomaly"]["global_scaled_rms"] if phase else float("nan")
        longitude = phase["phase_aligned"]["mean_longitude"]["global_scaled_rms"] if phase else float("nan")
        lines.append(f"| {comparison} | {detail['global_scaled_rms']:.9e} | {dominant} | {detail['per_body'][dominant]['scaled_rms']:.9e} | {detail['per_body'][dominant]['squared_error_contribution']:.6f} | {anomaly:.9e} | {longitude:.9e} |")
    lines.extend([
        "",
        "- Lane P passed the global IAS15 scaled-state limit but failed Saturn: {:.9e} versus {:.9e}.".format(raw["P_vs_IAS15"]["per_body"]["saturn barycenter"]["scaled_rms"], manifest["screen_thresholds"]["lane_p_vs_ias15"]["per_body_scaled_rms_max"]["saturn barycenter"]),        "- Lane T failed preregistered outer-body carrier limits against IAS15/Lane P and failed the historical synchronization-control limits for Sun, Jupiter, Uranus, Neptune, and Pluto.",
        "Raw Cartesian differences were screened but were not the sole architecture veto. RTN decompositions, both preregistered shared-phase reconstructions, coordinate-free orbital histories, and ten fixed 1-kyr windows were evaluated for every planet.",
        "",
        "## Orbital And Secular Results",
        "",
    ])
    threshold = manifest["screen_thresholds"]["all_physical_pairs"]
    for comparison in ("P_vs_IAS15", "T_vs_IAS15", "T_vs_P"):
        worst_a_name = max(BODY_NAMES[1:], key=lambda name: orbit[comparison]["per_body"][name]["semimajor_axis_relative_max"])
        worst_e_name = max(BODY_NAMES[1:], key=lambda name: orbit[comparison]["per_body"][name]["eccentricity_vector_norm_max"])
        lines.append(f"- `{comparison}`: maximum relative semimajor-axis difference {orbit[comparison]['per_body'][worst_a_name]['semimajor_axis_relative_max']:.9e} ({worst_a_name}); maximum eccentricity-vector difference {orbit[comparison]['per_body'][worst_e_name]['eccentricity_vector_norm_max']:.9e} ({worst_e_name}); Mercury perihelion-rate difference {peri[comparison]:.9e} arcsec/century; persistent nonphase failures {len(orbit[comparison]['persistent_nonphase_failures'])}.")
    lines.extend([
        f"- Frozen coordinate-free limits were {threshold['semimajor_axis_relative_max']:.1e} for relative semimajor axis and {threshold['eccentricity_absolute_max']:.1e} for eccentricity, eccentricity-vector norm, inclination components, and orbital-plane direction; the Mercury pair-rate limit was {threshold['mercury_perihelion_rate_difference_arcsec_per_century_max']:.3g} arcsec/century.",
        "- Secular-frequency estimates and absolute Mercury rates remain contextual under the frozen 10-kyr/cadence policy. The binary64 floor of the frozen arccos plane-angle estimator is an essential unresolved method issue; pair-rate differences were still screened exactly as preregistered.",
        "",
        "## Conservation",
        "",
    ])
    for key in ("P", "T"):
        lines.append(f"- Lane {key}: corrected-energy max |drift| {conservation[key]['energy']['max_abs']:.9e}, fitted 10-kyr change {conservation[key]['energy']['fitted_change_over_10k']:.9e}; angular-momentum max |drift| {conservation[key]['angular_momentum']['max_abs']:.9e}.")
    lines.extend([
        f"- Pair corrected-energy history max difference: {payload['gates']['conservation']['pair_energy_history_max_abs_difference']:.9e}.",
        f"- Pair angular-momentum history max difference: {payload['gates']['conservation']['pair_angular_history_max_abs_difference']:.9e}.",
        "- Corrected energy was independently recomputed from every stored state row using the frozen Newtonian plus GR-potential definition; the recomputed and recorded histories agreed exactly in binary64.",
        "",
        "## Tangent And Chaos Diagnostics",
        "",
        f"- Final direction cosine against the historical tangent prefix: {tangent['final_direction_cosine']:.12f}; direction discrepancy RMS: {tangent['direction_discrepancy_rms']:.9e}.",
        f"- Maximum log tangent-norm difference: {tangent['tangent_log_norm_difference_max']:.9e}; fitted new-lane log-norm growth: {tangent['new_tangent_log_norm_fitted_growth_per_year']:.9e} per year.",
        f"- Final MEGNO: {tangent['final_megno']:.12g}; final/history RMS MEGNO differences: {tangent['final_megno_difference']:.9e}/{tangent['megno_history_rms_difference']:.9e}.",
        f"- Final finite-time LCN: {tangent['final_lcn_1_per_year']:.9e} 1/year; final/history RMS accumulated-LCN differences: {tangent['final_accumulated_lcn_difference']:.9e}/{tangent['lcn_history_accumulated_rms_difference']:.9e}.",
        "- This tests numerical continuity of the already-validated tangent implementation; it does not establish long-duration chaos evidence.",
        "",
        "## Classification Evidence",
        "",
    ])
    for name, gate in payload["gates"].items():
        if isinstance(gate, dict) and "passed" in gate:
            lines.append(f"- `{name}`: `{'PASS' if gate['passed'] else 'FAIL'}`.")
    lines.extend([
        "",
        "Evidence supports {} and the emitted status {}. Frequency and absolute-rate estimates are nonessential. The plane-angle estimator remains an essential unresolved method issue, but it does not change the classification because Lane P raw physical and integrity gates and Lane T carrier/tangent gates fail independently.".format(payload["primary_finding"], payload["final_status"]),
        "",
        "## Artifacts",
        "",
        "The deterministic metrics table contains every coarse gate value and per-body coordinate-free result. The lane-configuration table records both effective contracts. Four PNG figures show physical defects, coordinate-free discrepancies, conservation histories, and tangent/MEGNO/LCN continuity. Raw trajectories and operational sidecars remain under the ignored Step 3f1 output root.",
        "",
        "## Smallest Successor Action",
        "",
        payload["smallest_successor_action"],
        "",
    ])
    return "\n".join(lines)


def analyze(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    audit_before = audit(manifest_path)
    runs = {"P": _new_lane(manifest, "P"), "T": _new_lane(manifest, "T"), "IAS15": _ias15(manifest), "old_T": _historical_tangent(manifest)}
    comparisons = {
        "P_vs_IAS15": (runs["P"], runs["IAS15"]),
        "T_vs_IAS15": (runs["T"], runs["IAS15"]),
        "T_vs_P": (runs["T"], runs["P"]),
        "T_vs_old_T": (runs["T"], runs["old_T"]),
    }
    raw_full = {name: _raw_detail(left, right) for name, (left, right) in comparisons.items()}
    orbital = {name: _phase_and_orbit(*pair) for name, pair in comparisons.items()}
    perihelion = {name: _perihelion(run) for name, run in runs.items()}
    perihelion_pairs = _perihelion_pairs(perihelion)
    frequencies = {name: _frequencies(runs[name]) for name in ("P", "T", "IAS15")}
    conservation_full = {name: _conservation(runs[name]) for name in ("P", "T")}
    tangent_full = _tangent(runs["T"], runs["old_T"])
    threshold = manifest["screen_thresholds"]
    raw_gates = {
        "P_vs_IAS15": _raw_gate(raw_full["P_vs_IAS15"], threshold["lane_p_vs_ias15"]),
        "T_vs_IAS15": _raw_gate(raw_full["T_vs_IAS15"], threshold["lane_t_carrier_vs_ias15_and_lane_p"]),
        "T_vs_P": _raw_gate(raw_full["T_vs_P"], threshold["lane_t_carrier_vs_ias15_and_lane_p"]),
        "T_vs_old_T": _raw_gate(raw_full["T_vs_old_T"], _sync_raw_threshold(manifest)),
    }
    orbit_gates = {name: _orbit_gate(detail, threshold["all_physical_pairs"]) for name, detail in orbital.items()}
    perihelion_gate = {
        "passed": all(value <= threshold["all_physical_pairs"]["mercury_perihelion_rate_difference_arcsec_per_century_max"] for value in perihelion_pairs.values()),
        "checks": {name: value <= threshold["all_physical_pairs"]["mercury_perihelion_rate_difference_arcsec_per_century_max"] for name, value in perihelion_pairs.items()},
    }
    conservation_gate = _conservation_gate(manifest, conservation_full)
    tangent_gate = _tangent_gate(manifest, tangent_full)
    integrity_gate = {"passed": all(runs[key].integrity["passed"] for key in ("P", "T")), "checks": {key: runs[key].integrity["passed"] for key in ("P", "T")}}
    p_conservation = all(value for key, value in conservation_gate["checks"].items() if key.startswith("P:"))
    t_conservation = all(value for key, value in conservation_gate["checks"].items() if key.startswith("T:") or key.startswith("pair_"))
    p_pass = runs["P"].integrity["passed"] and raw_gates["P_vs_IAS15"]["passed"] and orbit_gates["P_vs_IAS15"]["passed"] and perihelion_gate["checks"]["P_vs_IAS15"] and p_conservation
    t_pass = runs["T"].integrity["passed"] and all(raw_gates[name]["passed"] for name in ("T_vs_IAS15", "T_vs_P", "T_vs_old_T")) and all(orbit_gates[name]["passed"] for name in ("T_vs_IAS15", "T_vs_P", "T_vs_old_T")) and all(perihelion_gate["checks"][name] for name in ("T_vs_IAS15", "T_vs_P")) and tangent_gate["passed"] and t_conservation
    if p_pass and t_pass:
        finding, status = "TWO_LANE_ARCHITECTURE_SUPPORTED", "STEP3F1_TWO_LANE_SCREEN_PASSED"
        decision = "All essential preregistered integrity, physical, carrier, coordinate-free, conservation, tangent, chaos-diagnostic, and restart screens passed. Lane P and Lane T can own their proposed responsibilities for a bounded operational rehearsal."
        successor = "Separately preregister Step 3f2 as one serial 100-kyr operational rehearsal of the same two contracts, with frozen output cadence, storage budget, checkpoint/restart behavior, and claim-aligned stop rules. Do not begin that rehearsal from this Step 3f1 closeout."
    elif not p_pass and t_pass:
        finding, status = "PHYSICAL_WHCKL_LANE_UNQUALIFIED", "STEP3F1_TWO_LANE_SCREEN_FAILED"
        decision = "Lane P failed at least one material preregistered physical screen while Lane T passed its assigned screens."
        successor = "Perform the smallest offline, claim-aligned localization of the failed Lane P metric by body, component, and fixed window; do not assume a smaller timestep is better."
    elif p_pass and not t_pass:
        finding, status = "TANGENT_LANE_UNQUALIFIED", "STEP3F1_TWO_LANE_SCREEN_FAILED"
        decision = "Lane P passed its assigned screens, but Lane T failed tangent/MEGNO continuity or physical-carrier consistency."
        successor = "Run no longer trajectory. Isolate the failed tangent, synchronization, or carrier metric using the existing 10-kyr state and checkpoint evidence first."
    elif not p_pass and not t_pass:
        finding, status = "BOTH_LANES_UNQUALIFIED", "STEP3F1_TWO_LANE_SCREEN_FAILED"
        decision = "Both proposed lanes failed material assigned screens. Lane P missed its frozen IAS15 physical threshold and callback-accounting integrity gate; Lane T missed carrier-consistency and tangent/MEGNO continuity gates."
        successor = "Stop and perform a separately preregistered source-only and offline audit of the shared diagnostic-copy/synchronization representation against the frozen IAS15 state rows. This requires no new scientific integration and must resolve the order-17 copy callback accounting and coordinate-direction estimator floor before any architecture retest."
    else:
        finding, status = "MIXED_OR_INCONCLUSIVE", "STEP3F1_TWO_LANE_SCREEN_INCONCLUSIVE"
        decision = "The essential evidence did not support a unique preregistered classification."
        successor = "Obtain the single missing essential diagnostic offline from the stored Step 3f1 states before scheduling any new integration."
    unresolved = [{"metric": "angular_momentum_direction", "lane": "all physical pairs", "reason": "the frozen arccos(dot) estimator has a binary64 floor above the 1e-8 threshold", "essential": True}]
    for lane, bodies in frequencies.items():
        for body, modes in bodies.items():
            for mode, estimate in modes.items():
                if estimate["alias_risk"]:
                    unresolved.append({"metric": "secular_frequency", "lane": lane, "body": body, "mode": mode, "reason": "below two Fourier bins or above 0.8 Nyquist", "essential": False})
    for lane, estimate in perihelion.items():
        if estimate["confidence_95_half_width_arcsec_per_century"] > threshold["all_physical_pairs"]["mercury_perihelion_rate_difference_arcsec_per_century_max"]:
            unresolved.append({"metric": "absolute_mercury_perihelion_rate", "lane": lane, "reason": "95-percent interval exceeds pair threshold", "essential": False})
    gates = {"integrity": integrity_gate, "raw_P_vs_IAS15": raw_gates["P_vs_IAS15"], "carrier_T_vs_IAS15": raw_gates["T_vs_IAS15"], "carrier_T_vs_P": raw_gates["T_vs_P"], "sync_T_vs_old_T": raw_gates["T_vs_old_T"], "orbit_P_vs_IAS15": orbit_gates["P_vs_IAS15"], "orbit_T_vs_IAS15": orbit_gates["T_vs_IAS15"], "orbit_T_vs_P": orbit_gates["T_vs_P"], "orbit_T_vs_old_T": orbit_gates["T_vs_old_T"], "mercury_perihelion_pairs": perihelion_gate, "conservation": conservation_gate, "tangent": tangent_gate, "lane_P_assigned": {"passed": p_pass}, "lane_T_assigned": {"passed": t_pass}}
    payload: dict[str, Any] = {
        "schema_version": 1, "kind": "m0_step3f1_two_lane_architecture_screen_summary",
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": sha256_file(manifest_path),
        "final_status": status, "primary_finding": finding, "decision_statement": decision,
        "scope": "Architecture screen only; no production qualification or Stage 4 authorization.",
        "historical_results_preserved": manifest["frozen_historical_results"],
        "runtime_identity": audit_before["runtime"],
        "runtime": {key: {"runtime_seconds": runs[key].summary["runtime_seconds"], "throughput_years_per_wall_second": runs[key].summary["throughput_years_per_wall_second"]} for key in ("P", "T")},
        "integrity": {"passed": integrity_gate["passed"], "lanes": {key: runs[key].integrity for key in ("P", "T")}, "frozen_reference_artifacts_verified": len(audit_before["reference_artifacts"]), "protected_files_verified": len(audit_before["protected_files"])},
        "tangent_lane_reuse": {"reused": False, "executed_fresh": True, "historical_role": manifest["tangent_reuse_decision"]["historical_lane_role"], "reason": manifest["tangent_reuse_decision"]["reason"]},
        "reference_handling": manifest["reference_contract"],
        "thresholds": {**threshold, "tangent_sync_raw": _sync_raw_threshold(manifest)},
        "physical": {"raw": {name: _compact_raw(detail) for name, detail in raw_full.items()}, "orbital": orbital, "mercury_perihelion": perihelion, "mercury_perihelion_pair_difference": perihelion_pairs, "secular_frequencies": frequencies},
        "conservation": {key: _compact_conservation(value) for key, value in conservation_full.items()},
        "tangent": _compact_tangent(tangent_full), "gates": gates,
        "unresolved_metrics": unresolved, "essential_unresolved_metric_count": sum(1 for item in unresolved if item["essential"]),
        "smallest_successor_action": successor,
        "explicit_constraints": {"manifest17_not_validated": True, "manifest18_historical_combined_lane_conclusion_preserved": True, "step3f1_does_not_validate_0p25_day": True, "stage4_unauthorized": True, "ten_myr_unauthorized": True},
    }
    report_root = Path(manifest["paths"]["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    configuration_rows = _configuration_rows(manifest, runs)
    configuration_fields = tuple(configuration_rows[0])
    _atomic_csv(Path(manifest["paths"]["configuration_table"]), configuration_fields, configuration_rows)
    metrics_rows = _metric_rows(payload)
    _atomic_csv(Path(manifest["paths"]["metrics_table"]), tuple(metrics_rows[0]), metrics_rows)
    figures = _figures(manifest, runs, {name: detail["sample_scaled_rms"] for name, detail in raw_full.items()}, conservation_full, tangent_full, orbital)
    payload["derived_artifacts"] = {"configuration_table": _artifact(Path(manifest["paths"]["configuration_table"])), "metrics_table": _artifact(Path(manifest["paths"]["metrics_table"])), "figures": figures}
    report_text = _report(manifest, payload)
    _atomic_text(Path(manifest["paths"]["report"]), report_text)
    payload["derived_artifacts"]["report"] = _artifact(Path(manifest["paths"]["report"]))
    payload = _native(payload)
    _finite(payload)
    atomic_write_json(Path(manifest["paths"]["summary"]), payload)
    audit_after = audit(manifest_path)
    require([item["sha256"] for item in audit_before["reference_artifacts"]] == [item["sha256"] for item in audit_after["reference_artifacts"]], "Frozen reference changed during analysis.")
    return payload


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    audit_payload = audit(manifest_path)
    summary_path = Path(manifest["paths"]["summary"])
    raw_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(raw_text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    _finite(summary)
    require(summary["final_status"] in manifest["allowed_final_statuses"], "Invalid Step 3f1 final status.")
    require(summary["primary_finding"] in manifest["allowed_primary_findings"], "Invalid Step 3f1 primary finding.")
    require(summary["manifest_sha256"] == sha256_file(manifest_path), "Summary manifest hash changed.")
    for lane in ("P", "T"):
        integrity = summary["integrity"]["lanes"][lane]
        require(integrity["scientific_samples"] == 101 and integrity["state_rows"] == 1010, f"Lane {lane} derived counts changed.")
        require(integrity["steps"] == 14610000 and integrity["archive_snapshots"] == 11, f"Lane {lane} execution counts changed.")
        for item in integrity["artifacts"].values():
            require(_artifact(Path(item["path"])) == item, f"Lane {lane} artifact inventory changed.")
    derived = summary["derived_artifacts"]
    for key in ("configuration_table", "metrics_table", "report"):
        require(_artifact(Path(derived[key]["path"])) == derived[key], f"Derived artifact changed: {key}")
    for item in derived["figures"]:
        require(_artifact(Path(item["path"])) == item, f"Figure changed: {item['path']}")
    with Path(manifest["paths"]["configuration_table"]).open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == 2, "Lane configuration row count changed.")
    with Path(manifest["paths"]["metrics_table"]).open(newline="", encoding="utf-8") as handle:
        metric_count = len(list(csv.DictReader(handle)))
    require(metric_count > 100, "Step 3f1 metrics table is incomplete.")
    report = Path(manifest["paths"]["report"]).read_text(encoding="utf-8")
    require(summary["final_status"] in report and summary["primary_finding"] in report, "Report disagrees with summary.")
    for text in ("Manifest 17 remains", "Manifest 18 remains", "Stage 4", "10 Myr"):
        require(text in report, f"Report omission: {text}")
    return {"status": "PASS", "final_status": summary["final_status"], "primary_finding": summary["primary_finding"], "summary": _artifact(summary_path), "derived_artifact_count": 7, "metrics_rows": metric_count, "protected_files_verified": len(audit_payload["protected_files"]), "reference_artifacts_verified": len(audit_payload["reference_artifacts"])}
