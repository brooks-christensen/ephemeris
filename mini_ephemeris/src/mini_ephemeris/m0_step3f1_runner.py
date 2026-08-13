from __future__ import annotations

import csv
import hashlib
import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Sequence
import warnings

import numpy as np

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import CBackend, load_c_backend
from .long_term_stability_cli import (
    build_rebound_simulation,
    configure_rebound_simulationarchive,
    optional_import_module,
    rebound_state_from_sim,
    stability_body_list,
)
from .m0_energy_precision_diagnosis import float64_energy
from .m0_step3f1_contract import (
    AU_M,
    BODY_NAMES,
    DEFAULT_MANIFEST,
    JULIAN_YEAR_S,
    PROGRESS_FIELDS,
    ROOT,
    STATE_FIELDS,
    VELOCITY_SCALE,
    Step3f1Error,
    artifact_identity,
    canonical_hash,
    lane_paths,
    lane_payload,
    load_json,
    require,
    sha256_file,
    validate_manifest,
)
from .nbody import G_SI, NBodyState
from .rebound_gr_tangent_backend_cli import atomic_write_json, initial_condition_hash
from .stability_diagnostics import total_angular_momentum_vector


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _event(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _runtime_identity(rebound: Any, backend: CBackend) -> dict[str, Any]:
    rebound_library = Path(rebound.clibrebound._name)
    reboundx_library = rebound_library.with_name(
        "libreboundx.cpython-310-x86_64-linux-gnu.so"
    )
    return {
        "python": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "numpy": np.__version__,
        "rebound": rebound.__version__,
        "rebound_githash": getattr(rebound, "__githash__", None),
        "rebound_build": getattr(rebound, "__build__", None),
        "librebound_path": str(rebound_library),
        "librebound_sha256": sha256_file(rebound_library),
        "libreboundx_path": str(reboundx_library),
        "libreboundx_sha256": sha256_file(reboundx_library),
        "callback_library_path": str(backend.artifact_path),
        "callback_library_sha256": backend.build_metadata["artifact_sha256"],
        "callback_source_sha256": backend.build_metadata["source_sha256"],
        "callback_abi": backend.abi_metadata,
    }


def _audit_file_set(entries: dict[str, Sequence[Any]], label: str) -> list[dict[str, Any]]:
    output = []
    for name, entry in entries.items():
        path = ROOT / str(entry[0])
        require(path.is_file(), f"Missing {label} {name}: {path}")
        actual = sha256_file(path)
        require(actual == entry[1], f"Changed {label} {name}: {path}")
        if len(entry) >= 3:
            require(path.stat().st_size == int(entry[2]), f"Size changed for {label} {name}.")
        output.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return output


def _reference_entries() -> dict[str, Sequence[Any]]:
    manifest18 = load_json(
        ROOT / "ephemeris_experiment_runner/manifests/18_m0_step3e1_offline_state_diagnosis_v1.json",
        "Manifest 18",
    )
    entries: dict[str, Sequence[Any]] = {}
    for lane_id, values in manifest18["ias15_evidence_artifacts"].items():
        for artifact_name, entry in values.items():
            entries[f"{lane_id}:{artifact_name}"] = entry
    tangent = next(
        lane
        for lane in manifest18["stored_lanes"]
        if lane["run_id"] == "m0_conv_0p25d_1myr_s12345"
    )
    output_dir = Path(tangent["output_dir"])
    for artifact_name, entry in tangent["artifact_inventory"].items():
        path = (output_dir / entry[0]).resolve()
        entries[f"historical_tangent:{artifact_name}"] = [str(path), entry[1], entry[2]]
    return entries


def audit(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    require(
        _git("merge-base", "--is-ancestor", manifest["preregistration"]["starting_commit"], "HEAD")
        == "",
        "Starting commit is not an ancestor of HEAD.",
    )
    require(_git("cat-file", "-t", manifest["provenance"]["compiled_c_tag"]) == "tag", "Compiled-C tag is not annotated.")
    require(
        _git("rev-list", "-n", "1", manifest["provenance"]["compiled_c_tag"])
        == manifest["provenance"]["compiled_c_tag_commit"],
        "Compiled-C tag target changed.",
    )
    protected = _audit_file_set(
        {name: [name, digest] for name, digest in manifest["protected_files"].items()},
        "protected file",
    )
    source = _audit_file_set(manifest["source_artifacts"], "source artifact")
    references = _audit_file_set(_reference_entries(), "reference artifact")
    rebound = optional_import_module("rebound")
    require(rebound is not None, "REBOUND is unavailable.")
    backend = load_c_backend()
    runtime = _runtime_identity(rebound, backend)
    expected = manifest["runtime_identity"]
    for name in (
        "python",
        "python_compiler",
        "numpy",
        "rebound",
        "rebound_githash",
        "rebound_build",
        "librebound_sha256",
        "libreboundx_sha256",
        "callback_library_sha256",
        "callback_source_sha256",
    ):
        require(runtime[name] == expected[name], f"Runtime identity changed: {name}")
    require(
        runtime["callback_abi"]["c_sizeof_reb_simulation"]
        == expected["c_sizeof_reb_simulation"],
        "REBOUND simulation ABI changed.",
    )
    require(
        runtime["callback_abi"]["c_sizeof_reb_particle"]
        == expected["c_sizeof_reb_particle"],
        "REBOUND particle ABI changed.",
    )
    output_root = Path(manifest["paths"]["output_root"])
    allowed = {lane["id"] for lane in manifest["lane_contracts"].values()}
    allowed.add("zero_step_audit.json")
    if output_root.exists():
        unexpected = [child.name for child in output_root.iterdir() if child.name not in allowed]
        require(not unexpected, f"Unauthorized Step 3f1 output: {unexpected}")
    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "git_head": _git("rev-parse", "HEAD"),
        "compiled_c_tag_commit": manifest["provenance"]["compiled_c_tag_commit"],
        "protected_files": protected,
        "source_artifacts": source,
        "reference_artifacts": references,
        "runtime": runtime,
        "available_disk_bytes": os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
    }


def _initial_state(manifest: dict[str, Any]) -> tuple[tuple[str, ...], NBodyState]:
    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    require(tuple(bodies) == BODY_NAMES, "M0 body order changed.")
    state = initial_state_solar_system_barycentric(
        dt.datetime.fromisoformat(manifest["common_physical_contract"]["start_date"]),
        bodies=bodies,
        config=EphemerisConfig(kernel_path=manifest["paths"]["kernel"]),
    )
    require(
        initial_condition_hash(state, bodies)
        == manifest["common_physical_contract"]["initial_conditions_sha256"],
        "M0 initial-condition hash changed.",
    )
    return tuple(bodies), state


def _settings(sim: Any) -> dict[str, Any]:
    whfast = sim.ri_whfast
    return {
        "integrator": str(sim.integrator),
        "dt_seconds": float(sim.dt),
        "coordinates": str(whfast.coordinates),
        "kernel": str(whfast.kernel),
        "corrector": int(whfast.corrector),
        "corrector2": int(whfast.corrector2),
        "safe_mode": int(whfast.safe_mode),
        "keep_unsynchronized": int(whfast.keep_unsynchronized),
        "recalculate_coordinates_this_timestep": int(
            whfast.recalculate_coordinates_this_timestep
        ),
        "is_synchronized": int(whfast.is_synchronized),
        "N": int(sim.N),
        "N_real": int(sim.N_real),
        "N_var": int(sim.N_var),
        "N_var_config": int(sim.N_var_config),
        "force_is_velocity_dependent": int(sim.force_is_velocity_dependent),
    }


def _validate_settings(sim: Any, manifest: dict[str, Any], lane_key: str) -> None:
    lane = manifest["lane_contracts"][lane_key]
    common = manifest["common_physical_contract"]
    values = _settings(sim)
    require(values["integrator"] == "whfast", "Integrator changed.")
    require(values["dt_seconds"] == common["step_seconds"], "Timestep changed.")
    require(values["coordinates"] == common["coordinates"], "Coordinates changed.")
    require(values["kernel"] == lane["kernel"], "WHFast kernel changed.")
    for name in (
        "corrector",
        "corrector2",
        "safe_mode",
        "keep_unsynchronized",
        "recalculate_coordinates_this_timestep",
    ):
        require(values[name] == lane[name], f"Lane {lane_key} {name} changed.")
    require(values["N_real"] == 10, "Real-particle count changed.")
    require(values["force_is_velocity_dependent"] == 0, "Callback became velocity dependent.")
    if lane_key == "P":
        require(values["N"] == 10 and values["N_var"] == 0 and values["N_var_config"] == 0, "Lane P is not physical-only.")
    else:
        require(values["N"] == 20 and values["N_var"] == 10 and values["N_var_config"] == 1, "Lane T variation layout changed.")
        config = sim.var_config[0]
        require(int(config.index) == 10 and int(config.order) == 1 and int(config.testparticle) == -1, "Lane T variation identity changed.")


def _build_lane(
    manifest: dict[str, Any], lane_key: str
) -> tuple[Any, Any, CBackend, tuple[str, ...], NBodyState, dict[str, Any]]:
    rebound = optional_import_module("rebound")
    require(rebound is not None, "REBOUND is unavailable.")
    bodies, state0 = _initial_state(manifest)
    common = manifest["common_physical_contract"]
    lane = manifest["lane_contracts"][lane_key]
    sim = build_rebound_simulation(
        rebound,
        state0,
        integrator="whfast",
        step_s=common["step_seconds"],
        ias15_epsilon=1.0e-10,
    )
    whfast = sim.ri_whfast
    whfast.coordinates = common["coordinates"]
    whfast.kernel = lane["kernel"]
    whfast.corrector = lane["corrector"]
    whfast.corrector2 = lane["corrector2"]
    whfast.safe_mode = lane["safe_mode"]
    whfast.keep_unsynchronized = lane["keep_unsynchronized"]
    whfast.recalculate_coordinates_this_timestep = lane[
        "recalculate_coordinates_this_timestep"
    ]
    initial_real = np.array(
        [[p.x, p.y, p.z, p.vx, p.vy, p.vz, p.m] for p in sim.particles],
        dtype=np.float64,
    )
    if lane_key == "T":
        sim.init_megno(seed=lane["megno_seed"])
        after_real = np.array(
            [[p.x, p.y, p.z, p.vx, p.vy, p.vz, p.m] for p in sim.particles[:10]],
            dtype=np.float64,
        )
        require(np.array_equal(initial_real, after_real), "Variations back-reacted at initialization.")
    backend = load_c_backend()
    backend.attach(
        sim,
        coefficient_scale=common["gr_scale"],
        include_central_response=common["include_central_response"],
    )
    _validate_settings(sim, manifest, lane_key)
    stats = backend.stats(sim)
    require(stats["callback_invocations"] == 0, "Zero-step construction evaluated the callback.")
    require(stats["nonfinite_result_count"] == 0, "Zero-step callback state is nonfinite.")
    require(backend.hot_path_proof(sim)["addresses_match"] is True, "C callback is not directly attached.")
    construction = {
        "settings": _settings(sim),
        "initial_real_sha256": hashlib.sha256(initial_real.tobytes()).hexdigest(),
        "no_backreaction": True,
        "callback_stats": stats,
        "hot_path": backend.hot_path_proof(sim),
    }
    return rebound, sim, backend, bodies, state0, construction


def zero_step(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    audit_payload = audit(manifest_path)
    lanes = {}
    initial_hashes = []
    for lane_key in ("P", "T"):
        _, sim, backend, _, _, construction = _build_lane(manifest, lane_key)
        lanes[lane_key] = construction
        initial_hashes.append(construction["initial_real_sha256"])
        backend.library.me_gr_tangent_detach(ctypes.byref(sim))
    require(len(set(initial_hashes)) == 1, "Lane initial physical states differ.")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "integration_steps": 0,
        "force_evaluations": 0,
        "initial_state_equal": True,
        "lanes": lanes,
        "audit_runtime": audit_payload["runtime"],
    }
    output = Path(manifest["paths"]["output_root"]) / "zero_step_audit.json"
    atomic_write_json(output, payload)
    return payload


def _state_from_snapshot(snapshot: Any, masses: np.ndarray) -> NBodyState:
    return rebound_state_from_sim(snapshot, masses)


def _energy(state: NBodyState) -> dict[str, float]:
    return float64_energy(
        state.masses,
        state.positions,
        state.velocities,
        gravitational_constant=G_SI,
        speed_of_light=299792458.0,
        coefficient_scale=1.0,
    )


def _state_rows(
    snapshot: Any,
    manifest: dict[str, Any],
    lane_key: str,
    sample_index: int,
    step_count: int,
) -> list[dict[str, Any]]:
    lane = manifest["lane_contracts"][lane_key]
    fingerprint = lane["configuration_fingerprint"]
    identity = artifact_identity(manifest, lane_key)
    rows = []
    for body_index, body_name in enumerate(BODY_NAMES):
        particle = snapshot.particles[body_index]
        if lane_key == "T":
            variation = snapshot.particles[10 + body_index]
            variation_values: list[Any] = [
                0,
                float(variation.x),
                float(variation.y),
                float(variation.z),
                float(variation.vx),
                float(variation.vy),
                float(variation.vz),
            ]
        else:
            variation_values = ["", "", "", "", "", "", ""]
        rows.append(
            dict(
                zip(
                    STATE_FIELDS,
                    [
                        manifest["telemetry_contract"]["schema_version"],
                        fingerprint,
                        lane["id"],
                        identity,
                        sample_index,
                        step_count,
                        float(snapshot.t),
                        float(snapshot.t) / JULIAN_YEAR_S,
                        body_index,
                        body_name,
                        float(particle.m),
                        float(particle.x),
                        float(particle.y),
                        float(particle.z),
                        float(particle.vx),
                        float(particle.vy),
                        float(particle.vz),
                        *variation_values,
                    ],
                )
            )
        )
    return rows


def _tangent_values(snapshot: Any, lane_key: str) -> tuple[Any, Any]:
    if lane_key != "T":
        return "", ""
    values = []
    for particle in snapshot.particles[10:20]:
        values.extend(
            (
                particle.x / AU_M,
                particle.y / AU_M,
                particle.z / AU_M,
                particle.vx / VELOCITY_SCALE,
                particle.vy / VELOCITY_SCALE,
                particle.vz / VELOCITY_SCALE,
            )
        )
    norm = float(np.linalg.norm(np.asarray(values, dtype=np.float64)))
    require(math.isfinite(norm) and norm > 0.0, "Lane T tangent norm is invalid.")
    return norm, math.log(norm)


def _progress_row(
    sim: Any,
    snapshot: Any,
    backend: CBackend,
    state: NBodyState,
    manifest: dict[str, Any],
    lane_key: str,
    sample_index: int,
    step_count: int,
    energy_reference: float,
    angular_reference: float,
    sync_before: int,
) -> dict[str, Any]:
    lane = manifest["lane_contracts"][lane_key]
    whfast = sim.ri_whfast
    energy = _energy(state)
    angular = total_angular_momentum_vector(state)
    angular_norm = float(np.linalg.norm(angular))
    tangent_norm, tangent_log = _tangent_values(snapshot, lane_key)
    if lane_key == "T":
        megno: Any = float(sim.megno())
        lcn: Any = float(sim.lyapunov())
        require(math.isfinite(megno) and math.isfinite(lcn), "Lane T chaos diagnostic is nonfinite.")
    else:
        megno = ""
        lcn = ""
    stats = backend.stats(sim)
    sync_after = int(whfast.is_synchronized)
    require(sync_after == sync_before, "Scientific sampling changed the live synchronization state.")
    values = [
        manifest["telemetry_contract"]["schema_version"],
        lane["configuration_fingerprint"],
        lane["id"],
        artifact_identity(manifest, lane_key),
        sample_index,
        step_count,
        step_count,
        float(sim.t),
        float(sim.t) / JULIAN_YEAR_S,
        float(sim.dt),
        float(sim.dt_last_done),
        int(sim.steps_done),
        str(sim.integrator),
        str(whfast.coordinates),
        str(whfast.kernel),
        int(whfast.corrector),
        int(whfast.corrector2),
        int(whfast.safe_mode),
        int(whfast.keep_unsynchronized),
        int(whfast.recalculate_coordinates_this_timestep),
        sync_before,
        sync_after,
        1,
        int(sim.N_real),
        int(sim.N_var),
        int(sim.N_var_config),
        megno,
        lcn,
        tangent_norm,
        tangent_log,
        energy["newtonian"],
        energy["gr_potential"],
        energy["corrected"],
        (energy["corrected"] - energy_reference) / abs(energy_reference),
        float(angular[0]),
        float(angular[1]),
        float(angular[2]),
        angular_norm,
        (angular_norm - angular_reference) / abs(angular_reference),
        int(stats["callback_invocations"]),
        int(stats["nonfinite_result_count"]),
    ]
    return dict(zip(PROGRESS_FIELDS, values))


def _diagnostic_snapshot(sim: Any) -> Any:
    live_before = np.asarray(
        [[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles],
        dtype=np.float64,
    )
    sync_before = int(sim.ri_whfast.is_synchronized)
    steps_before = int(sim.steps_done)
    time_before = float(sim.t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        snapshot = sim.copy()
    live_after = np.asarray(
        [[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles],
        dtype=np.float64,
    )
    require(np.array_equal(live_before, live_after), "Diagnostic synchronization mutated live particles.")
    require(int(sim.ri_whfast.is_synchronized) == sync_before, "Diagnostic synchronization changed the live map.")
    require(int(sim.steps_done) == steps_before and float(sim.t) == time_before, "Diagnostic synchronization advanced the live lane.")
    return snapshot


def run_lane(manifest_path: Path, lane_key: str) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    require(lane_key in {"P", "T"}, "Only Lane P or Lane T is authorized.")
    audit_before = audit(manifest_path)
    require(
        audit_before["available_disk_bytes"]
        >= manifest["runtime_safeguards"]["disk_required_bytes"],
        "Available disk is below the preregistered requirement.",
    )
    paths = lane_paths(manifest, lane_key)
    require(not paths["directory"].exists(), f"Collision-safe lane exists: {paths['directory']}")
    paths["directory"].mkdir(parents=True)
    lane = manifest["lane_contracts"][lane_key]
    _event(paths["events"], f"START lane={lane_key} command={' '.join(sys.argv)}")
    print(json.dumps({"lane": lane_key, "configuration": lane_payload(manifest, lane_key), "fingerprint": lane["configuration_fingerprint"], "expected_steps": manifest["common_physical_contract"]["total_steps"]}, indent=2), flush=True)
    rebound, sim, backend, bodies, state0, construction = _build_lane(manifest, lane_key)
    configure_rebound_simulationarchive(
        sim,
        paths["archive"],
        interval_s=manifest["common_physical_contract"]["archive_cadence_years"]
        * JULIAN_YEAR_S,
        delete_existing=True,
    )
    reference_energy = _energy(state0)["corrected"]
    reference_angular = float(np.linalg.norm(total_angular_momentum_vector(state0)))
    common = manifest["common_physical_contract"]
    start = time.perf_counter()
    progress_rows = 0
    state_rows = 0
    try:
        with (
            paths["progress_partial"].open("w", newline="", encoding="utf-8") as progress_handle,
            paths["state_partial"].open("w", newline="", encoding="utf-8") as state_handle,
        ):
            progress_writer = csv.DictWriter(progress_handle, fieldnames=PROGRESS_FIELDS, lineterminator="\n")
            state_writer = csv.DictWriter(state_handle, fieldnames=STATE_FIELDS, lineterminator="\n")
            progress_writer.writeheader()
            state_writer.writeheader()
            for sample_index in range(common["scientific_samples"]):
                step_count = sample_index * common["steps_per_scientific_sample"]
                target_seconds = step_count * common["step_seconds"]
                if sample_index:
                    sim.integrate(target_seconds, exact_finish_time=common["exact_finish_time"])
                require(float(sim.t) == target_seconds, "Lane missed an exact integer target.")
                require(int(sim.steps_done) == step_count, "Lane step count differs from target.")
                if sample_index:
                    require(float(sim.dt_last_done) == common["step_seconds"], "A shortened endpoint step occurred.")
                _validate_settings(sim, manifest, lane_key)
                sync_before = int(sim.ri_whfast.is_synchronized)
                require(sync_before == (1 if sample_index == 0 else 0), "Unexpected live-map synchronization state.")
                snapshot = _diagnostic_snapshot(sim)
                state = _state_from_snapshot(snapshot, state0.masses)
                require(np.all(np.isfinite(state.positions)) and np.all(np.isfinite(state.velocities)), "Nonfinite physical state.")
                sample_state_rows = _state_rows(snapshot, manifest, lane_key, sample_index, step_count)
                progress = _progress_row(
                    sim,
                    snapshot,
                    backend,
                    state,
                    manifest,
                    lane_key,
                    sample_index,
                    step_count,
                    reference_energy,
                    reference_angular,
                    sync_before,
                )
                progress_writer.writerow(progress)
                state_writer.writerows(sample_state_rows)
                progress_handle.flush()
                state_handle.flush()
                os.fsync(progress_handle.fileno())
                os.fsync(state_handle.fileno())
                progress_rows += 1
                state_rows += len(sample_state_rows)
                atomic_write_json(
                    paths["status"],
                    {
                        "schema_version": 1,
                        "state": "RUNNING",
                        "lane": lane_key,
                        "lane_id": lane["id"],
                        "configuration_fingerprint": lane["configuration_fingerprint"],
                        "manifest_sha256": sha256_file(manifest_path),
                        "sample_index": sample_index,
                        "step_count": step_count,
                        "time_years": float(sim.t) / JULIAN_YEAR_S,
                        "callback_invocations": progress["callback_invocations"],
                        "nonfinite_result_count": progress["nonfinite_result_count"],
                    },
                )
                print(f"[step3f1] lane={lane_key} sample={sample_index + 1}/101 steps={step_count}", flush=True)
        os.replace(paths["progress_partial"], paths["progress"])
        os.replace(paths["state_partial"], paths["state"])
        elapsed = time.perf_counter() - start
        stats = backend.stats(sim)
        require(progress_rows == common["scientific_samples"], "Progress count mismatch.")
        require(state_rows == common["state_rows"], "State row count mismatch.")
        require(int(stats["callback_invocations"]) == lane["expected_callback_invocations"], "Callback accounting mismatch.")
        require(int(stats["nonfinite_result_count"]) == 0, "Nonfinite callback result.")
        require(elapsed <= manifest["runtime_safeguards"]["lane_runtime_ceiling_seconds"][lane_key], "Lane exceeded the preregistered runtime ceiling.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            archive = rebound.Simulationarchive(str(paths["archive"]))
        require(len(archive) == common["archive_snapshots"], "Archive snapshot count mismatch.")
        archive_times = [float(archive[index].t) / JULIAN_YEAR_S for index in range(len(archive))]
        require(archive_times == [1000.0 * index for index in range(11)], "Archive times changed.")
        atomic_write_json(
            paths["status"],
            {
                "schema_version": 1,
                "state": "COMPLETED",
                "lane": lane_key,
                "lane_id": lane["id"],
                "configuration_fingerprint": lane["configuration_fingerprint"],
                "manifest_sha256": sha256_file(manifest_path),
                "samples": progress_rows,
                "state_rows": state_rows,
                "steps": int(sim.steps_done),
                "time_years": float(sim.t) / JULIAN_YEAR_S,
                "callback_invocations": int(stats["callback_invocations"]),
                "nonfinite_result_count": int(stats["nonfinite_result_count"]),
            },
        )
        _event(paths["events"], f"COMPLETE lane={lane_key} runtime_seconds={elapsed:.9f}")
        summary = {
            "schema_version": 1,
            "status": "COMPLETED",
            "lane": lane_key,
            "lane_id": lane["id"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "configuration": lane_payload(manifest, lane_key),
            "configuration_fingerprint": lane["configuration_fingerprint"],
            "construction": construction,
            "final_settings": _settings(sim),
            "runtime_seconds": elapsed,
            "throughput_years_per_wall_second": common["duration_years"] / elapsed,
            "scientific_samples": progress_rows,
            "state_rows": state_rows,
            "steps": int(sim.steps_done),
            "archive_snapshots": len(archive),
            "archive_times_years": archive_times,
            "callback_stats": stats,
            "hot_path": backend.hot_path_proof(sim),
            "command": sys.argv,
            "provenance": {
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "git_head": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "runtime": audit_before["runtime"],
            },
        }
        inventory = {}
        for name in ("progress", "state", "archive", "status", "events"):
            path = paths[name]
            inventory[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        summary["artifact_inventory"] = inventory
        atomic_write_json(paths["summary"], summary)
        return summary
    except Exception as exc:
        atomic_write_json(
            paths["status"],
            {
                "schema_version": 1,
                "state": "FAILED",
                "lane": lane_key,
                "failure": str(exc),
                "manifest_sha256": sha256_file(manifest_path),
            },
        )
        _event(paths["events"], f"FAILED lane={lane_key} error={exc}")
        raise


def _read_main_sample(paths: dict[str, Path], sample_index: int) -> tuple[list[dict[str, str]], dict[str, str]]:
    with paths["state"].open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["sample_index"]) == sample_index]
    with paths["progress"].open(newline="", encoding="utf-8") as handle:
        progress = next(row for row in csv.DictReader(handle) if int(row["sample_index"]) == sample_index)
    require(len(rows) == 10, "Restart target state group is incomplete.")
    return rows, progress


def restart_check(manifest_path: Path, lane_key: str) -> dict[str, Any]:
    manifest = load_json(manifest_path, "Manifest 20")
    validate_manifest(manifest)
    require(lane_key in {"P", "T"}, "Only Lane P or Lane T is authorized.")
    audit_before = audit(manifest_path)
    paths = lane_paths(manifest, lane_key)
    require(paths["summary"].is_file(), "Completed lane is required before restart check.")
    archive_before = sha256_file(paths["archive"])
    rebound = optional_import_module("rebound")
    require(rebound is not None, "REBOUND is unavailable.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        archive = rebound.Simulationarchive(str(paths["archive"]))
        matches = [archive[index] for index in range(len(archive)) if float(archive[index].t) == 1000.0 * JULIAN_YEAR_S]
    require(len(matches) == 1, "The 1000-year restart checkpoint is unavailable.")
    sim = matches[0]
    _validate_settings(sim, manifest, lane_key)
    require(int(sim.ri_whfast.is_synchronized) == 0, "Archive did not preserve the unsynchronized live map.")
    backend = load_c_backend()
    backend.attach(sim, coefficient_scale=1.0, include_central_response=True)
    require(backend.hot_path_proof(sim)["addresses_match"] is True, "Restart callback attachment failed.")
    start_steps = int(sim.steps_done)
    target_seconds = 1100.0 * JULIAN_YEAR_S
    sim.integrate(target_seconds, exact_finish_time=1)
    require(float(sim.t) == target_seconds, "Restart branch missed the exact target.")
    require(int(sim.steps_done) - start_steps == 146100, "Restart branch step count changed.")
    require(float(sim.dt_last_done) == 21600.0, "Restart branch shortened its final step.")
    require(int(sim.ri_whfast.is_synchronized) == 0, "Restart branch lost its internal map.")
    snapshot = _diagnostic_snapshot(sim)
    main_rows, main_progress = _read_main_sample(paths, 11)
    branch_rows = _state_rows(snapshot, manifest, lane_key, 11, 11 * 146100)
    physical_fields = ("mass_kg", "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s")
    variation_fields = (
        "variation_x_m",
        "variation_y_m",
        "variation_z_m",
        "variation_vx_m_per_s",
        "variation_vy_m_per_s",
        "variation_vz_m_per_s",
    )
    physical_exact = all(
        float(main[field]) == float(branch[field])
        for main, branch in zip(main_rows, branch_rows)
        for field in physical_fields
    )
    variation_exact = lane_key == "P" or all(
        float(main[field]) == float(branch[field])
        for main, branch in zip(main_rows, branch_rows)
        for field in variation_fields
    )
    if lane_key == "T":
        megno_exact = float(main_progress["megno"]) == float(sim.megno())
        lcn_exact = float(main_progress["lcn_1_per_year"]) == float(sim.lyapunov())
    else:
        megno_exact = True
        lcn_exact = True
    stats = backend.stats(sim)
    expected_callbacks = manifest["restart_contract"]["expected_branch_callbacks"][lane_key]
    callback_accounting_passed = int(stats["callback_invocations"]) == expected_callbacks
    require(int(stats["nonfinite_result_count"]) == 0, "Restart callback produced a nonfinite result.")
    require(physical_exact and variation_exact and megno_exact and lcn_exact, "Restart closure is not exact.")
    require(sha256_file(paths["archive"]) == archive_before, "Restart check modified the archive.")
    payload = {
        "schema_version": 1,
        "status": "PASS" if callback_accounting_passed else "FAIL",
        "lane": lane_key,
        "lane_id": manifest["lane_contracts"][lane_key]["id"],
        "manifest_sha256": sha256_file(manifest_path),
        "fresh_process_pid": os.getpid(),
        "checkpoint_years": 1000,
        "target_years": 1100,
        "continuation_steps": 146100,
        "archive_is_synchronized_at_restore": 0,
        "archive_sha256_before": archive_before,
        "archive_sha256_after": sha256_file(paths["archive"]),
        "callback_reattached": True,
        "callback_invocations": int(stats["callback_invocations"]),
        "expected_callback_invocations": expected_callbacks,
        "callback_accounting_passed": callback_accounting_passed,
        "nonfinite_result_count": int(stats["nonfinite_result_count"]),
        "physical_state_exact_float64": physical_exact,
        "variation_state_exact_float64": variation_exact,
        "megno_exact_float64": megno_exact,
        "lcn_exact_float64": lcn_exact,
        "settings": _settings(sim),
        "reference_artifacts_verified": len(audit_before["reference_artifacts"]),
    }
    atomic_write_json(paths["restart_check"], payload)
    return payload
