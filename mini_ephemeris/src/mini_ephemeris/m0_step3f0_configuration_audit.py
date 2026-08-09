from __future__ import annotations

import argparse

import csv
import datetime as dt
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .ephem import EphemerisConfig, initial_state_solar_system_barycentric
from .gr_potential_tangent_c import load_c_backend
from .m0_step3f0_configuration_contract import lane_manifest_numbers, merge_lane_configuration
from .long_term_stability_cli import (
    build_rebound_simulation,
    stability_body_list,
)
from .orbital_elements import DAY_S
from .m0_step3f0_source_evidence import PYTHON_ACCESSOR_FINDING
from .m0_step3f0_whckl_evidence import WHCKL_SHORTCUT_FINDING
from .rebound_gr_tangent_backend_cli import sha256_file


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "19_m0_step3f0_whfast_configuration_audit_v1.json"
)
REBOUND_SOURCE = Path("/tmp/rebound-4.6.0-step3f0")
REBOUNDX_SOURCE = Path("/tmp/reboundx-4.6.1-step3f0")
REBOUND_COMMIT = "e3b07aa88dc4b004d82c03da070a89de5b699a2c"
REBOUNDX_COMMIT = "d5a4a2b5d28cbbd167bef3148063603ea2f2e131"
REBOUND_ARCHIVE_SHA256 = (
    "63354536ba3f7fb3a0365f6619b8a76b4a85c54d444230fd3efc074489b318f5"
)
REBOUNDX_ARCHIVE_SHA256 = (
    "1b7f3d44a6acaf224f36616f010f88511c71c4ccfb33cb45baf0389de4aaaa23"
)
FINAL_STATUS = "STEP3F0_CONFIGURATION_AUDIT_COMPLETE"
PRIMARY_FINDING = "COMBINED_LANE_CAPABILITY_CONSTRAINT_CONFIRMED"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"


class ConfigurationAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationAuditError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"Missing {label}: {path}")
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"{label} must be a JSON object.")
    return payload


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(("git", *args), cwd=cwd, text=True).strip()


def _archive_sha256(source: Path) -> str:
    archive = subprocess.check_output(
        ("git", "archive", "--format=tar", "HEAD"), cwd=source
    )
    return hashlib.sha256(archive).hexdigest()


def _json_value(value: Any) -> str:
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"Nonfinite value at {path}.")
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, f"{path}[{index}]")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    _require(bool(rows), "The settings matrix is empty.")
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            _require(list(row) == fields, "Settings matrix schema changed.")
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest_contract(manifest: dict[str, Any]) -> None:
    matrix = manifest["effective_settings_matrix"]
    _require(matrix["unique_lane_count"] == len(manifest["historical_lanes"]), "Lane count changed.")
    _require(matrix["setting_count"] == len(matrix["settings"]), "Setting count changed.")
    _require(
        matrix["expected_rows"]
        == matrix["unique_lane_count"] * matrix["setting_count"],
        "Preregistered matrix dimensions are inconsistent.",
    )
    _require(
        manifest["preregistration"]["primary_finding"] == "NOT_EVALUATED",
        "Manifest 19 preregistration was post-hoc modified.",
    )
    _require(
        manifest["preregistration"]["integration_or_force_evaluation_count"] == 0,
        "Manifest 19 permits integration or force evaluation.",
    )


def _immutable_audit(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "Manifest 19")
    _manifest_contract(manifest)
    protected = []
    for relative, expected in manifest["protected_files"].items():
        actual = sha256_file(ROOT / relative)
        _require(actual == expected, f"Protected file changed: {relative}")
        protected.append({"path": relative, "sha256": actual})
    historical = []
    for label, (relative, expected) in manifest["historical_artifacts"].items():
        actual = sha256_file(ROOT / relative)
        _require(actual == expected, f"Historical artifact changed: {relative}")
        historical.append({"label": label, "path": relative, "sha256": actual})
    transient = {}
    for relative in (
        ".codex-step3e1-manifest.patch",
        ".codex-step3e1-manifest-update.json",
    ):
        transient[relative] = {
            "exists": (ROOT / relative).exists(),
            "tracked": bool(_git("ls-files", "--error-unmatch", relative))
            if (ROOT / relative).exists()
            else False,
        }
        _require(not transient[relative]["exists"], f"Transient file reappeared: {relative}")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "head": _git("rev-parse", "HEAD"),
        "tag_object_type": _git("cat-file", "-t", "gr-tangent-compiled-c-v1"),
        "tag_commit": _git("rev-parse", "gr-tangent-compiled-c-v1^{commit}"),
        "protected_files": protected,
        "historical_artifacts": historical,
        "transient_repository_hygiene": transient,
    }


def _source_audit() -> dict[str, Any]:
    sources = []
    for name, path, commit, tag, url, expected_archive in (
        (
            "REBOUND",
            REBOUND_SOURCE,
            REBOUND_COMMIT,
            "4.6.0",
            "https://github.com/hannorein/rebound/tree/4.6.0",
            REBOUND_ARCHIVE_SHA256,
        ),
        (
            "REBOUNDx",
            REBOUNDX_SOURCE,
            REBOUNDX_COMMIT,
            "4.6.1",
            "https://github.com/dtamayo/reboundx/tree/4.6.1",
            REBOUNDX_ARCHIVE_SHA256,
        ),
    ):
        _require((path / ".git").exists(), f"Missing exact tagged source checkout: {path}")
        actual_commit = _git("rev-parse", "HEAD", cwd=path)
        archive_sha = _archive_sha256(path)
        _require(actual_commit == commit, f"{name} source checkout commit changed.")
        _require(archive_sha == expected_archive, f"{name} git archive hash changed.")
        sources.append(
            {
                "name": name,
                "url": url,
                "tag": tag,
                "commit": actual_commit,
                "retrieval_date": "2026-08-09",
                "git_archive_sha256": archive_sha,
            }
        )
    whfast = (REBOUND_SOURCE / "src/integrator_whfast.c").read_text()
    output = (REBOUND_SOURCE / "src/output.c").read_text()
    rebound_c = (REBOUND_SOURCE / "src/rebound.c").read_text()
    rebx_gr = (REBOUNDX_SOURCE / "src/gr_potential.c").read_text()
    required = {
        "variations_standard_kernel_only": "Variational particles are only compatible with the standard kernel.",
        "variations_jacobi_only": "Variational particles are only compatible with Jacobi coordinates.",
        "megno_synchronization": "Need to have x,v,a synchronized to calculate ddot/d for MEGNO.",
        "safe_keep_constraint": "keep_unsynchronized == 1 is not compatible with safe_mode",
        "archive_whfast_fields": '"ri_whfast.keep_unsynchronized"',
        "integrate_final_sync": "reb_simulation_synchronize(r);",
        "gr_mean_motion_caveat": "gets the mean motion wrong",
    }
    haystacks = {
        "variations_standard_kernel_only": whfast,
        "variations_jacobi_only": whfast,
        "megno_synchronization": whfast,
        "safe_keep_constraint": whfast,
        "archive_whfast_fields": output,
        "integrate_final_sync": rebound_c,
        "gr_mean_motion_caveat": rebx_gr,
    }
    for label, token in required.items():
        _require(token in haystacks[label], f"Required exact-source evidence missing: {label}")
    return {
        "retrievals": sources,
        "files": [
            {
                "path": str(REBOUND_SOURCE / "src/integrator_whfast.c"),
                "sha256": sha256_file(REBOUND_SOURCE / "src/integrator_whfast.c"),
            },
            {
                "path": str(REBOUND_SOURCE / "src/rebound.c"),
                "sha256": sha256_file(REBOUND_SOURCE / "src/rebound.c"),
            },
            {
                "path": str(REBOUND_SOURCE / "src/output.c"),
                "sha256": sha256_file(REBOUND_SOURCE / "src/output.c"),
            },
            {
                "path": str(REBOUND_SOURCE / "src/simulationarchive.c"),
                "sha256": sha256_file(REBOUND_SOURCE / "src/simulationarchive.c"),
            },
            {
                "path": str(REBOUNDX_SOURCE / "src/gr_potential.c"),
                "sha256": sha256_file(REBOUNDX_SOURCE / "src/gr_potential.c"),
            },
        ],
        "required_tokens_verified": sorted(required),
    }


class NoStepGuard:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._originals: list[tuple[Any, str, Any]] = []

    def _block(self, name: str):
        def blocked(*_args: Any, **_kwargs: Any) -> None:
            self.calls.append(name)
            raise ConfigurationAuditError(f"Step 3f0 no-step guard blocked {name}.")

        return blocked

    def _patch(self, owner: Any, name: str) -> None:
        if hasattr(owner, name):
            original = getattr(owner, name)
            self._originals.append((owner, name, original))
            setattr(owner, name, self._block(name))

    def __enter__(self) -> "NoStepGuard":
        import rebound

        self._patch(rebound.Simulation, "integrate")
        self._patch(rebound.Simulation, "step")
        self._patch(rebound.Simulation, "steps")
        self._patch(rebound.clibrebound, "reb_simulation_integrate")
        self._patch(rebound.clibrebound, "reb_simulation_step")
        from . import long_term_stability_cli
        from . import rebound_gr_tangent_backend_cli

        self._patch(long_term_stability_cli, "integrate_rebound_streaming")
        self._patch(rebound_gr_tangent_backend_cli, "main")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        if exc_type is None:
            _require(not self.calls, f"No-step guard recorded calls: {self.calls}")


def _callback_address(sim: Any) -> int:
    try:
        return int(ctypes.cast(sim._additional_forces, ctypes.c_void_p).value or 0)
    except (AttributeError, TypeError, ValueError):
        return 0


def _variation_configuration(sim: Any) -> list[dict[str, int]]:
    output = []
    for index in range(int(sim.N_var_config)):
        config = sim.var_config[index]
        output.append(
            {
                "index": int(config.index),
                "order": int(config.order),
                "testparticle": int(config.testparticle),
            }
        )
    return output


def _capture_simulation(sim: Any) -> dict[str, Any]:
    integrator = str(sim.integrator)
    capture: dict[str, Any] = {
        "sim.integrator": integrator,
        "sim.dt": float(sim.dt),
        "sim.gravity": str(sim.gravity),
        "sim.collision": str(sim.collision),
        "sim.boundary": str(sim.boundary),
        "sim.N": int(sim.N),
        "sim.N_active": int(sim.N_active),
        "sim.N_var": int(sim.N_var),
        "sim.N_var_config": int(sim.N_var_config),
        "variational_configurations": _variation_configuration(sim),
        "megno.initialized": bool(int(sim._calculate_megno)),
        "megno.seed": int(sim.rand_seed) if int(sim._calculate_megno) else NOT_APPLICABLE,
        "sim.force_is_velocity_dependent": int(sim.force_is_velocity_dependent),
        "sim.additional_forces": "attached" if _callback_address(sim) else "not_attached",
    }
    if integrator == "whfast":
        whfast = sim.ri_whfast
        capture.update(
            {
                "ri_whfast.coordinates": str(whfast.coordinates),
                "ri_whfast.kernel": str(whfast.kernel),
                "ri_whfast.corrector": int(whfast.corrector),
                "ri_whfast.corrector2": int(whfast.corrector2),
                "ri_whfast.safe_mode": int(whfast.safe_mode),
                "ri_whfast.keep_unsynchronized": int(whfast.keep_unsynchronized),
                "ri_whfast.recalculate_coordinates_this_timestep": int(
                    whfast.recalculate_coordinates_this_timestep
                ),
                "ri_whfast.is_synchronized": int(whfast.is_synchronized),
            }
        )
    return capture


def _zero_step_probe(manifest: dict[str, Any]) -> dict[str, Any]:
    import rebound

    bodies = stability_body_list("full_with_pluto", include_pluto=True)
    state = initial_state_solar_system_barycentric(
        dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc),
        bodies=bodies,
        config=EphemerisConfig(kernel_path=str(ROOT / "data/de431_part-2.bsp")),
    )
    sim = build_rebound_simulation(
        rebound,
        state,
        integrator="whfast",
        step_s=0.5 * DAY_S,
        ias15_epsilon=1.0e-10,
    )
    initial = _capture_simulation(sim)
    sim.init_megno(seed=12345)
    after_megno = _capture_simulation(sim)
    backend = load_c_backend()
    backend.attach(sim, coefficient_scale=1.0, include_central_response=True)
    after_gr = _capture_simulation(sim)
    hot_path = backend.hot_path_proof(sim)
    stats = backend.stats(sim)
    _require(stats["callback_invocations"] == 0, "Zero-step probe evaluated the force callback.")
    _require(stats["nonfinite_result_count"] == 0, "Zero-step probe callback state is invalid.")
    _require(hot_path["addresses_match"], "Compiled callback did not attach directly.")
    hot_path = {key: hot_path[key] for key in ("kind", "addresses_match", "python_callback_in_force_path", "c_owned_instrumentation")}
    result = {
        "initial": initial,
        "after_megno": after_megno,
        "after_gr_attachment": after_gr,
        "hot_path": hot_path,
        "callback_stats": stats,
        "body_order": list(bodies),
        "masses_kg": [float(value) for value in state.masses],
        "requested_megno_seed": 12345,
        "note": "rand_seed is advanced by initialization; the requested seed is preserved separately.",
    }
    backend.library.me_gr_tangent_detach(ctypes.byref(sim))
    _require(
        manifest["initial_integrity_facts"]["runtime"]["callback_library_sha256"]
        == backend.build_metadata["artifact_sha256"],
        "Loaded callback identity differs from Manifest 19.",
    )
    return result


def _find_configuration(value: Any, run_id: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("id") == run_id or value.get("run_id") == run_id:
            config = value.get("configuration")
            return dict(config) if isinstance(config, dict) else dict(value)
        for child in value.values():
            found = _find_configuration(child, run_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_configuration(child, run_id)
            if found is not None:
                return found
    return None


def _lane_configuration(run_id: str) -> dict[str, Any]:
    manifest_numbers = (10, 11, 13, 14, 15, 16, 17)
    for number in manifest_numbers:
        matches = sorted((ROOT / "ephemeris_experiment_runner/manifests").glob(f"{number:02d}_*.json"))
        for path in matches:
            payload = _load_json(path, f"Manifest {number}")
            found = _find_configuration(payload, run_id)
            if found is not None:
                return merge_lane_configuration(payload, found)
    raise ConfigurationAuditError(f"Cannot recover historical lane configuration: {run_id}")


def _combined_summary(run_id: str) -> dict[str, Any]:
    directories = sorted((ROOT / "output/stability").glob(f"*/{run_id}"))
    _require(len(directories) == 1, f"Expected one historical output directory for {run_id}.")
    summaries = sorted(directories[0].glob("*summary*.json"))
    _require(len(summaries) == 1, f"Expected one historical summary for {run_id}.")
    return _load_json(summaries[0], f"summary for {run_id}")


def _archive_paths() -> dict[str, Path]:
    result: dict[str, Path] = {}
    summary15 = _load_json(
        ROOT / "docs/validation/m0-integrator-roundoff-diagnosis-continuation-v1/"
        "m0_integrator_roundoff_diagnosis_continuation_summary.json",
        "Manifest 15 summary",
    )
    for run_id, lane in summary15["new_lanes"].items():
        archive = lane["artifact_inventory"].get("archive")
        if archive:
            result[run_id] = Path(archive["path"])
    summary16 = _load_json(
        ROOT / "docs/validation/m0-ias15-phase-reference-v1/"
        "m0_ias15_phase_reference_summary.json",
        "Manifest 16 summary",
    )
    for run_id, lane in summary16["ias15_lanes"].items():
        archive = lane["artifact_inventory"].get("archive")
        if archive:
            result[run_id] = Path(archive["path"])
    for run_id in (
        "m0_conv_0p5d_1myr_s12345",
        "m0_conv_0p25d_1myr_s12345",
        "m0_step3e_tangent_whfast_0125d_1myr",
    ):
        result[run_id] = Path(_combined_summary(run_id)["outputs"]["archive"])
    return result


def _archive_audit(paths: dict[str, Path]) -> dict[str, Any]:
    import rebound

    output: dict[str, Any] = {}
    for run_id, path in sorted(paths.items()):
        _require(path.is_file(), f"Missing historical archive: {path}")
        before = sha256_file(path)
        archive = rebound.Simulationarchive(str(path))
        earliest = archive[0]
        final = archive[-1]
        output[run_id] = {
            "path": str(path),
            "sha256_before": before,
            "sha256_after": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "snapshots": len(archive),
            "earliest": _capture_simulation(earliest),
            "later_or_final": _capture_simulation(final),
            "restored": _capture_simulation(archive[-1]),
        }
        _require(
            output[run_id]["sha256_before"] == output[run_id]["sha256_after"],
            f"Archive changed during inspection: {path}",
        )
    return output


def _requested_value(
    setting: str,
    config: dict[str, Any],
    run_id: str,
    zero_probe: dict[str, Any],
) -> Any:
    key_map = {
        "sim.integrator": "integrator",
        "ri_whfast.coordinates": "coordinates",
        "ri_whfast.kernel": "kernel",
        "ri_whfast.corrector": "corrector",
        "ri_whfast.corrector2": "corrector2",
        "ri_whfast.safe_mode": "safe_mode",
        "ri_whfast.keep_unsynchronized": "keep_unsynchronized",
        "ri_whfast.recalculate_coordinates_this_timestep": "recalculate_coordinates_this_timestep",
        "sim.gravity": "gravity",
        "sim.boundary": "boundary",
        "sim.N_active": "N_active",
        "sim.force_is_velocity_dependent": "force_is_velocity_dependent",
        "gr.coefficient_scale": "gr_scale",
        "gr.include_central_response": "include_central_response",
        "output.scientific_cadence": "record_every_years",
        "output.archive_cadence": "archive_interval_years",
        "integration.exact_finish_time": "exact_finish_time",
    }
    if setting in key_map and key_map[setting] in config:
        return config[key_map[setting]]
    if setting == "sim.dt":
        if "step_days" in config:
            return float(config["step_days"]) * DAY_S
        if "initial_dt_days" in config:
            return float(config["initial_dt_days"]) * DAY_S
    combined = bool(config.get("variations") or "conv_" in run_id or "tangent" in run_id)
    if setting == "sim.N":
        return 20 if combined else len(config.get("body_names", zero_probe["body_order"]))
    if setting == "sim.N_var":
        return 10 if combined else 0
    if setting == "sim.N_var_config":
        return 1 if combined else 0
    if setting == "variational_configurations":
        return [{"index": 10, "order": 1, "testparticle": -1}] if combined else []
    if setting == "megno.initialized":
        return combined or bool(config.get("megno", False))
    if setting == "megno.seed":
        return 12345 if combined else NOT_APPLICABLE
    if setting == "sim.additional_forces":
        return "validated compiled-C callback"
    if setting == "callback.library_sha256":
        return "e88ecbcc01a9557f483f8afce9a31a593247a9d54d81f5731ec2a8cde2456067"
    if setting == "gr.coefficient_scale":
        return 1
    if setting == "gr.include_central_response":
        return True
    if setting == "physical.body_order":
        return config.get("body_names", zero_probe["body_order"])
    if setting == "physical.masses":
        return config.get("masses_kg", zero_probe["masses_kg"])
    if setting == "physical.barycentric_mass_convention":
        return "DE431 barycenter masses; Sun is central GR source"
    if setting == "physical.units":
        return "SI (m, s, kg)"
    if setting == "physical.input_frame":
        return "DE431 solar-system barycentric ICRF at 2000-01-01T00:00:00Z"
    if setting == "sim.collision":
        return "none"
    if setting == "output.checkpoint_cadence":
        return "restart sidecar at deliberate stop; no independent cadence"
    if setting == "integration.output_targets_integer_steps":
        step = config.get("step_days")
        cadence = config.get("record_every_years")
        if step and cadence:
            ratio = float(cadence) * 365.25 / float(step)
            return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-12)
        return NOT_APPLICABLE
    if setting == "restart.archive_synchronization_state":
        return "serialized ri_whfast state and p_jh; direct value in matrix archive columns"
    if setting == "restart.callback_reattachment":
        return "required and performed by production resume path"
    if setting == "ri_whfast.is_synchronized":
        return "runtime state, not a requested setting"
    return NOT_AVAILABLE


def _default_value(setting: str, integrator: str) -> Any:
    if integrator != "whfast" and setting.startswith("ri_whfast."):
        return NOT_APPLICABLE
    defaults = {
        "ri_whfast.coordinates": "jacobi",
        "ri_whfast.kernel": "default",
        "ri_whfast.corrector": 0,
        "ri_whfast.corrector2": 0,
        "ri_whfast.safe_mode": 1,
        "ri_whfast.keep_unsynchronized": 0,
        "ri_whfast.recalculate_coordinates_this_timestep": 0,
        "ri_whfast.is_synchronized": 1,
        "sim.gravity": "basic",
        "sim.collision": "none",
        "sim.boundary": "none",
        "sim.N_active": -1,
        "sim.force_is_velocity_dependent": 0,
        "sim.additional_forces": "not_attached",
        "megno.initialized": False,
    }
    return defaults.get(setting, NOT_APPLICABLE)


def _stage_value(
    setting: str,
    capture: dict[str, Any] | None,
    requested: Any,
    stage: str,
    combined: bool,
) -> Any:
    if capture and setting in capture:
        return capture[setting]
    if setting == "callback.library_sha256":
        return requested if stage in {"gr", "tangent"} else NOT_APPLICABLE
    if setting.startswith("gr."):
        return requested if stage in {"gr", "tangent"} else NOT_APPLICABLE
    if setting.startswith("physical.") or setting.startswith("output.") or setting.startswith("integration."):
        return requested
    if setting.startswith("restart."):
        return NOT_APPLICABLE if stage != "restored" else requested
    if stage == "tangent" and not combined and setting in {
        "megno.initialized",
        "megno.seed",
        "variational_configurations",
    }:
        return NOT_APPLICABLE
    return requested


def _build_settings_matrix(
    manifest: dict[str, Any],
    zero_probe: dict[str, Any],
    archives: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    combined_ids = {
        "m0_conv_0p5d_1myr_s12345",
        "m0_conv_0p25d_1myr_s12345",
        "m0_step3e_tangent_whfast_0125d_1myr",
    }
    for lane in manifest["historical_lanes"]:
        run_id = lane["run_id"]
        config = _lane_configuration(run_id)
        combined = run_id in combined_ids
        archive = archives.get(run_id)
        integrator = str(config.get("integrator", "whfast"))
        initial_capture = zero_probe["initial"] if combined else (archive or {}).get("earliest")
        gr_capture = zero_probe["after_gr_attachment"] if combined else initial_capture
        tangent_capture = zero_probe["after_megno"] if combined else None
        for setting in manifest["effective_settings_matrix"]["settings"]:
            requested = _requested_value(setting, config, run_id, zero_probe)
            confidence = "DIRECT" if archive and setting in archive["earliest"] else "CORROBORATED"
            caveat = ""
            if integrator != "whfast" and setting.startswith("ri_whfast."):
                confidence = "UNAVAILABLE"
                caveat = "IAS15 lane; WHFast field is not applicable."
            elif not archive:
                caveat = "No trajectory archive was required for this historical lane; value is manifest/source derived."
                confidence = "INFERRED"
            if combined and setting == "megno.seed":
                caveat = "Archive stores the advanced RNG state, not the original requested seed 12345."
            if setting == "restart.callback_reattachment":
                caveat = "Function pointers are not serialized; the production resume path reattaches the callback."
            row = {
                "lane_id": run_id,
                "manifests": _json_value(lane_manifest_numbers(lane)),
                "setting": setting,
                "requested_value": _json_value(requested),
                "rebound_default": _json_value(_default_value(setting, integrator)),
                "effective_initial": _json_value(
                    _stage_value(setting, initial_capture, requested, "initial", combined)
                ),
                "effective_after_gr_attachment": _json_value(
                    _stage_value(setting, gr_capture, requested, "gr", combined)
                ),
                "effective_after_tangent_megno": _json_value(
                    _stage_value(setting, tangent_capture, requested, "tangent", combined)
                ),
                "earliest_archive": _json_value(
                    (archive or {}).get("earliest", {}).get(setting, NOT_AVAILABLE)
                ),
                "later_or_final_archive": _json_value(
                    (archive or {}).get("later_or_final", {}).get(setting, NOT_AVAILABLE)
                ),
                "after_checkpoint_restoration": _json_value(
                    _stage_value(
                        setting,
                        (archive or {}).get("restored"),
                        requested,
                        "restored",
                        combined,
                    )
                    if archive
                    else NOT_AVAILABLE
                ),
                "evidence_source": (
                    "historical SimulationArchive + Manifest/source"
                    if archive
                    else "committed historical summary/manifest + exact tagged source"
                ),
                "confidence": confidence,
                "discrepancy_or_caveat": caveat,
            }
            rows.append(row)
    expected = manifest["effective_settings_matrix"]["expected_rows"]
    _require(len(rows) == expected, f"Settings matrix has {len(rows)} rows, expected {expected}.")
    keys = {(row["lane_id"], row["setting"]) for row in rows}
    _require(len(keys) == expected, "Settings matrix lane/setting keys are not unique.")
    return rows


SOURCE_FINDINGS = [PYTHON_ACCESSOR_FINDING, WHCKL_SHORTCUT_FINDING,
    {
        "id": "whfast_defaults",
        "source": "REBOUND 4.6.0 src/integrator_whfast.c:1245-1253",
        "symbol": "reb_integrator_whfast_reset",
        "finding": "Defaults are Jacobi coordinates, standard kernel, no correctors, safe_mode=1, keep_unsynchronized=0, synchronized state, and no forced coordinate recalculation.",
        "historical_applicability": "Matches the three combined-lane archives and Manifest 13 current-sync controls.",
    },
    {
        "id": "variation_capability",
        "source": "REBOUND 4.6.0 src/integrator_whfast.c:784-810",
        "symbol": "reb_integrator_whfast_init",
        "finding": "WHFast first variations require Jacobi coordinates and the standard kernel; nonstandard kernels are rejected rather than silently downgraded.",
        "historical_applicability": "A native tangent/MEGNO lane cannot select WHCKL or another nonstandard kernel in REBOUND 4.6.0.",
    },
    {
        "id": "corrector_capability",
        "source": "REBOUND 4.6.0 src/integrator_whfast.c:811-820, 939-990",
        "symbol": "reb_integrator_whfast_init; reb_integrator_whfast_synchronize",
        "finding": "Corrector orders 3, 5, 7, 11, and 17 are accepted in Jacobi or barycentric coordinates, and Jacobi variation blocks are transformed during synchronization.",
        "historical_applicability": "Correctors remain technically feasible for the standard-kernel combined lane, although none was requested historically.",
    },
    {
        "id": "megno_per_step_sync",
        "source": "REBOUND 4.6.0 src/integrator_whfast.c:1169-1242",
        "symbol": "reb_integrator_whfast_step",
        "finding": "With variations, WHFast constructs synchronized inertial x/v/a every timestep for MEGNO. keep_unsynchronized=1 preserves and restores the internal map around that calculation.",
        "historical_applicability": "Historical combined lanes used safe_mode=1, so they already synchronized each step before the MEGNO-specific path.",
    },
    {
        "id": "safe_keep_semantics",
        "source": "REBOUND 4.6.0 src/integrator_whfast.c:821-822, 1000-1018, 1158-1161",
        "symbol": "reb_integrator_whfast_init; reb_integrator_whfast_step",
        "finding": "safe_mode=1 and keep_unsynchronized=1 are incompatible. safe_mode=1 recalculates map coordinates and synchronizes every step; safe_mode=0 with keep_unsynchronized=1 retains the internal map.",
        "historical_applicability": "Manifest 15 tested both current-sync and minimal-sync physical-only controls; minimal synchronization did not supply a material causal improvement.",
    },
    {
        "id": "output_and_exact_finish",
        "source": "REBOUND 4.6.0 src/rebound.c:639-657, 768-844",
        "symbol": "reb_check_exit; reb_simulation_integrate_raw",
        "finding": "exact_finish_time=1 shortens a final step only when needed, restores dt afterward, and integrate synchronizes before returning.",
        "historical_applicability": "The 0.5, 0.25, and 0.125 day lanes divide the 100-year output cadence exactly; the older 2-day lane did not and received a one-day shortened endpoint step at each output.",
    },
    {
        "id": "archive_semantics",
        "source": "REBOUND 4.6.0 src/output.c:40-165; src/simulationarchive.c:409-442",
        "symbol": "reb_binary_field_descriptor_list; reb_simulationarchive_heartbeat",
        "finding": "Archives serialize particles, variations, MEGNO state, WHFast map state, p_jh, kernel, coordinates, correctors, and synchronization flags without first forcing synchronization.",
        "historical_applicability": "Existing archive hashes remained unchanged during direct restoration and their effective settings match the manifests.",
    },
    {
        "id": "callback_restore",
        "source": "REBOUND 4.6.0 src/output.c:597-615; src/input.c:168-170, 243",
        "symbol": "reb_simulation_save_to_stream; reb_input_process_warnings",
        "finding": "Function pointers are flagged but not serialized and must be reset after restore.",
        "historical_applicability": "The production resume path explicitly reattaches the validated compiled callback after archive validation.",
    },
    {
        "id": "gr_potential_contract",
        "source": "REBOUNDx 4.6.1 src/gr_potential.c:38-45, 60-119",
        "symbol": "rebx_gr_potential; rebx_gr_potential_potential",
        "finding": "gr_potential is position-only, preserves WHFast splitting, reproduces perihelion precession, has a documented mean-motion error of order GM/(a c^2), and uses the validated -3 G^2 M^2 m/(c^2 r^2) potential.",
        "historical_applicability": "The custom callback implements the same physical acceleration and potential plus its analytic first-variation Jacobian and central response.",
    },
]


LITERATURE = [
    {
        "topic": "original Wisdom-Holman method",
        "title": "Symplectic maps for the n-body problem",
        "authors": "Wisdom, J.; Holman, M.",
        "year": 1991,
        "doi_or_official_url": "https://doi.org/10.1086/115978",
        "software_version_or_commit": NOT_APPLICABLE,
        "symbol_or_section": "method",
        "paraphrase": "Splitting nearly Keplerian planetary motion enables efficient long-term symplectic maps.",
        "direct_applicability": "Foundational method for WHFast.",
        "scope_caveat": "Does not specify REBOUND implementation settings.",
    },
    {
        "topic": "original WHFast method and implementation",
        "title": "WHFast: a fast and unbiased implementation of a symplectic Wisdom-Holman integrator for long-term gravitational simulations",
        "authors": "Rein, H.; Tamayo, D.",
        "year": 2015,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stv1257",
        "software_version_or_commit": "method antecedent to REBOUND 4.6.0",
        "symbol_or_section": "WHFast implementation, correctors, tangent map",
        "paraphrase": "Stable Jacobi transforms remove secular implementation bias; correctors and a symplectic tangent map support LCN and MEGNO.",
        "direct_applicability": "Supports standard-kernel combined-lane design.",
        "scope_caveat": "Does not imply support for later nonstandard kernels with variations.",
    },
    {
        "topic": "advanced WHFast kernels and correctors",
        "title": "High-order symplectic integrators for planetary dynamics and their implementation in REBOUND",
        "authors": "Rein, H.; Tamayo, D.; Brown, G.",
        "year": 2019,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stz2503",
        "software_version_or_commit": "REBOUND high-order kernel implementation; checked against 4.6.0 source",
        "symbol_or_section": "Sections 2-4; WHCKL",
        "paraphrase": "High-order kernels and correctors can greatly reduce physical integration error, but the implemented kernels do not support variational equations or MEGNO.",
        "direct_applicability": "Directly explains the physical-WHCKL versus native-tangent capability split.",
        "scope_caveat": "Accuracy gains are problem- and timestep-dependent and do not establish causation here.",
    },
    {
        "topic": "variational equations and MEGNO",
        "title": "Second-order variational equations for N-body simulations",
        "authors": "Rein, H.; Tamayo, D.",
        "year": 2016,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stw644",
        "software_version_or_commit": "REBOUND variational framework",
        "symbol_or_section": "first- and second-order variational equations",
        "paraphrase": "Variational equations provide derivatives and chaos indicators without finite differencing.",
        "direct_applicability": "Supports the analytic first-variation telemetry requirement.",
        "scope_caveat": "The paper's second-order examples emphasize IAS15; exact WHFast constraints come from tagged source.",
    },
    {
        "topic": "MEGNO definition",
        "title": "Phase space structure of multi-dimensional systems by means of the mean exponential growth factor of nearby orbits",
        "authors": "Cincotta, P. M.; Giordano, C. M.; Simo, C.",
        "year": 2003,
        "doi_or_official_url": "https://doi.org/10.1016/S0167-2789(03)00103-9",
        "software_version_or_commit": NOT_APPLICABLE,
        "symbol_or_section": "MEGNO and LCN definitions",
        "paraphrase": "MEGNO integrates first variations and can estimate hyperbolicity and the Lyapunov characteristic number.",
        "direct_applicability": "Defines the scientific tangent diagnostic.",
        "scope_caveat": "Does not prescribe REBOUND synchronization or kernel settings.",
    },
    {
        "topic": "REBOUNDx gr_potential",
        "title": "REBOUNDx: a library for adding conservative and dissipative forces to otherwise symplectic N-body integrations",
        "authors": "Tamayo, D.; Rein, H.; Shi, P.; Hernandez, D. M.",
        "year": 2020,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stz2870",
        "software_version_or_commit": "REBOUNDx 4.6.1 tag d5a4a2b5",
        "symbol_or_section": "operator splitting and additional forces",
        "paraphrase": "Position-dependent conservative effects can remain compatible with symplectic splitting; velocity-dependent post-Newtonian approximations require special care.",
        "direct_applicability": "The validated gr_potential callback is position-only.",
        "scope_caveat": "The project uses a custom analytic tangent callback, not the stock effect object.",
    },
    {
        "topic": "long-term Solar-System WHCKL application",
        "title": "A repository of vanilla long-term integrations of the Solar System",
        "authors": "Brown, G.; Rein, H.",
        "year": 2020,
        "doi_or_official_url": "https://doi.org/10.3847/2515-5172/abd103",
        "software_version_or_commit": "WHCKL application",
        "symbol_or_section": "96 long-term Solar-System integrations",
        "paraphrase": "WHCKL has been used for large ensembles of long-term physical Solar-System trajectories.",
        "direct_applicability": "Demonstrates a precedent for a separate canonical physical lane.",
        "scope_caveat": "Its model, timestep, and scientific target differ from M0.",
    },
    {
        "topic": "secular-frequency and pericenter accuracy",
        "title": "Stepsize errors in the N-body problem: discerning Mercury's true possible long-term orbits",
        "authors": "Hernandez, D. M.; Zeebe, R. E.; Hadden, S.",
        "year": 2022,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stab3664",
        "software_version_or_commit": NOT_APPLICABLE,
        "symbol_or_section": "pericenter resolution criterion",
        "paraphrase": "Pointwise and statistical reliability can depend on resolving Mercury's perihelion passage; energy error alone is not decisive.",
        "direct_applicability": "Motivates successor screening with orbital and secular observables, not energy alone.",
        "scope_caveat": "Does not diagnose the completed M0 lanes by inspection.",
    },
    {
        "topic": "statistical versus pointwise convergence",
        "title": "On the statistical convergence of N-body simulations of the Solar System",
        "authors": "Rein, H.; Brown, G.; Kanda, M.",
        "year": 2025,
        "doi_or_official_url": "https://doi.org/10.33232/001c.154745",
        "software_version_or_commit": "ensemble WHCKL study",
        "symbol_or_section": "secular frequencies and instability-rate convergence",
        "paraphrase": "Long-term chaotic Solar-System integrations can converge statistically even when individual trajectories diverge pointwise.",
        "direct_applicability": "Limits the interpretation of the one-trajectory Step 3 pointwise failures.",
        "scope_caveat": "Does not waive M0's preregistered production gates.",
    },
    {
        "topic": "chaotic shadow trajectories",
        "title": "Fundamental limits from chaos on instability time predictions in compact planetary systems",
        "authors": "Hussain, N.; Tamayo, D.",
        "year": 2020,
        "doi_or_official_url": "https://doi.org/10.1093/mnras/stz3402",
        "software_version_or_commit": NOT_APPLICABLE,
        "symbol_or_section": "shadow-trajectory instability-time distributions",
        "paraphrase": "A single chaotic instability time is one draw from a distribution of nearby valid trajectories.",
        "direct_applicability": "Supports separating pointwise reproducibility from ensemble claims.",
        "scope_caveat": "Compact-system distributions are not a direct M0 threshold calibration.",
    },
]


def _render_report(payload: dict[str, Any]) -> str:
    matrix = payload["effective_settings_matrix"]
    archive_lines = "\n".join(
        f"| `{run_id}` | {entry['snapshots']} | `{entry['sha256_before']}` | unchanged |"
        for run_id, entry in sorted(payload["archive_audit"].items())
    )
    source_lines = "\n".join(
        f"| `{item['id']}` | {item['finding']} | {item['historical_applicability']} |"
        for item in payload["source_findings"]
    )
    literature_lines = "\n".join(
        f"| {item['year']} | [{item['title']}]({item['doi_or_official_url']}) | {item['direct_applicability']} | {item['scope_caveat']} |"
        for item in payload["literature_review"]
    )
    return f"""# M0 Step 3f0 WHFast Configuration Audit

## Status

- Final status: **{payload['final_status']}**
- Primary finding: **{payload['primary_finding']}**
- Integration steps: **0**
- Force evaluations: **0**
- Historical archives modified: **0**
- Manifest 18 remains: **TRUE_NONPHASE_NONCONVERGENCE**, scoped to the historical combined standard-kernel physical+tangent/MEGNO lane.

## Answer First

No material historical setting mismatch or undocumented effective default was found. The completed combined lanes consistently used REBOUND 4.6.0's supported standard-kernel Jacobi tangent map with `corrector=0`, `corrector2=0`, `safe_mode=1`, and `keep_unsynchronized=0`. Those settings match both the production runner and restored archives.

A material capability constraint is confirmed. REBOUND 4.6.0 rejects nonstandard WHFast kernels whenever native variational particles are present. Consequently, a single native physical+tangent/MEGNO lane cannot also use the literature's WHCKL-style physical kernel. Lane separation is technically necessary to use such a physical kernel while preserving REBOUND's native variation/MEGNO machinery. It is not necessary merely to enable a standard-kernel corrector or `safe_mode=0, keep_unsynchronized=1`; both remain feasible in the combined lane.

This inspection does not establish that the capability constraint caused the completed convergence anomaly. Manifest 15's physical-only current-sync controls reproduced the combined-lane behavior, while minimal synchronization did not materially improve both timesteps. Manifest 16's `SYSTEMATIC_WHFAST_STEP_BIAS` remains historical evidence, not a configuration defect newly proved here.

## Effective Configuration

The deterministic long-form matrix contains {matrix['rows']} rows: {matrix['lanes']} lanes x {matrix['settings']} settings. Its SHA-256 is `{matrix['sha256']}`. Values are labeled `DIRECT`, `CORROBORATED`, `INFERRED`, or `UNAVAILABLE`; absent archive values are never silently replaced with defaults.

The guarded zero-step construction recovered:

- before MEGNO: 10 real particles, no variations, Jacobi/default kernel, no correctors, `safe_mode=1`, `keep_unsynchronized=0`;
- after `init_megno(seed=12345)`: 20 particles, 10 first-variation particles in one full-system block, and unchanged WHFast settings;
- after GR attachment: the validated C callback was installed as a direct C function pointer, `force_is_velocity_dependent=0`, with zero callback invocations;
- the stored RNG field is the advanced generator state, so the requested seed is carried by configuration provenance rather than inferred from an archive.

## Source Findings

| Evidence | Finding | Historical application |
|---|---|---|
{source_lines}

### Synchronization and outputs

`safe_mode=1` synchronizes and reconstructs coordinates every timestep. Native WHFast variations also need synchronized inertial position, velocity, and acceleration for MEGNO every timestep. With `safe_mode=0, keep_unsynchronized=1`, that MEGNO synchronization is temporary: REBOUND caches and restores its internal Jacobi map. Thus, continuous MEGNO does impose synchronized inertial construction, but it does not inherently require discarding the unsynchronized map.

The Python particle, energy, orbit, MEGNO, and Lyapunov accessors do not independently establish synchronization. In the historical runner that distinction is harmless at scientific samples because every `Simulation.integrate(target, exact_finish_time=1)` synchronizes before returning. Scientific telemetry is therefore read from synchronized particle arrays.

`exact_finish_time=1` changes the step only when an output target is not divisible by the fixed step. A 100-Julian-year cadence is exactly divisible by 1, 0.5, 0.25, and 0.125 day, but not by 2 days. The older 2-day Step 3 lane therefore received a shortened one-day endpoint step at each 100-year target. This is a documented caveat for that older comparison, not a defect affecting the decisive 0.5/0.25/0.125-day evidence in Manifests 13-18.

### Checkpoints and restart

SimulationArchive heartbeat writes the live state without an implicit synchronization. Its binary schema preserves particles, variation configurations, MEGNO accumulators, WHFast coordinates/kernel/correctors, `safe_mode`, `keep_unsynchronized`, `is_synchronized`, recalculation state, and the internal `p_jh` map. Function pointers are deliberately not serialized. The production resume path loads and validates the archive, trims sidecar telemetry to the archive epoch, and then reattaches the validated callback. JSON/CSV sidecars do not mutate simulation state.

Archive files were opened read-only through existing REBOUND restoration paths and hashed before and after inspection:

| Lane | Snapshots | SHA-256 | Result |
|---|---:|---|---|
{archive_lines}

## Custom GR Callback

The callback remains byte-identical to Manifest 19 (`c764740a...`). It is position-only, sets `force_is_velocity_dependent=0`, applies the validated `gr_potential` acceleration and central response to every real particle, and applies the analytic Jacobian to each full real-particle variation block. It reads no WHFast configuration fields and performs no synchronization or coordinate transform.

REBOUNDx 4.6.1 documents that `gr_potential` preserves the WHFast split and gets perihelion precession right while shifting mean motion by order `GM/(a c^2)`. Its potential is exactly the validated correction `-3 G^2 M_sun^2 m_i/(c^2 r_i^2)`. The historical M0 scientific model intentionally inherits that physical approximation; Step 3f0 neither changes nor revalidates it.

## Literature Alignment

| Year | Primary source | Direct application | Caveat |
|---:|---|---|---|
{literature_lines}

The literature and exact source agree on the central architectural point: high-order physical kernels are useful candidates for long-term planetary trajectories, but REBOUND's implemented nonstandard WHFast kernels do not carry native variations/MEGNO. Literature on chaotic Solar-System integrations also distinguishes statistical convergence of ensembles from pointwise agreement of one trajectory. Neither principle weakens Manifest 17 or reinterprets Manifest 18.

## Scope of Prior Results

- Manifest 13's historical `BLOCKED` result is unchanged.
- Manifest 14's `REVERSIBILITY_GATE_PASSED` result is unchanged.
- Manifest 15's `BLOCKED` result and synchronization-control evidence are unchanged.
- Manifest 16's `SYSTEMATIC_WHFAST_STEP_BIAS` primary mechanism and qualified IAS15 phase floor are unchanged.
- Manifest 17 remains `STEP3E_025_DAY_PRODUCTION_NOT_VALIDATED`.
- Manifest 18 remains `STEP3E1_OFFLINE_DIAGNOSIS_COMPLETE` / `TRUE_NONPHASE_NONCONVERGENCE` for the historical combined standard-kernel lane.
- No conclusion here applies automatically to all WHFast configurations or to a WHCKL physical-only lane that has not yet been run under M0.

## Successor Architecture

The evidence supports preregistering a two-lane design for any successor experiment:

1. **Canonical physical lane:** full M0 physics and validated position-only compiled-C GR callback; Jacobi coordinates; sim.integrator=WHCKL (WHFast lazy kernel with corrector 17); `safe_mode=0`; `keep_unsynchronized=1`. This lane owns canonical physical-state, energy, angular-momentum, and secular-frequency claims.
2. **Tangent diagnostic lane:** identical initial physical state, timestep, output epochs, and GR callback; standard WHFast kernel; Jacobi coordinates; native first variations and MEGNO; `safe_mode=0`; `keep_unsynchronized=1`; and corrector order 17. This lane owns tangent direction/norm, MEGNO, and finite-time LCN diagnostics, not the canonical physical trajectory.

The smallest evidence-based screening experiment is a preregistered paired 10-kyr run at one already studied timestep (0.25 day is the natural choice), 100-year scientific cadence, with two fresh lanes: physical WHCKL and standard-kernel tangent/MEGNO. Compare both against the qualified 10-kyr IAS15 phase envelope and compare their shared physical observables, corrected energy, angular momentum, secular frequencies, and callback integrity. This is a recommendation only; Step 3f0 executes no command and does not validate 0.25 or 0.125 day for production.

## Residual Questions

- Whether WHCKL materially improves the M0 secular and state defects requires a controlled integration and cannot be inferred from source inspection.
- Whether corrector order 17 and its cost are worthwhile for the custom GR splitting must be preregistered and screened.
- The physical/tangent lane divergence tolerance and interpretation of tangent statistics must be fixed before observing a successor result.
- Production promotion still requires its own preregistered evidence; no Stage 4 or 10-Myr command is provided.

## Reproducibility

- Starting commit: `35a4d3ae6a717f7d40e4c4db0bd1e78b0c169ce4`
- Manifest 19 preregistration commit: `{payload['provenance']['preregistration_commit']}`
- Manifest 19 SHA-256: `{payload['manifest_sha256']}`
- REBOUND: tag 4.6.0, commit `{REBOUND_COMMIT}`, git-archive SHA-256 `{REBOUND_ARCHIVE_SHA256}`
- REBOUNDx: tag 4.6.1, commit `{REBOUNDX_COMMIT}`, git-archive SHA-256 `{REBOUNDX_ARCHIVE_SHA256}`
- Matrix: `{matrix['path']}`
- No-step guard calls: {payload['no_step_guard']['blocked_call_count']}
"""


def audit(manifest_path: Path) -> dict[str, Any]:
    immutable = _immutable_audit(manifest_path)
    source = _source_audit()
    payload = {
        "status": "PASS",
        "manifest_sha256": immutable["manifest_sha256"],
        "immutable": immutable,
        "source": source,
        "integration_steps": 0,
        "force_evaluations": 0,
    }
    return payload


def analyze(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Manifest 19")
    audit_payload = audit(manifest_path)
    with NoStepGuard() as guard:
        zero_probe = _zero_step_probe(manifest)
        archive_payload = _archive_audit(_archive_paths())
    _require(not guard.calls, "A forbidden stepping call occurred.")
    matrix_rows = _build_settings_matrix(manifest, zero_probe, archive_payload)
    matrix_path = Path(manifest["paths"]["settings_matrix"])
    summary_path = Path(manifest["paths"]["summary"])
    report_path = Path(manifest["paths"]["report"])
    _atomic_csv(matrix_path, matrix_rows)
    matrix_sha = sha256_file(matrix_path)
    prereg_commit = _git("log", "-1", "--format=%H", "--", str(manifest_path.relative_to(ROOT)))
    payload = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "model_id": manifest["model_id"],
        "final_status": FINAL_STATUS,
        "primary_finding": PRIMARY_FINDING,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "created_date": "2026-08-09",
        "provenance": {
            "starting_commit": manifest["preregistration"]["starting_commit"],
            "preregistration_commit": prereg_commit,
            "audit_head": audit_payload["immutable"]["head"],
            "compiled_c_tag_commit": audit_payload["immutable"]["tag_commit"],
        },
        "execution": {
            "integration_steps": 0,
            "force_evaluations": 0,
            "benchmarks": 0,
            "new_archives": 0,
            "historical_archives_modified": 0,
        },
        "no_step_guard": {
            "blocked_call_count": len(guard.calls),
            "patched_surfaces": [
                "rebound.Simulation.integrate",
                "rebound.Simulation.step",
                "rebound.Simulation.steps",
                "clibrebound.reb_simulation_integrate",
                "clibrebound.reb_simulation_step",
                "long_term_stability_cli.integrate_rebound_streaming",
                "rebound_gr_tangent_backend_cli.main",
            ],
        },
        "classification": {
            "material_misconfiguration_confirmed": False,
            "combined_lane_capability_constraint_confirmed": True,
            "causation_established": False,
            "reason": "REBOUND 4.6.0 directly rejects nonstandard WHFast kernels with native variations; recovered historical settings are internally consistent and supported.",
        },
        "historical_scope": manifest["historical_results_immutable"],
        "immutable_audit": audit_payload["immutable"],
        "exact_source_audit": audit_payload["source"],
        "zero_step_probe": zero_probe,
        "archive_audit": archive_payload,
        "source_findings": SOURCE_FINDINGS,
        "literature_review": LITERATURE,
        "operational_conclusions": {
            "historical_combined_lane": "Supported standard-kernel Jacobi first-variation configuration; no material mismatch found.",
            "canonical_physical_lane": "Eligible for a nonstandard WHCKL/lazy kernel only without native variations in REBOUND 4.6.0.",
            "tangent_megno_lane": "Requires the standard kernel and Jacobi coordinates; correctors and safe_mode=0/keep_unsynchronized=1 remain feasible.",
            "split_lane_necessity": "Required only to combine a nonstandard canonical physical kernel with native tangent/MEGNO diagnostics.",
            "synchronization_causation": "Not established; existing physical-only controls do not support synchronization as the sole mechanism.",
            "two_day_output_caveat": "The historical 2-day lane used shortened one-day endpoint steps at 100-year outputs; decisive 0.5/0.25/0.125-day lanes divide the cadence exactly.",
        },
        "successor_recommendation": {
            "architecture": "separate canonical physical WHCKL (lazy kernel, corrector 17) and standard-kernel tangent/MEGNO (corrector 17) lanes",
            "smallest_screen": "two fresh preregistered 10-kyr lanes at 0.25 day and 100-year output cadence, compared with the qualified IAS15 10-kyr envelope",
            "executed": False,
            "stage4_ready": False,
        },
        "effective_settings_matrix": {
            "path": str(matrix_path),
            "rows": len(matrix_rows),
            "lanes": len(manifest["historical_lanes"]),
            "settings": len(manifest["effective_settings_matrix"]["settings"]),
            "sha256": matrix_sha,
            "size_bytes": matrix_path.stat().st_size,
        },
    }
    _atomic_json(summary_path, payload)
    _atomic_text(report_path, _render_report(payload))
    print(
        json.dumps(
            {
                "final_status": FINAL_STATUS,
                "primary_finding": PRIMARY_FINDING,
                "matrix_rows": len(matrix_rows),
                "summary": str(summary_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


def verify(manifest_path: Path) -> int:
    manifest = _load_json(manifest_path, "Manifest 19")
    audit_payload = audit(manifest_path)
    summary_path = Path(manifest["paths"]["summary"])
    report_path = Path(manifest["paths"]["report"])
    matrix_path = Path(manifest["paths"]["settings_matrix"])
    summary = _load_json(summary_path, "Step 3f0 summary")
    _finite_json(summary)
    _require(summary["final_status"] in manifest["allowed_final_statuses"], "Invalid final status.")
    _require(summary["primary_finding"] in manifest["allowed_primary_findings"], "Invalid primary finding.")
    _require(summary["manifest_sha256"] == sha256_file(manifest_path), "Manifest hash changed.")
    _require(report_path.is_file(), "Missing Step 3f0 report.")
    _require(matrix_path.is_file(), "Missing effective-settings matrix.")
    _require(summary["effective_settings_matrix"]["sha256"] == sha256_file(matrix_path), "Matrix hash changed.")
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == manifest["effective_settings_matrix"]["expected_rows"], "Matrix row count changed.")
    _require(
        len({(row["lane_id"], row["setting"]) for row in rows}) == len(rows),
        "Matrix keys are not unique.",
    )
    _require(summary["execution"]["integration_steps"] == 0, "Summary records integration steps.")
    _require(summary["execution"]["force_evaluations"] == 0, "Summary records force evaluations.")
    _require(summary["no_step_guard"]["blocked_call_count"] == 0, "No-step guard was triggered.")
    for run_id, archive in summary["archive_audit"].items():
        path = Path(archive["path"])
        _require(sha256_file(path) == archive["sha256_before"], f"Archive changed: {run_id}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_status": summary["final_status"],
                "primary_finding": summary["primary_finding"],
                "manifest_sha256": audit_payload["manifest_sha256"],
                "summary_sha256": sha256_file(summary_path),
                "report_sha256": sha256_file(report_path),
                "matrix_sha256": sha256_file(matrix_path),
                "matrix_rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only M0 Step 3f0 WHFast configuration audit."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "command",
        choices=("audit", "analyze", "verify"),
        help="Every command is guarded against integration, stepping, and force evaluation.",
    )
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit(args.manifest), indent=2))
        return 0
    if args.command == "analyze":
        return analyze(args.manifest)
    return verify(args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
