from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from .gr_potential_tangent import C_M_PER_S


C_BACKEND_API_VERSION = 1
BUILD_FLAGS = (
    "-std=c11",
    "-O3",
    "-fPIC",
    "-shared",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-fno-fast-math",
    "-ffp-contract=off",
)


class CBackendError(RuntimeError):
    pass


class CBackendBuildError(CBackendError):
    pass


class CBackendCompatibilityError(CBackendError):
    pass


class _CStats(ctypes.Structure):
    _fields_ = [
        ("callback_invocations", ctypes.c_uint64),
        ("real_gr_accel_norm_max", ctypes.c_double),
        ("real_gr_accel_norm_sum", ctypes.c_double),
        ("real_gr_accel_norm_count", ctypes.c_uint64),
        ("tangent_gr_accel_norm_max", ctypes.c_double),
        ("tangent_gr_accel_norm_sum", ctypes.c_double),
        ("tangent_gr_accel_norm_count", ctypes.c_uint64),
        ("nonfinite_result_count", ctypes.c_uint64),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def c_source_path() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "gr_potential_tangent.c"


def default_build_dir() -> Path:
    return package_root() / "build" / "gr_tangent_c"


def default_artifact_path() -> Path:
    return default_build_dir() / "libmini_ephemeris_gr_tangent.so"


def default_metadata_path() -> Path:
    return default_build_dir() / "build_metadata.json"


def rebound_header_path(rebound_module: Any) -> Path:
    return Path(rebound_module.__file__).resolve().parent / "rebound.h"


def build_c_backend(*, force: bool = False, compiler: str | None = None) -> dict[str, Any]:
    import rebound

    source = c_source_path()
    header = rebound_header_path(rebound)
    artifact = default_artifact_path()
    metadata_path = default_metadata_path()
    selected_compiler = compiler or os.environ.get("CC", "cc")
    compiler_path = shutil.which(selected_compiler)
    if compiler_path is None:
        raise CBackendBuildError(f"C compiler not found: {selected_compiler}")
    if not source.is_file():
        raise CBackendBuildError(f"C source not found: {source}")
    if not header.is_file():
        raise CBackendBuildError(f"Installed REBOUND header not found: {header}")

    source_hash = _sha256(source)
    header_hash = _sha256(header)
    expected = {
        "api_version": C_BACKEND_API_VERSION,
        "source_sha256": source_hash,
        "rebound_header_sha256": header_hash,
        "rebound_version": str(rebound.__version__),
        "compiler_path": str(Path(compiler_path).resolve()),
        "compiler_flags": list(BUILD_FLAGS),
    }
    if not force and artifact.is_file() and metadata_path.is_file():
        try:
            existing = json.loads(metadata_path.read_text())
        except Exception:
            existing = {}
        if all(existing.get(key) == value for key, value in expected.items()):
            return existing

    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_name(artifact.name + ".tmp")
    command = [
        compiler_path,
        *BUILD_FLAGS,
        "-I",
        str(header.parent),
        str(source),
        "-o",
        str(temporary),
        "-lm",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise CBackendBuildError(
            "C backend build failed.\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    compiler_identity = subprocess.run(
        [compiler_path, "--version"], text=True, capture_output=True, check=True
    ).stdout.splitlines()[0]
    os.replace(temporary, artifact)
    metadata = {
        **expected,
        "compiler_identity": compiler_identity,
        "build_command": command,
        "artifact_path": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "rebound_build": str(getattr(rebound, "__build__", "unknown")),
        "rebound_githash": str(getattr(rebound, "__githash__", "unknown")),
    }
    _atomic_json(metadata_path, metadata)
    return metadata


@dataclass(frozen=True)
class CBackend:
    library: Any
    artifact_path: Path
    build_metadata: dict[str, Any]
    abi_metadata: dict[str, Any]

    def attach(
        self,
        sim: Any,
        *,
        coefficient_scale: float = 1.0,
        c_m_per_s: float = C_M_PER_S,
        include_central_response: bool = True,
    ) -> None:
        import rebound

        result = self.library.me_gr_tangent_attach(
            ctypes.byref(sim),
            float(coefficient_scale),
            float(c_m_per_s),
            int(include_central_response),
        )
        if result == -2:
            raise CBackendError(
                "Cannot attach compiled GR tangent force: simulation already owns extras or callback state."
            )
        if result != 0:
            raise CBackendError(f"Cannot attach compiled GR tangent force (C error {result}).")
        if not self.is_attached(sim):
            raise CBackendError("Compiled GR tangent callback attachment did not verify.")
        sim._mini_ephemeris_gr_tangent_c_backend = self
        sim._mini_ephemeris_gr_tangent_c_config = {
            "coefficient_scale": float(coefficient_scale),
            "c_m_per_s": float(c_m_per_s),
            "include_central_response": bool(include_central_response),
            "rebound_version": rebound.__version__,
        }

    def is_attached(self, sim: Any) -> bool:
        return bool(self.library.me_gr_tangent_is_attached(ctypes.byref(sim)))

    def callback_pointer(self, sim: Any) -> int:
        return int(ctypes.cast(sim._additional_forces, ctypes.c_void_p).value or 0)

    def hot_path_proof(self, sim: Any) -> dict[str, Any]:
        installed = self.callback_pointer(sim)
        exported = int(self.library.me_gr_tangent_callback_address())
        return {
            "kind": "direct_rebound_c_function_pointer",
            "installed_callback_address": hex(installed),
            "exported_c_callback_address": hex(exported),
            "addresses_match": installed == exported and installed != 0,
            "python_callback_in_force_path": False,
            "c_owned_instrumentation": True,
        }

    def stats(self, sim: Any) -> dict[str, int | float | None]:
        raw = _CStats()
        result = self.library.me_gr_tangent_get_stats(ctypes.byref(sim), ctypes.byref(raw))
        if result != 0:
            raise CBackendError("Compiled GR tangent statistics are unavailable for this simulation.")
        return {
            "callback_invocations": int(raw.callback_invocations),
            "real_gr_accel_norm_max": float(raw.real_gr_accel_norm_max),
            "real_gr_accel_norm_sum": float(raw.real_gr_accel_norm_sum),
            "real_gr_accel_norm_count": int(raw.real_gr_accel_norm_count),
            "real_gr_accel_norm_mean": (
                float(raw.real_gr_accel_norm_sum) / raw.real_gr_accel_norm_count
                if raw.real_gr_accel_norm_count
                else None
            ),
            "tangent_gr_accel_norm_max": float(raw.tangent_gr_accel_norm_max),
            "tangent_gr_accel_norm_sum": float(raw.tangent_gr_accel_norm_sum),
            "tangent_gr_accel_norm_count": int(raw.tangent_gr_accel_norm_count),
            "tangent_gr_accel_norm_mean": (
                float(raw.tangent_gr_accel_norm_sum) / raw.tangent_gr_accel_norm_count
                if raw.tangent_gr_accel_norm_count
                else None
            ),
            "nonfinite_result_count": int(raw.nonfinite_result_count),
        }

    def pointwise(
        self,
        positions_m: np.ndarray,
        masses_kg: np.ndarray,
        delta_positions_m: np.ndarray | None,
        *,
        gravitational_constant: float,
        coefficient_scale: float = 1.0,
        c_m_per_s: float = C_M_PER_S,
        include_central_response: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        positions = np.ascontiguousarray(positions_m, dtype=np.float64)
        masses = np.ascontiguousarray(masses_kg, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or masses.shape != (positions.shape[0],):
            raise ValueError("positions must have shape (N, 3) and masses shape (N,)")
        delta = None if delta_positions_m is None else np.ascontiguousarray(delta_positions_m, dtype=np.float64)
        if delta is not None and delta.shape != positions.shape:
            raise ValueError("delta_positions must match positions shape")
        accelerations = np.zeros_like(positions)
        tangent = np.zeros_like(positions) if delta is not None else None
        pointer = ctypes.POINTER(ctypes.c_double)
        result = self.library.me_gr_tangent_pointwise(
            positions.shape[0],
            positions.ctypes.data_as(pointer),
            masses.ctypes.data_as(pointer),
            delta.ctypes.data_as(pointer) if delta is not None else None,
            float(gravitational_constant),
            float(coefficient_scale),
            float(c_m_per_s),
            int(include_central_response),
            accelerations.ctypes.data_as(pointer),
            tangent.ctypes.data_as(pointer) if tangent is not None else None,
        )
        if result != 0:
            raise CBackendError(f"Compiled pointwise evaluation failed (C error {result}).")
        return accelerations, tangent


def _configure_library(library: Any, rebound: Any) -> None:
    simulation_pointer = ctypes.POINTER(rebound.Simulation)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library.me_gr_tangent_api_version.restype = ctypes.c_uint32
    library.me_gr_tangent_sizeof_simulation.restype = ctypes.c_size_t
    library.me_gr_tangent_sizeof_particle.restype = ctypes.c_size_t
    library.me_gr_tangent_offsetof_additional_forces.restype = ctypes.c_size_t
    library.me_gr_tangent_offsetof_extras.restype = ctypes.c_size_t
    library.me_gr_tangent_callback_address.restype = ctypes.c_size_t
    library.me_gr_tangent_attach.argtypes = [
        simulation_pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ]
    library.me_gr_tangent_attach.restype = ctypes.c_int
    library.me_gr_tangent_detach.argtypes = [simulation_pointer]
    library.me_gr_tangent_detach.restype = ctypes.c_int
    library.me_gr_tangent_is_attached.argtypes = [simulation_pointer]
    library.me_gr_tangent_is_attached.restype = ctypes.c_int
    library.me_gr_tangent_get_stats.argtypes = [simulation_pointer, ctypes.POINTER(_CStats)]
    library.me_gr_tangent_get_stats.restype = ctypes.c_int
    library.me_gr_tangent_pointwise.argtypes = [
        ctypes.c_size_t,
        double_pointer,
        double_pointer,
        double_pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        double_pointer,
        double_pointer,
    ]
    library.me_gr_tangent_pointwise.restype = ctypes.c_int


def load_c_backend(*, artifact_path: Path | None = None) -> CBackend:
    import rebound

    artifact = (artifact_path or default_artifact_path()).resolve()
    metadata_path = artifact.parent / "build_metadata.json"
    if not artifact.is_file():
        raise CBackendCompatibilityError(
            f"Compiled GR tangent artifact not found: {artifact}. "
            "Run `python -m mini_ephemeris.gr_potential_tangent_c_build`."
        )
    if not metadata_path.is_file():
        raise CBackendCompatibilityError(f"Build metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        raise CBackendCompatibilityError(f"Unreadable build metadata {metadata_path}: {exc}") from exc
    expected = {
        "api_version": C_BACKEND_API_VERSION,
        "source_sha256": _sha256(c_source_path()),
        "rebound_header_sha256": _sha256(rebound_header_path(rebound)),
        "rebound_version": str(rebound.__version__),
        "artifact_sha256": _sha256(artifact),
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise CBackendCompatibilityError(
            "Compiled GR tangent artifact is stale or incompatible: " + json.dumps(mismatches, sort_keys=True)
        )
    try:
        library = ctypes.CDLL(str(artifact))
    except OSError as exc:
        raise CBackendCompatibilityError(f"Cannot load compiled artifact {artifact}: {exc}") from exc
    _configure_library(library, rebound)
    abi = {
        "api_version": int(library.me_gr_tangent_api_version()),
        "c_sizeof_reb_simulation": int(library.me_gr_tangent_sizeof_simulation()),
        "python_sizeof_reb_simulation": ctypes.sizeof(rebound.Simulation),
        "c_sizeof_reb_particle": int(library.me_gr_tangent_sizeof_particle()),
        "python_sizeof_reb_particle": ctypes.sizeof(rebound.Particle),
        "c_offsetof_additional_forces": int(library.me_gr_tangent_offsetof_additional_forces()),
        "python_offsetof_additional_forces": rebound.Simulation._additional_forces.offset,
        "c_offsetof_extras": int(library.me_gr_tangent_offsetof_extras()),
        "python_offsetof_extras": rebound.Simulation.extras.offset,
    }
    mismatched_abi = {
        "api_version": abi["api_version"] != C_BACKEND_API_VERSION,
        "simulation_size": abi["c_sizeof_reb_simulation"] != abi["python_sizeof_reb_simulation"],
        "particle_size": abi["c_sizeof_reb_particle"] != abi["python_sizeof_reb_particle"],
        "additional_forces_offset": abi["c_offsetof_additional_forces"]
        != abi["python_offsetof_additional_forces"],
        "extras_offset": abi["c_offsetof_extras"] != abi["python_offsetof_extras"],
    }
    failed = [name for name, mismatch in mismatched_abi.items() if mismatch]
    if failed:
        raise CBackendCompatibilityError(
            f"Compiled GR tangent ABI mismatch in {failed}: {json.dumps(abi, sort_keys=True)}"
        )
    return CBackend(library=library, artifact_path=artifact, build_metadata=metadata, abi_metadata=abi)
