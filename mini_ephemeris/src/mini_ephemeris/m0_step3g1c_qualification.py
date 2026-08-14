"""Guarded safety, integrity, and regeneration helpers for Step 3g1c."""

from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "mini_ephemeris/src/mini_ephemeris"
MANIFEST25 = ROOT / "ephemeris_experiment_runner/manifests/25_m0_step3g1c_kepler_drift_tangent_primitive_v1.json"
PREREGISTRATION_COMMIT = "d2ffe64742734f7b1337c74faa7705073bc1c203"
PYTEST_SITE_PACKAGES = Path("/home/peacelovephysics/sheet-music-generator/.venv/lib/python3.10/site-packages")
PYTEST_VERSION = "8.4.2"
PYTEST_IDENTITY_SHA256 = {
    "pytest/__init__.py": "66993a5e3905005e0981159b4794d10b1adacf341a58a44d696ad2c4442dcdc6",
    "pytest-8.4.2.dist-info/METADATA": "93d6c5ef0a9714d53716243035037b77fa7d5f970596c48433887cf57f7f675a",
    "pytest-8.4.2.dist-info/RECORD": "fba86b3aa6d34c9d73bc8b2ea69d6d69a65cd555c4ac84ef15a330232706f82f",
}

LOCAL_ALLOWED_PREFIXES = (
    "mini_ephemeris.v2",
    "mini_ephemeris.m0_step3g1a_requalification",
    "mini_ephemeris.m0_step3g1b_qualification",
    "mini_ephemeris.m0_step3g1b_qualification_runner",
    "mini_ephemeris.m0_step3g1b_reporting",
    "mini_ephemeris.m0_step3g1c_qualification",
    "mini_ephemeris.m0_step3g1c_qualification_runner",
    "mini_ephemeris.m0_step3g1c_reporting",
)
APPROVED_DEPENDENCY_ROOTS = {
    "pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments",
    "tomli", "exceptiongroup", "typing_extensions", "numpy",
}
TEST_MODULES = {
    "test_v2_foundation", "test_m0_step3g1a_requalification",
    "test_v2_jacobi", "test_m0_step3g1b_integrity",
    "test_m0_step3g1b_artifacts", "test_v2_kepler",
    "test_m0_step3g1c_integrity", "test_m0_step3g1c_artifacts",
}
EXPLICIT_FORBIDDEN_MODULES = (
    "step3g1c_forbidden_sentinel", "rebound", "reboundx",
)
FORBIDDEN_LOCAL_PREFIXES = (
    "mini_ephemeris.gr_", "mini_ephemeris.rebound_",
    "mini_ephemeris.nbody", "mini_ephemeris.m0_step3g0",
)
FORBIDDEN_LIBRARY_MARKERS = (
    "step3g1c_forbidden_library_sentinel", "libmini_ephemeris_gr_tangent",
    "gr_potential_tangent.so", "librebound", "libreboundx",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def manifest25() -> Mapping[str, Any]:
    value = strict_json(MANIFEST25)
    if not isinstance(value, dict):
        raise TypeError("Manifest 25 must be a JSON object")
    return value


def install_namespace_shell(root: Path = ROOT) -> None:
    existing = sys.modules.get("mini_ephemeris")
    if existing is not None:
        if getattr(existing, "__file__", None) is not None:
            raise RuntimeError("legacy mini_ephemeris package initialization already ran")
        return
    package_path = root / "mini_ephemeris/src/mini_ephemeris"
    package = types.ModuleType("mini_ephemeris")
    package.__package__ = "mini_ephemeris"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "mini_ephemeris", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["mini_ephemeris"] = package


def _is_allowed_local_module(fullname: str) -> bool:
    return fullname == "mini_ephemeris" or any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in LOCAL_ALLOWED_PREFIXES
    )


def _is_forbidden_module(fullname: str) -> bool:
    if any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in EXPLICIT_FORBIDDEN_MODULES
    ):
        return True
    if any(fullname.startswith(prefix) for prefix in FORBIDDEN_LOCAL_PREFIXES):
        return True
    return fullname.startswith("mini_ephemeris.") and not _is_allowed_local_module(fullname)


def reject_forbidden_library_path(path: object) -> None:
    if path is None:
        return
    normalized = os.fsdecode(path).lower()
    if any(marker in normalized for marker in FORBIDDEN_LIBRARY_MARKERS):
        raise RuntimeError(f"forbidden compiled library path: {normalized}")


def loaded_library_violations(maps_text: str | None = None) -> tuple[str, ...]:
    if maps_text is None:
        path = Path("/proc/self/maps")
        maps_text = path.read_text(encoding="utf-8") if path.exists() else ""
    lowered = maps_text.lower()
    return tuple(marker for marker in FORBIDDEN_LIBRARY_MARKERS if marker in lowered)


class Step3g1cImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.strict = False
        self.preloaded_roots: set[str] = set()

    def activate_strict(self) -> None:
        self.preloaded_roots = {name.split(".", 1)[0] for name in sys.modules}
        self.strict = True

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: object | None = None
    ) -> None:
        del path, target
        if _is_forbidden_module(fullname):
            raise ImportError(f"Manifest 25 forbids importing {fullname!r}")
        if not self.strict:
            return None
        root = fullname.split(".", 1)[0]
        allowed = (
            root in sys.stdlib_module_names
            or root.startswith("_sysconfigdata_")
            or root in self.preloaded_roots
            or root in APPROVED_DEPENDENCY_ROOTS
            or root in TEST_MODULES
            or _is_allowed_local_module(fullname)
        )
        if not allowed:
            raise ImportError(f"Manifest 25 import root is not allowlisted: {fullname!r}")
        return None


_ACTIVE_GUARD: Step3g1cImportGuard | None = None
_AUDIT_HOOK_INSTALLED = False


def install_guard() -> Step3g1cImportGuard:
    global _ACTIVE_GUARD, _AUDIT_HOOK_INSTALLED
    install_namespace_shell()
    if _ACTIVE_GUARD is None:
        _ACTIVE_GUARD = Step3g1cImportGuard()
        sys.meta_path.insert(0, _ACTIVE_GUARD)
    if not _AUDIT_HOOK_INSTALLED:
        def audit_hook(event: str, args: tuple[object, ...]) -> None:
            if event == "ctypes.dlopen" and args:
                reject_forbidden_library_path(args[0])

        sys.addaudithook(audit_hook)
        _AUDIT_HOOK_INSTALLED = True
    assert_protected_runtime_absent()
    return _ACTIVE_GUARD


def assert_protected_runtime_absent() -> None:
    forbidden = sorted(name for name in sys.modules if _is_forbidden_module(name))
    if forbidden:
        raise RuntimeError(f"forbidden modules are loaded: {forbidden}")
    libraries = loaded_library_violations()
    if libraries:
        raise RuntimeError(f"forbidden compiled libraries are loaded: {libraries}")


def _imports_for(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add("." * node.level + (node.module or ""))
    return tuple(sorted(imports))


def _find_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return method
    raise AssertionError(f"missing exact node {class_name}::{method_name}")


def _verify_node_ids(node_ids: Iterable[str]) -> None:
    parsed: dict[Path, ast.Module] = {}
    for node_id in node_ids:
        parts = node_id.split("::")
        if len(parts) != 3:
            raise AssertionError(f"node ID is not exact: {node_id}")
        relative, class_name, method_name = parts
        path = ROOT / relative
        tree = parsed.setdefault(
            path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        _find_method(tree, class_name, method_name)


def _subprocess_inventory(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess":
            continue
        owner: ast.AST | None = node
        while owner is not None and not isinstance(owner, ast.FunctionDef):
            owner = parents.get(owner)
        owner_name = owner.name if isinstance(owner, ast.FunctionDef) else "<module>"
        found.append(f"{path.relative_to(ROOT)}:{owner_name}:{node.func.attr}")
    return tuple(sorted(found))


def pytest_runtime_audit() -> Mapping[str, Any]:
    observed = {
        relative: sha256_file(PYTEST_SITE_PACKAGES / relative)
        for relative in PYTEST_IDENTITY_SHA256
    }
    if observed != PYTEST_IDENTITY_SHA256:
        raise AssertionError("audited pytest runtime identity changed")
    return {
        "version": PYTEST_VERSION,
        "site_packages": str(PYTEST_SITE_PACKAGES),
        "sha256": observed,
        "plugin_autoload_disabled": True,
    }


def _source_inventory() -> tuple[str, ...]:
    manifest = manifest25()
    paths = set(manifest["qualified_step3g1a_read_only_sha256"])
    paths.update(manifest["qualified_step3g1b_read_only_sha256"])
    paths.update(manifest["paths"].values())
    paths.update({
        "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_requalification.py",
        "mini_ephemeris/tests/test_m0_step3g1a_requalification.py",
    })
    paths.discard(manifest["paths"]["manifest"])
    paths.discard(manifest["paths"]["documentation_root"])
    return tuple(sorted(paths))


def static_safety_audit() -> Mapping[str, Any]:
    manifest = manifest25()
    for key in ("qualified_step3g1a_read_only_sha256", "qualified_step3g1b_read_only_sha256"):
        for relative, expected in manifest[key].items():
            if sha256_file(ROOT / relative) != expected:
                raise AssertionError(f"qualified prior file changed: {relative}")
    expected_v2 = {
        path for key in ("qualified_step3g1a_read_only_sha256", "qualified_step3g1b_read_only_sha256")
        for path in manifest[key] if path.startswith("mini_ephemeris/src/mini_ephemeris/v2/")
    }
    expected_v2.add(manifest["paths"]["implementation"])
    actual_v2 = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "mini_ephemeris/src/mini_ephemeris/v2").iterdir()
        if path.is_file() and path.suffix == ".py"
    }
    if actual_v2 != expected_v2:
        raise AssertionError("v2 source inventory differs from Manifest 25")
    source_paths = [ROOT / relative for relative in _source_inventory()]
    import_graph = {
        str(path.relative_to(ROOT)): list(_imports_for(path)) for path in source_paths
    }
    prohibited = sorted(
        module for modules in import_graph.values() for module in modules
        if module in {"rebound", "reboundx"}
        or module.startswith("mini_ephemeris.gr_")
        or module.startswith("mini_ephemeris.rebound_")
        or module.startswith("mini_ephemeris.m0_step3g0")
    )
    if prohibited:
        raise AssertionError(f"static import graph is unsafe: {prohibited}")
    selection = manifest["exact_test_selection"]
    keys = (
        "step3g1c_core_node_ids", "step3g1c_integrity_node_ids",
        "safe_step3g1a_regression_node_ids", "safe_step3g1b_regression_node_ids",
        "artifact_node_ids",
    )
    all_nodes = [node for key in keys for node in selection[key]]
    _verify_node_ids(all_nodes)
    subprocesses = _subprocess_inventory(ROOT / manifest["paths"]["qualification_helper"])
    owners = {":".join(value.split(":")[-2:]) for value in subprocesses}
    expected_owners = {
        "git_output:check_output", "run_fresh_artifact_probe:run",
        "run_fresh_kepler_probe:run",
    }
    if owners != expected_owners:
        raise AssertionError(f"subprocess closure changed: {subprocesses}")
    assert_protected_runtime_absent()
    return {
        "status": "PASS",
        "source_file_count": len(source_paths),
        "selected_node_count": len(all_nodes),
        "import_graph": import_graph,
        "forbidden_imports": [],
        "forbidden_library_mappings": list(loaded_library_violations()),
        "active_subprocess_call_sites": list(subprocesses),
        "legacy_package_init_bypassed": True,
        "legacy_nbody_absent": "mini_ephemeris.nbody" not in sys.modules,
        "pytest_runtime": pytest_runtime_audit(),
    }


GUARDED_KEPLER_PROBE_SOURCE = r"""
import importlib.machinery
import json
import math
from pathlib import Path
import sys
import types
root = Path(sys.argv[1])
package_path = root / "mini_ephemeris/src/mini_ephemeris"
package = types.ModuleType("mini_ephemeris")
package.__package__ = "mini_ephemeris"
package.__path__ = [str(package_path)]
package.__spec__ = importlib.machinery.ModuleSpec("mini_ephemeris", loader=None, is_package=True)
package.__spec__.submodule_search_locations = [str(package_path)]
sys.modules["mini_ephemeris"] = package
sys.path.insert(0, str(root / "mini_ephemeris/src"))
from mini_ephemeris.m0_step3g1c_qualification import assert_protected_runtime_absent, install_guard
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.v2.jacobi import build_jacobi_transform_plan
from mini_ephemeris.v2.kepler import CanonicalKeplerPairState, CanonicalKeplerPairTangent, build_kepler_pair_plan, kepler_drift_tangent
from mini_ephemeris.v2.model import CompiledLayout, PhysicalModel, SI_UNITS
layout = CompiledLayout(("inner", "outer"), "inner")
model = PhysicalModel(model_id="probe", schema_version="1", layout=layout,
    masses_kg={"inner": 4.0, "outer": 4.0}, gravitational_constant_si=5.0e13,
    units=SI_UNITS, enabled_effects=("synthetic-none",), provenance={"fixture": "probe"})
plan = build_kepler_pair_plan(
    model, build_jacobi_transform_plan(model), 1, minimum_periapsis_m=1.0e6)
speed = math.sqrt(plan.gravitational_parameter_m3_s2 / 2.0e7)
state = CanonicalKeplerPairState(
    q_m=(2.0e7, 0.0, 0.0),
    p_kg_m_per_s=(0.0, plan.reduced_mass_kg * speed, 0.0),
    unit_system_id="si_v1", layout_fingerprint=model.layout.fingerprint,
    model_fingerprint=model.fingerprint, pair_plan_fingerprint=plan.fingerprint)
tangent = CanonicalKeplerPairTangent(
    delta_q_m=(1.0, -2.0, 0.5),
    delta_p_kg_m_per_s=(0.25, -0.5, 0.125),
    unit_system_id="si_v1", layout_fingerprint=model.layout.fingerprint,
    model_fingerprint=model.fingerprint, pair_plan_fingerprint=plan.fingerprint)
result = kepler_drift_tangent(plan, state, tangent, 123.0)
assert_protected_runtime_absent()
print(json.dumps({"plan": plan.fingerprint, "state": result.state.canonical_bytes().decode(),
    "tangent": result.tangent.canonical_bytes().decode(), "branch": result.diagnostics.branch,
    "iterations": result.diagnostics.iterations,
    "legacy_nbody": "mini_ephemeris.nbody" in sys.modules},
    sort_keys=True, separators=(",", ":")))
""".strip()


GUARDED_ARTIFACT_PROBE_SOURCE = r"""
import importlib.machinery
from pathlib import Path
import sys
import types
root = Path(sys.argv[1])
destination = Path(sys.argv[2])
package_path = root / "mini_ephemeris/src/mini_ephemeris"
package = types.ModuleType("mini_ephemeris")
package.__package__ = "mini_ephemeris"
package.__path__ = [str(package_path)]
package.__spec__ = importlib.machinery.ModuleSpec("mini_ephemeris", loader=None, is_package=True)
package.__spec__.submodule_search_locations = [str(package_path)]
sys.modules["mini_ephemeris"] = package
sys.path.insert(0, str(root / "mini_ephemeris/src"))
from mini_ephemeris.m0_step3g1c_qualification import assert_protected_runtime_absent, install_guard
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.m0_step3g1c_reporting import generate_artifacts
generate_artifacts(destination)
assert_protected_runtime_absent()
""".strip()


def run_fresh_kepler_probe(hash_seed: int, locale_name: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["LC_ALL"] = locale_name
    result = subprocess.run(
        [sys.executable, "-I", "-c", GUARDED_KEPLER_PROBE_SOURCE, str(ROOT)],
        cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
    )
    return result.stdout


def run_fresh_artifact_probe(destination: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["LC_ALL"] = "C"
    subprocess.run(
        [sys.executable, "-I", "-c", GUARDED_ARTIFACT_PROBE_SOURCE, str(ROOT), str(destination)],
        cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
    )


def git_output(arguments: Sequence[str]) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_inherited_integrity() -> Mapping[str, Any]:
    from mini_ephemeris.m0_step3g1b_qualification import (
        verify_inherited_integrity as verify_step3g1b,
    )

    prior = verify_step3g1b()
    manifest = manifest25()
    expected = {}
    expected.update(manifest["protected_sources"])
    expected.update(manifest["qualified_step3g1a_read_only_sha256"])
    expected.update(manifest["qualified_step3g1b_read_only_sha256"])
    manifest_root = ROOT / "ephemeris_experiment_runner/manifests"
    for number, digest in manifest["controlled_baseline"]["manifest_sha256_13_through_24"].items():
        matches = tuple(manifest_root.glob(f"{int(number):02d}_*.json"))
        if len(matches) != 1 or sha256_file(matches[0]) != digest:
            raise AssertionError(f"historical Manifest {number} changed")
    artifact_root = ROOT / "docs/validation/m0-step3g1b-canonical-jacobi-tangent-primitives-v1"
    artifact_hashes = strict_json(artifact_root / "artifact_hashes.json")["sha256"]
    for name, digest in artifact_hashes.items():
        expected[str((artifact_root / name).relative_to(ROOT))] = digest
    mismatches = {
        relative: {"expected": digest, "observed": sha256_file(ROOT / relative)}
        for relative, digest in expected.items()
        if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != digest
    }
    if mismatches:
        raise AssertionError(f"inherited hash mismatch: {mismatches}")
    if prior["status"] != "PASS":
        raise AssertionError("Step 3g1b inherited integrity did not pass")
    return {
        "status": "PASS",
        "checked_hashes": prior["checked_hashes"] + len(expected) + 12,
        "protected_tags": prior["protected_tags"],
        "historical_manifests": 12,
    }
