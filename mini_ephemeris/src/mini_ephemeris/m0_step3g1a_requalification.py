"""Source-only safety and integrity helpers for Step 3g1a requalification."""

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
PYTEST_SITE_PACKAGES = Path("/home/peacelovephysics/sheet-music-generator/.venv/lib/python3.10/site-packages")
PYTEST_VERSION = "8.4.2"
PYTEST_IDENTITY_SHA256 = {
    "pytest/__init__.py": "66993a5e3905005e0981159b4794d10b1adacf341a58a44d696ad2c4442dcdc6",
    "pytest-8.4.2.dist-info/METADATA": "93d6c5ef0a9714d53716243035037b77fa7d5f970596c48433887cf57f7f675a",
    "pytest-8.4.2.dist-info/RECORD": "fba86b3aa6d34c9d73bc8b2ea69d6d69a65cd555c4ac84ef15a330232706f82f",
}
MANIFEST23 = (
    ROOT
    / "ephemeris_experiment_runner/manifests/"
    "23_m0_step3g1a_v2_foundation_requalification_v1.json"
)

LOCAL_ALLOWED_PREFIXES = (
    "mini_ephemeris.v2",
    "mini_ephemeris.m0_step3g1a_requalification",
    "mini_ephemeris.m0_step3g1a_requalification_runner",
    "mini_ephemeris.m0_step3g1a_requalification_reporting",
)
APPROVED_DEPENDENCY_ROOTS = {
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "packaging",
    "pygments",
    "tomli",
    "exceptiongroup",
    "typing_extensions",
}
TEST_MODULES = {
    "test_v2_foundation",
    "test_m0_step3g1a_requalification",
    "test_m0_step3g1a_requalification_artifacts",
}
EXPLICIT_FORBIDDEN_MODULES = (
    "requalification_forbidden_sentinel",
    "rebound",
    "reboundx",
)
FORBIDDEN_LIBRARY_MARKERS = (
    "requalification_forbidden_library_sentinel",
    "libmini_ephemeris_gr_tangent",
    "gr_potential_tangent.so",
    "librebound",
    "libreboundx",
)
FORBIDDEN_EXECUTABLE_NAMES = {
    "integrate",
    "step",
    "kepler",
    "kick",
    "lazy",
    "corrector",
    "whckl",
    "megno",
    "lcn",
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    """Read strict JSON and reject nonstandard numeric constants."""

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def manifest23() -> Mapping[str, Any]:
    """Return the frozen requalification manifest."""

    value = strict_json(MANIFEST23)
    if not isinstance(value, dict):
        raise TypeError("Manifest 23 must be a JSON object")
    return value


def pytest_runtime_audit() -> Mapping[str, Any]:
    """Verify the existing external pytest runtime without importing it."""

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
        "qualifying_commands_plugin_autoload_disabled": all(
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in manifest23()["exact_commands"][name]
            for name in ("static_safety_gate", "foundation_campaign", "artifact_campaign")
        ),
    }


def install_namespace_shell(root: Path = ROOT) -> None:
    """Install a package shell without executing legacy package initialization."""

    existing = sys.modules.get("mini_ephemeris")
    if existing is not None:
        package_file = getattr(existing, "__file__", None)
        if package_file is not None:
            raise RuntimeError("legacy mini_ephemeris package initialization already executed")
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
    return fullname.startswith("mini_ephemeris.") and not _is_allowed_local_module(fullname)


def reject_forbidden_library_path(path: object) -> None:
    """Reject a protected compiled-library marker without loading that library."""

    if path is None:
        return
    normalized = os.fsdecode(path).lower()
    if any(marker in normalized for marker in FORBIDDEN_LIBRARY_MARKERS):
        raise RuntimeError(f"forbidden compiled library path: {normalized}")


def loaded_library_violations(maps_text: str | None = None) -> tuple[str, ...]:
    """Return forbidden markers found in the current Linux process map."""

    if maps_text is None:
        maps_path = Path("/proc/self/maps")
        maps_text = maps_path.read_text(encoding="utf-8") if maps_path.exists() else ""
    lowered = maps_text.lower()
    return tuple(marker for marker in FORBIDDEN_LIBRARY_MARKERS if marker in lowered)


class RequalificationImportGuard(importlib.abc.MetaPathFinder):
    """Deny forbidden imports and optionally reject every unknown import root."""

    def __init__(self) -> None:
        self.strict = False
        self.preloaded_roots: set[str] = set()

    def activate_strict(self) -> None:
        """Allow only stdlib, preloaded approved dependencies, tests, and v2 helpers."""

        self.preloaded_roots = {name.split(".", 1)[0] for name in sys.modules}
        self.strict = True

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> None:
        """Raise before a forbidden or unapproved module can be resolved."""

        del path, target
        if _is_forbidden_module(fullname):
            raise ImportError(f"Manifest 23 forbids importing {fullname!r}")
        if not self.strict:
            return None
        root = fullname.split(".", 1)[0]
        allowed = (
            root in sys.stdlib_module_names
            or root in self.preloaded_roots
            or root in APPROVED_DEPENDENCY_ROOTS
            or root in TEST_MODULES
            or _is_allowed_local_module(fullname)
        )
        if not allowed:
            raise ImportError(f"Manifest 23 import root is not allowlisted: {fullname!r}")
        return None


_ACTIVE_GUARD: RequalificationImportGuard | None = None
_AUDIT_HOOK_INSTALLED = False


def install_guard() -> RequalificationImportGuard:
    """Install the deny-first import and compiled-library guards once."""

    global _ACTIVE_GUARD, _AUDIT_HOOK_INSTALLED
    install_namespace_shell()
    if _ACTIVE_GUARD is None:
        _ACTIVE_GUARD = RequalificationImportGuard()
        sys.meta_path.insert(0, _ACTIVE_GUARD)
    if not _AUDIT_HOOK_INSTALLED:
        def audit_hook(event: str, args: tuple[object, ...]) -> None:
            if event == "ctypes.dlopen" and args:
                reject_forbidden_library_path(args[0])

        sys.addaudithook(audit_hook)
        _AUDIT_HOOK_INSTALLED = True
    assert_protected_runtime_absent()
    return _ACTIVE_GUARD


def active_guard() -> RequalificationImportGuard:
    """Return the installed guard or fail closed."""

    if _ACTIVE_GUARD is None:
        raise RuntimeError("requalification guard is not installed")
    return _ACTIVE_GUARD


def assert_protected_runtime_absent() -> None:
    """Require no forbidden Python module, legacy package, or library mapping."""

    forbidden = sorted(name for name in sys.modules if _is_forbidden_module(name))
    if forbidden:
        raise RuntimeError(f"forbidden modules are loaded: {forbidden}")
    if "mini_ephemeris.nbody" in sys.modules:
        raise RuntimeError("legacy mini_ephemeris.nbody is loaded")
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
            prefix = "." * node.level
            imports.add(prefix + (node.module or ""))
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
            path,
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        _find_method(tree, class_name, method_name)


def _selected_historical_nodes_have_no_subprocess(manifest: Mapping[str, Any]) -> None:
    nodes = manifest["exact_test_selection"]["foundation_node_ids"]
    path = ROOT / manifest["exact_test_selection"]["historical_test_file"]["path"]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node_id in nodes:
        _, class_name, method_name = node_id.split("::")
        method = _find_method(tree, class_name, method_name)
        calls = {
            child.func.attr
            for child in ast.walk(method)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"run", "Popen", "check_call", "check_output"}
        }
        helper_calls = {
            child.func.attr
            for child in ast.walk(method)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "_run_probe"
        }
        if calls or helper_calls:
            raise AssertionError(f"selected historical node can spawn a subprocess: {node_id}")


def _subprocess_call_inventory(path: Path) -> tuple[str, ...]:
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
        while owner is not None and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parents.get(owner)
        owner_name = owner.name if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"
        found.append(f"{path.relative_to(ROOT)}:{owner_name}:{node.func.attr}")
    return tuple(sorted(found))


def static_safety_audit() -> Mapping[str, Any]:
    """Audit the exact source, node, import, and subprocess closure without imports."""

    manifest = manifest23()
    expected_v2 = manifest["v2_implementation_sha256"]
    actual_v2_paths = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "mini_ephemeris/src/mini_ephemeris/v2").iterdir()
        if path.is_file() and path.suffix == ".py"
    }
    if actual_v2_paths != set(expected_v2):
        raise AssertionError("v2 source inventory differs from Manifest 23")
    for relative, expected in expected_v2.items():
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"v2 source hash changed: {relative}")
    historical_test = manifest["exact_test_selection"]["historical_test_file"]
    if sha256_file(ROOT / historical_test["path"]) != historical_test["sha256"]:
        raise AssertionError("historical foundation test hash changed")

    source_paths = [ROOT / value for value in manifest["import_and_library_safety"]["static_source_files"]]
    for path in source_paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node_groups = manifest["exact_test_selection"]
    all_nodes = (
        list(node_groups["foundation_node_ids"])
        + list(node_groups["requalification_node_ids"])
        + list(node_groups["artifact_node_ids"])
    )
    _verify_node_ids(all_nodes)
    _selected_historical_nodes_have_no_subprocess(manifest)

    helper_path = ROOT / "mini_ephemeris/src/mini_ephemeris/m0_step3g1a_requalification.py"
    active_subprocesses = _subprocess_call_inventory(helper_path)
    expected_owners = {
        "run_fresh_v2_probe:run",
        "run_fresh_artifact_probe:run",
        "git_output:check_output",
    }
    observed_owners = {":".join(value.split(":")[-2:]) for value in active_subprocesses}
    if observed_owners != expected_owners:
        raise AssertionError(f"subprocess closure changed: {active_subprocesses}")

    legacy_init = ROOT / "mini_ephemeris/src/mini_ephemeris/__init__.py"
    legacy_tree = ast.parse(legacy_init.read_text(encoding="utf-8"), filename=str(legacy_init))
    imports_nbody = any(
        isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == "nbody"
        for node in ast.walk(legacy_tree)
    )
    if not imports_nbody:
        raise AssertionError("legacy package import hazard no longer matches preregistration")
    if "mini_ephemeris.nbody" in sys.modules:
        raise AssertionError("namespace bootstrap did not isolate legacy nbody")

    import_graph = {
        str(path.relative_to(ROOT)): list(_imports_for(path))
        for path in sorted(source_paths)
    }
    prohibited_imports = sorted(
        module
        for modules in import_graph.values()
        for module in modules
        if module in {"rebound", "reboundx"}
        or module.startswith("mini_ephemeris.gr_")
        or module.startswith("mini_ephemeris.rebound_")
        or module.startswith("mini_ephemeris.m0_step3g0")
    )
    if prohibited_imports:
        raise AssertionError(f"static import graph is unsafe: {prohibited_imports}")
    return {
        "status": "PASS",
        "pytest_runtime": pytest_runtime_audit(),
        "approved_dependency_roots": sorted(APPROVED_DEPENDENCY_ROOTS),
        "source_file_count": len(source_paths),
        "selected_node_count": len(all_nodes),
        "foundation_node_count": len(node_groups["foundation_node_ids"]),
        "requalification_node_count": len(node_groups["requalification_node_ids"]),
        "artifact_node_count": len(node_groups["artifact_node_ids"]),
        "import_graph": import_graph,
        "active_subprocess_call_sites": list(active_subprocesses),
        "dormant_historical_subprocess_nodes": node_groups["excluded_historical_nodes"][:2],
        "legacy_package_init_bypassed": True,
        "legacy_nbody_absent": "mini_ephemeris.nbody" not in sys.modules,
        "forbidden_imports": [],
        "forbidden_library_mappings": list(loaded_library_violations()),
    }


GUARDED_V2_PROBE_SOURCE = r"""
import importlib.machinery
import json
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
from mini_ephemeris.m0_step3g1a_requalification import assert_protected_runtime_absent, install_guard
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.v2 import CompiledLayout, ExactSeconds, MacroTimebase, PhysicalModel, SI_UNITS
layout = CompiledLayout(["sun", "planet"], "sun")
model = PhysicalModel(model_id="probe", schema_version="1", layout=layout,
    masses_kg={"planet": 2.0, "sun": 1.0}, gravitational_constant_si=6.67430e-11,
    units=SI_UNITS, enabled_effects={"effect-b", "effect-a"},
    provenance={"z": "last", "a": "first"})
timebase = MacroTimebase(ExactSeconds(-1, 3), ExactSeconds(1, 10), 2**62)
assert_protected_runtime_absent()
print(json.dumps({"legacy_nbody": "mini_ephemeris.nbody" in sys.modules,
    "model": model.fingerprint, "time": timebase.fingerprint,
    "target": timebase.at(2**60).canonical_payload()}, sort_keys=True, separators=(",", ":")))
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
from mini_ephemeris.m0_step3g1a_requalification import assert_protected_runtime_absent, install_guard
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.m0_step3g1a_requalification_reporting import generate_artifacts
generate_artifacts(destination)
assert_protected_runtime_absent()
""".strip()


def run_fresh_v2_probe(hash_seed: int, locale_name: str) -> str:
    """Run the exact guarded v2 fresh-process probe."""

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["LC_ALL"] = locale_name
    result = subprocess.run(
        [sys.executable, "-I", "-c", GUARDED_V2_PROBE_SOURCE, str(ROOT)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_fresh_artifact_probe(destination: Path) -> None:
    """Generate artifacts in one exact guarded fresh-process probe."""

    environment = os.environ.copy()
    environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    environment["PYTHONHASHSEED"] = "0"
    environment["LC_ALL"] = "C"
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            GUARDED_ARTIFACT_PROBE_SOURCE,
            str(ROOT),
            str(destination),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def git_output(arguments: Sequence[str]) -> str:
    """Run one preregistered read-only git identity command."""

    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()
