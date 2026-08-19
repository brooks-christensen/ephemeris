"""Fresh-process exact-node runner for Step 3g1d requalification."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
MANIFEST28 = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "28_m0_step3g1d_interaction_kick_requalification_v1.json"
)
GROUPS: Mapping[str, Mapping[str, Any]] = {
    "step3g1d_requalification": {
        "expected_count_key": "step3g1d_fresh_process_group",
        "hash_seed": "28101",
        "helper": "mini_ephemeris.m0_step3g1d_qualification",
        "keys": ("step3g1d_core_node_ids", "step3g1d_integrity_node_ids"),
    },
    "safe_step3g1a": {
        "expected_count_key": "safe_step3g1a_regression",
        "hash_seed": "28102",
        "helper": "mini_ephemeris.m0_step3g1a_requalification",
        "keys": ("safe_step3g1a_regression_node_ids",),
    },
    "safe_step3g1b": {
        "expected_count_key": "safe_step3g1b_regression",
        "hash_seed": "28103",
        "helper": "mini_ephemeris.m0_step3g1b_qualification",
        "keys": ("safe_step3g1b_regression_node_ids",),
    },
    "safe_step3g1c": {
        "expected_count_key": "safe_step3g1c_regression",
        "hash_seed": "28104",
        "helper": "mini_ephemeris.m0_step3g1c_qualification",
        "keys": ("safe_step3g1c_regression_node_ids",),
    },
}


def _bootstrap_namespace() -> None:
    if "mini_ephemeris" in sys.modules:
        raise RuntimeError("fresh worker already contains mini_ephemeris")
    package_path = ROOT / "mini_ephemeris/src/mini_ephemeris"
    package = types.ModuleType("mini_ephemeris")
    package.__package__ = "mini_ephemeris"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "mini_ephemeris", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["mini_ephemeris"] = package


def _manifest28() -> Mapping[str, Any]:
    value = json.loads(
        MANIFEST28.read_text(encoding="utf-8"),
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise TypeError("Manifest 28 must be a JSON object")
    return value


def _audited_pytest(helper):
    guard = helper.install_guard()
    importlib.import_module("numpy")
    site_packages = helper.PYTEST_SITE_PACKAGES
    sys.path.insert(0, str(site_packages))
    import pytest

    if (
        pytest.__version__ != helper.PYTEST_VERSION
        or Path(pytest.__file__).parent.parent != site_packages
    ):
        raise RuntimeError("audited pytest runtime identity changed")
    guard.activate_strict()
    return pytest


def _expected_passes(group_name: str) -> int:
    config = GROUPS[group_name]
    return int(
        _manifest28()["exact_test_selection"]["expected_counts"][
            config["expected_count_key"]
        ]
    )


def _group_nodes(group_name: str) -> list[str]:
    config = GROUPS[group_name]
    selection = _manifest28()["exact_test_selection"]
    nodes = [
        node
        for key in config["keys"]
        for node in selection[key]
    ]
    expected = _expected_passes(group_name)
    if len(nodes) != expected or len(nodes) != len(set(nodes)):
        raise RuntimeError(f"{group_name} literal-node count changed")
    return nodes


def _verify_frozen_groups() -> None:
    groups = _manifest28()["fresh_process_regression_isolation"]["groups"]
    observed = {
        value["name"]: {
            "hash_seed": value["hash_seed"],
            "expected_passes": value["expected_passes"],
        }
        for value in groups
    }
    expected = {
        name: {
            "hash_seed": config["hash_seed"],
            "expected_passes": _expected_passes(name),
        }
        for name, config in GROUPS.items()
    }
    if observed != expected:
        raise RuntimeError("fresh-process group contract changed")


def _run_worker(group_name: str) -> int:
    config = GROUPS[group_name]
    expected_environment = {
        "PYTHONHASHSEED": config["hash_seed"],
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(ROOT / "mini_ephemeris/src"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    observed_environment = {
        name: os.environ.get(name) for name in expected_environment
    }
    if observed_environment != expected_environment:
        raise RuntimeError(
            f"{group_name} worker environment changed: {observed_environment}"
        )

    _bootstrap_namespace()
    helper = importlib.import_module(config["helper"])
    pytest = _audited_pytest(helper)
    if group_name == "step3g1d_requalification":
        helper.static_safety_audit()
    nodes = _group_nodes(group_name)
    result = int(pytest.main(["-q", *nodes]))
    helper.assert_protected_runtime_absent()
    print(
        json.dumps(
            {
                "expected_passes": _expected_passes(group_name),
                "group": group_name,
                "pytest_exit_code": result,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return result


def _run_group_worker(group_name: str) -> int:
    config = GROUPS[group_name]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONHASHSEED": config["hash_seed"],
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(ROOT / "mini_ephemeris/src"),
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-group",
            group_name,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return int(result.returncode)


def run_core() -> int:
    _verify_frozen_groups()
    results = []
    for group_name in GROUPS:
        results.append(_run_group_worker(group_name))
    return max(results, default=0)


def run_safety_audit() -> int:
    _bootstrap_namespace()
    helper = importlib.import_module(
        "mini_ephemeris.m0_step3g1d_qualification"
    )
    _audited_pytest(helper)
    result = helper.static_safety_audit()
    helper.assert_protected_runtime_absent()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


def run_artifacts() -> int:
    _bootstrap_namespace()
    helper = importlib.import_module(
        "mini_ephemeris.m0_step3g1d_qualification"
    )
    pytest = _audited_pytest(helper)
    nodes = list(
        _manifest28()["exact_test_selection"]["artifact_node_ids"]
    )
    expected = int(
        _manifest28()["exact_test_selection"]["expected_counts"]["artifact"]
    )
    if len(nodes) != expected or len(nodes) != len(set(nodes)):
        raise RuntimeError("artifact literal-node count changed")
    result = int(pytest.main(["-q", *nodes]))
    helper.assert_protected_runtime_absent()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--safety-audit", action="store_true")
    modes.add_argument("--run-core", action="store_true")
    modes.add_argument("--run-artifacts", action="store_true")
    modes.add_argument("--worker-group", choices=tuple(GROUPS))
    arguments = parser.parse_args(argv)
    if arguments.worker_group is not None:
        return _run_worker(arguments.worker_group)
    if arguments.safety_audit:
        return run_safety_audit()
    if arguments.run_core:
        return run_core()
    return run_artifacts()


if __name__ == "__main__":
    raise SystemExit(main())
