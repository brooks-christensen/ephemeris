"""Exact-node guarded runner for Step 3g1d qualification."""

from __future__ import annotations

import argparse
import importlib.machinery
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[3]


def _bootstrap_namespace() -> None:
    package_path = ROOT / "mini_ephemeris/src/mini_ephemeris"
    package = types.ModuleType("mini_ephemeris")
    package.__package__ = "mini_ephemeris"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "mini_ephemeris", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["mini_ephemeris"] = package


_bootstrap_namespace()

from mini_ephemeris.m0_step3g1d_qualification import (  # noqa: E402
    PYTEST_SITE_PACKAGES,
    PYTEST_VERSION,
    assert_protected_runtime_absent,
    install_guard,
    manifest26,
    static_safety_audit,
)


def _remove_guard(class_name: str) -> None:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__name__ != class_name
    ]


def _run_pytest(node_ids: list[str]) -> int:
    import pytest

    result = pytest.main(["-q", *node_ids])
    return int(result)


def _audited_pytest():
    guard = install_guard()
    sys.path.insert(0, str(PYTEST_SITE_PACKAGES))
    import pytest

    if (
        pytest.__version__ != PYTEST_VERSION
        or Path(pytest.__file__).parent.parent != PYTEST_SITE_PACKAGES
    ):
        raise RuntimeError("audited pytest runtime identity changed")
    guard.activate_strict()
    static_safety_audit()
    return pytest


def run_core() -> int:
    _audited_pytest()
    selection = manifest26()["exact_test_selection"]
    results = []
    results.append(
        _run_pytest(
            list(selection["step3g1d_core_node_ids"])
            + list(selection["step3g1d_integrity_node_ids"])
            + list(selection["safe_step3g1a_regression_node_ids"])
        )
    )
    results.append(
        _run_pytest(list(selection["safe_step3g1b_regression_node_ids"]))
    )
    _remove_guard("Step3g1bImportGuard")
    results.append(
        _run_pytest(list(selection["safe_step3g1c_regression_node_ids"]))
    )
    _remove_guard("Step3g1cImportGuard")
    assert_protected_runtime_absent()
    return max(results)


def run_artifacts() -> int:
    _audited_pytest()
    selection = manifest26()["exact_test_selection"]
    result = _run_pytest(list(selection["artifact_node_ids"]))
    assert_protected_runtime_absent()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--safety-audit", action="store_true")
    modes.add_argument("--run-core", action="store_true")
    modes.add_argument("--run-artifacts", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.safety_audit:
        guard = install_guard()
        guard.activate_strict()
        result = static_safety_audit()
        assert_protected_runtime_absent()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if arguments.run_core:
        return run_core()
    return run_artifacts()


if __name__ == "__main__":
    raise SystemExit(main())
