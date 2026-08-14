"""Exact-node guarded runner for Step 3g1c qualification."""

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

from mini_ephemeris.m0_step3g1c_qualification import (  # noqa: E402
    PYTEST_SITE_PACKAGES,
    PYTEST_VERSION,
    assert_protected_runtime_absent,
    install_guard,
    manifest25,
    static_safety_audit,
)


def _run_pytest(
    node_ids: list[str],
    inherited_step3g1b_nodes: list[str] | None = None,
) -> int:
    guard = install_guard()
    import numpy

    sys.path.insert(0, str(PYTEST_SITE_PACKAGES))
    import pytest

    if (
        pytest.__version__ != PYTEST_VERSION
        or Path(pytest.__file__).parent.parent != PYTEST_SITE_PACKAGES
    ):
        raise RuntimeError("audited pytest runtime identity changed")
    del numpy
    guard.activate_strict()
    static_safety_audit()
    results = []
    if inherited_step3g1b_nodes:
        module_name = "mini_ephemeris.m0_step3g1c_qualification"
        qualification_module = sys.modules.pop(module_name)
        try:
            results.append(pytest.main(["-q", *inherited_step3g1b_nodes]))
        finally:
            sys.modules[module_name] = qualification_module
        sys.meta_path[:] = [
            finder for finder in sys.meta_path
            if finder.__class__.__name__ != "Step3g1bImportGuard"
        ]
    results.append(pytest.main(["-q", *node_ids]))
    assert_protected_runtime_absent()
    return max(int(result) for result in results)


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
    selection = manifest25()["exact_test_selection"]
    if arguments.run_core:
        nodes = (
            list(selection["step3g1c_core_node_ids"])
            + list(selection["step3g1c_integrity_node_ids"])
            + list(selection["safe_step3g1a_regression_node_ids"])
        )
        return _run_pytest(
            nodes, list(selection["safe_step3g1b_regression_node_ids"])
        )
    return _run_pytest(list(selection["artifact_node_ids"]))


if __name__ == "__main__":
    raise SystemExit(main())
