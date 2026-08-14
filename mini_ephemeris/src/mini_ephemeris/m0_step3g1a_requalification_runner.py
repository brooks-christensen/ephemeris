"""Exact-node guarded runner for the Step 3g1a requalification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]


def _bootstrap_namespace() -> None:
    import importlib.machinery
    import types

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

from mini_ephemeris.m0_step3g1a_requalification import (  # noqa: E402
    PYTEST_SITE_PACKAGES,
    PYTEST_VERSION,
    assert_protected_runtime_absent,
    install_guard,
    manifest23,
    static_safety_audit,
)


def _run_pytest(node_ids: list[str]) -> int:
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
    result = pytest.main(["-q", *node_ids])
    assert_protected_runtime_absent()
    return int(result)


def main(argv: list[str] | None = None) -> int:
    """Run the static gate or one exact preregistered pytest node list."""

    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--safety-audit", action="store_true")
    modes.add_argument("--run-foundation", action="store_true")
    modes.add_argument("--run-artifacts", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.safety_audit:
        guard = install_guard()
        guard.activate_strict()
        result = static_safety_audit()
        assert_protected_runtime_absent()
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    selection = manifest23()["exact_test_selection"]
    if arguments.run_foundation:
        nodes = list(selection["foundation_node_ids"]) + list(
            selection["requalification_node_ids"]
        )
        return _run_pytest(nodes)
    return _run_pytest(list(selection["artifact_node_ids"]))


if __name__ == "__main__":
    raise SystemExit(main())
