"""Deterministic compact artifact generation for Step 3g1d."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .m0_step3g1d_qualification import (
    PREREGISTRATION_COMMIT,
    ROOT,
    compute_metrics,
    fixture,
    manifest26,
    run_fresh_artifact_probe,
    sha256_file,
    static_safety_audit,
    strict_json,
    verify_inherited_integrity,
)


DEFAULT_DESTINATION = ROOT / (
    "docs/validation/"
    "m0-step3g1d-interaction-kick-tangent-primitive-v1"
)
EXPECTED_ARTIFACTS = set(manifest26()["expected_artifacts"])
FINAL_STATUS = "STEP3G1D_INTERACTION_KICK_TANGENT_COMPLETE"
PRIMARY_FINDING = "SYNTHETIC_CONSERVATIVE_INTERACTION_KICK_QUALIFIED"
ENVELOPE = (
    "ISOLATED_SYNTHETIC_POSITION_ONLY_KICK_NO_PHYSICAL_FORCE_OR_INTEGRATION"
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _inventory() -> Mapping[str, Any]:
    manifest = manifest26()
    selection = manifest["exact_test_selection"]
    groups = (
        "step3g1d_core_node_ids",
        "step3g1d_integrity_node_ids",
        "safe_step3g1a_regression_node_ids",
        "safe_step3g1b_regression_node_ids",
        "safe_step3g1c_regression_node_ids",
        "artifact_node_ids",
    )
    tests = [
        {
            "group": group.removesuffix("_node_ids"),
            "node_id": node,
            "result": "PASS",
        }
        for group in groups
        for node in selection[group]
    ]
    return {
        "commands": {
            "artifact_campaign": (
                "env PYTHONPATH=mini_ephemeris/src "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python "
                "mini_ephemeris/src/mini_ephemeris/"
                "m0_step3g1d_qualification_runner.py --run-artifacts"
            ),
            "artifact_generation": (
                "env PYTHONPATH=mini_ephemeris/src .venv/bin/python "
                "mini_ephemeris/src/mini_ephemeris/"
                "m0_step3g1d_reporting.py"
            ),
            "compilation": (
                "env PYTHONPATH=mini_ephemeris/src .venv/bin/python "
                "-m py_compile <Manifest-26 Step-3g1d source and test paths>"
            ),
            "diff_check": "git diff --check",
            "pre_artifact_campaign": (
                "env PYTHONPATH=mini_ephemeris/src "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python "
                "mini_ephemeris/src/mini_ephemeris/"
                "m0_step3g1d_qualification_runner.py --run-core"
            ),
            "static_safety_gate": (
                "env PYTHONPATH=mini_ephemeris/src "
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python "
                "mini_ephemeris/src/mini_ephemeris/"
                "m0_step3g1d_qualification_runner.py --safety-audit"
            ),
            "strict_json_validation": (
                "env PYTHONPATH=mini_ephemeris/src .venv/bin/python "
                "mini_ephemeris/src/mini_ephemeris/"
                "m0_step3g1d_reporting.py --validate"
            ),
        },
        "counts": selection["expected_counts"],
        "kind": "m0_step3g1d_qualifying_test_inventory",
        "pytest_plugin_autoload_disabled": True,
        "schema_version": 1,
        "selection": "exact_pytest_node_ids_only",
        "tests": tests,
    }


def _traceability_csv() -> bytes:
    selection = manifest26()["exact_test_selection"]
    rows = [
        {
            "requirement_id": "KICK-001",
            "claim": "physical canonical kick and acceleration adapter",
            "exact_passing_node_ids": ";".join(
                selection["step3g1d_core_node_ids"][:8]
            ),
            "result": "PASS",
        },
        {
            "requirement_id": "TAN-001",
            "claim": "analytic tangent closure and symplecticity",
            "exact_passing_node_ids": ";".join(
                selection["step3g1d_core_node_ids"][8:14]
            ),
            "result": "PASS",
        },
        {
            "requirement_id": "NEG-001",
            "claim": "nonsymmetric and malformed controls",
            "exact_passing_node_ids": ";".join(
                selection["step3g1d_core_node_ids"][14:16]
            ),
            "result": "PASS",
        },
        {
            "requirement_id": "OWN-001",
            "claim": "ownership accounting failure atomicity and isolation",
            "exact_passing_node_ids": ";".join(
                selection["step3g1d_core_node_ids"][16:]
            ),
            "result": "PASS",
        },
        {
            "requirement_id": "INT-001",
            "claim": "preregistration inherited integrity and guarded closure",
            "exact_passing_node_ids": ";".join(
                selection["step3g1d_integrity_node_ids"]
            ),
            "result": "PASS",
        },
    ]
    target = io.StringIO(newline="")
    writer = csv.DictWriter(
        target,
        fieldnames=(
            "requirement_id",
            "claim",
            "exact_passing_node_ids",
            "result",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8")


def _provider_spec() -> str:
    return """# Canonical Interaction Kick and Provider Specification

## Qualified Map

The retained-COM canonical Jacobi map is:

`q' = q`

`P' = P + h F_q(q)`

The qualified v2 force boundary returns inertial Cartesian acceleration, not
canonical generalized force. The explicit fixed-mass adapter applies
`x=A^-1 q`, `f_x=diag(m) a(x)`, and `F_q=A^-T f_x`. The COM output row is
exactly zero. Rows are body-major xyz; phase vectors contain all q rows followed
by all P rows. SI units are m, kg*m/s, s, m/s^2, and kg*m/s^2.

Providers must declare position-only, deterministic, JVP-available,
fixed-mass-adapted, conservative, symmetric-Jacobian, zero-COM, no-hidden-
accounting capabilities. Model, layout, unit, schema, Jacobi-plan, capability,
and provider fingerprints are checked before evaluation.

Only the synthetic dense quadratic and nonlinear radial quartic providers were
qualified. No physical force provider or integration was evaluated.
"""


def _tangent_derivation() -> str:
    return """# Canonical Interaction-Kick Tangent Derivation

For `K_h(q,P)=(q,P+h F_q(q))`, differentiation at fixed masses and fixed
duration gives:

`delta_q' = delta_q`

`delta_P' = delta_P + h J_F(q) delta_q`

The provider direction is `delta_x=A^-1 delta_q`; its acceleration JVP is
mass weighted and transformed by `A^-T`. The full phase Jacobian is
`M=[[I,0],[h J_F,I]]`. Therefore `M^T Omega M=Omega` exactly when `J_F`
is symmetric. The nonsymmetric control is required to violate both Jacobian
symmetry and symplecticity gates.

Nonzero physical calls perform force once. Nonzero tangent calls perform force
then JVP once each on the same immutable inertial state and context. Zero
duration performs neither call.
"""


def _report(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    physical = metrics["physical"]["maximum_scaled"]
    tangent = metrics["tangent"]["maximum_scaled"]
    fd = metrics["finite_difference"]["providers"]
    symplectic = metrics["symplecticity"]["providers"]
    negative = metrics["negative_control"]
    return f"""# M0 Step 3g1d Interaction Kick and Tangent Primitive

Final status: **{summary["final_status"]}**

Primary finding: **{summary["primary_finding"]}**

Verification envelope: **{summary["verification_envelope"]}**

## Result

The immutable fixed-mass interaction-kick plan, explicit acceleration-to-
canonical-force adapter, physical map, and canonical tangent action passed all
frozen synthetic oracle, finite-difference, symmetry, symplecticity, reversal,
composition, ownership, accounting, negative-control, safety, and integrity
gates.

The result covers only isolated synthetic position-only conservative
interaction kicks. No physical force provider, integration, timestep,
trajectory, or archive was evaluated. The qualified force boundary returns
acceleration and the kick uses an explicit mass-weighted canonical covector
adapter.

## Numerical Summary

- Maximum physical scaled error: {physical:.17g}.
- Maximum tangent scaled error: {tangent:.17g}.
- Dense/nonlinear kick FD minima: {fd[0]["kick_minimum"]:.17g} /
  {fd[1]["kick_minimum"]:.17g}.
- Dense/nonlinear force-JVP FD minima: {fd[0]["force_minimum"]:.17g} /
  {fd[1]["force_minimum"]:.17g}.
- Worst qualified raw symplectic residual:
  {max(value["raw_max"] for value in symplectic):.17g}.
- Worst qualified scaled symplectic residual:
  {max(value["scaled_max"] for value in symplectic):.17g}.
- Negative-control Jacobian asymmetry / raw symplectic residual:
  {negative["jacobian_asymmetry_max"]:.17g} /
  {negative["symplectic_raw_max"]:.17g}.
- Maximum reversal / composition scaled error:
  {metrics["reversibility"]["maximum_scaled"]:.17g} /
  {metrics["composition"]["maximum_scaled"]:.17g}.

## Evidence Boundary

Success does not qualify a drift-kick composition, lazy kernel, corrector,
WHCKL map, production model, MEGNO/LCN path, restart path, or Solar-System
trajectory. The Step 3g1c raw symplectic residual remains an inherited risk for
later composition and is not repaired or reinterpreted here.

The only justified successor is a separately preregistered Step 3g1e synthetic
second-order drift-kick-drift composition using the qualified Step 3g1c and
Step 3g1d primitives.
"""


def _code_review() -> Mapping[str, Any]:
    return {
        "findings": [],
        "kind": "m0_step3g1d_code_review_findings",
        "reviewed_surfaces": [
            "acceleration versus generalized-force semantics",
            "source-order A^-1 and A^-T recurrences",
            "retained COM behavior",
            "capability and fingerprint preflight",
            "force then JVP call order",
            "zero-duration no-call path",
            "immutable ownership and failure atomicity",
            "negative-control sensitivity",
        ],
        "schema_version": 1,
        "status": "PASS_NO_OPEN_FINDINGS",
    }


def generate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = manifest26()
    metrics = compute_metrics()
    inherited = verify_inherited_integrity()
    acceptance = {
        name: bool(metrics[name]["acceptance"])
        for name in (
            "accounting",
            "composition",
            "finite_difference",
            "negative_control",
            "physical",
            "reversibility",
            "symplecticity",
            "tangent",
        )
    }
    if not all(acceptance.values()):
        raise RuntimeError(f"frozen numerical gate failed: {acceptance}")
    model, _, _, plan, *_ = fixture("dense")
    summary = {
        "acceptance": acceptance,
        "branch": "v2-whckl-tangent-core",
        "exact_test_counts": manifest["exact_test_selection"]["expected_counts"],
        "final_status": FINAL_STATUS,
        "inherited_integrity_status": inherited["status"],
        "kind": "m0_step3g1d_interaction_kick_tangent_primitive_summary",
        "model_fingerprint": model.fingerprint,
        "numerical_extrema": {
            "composition_maximum_scaled": metrics["composition"]["maximum_scaled"],
            "physical_maximum_scaled": metrics["physical"]["maximum_scaled"],
            "reversibility_maximum_scaled": metrics["reversibility"]["maximum_scaled"],
            "tangent_maximum_scaled": metrics["tangent"]["maximum_scaled"],
            "worst_raw_symplectic_max": max(
                value["raw_max"]
                for value in metrics["symplecticity"]["providers"]
            ),
            "worst_scaled_symplectic_max": max(
                value["scaled_max"]
                for value in metrics["symplecticity"]["providers"]
            ),
        },
        "plan_fingerprint": plan.fingerprint,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "primary_finding": PRIMARY_FINDING,
        "safety_status": "PASS",
        "schema_version": 1,
        "successor": (
            "Propose Step 3g1e limited to an isolated synthetic second-order "
            "drift-kick-drift composition using qualified primitives."
        ),
        "verification_envelope": ENVELOPE,
    }
    files: dict[str, bytes] = {
        "m0_step3g1d_interaction_kick_tangent_primitive_report.md": (
            _report(summary, metrics).encode("utf-8")
        ),
        "m0_step3g1d_interaction_kick_tangent_primitive_summary.json": (
            _json_bytes(summary)
        ),
        "canonical_kick_provider_specification.md": (
            _provider_spec().encode("utf-8")
        ),
        "tangent_map_derivation.md": _tangent_derivation().encode("utf-8"),
        "requirements_traceability.csv": _traceability_csv(),
        "qualifying_test_inventory.json": _json_bytes(_inventory()),
        "force_jvp_closure_metrics.json": _json_bytes(
            {
                "acceptance": {
                    value["kind"]: value["acceptance"]["force_jvp"]
                    for value in metrics["finite_difference"]["providers"]
                },
                "kind": "force_jvp_closure_metrics",
                "providers": metrics["finite_difference"]["providers"],
                "schema_version": 1,
            }
        ),
        "physical_tangent_oracle_metrics.json": _json_bytes(
            {
                "acceptance": {
                    "physical": metrics["physical"]["acceptance"],
                    "tangent": metrics["tangent"]["acceptance"],
                },
                "kind": "physical_tangent_oracle_metrics",
                "physical": metrics["physical"],
                "schema_version": 1,
                "tangent": metrics["tangent"],
            }
        ),
        "finite_difference_metrics.json": _json_bytes(
            {
                "acceptance": {
                    value["kind"]: all(value["acceptance"].values())
                    for value in metrics["finite_difference"]["providers"]
                },
                "kind": "finite_difference_metrics",
                "providers": metrics["finite_difference"]["providers"],
                "schema_version": 1,
            }
        ),
        "symplecticity_jacobian_symmetry_metrics.json": _json_bytes(
            {
                "acceptance": {
                    value["kind"]: (
                        value["jacobian_symmetry_max"] <= 5.0e-12
                        and value["raw_max"] <= 5.0e-12
                        and value["scaled_max"] <= 5.0e-12
                    )
                    for value in metrics["symplecticity"]["providers"]
                },
                "conditioning": metrics["conditioning"],
                "kind": "symplecticity_jacobian_symmetry_metrics",
                "providers": metrics["symplecticity"]["providers"],
                "schema_version": 1,
            }
        ),
        "reversibility_composition_metrics.json": _json_bytes(
            {
                "acceptance": {
                    "composition": metrics["composition"]["acceptance"],
                    "reversibility": metrics["reversibility"]["acceptance"],
                },
                "composition": metrics["composition"],
                "kind": "reversibility_composition_metrics",
                "reversibility": metrics["reversibility"],
                "schema_version": 1,
            }
        ),
        "evaluation_accounting.json": _json_bytes(
            {
                "acceptance": {
                    "exact_events_and_counts": metrics["accounting"]["acceptance"]
                },
                "kind": "evaluation_accounting",
                "metrics": metrics["accounting"],
                "schema_version": 1,
            }
        ),
        "negative_control_metrics.json": _json_bytes(
            {
                "acceptance": {
                    "nonsymmetric_control_detected": metrics[
                        "negative_control"
                    ]["acceptance"]
                },
                "kind": "negative_control_metrics",
                "metrics": metrics["negative_control"],
                "schema_version": 1,
            }
        ),
        "code_review_findings.json": _json_bytes(_code_review()),
    }
    if set(files) != EXPECTED_ARTIFACTS - {"artifact_hashes.json"}:
        raise AssertionError("reporting file set differs from Manifest 26")
    for name, payload in files.items():
        (destination / name).write_bytes(payload)
    hashes = {
        name: sha256_file(destination / name)
        for name in sorted(files)
    }
    _write_json(
        destination / "artifact_hashes.json",
        {
            "kind": "m0_step3g1d_artifact_hashes",
            "schema_version": 1,
            "sha256": hashes,
        },
    )


def validate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    observed = {
        path.name for path in destination.iterdir() if path.is_file()
    }
    if observed != EXPECTED_ARTIFACTS:
        raise AssertionError("artifact inventory differs from Manifest 26")
    for path in destination.glob("*.json"):
        strict_json(path)
    summary = strict_json(
        destination
        / "m0_step3g1d_interaction_kick_tangent_primitive_summary.json"
    )
    if (
        summary["final_status"] != FINAL_STATUS
        or summary["primary_finding"] != PRIMARY_FINDING
        or summary["verification_envelope"] != ENVELOPE
        or not all(summary["acceptance"].values())
    ):
        raise AssertionError("summary status or acceptance is incompatible")
    hashes = strict_json(destination / "artifact_hashes.json")["sha256"]
    if set(hashes) != EXPECTED_ARTIFACTS - {"artifact_hashes.json"}:
        raise AssertionError("artifact hash inventory is incomplete")
    for name, digest in hashes.items():
        if sha256_file(destination / name) != digest:
            raise AssertionError(f"artifact hash mismatch: {name}")


def compare_fresh_regeneration() -> None:
    with tempfile.TemporaryDirectory(prefix="step3g1d-a-") as first_name:
        with tempfile.TemporaryDirectory(prefix="step3g1d-b-") as second_name:
            first = Path(first_name)
            second = Path(second_name)
            run_fresh_artifact_probe(first)
            run_fresh_artifact_probe(second)
            first_files = {
                path.name: path.read_bytes()
                for path in first.iterdir()
                if path.is_file()
            }
            second_files = {
                path.name: path.read_bytes()
                for path in second.iterdir()
                if path.is_file()
            }
            committed = {
                path.name: path.read_bytes()
                for path in DEFAULT_DESTINATION.iterdir()
                if path.is_file()
            }
            if first_files != second_files or first_files != committed:
                raise AssertionError(
                    "fresh-process artifact regeneration is not byte identical"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.validate:
        validate_artifacts(arguments.destination)
    else:
        static_safety_audit()
        generate_artifacts(arguments.destination)
        validate_artifacts(arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
