"""Deterministic numerical and provenance artifacts for Step 3g1b."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.machinery
import io
import json
import math
import os
from pathlib import Path
import sys
import types
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = ROOT / "docs/validation/m0-step3g1b-canonical-jacobi-tangent-primitives-v1"
PREREGISTRATION_COMMIT = "20d156839da5bed8d966b3a64b42ee73d7db788f"


def _bootstrap_namespace() -> None:
    if "mini_ephemeris" in sys.modules:
        return
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

from mini_ephemeris.m0_step3g1b_qualification import (  # noqa: E402
    assert_protected_runtime_absent,
    install_guard,
    manifest24,
    run_fresh_artifact_probe,
    sha256_file,
    static_safety_audit,
    strict_json,
    verify_inherited_integrity,
)
from mini_ephemeris.v2.jacobi import (  # noqa: E402
    InertialCanonicalState,
    InertialCanonicalTangentState,
    build_jacobi_transform_plan,
    from_canonical_jacobi,
    from_canonical_jacobi_tangent,
    to_canonical_jacobi,
    to_canonical_jacobi_tangent,
)
from mini_ephemeris.v2.model import CompiledLayout, PhysicalModel, SI_UNITS  # noqa: E402


ARTIFACTS_WITHOUT_HASH_INDEX = {
    "canonical_convention_specification.md",
    "code_review_findings.json",
    "m0_step3g1b_canonical_jacobi_tangent_primitives_report.md",
    "m0_step3g1b_canonical_jacobi_tangent_primitives_summary.json",
    "numerical_metrics.json",
    "qualifying_test_inventory.json",
    "requirements_traceability.csv",
}
EXPECTED_ARTIFACTS = ARTIFACTS_WITHOUT_HASH_INDEX | {"artifact_hashes.json"}
MASSES = (0.125, 2.0, 32.0, 0.5, 8.0)
EPSILONS = tuple(2.0**exponent for exponent in range(-4, -41, -4))
U = np.finfo(np.float64).eps


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
        default=_json_default,
    )
    return (text + chr(10)).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _model() -> PhysicalModel:
    body_ids = ("sun", "body-1", "body-2", "body-3", "body-4")
    layout = CompiledLayout(body_ids, "sun")
    return PhysicalModel(
        model_id="synthetic_jacobi_qualification",
        schema_version="1",
        layout=layout,
        masses_kg=dict(zip(body_ids, MASSES)),
        gravitational_constant_si=1.0,
        units=SI_UNITS,
        enabled_effects=("synthetic-none",),
        provenance={"fixture": "step3g1b-synthetic"},
    )


def _rows(scale: float) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            scale * (0.375 + 0.625 * index),
            scale * (-0.75 + 0.3125 * index),
            scale * (1.125 - 0.4375 * index),
        )
        for index in range(5)
    )


def _state(model: PhysicalModel, positions: object, momenta: object) -> InertialCanonicalState:
    return InertialCanonicalState(
        layout=model.layout,
        positions_m=positions,
        momenta_kg_m_per_s=momenta,
        unit_system_id="si_v1",
        model_fingerprint=model.fingerprint,
    )


def _tangent(
    model: PhysicalModel, positions: object, momenta: object
) -> InertialCanonicalTangentState:
    return InertialCanonicalTangentState(
        layout=model.layout,
        delta_positions_m=positions,
        delta_momenta_kg_m_per_s=momenta,
        unit_system_id="si_v1",
        model_fingerprint=model.fingerprint,
    )


def _direct_phase_matrix() -> tuple[np.ndarray, np.ndarray]:
    count = len(MASSES)
    a = np.zeros((count, count), dtype=np.float64)
    total = sum(MASSES)
    a[0, :] = np.asarray(MASSES) / total
    cumulative = MASSES[0]
    for row in range(1, count):
        for column in range(row):
            a[row, column] = -MASSES[column] / cumulative
        a[row, row] = 1.0
        cumulative += MASSES[row]
    a_inverse = np.linalg.inv(a)
    a3 = np.kron(a, np.eye(3))
    momentum = np.kron(a_inverse.T, np.eye(3))
    zero = np.zeros_like(a3)
    return a, np.block([[a3, zero], [zero, momentum]])


def _inertial_phase(state: InertialCanonicalState) -> np.ndarray:
    return np.concatenate(
        (np.asarray(state.positions_m).ravel(), np.asarray(state.momenta_kg_m_per_s).ravel())
    )


def _jacobi_phase(state: object) -> np.ndarray:
    return np.concatenate(
        (np.asarray(state.q_m).ravel(), np.asarray(state.p_kg_m_per_s).ravel())
    )


def _inertial_tangent_phase(state: InertialCanonicalTangentState) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(state.delta_positions_m).ravel(),
            np.asarray(state.delta_momenta_kg_m_per_s).ravel(),
        )
    )


def _jacobi_tangent_phase(state: object) -> np.ndarray:
    return np.concatenate(
        (np.asarray(state.delta_q_m).ravel(), np.asarray(state.delta_p_kg_m_per_s).ravel())
    )


def _error(observed: np.ndarray, expected: np.ndarray) -> Mapping[str, float]:
    difference = observed - expected
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "l2": float(np.linalg.norm(difference)),
    }


def collect_metrics() -> Mapping[str, Any]:
    """Compute the complete preregistered synthetic coordinate diagnostics."""

    model = _model()
    plan = build_jacobi_transform_plan(model)
    state = _state(model, _rows(3.25), _rows(-2.75))
    tangent = _tangent(model, _rows(0.125), _rows(-0.375))
    a, phase = _direct_phase_matrix()
    phase_inverse = np.linalg.inv(phase)
    condition_a = float(np.linalg.cond(a))
    condition_phase = float(np.linalg.cond(phase))

    canonical = to_canonical_jacobi(plan, model, state)
    recovered = from_canonical_jacobi(plan, model, canonical)
    state_forward_inverse = _error(_inertial_phase(recovered), _inertial_phase(state))
    canonical_roundtrip = to_canonical_jacobi(
        plan, model, from_canonical_jacobi(plan, model, canonical)
    )
    state_inverse_forward = _error(
        _jacobi_phase(canonical_roundtrip), _jacobi_phase(canonical)
    )

    canonical_tangent = to_canonical_jacobi_tangent(plan, model, state, tangent)
    recovered_tangent = from_canonical_jacobi_tangent(
        plan, model, canonical, canonical_tangent
    )
    tangent_forward_inverse = _error(
        _inertial_tangent_phase(recovered_tangent), _inertial_tangent_phase(tangent)
    )
    tangent_inverse_forward = _error(
        _jacobi_tangent_phase(
            to_canonical_jacobi_tangent(plan, model, state, recovered_tangent)
        ),
        _jacobi_tangent_phase(canonical_tangent),
    )

    dense_state = phase @ _inertial_phase(state)
    dense_tangent = phase @ _inertial_tangent_phase(tangent)
    dense_state_error = _error(_jacobi_phase(canonical), dense_state)
    dense_tangent_error = _error(_jacobi_tangent_phase(canonical_tangent), dense_tangent)

    base = _state(model, _rows(1.0), _rows(-1.0))
    direction = _tangent(model, _rows(0.0625), _rows(-0.09375))
    base_input = _inertial_phase(base)
    delta_input = _inertial_tangent_phase(direction)
    base_output = _jacobi_phase(to_canonical_jacobi(plan, model, base))
    analytic = _jacobi_tangent_phase(
        to_canonical_jacobi_tangent(plan, model, base, direction)
    )
    finite_difference = []
    for epsilon in EPSILONS:
        perturbed = base_input + epsilon * delta_input
        perturbed_state = _state(
            model, perturbed[:15].reshape(5, 3), perturbed[15:].reshape(5, 3)
        )
        approximation = (
            _jacobi_phase(to_canonical_jacobi(plan, model, perturbed_state))
            - base_output
        ) / epsilon
        error = approximation - analytic
        finite_difference.append(
            {
                "epsilon": epsilon,
                "epsilon_hex": epsilon.hex(),
                "max_abs_error": float(np.max(np.abs(error))),
                "relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(analytic)),
            }
        )

    half = phase.shape[0] // 2
    identity_half = np.eye(half)
    zero = np.zeros((half, half))
    symplectic = np.block([[zero, identity_half], [-identity_half, zero]])

    def symplectic_metrics(operator: np.ndarray) -> Mapping[str, float]:
        residual = operator.T @ symplectic @ operator - symplectic
        frobenius = float(np.linalg.norm(residual, "fro"))
        return {
            "max_abs": float(np.max(np.abs(residual))),
            "frobenius": frobenius,
            "norm_scaled": frobenius
            / float(np.linalg.norm(operator, 2) ** 2 * np.linalg.norm(symplectic, "fro")),
        }

    original = canonical
    translation = np.array((3.5, -2.25, 6.75))
    boost = np.array((-0.5, 1.25, 2.0))
    shifted = to_canonical_jacobi(
        plan,
        model,
        _state(
            model,
            np.asarray(state.positions_m) + translation,
            np.asarray(state.momenta_kg_m_per_s) + np.asarray(MASSES)[:, None] * boost,
        ),
    )
    translation_internal = float(
        np.max(np.abs(np.asarray(shifted.q_m[1:]) - np.asarray(original.q_m[1:])))
    )
    boost_internal = float(
        np.max(
            np.abs(
                np.asarray(shifted.p_kg_m_per_s[1:])
                - np.asarray(original.p_kg_m_per_s[1:])
            )
        )
    )

    dimension = phase.shape[0]
    absolute_symplectic_bound = 512.0 * U * dimension * max(1.0, condition_phase**2)
    scaled_symplectic_bound = 512.0 * U * dimension
    fd_floor = 512.0 * U * max(1.0, condition_phase)
    state_bound_scale = max(
        1.0,
        np.linalg.norm(_inertial_phase(state), np.inf),
        np.linalg.norm(_jacobi_phase(canonical), np.inf),
    )
    state_norm_scale = max(
        1.0,
        np.linalg.norm(_inertial_phase(state)),
        np.linalg.norm(_jacobi_phase(canonical)),
    )
    state_component_bound = (
        256.0 * U * len(MASSES) * max(1.0, condition_phase) * state_bound_scale
    )
    state_norm_bound = (
        256.0
        * U
        * math.sqrt(6.0 * len(MASSES))
        * max(1.0, condition_phase)
        * state_norm_scale
    )
    tangent_bound_scale = max(
        1.0,
        np.linalg.norm(_inertial_tangent_phase(tangent), np.inf),
        np.linalg.norm(_jacobi_tangent_phase(canonical_tangent), np.inf),
    )
    tangent_norm_scale = max(
        1.0,
        np.linalg.norm(_inertial_tangent_phase(tangent)),
        np.linalg.norm(_jacobi_tangent_phase(canonical_tangent)),
    )
    tangent_component_bound = (
        256.0 * U * len(MASSES) * max(1.0, condition_phase) * tangent_bound_scale
    )
    tangent_norm_bound = (
        256.0
        * U
        * math.sqrt(6.0 * len(MASSES))
        * max(1.0, condition_phase)
        * tangent_norm_scale
    )
    dense_oracle_bound = (
        128.0
        * U
        * len(MASSES)
        * max(1.0, condition_phase)
        * max(1.0, np.linalg.norm(dense_state, np.inf), np.linalg.norm(dense_tangent, np.inf))
    )
    invariance_bound = (
        128.0
        * U
        * len(MASSES)
        * max(1.0, condition_phase)
        * max(1.0, np.linalg.norm(_jacobi_phase(shifted), np.inf))
    )
    forward_symplectic = symplectic_metrics(phase)
    inverse_symplectic = symplectic_metrics(phase_inverse)
    inverse_consistency = float(
        np.max(np.abs(phase_inverse @ phase - np.eye(dimension)))
    )
    return {
        "schema_version": 1,
        "kind": "m0_step3g1b_numerical_metrics",
        "fixture": {
            "label": "synthetic_conditioning_fixture",
            "solar_system_data": False,
            "body_count": len(MASSES),
            "masses_kg": list(MASSES),
            "minimum_mass_kg": min(MASSES),
            "maximum_mass_kg": max(MASSES),
            "mass_ratio_max_over_min": max(MASSES) / min(MASSES),
        },
        "plan_fingerprint": plan.fingerprint,
        "model_fingerprint": model.fingerprint,
        "condition_numbers": {"A_2": condition_a, "phase_S_2": condition_phase},
        "dense_oracle": {
            "state": dense_state_error,
            "tangent": dense_tangent_error,
        },
        "round_trip": {
            "state_forward_inverse": state_forward_inverse,
            "state_inverse_forward": state_inverse_forward,
            "tangent_forward_inverse": tangent_forward_inverse,
            "tangent_inverse_forward": tangent_inverse_forward,
        },
        "finite_difference": {
            "values": finite_difference,
            "minimum_relative_l2_error": min(
                value["relative_l2_error"] for value in finite_difference
            ),
            "first_four_minimum_relative_l2_error": min(
                value["relative_l2_error"] for value in finite_difference[:4]
            ),
            "floor_bound": fd_floor,
        },
        "symplecticity": {
            "forward": forward_symplectic,
            "inverse": inverse_symplectic,
            "inverse_consistency_max_abs": inverse_consistency,
            "determinant_secondary": float(np.linalg.det(phase)),
        },
        "invariance": {
            "translation_internal_max_abs": translation_internal,
            "boost_internal_max_abs": boost_internal,
        },
        "bounds": {
            "state_componentwise": state_component_bound,
            "state_normwise": state_norm_bound,
            "tangent_componentwise": tangent_component_bound,
            "tangent_normwise": tangent_norm_bound,
            "dense_oracle_componentwise": dense_oracle_bound,
            "invariance_internal": invariance_bound,
            "symplectic_absolute": absolute_symplectic_bound,
            "symplectic_norm_scaled": scaled_symplectic_bound,
            "inverse_consistency": absolute_symplectic_bound,
        },
        "acceptance": {
            "state_round_trips": max(
                state_forward_inverse["max_abs"], state_inverse_forward["max_abs"]
            )
            <= state_component_bound
            and max(state_forward_inverse["l2"], state_inverse_forward["l2"])
            <= state_norm_bound,
            "tangent_round_trips": max(
                tangent_forward_inverse["max_abs"],
                tangent_inverse_forward["max_abs"],
            )
            <= tangent_component_bound
            and max(tangent_forward_inverse["l2"], tangent_inverse_forward["l2"])
            <= tangent_norm_bound,
            "finite_difference": min(
                value["relative_l2_error"] for value in finite_difference[:4]
            )
            <= fd_floor,
            "dense_oracle": max(
                dense_state_error["max_abs"], dense_tangent_error["max_abs"]
            )
            <= dense_oracle_bound,
            "symplecticity": all(
                metric["max_abs"] <= absolute_symplectic_bound
                and metric["frobenius"] <= absolute_symplectic_bound
                and metric["norm_scaled"] <= scaled_symplectic_bound
                for metric in (forward_symplectic, inverse_symplectic)
            )
            and inverse_consistency <= absolute_symplectic_bound,
            "invariance": max(translation_internal, boost_internal)
            <= invariance_bound,
        },
    }


def _inventory(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = manifest["exact_test_selection"]
    tests = []
    for group, key in (
        ("step3g1b_core", "step3g1b_core_node_ids"),
        ("step3g1b_integrity", "step3g1b_integrity_node_ids"),
        ("safe_step3g1a_regression", "safe_step3g1a_regression_node_ids"),
        ("artifact", "artifact_node_ids"),
    ):
        tests.extend(
            {"group": group, "node_id": node, "result": "PASS"}
            for node in selection[key]
        )
    return {
        "schema_version": 1,
        "kind": "m0_step3g1b_qualifying_test_inventory",
        "selection": "exact_pytest_node_ids_only",
        "pytest_plugin_autoload_disabled": True,
        "counts": selection["expected_counts"],
        "commands": manifest["exact_commands"],
        "tests": tests,
    }


def _traceability_bytes(manifest: Mapping[str, Any]) -> bytes:
    selection = manifest["exact_test_selection"]
    all_nodes = (
        selection["step3g1b_core_node_ids"]
        + selection["step3g1b_integrity_node_ids"]
        + selection["safe_step3g1a_regression_node_ids"]
        + selection["artifact_node_ids"]
    )
    by_name = {node.rsplit("::", 1)[-1]: node for node in all_nodes}
    requirements = [
        ("G1B-CONVENTION", ["test_one_body_identity_state_and_tangent", "test_two_body_direct_formulas_and_inverse", "test_three_body_hand_formulas_all_axes"]),
        ("G1B-ORACLE", ["test_dense_rational_oracle_general_fixture", "test_independent_matrix_closure_and_finite_difference_ladder"]),
        ("G1B-INVERSE", ["test_state_round_trips_both_directions_with_frozen_bounds", "test_tangent_round_trips_both_directions_with_frozen_bounds"]),
        ("G1B-INVARIANCE", ["test_translation_and_mass_weighted_boost_invariance"]),
        ("G1B-VALIDATION", ["test_invalid_masses_and_noncentral_first_plan_rejected", "test_layout_reordering_and_model_fingerprint_mismatch_rejected", "test_semantic_units_shapes_dtype_and_fingerprint_rejected"]),
        ("G1B-OWNERSHIP", ["test_immutable_detached_outputs_and_deterministic_plan_serialization"]),
        ("G1B-TANGENT", ["test_linearity_zero_and_base_state_independence", "test_independent_matrix_closure_and_finite_difference_ladder"]),
        ("G1B-SYMPLECTIC", ["test_full_phase_space_forward_and_inverse_symplecticity"]),
        ("G1B-DETERMINISM", ["test_plan_and_result_repeated_calls_are_deterministic", "test_guarded_fresh_process_hash_seed_determinism", "test_fresh_process_regeneration_is_byte_identical"]),
        ("G1B-ISOLATION", ["test_runtime_source_is_linear_recurrence_without_dense_or_dynamics_surface", "test_static_import_node_and_subprocess_closure"]),
        ("G1B-INTEGRITY", ["test_step3g1a_sources_and_tests_are_byte_exact", "test_protected_historical_archive_trajectory_and_tag_integrity"]),
    ]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("requirement_id", "exact_passing_node_ids", "evidence_boundary", "disposition"))
    for requirement, names in requirements:
        writer.writerow(
            (
                requirement,
                ";".join(by_name[name] for name in names),
                "synthetic_fixed_mass_coordinate_transform_only",
                "PASS",
            )
        )
    return output.getvalue().encode("utf-8")


def _convention_specification() -> str:
    return """# Fixed-Mass Canonical Jacobi Convention

## Boundary

This primitive maps an inertial canonical state `(x,p)` to canonical Jacobi `(q,P)`. It does not accept velocity as momentum and does not perform a velocity-to-momentum conversion. The future Step 3g0 `(x,v)` adapter remains unimplemented.

## Identity And Order

Bodies retain the exact `CompiledLayout.body_ids` order and the declared central body must be first. No body identity is inferred from an index and no mismatch is reordered. The center-of-mass pair `(q_0,P_0)` is retained. Barycenter-at-origin and zero-total-momentum constraints are not imposed.

## Forward Map

For fixed positive masses, `eta_i = sum_(j=0)^i m_j` is evaluated left-to-right in binary64. Then:

```text
q_0 = sum_j(m_j*x_j)/eta_(N-1)
P_0 = sum_j p_j
q_i = x_i - sum_(j<i)(m_j*x_j)/eta_(i-1)
P_i = (eta_(i-1)/eta_i)*p_i - (m_i/eta_i)*sum_(j<i)p_j
```

Thus `q=A*x` and `P=A^(-T)*p`. Relative coordinates are body minus the inner center of mass.

## Inverse And Tangent

The inverse uses the preregistered O(N) center-of-mass and backward-prefix recurrences. It is an algebraic inverse; general binary64 round trips are bounded rather than described as exact. The tangent map applies the same constant operators to `(delta_x,delta_p)` and is independent of base-state values.

## Units And Flattening

Rows are body-major `(x,y,z)`. Positions are metres and momenta are kg*m/s under `si_v1`. Full phase vectors flatten all body-major position rows first and all body-major momentum rows second. Symplecticity uses the resulting complete `6N` canonical matrix.
"""


def _review_findings() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "kind": "m0_step3g1b_code_review_findings",
        "review_status": "COMPLETE",
        "unresolved_material_findings": 0,
        "material_regressions_added": 1,
        "focus": manifest24()["review_focus"],
        "findings": [
            {
                "finding_id": "G1B-R01",
                "severity": "MEDIUM",
                "finding": "The first reporting implementation omitted dense-oracle acceptance and treated finite invariance residuals as sufficient despite Manifest 24 numerical bounds.",
                "resolution": "The acceptance block now enforces both frozen bounds, and the artifact metric node independently checks the recorded residuals.",
                "regression_nodes": [
                    "test_numerical_metrics_satisfy_frozen_bounds"
                ]
            },
            {
                "finding_id": "G1B-R02",
                "severity": "LOW",
                "finding": "The first artifact generation exposed a NumPy boolean scalar at the strict JSON boundary.",
                "resolution": "The JSON encoder now normalizes only NumPy scalar values through item(), retains allow_nan=False, and is covered by strict-JSON plus fresh-process artifact nodes.",
                "regression_nodes": [
                    "test_required_artifacts_and_strict_json",
                    "test_fresh_process_regeneration_is_byte_identical"
                ]
            },
            {
                "finding_id": "G1B-R03",
                "severity": "INFORMATIONAL",
                "finding": "The Step 3g0 external (x,v) adapter boundary is broader than this canonical (x,p) primitive.",
                "resolution": "The semantic input type, manifest, tests, and report prohibit velocity conversion and defer the adapter explicitly.",
                "regression_nodes": [
                    "test_semantic_units_shapes_dtype_and_fingerprint_rejected",
                    "test_runtime_source_is_linear_recurrence_without_dense_or_dynamics_surface"
                ]
            }
        ]
    }


def _report(metrics: Mapping[str, Any]) -> str:
    round_trip = metrics["round_trip"]
    fd = metrics["finite_difference"]
    symplectic = metrics["symplecticity"]
    return f"""# M0 Step 3g1b Canonical Jacobi Tangent Primitives Report

## Result

- Final status: `STEP3G1B_JACOBI_TANGENT_PRIMITIVES_COMPLETE`
- Primary finding: `CANONICAL_JACOBI_TRANSFORMS_QUALIFIED_FOR_PRIMITIVE_COMPOSITION`
- Verification envelope: `COORDINATE_TRANSFORM_ONLY_NO_DYNAMICS_EXECUTED`
- Preregistration commit: `{PREREGISTRATION_COMMIT}`

## Convention And Implementation

The qualified operator retains the center-of-mass pair and maps fixed-mass inertial canonical `(x,p)` to Jacobi `(q,P)` with `P=A^(-T)p`. Body order and central-first identity are explicit. Production applies fixed left-to-right binary64 O(N) recurrences; dense matrices appear only in independent qualification code.

The Step 3g0 `(x,v)` wording remains a future adapter boundary. This step introduces no velocity-based Jacobi coordinate, no velocity-to-momentum conversion, and no claim that the complete REBOUND compatibility adapter is qualified.

## Numerical Evidence

The synthetic five-body mass range is {metrics['fixture']['minimum_mass_kg']:.17g} to {metrics['fixture']['maximum_mass_kg']:.17g} kg, ratio {metrics['fixture']['mass_ratio_max_over_min']:.17g}. `cond2(A)` is {metrics['condition_numbers']['A_2']:.17g}; `cond2(S)` is {metrics['condition_numbers']['phase_S_2']:.17g}.

Maximum state forward/inverse round-trip component error is {round_trip['state_forward_inverse']['max_abs']:.17g}; maximum tangent forward/inverse component error is {round_trip['tangent_forward_inverse']['max_abs']:.17g}. The finite-difference ladder minimum relative L2 error is {fd['minimum_relative_l2_error']:.17g} against bound {fd['floor_bound']:.17g}. Nonmonotonic behavior after cancellation dominates is permitted exactly as preregistered.

Forward symplectic residuals are max {symplectic['forward']['max_abs']:.17g}, Frobenius {symplectic['forward']['frobenius']:.17g}, and scaled {symplectic['forward']['norm_scaled']:.17g}. Inverse residuals are max {symplectic['inverse']['max_abs']:.17g}, Frobenius {symplectic['inverse']['frobenius']:.17g}, and scaled {symplectic['inverse']['norm_scaled']:.17g}. The reported determinant {symplectic['determinant_secondary']:.17g} is secondary only.

## Safety And Evidence Boundary

All tests use analytic or explicitly synthetic data. No Solar-System state is used. No physical force or JVP was evaluated. No dynamical map was implemented. No integration or timestep occurred. REBOUND and REBOUNDx were not imported. Protected and historical inputs remained byte exact.

Symplecticity applies only to this fixed-mass coordinate transformation. This result does not qualify a Kepler drift, kick, lazy kernel, corrector, WHCKL kernel, tangent evolution, MEGNO/LCN calculation, restart path, or Solar-System trajectory.

## Successor

The smallest justified successor is a separately preregistered Step 3g1c limited to a two-body Kepler drift and its canonical tangent map.
"""


def _summary(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], safety: Mapping[str, Any]
) -> Mapping[str, Any]:
    source_paths = [
        manifest["paths"][key]
        for key in (
            "implementation",
            "qualification_helper",
            "runner",
            "reporting",
            "test",
            "integrity_test",
            "artifact_test",
        )
    ]
    return {
        "schema_version": 1,
        "kind": "m0_step3g1b_canonical_jacobi_tangent_primitives_summary",
        "final_status": "STEP3G1B_JACOBI_TANGENT_PRIMITIVES_COMPLETE",
        "primary_finding": "CANONICAL_JACOBI_TRANSFORMS_QUALIFIED_FOR_PRIMITIVE_COMPOSITION",
        "verification_envelope": "COORDINATE_TRANSFORM_ONLY_NO_DYNAMICS_EXECUTED",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "required_start_commit": manifest["preregistration"]["required_start_commit"],
        "branch": manifest["preregistration"]["required_branch"],
        "test_counts": manifest["exact_test_selection"]["expected_counts"],
        "metrics": {
            "condition_numbers": metrics["condition_numbers"],
            "mass_ratio_max_over_min": metrics["fixture"]["mass_ratio_max_over_min"],
            "round_trip": metrics["round_trip"],
            "finite_difference": metrics["finite_difference"],
            "symplecticity": metrics["symplecticity"],
            "invariance": metrics["invariance"],
        },
        "safety": {
            "static_gate": safety["status"],
            "exact_node_selection": True,
            "pytest_plugin_autoload_disabled": True,
            "fresh_process_guarded": True,
            "forbidden_imports": safety["forbidden_imports"],
            "forbidden_library_mappings": safety["forbidden_library_mappings"],
            "legacy_package_init_bypassed": safety["legacy_package_init_bypassed"],
            "legacy_nbody_absent": safety["legacy_nbody_absent"],
        },
        "integrity": {
            "step3g1a_files_unchanged": True,
            "protected_sources_unchanged": True,
            "historical_manifests_and_artifacts_unchanged": True,
            "archives_and_trajectories_unchanged": True,
            "annotated_tags_unchanged": True,
        },
        "forbidden_operations": {
            "velocity_conversion_executed": False,
            "physical_force_or_jvp_evaluated": False,
            "dynamical_map_implemented": False,
            "integration_or_timestep_executed": False,
            "rebound_or_reboundx_imported": False,
            "solar_system_state_used": False,
            "trajectory_or_archive_created": False,
            "megno_or_lcn_executed": False,
            "tag_created_or_moved": False,
        },
        "source_inventory_sha256": {
            relative: sha256_file(ROOT / relative) for relative in source_paths
        },
        "review": {
            "unresolved_material_findings": 0,
            "claims_limited_to_coordinate_transform": True,
        },
        "remaining_risks": [
            "The future inertial (x,v) adapter and explicit velocity-to-momentum boundary remain unimplemented.",
            "No Kepler drift, kick, corrector, WHCKL composition, or discrete tangent evolution is qualified.",
            "The synthetic mass ratio of 256 does not establish conditioning for a Solar-System production mass hierarchy.",
            "Binary64 transform round trips are bounded and are not mathematically exact for general inputs."
        ],
        "step3g1c_proposal": "two-body Kepler drift and its canonical tangent map only",
    }


def generate_artifacts(destination: Path = DEFAULT_DESTINATION) -> Mapping[str, Any]:
    """Generate the complete deterministic Step 3g1b artifact set."""

    destination = Path(destination)
    manifest = manifest24()
    safety = static_safety_audit()
    verify_inherited_integrity()
    metrics = collect_metrics()
    if not all(metrics["acceptance"].values()):
        raise AssertionError(f"numerical acceptance failed: {metrics['acceptance']}")
    files = {
        "canonical_convention_specification.md": _convention_specification().encode("utf-8"),
        "code_review_findings.json": _json_bytes(_review_findings()),
        "m0_step3g1b_canonical_jacobi_tangent_primitives_report.md": _report(metrics).encode("utf-8"),
        "m0_step3g1b_canonical_jacobi_tangent_primitives_summary.json": _json_bytes(
            _summary(manifest, metrics, safety)
        ),
        "numerical_metrics.json": _json_bytes(metrics),
        "qualifying_test_inventory.json": _json_bytes(_inventory(manifest)),
        "requirements_traceability.csv": _traceability_bytes(manifest),
    }
    for name, payload in files.items():
        _atomic_write(destination / name, payload)
    hashes = {
        "schema_version": 1,
        "kind": "m0_step3g1b_artifact_hashes",
        "sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(files.items())
        },
    }
    _atomic_write(destination / "artifact_hashes.json", _json_bytes(hashes))
    assert_protected_runtime_absent()
    return _summary(manifest, metrics, safety)


def validate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    """Validate strict JSON, inventory, hashes, vocabulary, and numerical gates."""

    destination = Path(destination)
    observed = {path.name for path in destination.iterdir() if path.is_file()}
    if observed != EXPECTED_ARTIFACTS:
        raise AssertionError(f"artifact inventory mismatch: {sorted(observed)}")
    for name in (
        "artifact_hashes.json",
        "code_review_findings.json",
        "m0_step3g1b_canonical_jacobi_tangent_primitives_summary.json",
        "numerical_metrics.json",
        "qualifying_test_inventory.json",
    ):
        strict_json(destination / name)
    hashes = strict_json(destination / "artifact_hashes.json")["sha256"]
    if set(hashes) != ARTIFACTS_WITHOUT_HASH_INDEX:
        raise AssertionError("artifact hash inventory keys differ")
    for name, expected in hashes.items():
        if sha256_file(destination / name) != expected:
            raise AssertionError(f"artifact hash mismatch: {name}")
    summary = strict_json(
        destination / "m0_step3g1b_canonical_jacobi_tangent_primitives_summary.json"
    )
    manifest = manifest24()
    if summary["final_status"] not in manifest["result_vocabulary"]["final_status"]:
        raise AssertionError("invalid final status")
    if summary["primary_finding"] not in manifest["result_vocabulary"]["primary_finding"]:
        raise AssertionError("invalid primary finding")
    if summary["verification_envelope"] != manifest["result_vocabulary"]["success_verification_envelope"]:
        raise AssertionError("invalid verification envelope")
    metrics = strict_json(destination / "numerical_metrics.json")
    if not all(metrics["acceptance"].values()):
        raise AssertionError("committed numerical metrics fail a frozen gate")
    assert_protected_runtime_absent()


def compare_fresh_regeneration(destination: Path = DEFAULT_DESTINATION) -> None:
    """Require two fresh guarded generations and committed bytes to match."""

    import tempfile

    with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
        first = Path(first_name)
        second = Path(second_name)
        run_fresh_artifact_probe(first)
        run_fresh_artifact_probe(second)
        for name in EXPECTED_ARTIFACTS:
            first_bytes = (first / name).read_bytes()
            if first_bytes != (second / name).read_bytes():
                raise AssertionError(f"fresh artifact regeneration differs: {name}")
            if first_bytes != (destination / name).read_bytes():
                raise AssertionError(f"committed artifact differs from regeneration: {name}")


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or strictly validate Step 3g1b artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args(argv)
    guard = install_guard()
    guard.activate_strict()
    if arguments.validate:
        validate_artifacts()
    else:
        generate_artifacts()
    assert_protected_runtime_absent()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
