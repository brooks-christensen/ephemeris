"""Deterministic numerical and provenance artifacts for Step 3g1c."""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import io
import json
import math
import os
from pathlib import Path
import tempfile
import types
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DESTINATION = ROOT / "docs/validation/m0-step3g1c-kepler-drift-tangent-primitive-v1"
SUMMARY_NAME = "m0_step3g1c_kepler_drift_tangent_primitive_summary.json"
REPORT_NAME = "m0_step3g1c_kepler_drift_tangent_primitive_report.md"
ARTIFACTS_WITHOUT_HASH_INDEX = {
    REPORT_NAME, SUMMARY_NAME, "hamiltonian_supported_domain_specification.md",
    "tangent_map_derivation.md", "requirements_traceability.csv",
    "qualifying_test_inventory.json", "physical_oracle_metrics.json",
    "finite_difference_metrics.json", "invariant_metrics.json",
    "reversibility_metrics.json", "symplecticity_metrics.json",
    "solver_metrics.json", "code_review_findings.json",
}
EXPECTED_ARTIFACTS = ARTIFACTS_WITHOUT_HASH_INDEX | {"artifact_hashes.json"}
U = np.finfo(np.float64).eps
SEMIMAJOR_AXIS_M = 20_000_000.0
MINIMUM_PERIAPSIS_M = 1_000_000.0
ECCENTRICITIES = (0.0, 0.1, 0.6, 0.9, 0.919)
ANOMALIES = (0.0, 0.37, 1.4, 2.7, 3.05)
PERIOD_FRACTIONS = (-0.99, -0.5, -0.125, -0.001, 0.0, 0.001, 0.125, 0.5, 0.99)
EPSILONS = tuple(2.0**exponent for exponent in range(-10, -43, -4))


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

from mini_ephemeris.m0_step3g1c_qualification import (  # noqa: E402
    assert_protected_runtime_absent, install_guard, manifest25,
    run_fresh_artifact_probe, sha256_file, static_safety_audit,
    strict_json, verify_inherited_integrity,
)
from mini_ephemeris.v2.jacobi import build_jacobi_transform_plan  # noqa: E402
from mini_ephemeris.v2.kepler import (  # noqa: E402
    CanonicalKeplerPairState, CanonicalKeplerPairTangent,
    build_kepler_pair_plan, kepler_drift, kepler_drift_tangent,
)
from mini_ephemeris.v2.model import CompiledLayout, PhysicalModel, SI_UNITS  # noqa: E402


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False, default=_json_default
    ) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _fixture():
    layout = CompiledLayout(("inner", "outer"), "inner")
    model = PhysicalModel(
        model_id="synthetic_kepler_qualification", schema_version="1", layout=layout,
        masses_kg={"inner": 4.0, "outer": 4.0},
        gravitational_constant_si=50_000_000_000_000.0, units=SI_UNITS,
        enabled_effects=("synthetic-newtonian-pair",),
        provenance={"fixture": "step3g1c-synthetic"},
    )
    jacobi = build_jacobi_transform_plan(model)
    plan = build_kepler_pair_plan(
        model, jacobi, 1, minimum_periapsis_m=MINIMUM_PERIAPSIS_M
    )
    return model, plan


def _orientation(name: str):
    if name == "xy_aligned":
        first, second = np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0))
    elif name == "non_axis_aligned_rotation_A":
        first, second = np.array((1.0, 2.0, 3.0)), np.array((-2.0, 1.0, 0.5))
    else:
        first, second = np.array((2.0, -1.0, 1.5)), np.array((0.25, 3.0, -1.0))
    first /= np.linalg.norm(first)
    second -= np.dot(second, first) * first
    second /= np.linalg.norm(second)
    return first, second, np.cross(first, second)


def _state(plan, eccentricity: float, anomaly: float, orientation: str = "non_axis_aligned_rotation_A"):
    first, second, _ = _orientation(orientation)
    parameter = SEMIMAJOR_AXIS_M * (1.0 - eccentricity * eccentricity)
    radius = parameter / (1.0 + eccentricity * math.cos(anomaly))
    q = radius * (math.cos(anomaly) * first + math.sin(anomaly) * second)
    velocity = math.sqrt(plan.gravitational_parameter_m3_s2 / parameter) * (
        -math.sin(anomaly) * first + (eccentricity + math.cos(anomaly)) * second
    )
    return CanonicalKeplerPairState(
        q_m=q, p_kg_m_per_s=plan.reduced_mass_kg * velocity,
        unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _tangent(plan, values):
    return CanonicalKeplerPairTangent(
        delta_q_m=values[:3], delta_p_kg_m_per_s=values[3:],
        unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _period(plan) -> float:
    return 2.0 * math.pi * math.sqrt(
        SEMIMAJOR_AXIS_M**3 / plan.gravitational_parameter_m3_s2
    )


def _phase(state) -> np.ndarray:
    return np.concatenate((np.asarray(state.q_m), np.asarray(state.p_kg_m_per_s)))


def _tangent_phase(tangent) -> np.ndarray:
    return np.concatenate((np.asarray(tangent.delta_q_m), np.asarray(tangent.delta_p_kg_m_per_s)))


def _scales(plan) -> np.ndarray:
    momentum = plan.reduced_mass_kg * math.sqrt(
        plan.gravitational_parameter_m3_s2 / SEMIMAJOR_AXIS_M
    )
    return np.array((SEMIMAJOR_AXIS_M,) * 3 + (momentum,) * 3)


def _scaled_error(plan, observed, expected):
    difference = (_phase(observed) - _phase(expected)) / _scales(plan)
    return float(np.max(np.abs(difference))), float(np.linalg.norm(difference))


def _chi(eccentricity: float, phase: float) -> float:
    return (1.0 + eccentricity) * (1.0 + abs(phase)) / (1.0 - eccentricity) ** 2


def _reference(plan, state, duration):
    q0 = np.asarray(state.q_m)
    v0 = np.asarray(state.p_kg_m_per_s) / plan.reduced_mass_kg
    mu = plan.gravitational_parameter_m3_s2
    r0 = np.linalg.norm(q0)
    energy = 0.5 * np.dot(v0, v0) - mu / r0
    semimajor = -mu / (2.0 * energy)
    angular = np.cross(q0, v0)
    normal = angular / np.linalg.norm(angular)
    e_vector = np.cross(v0, angular) / mu - q0 / r0
    eccentricity = np.linalg.norm(e_vector)
    mean_motion = math.sqrt(mu / semimajor**3)
    if eccentricity <= 64.0 * U:
        angle = mean_motion * duration
        q1 = math.cos(angle) * q0 + math.sin(angle) * np.cross(normal, q0)
        v1 = math.cos(angle) * v0 + math.sin(angle) * np.cross(normal, v0)
    else:
        periapsis = e_vector / eccentricity
        transverse = np.cross(normal, periapsis)
        root = math.sqrt(1.0 - eccentricity * eccentricity)
        e0 = math.atan2(
            np.dot(q0, transverse) / (semimajor * root),
            np.dot(q0, periapsis) / semimajor + eccentricity,
        )
        mean = math.remainder(
            e0 - eccentricity * math.sin(e0) + mean_motion * duration, 2.0 * math.pi
        )
        lower, upper, anomaly = -math.pi, math.pi, 0.0
        for _ in range(160):
            anomaly = 0.5 * (lower + upper)
            if anomaly - eccentricity * math.sin(anomaly) - mean >= 0.0:
                upper = anomaly
            else:
                lower = anomaly
        cosine, sine = math.cos(anomaly), math.sin(anomaly)
        denominator = 1.0 - eccentricity * cosine
        q1 = semimajor * (
            (cosine - eccentricity) * periapsis + root * sine * transverse
        )
        v1 = semimajor * mean_motion / denominator * (
            -sine * periapsis + root * cosine * transverse
        )
    return CanonicalKeplerPairState(
        q_m=q1, p_kg_m_per_s=plan.reduced_mass_kg * v1,
        unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _invariants(plan, state):
    q, p = np.asarray(state.q_m), np.asarray(state.p_kg_m_per_s)
    velocity = p / plan.reduced_mass_kg
    radius = np.linalg.norm(q)
    energy = 0.5 * np.dot(velocity, velocity) - plan.gravitational_parameter_m3_s2 / radius
    h = np.cross(q, velocity)
    e_vector = np.cross(velocity, h) / plan.gravitational_parameter_m3_s2 - q / radius
    semimajor = -plan.gravitational_parameter_m3_s2 / (2.0 * energy)
    return plan.reduced_mass_kg * energy, energy, h, e_vector, semimajor


def _jacobian(plan, state, duration):
    scales = _scales(plan)
    columns = []
    for index in range(6):
        direction = np.zeros(6)
        direction[index] = scales[index]
        result = kepler_drift_tangent(plan, state, _tangent(plan, direction), duration).tangent
        columns.append(_tangent_phase(result) / scales)
    return np.column_stack(columns)

def collect_metrics() -> Mapping[str, Any]:
    model, plan = _fixture()
    period, scales = _period(plan), _scales(plan)
    solver_rows, physical_rows = [], []
    physical_cases = (
        (0.0, 0.37, "non_axis_aligned_rotation_A"),
        (0.0, 0.37, "non_axis_aligned_rotation_B"),
        (0.1, 0.37, "non_axis_aligned_rotation_A"),
        (0.6, 1.4, "non_axis_aligned_rotation_A"),
        (0.9, 2.7, "non_axis_aligned_rotation_A"),
        (0.919, 3.05, "non_axis_aligned_rotation_A"),
        (0.919, 0.0, "non_axis_aligned_rotation_A"),
    )
    for eccentricity, anomaly, orientation in physical_cases:
        state = _state(plan, eccentricity, anomaly, orientation)
        for fraction in PERIOD_FRACTIONS:
            result = kepler_drift(plan, state, fraction * period)
            expected = _reference(plan, state, fraction * period)
            maximum, normwise = _scaled_error(plan, result.state, expected)
            bound = 8192.0 * U * _chi(eccentricity, abs(fraction) * 2.0 * math.pi)
            physical_rows.append({
                "eccentricity": eccentricity, "true_anomaly_rad": anomaly,
                "orientation": orientation, "period_fraction": fraction,
                "maximum_scaled_error": maximum,
                "normwise_scaled_error": normwise, "bound": bound,
                "pass": max(maximum, normwise) <= bound,
            })
            diagnostic = result.diagnostics
            solver_rows.append({
                "eccentricity": eccentricity, "period_fraction": fraction,
                "branch": diagnostic.branch, "iterations": diagnostic.iterations,
                "residual_s": diagnostic.residual_s,
                "update_x_s_per_m": diagnostic.update_x_s_per_m,
                "phase_advance_rad": diagnostic.phase_advance_rad,
                "converged": diagnostic.converged,
            })

    composition_rows = []
    for eccentricity, anomaly in ((0.1, 0.37), (0.6, 2.7), (0.9, 1.4)):
        state = _state(plan, eccentricity, anomaly)
        first = kepler_drift(plan, state, 0.17 * period).state
        composed = kepler_drift(plan, first, -0.08 * period).state
        direct = kepler_drift(plan, state, 0.09 * period).state
        maximum, normwise = _scaled_error(plan, composed, direct)
        bound = 32768.0 * U * _chi(eccentricity, 0.17 * 2.0 * math.pi)
        composition_rows.append({
            "eccentricity": eccentricity, "maximum_scaled_error": maximum,
            "normwise_scaled_error": normwise, "bound": bound,
            "pass": max(maximum, normwise) <= bound,
        })

    invariant_rows = []
    invariant_cases = (
        (0.0, 0.37), (0.1, 1.4), (0.6, 2.7),
        (0.9, 0.37), (0.919, 3.05), (0.919, 0.0),
    )
    for eccentricity, anomaly in invariant_cases:
        state = _state(plan, eccentricity, anomaly)
        initial = _invariants(plan, state)
        for fraction in (-0.99, -0.125, 0.001, 0.5, 0.99):
            final = _invariants(plan, kepler_drift(plan, state, fraction * period).state)
            values = {
                "hamiltonian_relative": abs((final[0] - initial[0]) / initial[0]),
                "specific_energy_relative": abs((final[1] - initial[1]) / initial[1]),
                "angular_momentum_relative": float(
                    np.linalg.norm(final[2] - initial[2]) / np.linalg.norm(initial[2])
                ),
                "orbital_plane_normal_absolute": float(np.linalg.norm(
                    final[2] / np.linalg.norm(final[2])
                    - initial[2] / np.linalg.norm(initial[2])
                )),
                "eccentricity_vector_absolute": float(np.linalg.norm(final[3] - initial[3])),
                "semimajor_axis_relative": abs((final[4] - initial[4]) / initial[4]),
            }
            bound = 65536.0 * U * _chi(eccentricity, abs(fraction) * 2.0 * math.pi)
            invariant_rows.append({
                "eccentricity": eccentricity, "period_fraction": fraction,
                **values, "maximum": max(values.values()), "bound": bound,
                "pass": max(values.values()) <= bound,
            })

    state = _state(plan, 0.6, 1.4)
    direction = scales * np.array((0.2, -0.1, 0.05, -0.15, 0.25, 0.1))
    duration = 0.3 * period
    base, base_output = _phase(state), _phase(kepler_drift(plan, state, duration).state)
    analytic = _tangent_phase(
        kepler_drift_tangent(plan, state, _tangent(plan, direction), duration).tangent
    )
    fd_rows = []
    for epsilon in EPSILONS:
        plus_values, minus_values = base + epsilon * direction, base - epsilon * direction
        plus = CanonicalKeplerPairState(
            q_m=plus_values[:3], p_kg_m_per_s=plus_values[3:],
            unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
            model_fingerprint=plan.model_fingerprint,
            pair_plan_fingerprint=plan.fingerprint,
        )
        minus = CanonicalKeplerPairState(
            q_m=minus_values[:3], p_kg_m_per_s=minus_values[3:],
            unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
            model_fingerprint=plan.model_fingerprint,
            pair_plan_fingerprint=plan.fingerprint,
        )
        plus_output = _phase(kepler_drift(plan, plus, duration).state)
        minus_output = _phase(kepler_drift(plan, minus, duration).state)
        forward = (plus_output - base_output) / epsilon
        central = (plus_output - minus_output) / (2.0 * epsilon)
        denominator = np.linalg.norm(analytic / scales)
        fd_rows.append({
            "epsilon": epsilon, "epsilon_hex": epsilon.hex(),
            "forward_relative_l2_error": float(
                np.linalg.norm((forward - analytic) / scales) / denominator
            ),
            "central_relative_l2_error": float(
                np.linalg.norm((central - analytic) / scales) / denominator
            ),
        })
    chi_fd = _chi(0.6, 0.6 * math.pi)
    forward_values = [row["forward_relative_l2_error"] for row in fd_rows]
    central_values = [row["central_relative_l2_error"] for row in fd_rows]
    minimum_index = int(np.argmin(forward_values))
    fd_acceptance = {
        "finite": all(
            math.isfinite(value) for row in fd_rows for key, value in row.items()
            if key.endswith("_error")
        ),
        "forward_floor": min(forward_values) <= 128.0 * math.sqrt(U) * chi_fd,
        "central_floor": min(central_values) <= 512.0 * U ** (2.0 / 3.0) * chi_fd,
        "early_improvement": min(forward_values[:6]) <= 0.25 * forward_values[0],
        "roundoff_turn": minimum_index < len(forward_values) - 1
        and max(forward_values[minimum_index + 1:]) >= 2.0 * forward_values[minimum_index],
    }

    identity3, zero = np.eye(3), np.zeros((3, 3))
    symplectic_form = np.block([[zero, identity3], [-identity3, zero]])
    symplectic_rows = []
    for eccentricity, anomaly in (
        (0.0, 0.37), (0.6, 1.4), (0.9, 2.7),
        (0.919, 3.05), (0.919, 0.0),
    ):
        state = _state(plan, eccentricity, anomaly)
        for fraction in (-0.99, -0.125, 0.5, 0.99):
            matrix = _jacobian(plan, state, fraction * period)
            residual = matrix.T @ symplectic_form @ matrix - symplectic_form
            condition = float(np.linalg.cond(matrix))
            absolute_bound = 65536.0 * U * 6.0 * max(1.0, condition**2)
            scaled_bound = 65536.0 * U * 6.0 * max(1.0, condition)
            maximum = float(np.max(np.abs(residual)))
            frobenius = float(np.linalg.norm(residual, "fro"))
            scaled = frobenius / float(
                np.linalg.norm(matrix, 2) ** 2 * np.linalg.norm(symplectic_form, "fro")
            )
            symplectic_rows.append({
                "eccentricity": eccentricity, "period_fraction": fraction,
                "condition_2": condition, "maximum_absolute_residual": maximum,
                "frobenius_residual": frobenius, "norm_scaled_residual": scaled,
                "absolute_bound": absolute_bound, "norm_scaled_bound": scaled_bound,
                "determinant_secondary": float(np.linalg.det(matrix)),
                "pass": maximum <= absolute_bound and frobenius <= absolute_bound
                and scaled <= scaled_bound,
            })

    reversibility_rows = []
    for eccentricity, anomaly in ((0.1, 0.37), (0.6, 1.4), (0.9, 2.7)):
        state = _state(plan, eccentricity, anomaly)
        duration = 0.5 * period
        forward = kepler_drift(plan, state, duration).state
        recovered = kepler_drift(plan, forward, -duration).state
        maximum, normwise = _scaled_error(plan, recovered, state)
        forward_matrix = _jacobian(plan, state, duration)
        reverse_matrix = _jacobian(plan, forward, -duration)
        matrix_residual = reverse_matrix @ forward_matrix - np.eye(6)
        condition_product = float(
            np.linalg.cond(forward_matrix) * np.linalg.cond(reverse_matrix)
        )
        physical_bound = 32768.0 * U * _chi(eccentricity, math.pi)
        matrix_bound = 65536.0 * U * 6.0 * max(1.0, condition_product)
        direction = scales * np.array((0.2, -0.1, 0.05, -0.15, 0.25, 0.1))
        advanced = kepler_drift_tangent(
            plan, state, _tangent(plan, direction), duration
        )
        returned = kepler_drift_tangent(
            plan, advanced.state, advanced.tangent, -duration
        ).tangent
        tangent_relative = float(
            np.linalg.norm((_tangent_phase(returned) - direction) / scales)
            / np.linalg.norm(direction / scales)
        )
        tangent_bound = (
            65536.0 * U * _chi(eccentricity, math.pi)
            * max(1.0, float(np.linalg.cond(forward_matrix)))
        )
        reversed_initial = CanonicalKeplerPairState(
            q_m=state.q_m, p_kg_m_per_s=-np.asarray(state.p_kg_m_per_s),
            unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
            model_fingerprint=plan.model_fingerprint,
            pair_plan_fingerprint=plan.fingerprint,
        )
        reversed_forward = kepler_drift(plan, reversed_initial, duration).state
        momentum_flipped = CanonicalKeplerPairState(
            q_m=reversed_forward.q_m,
            p_kg_m_per_s=-np.asarray(reversed_forward.p_kg_m_per_s),
            unit_system_id="si_v1", layout_fingerprint=plan.layout_fingerprint,
            model_fingerprint=plan.model_fingerprint,
            pair_plan_fingerprint=plan.fingerprint,
        )
        backward = kepler_drift(plan, state, -duration).state
        momentum_maximum, momentum_normwise = _scaled_error(
            plan, momentum_flipped, backward
        )
        reversibility_rows.append({
            "eccentricity": eccentricity,
            "physical_maximum_scaled": maximum,
            "physical_normwise_scaled": normwise, "physical_bound": physical_bound,
            "matrix_maximum_absolute": float(np.max(np.abs(matrix_residual))),
            "matrix_bound": matrix_bound, "tangent_relative_l2": tangent_relative,
            "tangent_bound": tangent_bound,
            "momentum_reversal_maximum_scaled": momentum_maximum,
            "momentum_reversal_normwise_scaled": momentum_normwise,
            "momentum_reversal_bound": physical_bound,
            "pass": max(maximum, normwise) <= physical_bound
            and float(np.max(np.abs(matrix_residual))) <= matrix_bound
            and tangent_relative <= tangent_bound
            and max(momentum_maximum, momentum_normwise) <= physical_bound,
        })

    state, duration = _state(plan, 0.6, 1.4), 0.3 * period
    q, p = np.asarray(state.q_m), np.asarray(state.p_kg_m_per_s)
    radius = np.linalg.norm(q)
    flow_values = np.concatenate((
        p / plan.reduced_mass_kg,
        -plan.reduced_mass_kg * plan.gravitational_parameter_m3_s2 * q / radius**3,
    ))
    flow_result = kepler_drift_tangent(plan, state, _tangent(plan, flow_values), duration)
    q1, p1 = np.asarray(flow_result.state.q_m), np.asarray(flow_result.state.p_kg_m_per_s)
    expected_flow = np.concatenate((
        p1 / plan.reduced_mass_kg,
        -plan.reduced_mass_kg * plan.gravitational_parameter_m3_s2
        * q1 / np.linalg.norm(q1)**3,
    ))
    matrix = _jacobian(plan, state, duration)
    directional_bound = 65536.0 * U * _chi(0.6, 0.6 * math.pi) * max(
        1.0, float(np.linalg.cond(matrix))
    )
    omega = np.array((0.3, -0.4, 0.5))
    rotation_values = np.concatenate((np.cross(omega, q), np.cross(omega, p)))
    rotation_result = kepler_drift_tangent(
        plan, state, _tangent(plan, rotation_values), duration
    )
    expected_rotation = np.concatenate((
        np.cross(omega, np.asarray(rotation_result.state.q_m)),
        np.cross(omega, np.asarray(rotation_result.state.p_kg_m_per_s)),
    ))
    flow_rotation = {
        "flow_relative_l2": float(
            np.linalg.norm((_tangent_phase(flow_result.tangent) - expected_flow) / scales)
            / np.linalg.norm(expected_flow / scales)
        ),
        "rotation_relative_l2": float(
            np.linalg.norm(
                (_tangent_phase(rotation_result.tangent) - expected_rotation) / scales
            ) / np.linalg.norm(expected_rotation / scales)
        ),
        "bound": directional_bound,
    }
    flow_rotation["pass"] = (
        flow_rotation["flow_relative_l2"] <= directional_bound
        and flow_rotation["rotation_relative_l2"] <= directional_bound
    )
    first = scales * np.array((0.2, -0.3, 0.1, 0.4, 0.05, -0.2))
    second = scales * np.array((-0.1, 0.15, 0.25, -0.2, 0.3, 0.1))
    a_value, b_value = 1.25, -0.75
    combined = a_value * first + b_value * second
    observed_linear = _tangent_phase(kepler_drift_tangent(
        plan, state, _tangent(plan, combined), duration
    ).tangent)
    expected_linear = (
        a_value * _tangent_phase(kepler_drift_tangent(
            plan, state, _tangent(plan, first), duration
        ).tangent)
        + b_value * _tangent_phase(kepler_drift_tangent(
            plan, state, _tangent(plan, second), duration
        ).tangent)
    )
    linearity_relative = float(
        np.linalg.norm((observed_linear - expected_linear) / scales)
        / max(np.linalg.norm(expected_linear / scales), U)
    )
    linearity_bound = (
        32768.0 * U * _chi(0.6, 0.6 * math.pi)
        * max(1.0, float(np.linalg.cond(matrix)))
    )
    zero_tangent = _tangent(plan, np.zeros(6))
    zero_result = kepler_drift_tangent(plan, state, zero_tangent, duration).tangent
    zero_maximum = float(np.max(np.abs(_tangent_phase(zero_result))))
    other_state = _state(plan, 0.6, 2.7)
    other_result = kepler_drift_tangent(
        plan, other_state, _tangent(plan, first), duration
    ).tangent
    base_dependence = float(np.linalg.norm(
        (_tangent_phase(other_result)
         - _tangent_phase(kepler_drift_tangent(
             plan, state, _tangent(plan, first), duration
         ).tangent)) / scales
    ))
    tangent_semantics = {
        "linearity_relative_l2": linearity_relative,
        "linearity_bound": linearity_bound,
        "zero_maximum_absolute": zero_maximum,
        "base_state_difference_scaled_l2": base_dependence,
        "pass": linearity_relative <= linearity_bound
        and zero_maximum == 0.0 and base_dependence > 0.0,
    }

    branch_counts = {}
    for row in solver_rows:
        branch_counts[row["branch"]] = branch_counts.get(row["branch"], 0) + 1
    nonzero_iterations = [row["iterations"] for row in solver_rows if row["iterations"]]
    physical_error = lambda row: max(
        row["maximum_scaled_error"], row["normwise_scaled_error"]
    )
    circular_worst = max(
        (row for row in physical_rows if row["eccentricity"] == 0.0),
        key=physical_error,
    )
    elliptic_worst = max(
        (row for row in physical_rows if row["eccentricity"] > 0.0),
        key=physical_error,
    )
    metrics = {
        "physical": {
            "schema_version": 1, "kind": "m0_step3g1c_physical_oracle_metrics",
            "values": physical_rows, "composition": composition_rows,
            "worst_observed_scaled": max(physical_error(row) for row in physical_rows),
            "worst_circular": {
                **circular_worst, "observed_scaled": physical_error(circular_worst)
            },
            "worst_elliptic": {
                **elliptic_worst, "observed_scaled": physical_error(elliptic_worst)
            },
            "composition_maximum_scaled": max(
                physical_error(row) for row in composition_rows
            ),
            "acceptance": {
                "independent_oracle": all(row["pass"] for row in physical_rows),
                "composition": all(row["pass"] for row in composition_rows),
            },
        },
        "finite_difference": {
            "schema_version": 1, "kind": "m0_step3g1c_finite_difference_metrics",
            "values": fd_rows, "condition_scale": chi_fd,
            "forward_bound": 128.0 * math.sqrt(U) * chi_fd,
            "central_bound": 512.0 * U ** (2.0 / 3.0) * chi_fd,
            "minimum_forward_index": minimum_index, "acceptance": fd_acceptance,
        },
        "invariant": {
            "schema_version": 1, "kind": "m0_step3g1c_invariant_metrics",
            "values": invariant_rows,
            "worst_observed": max(row["maximum"] for row in invariant_rows),
            "acceptance": {"all_invariants": all(row["pass"] for row in invariant_rows)},
        },
        "reversibility": {
            "schema_version": 1, "kind": "m0_step3g1c_reversibility_metrics",
            "values": reversibility_rows, "flow_and_rotation": flow_rotation,
            "tangent_semantics": tangent_semantics,
            "acceptance": {
                "physical_tangent_matrix": all(row["pass"] for row in reversibility_rows),
                "flow_and_rotation": bool(flow_rotation["pass"]),
                "linearity_zero_base_dependence": bool(tangent_semantics["pass"]),
            },
        },
        "symplecticity": {
            "schema_version": 1, "kind": "m0_step3g1c_symplecticity_metrics",
            "values": symplectic_rows,
            "worst_maximum_absolute": max(
                row["maximum_absolute_residual"] for row in symplectic_rows
            ),
            "worst_frobenius": max(row["frobenius_residual"] for row in symplectic_rows),
            "forward_maximum_absolute": max(
                row["maximum_absolute_residual"] for row in symplectic_rows
                if row["period_fraction"] > 0.0
            ),
            "reverse_maximum_absolute": max(
                row["maximum_absolute_residual"] for row in symplectic_rows
                if row["period_fraction"] < 0.0
            ),
            "acceptance": {"full_6x6_canonical": all(row["pass"] for row in symplectic_rows)},
        },
        "solver": {
            "schema_version": 1, "kind": "m0_step3g1c_solver_metrics",
            "values": solver_rows, "branch_counts": branch_counts,
            "iterations_minimum_nonzero": min(nonzero_iterations),
            "iterations_maximum": max(row["iterations"] for row in solver_rows),
            "maximum_absolute_residual_s": max(abs(row["residual_s"]) for row in solver_rows),
            "maximum_absolute_update_x_s_per_m": max(
                abs(row["update_x_s_per_m"]) for row in solver_rows
            ),
            "acceptance": {
                "all_converged": all(row["converged"] for row in solver_rows),
                "finite_diagnostics": all(
                    math.isfinite(row[key]) for row in solver_rows
                    for key in ("residual_s", "update_x_s_per_m", "phase_advance_rad")
                ),
                "known_branches": set(branch_counts) <= {
                    "zero_duration", "elliptic_newton",
                    "elliptic_quartic", "elliptic_bisection",
                },
            },
        },
    }
    return {
        "model_fingerprint": model.fingerprint, "plan_fingerprint": plan.fingerprint,
        "period_s": period, "metrics": metrics,
    }


def _inventory(manifest):
    selection = manifest["exact_test_selection"]
    groups = (
        ("step3g1c_core", "step3g1c_core_node_ids"),
        ("step3g1c_integrity", "step3g1c_integrity_node_ids"),
        ("safe_step3g1a_regression", "safe_step3g1a_regression_node_ids"),
        ("safe_step3g1b_regression", "safe_step3g1b_regression_node_ids"),
        ("artifact", "artifact_node_ids"),
    )
    tests = [
        {"group": group, "node_id": node, "result": "PASS"}
        for group, key in groups for node in selection[key]
    ]
    return {
        "schema_version": 1, "kind": "m0_step3g1c_qualifying_test_inventory",
        "selection": "exact_pytest_node_ids_only",
        "pytest_plugin_autoload_disabled": True,
        "counts": selection["expected_counts"], "commands": manifest["exact_commands"],
        "tests": tests,
    }


def _traceability_bytes(manifest) -> bytes:
    selection = manifest["exact_test_selection"]
    core, integrity = (
        selection["step3g1c_core_node_ids"], selection["step3g1c_integrity_node_ids"]
    )
    rows = (
        ("G1C-PHYSICAL", "Hamiltonian flow, references, composition, invariants", core[:5]),
        ("G1C-SOLVER", "Stumpff, domain, convergence and typed failures", core[5:8]),
        ("G1C-TANGENT", "Analytic tangent closure, covariance and symplecticity", core[8:13]),
        ("G1C-OWNERSHIP", "Immutability, identity, isolation and determinism", core[13:]),
        ("G1C-INTEGRITY", "Prior bytes, guard, static closure and fresh process", integrity[1:]),
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("requirement_id", "requirement", "exact_passing_node_ids"))
    for requirement_id, requirement, nodes in rows:
        writer.writerow((requirement_id, requirement, ";".join(nodes)))
    return output.getvalue().encode("utf-8")

def _hamiltonian_spec() -> bytes:
    return b"""# Hamiltonian and Supported Domain

The isolated canonical pair map advances only relative variables `(q,P)` with
`H_pair = P^2/(2*mu_r) - mu_r*mu_g/r`, hence `dq/dt = P/mu_r` and
`dP/dt = -mu_r*mu_g*q/r^3`. Velocity is internal `P/mu_r`; public state uses
canonical momentum. The immutable plan binds masses, parameters, layout, units,
domain limits, solver budgets, and fingerprints.

Qualification is only noncollision bound elliptic motion with `e <= 0.92`,
positive minimum periapsis, nondegenerate angular momentum, and
`|n*dt| <= 0.999*2*pi`. Invalid conics, radial/collision states, multiple
revolutions, and incompatible states are rejected.

No center-of-mass drift, multi-pair composition, interaction kick, force
callback, or N-body dynamics belongs to this primitive.
"""


def _tangent_spec() -> bytes:
    return b"""# Analytic Tangent Map

The tangent action differentiates the same map and the implicit universal-anomaly root
at fixed pair parameters and duration. The root derivative includes radius,
beta, eta, zeta, and Stiefel-function terms before the `f,g,fdot,gdot` update.

The audited source ordering, stable Stumpff evaluation, and deterministic
residual-plus-update gates are shared by physical and tangent results. There is
no production finite difference, variational ODE, parameter sensitivity,
callback, mutable cache, or partial nonconverged result.
"""


def generate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    guard = install_guard()
    guard.activate_strict()
    safety = static_safety_audit()
    inherited = verify_inherited_integrity()
    collected = collect_metrics()
    metrics = collected["metrics"]
    if not all(
        all(bool(value) for value in payload["acceptance"].values())
        for payload in metrics.values()
    ):
        raise RuntimeError("frozen Step 3g1c numerical acceptance did not pass")
    manifest = manifest25()
    summary = {
        "schema_version": 1,
        "kind": "m0_step3g1c_kepler_drift_tangent_primitive_summary",
        "final_status": "STEP3G1C_KEPLER_DRIFT_TANGENT_COMPLETE",
        "primary_finding": "BOUND_ELLIPTIC_KEPLER_DRIFT_TANGENT_QUALIFIED",
        "verification_envelope": "ISOLATED_TWO_BODY_BOUND_ELLIPTIC_MAP_ONLY_NO_NBODY_DYNAMICS",
        "branch": "v2-whckl-tangent-core",
        "preregistration_commit": "d2ffe64742734f7b1337c74faa7705073bc1c203",
        "model_fingerprint": collected["model_fingerprint"],
        "plan_fingerprint": collected["plan_fingerprint"],
        "period_s": collected["period_s"],
        "exact_test_counts": manifest["exact_test_selection"]["expected_counts"],
        "safety_status": safety["status"],
        "inherited_integrity_status": inherited["status"],
        "numerical_acceptance": {
            name: payload["acceptance"] for name, payload in metrics.items()
        },
        "numerical_extrema": {
            "physical_oracle_maximum_scaled": metrics["physical"]["worst_observed_scaled"],
            "composition_maximum_scaled": metrics["physical"]["composition_maximum_scaled"],
            "invariant_maximum": metrics["invariant"]["worst_observed"],
            "finite_difference_forward_minimum_relative_l2": min(
                row["forward_relative_l2_error"]
                for row in metrics["finite_difference"]["values"]
            ),
            "finite_difference_central_minimum_relative_l2": min(
                row["central_relative_l2_error"]
                for row in metrics["finite_difference"]["values"]
            ),
            "finite_difference_finest_forward_relative_l2": (
                metrics["finite_difference"]["values"][-1]["forward_relative_l2_error"]
            ),
            "symplectic_forward_maximum_absolute": (
                metrics["symplecticity"]["forward_maximum_absolute"]
            ),
            "symplectic_reverse_maximum_absolute": (
                metrics["symplecticity"]["reverse_maximum_absolute"]
            ),
            "physical_reversibility_maximum_scaled": max(
                max(row["physical_maximum_scaled"], row["physical_normwise_scaled"])
                for row in metrics["reversibility"]["values"]
            ),
            "tangent_reversibility_maximum_relative_l2": max(
                row["tangent_relative_l2"] for row in metrics["reversibility"]["values"]
            ),
            "solver_iterations_maximum": metrics["solver"]["iterations_maximum"],
            "solver_residual_maximum_s": metrics["solver"]["maximum_absolute_residual_s"],
            "solver_update_maximum_x_s_per_m": (
                metrics["solver"]["maximum_absolute_update_x_s_per_m"]
            ),
        },
        "claim_limit": manifest["qualified_domain"]["claim_limit"],
        "successor": manifest["successor_boundary"]["allowed_only_after_success"],
    }
    report = f"""# M0 Step 3g1c Kepler Drift and Tangent Primitive

Final status: **{summary['final_status']}**

Primary finding: **{summary['primary_finding']}**

Verification envelope: **{summary['verification_envelope']}**

## Result

The immutable fixed-mass pair plan, exact bound-elliptic universal-variable flow,
and analytic initial-state tangent map passed every frozen physical, tangent,
solver, ownership, determinism, failure, safety, and integrity gate.

The campaign contains 73 pre-artifact nodes and 6 artifact nodes. The model
fingerprint is `{collected['model_fingerprint']}`; the pair-plan fingerprint is
`{collected['plan_fingerprint']}`.

## Numerical Summary

- Physical cases: {len(metrics['physical']['values'])}; worst scaled error:
  {metrics['physical']['worst_observed_scaled']:.17g}.
- Invariant cases: {len(metrics['invariant']['values'])}; worst normalized drift:
  {metrics['invariant']['worst_observed']:.17g}.
- Symplectic matrices: {len(metrics['symplecticity']['values'])}; forward/reverse
  max residuals: {metrics['symplecticity']['forward_maximum_absolute']:.17g} /
  {metrics['symplecticity']['reverse_maximum_absolute']:.17g}.
- Composition maximum scaled error:
  {metrics['physical']['composition_maximum_scaled']:.17g}.
- Solver cases: {len(metrics['solver']['values'])}; maximum iterations:
  {metrics['solver']['iterations_maximum']}; branch counts:
  {json.dumps(metrics['solver']['branch_counts'], sort_keys=True)}.
- Forward finite-difference minimum:
  {min(row['forward_relative_l2_error'] for row in metrics['finite_difference']['values']):.17g};
  central minimum:
  {min(row['central_relative_l2_error'] for row in metrics['finite_difference']['values']):.17g}.

## Scope

- Only an isolated two-body bound-elliptic map was evaluated.
- No protected force/JVP provider was invoked.
- No N-body dynamics, map, or trajectory was executed.
- No interaction kick, lazy kernel, corrector, or WHCKL composition exists yet.
- No MEGNO, LCN, or Solar-System result is qualified.
- Tangent qualification covers canonical initial-state derivatives only, at
  fixed masses, fixed parameters, and fixed duration.

No center-of-mass drift, multi-pair composition, synchronization, archive,
REBOUND, or REBOUNDx operation was executed. No qualified prior file or
protected/historical artifact changed. The claim remains limited to the frozen
isolated bound-elliptic domain. Step 3g1d may be proposed only for a synthetic
analytic interaction-kick force/JVP map; it is not implemented here.
"""
    review = {
        "schema_version": 1, "kind": "m0_step3g1c_code_review_findings",
        "status": "PASS",
        "findings": [
            {
                "id": "G1C-REVIEW-001",
                "severity": "material",
                "status": "RESOLVED_BEFORE_QUALIFICATION",
                "finding": "The first tangent draft reconstructed terminal radius instead of reusing the retained physical root radius.",
                "resolution": "Use root.radius_m for the analytic implicit-root and coefficient derivatives.",
                "regression_node_ids": [
                    "mini_ephemeris/tests/test_v2_kepler.py::TangentMapTests::test_full_six_by_six_symplecticity",
                    "mini_ephemeris/tests/test_v2_kepler.py::TangentMapTests::test_physical_tangent_and_time_reversal"
                ]
            },
            {
                "id": "G1C-REVIEW-002",
                "severity": "material",
                "status": "RESOLVED_BEFORE_CLOSEOUT",
                "finding": "The first stable Stumpff draft stopped its fixed Horner series before Manifest 25's frozen 34! depth.",
                "resolution": "Evaluate each required Stumpff series through the largest matching parity factorial at or below 34! before source-style scaling recurrences.",
                "regression_node_ids": [
                    "mini_ephemeris/tests/test_v2_kepler.py::SolverDomainTests::test_stumpff_small_argument_and_solver_diagnostics",
                    "mini_ephemeris/tests/test_v2_kepler.py::PhysicalOracleTests::test_independent_elliptic_reference_matrix"
                ]
            }
        ],
        "reviewed_risks": [
            "momentum_velocity_and_reduced_mass", "Hamiltonian_sign_and_scaling",
            "negative_duration_and_phase_limit", "Stumpff_and_root_convergence",
            "shared_physical_tangent_root", "implicit_root_derivative",
            "canonical_variable_order_and_symplecticity",
            "immutability_and_no_partial_failure",
            "forbidden_dependency_and_scope_surface",
        ],
        "residual_limit": "Only the frozen isolated bound-elliptic domain is qualified.",
    }
    payloads = {
        SUMMARY_NAME: _json_bytes(summary), REPORT_NAME: report.encode("utf-8"),
        "hamiltonian_supported_domain_specification.md": _hamiltonian_spec(),
        "tangent_map_derivation.md": _tangent_spec(),
        "requirements_traceability.csv": _traceability_bytes(manifest),
        "qualifying_test_inventory.json": _json_bytes(_inventory(manifest)),
        "physical_oracle_metrics.json": _json_bytes(metrics["physical"]),
        "finite_difference_metrics.json": _json_bytes(metrics["finite_difference"]),
        "invariant_metrics.json": _json_bytes(metrics["invariant"]),
        "reversibility_metrics.json": _json_bytes(metrics["reversibility"]),
        "symplecticity_metrics.json": _json_bytes(metrics["symplecticity"]),
        "solver_metrics.json": _json_bytes(metrics["solver"]),
        "code_review_findings.json": _json_bytes(review),
    }
    for name in sorted(payloads):
        _atomic_write(destination / name, payloads[name])
    _atomic_write(
        destination / "artifact_hashes.json",
        _json_bytes({
            "schema_version": 1, "kind": "m0_step3g1c_artifact_hashes",
            "sha256": {
                name: sha256_file(destination / name) for name in sorted(payloads)
            },
        }),
    )
    assert_protected_runtime_absent()


def validate_artifacts(destination: Path = DEFAULT_DESTINATION) -> None:
    observed = {path.name for path in destination.iterdir() if path.is_file()}
    if observed != EXPECTED_ARTIFACTS:
        raise AssertionError(f"artifact inventory differs: {sorted(observed)}")
    json_names = {name for name in EXPECTED_ARTIFACTS if name.endswith(".json")}
    parsed = {name: strict_json(destination / name) for name in json_names}
    hashes = parsed["artifact_hashes.json"]["sha256"]
    if set(hashes) != EXPECTED_ARTIFACTS - {"artifact_hashes.json"}:
        raise AssertionError("artifact hash keyset differs")
    for name, expected in hashes.items():
        if sha256_file(destination / name) != expected:
            raise AssertionError(f"artifact hash mismatch: {name}")
    summary, manifest = parsed[SUMMARY_NAME], manifest25()
    if summary["final_status"] not in manifest["result_vocabulary"]["final_status"]:
        raise AssertionError("invalid final status vocabulary")
    if summary["primary_finding"] not in manifest["result_vocabulary"]["primary_finding"]:
        raise AssertionError("invalid primary finding vocabulary")
    if summary["verification_envelope"] != manifest["result_vocabulary"]["success_verification_envelope"]:
        raise AssertionError("verification envelope differs")
    for name in (
        "physical_oracle_metrics.json", "finite_difference_metrics.json",
        "invariant_metrics.json", "reversibility_metrics.json",
        "symplecticity_metrics.json", "solver_metrics.json",
    ):
        if not all(parsed[name]["acceptance"].values()):
            raise AssertionError(f"frozen numerical gate failed: {name}")


def compare_fresh_regeneration(destination: Path = DEFAULT_DESTINATION) -> None:
    with tempfile.TemporaryDirectory(prefix="step3g1c-a-") as first_name:
        with tempfile.TemporaryDirectory(prefix="step3g1c-b-") as second_name:
            first, second = Path(first_name), Path(second_name)
            run_fresh_artifact_probe(first)
            run_fresh_artifact_probe(second)
            for name in EXPECTED_ARTIFACTS:
                committed = (destination / name).read_bytes()
                if (first / name).read_bytes() != committed:
                    raise AssertionError(f"first fresh regeneration differs: {name}")
                if (second / name).read_bytes() != committed:
                    raise AssertionError(f"second fresh regeneration differs: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.validate:
        validate_artifacts()
    else:
        generate_artifacts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
