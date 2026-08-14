"""Analytic and independent qualification of the isolated v2 Kepler map."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.v2.errors import InvalidModel, InvalidState, LayoutMismatch
from mini_ephemeris.v2.jacobi import build_jacobi_transform_plan
from mini_ephemeris.v2.kepler import (
    CanonicalKeplerPairState,
    CanonicalKeplerPairTangent,
    KeplerConvergenceError,
    KeplerDomainError,
    _stumpff_cs5,
    _stumpff_series,
    build_kepler_pair_plan,
    kepler_drift,
    kepler_drift_tangent,
)
from mini_ephemeris.v2.model import CompiledLayout, PhysicalModel, SI_UNITS


ROOT = Path(__file__).resolve().parents[2]
U = np.finfo(np.float64).eps
SEMIMAJOR_AXIS_M = 20_000_000.0
MINIMUM_PERIAPSIS_M = 1_000_000.0
EPSILON_LADDER = tuple(2.0**exponent for exponent in range(-10, -43, -4))
PERIOD_FRACTIONS = (-0.99, -0.5, -0.125, -0.001, 0.0, 0.001, 0.125, 0.5, 0.99)


def _fixture():
    layout = CompiledLayout(("inner", "outer"), "inner")
    model = PhysicalModel(
        model_id="synthetic_kepler_qualification",
        schema_version="1",
        layout=layout,
        masses_kg={"inner": 4.0, "outer": 4.0},
        gravitational_constant_si=50_000_000_000_000.0,
        units=SI_UNITS,
        enabled_effects=("synthetic-newtonian-pair",),
        provenance={"fixture": "step3g1c-synthetic"},
    )
    jacobi_plan = build_jacobi_transform_plan(model)
    plan = build_kepler_pair_plan(
        model,
        jacobi_plan,
        1,
        minimum_periapsis_m=MINIMUM_PERIAPSIS_M,
    )
    return model, jacobi_plan, plan


def _orientation(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "xy_aligned":
        first = np.array((1.0, 0.0, 0.0))
        second = np.array((0.0, 1.0, 0.0))
    elif name == "non_axis_aligned_rotation_A":
        first = np.array((1.0, 2.0, 3.0))
        second = np.array((-2.0, 1.0, 0.5))
    else:
        first = np.array((2.0, -1.0, 1.5))
        second = np.array((0.25, 3.0, -1.0))
    first = first / np.linalg.norm(first)
    second = second - np.dot(second, first) * first
    second = second / np.linalg.norm(second)
    normal = np.cross(first, second)
    return first, second, normal


def _state_from_elements(
    plan,
    eccentricity: float,
    true_anomaly: float,
    orientation: str = "non_axis_aligned_rotation_A",
) -> CanonicalKeplerPairState:
    first, second, _ = _orientation(orientation)
    parameter = SEMIMAJOR_AXIS_M * (1.0 - eccentricity * eccentricity)
    radius = parameter / (1.0 + eccentricity * math.cos(true_anomaly))
    q_plane = radius * (
        math.cos(true_anomaly) * first + math.sin(true_anomaly) * second
    )
    speed_scale = math.sqrt(plan.gravitational_parameter_m3_s2 / parameter)
    velocity = speed_scale * (
        -math.sin(true_anomaly) * first
        + (eccentricity + math.cos(true_anomaly)) * second
    )
    momentum = plan.reduced_mass_kg * velocity
    return CanonicalKeplerPairState(
        q_m=q_plane,
        p_kg_m_per_s=momentum,
        unit_system_id="si_v1",
        layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _tangent(plan, delta_q, delta_p) -> CanonicalKeplerPairTangent:
    return CanonicalKeplerPairTangent(
        delta_q_m=delta_q,
        delta_p_kg_m_per_s=delta_p,
        unit_system_id="si_v1",
        layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _period(plan) -> float:
    return 2.0 * math.pi * math.sqrt(
        SEMIMAJOR_AXIS_M**3 / plan.gravitational_parameter_m3_s2
    )


def _phase(state: CanonicalKeplerPairState) -> np.ndarray:
    return np.concatenate((np.asarray(state.q_m), np.asarray(state.p_kg_m_per_s)))


def _tangent_phase(tangent: CanonicalKeplerPairTangent) -> np.ndarray:
    return np.concatenate(
        (np.asarray(tangent.delta_q_m), np.asarray(tangent.delta_p_kg_m_per_s))
    )


def _scales(plan) -> np.ndarray:
    momentum_scale = plan.reduced_mass_kg * math.sqrt(
        plan.gravitational_parameter_m3_s2 / SEMIMAJOR_AXIS_M
    )
    return np.array((SEMIMAJOR_AXIS_M,) * 3 + (momentum_scale,) * 3)


def _scaled_error(plan, observed, expected) -> tuple[float, float]:
    difference = (_phase(observed) - _phase(expected)) / _scales(plan)
    return float(np.max(np.abs(difference))), float(np.linalg.norm(difference))


def _condition_scale(eccentricity: float, phase_advance: float) -> float:
    return (1.0 + eccentricity) * (1.0 + abs(phase_advance)) / (1.0 - eccentricity) ** 2


def _reference_elliptic(
    plan, state: CanonicalKeplerPairState, duration_s: float
) -> CanonicalKeplerPairState:
    q0 = np.asarray(state.q_m, dtype=np.float64)
    v0 = np.asarray(state.p_kg_m_per_s, dtype=np.float64) / plan.reduced_mass_kg
    gravitational_parameter = plan.gravitational_parameter_m3_s2
    radius0 = np.linalg.norm(q0)
    energy = 0.5 * np.dot(v0, v0) - gravitational_parameter / radius0
    semimajor_axis = -gravitational_parameter / (2.0 * energy)
    angular = np.cross(q0, v0)
    angular_hat = angular / np.linalg.norm(angular)
    eccentricity_vector = np.cross(v0, angular) / gravitational_parameter - q0 / radius0
    eccentricity = np.linalg.norm(eccentricity_vector)
    mean_motion = math.sqrt(gravitational_parameter / semimajor_axis**3)
    if eccentricity <= 64.0 * U:
        angle = mean_motion * duration_s
        cosine = math.cos(angle)
        sine = math.sin(angle)
        q1 = cosine * q0 + sine * np.cross(angular_hat, q0)
        v1 = cosine * v0 + sine * np.cross(angular_hat, v0)
    else:
        periapsis_hat = eccentricity_vector / eccentricity
        transverse_hat = np.cross(angular_hat, periapsis_hat)
        root = math.sqrt(1.0 - eccentricity * eccentricity)
        cosine_e0 = np.dot(q0, periapsis_hat) / semimajor_axis + eccentricity
        sine_e0 = np.dot(q0, transverse_hat) / (semimajor_axis * root)
        eccentric_anomaly0 = math.atan2(sine_e0, cosine_e0)
        mean_anomaly = (
            eccentric_anomaly0
            - eccentricity * math.sin(eccentric_anomaly0)
            + mean_motion * duration_s
        )
        reduced_mean = math.remainder(mean_anomaly, 2.0 * math.pi)
        lower = -math.pi
        upper = math.pi
        eccentric_anomaly = 0.0
        for _ in range(160):
            eccentric_anomaly = 0.5 * (lower + upper)
            residual = (
                eccentric_anomaly
                - eccentricity * math.sin(eccentric_anomaly)
                - reduced_mean
            )
            if residual >= 0.0:
                upper = eccentric_anomaly
            else:
                lower = eccentric_anomaly
        cosine_e = math.cos(eccentric_anomaly)
        sine_e = math.sin(eccentric_anomaly)
        denominator = 1.0 - eccentricity * cosine_e
        q1 = semimajor_axis * (
            (cosine_e - eccentricity) * periapsis_hat
            + root * sine_e * transverse_hat
        )
        v1 = semimajor_axis * mean_motion / denominator * (
            -sine_e * periapsis_hat + root * cosine_e * transverse_hat
        )
    return CanonicalKeplerPairState(
        q_m=q1,
        p_kg_m_per_s=plan.reduced_mass_kg * v1,
        unit_system_id="si_v1",
        layout_fingerprint=plan.layout_fingerprint,
        model_fingerprint=plan.model_fingerprint,
        pair_plan_fingerprint=plan.fingerprint,
    )


def _invariants(plan, state: CanonicalKeplerPairState):
    q = np.asarray(state.q_m)
    p = np.asarray(state.p_kg_m_per_s)
    velocity = p / plan.reduced_mass_kg
    radius = np.linalg.norm(q)
    specific_energy = 0.5 * np.dot(velocity, velocity) - plan.gravitational_parameter_m3_s2 / radius
    h = np.cross(q, velocity)
    evec = np.cross(velocity, h) / plan.gravitational_parameter_m3_s2 - q / radius
    semimajor_axis = -plan.gravitational_parameter_m3_s2 / (2.0 * specific_energy)
    hamiltonian = plan.reduced_mass_kg * specific_energy
    return hamiltonian, specific_energy, h, evec, semimajor_axis


def _normalized_jacobian(plan, state, duration_s) -> np.ndarray:
    scales = _scales(plan)
    columns = []
    for index in range(6):
        direction = np.zeros(6)
        direction[index] = scales[index]
        tangent = _tangent(plan, direction[:3], direction[3:])
        result = kepler_drift_tangent(plan, state, tangent, duration_s).tangent
        columns.append(_tangent_phase(result) / scales)
    return np.column_stack(columns)


class PhysicalOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        _, _, self.plan = _fixture()

    def test_zero_duration_identity_without_root_iteration(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        tangent = _tangent(self.plan, (1.0, -2.0, 3.0), (4.0, 5.0, -6.0))
        physical = kepler_drift(self.plan, state, 0.0)
        varied = kepler_drift_tangent(self.plan, state, tangent, 0.0)
        self.assertEqual(physical.state, state)
        self.assertEqual(varied.state, state)
        self.assertEqual(varied.tangent, tangent)
        self.assertEqual(physical.diagnostics.branch, "zero_duration")
        self.assertEqual(physical.diagnostics.iterations, 0)

    def test_circular_orbit_rigid_rotation_multiple_planes_and_signs(self) -> None:
        period = _period(self.plan)
        for orientation in (
            "non_axis_aligned_rotation_A",
            "non_axis_aligned_rotation_B",
        ):
            state = _state_from_elements(self.plan, 0.0, 0.37, orientation)
            for fraction in (-0.99, -0.125, 0.001, 0.5, 0.99):
                duration = fraction * period
                observed = kepler_drift(self.plan, state, duration).state
                expected = _reference_elliptic(self.plan, state, duration)
                maximum, normwise = _scaled_error(self.plan, observed, expected)
                bound = 8192.0 * U * _condition_scale(0.0, abs(fraction) * 2.0 * math.pi)
                self.assertLessEqual(max(maximum, normwise), bound)

    def test_independent_elliptic_reference_matrix(self) -> None:
        period = _period(self.plan)
        cases = ((0.1, 0.37), (0.6, 1.4), (0.9, 2.7), (0.919, 3.05), (0.919, 0.0))
        for eccentricity, anomaly in cases:
            state = _state_from_elements(self.plan, eccentricity, anomaly)
            for fraction in PERIOD_FRACTIONS:
                duration = fraction * period
                observed = kepler_drift(self.plan, state, duration).state
                expected = _reference_elliptic(self.plan, state, duration)
                maximum, normwise = _scaled_error(self.plan, observed, expected)
                phase = abs(fraction) * 2.0 * math.pi
                bound = 8192.0 * U * _condition_scale(eccentricity, phase)
                self.assertLessEqual(max(maximum, normwise), bound)

    def test_composition_and_negative_duration(self) -> None:
        period = _period(self.plan)
        for eccentricity, anomaly in ((0.1, 0.37), (0.6, 2.7), (0.9, 1.4)):
            state = _state_from_elements(self.plan, eccentricity, anomaly)
            first_duration = 0.17 * period
            second_duration = -0.08 * period
            first = kepler_drift(self.plan, state, first_duration).state
            composed = kepler_drift(self.plan, first, second_duration).state
            direct = kepler_drift(
                self.plan, state, first_duration + second_duration
            ).state
            maximum, normwise = _scaled_error(self.plan, composed, direct)
            chi = _condition_scale(eccentricity, 0.17 * 2.0 * math.pi)
            self.assertLessEqual(max(maximum, normwise), 32768.0 * U * chi)

    def test_invariants_across_frozen_matrix(self) -> None:
        period = _period(self.plan)
        for eccentricity, anomaly in ((0.0, 0.37), (0.1, 1.4), (0.6, 2.7), (0.9, 0.37), (0.919, 3.05), (0.919, 0.0)):
            state = _state_from_elements(self.plan, eccentricity, anomaly)
            initial = _invariants(self.plan, state)
            for fraction in (-0.99, -0.125, 0.001, 0.5, 0.99):
                result = kepler_drift(self.plan, state, fraction * period).state
                final = _invariants(self.plan, result)
                values = [
                    abs((final[0] - initial[0]) / initial[0]),
                    abs((final[1] - initial[1]) / initial[1]),
                    np.linalg.norm(final[2] - initial[2]) / np.linalg.norm(initial[2]),
                    np.linalg.norm(final[3] - initial[3]),
                    abs((final[4] - initial[4]) / initial[4]),
                ]
                chi = _condition_scale(eccentricity, abs(fraction) * 2.0 * math.pi)
                self.assertLessEqual(max(values), 65536.0 * U * chi)


class SolverDomainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.jacobi_plan, self.plan = _fixture()

    def test_stumpff_small_argument_and_solver_diagnostics(self) -> None:
        for z_value in (0.0, 2.0**-40, 2.0**-20, 0.09):
            observed = _stumpff_cs5(z_value)
            for index in range(6):
                expected = sum(
                    (-z_value) ** term / math.factorial(index + 2 * term)
                    for term in range(30)
                )
                self.assertAlmostEqual(observed[index], expected, delta=16.0 * U)
        for index in range(6):
            maximum_term = (34 - index) // 2
            expected_frozen_series = sum(
                (-1.0) ** term / math.factorial(index + 2 * term)
                for term in range(maximum_term + 1)
            )
            self.assertAlmostEqual(
                _stumpff_series(1.0, index),
                expected_frozen_series,
                delta=4.0 * U,
            )
        state = _state_from_elements(self.plan, 0.6, 1.4)
        result = kepler_drift(self.plan, state, 0.5 * _period(self.plan))
        repeated = kepler_drift(self.plan, state, 0.5 * _period(self.plan))
        self.assertEqual(result, repeated)
        self.assertTrue(result.diagnostics.converged)
        self.assertGreater(result.diagnostics.iterations, 0)
        self.assertIn(
            result.diagnostics.branch,
            {"elliptic_newton", "elliptic_quartic", "elliptic_bisection"},
        )
        self.assertTrue(math.isfinite(result.diagnostics.residual_s))

    def test_rejects_invalid_parameters_states_conics_and_domain(self) -> None:
        with self.assertRaises(InvalidModel):
            PhysicalModel(
                model_id="bad",
                schema_version="1",
                layout=self.model.layout,
                masses_kg={"inner": 4.0, "outer": 0.0},
                gravitational_constant_si=1.0,
                units=SI_UNITS,
                enabled_effects=("synthetic",),
                provenance={"fixture": "bad"},
            )
        with self.assertRaises(InvalidModel):
            build_kepler_pair_plan(
                self.model,
                self.jacobi_plan,
                1,
                minimum_periapsis_m=0.0,
            )
        zero = _state_from_elements(self.plan, 0.1, 0.0)
        zero_radius = CanonicalKeplerPairState(
            q_m=(0.0, 0.0, 0.0),
            p_kg_m_per_s=zero.p_kg_m_per_s,
            unit_system_id="si_v1",
            layout_fingerprint=self.plan.layout_fingerprint,
            model_fingerprint=self.plan.model_fingerprint,
            pair_plan_fingerprint=self.plan.fingerprint,
        )
        with self.assertRaises(KeplerDomainError):
            kepler_drift(self.plan, zero_radius, 1.0)
        escape_speed = math.sqrt(
            2.0 * self.plan.gravitational_parameter_m3_s2 / SEMIMAJOR_AXIS_M
        )
        for speed_factor in (1.0, 1.1):
            conic = CanonicalKeplerPairState(
                q_m=(SEMIMAJOR_AXIS_M, 0.0, 0.0),
                p_kg_m_per_s=(
                    0.0,
                    self.plan.reduced_mass_kg * escape_speed * speed_factor,
                    0.0,
                ),
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
            with self.assertRaises(KeplerDomainError):
                kepler_drift(self.plan, conic, 1.0)
        radial = CanonicalKeplerPairState(
            q_m=(SEMIMAJOR_AXIS_M, 0.0, 0.0),
            p_kg_m_per_s=(1000.0, 0.0, 0.0),
            unit_system_id="si_v1",
            layout_fingerprint=self.plan.layout_fingerprint,
            model_fingerprint=self.plan.model_fingerprint,
            pair_plan_fingerprint=self.plan.fingerprint,
        )
        with self.assertRaises(KeplerDomainError):
            kepler_drift(self.plan, radial, 1.0)
        with self.assertRaises(KeplerDomainError):
            kepler_drift(self.plan, _state_from_elements(self.plan, 0.93, 0.0), 1.0)
        with self.assertRaises(KeplerDomainError):
            kepler_drift(
                self.plan,
                _state_from_elements(self.plan, 0.6, 1.4),
                1.0001 * _period(self.plan),
            )
        with self.assertRaises(KeplerDomainError):
            kepler_drift(self.plan, zero, math.inf)
        with self.assertRaises(InvalidState):
            CanonicalKeplerPairState(
                q_m=(math.nan, 0.0, 0.0),
                p_kg_m_per_s=(0.0, 0.0, 0.0),
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )

    def test_nonconvergence_is_typed_and_returns_no_partial_result(self) -> None:
        limited = build_kepler_pair_plan(
            self.model,
            self.jacobi_plan,
            1,
            minimum_periapsis_m=MINIMUM_PERIAPSIS_M,
            newton_max_iterations=1,
            quartic_max_iterations=1,
            bisection_max_iterations=1,
        )
        state = _state_from_elements(limited, 0.919, 0.0)
        tangent = _tangent(limited, (1.0, 2.0, 3.0), (-4.0, 5.0, -6.0))
        before_state = state.canonical_bytes()
        before_tangent = tangent.canonical_bytes()
        with self.assertRaises(KeplerConvergenceError):
            kepler_drift(limited, state, 0.99 * _period(limited))
        with self.assertRaises(KeplerConvergenceError):
            kepler_drift_tangent(limited, state, tangent, 0.99 * _period(limited))
        self.assertEqual(state.canonical_bytes(), before_state)
        self.assertEqual(tangent.canonical_bytes(), before_tangent)


class TangentMapTests(unittest.TestCase):
    def setUp(self) -> None:
        _, _, self.plan = _fixture()
        self.period = _period(self.plan)
        self.scales = _scales(self.plan)

    def test_zero_linearity_determinism_and_base_state_dependence(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        other_state = _state_from_elements(self.plan, 0.6, 2.7)
        first_values = self.scales * np.array((0.2, -0.3, 0.1, 0.4, 0.05, -0.2))
        second_values = self.scales * np.array((-0.1, 0.15, 0.25, -0.2, 0.3, 0.1))
        first = _tangent(self.plan, first_values[:3], first_values[3:])
        second = _tangent(self.plan, second_values[:3], second_values[3:])
        a_value, b_value = 1.25, -0.75
        combined_values = a_value * first_values + b_value * second_values
        combined = _tangent(self.plan, combined_values[:3], combined_values[3:])
        duration = 0.3 * self.period
        observed = _tangent_phase(
            kepler_drift_tangent(self.plan, state, combined, duration).tangent
        )
        expected = a_value * _tangent_phase(
            kepler_drift_tangent(self.plan, state, first, duration).tangent
        ) + b_value * _tangent_phase(
            kepler_drift_tangent(self.plan, state, second, duration).tangent
        )
        matrix = _normalized_jacobian(self.plan, state, duration)
        condition = np.linalg.cond(matrix)
        chi = _condition_scale(0.6, 0.6 * math.pi)
        relative = np.linalg.norm((observed - expected) / self.scales) / max(
            np.linalg.norm(expected / self.scales), U
        )
        self.assertLessEqual(relative, 32768.0 * U * chi * max(1.0, condition))
        zero = _tangent(self.plan, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        zero_result = kepler_drift_tangent(self.plan, state, zero, duration)
        self.assertEqual(zero_result.tangent.delta_q_m, (0.0, 0.0, 0.0))
        self.assertEqual(zero_result.tangent.delta_p_kg_m_per_s, (0.0, 0.0, 0.0))
        repeated = kepler_drift_tangent(self.plan, state, first, duration)
        self.assertEqual(repeated, kepler_drift_tangent(self.plan, state, first, duration))
        other = kepler_drift_tangent(self.plan, other_state, first, duration)
        self.assertFalse(np.allclose(_tangent_phase(repeated.tangent), _tangent_phase(other.tangent), rtol=1e-8, atol=0.0))

    def test_forward_and_central_finite_difference_ladders(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        direction_scaled = np.array((0.2, -0.1, 0.05, -0.15, 0.25, 0.1))
        direction_values = self.scales * direction_scaled
        direction = _tangent(self.plan, direction_values[:3], direction_values[3:])
        duration = 0.3 * self.period
        base_output = _phase(kepler_drift(self.plan, state, duration).state)
        analytic = _tangent_phase(
            kepler_drift_tangent(self.plan, state, direction, duration).tangent
        )
        base = _phase(state)
        forward_errors = []
        central_errors = []
        for epsilon in EPSILON_LADDER:
            plus_values = base + epsilon * direction_values
            minus_values = base - epsilon * direction_values
            plus = CanonicalKeplerPairState(
                q_m=plus_values[:3],
                p_kg_m_per_s=plus_values[3:],
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
            minus = CanonicalKeplerPairState(
                q_m=minus_values[:3],
                p_kg_m_per_s=minus_values[3:],
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
            plus_output = _phase(kepler_drift(self.plan, plus, duration).state)
            minus_output = _phase(kepler_drift(self.plan, minus, duration).state)
            forward = (plus_output - base_output) / epsilon
            central = (plus_output - minus_output) / (2.0 * epsilon)
            forward_errors.append(
                np.linalg.norm((forward - analytic) / self.scales)
                / np.linalg.norm(analytic / self.scales)
            )
            central_errors.append(
                np.linalg.norm((central - analytic) / self.scales)
                / np.linalg.norm(analytic / self.scales)
            )
        chi = _condition_scale(0.6, 0.6 * math.pi)
        self.assertLessEqual(min(forward_errors), 128.0 * math.sqrt(U) * chi)
        self.assertLessEqual(min(central_errors), 512.0 * U ** (2.0 / 3.0) * chi)
        self.assertLessEqual(min(forward_errors[:6]), 0.25 * forward_errors[0])
        minimum_index = int(np.argmin(forward_errors))
        self.assertLess(minimum_index, len(forward_errors) - 1)
        self.assertGreaterEqual(max(forward_errors[minimum_index + 1 :]), 2.0 * forward_errors[minimum_index])

    def test_flow_direction_and_rotational_covariance(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        duration = 0.3 * self.period
        q = np.asarray(state.q_m)
        p = np.asarray(state.p_kg_m_per_s)
        radius = np.linalg.norm(q)
        flow_values = np.concatenate(
            (
                p / self.plan.reduced_mass_kg,
                -self.plan.reduced_mass_kg
                * self.plan.gravitational_parameter_m3_s2
                * q
                / radius**3,
            )
        )
        flow = _tangent(self.plan, flow_values[:3], flow_values[3:])
        result = kepler_drift_tangent(self.plan, state, flow, duration)
        q1 = np.asarray(result.state.q_m)
        p1 = np.asarray(result.state.p_kg_m_per_s)
        radius1 = np.linalg.norm(q1)
        expected_flow = np.concatenate(
            (
                p1 / self.plan.reduced_mass_kg,
                -self.plan.reduced_mass_kg
                * self.plan.gravitational_parameter_m3_s2
                * q1
                / radius1**3,
            )
        )
        relative_flow = np.linalg.norm(
            (_tangent_phase(result.tangent) - expected_flow) / self.scales
        ) / np.linalg.norm(expected_flow / self.scales)
        matrix = _normalized_jacobian(self.plan, state, duration)
        bound = 65536.0 * U * _condition_scale(0.6, 0.6 * math.pi) * max(1.0, np.linalg.cond(matrix))
        self.assertLessEqual(relative_flow, bound)

        omega = np.array((0.3, -0.4, 0.5))
        rotation_values = np.concatenate((np.cross(omega, q), np.cross(omega, p)))
        rotation = _tangent(self.plan, rotation_values[:3], rotation_values[3:])
        rotated = kepler_drift_tangent(self.plan, state, rotation, duration)
        expected_rotation = np.concatenate(
            (
                np.cross(omega, np.asarray(rotated.state.q_m)),
                np.cross(omega, np.asarray(rotated.state.p_kg_m_per_s)),
            )
        )
        relative_rotation = np.linalg.norm(
            (_tangent_phase(rotated.tangent) - expected_rotation) / self.scales
        ) / np.linalg.norm(expected_rotation / self.scales)
        self.assertLessEqual(relative_rotation, bound)

    def test_full_six_by_six_symplecticity(self) -> None:
        identity3 = np.eye(3)
        zero = np.zeros((3, 3))
        symplectic = np.block([[zero, identity3], [-identity3, zero]])
        for eccentricity, anomaly in ((0.0, 0.37), (0.6, 1.4), (0.9, 2.7), (0.919, 3.05), (0.919, 0.0)):
            state = _state_from_elements(self.plan, eccentricity, anomaly)
            for fraction in (-0.99, -0.125, 0.5, 0.99):
                matrix = _normalized_jacobian(self.plan, state, fraction * self.period)
                residual = matrix.T @ symplectic @ matrix - symplectic
                condition = np.linalg.cond(matrix)
                absolute_bound = 65536.0 * U * 6.0 * max(1.0, condition**2)
                scaled_bound = 65536.0 * U * 6.0 * max(1.0, condition)
                self.assertLessEqual(np.max(np.abs(residual)), absolute_bound)
                self.assertLessEqual(np.linalg.norm(residual, "fro"), absolute_bound)
                scaled = np.linalg.norm(residual, "fro") / (
                    np.linalg.norm(matrix, 2) ** 2 * np.linalg.norm(symplectic, "fro")
                )
                self.assertLessEqual(scaled, scaled_bound)
                self.assertTrue(math.isfinite(float(np.linalg.det(matrix))))

    def test_physical_tangent_and_time_reversal(self) -> None:
        identity = np.eye(6)
        for eccentricity, anomaly in ((0.1, 0.37), (0.6, 1.4), (0.9, 2.7)):
            state = _state_from_elements(self.plan, eccentricity, anomaly)
            duration = 0.5 * self.period
            forward = kepler_drift(self.plan, state, duration).state
            recovered = kepler_drift(self.plan, forward, -duration).state
            maximum, normwise = _scaled_error(self.plan, recovered, state)
            chi = _condition_scale(eccentricity, math.pi)
            self.assertLessEqual(max(maximum, normwise), 32768.0 * U * chi)
            forward_matrix = _normalized_jacobian(self.plan, state, duration)
            reverse_matrix = _normalized_jacobian(self.plan, forward, -duration)
            matrix_residual = reverse_matrix @ forward_matrix - identity
            condition_product = np.linalg.cond(forward_matrix) * np.linalg.cond(reverse_matrix)
            self.assertLessEqual(
                np.max(np.abs(matrix_residual)),
                65536.0 * U * 6.0 * max(1.0, condition_product),
            )
            direction_values = self.scales * np.array((0.2, -0.1, 0.05, -0.15, 0.25, 0.1))
            direction = _tangent(self.plan, direction_values[:3], direction_values[3:])
            advanced = kepler_drift_tangent(self.plan, state, direction, duration)
            returned = kepler_drift_tangent(
                self.plan, advanced.state, advanced.tangent, -duration
            ).tangent
            relative = np.linalg.norm(
                (_tangent_phase(returned) - direction_values) / self.scales
            ) / np.linalg.norm(direction_values / self.scales)
            self.assertLessEqual(
                relative,
                65536.0 * U * chi * max(1.0, np.linalg.cond(forward_matrix)),
            )
            reversed_initial = CanonicalKeplerPairState(
                q_m=state.q_m,
                p_kg_m_per_s=-np.asarray(state.p_kg_m_per_s),
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
            reversed_forward = kepler_drift(
                self.plan, reversed_initial, duration
            ).state
            momentum_flipped = CanonicalKeplerPairState(
                q_m=reversed_forward.q_m,
                p_kg_m_per_s=-np.asarray(reversed_forward.p_kg_m_per_s),
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
            backward = kepler_drift(self.plan, state, -duration).state
            maximum, normwise = _scaled_error(self.plan, momentum_flipped, backward)
            self.assertLessEqual(max(maximum, normwise), 32768.0 * U * chi)


class OwnershipIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.jacobi_plan, self.plan = _fixture()

    def test_inputs_outputs_immutable_detached_and_nonmutating_on_success(self) -> None:
        q = [SEMIMAJOR_AXIS_M, 0.0, 0.0]
        speed = math.sqrt(self.plan.gravitational_parameter_m3_s2 / SEMIMAJOR_AXIS_M)
        p = [0.0, self.plan.reduced_mass_kg * speed, 0.0]
        state = CanonicalKeplerPairState(
            q_m=q,
            p_kg_m_per_s=p,
            unit_system_id="si_v1",
            layout_fingerprint=self.plan.layout_fingerprint,
            model_fingerprint=self.plan.model_fingerprint,
            pair_plan_fingerprint=self.plan.fingerprint,
        )
        q[0] = -1.0
        p[1] = -1.0
        before = state.canonical_bytes()
        result = kepler_drift(self.plan, state, 0.125 * _period(self.plan))
        self.assertEqual(state.canonical_bytes(), before)
        self.assertNotEqual(result.state.q_m[0], -1.0)
        with self.assertRaises(TypeError):
            result.state.q_m[0] = 1.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            self.plan.reduced_mass_kg = 1.0  # type: ignore[misc]

    def test_nonmutation_on_failure_and_fingerprint_unit_shape_rejection(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        tangent = _tangent(self.plan, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        before_state = state.canonical_bytes()
        before_tangent = tangent.canonical_bytes()
        wrong = CanonicalKeplerPairState(
            q_m=state.q_m,
            p_kg_m_per_s=state.p_kg_m_per_s,
            unit_system_id="si_v1",
            layout_fingerprint="0" * 64,
            model_fingerprint=self.plan.model_fingerprint,
            pair_plan_fingerprint=self.plan.fingerprint,
        )
        with self.assertRaises(LayoutMismatch):
            kepler_drift(self.plan, wrong, 1.0)
        with self.assertRaises(InvalidState):
            CanonicalKeplerPairState(
                q_m=(1.0, 2.0),
                p_kg_m_per_s=(1.0, 2.0, 3.0),
                unit_system_id="si_v1",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
        with self.assertRaises(InvalidState):
            CanonicalKeplerPairTangent(
                delta_q_m=(1.0, 2.0, 3.0),
                delta_p_kg_m_per_s=(1.0, 2.0, 3.0),
                unit_system_id="not-si",
                layout_fingerprint=self.plan.layout_fingerprint,
                model_fingerprint=self.plan.model_fingerprint,
                pair_plan_fingerprint=self.plan.fingerprint,
            )
        with self.assertRaises(KeplerDomainError):
            kepler_drift_tangent(
                self.plan, state, tangent, 1.1 * _period(self.plan)
            )
        self.assertEqual(state.canonical_bytes(), before_state)
        self.assertEqual(tangent.canonical_bytes(), before_tangent)

    def test_observer_accounting_noninterference_and_determinism(self) -> None:
        state = _state_from_elements(self.plan, 0.6, 1.4)
        tangent = _tangent(self.plan, (1.0, 2.0, 3.0), (-4.0, 5.0, -6.0))
        duration = 0.125 * _period(self.plan)
        first = kepler_drift_tangent(self.plan, state, tangent, duration)
        second = kepler_drift_tangent(self.plan, state, tangent, duration)
        self.assertEqual(first, second)
        self.assertEqual(self.plan.fingerprint, build_kepler_pair_plan(
            self.model,
            self.jacobi_plan,
            1,
            minimum_periapsis_m=MINIMUM_PERIAPSIS_M,
        ).fingerprint)

    def test_source_has_no_forbidden_dependencies_or_operations(self) -> None:
        path = ROOT / "mini_ephemeris/src/mini_ephemeris/v2/kepler.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertFalse(imported & {"numpy", "rebound", "reboundx", "scipy"})
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(calls & {"integrate", "step", "evaluate", "jvp", "observe"})
        names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertFalse(
            names
            & {
                "kick",
                "lazy",
                "corrector",
                "whckl",
                "megno",
                "lcn",
                "restart",
                "finite_difference",
            }
        )


if __name__ == "__main__":
    unittest.main()
