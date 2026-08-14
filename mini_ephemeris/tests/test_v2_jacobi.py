"""Independent analytic and numerical qualification of v2 Jacobi primitives."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest

import numpy as np

from mini_ephemeris.v2.errors import InvalidModel, InvalidState, LayoutMismatch
from mini_ephemeris.v2.jacobi import (
    InertialCanonicalState,
    InertialCanonicalTangentState,
    JacobiTransformPlan,
    build_jacobi_transform_plan,
    canonical_jacobi_state_bytes,
    canonical_jacobi_tangent_bytes,
    from_canonical_jacobi,
    from_canonical_jacobi_tangent,
    to_canonical_jacobi,
    to_canonical_jacobi_tangent,
)
from mini_ephemeris.v2.model import CompiledLayout, PhysicalModel, SI_UNITS
from mini_ephemeris.v2.state import (
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    InertialCartesianState,
)


ROOT = Path(__file__).resolve().parents[2]
U = np.finfo(np.float64).eps
GENERAL_MASSES = (0.125, 2.0, 32.0, 0.5, 8.0)
EPSILON_LADDER = tuple(2.0**exponent for exponent in range(-4, -41, -4))


def _model(masses: tuple[float, ...], *, order: tuple[str, ...] | None = None) -> PhysicalModel:
    body_ids = order or tuple("sun" if index == 0 else f"body-{index}" for index in range(len(masses)))
    layout = CompiledLayout(body_ids, "sun")
    return PhysicalModel(
        model_id="synthetic_jacobi_qualification",
        schema_version="1",
        layout=layout,
        masses_kg=dict(zip(body_ids, masses)),
        gravitational_constant_si=1.0,
        units=SI_UNITS,
        enabled_effects=("synthetic-none",),
        provenance={"fixture": "step3g1b-synthetic"},
    )


def _state(
    model: PhysicalModel,
    positions: object,
    momenta: object,
) -> InertialCanonicalState:
    return InertialCanonicalState(
        layout=model.layout,
        positions_m=positions,
        momenta_kg_m_per_s=momenta,
        unit_system_id="si_v1",
        model_fingerprint=model.fingerprint,
    )


def _tangent(
    model: PhysicalModel,
    positions: object,
    momenta: object,
) -> InertialCanonicalTangentState:
    return InertialCanonicalTangentState(
        layout=model.layout,
        delta_positions_m=positions,
        delta_momenta_kg_m_per_s=momenta,
        unit_system_id="si_v1",
        model_fingerprint=model.fingerprint,
    )


def _fixture_rows(count: int, scale: float = 1.0) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (
            scale * (0.375 + 0.625 * index),
            scale * (-0.75 + 0.3125 * index),
            scale * (1.125 - 0.4375 * index),
        )
        for index in range(count)
    )


def _fraction_matrix(masses: tuple[float, ...]) -> list[list[Fraction]]:
    values = [Fraction.from_float(value) for value in masses]
    total = sum(values, Fraction(0))
    matrix = [[Fraction(0) for _ in values] for _ in values]
    for column, mass in enumerate(values):
        matrix[0][column] = mass / total
    cumulative = values[0]
    for row in range(1, len(values)):
        for column in range(row):
            matrix[row][column] = -values[column] / cumulative
        matrix[row][row] = Fraction(1)
        cumulative += values[row]
    return matrix


def _fraction_inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    count = len(matrix)
    augmented = [
        row.copy() + [Fraction(int(row_index == column)) for column in range(count)]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(count):
        pivot = next(row for row in range(column, count) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(count):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [row[count:] for row in augmented]


def _dense_operators(masses: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractions = _fraction_matrix(masses)
    inverse = _fraction_inverse(fractions)
    a = np.array([[float(value) for value in row] for row in fractions], dtype=np.float64)
    a_inverse = np.array(
        [[float(value) for value in row] for row in inverse], dtype=np.float64
    )
    a3 = np.kron(a, np.eye(3, dtype=np.float64))
    momentum = np.kron(a_inverse.T, np.eye(3, dtype=np.float64))
    zero = np.zeros_like(a3)
    phase = np.block([[a3, zero], [zero, momentum]])
    return a, a_inverse, phase


def _phase_from_inertial(state: InertialCanonicalState) -> np.ndarray:
    return np.concatenate(
        (np.asarray(state.positions_m).ravel(), np.asarray(state.momenta_kg_m_per_s).ravel())
    )


def _phase_from_jacobi(state: CanonicalJacobiState) -> np.ndarray:
    return np.concatenate((np.asarray(state.q_m).ravel(), np.asarray(state.p_kg_m_per_s).ravel()))


def _phase_from_inertial_tangent(state: InertialCanonicalTangentState) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(state.delta_positions_m).ravel(),
            np.asarray(state.delta_momenta_kg_m_per_s).ravel(),
        )
    )


def _phase_from_jacobi_tangent(state: CanonicalJacobiTangentState) -> np.ndarray:
    return np.concatenate(
        (np.asarray(state.delta_q_m).ravel(), np.asarray(state.delta_p_kg_m_per_s).ravel())
    )


def _bounds(values: np.ndarray, body_count: int, condition: float) -> tuple[float, float]:
    component = 256.0 * U * body_count * max(1.0, condition) * max(1.0, np.linalg.norm(values, np.inf))
    normwise = (
        256.0
        * U
        * math.sqrt(6.0 * body_count)
        * max(1.0, condition)
        * max(1.0, np.linalg.norm(values))
    )
    return component, normwise


class AnalyticOracleTests(unittest.TestCase):
    def test_one_body_identity_state_and_tangent(self) -> None:
        model = _model((2.0,))
        plan = build_jacobi_transform_plan(model)
        state = _state(model, ((1.0, -2.0, 4.0),), ((3.0, 5.0, -7.0),))
        tangent = _tangent(model, ((0.5, -0.25, 0.125),), ((-1.0, 2.0, 4.0),))
        canonical = to_canonical_jacobi(plan, model, state)
        transformed_tangent = to_canonical_jacobi_tangent(plan, model, state, tangent)
        self.assertEqual(canonical.q_m, state.positions_m)
        self.assertEqual(canonical.p_kg_m_per_s, state.momenta_kg_m_per_s)
        self.assertEqual(transformed_tangent.delta_q_m, tangent.delta_positions_m)
        self.assertEqual(
            transformed_tangent.delta_p_kg_m_per_s, tangent.delta_momenta_kg_m_per_s
        )
        self.assertEqual(from_canonical_jacobi(plan, model, canonical), state)
        self.assertEqual(
            from_canonical_jacobi_tangent(plan, model, canonical, transformed_tangent),
            tangent,
        )

    def test_two_body_direct_formulas_and_inverse(self) -> None:
        model = _model((1.0, 1.0))
        plan = build_jacobi_transform_plan(model)
        x0, x1 = (2.0, -4.0, 8.0), (6.0, 2.0, -2.0)
        p0, p1 = (4.0, -8.0, 2.0), (-2.0, 6.0, 10.0)
        state = _state(model, (x0, x1), (p0, p1))
        result = to_canonical_jacobi(plan, model, state)
        expected_q = (
            tuple((a + b) / 2.0 for a, b in zip(x0, x1)),
            tuple(b - a for a, b in zip(x0, x1)),
        )
        expected_p = (
            tuple(a + b for a, b in zip(p0, p1)),
            tuple((b - a) / 2.0 for a, b in zip(p0, p1)),
        )
        self.assertEqual(result.q_m, expected_q)
        self.assertEqual(result.p_kg_m_per_s, expected_p)
        inverse = from_canonical_jacobi(plan, model, result)
        self.assertEqual(inverse.positions_m, (x0, x1))
        self.assertEqual(inverse.momenta_kg_m_per_s, (p0, p1))

    def test_three_body_hand_formulas_all_axes(self) -> None:
        masses = (1.0, 2.0, 5.0)
        model = _model(masses)
        plan = build_jacobi_transform_plan(model)
        x = ((2.0, -1.0, 3.0), (5.0, 7.0, -2.0), (-4.0, 6.0, 9.0))
        p = ((3.0, -5.0, 2.0), (-7.0, 4.0, 8.0), (6.0, 11.0, -9.0))
        result = to_canonical_jacobi(plan, model, _state(model, x, p))
        expected_q0 = tuple((x[0][axis] + 2.0 * x[1][axis] + 5.0 * x[2][axis]) / 8.0 for axis in range(3))
        expected_q1 = tuple(x[1][axis] - x[0][axis] for axis in range(3))
        expected_q2 = tuple(x[2][axis] - (x[0][axis] + 2.0 * x[1][axis]) / 3.0 for axis in range(3))
        expected_p0 = tuple(sum(row[axis] for row in p) for axis in range(3))
        expected_p1 = tuple(p[1][axis] / 3.0 - 2.0 * p[0][axis] / 3.0 for axis in range(3))
        expected_p2 = tuple(3.0 * p[2][axis] / 8.0 - 5.0 * (p[0][axis] + p[1][axis]) / 8.0 for axis in range(3))
        np.testing.assert_allclose(result.q_m, (expected_q0, expected_q1, expected_q2), rtol=0.0, atol=4.0 * U)
        np.testing.assert_allclose(result.p_kg_m_per_s, (expected_p0, expected_p1, expected_p2), rtol=0.0, atol=8.0 * U)
        inverse = from_canonical_jacobi(plan, model, result)
        np.testing.assert_allclose(inverse.positions_m, x, rtol=0.0, atol=16.0 * U)
        np.testing.assert_allclose(inverse.momenta_kg_m_per_s, p, rtol=0.0, atol=32.0 * U)
        self.assertFalse(np.allclose(np.asarray(result.q_m[0]), 0.0))
        self.assertFalse(np.allclose(np.asarray(result.p_kg_m_per_s[0]), 0.0))

    def test_dense_rational_oracle_general_fixture(self) -> None:
        model = _model(GENERAL_MASSES)
        plan = build_jacobi_transform_plan(model)
        state = _state(model, _fixture_rows(5), _fixture_rows(5, -1.75))
        result = to_canonical_jacobi(plan, model, state)
        _, _, phase = _dense_operators(GENERAL_MASSES)
        expected = phase @ _phase_from_inertial(state)
        observed = _phase_from_jacobi(result)
        condition = np.linalg.cond(phase)
        bound = 128.0 * U * 5 * condition * max(1.0, np.linalg.norm(expected, np.inf))
        self.assertLessEqual(np.max(np.abs(observed - expected)), bound)


class RoundTripAndInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model(GENERAL_MASSES)
        self.plan = build_jacobi_transform_plan(self.model)
        self.state = _state(self.model, _fixture_rows(5, 3.25), _fixture_rows(5, -2.75))
        self.tangent = _tangent(self.model, _fixture_rows(5, 0.125), _fixture_rows(5, -0.375))
        _, _, self.phase = _dense_operators(GENERAL_MASSES)

    def test_state_round_trips_both_directions_with_frozen_bounds(self) -> None:
        initial = _phase_from_inertial(self.state)
        recovered = _phase_from_inertial(
            from_canonical_jacobi(
                self.plan, self.model, to_canonical_jacobi(self.plan, self.model, self.state)
            )
        )
        component, normwise = _bounds(initial, 5, np.linalg.cond(self.phase))
        self.assertLessEqual(np.max(np.abs(recovered - initial)), component)
        self.assertLessEqual(np.linalg.norm(recovered - initial), normwise)

        canonical = to_canonical_jacobi(self.plan, self.model, self.state)
        initial_canonical = _phase_from_jacobi(canonical)
        recovered_canonical = _phase_from_jacobi(
            to_canonical_jacobi(
                self.plan,
                self.model,
                from_canonical_jacobi(self.plan, self.model, canonical),
            )
        )
        component, normwise = _bounds(initial_canonical, 5, np.linalg.cond(self.phase))
        self.assertLessEqual(np.max(np.abs(recovered_canonical - initial_canonical)), component)
        self.assertLessEqual(np.linalg.norm(recovered_canonical - initial_canonical), normwise)

    def test_tangent_round_trips_both_directions_with_frozen_bounds(self) -> None:
        canonical = to_canonical_jacobi(self.plan, self.model, self.state)
        transformed = to_canonical_jacobi_tangent(
            self.plan, self.model, self.state, self.tangent
        )
        recovered = from_canonical_jacobi_tangent(
            self.plan, self.model, canonical, transformed
        )
        initial = _phase_from_inertial_tangent(self.tangent)
        observed = _phase_from_inertial_tangent(recovered)
        component, normwise = _bounds(initial, 5, np.linalg.cond(self.phase))
        self.assertLessEqual(np.max(np.abs(observed - initial)), component)
        self.assertLessEqual(np.linalg.norm(observed - initial), normwise)

        transformed_again = to_canonical_jacobi_tangent(
            self.plan, self.model, self.state, recovered
        )
        initial_canonical = _phase_from_jacobi_tangent(transformed)
        observed_canonical = _phase_from_jacobi_tangent(transformed_again)
        component, normwise = _bounds(initial_canonical, 5, np.linalg.cond(self.phase))
        self.assertLessEqual(np.max(np.abs(observed_canonical - initial_canonical)), component)
        self.assertLessEqual(np.linalg.norm(observed_canonical - initial_canonical), normwise)

    def test_translation_and_mass_weighted_boost_invariance(self) -> None:
        original = to_canonical_jacobi(self.plan, self.model, self.state)
        translation = np.array((3.5, -2.25, 6.75))
        velocity_boost = np.array((-0.5, 1.25, 2.0))
        positions = np.asarray(self.state.positions_m) + translation
        momenta = np.asarray(self.state.momenta_kg_m_per_s) + np.asarray(GENERAL_MASSES)[:, None] * velocity_boost
        shifted = to_canonical_jacobi(
            self.plan, self.model, _state(self.model, positions, momenta)
        )
        np.testing.assert_allclose(
            np.asarray(shifted.q_m[0]) - np.asarray(original.q_m[0]),
            translation,
            rtol=0.0,
            atol=16.0 * U,
        )
        np.testing.assert_allclose(np.asarray(shifted.q_m[1:]), original.q_m[1:], rtol=0.0, atol=64.0 * U)
        np.testing.assert_allclose(
            np.asarray(shifted.p_kg_m_per_s[0]) - np.asarray(original.p_kg_m_per_s[0]),
            sum(GENERAL_MASSES) * velocity_boost,
            rtol=0.0,
            atol=256.0 * U,
        )
        np.testing.assert_allclose(
            np.asarray(shifted.p_kg_m_per_s[1:]),
            original.p_kg_m_per_s[1:],
            rtol=0.0,
            atol=256.0 * U,
        )


class ValidationOwnershipTests(unittest.TestCase):
    def test_invalid_masses_and_noncentral_first_plan_rejected(self) -> None:
        with self.assertRaises(InvalidModel):
            _model((1.0, 0.0))
        with self.assertRaises(InvalidModel):
            _model((1.0, math.nan))
        with self.assertRaises(InvalidModel):
            build_jacobi_transform_plan(
                _model((1.0, 2.0), order=("body-1", "sun"))
            )
        layout = CompiledLayout(("sun", "body-1"), "sun")
        with self.assertRaises(InvalidModel):
            PhysicalModel(
                model_id="missing",
                schema_version="1",
                layout=layout,
                masses_kg={"sun": 1.0},
                gravitational_constant_si=1.0,
                units=SI_UNITS,
                enabled_effects=("none",),
                provenance={"fixture": "missing"},
            )

    def test_layout_reordering_and_model_fingerprint_mismatch_rejected(self) -> None:
        model = _model((1.0, 2.0, 4.0))
        plan = build_jacobi_transform_plan(model)
        reordered = _model((1.0, 4.0, 2.0), order=("sun", "body-2", "body-1"))
        state = _state(reordered, _fixture_rows(3), _fixture_rows(3, -1.0))
        with self.assertRaises(LayoutMismatch):
            to_canonical_jacobi(plan, model, state)
        changed_model = PhysicalModel(
            model_id=model.model_id,
            schema_version=model.schema_version,
            layout=model.layout,
            masses_kg={"sun": 1.0, "body-1": 2.0, "body-2": 5.0},
            gravitational_constant_si=1.0,
            units=SI_UNITS,
            enabled_effects=("synthetic-none",),
            provenance={"fixture": "step3g1b-synthetic"},
        )
        with self.assertRaises(LayoutMismatch):
            to_canonical_jacobi(plan, changed_model, _state(changed_model, _fixture_rows(3), _fixture_rows(3)))

    def test_semantic_units_shapes_dtype_and_fingerprint_rejected(self) -> None:
        model = _model((1.0, 2.0))
        plan = build_jacobi_transform_plan(model)
        with self.assertRaises(InvalidState):
            _state(model, ((1.0, 2.0), (3.0, 4.0, 5.0)), _fixture_rows(2))
        with self.assertRaises(InvalidState):
            _state(model, (("1", 2.0, 3.0), (4.0, 5.0, 6.0)), _fixture_rows(2))
        with self.assertRaises(InvalidState):
            InertialCanonicalState(
                layout=model.layout,
                positions_m=_fixture_rows(2),
                momenta_kg_m_per_s=_fixture_rows(2),
                unit_system_id="astronomical",
                model_fingerprint=model.fingerprint,
            )
        with self.assertRaises(InvalidState):
            InertialCanonicalState(
                layout=model.layout,
                positions_m=_fixture_rows(2),
                momenta_kg_m_per_s=_fixture_rows(2),
                unit_system_id="si_v1",
                model_fingerprint="not-a-fingerprint",
            )
        wrong_fingerprint = InertialCanonicalState(
            layout=model.layout,
            positions_m=_fixture_rows(2),
            momenta_kg_m_per_s=_fixture_rows(2),
            unit_system_id="si_v1",
            model_fingerprint="0" * 64,
        )
        with self.assertRaises(LayoutMismatch):
            to_canonical_jacobi(plan, model, wrong_fingerprint)
        velocity_state = InertialCartesianState(
            model.layout, _fixture_rows(2), _fixture_rows(2), "si_v1"
        )
        with self.assertRaises(InvalidState):
            to_canonical_jacobi(plan, model, velocity_state)  # type: ignore[arg-type]

    def test_immutable_detached_outputs_and_deterministic_plan_serialization(self) -> None:
        model = _model((1.0, 2.0, 4.0))
        positions = [list(row) for row in _fixture_rows(3)]
        momenta = [list(row) for row in _fixture_rows(3, -2.0)]
        state = _state(model, positions, momenta)
        positions[0][0] = 999.0
        momenta[0][0] = 999.0
        self.assertNotEqual(state.positions_m[0][0], 999.0)
        self.assertNotEqual(state.momenta_kg_m_per_s[0][0], 999.0)
        plan = build_jacobi_transform_plan(model)
        result = to_canonical_jacobi(plan, model, state)
        with self.assertRaises(TypeError):
            result.q_m[0][0] = 1.0  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            plan.masses_kg = (9.0,)  # type: ignore[misc]
        self.assertEqual(plan.canonical_bytes(), build_jacobi_transform_plan(model).canonical_bytes())
        self.assertEqual(plan.fingerprint, build_jacobi_transform_plan(model).fingerprint)
        self.assertEqual(
            canonical_jacobi_state_bytes(plan, model, result),
            canonical_jacobi_state_bytes(plan, model, result),
        )


class TangentClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _model(GENERAL_MASSES)
        self.plan = build_jacobi_transform_plan(self.model)
        self.base_a = _state(self.model, _fixture_rows(5), _fixture_rows(5, -1.0))
        self.base_b = _state(self.model, _fixture_rows(5, 9.0), _fixture_rows(5, 7.0))

    def test_linearity_zero_and_base_state_independence(self) -> None:
        first = _tangent(self.model, _fixture_rows(5, 0.25), _fixture_rows(5, -0.5))
        second = _tangent(self.model, _fixture_rows(5, -0.75), _fixture_rows(5, 0.125))
        a, b = 1.5, -0.625
        combined = _tangent(
            self.model,
            a * np.asarray(first.delta_positions_m) + b * np.asarray(second.delta_positions_m),
            a * np.asarray(first.delta_momenta_kg_m_per_s) + b * np.asarray(second.delta_momenta_kg_m_per_s),
        )
        output = _phase_from_jacobi_tangent(
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, combined)
        )
        expected = a * _phase_from_jacobi_tangent(
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, first)
        ) + b * _phase_from_jacobi_tangent(
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, second)
        )
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=512.0 * U)
        zero = _tangent(self.model, np.zeros((5, 3)), np.zeros((5, 3)))
        self.assertTrue(
            np.array_equal(
                _phase_from_jacobi_tangent(
                    to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, zero)
                ),
                np.zeros(30),
            )
        )
        self.assertEqual(
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, first),
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_b, first),
        )

    def test_independent_matrix_closure_and_finite_difference_ladder(self) -> None:
        direction = _tangent(self.model, _fixture_rows(5, 0.0625), _fixture_rows(5, -0.09375))
        analytic = _phase_from_jacobi_tangent(
            to_canonical_jacobi_tangent(self.plan, self.model, self.base_a, direction)
        )
        _, _, phase = _dense_operators(GENERAL_MASSES)
        dense = phase @ _phase_from_inertial_tangent(direction)
        bound = 128.0 * U * 5 * np.linalg.cond(phase) * max(1.0, np.linalg.norm(dense, np.inf))
        self.assertLessEqual(np.max(np.abs(analytic - dense)), bound)

        base_output = _phase_from_jacobi(to_canonical_jacobi(self.plan, self.model, self.base_a))
        base_phase = _phase_from_inertial(self.base_a)
        delta_phase = _phase_from_inertial_tangent(direction)
        relative_errors = []
        for epsilon in EPSILON_LADDER:
            perturbed = base_phase + epsilon * delta_phase
            perturbed_state = _state(
                self.model, perturbed[:15].reshape(5, 3), perturbed[15:].reshape(5, 3)
            )
            difference = (
                _phase_from_jacobi(to_canonical_jacobi(self.plan, self.model, perturbed_state))
                - base_output
            ) / epsilon
            self.assertTrue(np.all(np.isfinite(difference)))
            relative_errors.append(np.linalg.norm(difference - analytic) / np.linalg.norm(analytic))
        floor = 512.0 * U * max(1.0, np.linalg.cond(phase))
        self.assertLessEqual(min(relative_errors[:4]), floor)
        self.assertLessEqual(min(relative_errors), floor)


class SymplecticityTests(unittest.TestCase):
    def test_full_phase_space_forward_and_inverse_symplecticity(self) -> None:
        _, _, phase = _dense_operators(GENERAL_MASSES)
        inverse = np.linalg.inv(phase)
        half = phase.shape[0] // 2
        identity = np.eye(half)
        zero = np.zeros((half, half))
        symplectic = np.block([[zero, identity], [-identity, zero]])
        dimension = phase.shape[0]
        condition = np.linalg.cond(phase)
        absolute_bound = 512.0 * U * dimension * max(1.0, condition**2)
        scaled_bound = 512.0 * U * dimension
        for operator in (phase, inverse):
            residual = operator.T @ symplectic @ operator - symplectic
            self.assertLessEqual(np.max(np.abs(residual)), absolute_bound)
            self.assertLessEqual(np.linalg.norm(residual, "fro"), absolute_bound)
            scaled = np.linalg.norm(residual, "fro") / (
                np.linalg.norm(operator, 2) ** 2 * np.linalg.norm(symplectic, "fro")
            )
            self.assertLessEqual(scaled, scaled_bound)
        self.assertLessEqual(
            np.max(np.abs(inverse @ phase - np.eye(dimension))), absolute_bound
        )
        self.assertTrue(math.isfinite(float(np.linalg.det(phase))))


class IsolationTests(unittest.TestCase):
    def test_runtime_source_is_linear_recurrence_without_dense_or_dynamics_surface(self) -> None:
        path = ROOT / "mini_ephemeris/src/mini_ephemeris/v2/jacobi.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertFalse(imported & {"numpy", "rebound", "reboundx"})
        function_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        self.assertFalse(
            function_names
            & {"integrate", "step", "kepler", "kick", "lazy", "corrector", "whckl"}
        )
        self.assertFalse(any(isinstance(node, ast.MatMult) for node in ast.walk(tree)))
        self.assertNotIn("velocity", " ".join(function_names))

    def test_plan_and_result_repeated_calls_are_deterministic(self) -> None:
        model = _model(GENERAL_MASSES)
        state = _state(model, _fixture_rows(5), _fixture_rows(5, -1.0))
        tangent = _tangent(model, _fixture_rows(5, 0.25), _fixture_rows(5, -0.125))
        plans = [build_jacobi_transform_plan(model) for _ in range(3)]
        self.assertEqual(len({plan.fingerprint for plan in plans}), 1)
        results = [to_canonical_jacobi(plans[0], model, state) for _ in range(3)]
        self.assertEqual(results[0], results[1])
        tangents = [
            to_canonical_jacobi_tangent(plans[0], model, state, tangent) for _ in range(3)
        ]
        self.assertEqual(tangents[0], tangents[1])
        self.assertEqual(
            canonical_jacobi_tangent_bytes(plans[0], model, tangents[0]),
            canonical_jacobi_tangent_bytes(plans[0], model, tangents[1]),
        )
        json.loads(plans[0].canonical_bytes())
