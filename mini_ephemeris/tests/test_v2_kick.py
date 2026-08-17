"""Qualification tests for the isolated synthetic canonical interaction kick."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import math
import unittest
from unittest import mock

import numpy as np

from mini_ephemeris.m0_step3g1d_qualification import (
    AFFINE_EXACT,
    NONLINEAR_SMOOTH,
    COMPOSITION_CAP,
    FD_CAP,
    PHYSICAL_CAP,
    REVERSIBILITY_CAP,
    SYMMETRY_CAP,
    SYMPLECTIC_CAP,
    TANGENT_CAP,
    analytic_tangent_matrix,
    canonical_force_and_jacobian,
    expected_physical,
    assess_finite_difference_ladder,
    expected_tangent,
    finite_difference_gate_spec,
    finite_difference_series,
    fixture,
    phase,
    runtime_tangent_matrix,
    scaled_error,
    symplectic_residual,
    synthetic_model,
    tangent_phase,
)
from mini_ephemeris.v2.errors import (
    InvalidState,
    KernelContractError,
    LayoutMismatch,
)
from mini_ephemeris.v2.jacobi import build_jacobi_transform_plan
from mini_ephemeris.v2.kernels import evaluate_force, evaluate_jvp
from mini_ephemeris.v2.kick import (
    InteractionProviderCapabilities,
    apply_interaction_kick,
    apply_interaction_kick_tangent,
    build_interaction_kick_plan,
)
from mini_ephemeris.v2.model import CompiledLayout
from mini_ephemeris.v2.state import (
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    CartesianAcceleration,
    CartesianAccelerationJVP,
)
from mini_ephemeris.v2.timebase import ExactSeconds


ROOT = Path(__file__).resolve().parents[2]


class ProviderProxy:
    """Test-only immutable-by-convention malformed provider adapter."""

    def __init__(self, delegate, variant, *, capabilities=None, fingerprint=None):
        self.delegate = delegate
        self.variant = variant
        self.capabilities = capabilities or delegate.capabilities
        self.provider_fingerprint = fingerprint or delegate.provider_fingerprint

    def evaluate(self, model, state, context):
        if self.variant == "wrong_type":
            return ()
        if self.variant == "wrong_shape":
            return CartesianAcceleration(
                model.layout, ((0.0, 0.0, 0.0),), "si_v1"
            )
        if self.variant == "nonfinite":
            rows = [(0.0, 0.0, 0.0) for _ in model.layout.body_ids]
            rows[0] = (float("nan"), 0.0, 0.0)
            return CartesianAcceleration(model.layout, rows, "si_v1")
        if self.variant == "wrong_layout":
            layout = CompiledLayout(
                tuple(reversed(model.layout.body_ids)),
                model.layout.central_body,
            )
            return CartesianAcceleration(
                layout,
                [(0.0, 0.0, 0.0) for _ in layout.body_ids],
                "si_v1",
            )
        result = self.delegate.evaluate(model, state, context)
        if self.variant in {"roundoff_offset", "above_bound"}:
            rows = [list(row) for row in result.values_m_per_s2]
            if self.variant == "roundoff_offset":
                for _ in range(32):
                    rows[0][0] = math.nextafter(rows[0][0], math.inf)
            else:
                rows[0][0] += 0.5
            return CartesianAcceleration(model.layout, rows, "si_v1")
        return result

    def jvp(self, model, state, direction, context):
        if self.variant == "wrong_jvp_type":
            return ()
        if self.variant == "nonfinite_jvp":
            rows = [(0.0, 0.0, 0.0) for _ in model.layout.body_ids]
            rows[0] = (float("inf"), 0.0, 0.0)
            return CartesianAccelerationJVP(model.layout, rows, "si_v1")
        return self.delegate.jvp(model, state, direction, context)


def _apply(kind="dense", duration=ExactSeconds(7, 4)):
    model, jacobi, provider, plan, state, tangent, context = fixture(kind)
    physical = apply_interaction_kick(
        plan, jacobi, model, provider, state, duration, context
    )
    varied = apply_interaction_kick_tangent(
        plan, jacobi, model, provider, state, tangent, duration, context
    )
    return (
        model,
        jacobi,
        provider,
        plan,
        state,
        tangent,
        context,
        physical,
        varied,
    )


class FiniteDifferenceGateTests(unittest.TestCase):
    def test_affine_exact_zero_improvements_passes(self) -> None:
        spec = finite_difference_gate_spec("dense")
        values = [1.0e-15 * (index + 1) for index in range(10)]
        result = assess_finite_difference_ladder(
            spec, values, [1.0] * 10, 0.0
        )
        self.assertEqual(spec.derivative_class, AFFINE_EXACT)
        self.assertEqual(result["early_improvements"], 0)
        self.assertTrue(result["oracle_pass"])
        self.assertTrue(result["roundoff_model"]["consistent"])
        self.assertTrue(result["acceptance"])

    def test_incorrect_affine_derivative_fails_oracle_and_cap(self) -> None:
        spec = finite_difference_gate_spec("dense")
        result = assess_finite_difference_ladder(
            spec,
            [2.0 * FD_CAP] * 10,
            [1.0] * 10,
            2.0 * TANGENT_CAP,
        )
        self.assertFalse(result["oracle_pass"])
        self.assertFalse(result["acceptance"])

    def test_nonlinear_smooth_required_pattern_passes(self) -> None:
        spec = finite_difference_gate_spec("nonlinear")
        values = [
            1.0e-4,
            1.0e-6,
            1.0e-8,
            1.0e-10,
            2.0e-10,
            4.0e-10,
            8.0e-10,
            1.6e-9,
            3.2e-9,
            6.4e-9,
        ]
        result = assess_finite_difference_ladder(
            spec, values, [math.nan] * 10, math.nan
        )
        self.assertEqual(spec.derivative_class, NONLINEAR_SMOOTH)
        self.assertEqual(result["early_improvements"], 3)
        self.assertFalse(result["diagnostic_inputs_finite"])
        self.assertTrue(result["acceptance"])

    def test_nonlinear_smooth_missing_pattern_fails(self) -> None:
        spec = finite_difference_gate_spec("nonlinear")
        values = [1.0e-10 * (index + 1) for index in range(10)]
        result = assess_finite_difference_ladder(
            spec, values, [1.0] * 10, 0.0
        )
        self.assertEqual(result["early_improvements"], 0)
        self.assertFalse(result["acceptance"])

    def test_derivative_class_is_immutable_across_evaluation(self) -> None:
        spec = finite_difference_gate_spec("dense")
        first = assess_finite_difference_ladder(
            spec,
            [1.0e-15 * (index + 1) for index in range(10)],
            [1.0] * 10,
            0.0,
        )
        self.assertTrue(first["acceptance"])
        with self.assertRaises(FrozenInstanceError):
            spec.derivative_class = NONLINEAR_SMOOTH
        tampered = replace(spec, derivative_class=NONLINEAR_SMOOTH)
        with self.assertRaisesRegex(
            ValueError, "derivative classification changed"
        ):
            assess_finite_difference_ladder(
                tampered,
                [1.0e-15] * 10,
                [1.0] * 10,
                0.0,
            )

    def test_gate_classification_and_threshold_serialization_is_deterministic(
        self,
    ) -> None:
        first = finite_difference_gate_spec("dense")
        second = finite_difference_gate_spec("dense")
        nonlinear = finite_difference_gate_spec("nonlinear")
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.canonical_payload()["absolute_cap_hex"],
            FD_CAP.hex(),
        )
        self.assertIsNone(first.minimum_early_improvements)
        self.assertEqual(nonlinear.minimum_early_improvements, 3)
        self.assertTrue(nonlinear.require_roundoff_turn)
        self.assertNotEqual(first.fingerprint, nonlinear.fingerprint)


class PlanSemanticsTests(unittest.TestCase):
    def test_acceleration_to_canonical_force_dense_oracle_and_com_row(self) -> None:
        *_, state, _, _, physical, _ = _apply("dense", ExactSeconds(1))
        expected_force, _, _ = canonical_force_and_jacobian("dense", state.q_m)
        observed = np.asarray(physical.canonical_force_kg_m_per_s2).reshape(-1)
        self.assertLessEqual(
            np.max(np.abs(observed - expected_force)), PHYSICAL_CAP
        )
        self.assertEqual(
            physical.canonical_force_kg_m_per_s2[0], (0.0, 0.0, 0.0)
        )
        self.assertEqual(physical.state.q_m, state.q_m)
        self.assertEqual(physical.state.p_kg_m_per_s[0], state.p_kg_m_per_s[0])

    def test_plan_and_capability_fingerprints_are_deterministic(self) -> None:
        first = fixture("dense")
        second = fixture("dense")
        self.assertEqual(first[3].fingerprint, second[3].fingerprint)
        self.assertEqual(first[3].canonical_bytes(), second[3].canonical_bytes())
        self.assertEqual(
            first[2].capabilities.fingerprint,
            second[2].capabilities.fingerprint,
        )
        self.assertNotEqual(first[3].fingerprint, fixture("nonlinear")[3].fingerprint)

    def test_incompatible_capabilities_are_rejected_before_evaluation(self) -> None:
        model, jacobi, provider, _, *_ = fixture("dense")
        for changes in (
            {"conservative_canonical_force": False},
            {"symmetric_canonical_jacobian": False},
            {"position_only": False},
            {"raw_force_output": "canonical_generalized_force"},
            {"canonical_adapter": "implicit_acceleration_relabel"},
        ):
            with self.subTest(changes=changes):
                capabilities = replace(provider.capabilities, **changes)
                with self.assertRaises(KernelContractError):
                    build_interaction_kick_plan(model, jacobi, capabilities)
        other = synthetic_model()
        other_layout = CompiledLayout(
            tuple(reversed(other.layout.body_ids)), other.layout.central_body
        )
        mismatched = replace(
            provider.capabilities,
            layout_fingerprint=other_layout.fingerprint,
        )
        with self.assertRaises(LayoutMismatch):
            build_interaction_kick_plan(model, jacobi, mismatched)

    def test_below_bound_roundoff_residual_is_recorded_and_projected(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("dense")
        result = apply_interaction_kick(
            plan,
            jacobi,
            model,
            ProviderProxy(provider, "roundoff_offset"),
            state,
            ExactSeconds(1),
            context,
        )
        diagnostics = result.metadata.force_com_projection
        self.assertIsNotNone(diagnostics)
        self.assertGreater(diagnostics.raw_residual_norm_kg_m_per_s2, 0.0)
        self.assertLessEqual(
            diagnostics.raw_residual_norm_kg_m_per_s2,
            diagnostics.derived_bound_norm_kg_m_per_s2,
        )
        for residual, bound in zip(
            diagnostics.raw_residual_kg_m_per_s2,
            diagnostics.component_bounds_kg_m_per_s2,
        ):
            self.assertLessEqual(abs(residual), bound)
        self.assertTrue(diagnostics.projection_applied)

    def test_exact_projected_com_force(self) -> None:
        for kind in ("dense", "nonlinear"):
            values = _apply(kind)
            physical = values[7]
            self.assertEqual(
                physical.canonical_force_kg_m_per_s2[0],
                (0.0, 0.0, 0.0),
            )
            diagnostics = physical.metadata.force_com_projection
            self.assertIsNotNone(diagnostics)
            self.assertTrue(diagnostics.projection_applied)
            self.assertEqual(diagnostics.accumulated_force_terms, 4)
            self.assertEqual(diagnostics.rounded_operation_count, 7)


class PhysicalKickTests(unittest.TestCase):
    def test_zero_duration_identity_and_no_calls(self) -> None:
        model, jacobi, provider, plan, state, tangent, context = fixture("dense")
        with mock.patch(
            "mini_ephemeris.v2.kick.evaluate_force", wraps=evaluate_force
        ) as force_call, mock.patch(
            "mini_ephemeris.v2.kick.evaluate_jvp", wraps=evaluate_jvp
        ) as jvp_call:
            physical = apply_interaction_kick(
                plan, jacobi, model, provider, state, ExactSeconds(0), context
            )
            varied = apply_interaction_kick_tangent(
                plan,
                jacobi,
                model,
                provider,
                state,
                tangent,
                ExactSeconds(0),
                context,
            )
        self.assertEqual(force_call.call_count, 0)
        self.assertEqual(jvp_call.call_count, 0)
        self.assertEqual(physical.state, state)
        self.assertEqual(varied.state, state)
        self.assertEqual(varied.tangent, tangent)
        self.assertIsNot(physical.state, state)

    def test_dense_quadratic_physical_oracle_for_signed_durations(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("dense")
        for duration in (
            ExactSeconds(1),
            ExactSeconds(-1),
            ExactSeconds(1, 1024),
            ExactSeconds(-7, 4),
        ):
            result = apply_interaction_kick(
                plan, jacobi, model, provider, state, duration, context
            )
            expected = expected_physical(
                state, "dense", duration.to_binary64()
            )
            self.assertLessEqual(
                scaled_error(phase(result.state), phase(expected))[0],
                PHYSICAL_CAP,
            )

    def test_nonlinear_quartic_physical_oracle(self) -> None:
        *_, state, _, _, physical, _ = _apply("nonlinear")
        expected = expected_physical(state, "nonlinear", 1.75)
        self.assertLessEqual(
            scaled_error(phase(physical.state), phase(expected))[0],
            PHYSICAL_CAP,
        )

    def test_reversal_and_composition(self) -> None:
        for kind in ("dense", "nonlinear"):
            model, jacobi, provider, plan, state, _, context = fixture(kind)
            forward = apply_interaction_kick(
                plan, jacobi, model, provider, state, ExactSeconds(7, 4), context
            ).state
            backward = apply_interaction_kick(
                plan, jacobi, model, provider, forward, ExactSeconds(-7, 4), context
            ).state
            self.assertLessEqual(
                scaled_error(phase(backward), phase(state))[0],
                REVERSIBILITY_CAP,
            )
            first = apply_interaction_kick(
                plan, jacobi, model, provider, state, ExactSeconds(1, 2), context
            ).state
            second = apply_interaction_kick(
                plan, jacobi, model, provider, first, ExactSeconds(5, 4), context
            ).state
            direct = apply_interaction_kick(
                plan, jacobi, model, provider, state, ExactSeconds(7, 4), context
            ).state
            self.assertLessEqual(
                scaled_error(phase(second), phase(direct))[0],
                COMPOSITION_CAP,
            )

    def test_repeated_calls_are_deterministic(self) -> None:
        values = _apply("nonlinear")
        args = (
            values[3], values[1], values[0], values[2], values[4],
            ExactSeconds(7, 4), values[6],
        )
        first = apply_interaction_kick(*args)
        second = apply_interaction_kick(*args)
        self.assertEqual(first, second)


class TangentKickTests(unittest.TestCase):
    def test_dense_and_nonlinear_analytic_tangent_oracles(self) -> None:
        for kind in ("dense", "nonlinear"):
            values = _apply(kind)
            state, tangent, varied = values[4], values[5], values[8]
            expected = expected_tangent(state, tangent, kind, 1.75)
            self.assertLessEqual(
                scaled_error(
                    tangent_phase(varied.tangent), tangent_phase(expected)
                )[0],
                TANGENT_CAP,
            )

    def test_zero_linearity_and_nonlinear_base_state_dependence(self) -> None:
        model, jacobi, provider, plan, state, tangent, context = fixture(
            "nonlinear"
        )
        zero = CanonicalJacobiTangentState(
            model.layout, np.zeros((4, 3)), np.zeros((4, 3)), "si_v1"
        )
        zero_result = apply_interaction_kick_tangent(
            plan, jacobi, model, provider, state, zero, ExactSeconds(1), context
        )
        self.assertEqual(tangent_phase(zero_result.tangent).tolist(), [0.0] * 24)
        half = CanonicalJacobiTangentState(
            model.layout,
            np.asarray(tangent.delta_q_m) * 0.5,
            np.asarray(tangent.delta_p_kg_m_per_s) * 0.5,
            "si_v1",
        )
        full_result = apply_interaction_kick_tangent(
            plan, jacobi, model, provider, state, tangent, ExactSeconds(1), context
        )
        half_result = apply_interaction_kick_tangent(
            plan, jacobi, model, provider, state, half, ExactSeconds(1), context
        )
        self.assertTrue(
            np.allclose(
                tangent_phase(full_result.tangent),
                2.0 * tangent_phase(half_result.tangent),
                rtol=0.0,
                atol=TANGENT_CAP,
            )
        )
        moved_q = np.asarray(state.q_m).copy()
        moved_q[2, 0] += 0.25
        moved = CanonicalJacobiState(
            model.layout, moved_q, state.p_kg_m_per_s, "si_v1"
        )
        moved_result = apply_interaction_kick_tangent(
            plan, jacobi, model, provider, moved, tangent, ExactSeconds(1), context
        )
        self.assertFalse(
            np.array_equal(
                np.asarray(full_result.canonical_force_jvp_kg_m_per_s2),
                np.asarray(moved_result.canonical_force_jvp_kg_m_per_s2),
            )
        )

    def test_exact_projected_com_jvp_output(self) -> None:
        for kind in ("dense", "nonlinear"):
            varied = _apply(kind)[8]
            self.assertEqual(
                varied.canonical_force_jvp_kg_m_per_s2[0],
                (0.0, 0.0, 0.0),
            )
            diagnostics = varied.metadata.jvp_com_projection
            self.assertIsNotNone(diagnostics)
            self.assertTrue(diagnostics.projection_applied)
            self.assertLessEqual(
                diagnostics.raw_residual_norm_kg_m_per_s2,
                diagnostics.derived_bound_norm_kg_m_per_s2,
            )

    def test_com_only_tangent_has_no_internal_force_effect(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("nonlinear")
        tangent = CanonicalJacobiTangentState(
            model.layout,
            ((0.25, -0.5, 1.0), (0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.0),) * 4,
            "si_v1",
        )
        result = apply_interaction_kick_tangent(
            plan,
            jacobi,
            model,
            provider,
            state,
            tangent,
            ExactSeconds(7, 4),
            context,
        )
        self.assertEqual(
            result.canonical_force_jvp_kg_m_per_s2,
            ((0.0, 0.0, 0.0),) * 4,
        )
        self.assertEqual(result.tangent, tangent)

    def test_projected_force_jvp_finite_difference_closure(self) -> None:
        for kind in ("dense", "nonlinear"):
            metrics = finite_difference_series(kind)
            self.assertTrue(metrics["acceptance"]["force_jvp"])
            self.assertLessEqual(metrics["force_minimum"], FD_CAP)

    def test_projected_jacobian_symmetry(self) -> None:
        for kind in ("dense", "nonlinear"):
            matrix = runtime_tangent_matrix(kind, ExactSeconds(7, 4))
            projected_jacobian = matrix[12:, :12] / 1.75
            self.assertLessEqual(
                np.max(
                    np.abs(
                        projected_jacobian - projected_jacobian.T
                    )
                ),
                SYMMETRY_CAP,
            )
            self.assertTrue(
                np.array_equal(
                    projected_jacobian[0:3],
                    np.zeros((3, 12)),
                )
            )

    def test_projected_kick_symplecticity(self) -> None:
        for kind in ("dense", "nonlinear"):
            matrix = runtime_tangent_matrix(kind, ExactSeconds(7, 4))
            self.assertLessEqual(
                symplectic_residual(matrix)[0],
                SYMPLECTIC_CAP,
            )
            scale = np.diag(np.asarray([4.0] * 12 + [0.25] * 12))
            self.assertLessEqual(
                symplectic_residual(
                    np.linalg.inv(scale) @ matrix @ scale
                )[0],
                SYMPLECTIC_CAP,
            )

    def test_complete_kick_finite_difference_ladder(self) -> None:
        for kind in ("dense", "nonlinear"):
            metrics = finite_difference_series(kind)
            self.assertTrue(metrics["acceptance"]["kick_tangent"])
            self.assertLessEqual(metrics["kick_minimum"], FD_CAP)

    def test_force_and_jvp_finite_difference_closure(self) -> None:
        for kind in ("dense", "nonlinear"):
            metrics = finite_difference_series(kind)
            self.assertTrue(metrics["acceptance"]["force_jvp"])
            self.assertLessEqual(metrics["force_minimum"], FD_CAP)

    def test_raw_and_scaled_full_phase_symplecticity(self) -> None:
        for kind in ("dense", "nonlinear"):
            state = fixture(kind)[4]
            matrix = runtime_tangent_matrix(kind, ExactSeconds(7, 4))
            expected = analytic_tangent_matrix(kind, state.q_m, 1.75)
            self.assertLessEqual(np.max(np.abs(matrix - expected)), TANGENT_CAP)
            self.assertLessEqual(symplectic_residual(matrix)[0], SYMPLECTIC_CAP)
            scale = np.diag(np.asarray([4.0] * 12 + [0.25] * 12))
            scaled = np.linalg.inv(scale) @ matrix @ scale
            self.assertLessEqual(symplectic_residual(scaled)[0], SYMPLECTIC_CAP)
            _, jacobian, _ = canonical_force_and_jacobian(kind, state.q_m)
            self.assertLessEqual(
                np.max(np.abs(jacobian - jacobian.T)), SYMMETRY_CAP
            )

    def test_tangent_reversal_and_composition(self) -> None:
        for kind in ("dense", "nonlinear"):
            model, jacobi, provider, plan, state, tangent, context = fixture(kind)
            forward = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, state, tangent,
                ExactSeconds(7, 4), context
            )
            backward = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, forward.state, forward.tangent,
                ExactSeconds(-7, 4), context
            )
            self.assertLessEqual(
                scaled_error(
                    tangent_phase(backward.tangent), tangent_phase(tangent)
                )[0],
                REVERSIBILITY_CAP,
            )
            first = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, state, tangent,
                ExactSeconds(1, 2), context
            )
            second = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, first.state, first.tangent,
                ExactSeconds(5, 4), context
            )
            direct = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, state, tangent,
                ExactSeconds(7, 4), context
            )
            self.assertLessEqual(
                scaled_error(
                    tangent_phase(second.tangent), tangent_phase(direct.tangent)
                )[0],
                COMPOSITION_CAP,
            )


class NegativeControlTests(unittest.TestCase):
    def test_above_bound_com_residual_is_rejected(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("dense")
        with self.assertRaisesRegex(
            KernelContractError,
            "derived binary64 COM-force closure bound",
        ):
            apply_interaction_kick(
                plan,
                jacobi,
                model,
                ProviderProxy(provider, "above_bound"),
                state,
                ExactSeconds(1),
                context,
            )

    def test_projection_does_not_hide_nonconservative_or_nonclosing_provider(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("dense")
        with self.assertRaises(KernelContractError):
            apply_interaction_kick(
                plan,
                jacobi,
                model,
                ProviderProxy(provider, "above_bound"),
                state,
                ExactSeconds(1),
                context,
            )
        nonconservative = replace(
            provider.capabilities,
            provider_id="step3g1d_nonconservative_projection_control",
            provider_fingerprint="3" * 64,
            conservative_canonical_force=False,
            symmetric_canonical_jacobian=False,
        )
        with self.assertRaises(KernelContractError):
            build_interaction_kick_plan(model, jacobi, nonconservative)

    def test_nonsymmetric_control_is_detected_and_fails_symplecticity(self) -> None:
        model, jacobi, provider, _, state, *_ = fixture("dense")
        _, jacobian, _ = canonical_force_and_jacobian(
            "nonsymmetric", state.q_m
        )
        self.assertGreaterEqual(
            np.max(np.abs(jacobian - jacobian.T)), 0.1
        )
        matrix = np.eye(24)
        matrix[12:, :12] = jacobian
        self.assertGreaterEqual(symplectic_residual(matrix)[0], 1.0e-5)
        capabilities = replace(
            provider.capabilities,
            provider_id="step3g1d_nonsymmetric_control",
            provider_fingerprint="1" * 64,
            conservative_canonical_force=False,
            symmetric_canonical_jacobian=False,
        )
        with self.assertRaises(KernelContractError):
            build_interaction_kick_plan(model, jacobi, capabilities)

    def test_malformed_layout_fingerprint_shape_semantics_and_nonfinite_reject(self) -> None:
        model, jacobi, provider, plan, state, tangent, context = fixture("dense")
        bad_fingerprint = ProviderProxy(
            provider, "delegate", fingerprint="2" * 64
        )
        with self.assertRaises(KernelContractError):
            apply_interaction_kick(
                plan, jacobi, model, bad_fingerprint, state,
                ExactSeconds(1), context
            )
        for variant in (
            "wrong_type", "wrong_shape", "wrong_layout", "nonfinite"
        ):
            with self.subTest(variant=variant), self.assertRaises(
                (InvalidState, KernelContractError)
            ):
                apply_interaction_kick(
                    plan, jacobi, model, ProviderProxy(provider, variant),
                    state, ExactSeconds(1), context
                )
        for variant in ("wrong_jvp_type", "nonfinite_jvp"):
            with self.subTest(variant=variant), self.assertRaises(
                (InvalidState, KernelContractError)
            ):
                apply_interaction_kick_tangent(
                    plan, jacobi, model, ProviderProxy(provider, variant),
                    state, tangent, ExactSeconds(1), context
                )
        mislabeled = replace(
            provider.capabilities,
            raw_force_output="canonical_generalized_force",
        )
        with self.assertRaises(KernelContractError):
            build_interaction_kick_plan(model, jacobi, mislabeled)
        inconsistent = replace(
            provider.capabilities,
            conservative_canonical_force=True,
            symmetric_canonical_jacobian=False,
        )
        with self.assertRaises(KernelContractError):
            build_interaction_kick_plan(model, jacobi, inconsistent)


class OwnershipAccountingTests(unittest.TestCase):
    def test_inputs_outputs_are_immutable_detached_and_nonmutating(self) -> None:
        values = _apply("dense")
        state, tangent, physical, varied = (
            values[4], values[5], values[7], values[8]
        )
        original_state = phase(state).copy()
        original_tangent = tangent_phase(tangent).copy()
        self.assertIsNot(physical.state, state)
        self.assertIsNot(varied.tangent, tangent)
        self.assertTrue(np.array_equal(phase(state), original_state))
        self.assertTrue(np.array_equal(tangent_phase(tangent), original_tangent))
        with self.assertRaises(FrozenInstanceError):
            values[3].body_count = 7
        with self.assertRaises(FrozenInstanceError):
            physical.metadata.force_evaluations = 2
        with self.assertRaises(FrozenInstanceError):
            physical.metadata.force_com_projection.projection_applied = False

    def test_exact_call_order_counts_and_disjoint_events(self) -> None:
        model, jacobi, provider, plan, state, tangent, context = fixture("dense")
        with mock.patch(
            "mini_ephemeris.v2.kick.evaluate_force", wraps=evaluate_force
        ) as force_call, mock.patch(
            "mini_ephemeris.v2.kick.evaluate_jvp", wraps=evaluate_jvp
        ) as jvp_call:
            physical = apply_interaction_kick(
                plan, jacobi, model, provider, state, ExactSeconds(1), context
            )
            self.assertEqual((force_call.call_count, jvp_call.call_count), (1, 0))
            varied = apply_interaction_kick_tangent(
                plan, jacobi, model, provider, state, tangent,
                ExactSeconds(1), context
            )
            self.assertEqual((force_call.call_count, jvp_call.call_count), (2, 1))
        self.assertEqual(
            [event.operation for event in physical.metadata.events], ["force"]
        )
        self.assertEqual(
            [event.operation for event in varied.metadata.events],
            ["force", "jvp"],
        )
        self.assertTrue(
            all(
                event.domain.value == "map_stage"
                for event in varied.metadata.events
            )
        )
        self.assertEqual(varied.metadata.observer_evaluations, 0)
        self.assertEqual(varied.metadata.synchronization_evaluations, 0)

    def test_failures_return_no_partial_result_and_do_not_mutate(self) -> None:
        model, jacobi, provider, plan, state, _, context = fixture("dense")
        before = phase(state).copy()
        with self.assertRaises(KernelContractError):
            apply_interaction_kick(
                plan, jacobi, model, ProviderProxy(provider, "wrong_type"),
                state, ExactSeconds(1), context
            )
        self.assertTrue(np.array_equal(phase(state), before))

    def test_source_is_isolated_from_forbidden_operations(self) -> None:
        path = ROOT / "mini_ephemeris/src/mini_ephemeris/v2/kick.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any(
                name.startswith(("rebound", "mini_ephemeris.gr_"))
                for name in imports
            )
        )
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(
            {"integrate", "synchronize", "observe", "save", "archive"}
            .isdisjoint(called)
        )


if __name__ == "__main__":
    unittest.main()
