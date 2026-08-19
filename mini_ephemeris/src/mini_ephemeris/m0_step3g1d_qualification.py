"""Synthetic fixtures, metrics, and guarded helpers for Step 3g1d."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v2.accounting import AccountingDomain
from .v2.canonical import canonical_json_bytes, finite_binary64_hex, sha256_hex
from .v2.jacobi import JacobiTransformPlan, build_jacobi_transform_plan
from .v2.kernels import ForceEvaluationContext
from .v2.kick import (
    InteractionProviderCapabilities,
    apply_interaction_kick,
    apply_interaction_kick_tangent,
    build_interaction_kick_plan,
)
from .v2.model import CompiledLayout, PhysicalModel, SI_UNITS
from .v2.state import (
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    CartesianAcceleration,
    CartesianAccelerationJVP,
)
from .v2.timebase import ControlTime, ExactSeconds


ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "mini_ephemeris/src/mini_ephemeris"
MANIFEST26 = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "26_m0_step3g1d_interaction_kick_tangent_primitive_v1.json"
)
MANIFEST27 = ROOT / (
    "ephemeris_experiment_runner/manifests/"
    "27_m0_step3g1d_interaction_kick_corrective_completion_v1.json"
)
PREREGISTRATION_COMMIT = "068695ace38a68ff83238668b5630867f8572a9b"
PYTEST_SITE_PACKAGES = Path(
    "/home/peacelovephysics/sheet-music-generator/.venv/lib/python3.10/site-packages"
)
PYTEST_VERSION = "8.4.2"
PYTEST_IDENTITY_SHA256 = {
    "pytest/__init__.py": (
        "66993a5e3905005e0981159b4794d10b1adacf341a58a44d696ad2c4442dcdc6"
    ),
    "pytest-8.4.2.dist-info/METADATA": (
        "93d6c5ef0a9714d53716243035037b77fa7d5f970596c48433887cf57f7f675a"
    ),
    "pytest-8.4.2.dist-info/RECORD": (
        "fba86b3aa6d34c9d73bc8b2ea69d6d69a65cd555c4ac84ef15a330232706f82f"
    ),
}
EPSILON_EXPONENTS = (-4, -8, -12, -16, -20, -24, -28, -32, -36, -40)
PHYSICAL_CAP = 2.0e-12
TANGENT_CAP = 4.0e-12
FD_CAP = 2.0e-7
SYMMETRY_CAP = 5.0e-12
SYMPLECTIC_CAP = 5.0e-12
REVERSIBILITY_CAP = 4.0e-12
COMPOSITION_CAP = 4.0e-12
AFFINE_EXACT = "AFFINE_EXACT"
NONLINEAR_SMOOTH = "NONLINEAR_SMOOTH"
FD_GATE_SCHEMA = "step3g1d.finite_difference_gate/2"
BINARY64_UNIT_ROUNDOFF = 2.0**-53
# A scalar path uses at most 98 rounded operations: inverse Jacobi (15),
# three dense dot products (23, 17, 23), and adapter/kick/difference
# arithmetic (20). The next power of two is the conservative gamma_k envelope.
AFFINE_ROUNDOFF_OPERATION_COUNT = 128
AFFINE_ROUNDOFF_MODEL = (
    "gamma_128*max(1,evaluation_scale_ratio)/epsilon; "
    "128 is the next power-of-two envelope above 98 scalar operations"
)


@dataclass(frozen=True)
class FiniteDifferenceGateSpec:
    """Immutable method selection derived from the fixture definition."""

    fixture_kind: str
    derivative_class: str
    epsilon_exponents_base2: tuple[int, ...]
    absolute_cap: float
    oracle_cap: float
    minimum_early_improvements: int | None
    require_roundoff_turn: bool
    affine_roundoff_operation_count: int | None
    unit_roundoff: float = BINARY64_UNIT_ROUNDOFF
    schema: str = FD_GATE_SCHEMA

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "absolute_cap_hex": finite_binary64_hex(
                self.absolute_cap, "finite-difference cap"
            ),
            "affine_roundoff_operation_count": (
                self.affine_roundoff_operation_count
            ),
            "affine_roundoff_operation_model": (
                AFFINE_ROUNDOFF_MODEL
                if self.affine_roundoff_operation_count is not None
                else None
            ),
            "derivative_class": self.derivative_class,
            "epsilon_exponents_base2": list(
                self.epsilon_exponents_base2
            ),
            "fixture_kind": self.fixture_kind,
            "minimum_early_improvements": (
                self.minimum_early_improvements
            ),
            "oracle_cap_hex": finite_binary64_hex(
                self.oracle_cap, "finite-difference oracle cap"
            ),
            "require_roundoff_turn": self.require_roundoff_turn,
            "schema": self.schema,
            "unit_roundoff_hex": finite_binary64_hex(
                self.unit_roundoff, "binary64 unit roundoff"
            ),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        return sha256_hex(self.canonical_bytes())


def finite_difference_gate_spec(kind: str) -> FiniteDifferenceGateSpec:
    """Classify a fixture analytically, before any ladder is evaluated."""

    if kind == "dense":
        return FiniteDifferenceGateSpec(
            fixture_kind=kind,
            derivative_class=AFFINE_EXACT,
            epsilon_exponents_base2=EPSILON_EXPONENTS,
            absolute_cap=FD_CAP,
            oracle_cap=TANGENT_CAP,
            minimum_early_improvements=None,
            require_roundoff_turn=False,
            affine_roundoff_operation_count=(
                AFFINE_ROUNDOFF_OPERATION_COUNT
            ),
        )
    if kind == "nonlinear":
        return FiniteDifferenceGateSpec(
            fixture_kind=kind,
            derivative_class=NONLINEAR_SMOOTH,
            epsilon_exponents_base2=EPSILON_EXPONENTS,
            absolute_cap=FD_CAP,
            oracle_cap=TANGENT_CAP,
            minimum_early_improvements=3,
            require_roundoff_turn=True,
            affine_roundoff_operation_count=None,
        )
    raise ValueError(f"unknown finite-difference fixture {kind!r}")


def _require_frozen_gate_spec(spec: FiniteDifferenceGateSpec) -> None:
    if (
        not isinstance(spec, FiniteDifferenceGateSpec)
        or spec != finite_difference_gate_spec(spec.fixture_kind)
    ):
        raise ValueError(
            "finite-difference derivative classification changed"
        )


def assess_finite_difference_ladder(
    spec: FiniteDifferenceGateSpec,
    values: Sequence[float],
    evaluation_scale_ratios: Sequence[float],
    oracle_error: float,
) -> Mapping[str, Any]:
    """Apply the frozen derivative-class-specific ladder method."""

    _require_frozen_gate_spec(spec)
    if (
        len(values) != len(spec.epsilon_exponents_base2)
        or len(evaluation_scale_ratios) != len(values)
        or not values
    ):
        raise ValueError("finite-difference ladder shape changed")
    ladder_finite = all(math.isfinite(value) for value in values)
    diagnostic_inputs_finite = all(
        math.isfinite(value)
        for value in (*evaluation_scale_ratios, oracle_error)
    )
    if any(value < 0.0 for value in values) or any(
        value < 0.0 for value in evaluation_scale_ratios
    ):
        raise ValueError("finite-difference diagnostics must be nonnegative")
    minimum = min(values)
    minimum_index = int(np.argmin(values))
    early_improvements = sum(
        values[index + 1] < values[index]
        for index in range(min(4, len(values) - 1))
    )
    oracle_pass = math.isfinite(oracle_error) and (
        oracle_error <= spec.oracle_cap
    )

    roundoff_bounds: list[float] = []
    roundoff_consistent: bool | None = None
    if spec.derivative_class == AFFINE_EXACT:
        operation_count = spec.affine_roundoff_operation_count
        if operation_count is None:
            raise ValueError("affine roundoff operation count is missing")
        denominator = 1.0 - operation_count * spec.unit_roundoff
        gamma = operation_count * spec.unit_roundoff / denominator
        for exponent, scale_ratio in zip(
            spec.epsilon_exponents_base2, evaluation_scale_ratios
        ):
            epsilon = 2.0**exponent
            roundoff_bounds.append(
                gamma * max(1.0, scale_ratio) / epsilon
            )
        roundoff_consistent = (
            ladder_finite
            and diagnostic_inputs_finite
            and all(
                value <= bound
                for value, bound in zip(values, roundoff_bounds)
            )
        )
        acceptance = (
            ladder_finite
            and oracle_pass
            and values[0] <= spec.absolute_cap
            and minimum <= spec.absolute_cap
            and roundoff_consistent
        )
    elif spec.derivative_class == NONLINEAR_SMOOTH:
        acceptance = (
            ladder_finite
            and minimum <= spec.absolute_cap
            and early_improvements
            >= int(spec.minimum_early_improvements or 0)
            and minimum_index < len(values) - 1
            and values[-1] > minimum
        )
    else:
        raise ValueError("unknown finite-difference derivative class")

    return {
        "acceptance": acceptance,
        "derivative_class": spec.derivative_class,
        "diagnostic_inputs_finite": diagnostic_inputs_finite,
        "early_improvements": early_improvements,
        "finite": ladder_finite,
        "largest_epsilon_error": values[0],
        "minimum": minimum,
        "minimum_index": minimum_index,
        "oracle_error": oracle_error,
        "oracle_pass": oracle_pass,
        "roundoff_model": {
            "applied": spec.derivative_class == AFFINE_EXACT,
            "bounds": roundoff_bounds,
            "consistent": roundoff_consistent,
            "formula": (
                "gamma_k*max(1,evaluation_scale_ratio)/epsilon"
                if spec.derivative_class == AFFINE_EXACT
                else "NONLINEAR_SMOOTH convergence then roundoff turn"
            ),
            "operation_count": spec.affine_roundoff_operation_count,
        },
    }


K_DENSE = np.array(
    [
        [1.0, 0.125, 0.0, 0.0625, 0.0, -0.03125, 0.0, 0.015625, 0.0],
        [0.125, 1.1, -0.09375, 0.0, 0.046875, 0.0, -0.0234375, 0.0, 0.01171875],
        [0.0, -0.09375, 1.2, 0.078125, 0.0, 0.0390625, 0.0, -0.01953125, 0.0],
        [0.0625, 0.0, 0.078125, 1.3, -0.0546875, 0.0, 0.02734375, 0.0, -0.013671875],
        [0.0, 0.046875, 0.0, -0.0546875, 1.4, 0.109375, 0.0, 0.025, 0.0],
        [-0.03125, 0.0, 0.0390625, 0.0, 0.109375, 1.5, -0.0703125, 0.0, 0.03515625],
        [0.0, -0.0234375, 0.0, 0.02734375, 0.0, -0.0703125, 1.6, 0.0859375, 0.0],
        [0.015625, 0.0, -0.01953125, 0.0, 0.025, 0.0, 0.0859375, 1.7, -0.1015625],
        [0.0, 0.01171875, 0.0, -0.013671875, 0.0, 0.03515625, 0.0, -0.1015625, 1.8],
    ],
    dtype=np.float64,
)
NONLINEAR_ALPHA = 0.03125
NONLINEAR_BETA = 0.0078125


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def manifest26() -> Mapping[str, Any]:
    value = strict_json(MANIFEST26)
    if not isinstance(value, dict):
        raise TypeError("Manifest 26 must be a JSON object")
    return value


def manifest27() -> Mapping[str, Any]:
    value = strict_json(MANIFEST27)
    if not isinstance(value, dict):
        raise TypeError("Manifest 27 must be a JSON object")
    return value


def synthetic_model() -> PhysicalModel:
    layout = CompiledLayout(("center", "body_1", "body_2", "body_3"), "center")
    return PhysicalModel(
        model_id="step3g1d_synthetic_interaction",
        schema_version="1",
        layout=layout,
        masses_kg={
            "center": 5.0,
            "body_1": 2.0,
            "body_2": 3.0,
            "body_3": 7.0,
        },
        gravitational_constant_si=1.0,
        units=SI_UNITS,
        enabled_effects=("synthetic-position-only-interaction",),
        provenance={"fixture": "manifest-26"},
    )


def synthetic_state(model: PhysicalModel) -> CanonicalJacobiState:
    return CanonicalJacobiState(
        model.layout,
        (
            (0.125, -0.25, 0.5),
            (1.0, -0.75, 0.25),
            (-0.5, 1.25, -1.0),
            (0.75, 0.5, -0.625),
        ),
        (
            (0.0, 0.0, 0.0),
            (0.375, -0.625, 0.875),
            (-1.0, 0.5, 0.25),
            (0.75, 1.125, -0.375),
        ),
        "si_v1",
    )


def synthetic_tangent(model: PhysicalModel) -> CanonicalJacobiTangentState:
    return CanonicalJacobiTangentState(
        model.layout,
        (
            (0.25, -0.125, 0.0625),
            (-0.375, 0.5, 0.125),
            (0.75, -0.25, 0.625),
            (-0.5, -0.875, 0.375),
        ),
        (
            (0.125, 0.25, -0.5),
            (-0.75, 0.375, 0.625),
            (0.5, -0.125, -0.25),
            (0.875, 0.75, -0.375),
        ),
        "si_v1",
    )


def jacobi_matrix(plan: JacobiTransformPlan) -> np.ndarray:
    """Return the independent dense q=A*x oracle in body-major order."""

    count = len(plan.masses_kg)
    body = np.zeros((count, count), dtype=np.float64)
    body[0, :] = np.asarray(plan.masses_kg) / plan.cumulative_masses_kg[-1]
    for index in range(1, count):
        body[index, :index] = (
            -np.asarray(plan.masses_kg[:index])
            / plan.cumulative_masses_kg[index - 1]
        )
        body[index, index] = 1.0
    return np.kron(body, np.eye(3, dtype=np.float64))


def phase(state: CanonicalJacobiState) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(state.q_m).reshape(-1),
            np.asarray(state.p_kg_m_per_s).reshape(-1),
        )
    )


def tangent_phase(tangent: CanonicalJacobiTangentState) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(tangent.delta_q_m).reshape(-1),
            np.asarray(tangent.delta_p_kg_m_per_s).reshape(-1),
        )
    )


def canonical_force_and_jacobian(
    kind: str, q_rows: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray, float]:
    q = np.asarray(q_rows, dtype=np.float64).reshape(-1)
    u = q[3:]
    force = np.zeros(12, dtype=np.float64)
    jacobian = np.zeros((12, 12), dtype=np.float64)
    if kind == "dense":
        force[3:] = -(K_DENSE @ u)
        jacobian[3:, 3:] = -K_DENSE
        potential = 0.5 * float(u @ K_DENSE @ u)
    elif kind == "nonlinear":
        radius2 = float(u @ u)
        coefficient = NONLINEAR_ALPHA + NONLINEAR_BETA * radius2
        force[3:] = -coefficient * u
        jacobian[3:, 3:] = (
            -coefficient * np.eye(9)
            - 2.0 * NONLINEAR_BETA * np.outer(u, u)
        )
        potential = (
            0.5 * NONLINEAR_ALPHA * radius2
            + 0.25 * NONLINEAR_BETA * radius2 * radius2
        )
    elif kind == "nonsymmetric":
        nonsymmetric = K_DENSE.copy()
        nonsymmetric[0, 4] += 0.25
        nonsymmetric[4, 0] -= 0.25
        force[3:] = -(nonsymmetric @ u)
        jacobian[3:, 3:] = -nonsymmetric
        potential = math.nan
    else:
        raise ValueError(f"unknown synthetic provider kind {kind!r}")
    return force, jacobian, potential


def _provider_identity(model: PhysicalModel, kind: str) -> str:
    payload = {
        "alpha_hex": finite_binary64_hex(NONLINEAR_ALPHA, "alpha"),
        "beta_hex": finite_binary64_hex(NONLINEAR_BETA, "beta"),
        "k_hex": [
            [finite_binary64_hex(value, "K") for value in row]
            for row in K_DENSE
        ],
        "kind": kind,
        "model_fingerprint": model.fingerprint,
        "schema": "step3g1d.synthetic_provider/1",
    }
    return sha256_hex(canonical_json_bytes(payload))


@dataclass(frozen=True, init=False)
class SyntheticInteractionProvider:
    """Test-only inertial acceleration provider derived from canonical U(q)."""

    kind: str
    matrix_A: tuple[tuple[float, ...], ...]
    masses_kg: tuple[float, ...]
    capabilities: InteractionProviderCapabilities
    provider_fingerprint: str

    def __init__(
        self, model: PhysicalModel, plan: JacobiTransformPlan, kind: str
    ) -> None:
        if kind not in {"dense", "nonlinear"}:
            raise ValueError("qualified provider kind must be dense or nonlinear")
        provider_fingerprint = _provider_identity(model, kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "matrix_A",
            tuple(
                tuple(float(value) for value in row)
                for row in jacobi_matrix(plan)
            ),
        )
        object.__setattr__(self, "masses_kg", tuple(plan.masses_kg))
        object.__setattr__(self, "provider_fingerprint", provider_fingerprint)
        object.__setattr__(
            self,
            "capabilities",
            InteractionProviderCapabilities(
                provider_id=f"step3g1d_{kind}_synthetic",
                provider_fingerprint=provider_fingerprint,
                model_fingerprint=model.fingerprint,
                layout_fingerprint=model.layout.fingerprint,
            ),
        )

    def _acceleration(
        self,
        positions: Sequence[Sequence[float]],
        direction: Sequence[Sequence[float]] | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(self.matrix_A)
        x = np.asarray(positions, dtype=np.float64).reshape(-1)
        q = matrix @ x
        force, jacobian, _ = canonical_force_and_jacobian(
            self.kind, q.reshape((-1, 3))
        )
        canonical_value = force
        if direction is not None:
            delta_x = np.asarray(direction, dtype=np.float64).reshape(-1)
            canonical_value = jacobian @ (matrix @ delta_x)
        inertial_force = matrix.T @ canonical_value
        return (
            inertial_force.reshape((-1, 3))
            / np.asarray(self.masses_kg)[:, None]
        )

    def evaluate(self, model, state, context):
        del context
        return CartesianAcceleration(
            model.layout, self._acceleration(state.positions_m), "si_v1"
        )

    def jvp(self, model, state, direction, context):
        del context
        return CartesianAccelerationJVP(
            model.layout,
            self._acceleration(
                state.positions_m, direction.delta_positions_m
            ),
            "si_v1",
        )


def fixture(kind: str = "dense"):
    model = synthetic_model()
    jacobi_plan = build_jacobi_transform_plan(model)
    provider = SyntheticInteractionProvider(model, jacobi_plan, kind)
    kick_plan = build_interaction_kick_plan(
        model, jacobi_plan, provider.capabilities
    )
    state = synthetic_state(model)
    tangent = synthetic_tangent(model)
    context = ForceEvaluationContext(
        ControlTime(17, ExactSeconds(17, 8)),
        AccountingDomain.MAP_STAGE,
        f"step3g1d-{kind}",
    )
    return (
        model,
        jacobi_plan,
        provider,
        kick_plan,
        state,
        tangent,
        context,
    )


def expected_physical(
    state: CanonicalJacobiState, kind: str, seconds: float
) -> CanonicalJacobiState:
    force, _, _ = canonical_force_and_jacobian(kind, state.q_m)
    expected = phase(state).copy()
    expected[12:] += seconds * force
    return CanonicalJacobiState(
        state.layout,
        expected[:12].reshape((-1, 3)),
        expected[12:].reshape((-1, 3)),
        "si_v1",
    )


def expected_tangent(
    state: CanonicalJacobiState,
    tangent: CanonicalJacobiTangentState,
    kind: str,
    seconds: float,
) -> CanonicalJacobiTangentState:
    _, jacobian, _ = canonical_force_and_jacobian(kind, state.q_m)
    expected = tangent_phase(tangent).copy()
    expected[12:] += seconds * (jacobian @ expected[:12])
    return CanonicalJacobiTangentState(
        tangent.layout,
        expected[:12].reshape((-1, 3)),
        expected[12:].reshape((-1, 3)),
        "si_v1",
    )


def scaled_error(
    observed: np.ndarray, expected: np.ndarray
) -> tuple[float, float]:
    difference = (observed - expected) / (1.0 + np.abs(expected))
    return float(np.max(np.abs(difference))), float(np.linalg.norm(difference))


def symplectic_residual(matrix: np.ndarray) -> tuple[float, float]:
    identity = np.eye(matrix.shape[0] // 2)
    zeros = np.zeros_like(identity)
    omega = np.block([[zeros, identity], [-identity, zeros]])
    residual = matrix.T @ omega @ matrix - omega
    return float(np.max(np.abs(residual))), float(np.linalg.norm(residual))


def analytic_tangent_matrix(
    kind: str, q_rows: Sequence[Sequence[float]], seconds: float
) -> np.ndarray:
    _, jacobian, _ = canonical_force_and_jacobian(kind, q_rows)
    matrix = np.eye(24)
    matrix[12:, :12] = seconds * jacobian
    return matrix


def runtime_tangent_matrix(kind: str, duration: ExactSeconds) -> np.ndarray:
    (
        model,
        jacobi_plan,
        provider,
        kick_plan,
        state,
        _,
        context,
    ) = fixture(kind)
    columns = []
    for index in range(24):
        direction = np.zeros(24)
        direction[index] = 1.0
        tangent = CanonicalJacobiTangentState(
            model.layout,
            direction[:12].reshape((-1, 3)),
            direction[12:].reshape((-1, 3)),
            "si_v1",
        )
        result = apply_interaction_kick_tangent(
            kick_plan,
            jacobi_plan,
            model,
            provider,
            state,
            tangent,
            duration,
            context,
        )
        columns.append(tangent_phase(result.tangent))
    return np.column_stack(columns)


def finite_difference_series(kind: str) -> Mapping[str, Any]:
    gate_spec = finite_difference_gate_spec(kind)
    (
        model,
        jacobi_plan,
        provider,
        kick_plan,
        state,
        tangent,
        context,
    ) = fixture(kind)
    if provider.kind != gate_spec.fixture_kind:
        raise ValueError("fixture derivative classification mismatch")
    base = phase(state)
    direction = tangent_phase(tangent)
    duration = ExactSeconds(7, 4)
    analytic = apply_interaction_kick_tangent(
        kick_plan,
        jacobi_plan,
        model,
        provider,
        state,
        tangent,
        duration,
        context,
    )
    analytic_kick = tangent_phase(analytic.tangent)
    analytic_force = np.asarray(
        analytic.canonical_force_jvp_kg_m_per_s2
    ).reshape(-1)
    _, expected_jacobian, _ = canonical_force_and_jacobian(
        kind, state.q_m
    )
    expected_force_jvp = expected_jacobian @ direction[:12]
    force_oracle_error = float(
        np.linalg.norm(analytic_force - expected_force_jvp)
        / max(np.linalg.norm(expected_force_jvp), 1.0)
    )
    expected_kick = expected_tangent(
        state, tangent, kind, duration.to_binary64()
    )
    kick_oracle_error = scaled_error(
        analytic_kick, tangent_phase(expected_kick)
    )[0]

    kick_errors = []
    force_errors = []
    kick_scale_ratios = []
    force_scale_ratios = []
    rows = []
    for exponent in gate_spec.epsilon_exponents_base2:
        epsilon = 2.0**exponent
        plus = base + epsilon * direction
        minus = base - epsilon * direction
        plus_state = CanonicalJacobiState(
            model.layout,
            plus[:12].reshape((-1, 3)),
            plus[12:].reshape((-1, 3)),
            "si_v1",
        )
        minus_state = CanonicalJacobiState(
            model.layout,
            minus[:12].reshape((-1, 3)),
            minus[12:].reshape((-1, 3)),
            "si_v1",
        )
        plus_result = apply_interaction_kick(
            kick_plan,
            jacobi_plan,
            model,
            provider,
            plus_state,
            duration,
            context,
        )
        minus_result = apply_interaction_kick(
            kick_plan,
            jacobi_plan,
            model,
            provider,
            minus_state,
            duration,
            context,
        )
        plus_phase = phase(plus_result.state)
        minus_phase = phase(minus_result.state)
        plus_force = np.asarray(
            plus_result.canonical_force_kg_m_per_s2
        ).reshape(-1)
        minus_force = np.asarray(
            minus_result.canonical_force_kg_m_per_s2
        ).reshape(-1)
        numerical_kick = (
            plus_phase - minus_phase
        ) / (2.0 * epsilon)
        numerical_force = (
            plus_force - minus_force
        ) / (2.0 * epsilon)
        kick_denominator = max(np.linalg.norm(analytic_kick), 1.0)
        force_denominator = max(np.linalg.norm(analytic_force), 1.0)
        kick_error = float(
            np.linalg.norm(numerical_kick - analytic_kick)
            / kick_denominator
        )
        force_error = float(
            np.linalg.norm(numerical_force - analytic_force)
            / force_denominator
        )
        kick_scale_ratio = float(
            (np.linalg.norm(plus_phase) + np.linalg.norm(minus_phase))
            / (2.0 * kick_denominator)
        )
        force_scale_ratio = float(
            (np.linalg.norm(plus_force) + np.linalg.norm(minus_force))
            / (2.0 * force_denominator)
        )
        kick_errors.append(kick_error)
        force_errors.append(force_error)
        kick_scale_ratios.append(kick_scale_ratio)
        force_scale_ratios.append(force_scale_ratio)
        rows.append(
            {
                "epsilon": epsilon,
                "exponent_base2": exponent,
                "force_evaluation_scale_ratio": force_scale_ratio,
                "force_jvp_relative_l2": force_error,
                "kick_evaluation_scale_ratio": kick_scale_ratio,
                "kick_relative_l2": kick_error,
            }
        )

    _require_frozen_gate_spec(gate_spec)
    if provider.kind != gate_spec.fixture_kind:
        raise ValueError("fixture derivative class changed after evaluation")
    force_gate = assess_finite_difference_ladder(
        gate_spec,
        force_errors,
        force_scale_ratios,
        force_oracle_error,
    )
    kick_gate = assess_finite_difference_ladder(
        gate_spec,
        kick_errors,
        kick_scale_ratios,
        kick_oracle_error,
    )
    for index, row in enumerate(rows):
        force_bounds = force_gate["roundoff_model"]["bounds"]
        kick_bounds = kick_gate["roundoff_model"]["bounds"]
        row["force_affine_roundoff_bound"] = (
            force_bounds[index] if force_bounds else None
        )
        row["kick_affine_roundoff_bound"] = (
            kick_bounds[index] if kick_bounds else None
        )
    return {
        "acceptance": {
            "force_jvp": force_gate["acceptance"],
            "kick_tangent": kick_gate["acceptance"],
        },
        "derivative_class": gate_spec.derivative_class,
        "force_gate": force_gate,
        "force_minimum": force_gate["minimum"],
        "force_minimum_index": force_gate["minimum_index"],
        "gate_spec": gate_spec.canonical_payload(),
        "gate_spec_fingerprint": gate_spec.fingerprint,
        "kick_gate": kick_gate,
        "kick_minimum": kick_gate["minimum"],
        "kick_minimum_index": kick_gate["minimum_index"],
        "kind": kind,
        "rows": rows,
    }


def _accounting_payload(metadata) -> Mapping[str, Any]:
    return {
        "events": [event.operation for event in metadata.events],
        "force": metadata.force_evaluations,
        "jvp": metadata.jvp_evaluations,
        "observer": metadata.observer_evaluations,
        "synchronization": metadata.synchronization_evaluations,
    }


def _projection_payload(diagnostics) -> Mapping[str, Any]:
    component_ratios = []
    for residual, bound in zip(
        diagnostics.raw_residual_kg_m_per_s2,
        diagnostics.component_bounds_kg_m_per_s2,
    ):
        component_ratios.append(
            abs(residual) / bound
            if bound > 0.0
            else (0.0 if residual == 0.0 else math.inf)
        )
    norm_ratio = (
        diagnostics.raw_residual_norm_kg_m_per_s2
        / diagnostics.derived_bound_norm_kg_m_per_s2
        if diagnostics.derived_bound_norm_kg_m_per_s2 > 0.0
        else (
            0.0
            if diagnostics.raw_residual_norm_kg_m_per_s2 == 0.0
            else math.inf
        )
    )
    return {
        "accumulated_force_terms": diagnostics.accumulated_force_terms,
        "component_bounds_kg_m_per_s2": list(
            diagnostics.component_bounds_kg_m_per_s2
        ),
        "component_ratio_max": max(component_ratios),
        "componentwise_absolute_force_sums_kg_m_per_s2": list(
            diagnostics.componentwise_absolute_force_sums_kg_m_per_s2
        ),
        "derived_bound_norm_kg_m_per_s2": (
            diagnostics.derived_bound_norm_kg_m_per_s2
        ),
        "gamma": diagnostics.gamma,
        "norm_ratio": norm_ratio,
        "projection_applied": diagnostics.projection_applied,
        "raw_residual_kg_m_per_s2": list(
            diagnostics.raw_residual_kg_m_per_s2
        ),
        "raw_residual_norm_kg_m_per_s2": (
            diagnostics.raw_residual_norm_kg_m_per_s2
        ),
        "rounded_operation_count": diagnostics.rounded_operation_count,
        "transform_condition_inf": diagnostics.transform_condition_inf,
        "within_bound": (
            max(component_ratios) <= 1.0 and norm_ratio <= 1.0
        ),
    }


def compute_metrics() -> Mapping[str, Any]:
    physical_cases = []
    tangent_cases = []
    accounting_cases = []
    projection_cases = []
    symplectic_cases = []
    reversibility_cases = []
    composition_cases = []
    durations = (
        ExactSeconds(1),
        ExactSeconds(-1),
        ExactSeconds(1, 1024),
        ExactSeconds(-1, 1024),
        ExactSeconds(7, 4),
        ExactSeconds(-7, 4),
    )
    for kind in ("dense", "nonlinear"):
        (
            model,
            jacobi_plan,
            provider,
            kick_plan,
            state,
            tangent,
            context,
        ) = fixture(kind)
        for duration in durations:
            seconds = duration.to_binary64()
            physical = apply_interaction_kick(
                kick_plan,
                jacobi_plan,
                model,
                provider,
                state,
                duration,
                context,
            )
            varied = apply_interaction_kick_tangent(
                kick_plan,
                jacobi_plan,
                model,
                provider,
                state,
                tangent,
                duration,
                context,
            )
            physical_error = scaled_error(
                phase(physical.state),
                phase(expected_physical(state, kind, seconds)),
            )
            tangent_error = scaled_error(
                tangent_phase(varied.tangent),
                tangent_phase(expected_tangent(state, tangent, kind, seconds)),
            )
            physical_cases.append(
                {
                    "duration_s": seconds,
                    "kind": kind,
                    "max_scaled": physical_error[0],
                    "relative_l2": physical_error[1],
                }
            )
            tangent_cases.append(
                {
                    "duration_s": seconds,
                    "kind": kind,
                    "max_scaled": tangent_error[0],
                    "relative_l2": tangent_error[1],
                }
            )
            accounting_cases.append(
                {
                    "duration_s": seconds,
                    "kind": kind,
                    "physical": _accounting_payload(physical.metadata),
                    "tangent": _accounting_payload(varied.metadata),
                }
            )
            for channel, diagnostics, force_rows in (
                (
                    "physical_force",
                    physical.metadata.force_com_projection,
                    physical.canonical_force_kg_m_per_s2,
                ),
                (
                    "tangent_force",
                    varied.metadata.force_com_projection,
                    varied.canonical_force_kg_m_per_s2,
                ),
                (
                    "tangent_jvp",
                    varied.metadata.jvp_com_projection,
                    varied.canonical_force_jvp_kg_m_per_s2,
                ),
            ):
                if diagnostics is None:
                    raise AssertionError("nonzero kick omitted projection evidence")
                payload = dict(_projection_payload(diagnostics))
                payload.update(
                    {
                        "channel": channel,
                        "duration_s": seconds,
                        "exact_zero_com_output": (
                            force_rows[0] == (0.0, 0.0, 0.0)
                        ),
                        "kind": kind,
                    }
                )
                projection_cases.append(payload)
        matrix = runtime_tangent_matrix(kind, ExactSeconds(7, 4))
        raw = symplectic_residual(matrix)
        scale = np.diag(np.asarray([4.0] * 12 + [0.25] * 12))
        scaled = symplectic_residual(np.linalg.inv(scale) @ matrix @ scale)
        _, jacobian, _ = canonical_force_and_jacobian(kind, state.q_m)
        analytic = analytic_tangent_matrix(kind, state.q_m, 1.75)
        symplectic_cases.append(
            {
                "analytic_matrix_max_error": float(
                    np.max(np.abs(matrix - analytic))
                ),
                "jacobian_symmetry_max": float(
                    np.max(np.abs(jacobian - jacobian.T))
                ),
                "kind": kind,
                "raw_frobenius": raw[1],
                "raw_max": raw[0],
                "scaled_frobenius": scaled[1],
                "scaled_max": scaled[0],
            }
        )
        forward = apply_interaction_kick_tangent(
            kick_plan,
            jacobi_plan,
            model,
            provider,
            state,
            tangent,
            ExactSeconds(7, 4),
            context,
        )
        backward = apply_interaction_kick_tangent(
            kick_plan,
            jacobi_plan,
            model,
            provider,
            forward.state,
            forward.tangent,
            ExactSeconds(-7, 4),
            context,
        )
        reversibility_cases.append(
            {
                "kind": kind,
                "physical_max_scaled": scaled_error(
                    phase(backward.state), phase(state)
                )[0],
                "tangent_max_scaled": scaled_error(
                    tangent_phase(backward.tangent), tangent_phase(tangent)
                )[0],
            }
        )
        first = apply_interaction_kick_tangent(
            kick_plan, jacobi_plan, model, provider, state, tangent,
            ExactSeconds(1, 2), context
        )
        second = apply_interaction_kick_tangent(
            kick_plan, jacobi_plan, model, provider, first.state, first.tangent,
            ExactSeconds(5, 4), context
        )
        direct = apply_interaction_kick_tangent(
            kick_plan, jacobi_plan, model, provider, state, tangent,
            ExactSeconds(7, 4), context
        )
        composition_cases.append(
            {
                "kind": kind,
                "physical_max_scaled": scaled_error(
                    phase(second.state), phase(direct.state)
                )[0],
                "tangent_max_scaled": scaled_error(
                    tangent_phase(second.tangent), tangent_phase(direct.tangent)
                )[0],
            }
        )

    zero_fixture = fixture("dense")
    zero_physical = apply_interaction_kick(
        zero_fixture[3], zero_fixture[1], zero_fixture[0], zero_fixture[2],
        zero_fixture[4], ExactSeconds(0), zero_fixture[6]
    )
    zero_tangent = apply_interaction_kick_tangent(
        zero_fixture[3], zero_fixture[1], zero_fixture[0], zero_fixture[2],
        zero_fixture[4], zero_fixture[5], ExactSeconds(0), zero_fixture[6]
    )
    finite_difference = [
        finite_difference_series(kind) for kind in ("dense", "nonlinear")
    ]
    _, negative_jacobian, _ = canonical_force_and_jacobian(
        "nonsymmetric", synthetic_state(synthetic_model()).q_m
    )
    negative_map = np.eye(24)
    negative_map[12:, :12] = negative_jacobian
    negative_symplectic = symplectic_residual(negative_map)
    maximum_physical = max(value["max_scaled"] for value in physical_cases)
    maximum_tangent = max(value["max_scaled"] for value in tangent_cases)
    maximum_reversal = max(
        max(value["physical_max_scaled"], value["tangent_max_scaled"])
        for value in reversibility_cases
    )
    maximum_composition = max(
        max(value["physical_max_scaled"], value["tangent_max_scaled"])
        for value in composition_cases
    )
    expected_physical_accounting = {
        "events": ["force"], "force": 1, "jvp": 0,
        "observer": 0, "synchronization": 0,
    }
    expected_tangent_accounting = {
        "events": ["force", "jvp"], "force": 1, "jvp": 1,
        "observer": 0, "synchronization": 0,
    }
    return {
        "accounting": {
            "acceptance": (
                all(
                    value["physical"] == expected_physical_accounting
                    and value["tangent"] == expected_tangent_accounting
                    for value in accounting_cases
                )
                and _accounting_payload(zero_physical.metadata)
                == {"events": [], "force": 0, "jvp": 0, "observer": 0,
                    "synchronization": 0}
                and _accounting_payload(zero_tangent.metadata)
                == {"events": [], "force": 0, "jvp": 0, "observer": 0,
                    "synchronization": 0}
                and zero_physical.metadata.force_com_projection is None
                and zero_physical.metadata.jvp_com_projection is None
                and zero_tangent.metadata.force_com_projection is None
                and zero_tangent.metadata.jvp_com_projection is None
            ),
            "cases": accounting_cases,
            "zero_duration": {
                "physical": _accounting_payload(zero_physical.metadata),
                "tangent": _accounting_payload(zero_tangent.metadata),
            },
        },
        "com_projection": {
            "acceptance": all(
                value["within_bound"]
                and value["projection_applied"]
                and value["exact_zero_com_output"]
                for value in projection_cases
            ),
            "cases": projection_cases,
            "maximum_component_ratio": max(
                value["component_ratio_max"] for value in projection_cases
            ),
            "maximum_derived_bound_norm_kg_m_per_s2": max(
                value["derived_bound_norm_kg_m_per_s2"]
                for value in projection_cases
            ),
            "maximum_norm_ratio": max(
                value["norm_ratio"] for value in projection_cases
            ),
            "maximum_raw_residual_norm_kg_m_per_s2": max(
                value["raw_residual_norm_kg_m_per_s2"]
                for value in projection_cases
            ),
        },
        "composition": {
            "acceptance": maximum_composition <= COMPOSITION_CAP,
            "maximum_scaled": maximum_composition,
            "providers": composition_cases,
        },
        "conditioning": {
            "canonical_scale_cond2": 16.0,
            "com_transform_condition_inf": zero_fixture[
                3
            ].com_transform_condition_inf,
            "dense_k_cond2": float(np.linalg.cond(K_DENSE)),
            "jacobi_cond2": float(
                np.linalg.cond(
                    jacobi_matrix(build_jacobi_transform_plan(synthetic_model()))
                )
            ),
        },
        "finite_difference": {
            "acceptance": all(
                all(value["acceptance"].values())
                for value in finite_difference
            ),
            "providers": finite_difference,
        },
        "negative_control": {
            "acceptance": (
                float(
                    np.max(
                        np.abs(negative_jacobian - negative_jacobian.T)
                    )
                ) >= 0.1
                and negative_symplectic[0] >= 1.0e-5
            ),
            "jacobian_asymmetry_max": float(
                np.max(np.abs(negative_jacobian - negative_jacobian.T))
            ),
            "symplectic_raw_frobenius": negative_symplectic[1],
            "symplectic_raw_max": negative_symplectic[0],
        },
        "physical": {
            "acceptance": maximum_physical <= PHYSICAL_CAP,
            "cases": physical_cases,
            "maximum_scaled": maximum_physical,
        },
        "reversibility": {
            "acceptance": maximum_reversal <= REVERSIBILITY_CAP,
            "maximum_scaled": maximum_reversal,
            "providers": reversibility_cases,
        },
        "symplecticity": {
            "acceptance": all(
                value["analytic_matrix_max_error"] <= TANGENT_CAP
                and value["jacobian_symmetry_max"] <= SYMMETRY_CAP
                and value["raw_max"] <= SYMPLECTIC_CAP
                and value["scaled_max"] <= SYMPLECTIC_CAP
                for value in symplectic_cases
            ),
            "providers": symplectic_cases,
        },
        "tangent": {
            "acceptance": maximum_tangent <= TANGENT_CAP,
            "cases": tangent_cases,
            "maximum_scaled": maximum_tangent,
        },
    }


LOCAL_ALLOWED_PREFIXES = (
    "mini_ephemeris.v2",
    "mini_ephemeris.m0_step3g1a_requalification",
    "mini_ephemeris.m0_step3g1b_qualification",
    "mini_ephemeris.m0_step3g1b_qualification_runner",
    "mini_ephemeris.m0_step3g1b_reporting",
    "mini_ephemeris.m0_step3g1c_qualification",
    "mini_ephemeris.m0_step3g1c_qualification_runner",
    "mini_ephemeris.m0_step3g1c_reporting",
    "mini_ephemeris.m0_step3g1d_qualification",
    "mini_ephemeris.m0_step3g1d_qualification_runner",
    "mini_ephemeris.m0_step3g1d_reporting",
)
APPROVED_DEPENDENCY_ROOTS = {
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "packaging",
    "pygments",
    "tomli",
    "exceptiongroup",
    "typing_extensions",
    "numpy",
}
TEST_MODULES = {
    "test_v2_foundation",
    "test_m0_step3g1a_requalification",
    "test_v2_jacobi",
    "test_m0_step3g1b_integrity",
    "test_m0_step3g1b_artifacts",
    "test_v2_kepler",
    "test_m0_step3g1c_integrity",
    "test_m0_step3g1c_artifacts",
    "test_v2_kick",
    "test_m0_step3g1d_integrity",
    "test_m0_step3g1d_artifacts",
}
EXPLICIT_FORBIDDEN_MODULES = (
    "step3g1d_forbidden_sentinel",
    "rebound",
    "reboundx",
)
FORBIDDEN_LOCAL_PREFIXES = (
    "mini_ephemeris.gr_",
    "mini_ephemeris.rebound_",
    "mini_ephemeris.nbody",
    "mini_ephemeris.m0_step3g0",
)
FORBIDDEN_LIBRARY_MARKERS = (
    "step3g1d_forbidden_library_sentinel",
    "libmini_ephemeris_gr_tangent",
    "gr_potential_tangent.so",
    "librebound",
    "libreboundx",
)


def install_namespace_shell(root: Path = ROOT) -> None:
    existing = sys.modules.get("mini_ephemeris")
    if existing is not None:
        if getattr(existing, "__file__", None) is not None:
            raise RuntimeError(
                "legacy mini_ephemeris package initialization already ran"
            )
        return
    package_path = root / "mini_ephemeris/src/mini_ephemeris"
    package = types.ModuleType("mini_ephemeris")
    package.__package__ = "mini_ephemeris"
    package.__path__ = [str(package_path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "mini_ephemeris", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_path)]
    sys.modules["mini_ephemeris"] = package


def _is_allowed_local_module(fullname: str) -> bool:
    return fullname == "mini_ephemeris" or any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in LOCAL_ALLOWED_PREFIXES
    )


def _is_forbidden_module(fullname: str) -> bool:
    if any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in EXPLICIT_FORBIDDEN_MODULES
    ):
        return True
    if any(fullname.startswith(prefix) for prefix in FORBIDDEN_LOCAL_PREFIXES):
        return True
    return (
        fullname.startswith("mini_ephemeris.")
        and not _is_allowed_local_module(fullname)
    )


def reject_forbidden_library_path(path: object) -> None:
    if path is None:
        return
    normalized = os.fsdecode(path).lower()
    if any(marker in normalized for marker in FORBIDDEN_LIBRARY_MARKERS):
        raise RuntimeError(f"forbidden compiled library path: {normalized}")


def loaded_library_violations(
    maps_text: str | None = None,
) -> tuple[str, ...]:
    if maps_text is None:
        path = Path("/proc/self/maps")
        maps_text = path.read_text(encoding="utf-8") if path.exists() else ""
    lowered = maps_text.lower()
    return tuple(
        marker for marker in FORBIDDEN_LIBRARY_MARKERS if marker in lowered
    )


class Step3g1dImportGuard(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.strict = False
        self.preloaded_roots: set[str] = set()

    def activate_strict(self) -> None:
        self.preloaded_roots = {
            name.split(".", 1)[0] for name in sys.modules
        }
        self.strict = True

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> None:
        del path, target
        if _is_forbidden_module(fullname):
            raise ImportError(f"Manifest 27 forbids importing {fullname!r}")
        if not self.strict:
            return None
        root = fullname.split(".", 1)[0]
        allowed = (
            root in sys.stdlib_module_names
            or root.startswith("_sysconfigdata_")
            or root in self.preloaded_roots
            or root in APPROVED_DEPENDENCY_ROOTS
            or root in TEST_MODULES
            or _is_allowed_local_module(fullname)
        )
        if not allowed:
            raise ImportError(
                f"Manifest 27 import root is not allowlisted: {fullname!r}"
            )
        return None


_ACTIVE_GUARD: Step3g1dImportGuard | None = None
_AUDIT_HOOK_INSTALLED = False


def assert_protected_runtime_absent() -> None:
    forbidden = sorted(
        name for name in sys.modules if _is_forbidden_module(name)
    )
    if forbidden:
        raise RuntimeError(f"forbidden modules are loaded: {forbidden}")
    libraries = loaded_library_violations()
    if libraries:
        raise RuntimeError(f"forbidden compiled libraries are loaded: {libraries}")


def install_guard() -> Step3g1dImportGuard:
    global _ACTIVE_GUARD, _AUDIT_HOOK_INSTALLED
    install_namespace_shell()
    if _ACTIVE_GUARD is None:
        _ACTIVE_GUARD = Step3g1dImportGuard()
        sys.meta_path.insert(0, _ACTIVE_GUARD)
    if not _AUDIT_HOOK_INSTALLED:

        def audit_hook(event: str, args: tuple[object, ...]) -> None:
            if event == "ctypes.dlopen" and args:
                reject_forbidden_library_path(args[0])

        sys.addaudithook(audit_hook)
        _AUDIT_HOOK_INSTALLED = True
    assert_protected_runtime_absent()
    return _ACTIVE_GUARD


def _imports_for(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add("." * node.level + (node.module or ""))
    return tuple(sorted(imports))


def _verify_node_ids(node_ids: Iterable[str]) -> None:
    parsed: dict[Path, ast.Module] = {}
    for node_id in node_ids:
        parts = node_id.split("::")
        if len(parts) != 3:
            raise AssertionError(f"node ID is not exact: {node_id}")
        relative, class_name, method_name = parts
        path = ROOT / relative
        tree = parsed.setdefault(
            path,
            ast.parse(
                path.read_text(encoding="utf-8"), filename=str(path)
            ),
        )
        found = any(
            isinstance(node, ast.ClassDef)
            and node.name == class_name
            and any(
                isinstance(method, ast.FunctionDef)
                and method.name == method_name
                for method in node.body
            )
            for node in tree.body
        )
        if not found:
            raise AssertionError(
                f"missing exact node {class_name}::{method_name}"
            )


def _subprocess_inventory(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    found = []
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or not isinstance(node.func.value, ast.Name)
            or node.func.value.id != "subprocess"
        ):
            continue
        owner: ast.AST | None = node
        while owner is not None and not isinstance(owner, ast.FunctionDef):
            owner = parents.get(owner)
        owner_name = (
            owner.name if isinstance(owner, ast.FunctionDef) else "<module>"
        )
        found.append(
            f"{path.relative_to(ROOT)}:{owner_name}:{node.func.attr}"
        )
    return tuple(sorted(found))


def pytest_runtime_audit() -> Mapping[str, Any]:
    observed = {
        relative: sha256_file(PYTEST_SITE_PACKAGES / relative)
        for relative in PYTEST_IDENTITY_SHA256
    }
    if observed != PYTEST_IDENTITY_SHA256:
        raise AssertionError("audited pytest runtime identity changed")
    return {
        "plugin_autoload_disabled": True,
        "sha256": observed,
        "site_packages": str(PYTEST_SITE_PACKAGES),
        "version": PYTEST_VERSION,
    }


def static_safety_audit() -> Mapping[str, Any]:
    manifest = manifest27()
    for relative, expected in manifest["qualified_read_only_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"qualified prior file changed: {relative}")
    expected_v2 = {
        path
        for path in manifest["qualified_read_only_sha256"]
        if path.startswith("mini_ephemeris/src/mini_ephemeris/v2/")
    }
    expected_v2.add(manifest["paths"]["implementation"])
    actual_v2 = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "mini_ephemeris/src/mini_ephemeris/v2").iterdir()
        if path.is_file() and path.suffix == ".py"
    }
    if actual_v2 != expected_v2:
        raise AssertionError("v2 source inventory differs from Manifest 27")

    source_paths = [
        ROOT / manifest["paths"][key]
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
    import_graph = {
        str(path.relative_to(ROOT)): list(_imports_for(path))
        for path in source_paths
    }
    prohibited = sorted(
        module
        for modules in import_graph.values()
        for module in modules
        if module in {"rebound", "reboundx"}
        or module.startswith("mini_ephemeris.gr_")
        or module.startswith("mini_ephemeris.rebound_")
        or module.startswith("mini_ephemeris.m0_step3g0")
    )
    if prohibited:
        raise AssertionError(f"static import graph is unsafe: {prohibited}")

    selection = manifest["exact_test_selection"]
    keys = (
        "step3g1d_core_node_ids",
        "step3g1d_integrity_node_ids",
        "safe_step3g1a_regression_node_ids",
        "safe_step3g1b_regression_node_ids",
        "safe_step3g1c_regression_node_ids",
        "artifact_node_ids",
    )
    all_nodes = [node for key in keys for node in selection[key]]
    if len(all_nodes) != 118 or len(set(all_nodes)) != 118:
        raise AssertionError("Manifest 27 literal-node inventory changed")
    _verify_node_ids(all_nodes)

    helper_path = ROOT / manifest["paths"]["qualification_helper"]
    runner_path = ROOT / manifest["paths"]["runner"]
    subprocesses = tuple(
        sorted(
            _subprocess_inventory(helper_path)
            + _subprocess_inventory(runner_path)
        )
    )
    expected_subprocesses = {
        (
            "mini_ephemeris/src/mini_ephemeris/"
            "m0_step3g1d_qualification.py:git_output:check_output"
        ),
        (
            "mini_ephemeris/src/mini_ephemeris/"
            "m0_step3g1d_qualification.py:run_fresh_artifact_probe:run"
        ),
        (
            "mini_ephemeris/src/mini_ephemeris/"
            "m0_step3g1d_qualification.py:run_fresh_kick_probe:run"
        ),
        (
            "mini_ephemeris/src/mini_ephemeris/"
            "m0_step3g1d_qualification_runner.py:_run_group_worker:run"
        ),
    }
    if set(subprocesses) != expected_subprocesses:
        raise AssertionError(f"subprocess closure changed: {subprocesses}")
    assert_protected_runtime_absent()
    return {
        "active_subprocess_call_sites": list(subprocesses),
        "forbidden_imports": [],
        "forbidden_library_mappings": list(loaded_library_violations()),
        "import_graph": import_graph,
        "legacy_nbody_absent": "mini_ephemeris.nbody" not in sys.modules,
        "legacy_package_init_bypassed": True,
        "pytest_runtime": pytest_runtime_audit(),
        "selected_node_count": len(all_nodes),
        "source_file_count": len(source_paths),
        "status": "PASS",
    }


GUARDED_KICK_PROBE_SOURCE = r"""
import importlib.machinery
import json
from pathlib import Path
import sys
import types
root = Path(sys.argv[1])
package_path = root / "mini_ephemeris/src/mini_ephemeris"
package = types.ModuleType("mini_ephemeris")
package.__package__ = "mini_ephemeris"
package.__path__ = [str(package_path)]
package.__spec__ = importlib.machinery.ModuleSpec(
    "mini_ephemeris", loader=None, is_package=True)
package.__spec__.submodule_search_locations = [str(package_path)]
sys.modules["mini_ephemeris"] = package
sys.path.insert(0, str(root / "mini_ephemeris/src"))
from mini_ephemeris.m0_step3g1d_qualification import (
    assert_protected_runtime_absent, fixture, install_guard, phase, tangent_phase)
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.v2.kick import apply_interaction_kick_tangent
from mini_ephemeris.v2.timebase import ExactSeconds
model, jacobi, provider, plan, state, tangent, context = fixture("nonlinear")
result = apply_interaction_kick_tangent(
    plan, jacobi, model, provider, state, tangent, ExactSeconds(7, 4), context)
assert_protected_runtime_absent()
print(json.dumps({
    "events": [event.operation for event in result.metadata.events],
    "legacy_nbody": "mini_ephemeris.nbody" in sys.modules,
    "plan": plan.fingerprint,
    "projection": {
        "force_bound": result.metadata.force_com_projection.derived_bound_norm_kg_m_per_s2,
        "force_raw": result.metadata.force_com_projection.raw_residual_norm_kg_m_per_s2,
        "jvp_bound": result.metadata.jvp_com_projection.derived_bound_norm_kg_m_per_s2,
        "jvp_raw": result.metadata.jvp_com_projection.raw_residual_norm_kg_m_per_s2,
    },
    "state": phase(result.state).tolist(),
    "tangent": tangent_phase(result.tangent).tolist(),
}, sort_keys=True, separators=(",", ":"), allow_nan=False))
""".strip()


GUARDED_ARTIFACT_PROBE_SOURCE = r"""
import importlib.machinery
from pathlib import Path
import sys
import types
root = Path(sys.argv[1])
destination = Path(sys.argv[2])
package_path = root / "mini_ephemeris/src/mini_ephemeris"
package = types.ModuleType("mini_ephemeris")
package.__package__ = "mini_ephemeris"
package.__path__ = [str(package_path)]
package.__spec__ = importlib.machinery.ModuleSpec(
    "mini_ephemeris", loader=None, is_package=True)
package.__spec__.submodule_search_locations = [str(package_path)]
sys.modules["mini_ephemeris"] = package
sys.path.insert(0, str(root / "mini_ephemeris/src"))
from mini_ephemeris.m0_step3g1d_qualification import (
    assert_protected_runtime_absent, install_guard)
guard = install_guard()
guard.activate_strict()
from mini_ephemeris.m0_step3g1d_reporting import generate_artifacts
generate_artifacts(destination)
assert_protected_runtime_absent()
""".strip()


def run_fresh_kick_probe(hash_seed: int, locale_name: str) -> str:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["LC_ALL"] = locale_name
    environment["LANG"] = locale_name
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            GUARDED_KICK_PROBE_SOURCE,
            str(ROOT),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_fresh_artifact_probe(
    destination: Path,
    hash_seed: int,
    locale_name: str,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["LC_ALL"] = locale_name
    environment["LANG"] = locale_name
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            GUARDED_ARTIFACT_PROBE_SOURCE,
            str(ROOT),
            str(destination),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def git_output(arguments: Sequence[str]) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).rstrip("\n")


def verify_generated_provenance() -> Mapping[str, Any]:
    manifest = manifest27()
    generated = manifest["generated_provenance"]
    frozen = generated["manifests_13_through_25_sha256"]
    manifest_root = ROOT / "ephemeris_experiment_runner/manifests"
    actual = {
        name: sha256_file(manifest_root / name)
        for name in frozen
    }
    if set(actual) != set(frozen) or any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in frozen.values()
    ):
        raise AssertionError("historical hash table is malformed or incomplete")
    if actual != frozen:
        raise AssertionError("generated historical hashes changed")

    manifest25 = strict_json(
        manifest_root
        / "25_m0_step3g1c_kepler_drift_tangent_primitive_v1.json"
    )
    manifest25_values = manifest25["controlled_baseline"][
        "manifest_sha256_13_through_24"
    ]
    for number in range(13, 25):
        matches = [
            name for name in actual if name.startswith(f"{number:02d}_")
        ]
        if len(matches) != 1 or actual[matches[0]] != manifest25_values[str(number)]:
            raise AssertionError(
                f"Manifest 25 provenance mismatch for Manifest {number}"
            )

    copied = manifest26()["inherited_integrity"][
        "manifests_13_through_25_sha256"
    ]
    observed_errors = [
        {
            "actual_sha256": actual[name],
            "file": name,
            "manifest26_sha256": copied[name],
        }
        for name in sorted(actual)
        if copied.get(name) != actual[name]
    ]
    if observed_errors != generated["manifest26_errors"]:
        raise AssertionError("Manifest 26 error inventory changed")
    if len(observed_errors) != 12 or {
        int(value["file"].split("_", 1)[0])
        for value in observed_errors
    } != set(range(13, 25)):
        raise AssertionError("Manifest 26 error set is not exactly 13 through 24")
    return {
        "historical_manifests": len(actual),
        "manifest25_matches": True,
        "manifest26_error_count": len(observed_errors),
        "status": "PASS",
    }


def verify_inherited_integrity() -> Mapping[str, Any]:
    from mini_ephemeris.m0_step3g1c_qualification import (
        verify_inherited_integrity as verify_step3g1c,
    )

    prior = verify_step3g1c()
    manifest = manifest27()
    provenance = verify_generated_provenance()
    expected = dict(manifest["protected_sources"])
    expected.update(manifest["qualified_read_only_sha256"])
    mismatches = {
        relative: {
            "expected": digest,
            "observed": (
                sha256_file(ROOT / relative)
                if (ROOT / relative).is_file()
                else None
            ),
        }
        for relative, digest in expected.items()
        if not (ROOT / relative).is_file()
        or sha256_file(ROOT / relative) != digest
    }
    if mismatches:
        raise AssertionError(f"inherited hash mismatch: {mismatches}")

    for relative, digest in manifest["corrective_start_sha256"].items():
        if not relative.startswith(
            "docs/validation/m0-step3g1d-interaction-kick-tangent-primitive-v1/"
        ):
            continue
        if sha256_file(ROOT / relative) != digest:
            raise AssertionError(f"Manifest 26 blocked evidence changed: {relative}")
    if sha256_file(MANIFEST26) != manifest[
        "manifest26_permanent_disposition"
    ]["manifest_sha256"]:
        raise AssertionError("Manifest 26 changed")

    artifact_root = ROOT / (
        "docs/validation/"
        "m0-step3g1c-kepler-drift-tangent-primitive-v1"
    )
    artifact_hashes = strict_json(
        artifact_root / "artifact_hashes.json"
    )["sha256"]
    for name, digest in artifact_hashes.items():
        if sha256_file(artifact_root / name) != digest:
            raise AssertionError(f"Step 3g1c artifact changed: {name}")
    for tag, identity in manifest["protected_tags"].items():
        if (
            git_output(["rev-parse", f"{tag}^{{}}"]) != identity["commit"]
            or git_output(["rev-parse", tag]) != identity["tag_object"]
        ):
            raise AssertionError(f"protected tag changed: {tag}")
    if prior["status"] != "PASS" or provenance["status"] != "PASS":
        raise AssertionError("inherited integrity did not pass")
    return {
        "checked_hashes": (
            prior["checked_hashes"]
            + len(expected)
            + provenance["historical_manifests"]
            + len(artifact_hashes)
            + 3
        ),
        "historical_manifests": provenance["historical_manifests"],
        "manifest26_error_count": provenance["manifest26_error_count"],
        "protected_tags": 2,
        "status": "PASS",
    }
