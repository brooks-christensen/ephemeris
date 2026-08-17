"""Isolated canonical interaction kick driven by pure acceleration/JVP providers.

The public force boundary returns inertial Cartesian acceleration.  This module
performs the explicit fixed-mass covector conversion required by the canonical
Jacobi kick.  It does not compose a drift, synchronize, observe, integrate, or
retain mutable evaluation state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Protocol, Tuple

from .accounting import AccountingDomain, AccountingEvent
from .canonical import canonical_json_bytes, finite_binary64_hex, sha256_hex
from .errors import InvalidModel, InvalidState, KernelContractError, LayoutMismatch
from .jacobi import (
    AXIS_ORDER,
    COORDINATE_CONVENTION,
    PHASE_ORDER,
    JacobiTransformPlan,
)
from .kernels import ForceEvaluationContext, evaluate_force, evaluate_jvp
from .model import PhysicalModel
from .state import (
    CanonicalJacobiState,
    CanonicalJacobiTangentState,
    CartesianPositionTangent,
    InertialCartesianState,
    VectorRows,
    require_canonical_tangent_compatible,
)
from .timebase import ExactSeconds


CAPABILITY_SCHEMA = "v2.interaction_provider_capabilities/1"
PLAN_SCHEMA = "v2.interaction_kick_plan/1"
RAW_FORCE_OUTPUT = "inertial_cartesian_acceleration"
RAW_JVP_OUTPUT = "inertial_cartesian_acceleration_jvp"
CANONICAL_ADAPTER = "mass_weight_then_A_inverse_transpose_v1"
COM_CLOSURE_SCHEMA = "binary64_internal_force_com_closure_v1"
STATE_SCHEMA = "canonical_jacobi_state_v1"
TANGENT_SCHEMA = "canonical_jacobi_tangent_v1"
UNIT_SYSTEM = "si_v1"
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ZERO_ROW = (0.0, 0.0, 0.0)


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise KernelContractError(f"{field} must be a nonempty trimmed string")
    return value


def _require_fingerprint(value: str, field: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise KernelContractError(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _require_bool(value: bool, field: str) -> bool:
    if not isinstance(value, bool):
        raise KernelContractError(f"{field} must be boolean")
    return value


@dataclass(frozen=True, init=False)
class InteractionProviderCapabilities:
    """Immutable declaration required before a provider may drive a kick."""

    provider_id: str
    provider_fingerprint: str
    model_fingerprint: str
    layout_fingerprint: str
    raw_force_output: str
    raw_jvp_output: str
    canonical_adapter: str
    state_schema: str
    tangent_schema: str
    unit_system_id: str
    position_only: bool
    jvp_available: bool
    fixed_mass_canonical_adapter: bool
    conservative_canonical_force: bool
    symmetric_canonical_jacobian: bool
    zero_center_of_mass_force: bool
    deterministic: bool
    no_hidden_accounting: bool
    schema: str

    def __init__(
        self,
        *,
        provider_id: str,
        provider_fingerprint: str,
        model_fingerprint: str,
        layout_fingerprint: str,
        raw_force_output: str = RAW_FORCE_OUTPUT,
        raw_jvp_output: str = RAW_JVP_OUTPUT,
        canonical_adapter: str = CANONICAL_ADAPTER,
        state_schema: str = STATE_SCHEMA,
        tangent_schema: str = TANGENT_SCHEMA,
        unit_system_id: str = UNIT_SYSTEM,
        position_only: bool = True,
        jvp_available: bool = True,
        fixed_mass_canonical_adapter: bool = True,
        conservative_canonical_force: bool = True,
        symmetric_canonical_jacobian: bool = True,
        zero_center_of_mass_force: bool = True,
        deterministic: bool = True,
        no_hidden_accounting: bool = True,
        schema: str = CAPABILITY_SCHEMA,
    ) -> None:
        object.__setattr__(self, "provider_id", _require_text(provider_id, "provider_id"))
        object.__setattr__(
            self,
            "provider_fingerprint",
            _require_fingerprint(provider_fingerprint, "provider_fingerprint"),
        )
        object.__setattr__(
            self,
            "model_fingerprint",
            _require_fingerprint(model_fingerprint, "model_fingerprint"),
        )
        object.__setattr__(
            self,
            "layout_fingerprint",
            _require_fingerprint(layout_fingerprint, "layout_fingerprint"),
        )
        for field, value in (
            ("raw_force_output", raw_force_output),
            ("raw_jvp_output", raw_jvp_output),
            ("canonical_adapter", canonical_adapter),
            ("state_schema", state_schema),
            ("tangent_schema", tangent_schema),
            ("unit_system_id", unit_system_id),
            ("schema", schema),
        ):
            object.__setattr__(self, field, _require_text(value, field))
        for field, value in (
            ("position_only", position_only),
            ("jvp_available", jvp_available),
            ("fixed_mass_canonical_adapter", fixed_mass_canonical_adapter),
            ("conservative_canonical_force", conservative_canonical_force),
            ("symmetric_canonical_jacobian", symmetric_canonical_jacobian),
            ("zero_center_of_mass_force", zero_center_of_mass_force),
            ("deterministic", deterministic),
            ("no_hidden_accounting", no_hidden_accounting),
        ):
            object.__setattr__(self, field, _require_bool(value, field))

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "canonical_adapter": self.canonical_adapter,
            "conservative_canonical_force": self.conservative_canonical_force,
            "deterministic": self.deterministic,
            "fixed_mass_canonical_adapter": self.fixed_mass_canonical_adapter,
            "jvp_available": self.jvp_available,
            "layout_fingerprint": self.layout_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "no_hidden_accounting": self.no_hidden_accounting,
            "position_only": self.position_only,
            "provider_fingerprint": self.provider_fingerprint,
            "provider_id": self.provider_id,
            "raw_force_output": self.raw_force_output,
            "raw_jvp_output": self.raw_jvp_output,
            "schema": self.schema,
            "state_schema": self.state_schema,
            "symmetric_canonical_jacobian": self.symmetric_canonical_jacobian,
            "tangent_schema": self.tangent_schema,
            "unit_system_id": self.unit_system_id,
            "zero_center_of_mass_force": self.zero_center_of_mass_force,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_hex(canonical_json_bytes(self.canonical_payload()))


class InteractionProvider(Protocol):
    """Pure force/JVP provider with immutable kick capability metadata."""

    capabilities: InteractionProviderCapabilities
    provider_fingerprint: str

    def evaluate(self, model: PhysicalModel, state: InertialCartesianState, context: ForceEvaluationContext): ...

    def jvp(self, model: PhysicalModel, state: InertialCartesianState, direction: CartesianPositionTangent, context: ForceEvaluationContext): ...


def _com_closure_parameters(
    plan: JacobiTransformPlan,
) -> tuple[float, int, float]:
    """Return kappa_inf(A^-T), the rounded-op count, and gamma_k."""

    count = len(plan.masses_kg)
    covector_row_sums = [float(count)]
    for index in range(1, count):
        covector_row_sums.append(
            math.fsum(
                (
                    index * abs(plan.body_mass_fractions[index]),
                    abs(plan.inner_mass_fractions[index]),
                )
            )
        )
    norm_a_inverse_transpose = max(covector_row_sums)

    total_mass = plan.cumulative_masses_kg[-1]
    position_column_sums = []
    for column in range(count):
        terms = [abs(plan.masses_kg[column] / total_mass)]
        for row in range(1, count):
            if column < row:
                terms.append(
                    abs(
                        plan.masses_kg[column]
                        / plan.cumulative_masses_kg[row - 1]
                    )
                )
            elif column == row:
                terms.append(1.0)
        position_column_sums.append(math.fsum(terms))
    norm_a_transpose = max(position_column_sums)
    condition = norm_a_inverse_transpose * norm_a_transpose

    operation_count = 2 * count - 1
    unit_roundoff = 2.0**-53
    denominator = 1.0 - operation_count * unit_roundoff
    gamma = operation_count * unit_roundoff / denominator
    if (
        not math.isfinite(condition)
        or condition < 1.0
        or operation_count < 1
        or denominator <= 0.0
        or not math.isfinite(gamma)
    ):
        raise InvalidModel("invalid binary64 COM-closure parameters")
    return condition, operation_count, gamma


@dataclass(frozen=True, init=False)
class InteractionKickPlan:
    """Immutable identity and capability plan for one interaction provider."""

    model_fingerprint: str
    layout_fingerprint: str
    jacobi_plan_fingerprint: str
    provider_fingerprint: str
    capability_fingerprint: str
    capabilities: InteractionProviderCapabilities
    unit_system_id: str
    coordinate_convention: str
    axis_order: str
    phase_order: str
    body_count: int
    com_closure_schema: str
    com_transform_condition_inf: float
    com_roundoff_operation_count: int
    com_roundoff_gamma: float
    schema: str
    fingerprint: str

    def __init__(
        self,
        model: PhysicalModel,
        jacobi_plan: JacobiTransformPlan,
        capabilities: InteractionProviderCapabilities,
    ) -> None:
        _require_plan_inputs(model, jacobi_plan, capabilities)
        _require_qualified_capabilities(capabilities)
        object.__setattr__(self, "model_fingerprint", model.fingerprint)
        object.__setattr__(self, "layout_fingerprint", model.layout.fingerprint)
        object.__setattr__(self, "jacobi_plan_fingerprint", jacobi_plan.fingerprint)
        object.__setattr__(self, "provider_fingerprint", capabilities.provider_fingerprint)
        object.__setattr__(self, "capability_fingerprint", capabilities.fingerprint)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "unit_system_id", UNIT_SYSTEM)
        object.__setattr__(self, "coordinate_convention", COORDINATE_CONVENTION)
        object.__setattr__(self, "axis_order", AXIS_ORDER)
        object.__setattr__(self, "phase_order", PHASE_ORDER)
        object.__setattr__(self, "body_count", len(model.layout.body_ids))
        condition, operation_count, gamma = _com_closure_parameters(jacobi_plan)
        object.__setattr__(self, "com_closure_schema", COM_CLOSURE_SCHEMA)
        object.__setattr__(self, "com_transform_condition_inf", condition)
        object.__setattr__(self, "com_roundoff_operation_count", operation_count)
        object.__setattr__(self, "com_roundoff_gamma", gamma)
        object.__setattr__(self, "schema", PLAN_SCHEMA)
        object.__setattr__(self, "fingerprint", sha256_hex(self.canonical_bytes()))

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "axis_order": self.axis_order,
            "body_count": self.body_count,
            "capabilities": self.capabilities.canonical_payload(),
            "capability_fingerprint": self.capability_fingerprint,
            "com_closure_schema": self.com_closure_schema,
            "com_roundoff_gamma_hex": finite_binary64_hex(
                self.com_roundoff_gamma, "COM roundoff gamma"
            ),
            "com_roundoff_operation_count": self.com_roundoff_operation_count,
            "com_transform_condition_inf_hex": finite_binary64_hex(
                self.com_transform_condition_inf, "COM transform condition"
            ),
            "coordinate_convention": self.coordinate_convention,
            "jacobi_plan_fingerprint": self.jacobi_plan_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "phase_order": self.phase_order,
            "provider_fingerprint": self.provider_fingerprint,
            "schema": self.schema,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.canonical_payload())


@dataclass(frozen=True)
class ComProjectionDiagnostics:
    """Immutable evidence for one accepted internal-force COM projection."""

    raw_residual_kg_m_per_s2: Tuple[float, float, float]
    component_bounds_kg_m_per_s2: Tuple[float, float, float]
    componentwise_absolute_force_sums_kg_m_per_s2: Tuple[float, float, float]
    raw_residual_norm_kg_m_per_s2: float
    derived_bound_norm_kg_m_per_s2: float
    transform_condition_inf: float
    accumulated_force_terms: int
    rounded_operation_count: int
    gamma: float
    projection_applied: bool


@dataclass(frozen=True)
class KickEvaluationMetadata:
    """Detached immutable accounting and COM-projection evidence."""

    events: Tuple[AccountingEvent, ...]
    force_evaluations: int
    jvp_evaluations: int
    observer_evaluations: int
    synchronization_evaluations: int
    request_id: str
    plan_fingerprint: str
    force_com_projection: ComProjectionDiagnostics | None
    jvp_com_projection: ComProjectionDiagnostics | None


@dataclass(frozen=True)
class InteractionKickResult:
    state: CanonicalJacobiState
    canonical_force_kg_m_per_s2: VectorRows
    metadata: KickEvaluationMetadata


@dataclass(frozen=True)
class InteractionKickTangentResult:
    state: CanonicalJacobiState
    tangent: CanonicalJacobiTangentState
    canonical_force_kg_m_per_s2: VectorRows
    canonical_force_jvp_kg_m_per_s2: VectorRows
    metadata: KickEvaluationMetadata


def _require_plan_inputs(
    model: PhysicalModel,
    jacobi_plan: JacobiTransformPlan,
    capabilities: InteractionProviderCapabilities,
) -> None:
    if not isinstance(model, PhysicalModel):
        raise InvalidModel("interaction kick requires a PhysicalModel")
    if not isinstance(jacobi_plan, JacobiTransformPlan):
        raise InvalidModel("interaction kick requires a JacobiTransformPlan")
    if not isinstance(capabilities, InteractionProviderCapabilities):
        raise KernelContractError("provider capabilities have the wrong semantic type")
    if jacobi_plan.model_fingerprint != model.fingerprint:
        raise LayoutMismatch("Jacobi plan and model fingerprints differ")
    if jacobi_plan.layout != model.layout:
        raise LayoutMismatch("Jacobi plan and model layouts differ")
    if capabilities.model_fingerprint != model.fingerprint:
        raise LayoutMismatch("provider capability and model fingerprints differ")
    if capabilities.layout_fingerprint != model.layout.fingerprint:
        raise LayoutMismatch("provider capability and model layouts differ")


def _require_qualified_capabilities(capabilities: InteractionProviderCapabilities) -> None:
    literals = {
        "schema": CAPABILITY_SCHEMA,
        "raw_force_output": RAW_FORCE_OUTPUT,
        "raw_jvp_output": RAW_JVP_OUTPUT,
        "canonical_adapter": CANONICAL_ADAPTER,
        "state_schema": STATE_SCHEMA,
        "tangent_schema": TANGENT_SCHEMA,
        "unit_system_id": UNIT_SYSTEM,
    }
    for field, expected in literals.items():
        if getattr(capabilities, field) != expected:
            raise KernelContractError(f"provider capability {field} is incompatible")
    for field in (
        "position_only",
        "jvp_available",
        "fixed_mass_canonical_adapter",
        "conservative_canonical_force",
        "symmetric_canonical_jacobian",
        "zero_center_of_mass_force",
        "deterministic",
        "no_hidden_accounting",
    ):
        if not getattr(capabilities, field):
            raise KernelContractError(f"provider capability {field} is required")


def build_interaction_kick_plan(
    model: PhysicalModel,
    jacobi_plan: JacobiTransformPlan,
    capabilities: InteractionProviderCapabilities,
) -> InteractionKickPlan:
    return InteractionKickPlan(model, jacobi_plan, capabilities)


def _require_identity(
    plan: InteractionKickPlan,
    jacobi_plan: JacobiTransformPlan,
    model: PhysicalModel,
    provider: InteractionProvider,
    state: CanonicalJacobiState,
    context: ForceEvaluationContext,
) -> None:
    if not isinstance(plan, InteractionKickPlan):
        raise InvalidModel("plan must be InteractionKickPlan")
    if not isinstance(state, CanonicalJacobiState):
        raise InvalidState("interaction kick requires CanonicalJacobiState")
    if not isinstance(context, ForceEvaluationContext):
        raise KernelContractError("interaction kick requires ForceEvaluationContext")
    if context.domain is not AccountingDomain.MAP_STAGE:
        raise KernelContractError("interaction kick context must use MAP_STAGE")
    _require_plan_inputs(model, jacobi_plan, plan.capabilities)
    if plan.model_fingerprint != model.fingerprint:
        raise LayoutMismatch("kick plan and model fingerprints differ")
    if plan.layout_fingerprint != model.layout.fingerprint or state.layout != model.layout:
        raise LayoutMismatch("kick plan, state, and model layouts differ")
    if plan.jacobi_plan_fingerprint != jacobi_plan.fingerprint:
        raise LayoutMismatch("kick and Jacobi plan fingerprints differ")
    if state.unit_system_id != UNIT_SYSTEM or model.units.identifier != UNIT_SYSTEM:
        raise LayoutMismatch("kick requires matching SI units")
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, InteractionProviderCapabilities):
        raise KernelContractError("provider lacks typed immutable capabilities")
    if capabilities != plan.capabilities or capabilities.fingerprint != plan.capability_fingerprint:
        raise KernelContractError("provider capabilities do not match the kick plan")
    if getattr(provider, "provider_fingerprint", None) != plan.provider_fingerprint:
        raise KernelContractError("provider fingerprint does not match the kick plan")


def _duration_seconds(duration: ExactSeconds) -> float:
    if not isinstance(duration, ExactSeconds):
        raise InvalidState("kick duration must be ExactSeconds")
    try:
        seconds = duration.to_binary64()
    except OverflowError as exc:
        raise InvalidState("kick duration is not finite in binary64") from exc
    if not math.isfinite(seconds):
        raise InvalidState("kick duration is not finite in binary64")
    return seconds


def _inverse_position_rows(plan: JacobiTransformPlan, rows: VectorRows) -> VectorRows:
    count = len(rows)
    output = [[0.0, 0.0, 0.0] for _ in range(count)]
    first = [rows[0][axis] for axis in range(3)]
    for index in range(1, count):
        beta = plan.body_mass_fractions[index]
        for axis in range(3):
            first[axis] = first[axis] - beta * rows[index][axis]
    output[0] = first
    inner_com = first.copy()
    for index in range(1, count):
        beta = plan.body_mass_fractions[index]
        for axis in range(3):
            output[index][axis] = rows[index][axis] + inner_com[axis]
            inner_com[axis] = inner_com[axis] + beta * rows[index][axis]
    return tuple(tuple(row) for row in output)  # type: ignore[return-value]


def _forward_covector_rows(plan: JacobiTransformPlan, rows: VectorRows) -> VectorRows:
    count = len(rows)
    output = [[0.0, 0.0, 0.0] for _ in range(count)]
    prefix = [rows[0][axis] for axis in range(3)]
    for index in range(1, count):
        alpha = plan.inner_mass_fractions[index]
        beta = plan.body_mass_fractions[index]
        for axis in range(3):
            output[index][axis] = alpha * rows[index][axis] - beta * prefix[axis]
            prefix[axis] = prefix[axis] + rows[index][axis]
    output[0] = prefix
    return tuple(tuple(row) for row in output)  # type: ignore[return-value]


def _project_internal_rows(rows: VectorRows) -> VectorRows:
    return (_ZERO_ROW,) + rows[1:]


def _inertial_state(plan: JacobiTransformPlan, model: PhysicalModel, q: VectorRows) -> InertialCartesianState:
    positions = _inverse_position_rows(plan, q)
    zeros = tuple(_ZERO_ROW for _ in positions)
    return InertialCartesianState(
        layout=model.layout,
        positions_m=positions,
        velocities_m_per_s=zeros,
        unit_system_id=UNIT_SYSTEM,
    )


def _canonical_force(
    kick_plan: InteractionKickPlan,
    jacobi_plan: JacobiTransformPlan,
    acceleration: VectorRows,
) -> tuple[VectorRows, ComProjectionDiagnostics]:
    inertial_force = tuple(
        tuple(jacobi_plan.masses_kg[index] * component for component in row)
        for index, row in enumerate(acceleration)
    )
    if not all(math.isfinite(value) for row in inertial_force for value in row):
        raise KernelContractError("mass-weighted provider force is nonfinite")

    force = _forward_covector_rows(jacobi_plan, inertial_force)
    raw_residual = force[0]
    try:
        component_sums = tuple(
            math.fsum(abs(row[axis]) for row in inertial_force)
            for axis in range(3)
        )
    except OverflowError as exc:
        raise KernelContractError(
            "mass-weighted absolute-force scale overflowed"
        ) from exc
    factor = (
        kick_plan.com_roundoff_gamma
        * kick_plan.com_transform_condition_inf
    )
    component_bounds = tuple(factor * scale for scale in component_sums)
    raw_norm = math.hypot(*raw_residual)
    bound_norm = math.hypot(*component_bounds)
    values = (
        *raw_residual,
        *component_sums,
        *component_bounds,
        raw_norm,
        bound_norm,
    )
    if not all(math.isfinite(value) for value in values):
        raise KernelContractError("COM-closure diagnostic is nonfinite")
    if (
        any(
            abs(residual) > bound
            for residual, bound in zip(raw_residual, component_bounds)
        )
        or raw_norm > bound_norm
    ):
        raise KernelContractError(
            "provider violates the derived binary64 COM-force closure bound"
        )

    projected = list(force)
    projected[0] = _ZERO_ROW
    diagnostics = ComProjectionDiagnostics(
        raw_residual_kg_m_per_s2=raw_residual,
        component_bounds_kg_m_per_s2=component_bounds,
        componentwise_absolute_force_sums_kg_m_per_s2=component_sums,
        raw_residual_norm_kg_m_per_s2=raw_norm,
        derived_bound_norm_kg_m_per_s2=bound_norm,
        transform_condition_inf=kick_plan.com_transform_condition_inf,
        accumulated_force_terms=kick_plan.body_count,
        rounded_operation_count=kick_plan.com_roundoff_operation_count,
        gamma=kick_plan.com_roundoff_gamma,
        projection_applied=True,
    )
    return tuple(projected), diagnostics


def _metadata(
    plan: InteractionKickPlan,
    context: ForceEvaluationContext,
    *,
    tangent: bool,
    zero: bool,
    force_projection: ComProjectionDiagnostics | None = None,
    jvp_projection: ComProjectionDiagnostics | None = None,
) -> KickEvaluationMetadata:
    if zero:
        if force_projection is not None or jvp_projection is not None:
            raise KernelContractError("zero-duration metadata cannot project")
    elif force_projection is None or (tangent != (jvp_projection is not None)):
        raise KernelContractError("COM-projection metadata is incomplete")
    names = () if zero else (("force", "jvp") if tangent else ("force",))
    events = tuple(AccountingEvent(AccountingDomain.MAP_STAGE, name) for name in names)
    return KickEvaluationMetadata(
        events=events,
        force_evaluations=0 if zero else 1,
        jvp_evaluations=1 if tangent and not zero else 0,
        observer_evaluations=0,
        synchronization_evaluations=0,
        request_id=context.request_id,
        plan_fingerprint=plan.fingerprint,
        force_com_projection=force_projection,
        jvp_com_projection=jvp_projection,
    )


def apply_interaction_kick(
    plan: InteractionKickPlan,
    jacobi_plan: JacobiTransformPlan,
    model: PhysicalModel,
    provider: InteractionProvider,
    state: CanonicalJacobiState,
    duration: ExactSeconds,
    context: ForceEvaluationContext,
) -> InteractionKickResult:
    """Apply one physical position-only kick with exactly one force call."""

    _require_identity(plan, jacobi_plan, model, provider, state, context)
    seconds = _duration_seconds(duration)
    if duration.numerator == 0:
        detached = CanonicalJacobiState(
            state.layout, state.q_m, state.p_kg_m_per_s, state.unit_system_id
        )
        return InteractionKickResult(detached, tuple(_ZERO_ROW for _ in state.q_m), _metadata(plan, context, tangent=False, zero=True))
    projected_q = _project_internal_rows(state.q_m)
    inertial = _inertial_state(jacobi_plan, model, projected_q)
    acceleration = evaluate_force(provider, model, inertial, context)
    force, force_projection = _canonical_force(
        plan, jacobi_plan, acceleration.values_m_per_s2
    )
    momenta = tuple(
        tuple(p + seconds * f for p, f in zip(p_row, f_row))
        for p_row, f_row in zip(state.p_kg_m_per_s, force)
    )
    result = CanonicalJacobiState(state.layout, state.q_m, momenta, state.unit_system_id)
    return InteractionKickResult(
        result,
        force,
        _metadata(
            plan,
            context,
            tangent=False,
            zero=False,
            force_projection=force_projection,
        ),
    )


def apply_interaction_kick_tangent(
    plan: InteractionKickPlan,
    jacobi_plan: JacobiTransformPlan,
    model: PhysicalModel,
    provider: InteractionProvider,
    state: CanonicalJacobiState,
    tangent: CanonicalJacobiTangentState,
    duration: ExactSeconds,
    context: ForceEvaluationContext,
) -> InteractionKickTangentResult:
    """Apply one physical kick and its canonical tangent action."""

    _require_identity(plan, jacobi_plan, model, provider, state, context)
    if not isinstance(tangent, CanonicalJacobiTangentState):
        raise InvalidState("tangent kick requires CanonicalJacobiTangentState")
    require_canonical_tangent_compatible(state, tangent)
    seconds = _duration_seconds(duration)
    if duration.numerator == 0:
        detached_state = CanonicalJacobiState(
            state.layout, state.q_m, state.p_kg_m_per_s, state.unit_system_id
        )
        detached_tangent = CanonicalJacobiTangentState(
            tangent.layout,
            tangent.delta_q_m,
            tangent.delta_p_kg_m_per_s,
            tangent.unit_system_id,
        )
        zeros = tuple(_ZERO_ROW for _ in state.q_m)
        return InteractionKickTangentResult(
            detached_state,
            detached_tangent,
            zeros,
            zeros,
            _metadata(plan, context, tangent=True, zero=True),
        )
    projected_q = _project_internal_rows(state.q_m)
    inertial = _inertial_state(jacobi_plan, model, projected_q)
    acceleration = evaluate_force(provider, model, inertial, context)
    force, force_projection = _canonical_force(
        plan, jacobi_plan, acceleration.values_m_per_s2
    )
    projected_delta_q = _project_internal_rows(tangent.delta_q_m)
    delta_positions = _inverse_position_rows(jacobi_plan, projected_delta_q)
    direction = CartesianPositionTangent(model.layout, delta_positions, UNIT_SYSTEM)
    acceleration_jvp = evaluate_jvp(provider, model, inertial, direction, context)
    force_jvp, jvp_projection = _canonical_force(
        plan, jacobi_plan, acceleration_jvp.values_m_per_s2
    )
    momenta = tuple(
        tuple(p + seconds * f for p, f in zip(p_row, f_row))
        for p_row, f_row in zip(state.p_kg_m_per_s, force)
    )
    delta_momenta = tuple(
        tuple(dp + seconds * df for dp, df in zip(dp_row, df_row))
        for dp_row, df_row in zip(tangent.delta_p_kg_m_per_s, force_jvp)
    )
    result_state = CanonicalJacobiState(state.layout, state.q_m, momenta, state.unit_system_id)
    result_tangent = CanonicalJacobiTangentState(
        tangent.layout, tangent.delta_q_m, delta_momenta, tangent.unit_system_id
    )
    return InteractionKickTangentResult(
        result_state,
        result_tangent,
        force,
        force_jvp,
        _metadata(
            plan,
            context,
            tangent=True,
            zero=False,
            force_projection=force_projection,
            jvp_projection=jvp_projection,
        ),
    )


__all__ = [
    "CANONICAL_ADAPTER",
    "CAPABILITY_SCHEMA",
    "COM_CLOSURE_SCHEMA",
    "ComProjectionDiagnostics",
    "InteractionKickPlan",
    "InteractionKickResult",
    "InteractionKickTangentResult",
    "InteractionProvider",
    "InteractionProviderCapabilities",
    "KickEvaluationMetadata",
    "PLAN_SCHEMA",
    "RAW_FORCE_OUTPUT",
    "RAW_JVP_OUTPUT",
    "STATE_SCHEMA",
    "TANGENT_SCHEMA",
    "apply_interaction_kick",
    "apply_interaction_kick_tangent",
    "build_interaction_kick_plan",
]
