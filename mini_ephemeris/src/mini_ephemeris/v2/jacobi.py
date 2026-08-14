"""Fixed-mass canonical Jacobi coordinate and tangent transformations.

This module implements only the linear canonical change of coordinates between
inertial ``(x,p)`` and Jacobi ``(q,P)`` values.  It performs no velocity
conversion, force evaluation, time evolution, synchronization, or accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .canonical import canonical_json_bytes, finite_binary64_hex, sha256_hex
from .errors import InvalidModel, InvalidState, LayoutMismatch
from .model import BodyId, CompiledLayout, PhysicalModel
from .state import CanonicalJacobiState, CanonicalJacobiTangentState


Vector3 = Tuple[float, float, float]
VectorRows = Tuple[Vector3, ...]

PLAN_SCHEMA = "v2.canonical_jacobi_transform_plan/1"
INERTIAL_STATE_SCHEMA = "v2.inertial_canonical_state/1"
INERTIAL_TANGENT_SCHEMA = "v2.inertial_canonical_tangent_state/1"
AXIS_ORDER = "body-major rows; each row is (x,y,z)"
PHASE_ORDER = "all body-major position rows, then all body-major momentum rows"
COORDINATE_CONVENTION = "com-pair-retained; relative=body-minus-inner-com"
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _immutable_vectors(
    values: Iterable[Sequence[float]], expected_rows: int, field: str
) -> VectorRows:
    rows = []
    for raw_row in values:
        components = tuple(raw_row)
        if len(components) != 3:
            raise InvalidState(f"{field} rows must each have exactly three components")
        if any(
            isinstance(component, bool) or not isinstance(component, Real)
            for component in components
        ):
            raise InvalidState(f"{field} components must be real numbers")
        row = tuple(float(component) for component in components)
        if not all(math.isfinite(component) for component in row):
            raise InvalidState(f"{field} must contain only finite binary64 values")
        rows.append(row)
    if len(rows) != expected_rows:
        raise InvalidState(f"{field} row count does not match the explicit body layout")
    return tuple(rows)


def _require_si(unit_system_id: str) -> str:
    if unit_system_id != "si_v1":
        raise InvalidState("canonical transforms require unit_system_id='si_v1'")
    return unit_system_id


def _require_fingerprint(value: str, field: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_PATTERN.fullmatch(value):
        raise InvalidState(f"{field} must be a lowercase SHA-256 fingerprint")
    return value


def _rows_payload(rows: VectorRows) -> list[list[str]]:
    return [
        [finite_binary64_hex(component, "state component") for component in row]
        for row in rows
    ]


@dataclass(frozen=True, init=False)
class InertialCanonicalState:
    """Immutable inertial canonical ``(x,p)`` state in SI units.

    This semantic type cannot be substituted with ``InertialCartesianState``:
    its second field is momentum in kg*m/s, never velocity.
    """

    layout: CompiledLayout
    positions_m: VectorRows
    momenta_kg_m_per_s: VectorRows
    unit_system_id: str
    model_fingerprint: str

    def __init__(
        self,
        *,
        layout: CompiledLayout,
        positions_m: Iterable[Sequence[float]],
        momenta_kg_m_per_s: Iterable[Sequence[float]],
        unit_system_id: str,
        model_fingerprint: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("inertial canonical state layout must be CompiledLayout")
        count = len(layout.body_ids)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self, "positions_m", _immutable_vectors(positions_m, count, "positions_m")
        )
        object.__setattr__(
            self,
            "momenta_kg_m_per_s",
            _immutable_vectors(momenta_kg_m_per_s, count, "momenta_kg_m_per_s"),
        )
        object.__setattr__(self, "unit_system_id", _require_si(unit_system_id))
        object.__setattr__(
            self,
            "model_fingerprint",
            _require_fingerprint(model_fingerprint, "model_fingerprint"),
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return the deterministic state serialization payload."""

        return {
            "layout_fingerprint": self.layout.fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "momenta_kg_m_per_s_hex": _rows_payload(self.momenta_kg_m_per_s),
            "positions_m_hex": _rows_payload(self.positions_m),
            "schema": INERTIAL_STATE_SCHEMA,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize this state deterministically without mutable aliases."""

        return canonical_json_bytes(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 identity of the exact canonical state bytes."""

        return sha256_hex(self.canonical_bytes())


@dataclass(frozen=True, init=False)
class InertialCanonicalTangentState:
    """Immutable inertial canonical tangent ``(delta_x,delta_p)`` in SI units."""

    layout: CompiledLayout
    delta_positions_m: VectorRows
    delta_momenta_kg_m_per_s: VectorRows
    unit_system_id: str
    model_fingerprint: str

    def __init__(
        self,
        *,
        layout: CompiledLayout,
        delta_positions_m: Iterable[Sequence[float]],
        delta_momenta_kg_m_per_s: Iterable[Sequence[float]],
        unit_system_id: str,
        model_fingerprint: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("inertial tangent layout must be CompiledLayout")
        count = len(layout.body_ids)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "delta_positions_m",
            _immutable_vectors(delta_positions_m, count, "delta_positions_m"),
        )
        object.__setattr__(
            self,
            "delta_momenta_kg_m_per_s",
            _immutable_vectors(
                delta_momenta_kg_m_per_s, count, "delta_momenta_kg_m_per_s"
            ),
        )
        object.__setattr__(self, "unit_system_id", _require_si(unit_system_id))
        object.__setattr__(
            self,
            "model_fingerprint",
            _require_fingerprint(model_fingerprint, "model_fingerprint"),
        )

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return the deterministic tangent serialization payload."""

        return {
            "delta_momenta_kg_m_per_s_hex": _rows_payload(
                self.delta_momenta_kg_m_per_s
            ),
            "delta_positions_m_hex": _rows_payload(self.delta_positions_m),
            "layout_fingerprint": self.layout.fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "schema": INERTIAL_TANGENT_SCHEMA,
            "unit_system_id": self.unit_system_id,
        }

    def canonical_bytes(self) -> bytes:
        """Serialize this tangent deterministically."""

        return canonical_json_bytes(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 identity of exact tangent bytes."""

        return sha256_hex(self.canonical_bytes())


@dataclass(frozen=True, init=False)
class JacobiTransformPlan:
    """Immutable fixed-mass coefficients and provenance for Jacobi transforms."""

    layout: CompiledLayout
    body_ids: Tuple[BodyId, ...]
    layout_fingerprint: str
    model_fingerprint: str
    masses_kg: Tuple[float, ...]
    cumulative_masses_kg: Tuple[float, ...]
    inner_mass_fractions: Tuple[float, ...]
    body_mass_fractions: Tuple[float, ...]
    schema: str
    coordinate_convention: str
    axis_order: str
    phase_order: str
    fingerprint: str

    def __init__(self, model: PhysicalModel) -> None:
        if not isinstance(model, PhysicalModel):
            raise InvalidModel("Jacobi transform plan requires a PhysicalModel")
        layout = model.layout
        if layout.central_body != layout.body_ids[0]:
            raise InvalidModel("canonical Jacobi layout requires the central body first")
        if model.units.identifier != "si_v1":
            raise InvalidModel("canonical Jacobi transform plan requires SI units")
        masses = tuple(model.mass_kg(body_id) for body_id in layout.body_ids)
        cumulative = []
        running = 0.0
        for mass in masses:
            running = running + mass
            if not math.isfinite(running) or running <= 0.0:
                raise InvalidModel("cumulative masses must remain finite and positive")
            cumulative.append(running)
        inner_fractions = [1.0]
        body_fractions = [1.0]
        for index in range(1, len(masses)):
            inner_fractions.append(cumulative[index - 1] / cumulative[index])
            body_fractions.append(masses[index] / cumulative[index])

        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "body_ids", tuple(layout.body_ids))
        object.__setattr__(self, "layout_fingerprint", layout.fingerprint)
        object.__setattr__(self, "model_fingerprint", model.fingerprint)
        object.__setattr__(self, "masses_kg", masses)
        object.__setattr__(self, "cumulative_masses_kg", tuple(cumulative))
        object.__setattr__(self, "inner_mass_fractions", tuple(inner_fractions))
        object.__setattr__(self, "body_mass_fractions", tuple(body_fractions))
        object.__setattr__(self, "schema", PLAN_SCHEMA)
        object.__setattr__(self, "coordinate_convention", COORDINATE_CONVENTION)
        object.__setattr__(self, "axis_order", AXIS_ORDER)
        object.__setattr__(self, "phase_order", PHASE_ORDER)
        object.__setattr__(self, "fingerprint", sha256_hex(self.canonical_bytes()))

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return every material coefficient and convention deterministically."""

        return {
            "axis_order": self.axis_order,
            "body_ids": [body.value for body in self.body_ids],
            "body_mass_fractions_hex": [
                finite_binary64_hex(value, "body mass fraction")
                for value in self.body_mass_fractions
            ],
            "coordinate_convention": self.coordinate_convention,
            "cumulative_masses_kg_hex": [
                finite_binary64_hex(value, "cumulative mass")
                for value in self.cumulative_masses_kg
            ],
            "inner_mass_fractions_hex": [
                finite_binary64_hex(value, "inner mass fraction")
                for value in self.inner_mass_fractions
            ],
            "layout_fingerprint": self.layout_fingerprint,
            "masses_kg_hex": [
                finite_binary64_hex(value, "mass") for value in self.masses_kg
            ],
            "model_fingerprint": self.model_fingerprint,
            "phase_order": self.phase_order,
            "schema": self.schema,
            "unit_system_id": "si_v1",
        }

    def canonical_bytes(self) -> bytes:
        """Serialize the frozen plan without including its derived fingerprint."""

        return canonical_json_bytes(self.canonical_payload())


def build_jacobi_transform_plan(model: PhysicalModel) -> JacobiTransformPlan:
    """Build one immutable fixed-mass transform plan from an exact model."""

    return JacobiTransformPlan(model)


def _require_plan_model(plan: JacobiTransformPlan, model: PhysicalModel) -> None:
    if not isinstance(plan, JacobiTransformPlan):
        raise InvalidModel("plan must be JacobiTransformPlan")
    if not isinstance(model, PhysicalModel):
        raise InvalidModel("model must be PhysicalModel")
    if plan.model_fingerprint != model.fingerprint:
        raise LayoutMismatch("transform plan and model fingerprints differ")
    if plan.layout != model.layout or plan.layout_fingerprint != model.layout.fingerprint:
        raise LayoutMismatch("transform plan and model layouts differ")


def _require_state_identity(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    layout: CompiledLayout,
    unit_system_id: str,
    model_fingerprint: str | None = None,
) -> None:
    _require_plan_model(plan, model)
    if layout != plan.layout or layout.fingerprint != plan.layout_fingerprint:
        raise LayoutMismatch("state body identities or order differ from transform plan")
    if unit_system_id != "si_v1" or unit_system_id != model.units.identifier:
        raise LayoutMismatch("state, plan, and model unit systems differ")
    if model_fingerprint is not None and model_fingerprint != plan.model_fingerprint:
        raise LayoutMismatch("state and transform plan model fingerprints differ")


def _forward_position_rows(plan: JacobiTransformPlan, rows: VectorRows) -> VectorRows:
    count = len(rows)
    output = [[0.0, 0.0, 0.0] for _ in range(count)]
    weighted_prefix = [plan.masses_kg[0] * rows[0][axis] for axis in range(3)]
    for index in range(1, count):
        eta_previous = plan.cumulative_masses_kg[index - 1]
        for axis in range(3):
            output[index][axis] = rows[index][axis] - weighted_prefix[axis] / eta_previous
            weighted_prefix[axis] = (
                weighted_prefix[axis] + plan.masses_kg[index] * rows[index][axis]
            )
    eta_total = plan.cumulative_masses_kg[-1]
    for axis in range(3):
        output[0][axis] = weighted_prefix[axis] / eta_total
    return tuple(tuple(row) for row in output)  # type: ignore[return-value]


def _forward_momentum_rows(plan: JacobiTransformPlan, rows: VectorRows) -> VectorRows:
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


def _inverse_momentum_rows(plan: JacobiTransformPlan, rows: VectorRows) -> VectorRows:
    count = len(rows)
    output = [[0.0, 0.0, 0.0] for _ in range(count)]
    prefix = [rows[0][axis] for axis in range(3)]
    for index in range(count - 1, 0, -1):
        alpha = plan.inner_mass_fractions[index]
        beta = plan.body_mass_fractions[index]
        for axis in range(3):
            output[index][axis] = rows[index][axis] + beta * prefix[axis]
            prefix[axis] = alpha * prefix[axis] - rows[index][axis]
    output[0] = prefix
    return tuple(tuple(row) for row in output)  # type: ignore[return-value]


def to_canonical_jacobi(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    state: InertialCanonicalState,
) -> CanonicalJacobiState:
    """Transform inertial canonical ``(x,p)`` to Jacobi ``(q,P)`` in O(N)."""

    if not isinstance(state, InertialCanonicalState):
        raise InvalidState("forward Jacobi transform requires InertialCanonicalState")
    _require_state_identity(
        plan, model, state.layout, state.unit_system_id, state.model_fingerprint
    )
    return CanonicalJacobiState(
        layout=plan.layout,
        q_m=_forward_position_rows(plan, state.positions_m),
        p_kg_m_per_s=_forward_momentum_rows(plan, state.momenta_kg_m_per_s),
        unit_system_id="si_v1",
    )


def from_canonical_jacobi(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    state: CanonicalJacobiState,
) -> InertialCanonicalState:
    """Apply the algebraic inverse Jacobi transform in O(N)."""

    if not isinstance(state, CanonicalJacobiState):
        raise InvalidState("inverse Jacobi transform requires CanonicalJacobiState")
    _require_state_identity(plan, model, state.layout, state.unit_system_id)
    return InertialCanonicalState(
        layout=plan.layout,
        positions_m=_inverse_position_rows(plan, state.q_m),
        momenta_kg_m_per_s=_inverse_momentum_rows(plan, state.p_kg_m_per_s),
        unit_system_id="si_v1",
        model_fingerprint=plan.model_fingerprint,
    )


def to_canonical_jacobi_tangent(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    base_state: InertialCanonicalState,
    tangent: InertialCanonicalTangentState,
) -> CanonicalJacobiTangentState:
    """Apply the constant forward canonical operator to a tangent direction."""

    if not isinstance(base_state, InertialCanonicalState):
        raise InvalidState("forward tangent transform requires inertial canonical base")
    if not isinstance(tangent, InertialCanonicalTangentState):
        raise InvalidState("forward tangent transform requires inertial canonical tangent")
    _require_state_identity(
        plan,
        model,
        base_state.layout,
        base_state.unit_system_id,
        base_state.model_fingerprint,
    )
    _require_state_identity(
        plan,
        model,
        tangent.layout,
        tangent.unit_system_id,
        tangent.model_fingerprint,
    )
    return CanonicalJacobiTangentState(
        layout=plan.layout,
        delta_q_m=_forward_position_rows(plan, tangent.delta_positions_m),
        delta_p_kg_m_per_s=_forward_momentum_rows(
            plan, tangent.delta_momenta_kg_m_per_s
        ),
        unit_system_id="si_v1",
    )


def from_canonical_jacobi_tangent(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    base_state: CanonicalJacobiState,
    tangent: CanonicalJacobiTangentState,
) -> InertialCanonicalTangentState:
    """Apply the constant inverse canonical operator to a tangent direction."""

    if not isinstance(base_state, CanonicalJacobiState):
        raise InvalidState("inverse tangent transform requires canonical Jacobi base")
    if not isinstance(tangent, CanonicalJacobiTangentState):
        raise InvalidState("inverse tangent transform requires canonical Jacobi tangent")
    _require_state_identity(plan, model, base_state.layout, base_state.unit_system_id)
    _require_state_identity(plan, model, tangent.layout, tangent.unit_system_id)
    return InertialCanonicalTangentState(
        layout=plan.layout,
        delta_positions_m=_inverse_position_rows(plan, tangent.delta_q_m),
        delta_momenta_kg_m_per_s=_inverse_momentum_rows(
            plan, tangent.delta_p_kg_m_per_s
        ),
        unit_system_id="si_v1",
        model_fingerprint=plan.model_fingerprint,
    )


def canonical_jacobi_state_bytes(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    state: CanonicalJacobiState,
) -> bytes:
    """Serialize one plan-bound canonical state deterministically."""

    if not isinstance(state, CanonicalJacobiState):
        raise InvalidState("canonical state serialization requires CanonicalJacobiState")
    _require_state_identity(plan, model, state.layout, state.unit_system_id)
    return canonical_json_bytes(
        {
            "layout_fingerprint": plan.layout_fingerprint,
            "model_fingerprint": plan.model_fingerprint,
            "p_kg_m_per_s_hex": _rows_payload(state.p_kg_m_per_s),
            "plan_fingerprint": plan.fingerprint,
            "q_m_hex": _rows_payload(state.q_m),
            "schema": "v2.plan_bound_canonical_jacobi_state/1",
            "unit_system_id": state.unit_system_id,
        }
    )


def canonical_jacobi_tangent_bytes(
    plan: JacobiTransformPlan,
    model: PhysicalModel,
    tangent: CanonicalJacobiTangentState,
) -> bytes:
    """Serialize one plan-bound canonical tangent deterministically."""

    if not isinstance(tangent, CanonicalJacobiTangentState):
        raise InvalidState("tangent serialization requires CanonicalJacobiTangentState")
    _require_state_identity(plan, model, tangent.layout, tangent.unit_system_id)
    return canonical_json_bytes(
        {
            "delta_p_kg_m_per_s_hex": _rows_payload(tangent.delta_p_kg_m_per_s),
            "delta_q_m_hex": _rows_payload(tangent.delta_q_m),
            "layout_fingerprint": plan.layout_fingerprint,
            "model_fingerprint": plan.model_fingerprint,
            "plan_fingerprint": plan.fingerprint,
            "schema": "v2.plan_bound_canonical_jacobi_tangent/1",
            "unit_system_id": tangent.unit_system_id,
        }
    )
