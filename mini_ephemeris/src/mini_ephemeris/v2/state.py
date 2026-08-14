"""Semantically distinct immutable public state contracts for v2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence, Tuple

from .errors import InvalidState, LayoutMismatch
from .model import CompiledLayout
from .timebase import ControlTime


Vector3 = Tuple[float, float, float]
VectorRows = Tuple[Vector3, ...]


def _immutable_vectors(
    values: Iterable[Sequence[float]], expected_rows: int, field: str
) -> VectorRows:
    rows = []
    for raw_row in values:
        components = tuple(raw_row)
        if any(isinstance(component, bool) or not isinstance(component, Real) for component in components):
            raise InvalidState(f"{field} components must be real numbers, not coercible values")
        row = tuple(float(component) for component in components)
        if len(row) != 3:
            raise InvalidState(f"{field} rows must each have exactly three components")
        if not all(math.isfinite(component) for component in row):
            raise InvalidState(f"{field} must contain only finite values")
        rows.append(row)
    if len(rows) != expected_rows:
        raise InvalidState(f"{field} row count does not match the explicit body layout")
    return tuple(rows)


def _unit_id(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InvalidState("unit_system_id must be a nonempty trimmed string")
    if value != "si_v1":
        raise InvalidState("these concrete public state types require unit_system_id='si_v1'")
    return value


@dataclass(frozen=True, init=False)
class InertialCartesianState:
    """Immutable inertial Cartesian `(x,v)` state in declared public units.

    Positions and velocities share the explicit body layout. The tuple-backed
    representation is detached and may cross a canonical serialization boundary.
    It is not a canonical `(q,p)` state.
    """

    layout: CompiledLayout
    positions_m: VectorRows
    velocities_m_per_s: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        positions_m: Iterable[Sequence[float]],
        velocities_m_per_s: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("state layout must be a CompiledLayout")
        count = len(layout.body_ids)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "positions_m", _immutable_vectors(positions_m, count, "positions_m"))
        object.__setattr__(
            self,
            "velocities_m_per_s",
            _immutable_vectors(velocities_m_per_s, count, "velocities_m_per_s"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


@dataclass(frozen=True, init=False)
class CanonicalJacobiState:
    """Immutable canonical Jacobi `(q,p)` state.

    `p` is canonical momentum in kg*m/s, never velocity. Construction performs
    no coordinate or velocity-to-momentum conversion. The value may cross a
    serialization boundary after an external schema supplies exact encoding.
    """

    layout: CompiledLayout
    q_m: VectorRows
    p_kg_m_per_s: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        q_m: Iterable[Sequence[float]],
        p_kg_m_per_s: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("state layout must be a CompiledLayout")
        count = len(layout.body_ids)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "q_m", _immutable_vectors(q_m, count, "q_m"))
        object.__setattr__(
            self,
            "p_kg_m_per_s",
            _immutable_vectors(p_kg_m_per_s, count, "p_kg_m_per_s"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


@dataclass(frozen=True, init=False)
class CanonicalJacobiTangentState:
    """Immutable canonical Jacobi tangent `(delta_q,delta_p)`.

    The tangent uses the exact physical state's body ordering and canonical
    momentum convention. It contains no normalization, rescaling, or MEGNO state.
    """

    layout: CompiledLayout
    delta_q_m: VectorRows
    delta_p_kg_m_per_s: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        delta_q_m: Iterable[Sequence[float]],
        delta_p_kg_m_per_s: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("tangent layout must be a CompiledLayout")
        count = len(layout.body_ids)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "delta_q_m", _immutable_vectors(delta_q_m, count, "delta_q_m"))
        object.__setattr__(
            self,
            "delta_p_kg_m_per_s",
            _immutable_vectors(delta_p_kg_m_per_s, count, "delta_p_kg_m_per_s"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


def require_canonical_tangent_compatible(
    state: CanonicalJacobiState,
    tangent: CanonicalJacobiTangentState,
) -> None:
    """Reject a canonical state/tangent pair with different layout or units."""

    if state.layout != tangent.layout:
        raise LayoutMismatch("canonical state and tangent body layouts differ")
    if state.unit_system_id != tangent.unit_system_id:
        raise LayoutMismatch("canonical state and tangent unit systems differ")

@dataclass(frozen=True, init=False)
class CartesianPositionTangent:
    """Immutable inertial Cartesian position direction used by a force JVP."""

    layout: CompiledLayout
    delta_positions_m: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        delta_positions_m: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("direction layout must be a CompiledLayout")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "delta_positions_m",
            _immutable_vectors(delta_positions_m, len(layout.body_ids), "delta_positions_m"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


@dataclass(frozen=True, init=False)
class CartesianAcceleration:
    """Immutable force-evaluation result in inertial Cartesian acceleration."""

    layout: CompiledLayout
    values_m_per_s2: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        values_m_per_s2: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("acceleration layout must be a CompiledLayout")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "values_m_per_s2",
            _immutable_vectors(values_m_per_s2, len(layout.body_ids), "values_m_per_s2"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


@dataclass(frozen=True, init=False)
class CartesianAccelerationJVP:
    """Immutable JVP result `delta_a=J_a(x) delta_x` in Cartesian units."""

    layout: CompiledLayout
    values_m_per_s2: VectorRows
    unit_system_id: str

    def __init__(
        self,
        layout: CompiledLayout,
        values_m_per_s2: Iterable[Sequence[float]],
        unit_system_id: str,
    ) -> None:
        if not isinstance(layout, CompiledLayout):
            raise InvalidState("JVP layout must be a CompiledLayout")
        object.__setattr__(self, "layout", layout)
        object.__setattr__(
            self,
            "values_m_per_s2",
            _immutable_vectors(values_m_per_s2, len(layout.body_ids), "values_m_per_s2"),
        )
        object.__setattr__(self, "unit_system_id", _unit_id(unit_system_id))


@dataclass(frozen=True, init=False)
class ObserverSnapshot:
    """Detached immutable observer input at one exact macro-step identity.

    The snapshot owns normalized tuple storage and may cross a serialization
    boundary. It contains no live-map handle, mutable buffer, callback, or counter.
    """

    time: ControlTime
    state: InertialCartesianState
    metadata: Tuple[Tuple[str, str], ...]

    def __init__(
        self,
        time: ControlTime,
        state: InertialCartesianState,
        metadata: Mapping[str, str],
    ) -> None:
        if not isinstance(time, ControlTime):
            raise InvalidState("snapshot time must be a ControlTime")
        if not isinstance(state, InertialCartesianState):
            raise InvalidState("snapshot state must be inertial Cartesian")
        items = tuple(metadata.items())
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key
            or not value
            or key.strip() != key
            or value.strip() != value
            for key, value in items
        ):
            raise InvalidState("snapshot metadata keys and values must be nonempty trimmed strings")
        normalized = tuple(sorted(items))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "metadata", normalized)
