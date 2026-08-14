"""Pure semantic force and JVP interfaces; no physical provider is included."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .accounting import AccountingDomain
from .errors import KernelContractError, LayoutMismatch
from .model import PhysicalModel
from .state import (
    CartesianAcceleration,
    CartesianAccelerationJVP,
    CartesianPositionTangent,
    InertialCartesianState,
)
from .timebase import ControlTime


@dataclass(frozen=True)
class ForceEvaluationContext:
    """Immutable semantic context supplied explicitly to a pure kernel.

    The context labels purpose but contains no mutable counter, observer callback,
    synchronization handle, output history, or global registration.
    """

    time: ControlTime
    domain: AccountingDomain
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.time, ControlTime):
            raise KernelContractError("context time must be ControlTime")
        if not isinstance(self.domain, AccountingDomain):
            raise KernelContractError("context domain must be AccountingDomain")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise KernelContractError("request_id must be a nonempty string")


class ForceProvider(Protocol):
    """Backend-independent deterministic Cartesian acceleration provider.

    Implementations must not mutate any input, synchronize a map, invoke an
    observer, update counters, inspect output history, or use mutable globals.
    """

    def evaluate(
        self,
        model: PhysicalModel,
        state: InertialCartesianState,
        context: ForceEvaluationContext,
    ) -> CartesianAcceleration:
        """Return acceleration for the exact model, state, and control context."""


class JVPProvider(Protocol):
    """Backend-independent first directional derivative of a force provider.

    For fixed model, state, and context, output must be linear in `direction`.
    No force value is implied or substituted by this distinct operation.
    """

    def jvp(
        self,
        model: PhysicalModel,
        state: InertialCartesianState,
        direction: CartesianPositionTangent,
        context: ForceEvaluationContext,
    ) -> CartesianAccelerationJVP:
        """Return `J_a(x) delta_x` without mutating or observing any input."""


def _check_state_model(model: PhysicalModel, state: InertialCartesianState) -> None:
    if model.layout != state.layout:
        raise LayoutMismatch("model and state body layouts differ")
    if model.units.identifier != state.unit_system_id:
        raise LayoutMismatch("model and state unit systems differ")


def evaluate_force(
    provider: ForceProvider,
    model: PhysicalModel,
    state: InertialCartesianState,
    context: ForceEvaluationContext,
) -> CartesianAcceleration:
    """Validate and invoke the immutable semantic force boundary.

    The immutable return type is intentionally distinct from a future private
    caller-owned-buffer `evaluate_into` hot-loop API.
    """

    _check_state_model(model, state)
    result = provider.evaluate(model, state, context)
    if not isinstance(result, CartesianAcceleration):
        raise KernelContractError("force provider returned the wrong semantic type")
    if result.layout != model.layout or result.unit_system_id != model.units.identifier:
        raise KernelContractError("force result layout or units do not match the model")
    return result


def evaluate_jvp(
    provider: JVPProvider,
    model: PhysicalModel,
    state: InertialCartesianState,
    direction: CartesianPositionTangent,
    context: ForceEvaluationContext,
) -> CartesianAccelerationJVP:
    """Validate and invoke the distinct pure force-JVP semantic boundary."""

    _check_state_model(model, state)
    if direction.layout != model.layout:
        raise LayoutMismatch("model and JVP direction body layouts differ")
    if direction.unit_system_id != model.units.identifier:
        raise LayoutMismatch("model and JVP direction unit systems differ")
    result = provider.jvp(model, state, direction, context)
    if not isinstance(result, CartesianAccelerationJVP):
        raise KernelContractError("JVP provider returned the wrong semantic type")
    if result.layout != model.layout or result.unit_system_id != model.units.identifier:
        raise KernelContractError("JVP result layout or units do not match the model")
    return result
