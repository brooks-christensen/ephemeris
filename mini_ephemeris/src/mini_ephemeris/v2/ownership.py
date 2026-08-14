"""Detached observer-copy boundary for the v2 foundation."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence, TypeVar

from .model import CompiledLayout
from .state import InertialCartesianState, ObserverSnapshot
from .timebase import ControlTime


ObserverResult = TypeVar("ObserverResult")


def capture_observer_snapshot(
    *,
    layout: CompiledLayout,
    positions_m: Iterable[Sequence[float]],
    velocities_m_per_s: Iterable[Sequence[float]],
    unit_system_id: str,
    time: ControlTime,
    metadata: Mapping[str, str],
) -> ObserverSnapshot:
    """Copy mutable source values into an immutable detached snapshot.

    The future map engine owns any live mutable buffers. This function receives
    only values, normalizes them to tuples, and returns no live handle or alias.
    """

    state = InertialCartesianState(
        layout=layout,
        positions_m=positions_m,
        velocities_m_per_s=velocities_m_per_s,
        unit_system_id=unit_system_id,
    )
    return ObserverSnapshot(time=time, state=state, metadata=metadata)


def observe(
    snapshot: ObserverSnapshot,
    observer: Callable[[ObserverSnapshot], ObserverResult],
) -> ObserverResult:
    """Run an observer with an immutable snapshot as its only state input."""

    if not isinstance(snapshot, ObserverSnapshot):
        raise TypeError("observer input must be an ObserverSnapshot")
    return observer(snapshot)
