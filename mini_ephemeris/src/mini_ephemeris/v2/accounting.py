"""Typed evaluation domains without mutable counters or hidden side effects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccountingDomain(str, Enum):
    """Disjoint future accounting domains fixed by the Step 3g1a contract."""

    MAP_STAGE = "map_stage"
    CORRECTOR_SYNCHRONIZATION = "corrector_synchronization"
    OBSERVER_ONLY = "observer_only"
    SERIALIZATION_RESTART = "serialization_restart"


@dataclass(frozen=True)
class AccountingEvent:
    """Immutable event identity; it does not increment or own a counter."""

    domain: AccountingDomain
    operation: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, AccountingDomain):
            raise TypeError("domain must be an AccountingDomain")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a nonempty string")
