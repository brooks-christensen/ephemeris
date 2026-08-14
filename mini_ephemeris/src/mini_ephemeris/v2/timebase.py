"""Exact integer-indexed macro-step control time for the v2 foundation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from .canonical import canonical_json_bytes, sha256_hex
from .errors import InvalidTimebase


MAX_ABS_STEP_INDEX = 2**63 - 1


def _checked_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTimebase(f"{field} must be an integer")
    return value


@dataclass(frozen=True, init=False)
class ExactSeconds:
    """Reduced exact rational seconds suitable for canonical serialization."""

    numerator: int
    denominator: int

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        numerator = _checked_int(numerator, "numerator")
        denominator = _checked_int(denominator, "denominator")
        if denominator == 0:
            raise InvalidTimebase("denominator must not be zero")
        value = Fraction(numerator, denominator)
        object.__setattr__(self, "numerator", value.numerator)
        object.__setattr__(self, "denominator", value.denominator)

    def canonical_payload(self) -> Mapping[str, int]:
        """Return exact numerator and positive denominator."""

        return {"denominator": self.denominator, "numerator": self.numerator}

    @classmethod
    def from_canonical_payload(cls, payload: Mapping[str, Any]) -> "ExactSeconds":
        """Reconstruct exact seconds from a strict two-integer payload."""

        if set(payload) != {"numerator", "denominator"}:
            raise InvalidTimebase("exact-seconds payload has unknown or missing fields")
        return cls(payload["numerator"], payload["denominator"])

    def to_binary64(self) -> float:
        """Convert to binary64 at the explicit numerical boundary."""

        return self.numerator / self.denominator


@dataclass(frozen=True)
class ControlTime:
    """Exact macro-step identity and its exact absolute time in seconds."""

    step_index: int
    seconds: ExactSeconds

    def __post_init__(self) -> None:
        _checked_int(self.step_index, "step_index")
        if not isinstance(self.seconds, ExactSeconds):
            raise InvalidTimebase("seconds must be ExactSeconds")

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return exact step and rational time identity."""

        return {
            "schema": "v2.control_time/1",
            "seconds": self.seconds.canonical_payload(),
            "step_index": self.step_index,
        }


@dataclass(frozen=True)
class MacroTimebase:
    """Derive target times from epoch plus integer index times interval.

    This controls macro-step, observation, and restart identity. It makes no
    claim that internal floating WHCKL stage times are integer-exact.
    """

    epoch: ExactSeconds
    interval: ExactSeconds
    max_abs_step_index: int = MAX_ABS_STEP_INDEX

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, ExactSeconds) or not isinstance(self.interval, ExactSeconds):
            raise InvalidTimebase("epoch and interval must be ExactSeconds")
        if self.interval.numerator <= 0:
            raise InvalidTimebase("macro-step interval must be positive")
        limit = _checked_int(self.max_abs_step_index, "max_abs_step_index")
        if limit < 0:
            raise InvalidTimebase("max_abs_step_index must be nonnegative")

    def at(self, step_index: int) -> ControlTime:
        """Return exact direct-index time without iterative accumulation."""

        index = _checked_int(step_index, "step_index")
        if abs(index) > self.max_abs_step_index:
            raise InvalidTimebase("step_index exceeds the declared deterministic bound")
        epoch = Fraction(self.epoch.numerator, self.epoch.denominator)
        interval = Fraction(self.interval.numerator, self.interval.denominator)
        target = epoch + index * interval
        return ControlTime(index, ExactSeconds(target.numerator, target.denominator))

    def canonical_payload(self) -> Mapping[str, Any]:
        """Return the exact restart-safe timebase payload."""

        return {
            "epoch": self.epoch.canonical_payload(),
            "interval": self.interval.canonical_payload(),
            "max_abs_step_index": self.max_abs_step_index,
            "schema": "v2.macro_timebase/1",
        }

    @classmethod
    def from_canonical_payload(cls, payload: Mapping[str, Any]) -> "MacroTimebase":
        """Reconstruct a timebase from its exact canonical fields."""

        required = {"schema", "epoch", "interval", "max_abs_step_index"}
        if set(payload) != required or payload.get("schema") != "v2.macro_timebase/1":
            raise InvalidTimebase("timebase payload schema or fields are incompatible")
        return cls(
            epoch=ExactSeconds.from_canonical_payload(payload["epoch"]),
            interval=ExactSeconds.from_canonical_payload(payload["interval"]),
            max_abs_step_index=payload["max_abs_step_index"],
        )

    def canonical_bytes(self) -> bytes:
        """Serialize the exact timebase deterministically."""

        return canonical_json_bytes(self.canonical_payload())

    @property
    def fingerprint(self) -> str:
        """Return the exact control-time identity hash."""

        return sha256_hex(self.canonical_bytes())
