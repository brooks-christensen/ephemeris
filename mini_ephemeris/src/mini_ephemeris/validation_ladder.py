"""Result machinery for the validation ladder.

The defect this module exists to make impossible
---------------------------------------------------
``m0_step3g1d_reporting.py`` builds a 124-node test inventory by transcribing
the manifest's node lists and stamping a literal ``"result": "PASS"`` on each
one. It never invokes pytest. The artifact would be byte-identical if every
test failed, and no module in the project could emit a failure status even if
one had been detected.

So the requirement here is not "add a FAIL constant". It is that a passing
result be *unconstructable* without a comparison that could have gone the other
way. :class:`RungResult` enforces that in ``__post_init__``: a ``PASS`` whose
measured value lies outside its own acceptance window, or whose stated
conditions are not all met, raises ``ValueError`` at construction. Transcribing
a PASS is a crash, not a document.

Design rules
------------
* A rung declares its target and acceptance window **before** it runs. The
  window is data, carried in the result, and appears in the report next to the
  measurement.
* A non-finite or missing measurement is a FAIL, never a skip and never a pass.
* An exception inside a rung is an ERROR, which is a failure, not an absence.
* The ladder halts at the first failure. Rungs after the halt are ``NOT_RUN``
  and say so; they are not counted as passing.
* ``overall_status`` is derived from the results and cannot be set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import time
import traceback
from typing import Callable, Sequence

__all__ = [
    "RungStatus",
    "RungResult",
    "LadderReport",
    "evaluate_rung",
    "run_ladder",
    "LadderRung",
]


class RungStatus(str, Enum):
    """Outcome of one rung. ``PASS`` is the only one that means anything good."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"

    def is_failure(self) -> bool:
        return self in (RungStatus.FAIL, RungStatus.ERROR)


def _within(value: float | None, window: tuple[float, float] | None) -> bool:
    if window is None:
        return True
    if value is None or not math.isfinite(value):
        return False
    low, high = window
    return low <= value <= high


@dataclass(frozen=True)
class RungResult:
    """One rung's outcome, with the evidence that produced it.

    ``status`` is validated against ``measured``, ``acceptance`` and
    ``conditions`` on construction. A PASS that does not survive that check is
    a ``ValueError``.
    """

    rung: str
    name: str
    status: RungStatus
    measured: float | None = None
    target: float | None = None
    acceptance: tuple[float, float] | None = None
    unit: str = ""
    conditions: tuple[tuple[str, bool], ...] = ()
    duration_seconds: float = 0.0
    evidence: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, RungStatus):
            raise TypeError("status must be a RungStatus")
        if self.acceptance is not None:
            low, high = self.acceptance
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"rung {self.rung}: malformed acceptance window")
        if self.status is not RungStatus.PASS:
            return

        # Everything below is the guard against a stamped PASS.
        if self.acceptance is not None and not _within(self.measured, self.acceptance):
            raise ValueError(
                f"rung {self.rung} ({self.name}): cannot be PASS with measured "
                f"{self.measured!r} outside acceptance {self.acceptance!r}"
            )
        if self.acceptance is None and self.measured is None and not self.conditions:
            raise ValueError(
                f"rung {self.rung} ({self.name}): cannot be PASS with nothing "
                "measured and no conditions checked -- a PASS must rest on a "
                "comparison that could have failed"
            )
        unmet = [label for label, ok in self.conditions if not ok]
        if unmet:
            raise ValueError(
                f"rung {self.rung} ({self.name}): cannot be PASS with unmet "
                f"conditions: {unmet}"
            )

    def to_dict(self) -> dict:
        return {
            "rung": self.rung,
            "name": self.name,
            "status": self.status.value,
            "measured": self.measured,
            "target": self.target,
            "acceptance": list(self.acceptance) if self.acceptance else None,
            "unit": self.unit,
            "conditions": [
                {"condition": label, "met": bool(ok)} for label, ok in self.conditions
            ],
            "duration_seconds": round(self.duration_seconds, 3),
            "evidence": self.evidence,
            "notes": list(self.notes),
        }

    def one_line(self) -> str:
        if self.measured is None or not math.isfinite(self.measured):
            value = "     --     "
        else:
            value = f"{self.measured:12.6g}"
        if self.acceptance is None:
            window = ""
        else:
            window = f"  window [{self.acceptance[0]:.6g}, {self.acceptance[1]:.6g}]"
        target = "" if self.target is None else f"  target {self.target:.6g}"
        return (f"  {self.status.value:<8} {self.rung:>3}  {self.name:<44} "
                f"{value} {self.unit}{target}{window}")


def evaluate_rung(
    rung: str,
    name: str,
    *,
    measured: float | None,
    acceptance: tuple[float, float] | None,
    target: float | None = None,
    unit: str = "",
    conditions: Sequence[tuple[str, bool]] = (),
    duration_seconds: float = 0.0,
    evidence: dict | None = None,
    notes: Sequence[str] = (),
) -> RungResult:
    """Decide PASS/FAIL from the measurement. The only way a PASS is produced."""

    conditions = tuple((str(label), bool(ok)) for label, ok in conditions)
    notes = tuple(notes)
    passed = _within(measured, acceptance) and all(ok for _, ok in conditions)
    if acceptance is None and measured is None and not conditions:
        passed = False
        notes = notes + ("no acceptance window and no conditions: cannot pass",)
    if measured is not None and not math.isfinite(measured):
        passed = False
        notes = notes + ("measurement is not finite",)
    return RungResult(
        rung=rung,
        name=name,
        status=RungStatus.PASS if passed else RungStatus.FAIL,
        measured=measured,
        target=target,
        acceptance=acceptance,
        unit=unit,
        conditions=conditions,
        duration_seconds=duration_seconds,
        evidence=evidence or {},
        notes=notes,
    )


LadderRung = Callable[[], RungResult]


@dataclass(frozen=True)
class LadderReport:
    results: tuple[RungResult, ...]
    halted_at: str | None
    total_seconds: float
    scope: str = ""
    not_implemented: tuple[tuple[str, str], ...] = ()

    @property
    def overall_status(self) -> RungStatus:
        if not self.results:
            return RungStatus.FAIL
        for result in self.results:
            if result.status is not RungStatus.PASS:
                return result.status
        return RungStatus.PASS

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "scope": self.scope,
            "not_implemented": [
                {"rung": rung, "name": name} for rung, name in self.not_implemented
            ],
            "halted_at_rung": self.halted_at,
            "total_seconds": round(self.total_seconds, 3),
            "rungs": [result.to_dict() for result in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def render(self) -> str:
        lines = ["", "VALIDATION LADDER", ""]
        lines.extend(result.one_line() for result in self.results)
        for result in self.results:
            if result.notes:
                lines.append("")
                lines.append(f"  rung {result.rung} notes:")
                lines.extend(f"    - {note}" for note in result.notes)
        lines.append("")
        scope = f" ({self.scope})" if self.scope else ""
        lines.append(f"  OVERALL{scope}: {self.overall_status.value}")
        if self.not_implemented:
            lines.append("")
            lines.append("  Declared but NOT implemented in this harness -- a PASS above")
            lines.append("  says nothing about these:")
            for rung, name in self.not_implemented:
                lines.append(f"    {rung:>3}  {name}")
        if self.halted_at is not None:
            lines.append(f"  halted at rung {self.halted_at}; later rungs were not run")
        lines.append("")
        return "\n".join(lines)


def run_ladder(
    rungs: Sequence[tuple[str, str, LadderRung]],
    *,
    halt_on_failure: bool = True,
    scope: str = "",
    not_implemented: Sequence[tuple[str, str]] = (),
) -> LadderReport:
    """Run rungs in order. Halt at the first failure; do not pass what did not run.

    Each entry is ``(rung_id, name, callable)``. The callable returns a
    :class:`RungResult`; anything it raises becomes an ERROR result.
    """

    results: list[RungResult] = []
    halted_at: str | None = None
    started = time.monotonic()

    for rung_id, name, run in rungs:
        if halted_at is not None:
            results.append(
                RungResult(
                    rung=rung_id,
                    name=name,
                    status=RungStatus.NOT_RUN,
                    notes=(f"not attempted: rung {halted_at} failed",),
                )
            )
            continue
        rung_started = time.monotonic()
        try:
            result = run()
            if not isinstance(result, RungResult):
                raise TypeError(
                    f"rung {rung_id} returned {type(result).__name__}, not RungResult"
                )
        except Exception:  # noqa: BLE001 - an exception is a failure, not an absence
            result = RungResult(
                rung=rung_id,
                name=name,
                status=RungStatus.ERROR,
                duration_seconds=time.monotonic() - rung_started,
                notes=tuple(traceback.format_exc().strip().splitlines()[-3:]),
            )
        results.append(result)
        if result.status.is_failure() and halt_on_failure:
            halted_at = rung_id

    return LadderReport(
        results=tuple(results),
        halted_at=halted_at,
        total_seconds=time.monotonic() - started,
        scope=scope,
        not_implemented=tuple(not_implemented),
    )
