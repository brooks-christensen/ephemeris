#!/usr/bin/env python3
"""Run the validation ladder.

    python3 scripts/run_validation_ladder.py                 # rungs 0-2
    python3 scripts/run_validation_ladder.py --json out.json
    python3 scripts/run_validation_ladder.py --self-test     # prove FAIL works

Rungs must pass in order; the ladder halts at the first failure and the rungs
after it are reported NOT_RUN rather than being quietly omitted.

Run --self-test before trusting a green run. A verification tool that has never
been shown to fail is a green light you painted yourself: the whole reason this
ladder exists is that the previous certification path stamped PASS on 124 nodes
without running anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_ephemeris" / "src"))

from mini_ephemeris.ladder_rungs import (  # noqa: E402
    PENDING_RUNGS,
    rung0_unit_tests,
    rung1_integrable_two_body,
    rung2a_cat_map,
    rung2b_standard_map,
    rung2c_known_nbody_exponent,
)
from mini_ephemeris.validation_ladder import (  # noqa: E402
    RungResult,
    RungStatus,
    evaluate_rung,
    run_ladder,
)

RUNGS = (
    ("0", "unit tests actually executed", rung0_unit_tests),
    ("1", "integrable two-body reads as regular", rung1_integrable_two_body),
    ("2a", "cat map recovers its exact Lyapunov exponent", rung2a_cat_map),
    ("2b", "standard map recovers ln(K/2)", rung2b_standard_map),
    ("2c", "known N-body exponent recovered", rung2c_known_nbody_exponent),
)


def self_test() -> int:
    """Show that failure is reachable, halts the ladder, and cannot be faked."""

    checks: list[tuple[str, bool]] = []

    def good() -> RungResult:
        return evaluate_rung("A", "correct measurement", measured=1.0,
                             target=1.0, acceptance=(0.9, 1.1))

    def wrong() -> RungResult:
        return evaluate_rung("B", "measurement misses its window", measured=5.0,
                             target=1.0, acceptance=(0.9, 1.1))

    def explodes() -> RungResult:
        raise RuntimeError("deliberate failure inside a rung")

    def not_finite() -> RungResult:
        return evaluate_rung("D", "non-finite measurement", measured=float("nan"),
                             target=1.0, acceptance=(0.9, 1.1))

    report = run_ladder((("A", "good", good), ("B", "wrong", wrong),
                         ("C", "explodes", explodes)))
    statuses = [r.status for r in report.results]
    checks.append(("a correct rung passes", statuses[0] is RungStatus.PASS))
    checks.append(("a wrong measurement FAILs", statuses[1] is RungStatus.FAIL))
    checks.append(("the ladder halts after a failure", statuses[2] is RungStatus.NOT_RUN))
    checks.append(("overall status is FAIL", report.overall_status is RungStatus.FAIL))
    checks.append(("the halt point is recorded", report.halted_at == "B"))

    report_error = run_ladder((("C", "explodes", explodes),))
    checks.append(("a raising rung becomes ERROR",
                   report_error.results[0].status is RungStatus.ERROR))
    checks.append(("an ERROR is not a pass",
                   report_error.overall_status is not RungStatus.PASS))

    checks.append(("a NaN measurement FAILs", not_finite().status is RungStatus.FAIL))

    try:
        RungResult(rung="X", name="stamped pass", status=RungStatus.PASS,
                   measured=5.0, acceptance=(0.9, 1.1))
        stamped_rejected = False
    except ValueError:
        stamped_rejected = True
    checks.append(("a PASS outside its window cannot be constructed", stamped_rejected))

    try:
        RungResult(rung="Y", name="empty pass", status=RungStatus.PASS)
        empty_rejected = False
    except ValueError:
        empty_rejected = True
    checks.append(("a PASS with nothing measured cannot be constructed", empty_rejected))

    try:
        RungResult(rung="Z", name="unmet condition", status=RungStatus.PASS,
                   measured=1.0, acceptance=(0.9, 1.1),
                   conditions=(("something required", False),))
        condition_rejected = False
    except ValueError:
        condition_rejected = True
    checks.append(("a PASS with an unmet condition cannot be constructed",
                   condition_rejected))

    print()
    print("LADDER SELF-TEST")
    print()
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'BAD '}  {label}")
    print()
    if all(ok for _, ok in checks):
        print("  Self-test passed: failures are reachable, halting works, and a")
        print("  PASS cannot be written without a comparison that could have failed.")
        print()
        return 0
    print("  SELF-TEST FAILED. Results from this ladder cannot be trusted.")
    print()
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the machine-readable report here")
    parser.add_argument("--self-test", action="store_true",
                        help="prove the ladder can report failure, then exit")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="run every rung even after one fails (diagnostic only)")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    report = run_ladder(
        RUNGS,
        halt_on_failure=not args.continue_on_failure,
        scope="rungs 0-2 only",
        not_implemented=PENDING_RUNGS,
    )
    print(report.render())
    for result in report.results:
        if result.evidence:
            print(f"  rung {result.rung} evidence:")
            for key, value in result.evidence.items():
                print(f"    {key}: {value}")
            print()
    if args.json is not None:
        args.json.write_text(report.to_json())
        print(f"  wrote {args.json}")
    return 0 if report.overall_status is RungStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
