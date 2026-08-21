#!/usr/bin/env python3
"""Break the estimator on purpose; check the ladder notices.

A ladder that has only ever been seen to pass is not evidence. Each mutation
below restores a defect this project actually shipped, or a near neighbour of
one. The ladder must go red for every single one. A mutation that SURVIVES is a
hole in the ladder, and is reported as a failure of this script.

    python3 scripts/mutation_test_ladder.py
    python3 scripts/mutation_test_ladder.py --keep    # leave the trees for inspection

Note the difference between a surviving mutation and an inert one. Setting
CHAOTIC_RATIO_MIN to 0 looks like it should make everything chaotic, but the
classifier tests `ratio <= REGULAR_RATIO_MAX` first, so for the systems in
rungs 1-2 that branch is never reached and the mutation changes no observable
behaviour. That is an invalid mutation, not a hole. Where a mutation is inert,
say so rather than counting it as a pass.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIAG = "mini_ephemeris/src/mini_ephemeris/chaos_estimator_diagnostics.py"
RUNGS = "mini_ephemeris/src/mini_ephemeris/ladder_rungs.py"

# (id, what defect this restores, file, find, replace)
MUTATIONS = (
    ("M1", "MEGNO factor back to the original 0.5", DIAG,
     "MEGNO_MEAN_TO_LYAPUNOV = 2.0", "MEGNO_MEAN_TO_LYAPUNOV = 0.5"),
    ("M2", "drop the factor 2 in the MEGNO reconstruction", DIAG,
     "y_inst[1:] = 2.0 * (log_norm[1:] - integral_log[1:] / t[1:])",
     "y_inst[1:] = 1.0 * (log_norm[1:] - integral_log[1:] / t[1:])"),
    ("M3", "widen the regular band so chaos reads as regular", DIAG,
     "REGULAR_RATIO_MAX = 0.70", "REGULAR_RATIO_MAX = 1.10"),
    ("M4", "narrow the regular band so regular motion reads as ambiguous", DIAG,
     "REGULAR_RATIO_MAX = 0.70", "REGULAR_RATIO_MAX = 0.20"),
    ("M5", "report the line-fit slope as the exponent (the original bug)", RUNGS,
     "measured=long_result.halving_ratio,", "measured=long_result.line_fit_lambda,"),
    ("M6", "forget to renormalise the tangent vector", RUNGS,
     "        tangent = (tangent[0] / norm, tangent[1] / norm)",
     "        tangent = (tangent[0], tangent[1])"),
    ("M7", "accumulate log of the squared norm (factor-2 slip)", RUNGS,
     "        cumulative += math.log(norm)\n        tangent = (tangent[0] / norm",
     "        cumulative += math.log(norm * norm)\n        tangent = (tangent[0] / norm"),
    ("M8", "stale variational acceleration across renormalisation", RUNGS,
     "            dvx *= scale\n            dvy *= scale",
     "            dvx *= 1.0\n            dvy *= 1.0"),
)

STATUS = re.compile(r"OVERALL[^:]*:\s*(\w+)")
RED = re.compile(r"^  (FAIL|ERROR)\s+(\S+)", re.MULTILINE)


def run_one(mutation, keep: bool) -> tuple[str, str, str]:
    ident, description, relative, find, replace = mutation
    workdir = Path(tempfile.mkdtemp(prefix=f"ladder-mut-{ident}-"))
    tree = workdir / "repo"
    shutil.copytree(REPO, tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = tree / relative
    source = target.read_text()
    if find not in source:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)
        return "NOT APPLIED", "", "pattern not found -- the mutation is stale"
    target.write_text(source.replace(find, replace, 1))

    completed = subprocess.run(
        [sys.executable, "scripts/run_validation_ladder.py", "--continue-on-failure"],
        cwd=tree, capture_output=True, text=True, timeout=900,
    )
    output = completed.stdout + completed.stderr
    match = STATUS.search(output)
    status = match.group(1) if match else "UNPARSED"
    red = " ".join(rung for _kind, rung in RED.findall(output)) or "none"
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return status, red, str(tree) if keep else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    print()
    print("MUTATION TESTS -- the ladder must go red for every one")
    print()
    survivors: list[str] = []
    for mutation in MUTATIONS:
        ident, description = mutation[0], mutation[1]
        status, red, where = run_one(mutation, args.keep)
        caught = status in ("FAIL", "ERROR")
        marker = "caught " if caught else "SURVIVED"
        print(f"  {marker}  {ident}  {description}")
        print(f"            overall={status}   red rungs: {red}")
        if where:
            print(f"            tree: {where}")
        if not caught:
            survivors.append(f"{ident}: {description}")

    print()
    if survivors:
        print(f"  {len(survivors)} mutation(s) SURVIVED. Each is a hole in the ladder:")
        for line in survivors:
            print(f"    - {line}")
        print()
        print("  A green ladder does not certify an estimator it cannot break.")
        print()
        return 1
    print(f"  All {len(MUTATIONS)} mutations were caught.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
