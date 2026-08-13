from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .m0_step3f1_contract import DEFAULT_MANIFEST
from .m0_step3f1_runner import audit, restart_check, run_lane, zero_step
from .m0_step3f1_workflow import analyze, verify


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the preregistered M0 Step 3f1 architecture screen.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("zero-step")
    run = subparsers.add_parser("run-lane")
    run.add_argument("--lane", choices=("P", "T"), required=True)
    restart = subparsers.add_parser("restart-check")
    restart.add_argument("--lane", choices=("P", "T"), required=True)
    subparsers.add_parser("analyze")
    subparsers.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = args.manifest.resolve()
    if args.command == "audit":
        payload = audit(manifest)
    elif args.command == "zero-step":
        payload = zero_step(manifest)
    elif args.command == "run-lane":
        payload = run_lane(manifest, args.lane)
    elif args.command == "restart-check":
        payload = restart_check(manifest, args.lane)
    elif args.command == "analyze":
        payload = analyze(manifest)
    else:
        payload = verify(manifest)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
