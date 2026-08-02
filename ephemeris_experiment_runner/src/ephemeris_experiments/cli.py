from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .manifest import ManifestError, load_manifest
from .runner import RunnerError, run_experiment
from .state import load_approvals, load_state, save_approvals
from .validators import run_gates


def _state_paths(spec):
    state_dir = Path(spec.state_dir).expanduser().resolve()
    return state_dir, state_dir / "state.json", state_dir / "approvals.json"


def cmd_plan(args: argparse.Namespace) -> int:
    spec = load_manifest(args.manifest)
    print(f"EXPERIMENT: {spec.title}")
    print(spec.description)
    print(f"state: {spec.state_dir}")
    for index, stage in enumerate(spec.stages, start=1):
        approval = " [manual approval before start]" if stage.approval_required_before else ""
        print(f"\n{index}. {stage.stage_id}: {stage.title}{approval}")
        print(f"   objective: {stage.objective}")
        print(f"   target: {stage.target_years if stage.target_years is not None else 'n/a'} years")
        print(f"   depends on: {', '.join(stage.depends_on) if stage.depends_on else 'none'}")
        print("   command: " + " ".join(stage.command))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_manifest(args.manifest)
    return run_experiment(spec, resume=args.resume, dry_run=args.dry_run)


def cmd_status(args: argparse.Namespace) -> int:
    spec = load_manifest(args.manifest)
    _, state_path, _ = _state_paths(spec)
    while True:
        state = load_state(state_path)
        print(json.dumps(state, indent=2, sort_keys=True))
        if not args.watch:
            return 0
        print("\n--- refresh ---\n")
        time.sleep(args.interval)


def cmd_approve(args: argparse.Namespace) -> int:
    spec = load_manifest(args.manifest)
    valid = {stage.stage_id for stage in spec.stages}
    if args.stage_id not in valid:
        print(f"Unknown stage id: {args.stage_id}", file=sys.stderr)
        return 2
    _, _, approvals_path = _state_paths(spec)
    approvals = load_approvals(approvals_path)
    approvals.add(args.stage_id)
    save_approvals(approvals_path, approvals)
    print(f"Approved stage {args.stage_id}. This records approval only; it does not start the run.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_manifest(args.manifest)
    selected = [stage for stage in spec.stages if args.stage_id in (None, stage.stage_id)]
    if not selected:
        print(f"Unknown stage id: {args.stage_id}", file=sys.stderr)
        return 2
    failed = False
    for stage in selected:
        print(f"\nValidating {stage.stage_id}")
        results = run_gates(stage.gates, stage.cwd)
        for result in results:
            print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}: {result.detail}")
            failed |= not result.passed
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ephem-exp")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Print the complete staged experiment plan")
    plan.add_argument("manifest")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="Run stages serially with progress and gates")
    run.add_argument("manifest")
    run.add_argument("--resume", action="store_true", help="Skip passed stages and append configured resume args when safe")
    run.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Show runner state")
    status.add_argument("manifest")
    status.add_argument("--watch", action="store_true")
    status.add_argument("--interval", type=int, default=60)
    status.set_defaults(func=cmd_status)

    approve = sub.add_parser("approve", help="Approve a manually gated stage")
    approve.add_argument("manifest")
    approve.add_argument("stage_id")
    approve.set_defaults(func=cmd_approve)

    validate = sub.add_parser("validate", help="Run postprocessing gates without launching simulations")
    validate.add_argument("manifest")
    validate.add_argument("stage_id", nargs="?")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(args.func(args))
    except (ManifestError, RunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
