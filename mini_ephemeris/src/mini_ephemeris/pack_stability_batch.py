from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


INCLUDE_PATTERNS = (
    "batch_driver.log",
    "*.log",
    "summary_*.json",
    "megno_summary_*.json",
    "shadow_lyapunov_summary_*.json",
    "shadow_separation_*.csv",
    "shadow_growth_*.png",
    "shadow_fit_diagnostics_*.csv",
    "shadow_fit_diagnostics_*.json",
    "shadow_fit_window_comparison_*.png",
    "shadow_metric_scan_*.csv",
    "shadow_metric_scan_*.json",
    "shadow_metric_scan_*.md",
    "rebound_megno*.json",
    "rebound_megno*.csv",
    "rebound_megno*.md",
    "backend_comparison*.csv",
    "backend_comparison*.json",
    "backend_comparison*.md",
    "invariants_*.csv",
    "orbital_elements_*.csv",
    "min_separations_*.csv",
    "megno_*.csv",
    "stability_timeseries_*.csv",
    "*.png",
    "*.sh",
    "*manifest*.csv",
    "*manifest*.tsv",
    "*manifest*.json",
)

ARCHIVE_PATTERNS = ("*.bin", "*.sa", "*.simarchive")
STANDARD_EXPECTED_PATTERNS = (
    "summary_*.json",
    "invariants_*.csv",
    "orbital_elements_*.csv",
    "min_separations_*.csv",
)
SHADOW_EXPECTED_PATTERNS = (
    "shadow_lyapunov_summary_*.json",
    "shadow_separation_*.csv",
    "shadow_growth_*.png",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package stability-mode run artifacts into a reproducibility zip."
    )
    parser.add_argument("--batch-dir", type=Path, default=None, help="Directory containing a batch run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="General stability output directory.")
    parser.add_argument("--tag", default=None, help="Optional tag filter for run artifacts.")
    parser.add_argument("--include-archives", action="store_true", help="Include SimulationArchive .bin/.sa files.")
    parser.add_argument(
        "--include-large-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include large CSV files. Use --no-include-large-csv to exclude them.",
    )
    parser.add_argument("--max-file-mb", type=float, default=None, help="Skip files larger than this size.")
    parser.add_argument("--output-zip", type=Path, default=None, help="Explicit output zip path.")
    return parser


def matches_any(path: Path, patterns: tuple[str, ...]) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def git_commit_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def iter_candidate_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files.append(path)
    return sorted(set(files))


def should_include(path: Path, *, tag: str | None, include_archives: bool, include_large_csv: bool) -> bool:
    if tag and tag not in path.name and "manifest" not in path.name and path.suffix != ".sh":
        return False
    if matches_any(path, ARCHIVE_PATTERNS):
        return include_archives
    if path.suffix == ".csv" and not include_large_csv:
        return False
    return matches_any(path, INCLUDE_PATTERNS)


def detect_shadow_mode(*, roots: list[Path], tag: str | None, included: list[dict], excluded: list[dict]) -> bool:
    if tag and "shadow" in tag.lower():
        return True
    if any("shadow" in root.name.lower() for root in roots):
        return True
    candidates = included + excluded
    return any(Path(item["path"]).name.startswith("shadow_") for item in candidates)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_dir is None and args.output_dir is None:
        parser.error("Provide --batch-dir or --output-dir.")

    roots = [path.resolve() for path in (args.batch_dir, args.output_dir) if path is not None]
    primary_root = roots[0]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.tag or primary_root.name or timestamp
    label = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in label)
    output_zip = args.output_zip or primary_root / f"stability_batch_{label}_{timestamp}.zip"
    manifest_path = output_zip.with_name(f"stability_batch_manifest_{label}_{timestamp}.json")

    included: list[dict] = []
    excluded: list[dict] = []
    max_bytes = None if args.max_file_mb is None else int(args.max_file_mb * 1024 * 1024)

    for path in iter_candidate_files(roots):
        reason = ""
        include = should_include(
            path,
            tag=args.tag,
            include_archives=args.include_archives,
            include_large_csv=args.include_large_csv,
        )
        if not include:
            reason = "pattern/tag/archive filter"
        elif max_bytes is not None and path.stat().st_size > max_bytes:
            include = False
            reason = f"larger than --max-file-mb ({args.max_file_mb:g})"

        entry = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "reason": reason,
        }
        if include:
            included.append(entry)
        else:
            excluded.append(entry)

    shadow_mode = detect_shadow_mode(
        roots=roots,
        tag=args.tag,
        included=included,
        excluded=excluded,
    )
    expected_patterns = SHADOW_EXPECTED_PATTERNS if shadow_mode else STANDARD_EXPECTED_PATTERNS

    warnings: list[str] = []
    for pattern in expected_patterns:
        if not any(fnmatch.fnmatch(Path(item["path"]).name, pattern) for item in included):
            warnings.append(f"missing expected artifact pattern: {pattern}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in included:
            path = Path(item["path"])
            try:
                arcname = path.relative_to(primary_root)
            except ValueError:
                arcname = Path(path.name)
            zf.write(path, arcname=str(arcname))

    manifest = {
        "created_utc": timestamp,
        "tag": args.tag,
        "roots": [str(root) for root in roots],
        "output_zip": str(output_zip),
        "git_commit": git_commit_hash(Path.cwd()),
        "python": sys.version,
        "detected_mode": "shadow" if shadow_mode else "standard",
        "expected_patterns": list(expected_patterns),
        "include_archives": args.include_archives,
        "include_large_csv": args.include_large_csv,
        "max_file_mb": args.max_file_mb,
        "included": included,
        "excluded": excluded,
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote zip: {output_zip}")
    print(f"wrote manifest: {manifest_path}")
    print(f"included files: {len(included)}")
    print(f"excluded files: {len(excluded)}")
    for warning in warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
