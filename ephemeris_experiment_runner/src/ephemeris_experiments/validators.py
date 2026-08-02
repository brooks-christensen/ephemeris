from __future__ import annotations

import csv
import glob
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GateResult:
    passed: bool
    name: str
    detail: str


def _matches(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(Path(path) for path in glob.glob(pattern))
    return sorted({p.resolve() for p in found if p.is_file()})


def _first_path(data: Any, paths: list[str]) -> Any:
    for path in paths:
        current = data
        ok = True
        for piece in path.split("."):
            if isinstance(current, dict) and piece in current:
                current = current[piece]
            else:
                ok = False
                break
        if ok:
            return current
    raise KeyError(paths)


def required_glob(gate: dict[str, Any]) -> GateResult:
    patterns = gate.get("patterns") or [gate["pattern"]]
    minimum = int(gate.get("min_count", 1))
    matches = _matches(patterns)
    return GateResult(
        len(matches) >= minimum,
        gate.get("name", "required_glob"),
        f"found {len(matches)} file(s); required {minimum}: {', '.join(patterns)}",
    )


def csv_integrity(gate: dict[str, Any]) -> GateResult:
    patterns = gate.get("patterns") or [gate["pattern"]]
    matches = _matches(patterns)
    if not matches:
        return GateResult(False, gate.get("name", "csv_integrity"), "no matching CSV")
    path = max(matches, key=lambda p: p.stat().st_mtime)
    if b"\x00" in path.read_bytes():
        return GateResult(False, gate.get("name", "csv_integrity"), f"NUL byte found in {path}")

    time_candidates = gate.get("time_columns", ["time_years", "t_years", "time_yr"])
    finite_columns = gate.get("finite_columns", [])
    times: list[float] = []
    missing_finite: dict[str, int] = {name: 0 for name in finite_columns}
    rows = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        time_col = next((name for name in time_candidates if name in headers), None)
        if time_col is None:
            return GateResult(False, gate.get("name", "csv_integrity"), f"no time column in {headers}")
        for row in reader:
            rows += 1
            try:
                value = float(row[time_col])
            except (TypeError, ValueError):
                return GateResult(False, gate.get("name", "csv_integrity"), f"invalid time on row {rows}")
            if not math.isfinite(value):
                return GateResult(False, gate.get("name", "csv_integrity"), f"non-finite time on row {rows}")
            times.append(value)
            for column in finite_columns:
                raw = row.get(column)
                try:
                    parsed = float(raw) if raw not in (None, "") else math.nan
                except ValueError:
                    parsed = math.nan
                if not math.isfinite(parsed):
                    missing_finite[column] += 1

    if rows < int(gate.get("min_rows", 2)):
        return GateResult(False, gate.get("name", "csv_integrity"), f"only {rows} row(s) in {path}")
    duplicates = len(times) - len(set(times))
    monotonic = all(right > left for left, right in zip(times, times[1:]))
    if not monotonic or duplicates:
        return GateResult(
            False,
            gate.get("name", "csv_integrity"),
            f"time monotonic={monotonic}, duplicate count={duplicates} in {path}",
        )

    target = gate.get("target_years")
    if target is not None:
        tolerance_fraction = float(gate.get("final_time_tolerance_fraction", 0.002))
        tolerance_absolute = float(gate.get("final_time_tolerance_years", 0.0))
        tolerance = max(abs(float(target)) * tolerance_fraction, tolerance_absolute)
        if abs(times[-1] - float(target)) > tolerance:
            return GateResult(
                False,
                gate.get("name", "csv_integrity"),
                f"final time {times[-1]:.6g} differs from target {float(target):.6g} by more than {tolerance:.6g}",
            )

    allowed_missing_fraction = float(gate.get("allowed_missing_fraction", 0.0))
    bad_columns = {
        name: count for name, count in missing_finite.items() if count / max(rows, 1) > allowed_missing_fraction
    }
    if bad_columns:
        return GateResult(False, gate.get("name", "csv_integrity"), f"non-finite critical values: {bad_columns}")

    return GateResult(
        True,
        gate.get("name", "csv_integrity"),
        f"{rows} rows, strictly monotonic, final time {times[-1]:.6g}: {path}",
    )


def json_metrics(gate: dict[str, Any]) -> GateResult:
    patterns = gate.get("patterns") or [gate["pattern"]]
    matches = _matches(patterns)
    if not matches:
        return GateResult(False, gate.get("name", "json_metrics"), "no matching JSON")
    path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return GateResult(False, gate.get("name", "json_metrics"), f"cannot parse {path}: {exc}")

    details: list[str] = []
    for check in gate.get("checks", []):
        paths = check.get("paths") or [check["path"]]
        try:
            raw = _first_path(data, paths)
        except KeyError:
            if check.get("optional", False):
                continue
            return GateResult(False, gate.get("name", "json_metrics"), f"none of {paths} found in {path}")
        if "allowed" in check:
            if raw not in check["allowed"]:
                return GateResult(False, gate.get("name", "json_metrics"), f"{paths[0]}={raw!r} not in {check['allowed']}")
            details.append(f"{paths[0]}={raw}")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return GateResult(False, gate.get("name", "json_metrics"), f"{paths[0]} is not numeric: {raw!r}")
        if not math.isfinite(value):
            return GateResult(False, gate.get("name", "json_metrics"), f"{paths[0]} is non-finite")
        if "min" in check and value < float(check["min"]):
            return GateResult(False, gate.get("name", "json_metrics"), f"{paths[0]}={value:.6g} below {check['min']}")
        if "max" in check and value > float(check["max"]):
            return GateResult(False, gate.get("name", "json_metrics"), f"{paths[0]}={value:.6g} above {check['max']}")
        if "abs_max" in check and abs(value) > float(check["abs_max"]):
            return GateResult(False, gate.get("name", "json_metrics"), f"abs({paths[0]})={abs(value):.6g} above {check['abs_max']}")
        details.append(f"{paths[0]}={value:.6g}")
    return GateResult(True, gate.get("name", "json_metrics"), f"{'; '.join(details)} in {path}")


def command_gate(gate: dict[str, Any], cwd: str) -> GateResult:
    command = [str(part) for part in gate["command"]]
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    detail = (completed.stdout + "\n" + completed.stderr).strip()[-2000:]
    return GateResult(
        completed.returncode == 0,
        gate.get("name", "command"),
        f"exit={completed.returncode}; {detail}",
    )


def benettin_scientific_control(gate: dict[str, Any]) -> GateResult:
    patterns = gate.get("patterns") or [gate["pattern"]]
    matches = _matches(patterns)
    if not matches:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), "no matching summary JSON")
    path = max(matches, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"cannot parse {path}: {exc}")

    details: list[str] = []
    target_years = float(gate.get("target_years", data.get("duration_years", math.nan)))
    actual_years = float(data.get("actual_time_years", math.nan))
    tolerance = max(float(gate.get("final_time_tolerance_years", 0.0)), target_years * float(gate.get("final_time_tolerance_fraction", 0.002)))
    if not math.isfinite(actual_years) or abs(actual_years - target_years) > tolerance:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"actual_time_years={actual_years:.6g}, target={target_years:.6g}")
    details.append(f"actual_time_years={actual_years:.6g}")

    classification = data.get("classification_hint")
    allowed = gate.get("allowed_classifications", ["regular_likely"])
    if classification not in allowed:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"classification={classification!r} not in {allowed}")
    details.append(f"classification={classification}")

    lcn = float(data.get("finite_time_lcn_1_per_year", math.nan))
    if not math.isfinite(lcn):
        return GateResult(False, gate.get("name", "benettin_scientific_control"), "finite_time_lcn_1_per_year is not finite")
    max_abs_lcn = float(gate.get("max_abs_lcn_1_per_year", math.inf))
    if abs(lcn) > max_abs_lcn:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"abs(LCN)={abs(lcn):.6g} above {max_abs_lcn:.6g}")
    details.append(f"LCN={lcn:.6g}")

    max_ref_energy = float(gate.get("max_reference_relative_energy_error", math.inf))
    max_ref_l = float(gate.get("max_reference_relative_angular_momentum_error", math.inf))
    ref_energy = abs(float(data.get("max_abs_reference_relative_energy_error", 0.0)))
    ref_l = abs(float(data.get("max_abs_reference_relative_angular_momentum_error", 0.0)))
    if ref_energy > max_ref_energy:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"reference energy drift {ref_energy:.6g} above {max_ref_energy:.6g}")
    if ref_l > max_ref_l:
        return GateResult(False, gate.get("name", "benettin_scientific_control"), f"reference angular momentum drift {ref_l:.6g} above {max_ref_l:.6g}")
    details.append(f"ref_dE={ref_energy:.6g}; ref_dL={ref_l:.6g}")

    if gate.get("require_standalone_reference_agreement", False):
        max_pos = float(gate.get("max_reference_standalone_position_delta_m", math.inf))
        max_vel = float(gate.get("max_reference_standalone_velocity_delta_m_s", math.inf))
        pos = abs(float(data.get("reference_standalone_max_position_delta_m", math.inf)))
        vel = abs(float(data.get("reference_standalone_max_velocity_delta_m_s", math.inf)))
        if pos > max_pos:
            return GateResult(False, gate.get("name", "benettin_scientific_control"), f"standalone position delta {pos:.6g} m above {max_pos:.6g}")
        if vel > max_vel:
            return GateResult(False, gate.get("name", "benettin_scientific_control"), f"standalone velocity delta {vel:.6g} m/s above {max_vel:.6g}")
        details.append(f"standalone_pos={pos:.6g}; standalone_vel={vel:.6g}")

    if data.get("stable_positive_late_time_plateau", False):
        return GateResult(False, gate.get("name", "benettin_scientific_control"), "stable positive late-time plateau flagged")

    progress_patterns = gate.get("progress_patterns", [])
    if gate.get("require_lcn_trends_toward_zero", False):
        progress_matches = _matches(progress_patterns)
        if not progress_matches:
            return GateResult(False, gate.get("name", "benettin_scientific_control"), "no progress CSV for LCN trend check")
        progress_path = max(progress_matches, key=lambda p: p.stat().st_mtime)
        samples: list[tuple[float, float]] = []
        with progress_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    t = float(row.get("time_years", "nan"))
                    lcn_value = float(row.get("finite_time_lcn_1_per_year", "nan"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(t) and math.isfinite(lcn_value):
                    samples.append((t, lcn_value))
        min_samples = int(gate.get("lcn_trend_min_samples", 4))
        if len(samples) < min_samples:
            return GateResult(False, gate.get("name", "benettin_scientific_control"), f"only {len(samples)} finite LCN samples for trend check")
        split_fraction = float(gate.get("lcn_trend_split_fraction", 0.5))
        split_time = samples[0][0] + split_fraction * (samples[-1][0] - samples[0][0])
        early = [abs(value) for t, value in samples if t <= split_time]
        late = [abs(value) for t, value in samples if t >= split_time]
        if not early or not late:
            return GateResult(False, gate.get("name", "benettin_scientific_control"), "LCN trend check lacks early or late samples")
        early_median = sorted(early)[len(early) // 2]
        late_median = sorted(late)[len(late) // 2]
        max_ratio = float(gate.get("lcn_late_to_early_max_ratio", 1.0))
        late_abs_max = float(gate.get("lcn_late_abs_max", math.inf))
        if late_median > early_median * max_ratio and late_median > late_abs_max:
            return GateResult(
                False,
                gate.get("name", "benettin_scientific_control"),
                f"LCN does not trend toward zero: early median={early_median:.6g}, late median={late_median:.6g}",
            )
        details.append(f"LCN_trend early_median={early_median:.6g}; late_median={late_median:.6g}")

    return GateResult(True, gate.get("name", "benettin_scientific_control"), f"{'; '.join(details)} in {path}")


def run_gates(gates: tuple[dict[str, Any], ...], cwd: str) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in gates:
        kind = gate.get("kind")
        if kind == "required_glob":
            result = required_glob(gate)
        elif kind == "csv_integrity":
            result = csv_integrity(gate)
        elif kind == "json_metrics":
            result = json_metrics(gate)
        elif kind == "command":
            result = command_gate(gate, cwd)
        elif kind == "benettin_scientific_control":
            result = benettin_scientific_control(gate)
        else:
            result = GateResult(False, str(kind), f"unknown gate kind: {kind}")
        results.append(result)
        if not result.passed and gate.get("stop_on_failure", True):
            break
    return results
