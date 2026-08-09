from __future__ import annotations

from typing import Any


def merge_lane_configuration(
    manifest: dict[str, Any], lane_configuration: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("scientific_configuration", "common_configuration"):
        common = manifest.get(key)
        if isinstance(common, dict):
            merged.update(common)
    merged.update(lane_configuration)
    command = merged.get("command")
    if not isinstance(command, list):
        return merged
    value_options = {
        "--step-days": ("step_days", float),
        "--duration-years": ("duration_years", float),
        "--record-every-years": ("record_every_years", float),
        "--archive-interval-years": ("archive_interval_years", float),
        "--megno-seed": ("megno_seed", int),
        "--gr-scale": ("gr_scale", float),
        "--model-scope": ("model_scope", str),
    }
    for index, token in enumerate(command[:-1]):
        if token in value_options:
            key, converter = value_options[token]
            merged[key] = converter(command[index + 1])
    if "mini_ephemeris.rebound_gr_tangent_backend_cli" in command:
        merged.setdefault("integrator", "whfast")
        merged.setdefault("variations", True)
        merged.setdefault("megno", True)
        merged.setdefault("exact_finish_time", 1)
    return merged


def lane_manifest_numbers(lane: dict[str, Any]) -> list[int]:
    values = set(lane["manifests"])
    if "origin_manifest" in lane:
        values.add(int(lane["origin_manifest"]))
    return sorted(values)
