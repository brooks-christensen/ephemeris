from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class LunarCalibration:
    """
    Named empirical lunar calibration profile.

    These parameters are intended for short-range ephemeris matching against a
    specific JPL/book convention. They should not be treated as general-purpose
    lunar physics for long-term stability studies.
    """

    name: str
    moon_dv_t_mm_s: float
    moon_a_t_1e_15_m_s2: float = 0.0

    description: str = ""
    fit_start_date: str | None = None
    fit_end_date: str | None = None
    validation_start_date: str | None = None
    validation_end_date: str | None = None
    objective: str | None = None
    model_notes: str | None = None

    lon_rms_arcsec: float | None = None
    lon_peak_abs_arcsec: float | None = None
    lat_rms_arcsec: float | None = None
    dist_rms_km: float | None = None

    @classmethod
    def from_mapping(cls, name: str, data: dict[str, Any]) -> "LunarCalibration":
        return cls(
            name=name,
            moon_dv_t_mm_s=float(data["moon_dv_t_mm_s"]),
            moon_a_t_1e_15_m_s2=float(data.get("moon_a_t_1e_15_m_s2", 0.0)),
            description=str(data.get("description", "")),
            fit_start_date=data.get("fit_start_date"),
            fit_end_date=data.get("fit_end_date"),
            validation_start_date=data.get("validation_start_date"),
            validation_end_date=data.get("validation_end_date"),
            objective=data.get("objective"),
            model_notes=data.get("model_notes"),
            lon_rms_arcsec=_optional_float(data.get("lon_rms_arcsec")),
            lon_peak_abs_arcsec=_optional_float(data.get("lon_peak_abs_arcsec")),
            lat_rms_arcsec=_optional_float(data.get("lat_rms_arcsec")),
            dist_rms_km=_optional_float(data.get("dist_rms_km")),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _load_json_profiles(path: str | Path) -> dict[str, LunarCalibration]:
    path = Path(path)

    with path.open() as f:
        raw = json.load(f)

    # Preferred format:
    # {
    #   "profiles": {
    #       "profile_name": {...}
    #   }
    # }
    profile_data = raw.get("profiles", raw)

    profiles: dict[str, LunarCalibration] = {}
    for name, data in profile_data.items():
        profiles[name] = LunarCalibration.from_mapping(name, data)

    return profiles


def load_lunar_calibration_profile(
    name: str,
    *,
    calibration_file: str | Path | None = None,
) -> LunarCalibration:
    """
    Load a named lunar calibration profile.

    Currently this loads from a JSON file. Keeping this as a helper makes it
    easy to add built-in profiles later without changing the CLIs.
    """
    if calibration_file is None:
        raise ValueError(
            "A --lunar-calibration-file path is required when using "
            "--lunar-calibration-profile."
        )

    profiles = _load_json_profiles(calibration_file)

    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(
            f"Lunar calibration profile {name!r} not found in "
            f"{calibration_file}. Available profiles: {available}"
        )

    return profiles[name]


def resolve_lunar_correction_values(
    *,
    profile_name: str | None,
    calibration_file: str | Path | None,
    moon_dv_t_mm_s: float | None,
    moon_a_t_1e_15_m_s2: float | None,
) -> tuple[float, float, LunarCalibration | None]:
    """
    Resolve final lunar correction values from optional profile + CLI overrides.

    Rule:
      - If no profile is selected, missing values default to 0.
      - If a profile is selected, profile values are used.
      - Explicit CLI values override profile values.
    """
    profile = None

    if profile_name is not None:
        profile = load_lunar_calibration_profile(
            profile_name,
            calibration_file=calibration_file,
        )

    if profile is None:
        resolved_dv = 0.0 if moon_dv_t_mm_s is None else float(moon_dv_t_mm_s)
        resolved_at = (
            0.0
            if moon_a_t_1e_15_m_s2 is None
            else float(moon_a_t_1e_15_m_s2)
        )
    else:
        resolved_dv = (
            profile.moon_dv_t_mm_s
            if moon_dv_t_mm_s is None
            else float(moon_dv_t_mm_s)
        )
        resolved_at = (
            profile.moon_a_t_1e_15_m_s2
            if moon_a_t_1e_15_m_s2 is None
            else float(moon_a_t_1e_15_m_s2)
        )

    return resolved_dv, resolved_at, profile


def save_lunar_calibration_profile(
    profile: LunarCalibration,
    path: str | Path,
) -> None:
    """
    Save or update one profile in a JSON calibration file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open() as f:
            raw = json.load(f)
    else:
        raw = {"profiles": {}}

    if "profiles" not in raw:
        raw = {"profiles": raw}

    raw["profiles"][profile.name] = {
        k: v for k, v in asdict(profile).items() if k != "name" and v is not None
    }

    with path.open("w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)
        f.write("\n")