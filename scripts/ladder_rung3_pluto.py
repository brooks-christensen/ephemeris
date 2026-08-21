#!/usr/bin/env python3
"""Rung 3 of the validation ladder: Pluto's Lyapunov time.

Target ~20 Myr (Applegate et al. 1986; Sussman & Wisdom 1988), acceptance
window 10-40 Myr as fixed in docs/PLAN.md before the run.

Configuration: Sun (with the terrestrial planet masses folded in) plus the four
giant planets and Pluto -- the same system Sussman & Wisdom used. Pluto's chaos
comes from the 3:2 resonance with Neptune and the associated Kozai libration,
so the inner planets are not needed.

Two correlated diagnostics are recorded for each variational integration:

  * Running tangent growth, from the variational vector itself. REBOUND's MEGNO
    machinery carries first-order variational particles without renormalising
    them, so ln|delta(t)| is readable directly. The S(T)/T estimate and its
    duration-halving ratio go through the project's own analyze_growth.
  * MEGNO, as 2 x d<Y>/dt, using the factor measured in
    scripts/measure_megno_convention.py.

Both diagnostics use the same variational trajectory; their agreement is a
consistency check, not independent physical validation. sim.lyapunov() is
deliberately not used because it is the least-squares slope of <Y> and returns
lambda/2 in the installed REBOUND convention.

The coarse lane retains a 200 Myr checkpoint. A rung cannot pass until a 200 Myr
dt/2 lane supplies --compare-to and all five fixed tangent seeds meet every
condition in docs/PLAN.md.

    python3 scripts/ladder_rung3_pluto.py --years 4e8 --dt 0.4 --json rung3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from skyfield.api import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_ephemeris" / "src"))

try:
    import rebound
except ImportError:  # pragma: no cover
    sys.exit("REBOUND is not installed in this environment; run inside .venv")

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    MEGNO_MEAN_TO_LYAPUNOV,
    analyze_growth,
)
from mini_ephemeris.ephem import BODY_GM_KM3_S2  # noqa: E402
from mini_ephemeris.validation_ladder import evaluate_rung  # noqa: E402

AU_KM = 149_597_870.700
JULIAN_YEAR_SECONDS = 365.25 * 86_400.0
J2000_TT_JD = 2_451_545.0
DEFAULT_EPHEMERIS = Path(__file__).resolve().parents[1] / "data" / "de440s.bsp"

CENTRAL_COMPONENTS = (
    "sun",
    "mercury barycenter",
    "venus barycenter",
    "earth barycenter",
    "mars barycenter",
)
OUTER_BODIES = (
    "jupiter barycenter",
    "saturn barycenter",
    "uranus barycenter",
    "neptune barycenter",
    "pluto barycenter",
)
DEFAULT_TANGENT_SEEDS = (12345, 23456, 34567, 45678, 56789)

TARGET_LYAPUNOV_TIME_YEARS = 2.0e7
ACCEPTANCE_YEARS = (1.0e7, 4.0e7)

# Pre-flight physics check, added after the first attempt at this rung.
#
# Pluto's ~20 Myr Lyapunov time is a property of a Pluto *protected by the 3:2
# mean-motion resonance with Neptune*: the resonant argument
#
#     phi = 3*lambda_Pluto - 2*lambda_Neptune - varpi_Pluto
#
# librates about 180 deg, which keeps Pluto away from Neptune even though its
# perihelion (29.7 AU) lies inside Neptune's orbit (30.1 AU). Take Pluto out of
# the resonance and you are measuring a different system.
#
# The original rounded J2000 elements made phi circulate and could not represent
# the target system. This harness therefore loads one barycentric epoch from a
# hashed DE44x kernel and folds the terrestrial masses and states into the
# central particle. The check below remains a hard precondition so that a number
# produced from an unprotected Pluto can never be reported as Pluto's Lyapunov
# time.
MAX_LIBRATION_AMPLITUDE_DEG = 330.0
RESONANCE_CHECK_YEARS = 3.0e5
RESONANCE_CHECK_SAMPLES = 3000
ENERGY_DRIFT_LIMIT = 1.0e-9
TIMESTEP_RELATIVE_LIMIT = 0.10
ESTIMATOR_RELATIVE_LIMIT = 0.20
SATURATION_LOG_NORM_LIMIT = 300.0
CONVERGENCE_CHECKPOINT_YEARS = 2.0e8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skyfield_state(eph: object, name: str, epoch: object) -> tuple[np.ndarray, np.ndarray]:
    state = eph[name].at(epoch)
    position = np.asarray(state.position.au, dtype=float)
    velocity = (
        np.asarray(state.velocity.km_per_s, dtype=float)
        * JULIAN_YEAR_SECONDS
        / AU_KM
    )
    return position, velocity


def load_ephemeris_initial_conditions(
    ephemeris: Path = DEFAULT_EPHEMERIS,
) -> tuple[list[dict], dict]:
    """Load one barycentric DE44x epoch and fold terrestrial bodies centrally."""

    kernel_path = ephemeris.expanduser().resolve()
    if not kernel_path.is_file():
        raise FileNotFoundError(f"ephemeris kernel not found: {kernel_path}")

    timescale = load.timescale()
    kernel = load(str(kernel_path))
    try:
        epoch = timescale.tt_jd(J2000_TT_JD)
        names = CENTRAL_COMPONENTS + OUTER_BODIES
        states = {
            name: _skyfield_state(kernel, name, epoch)
            for name in names
        }
    finally:
        kernel.close()

    central_gm = sum(BODY_GM_KM3_S2[name] for name in CENTRAL_COMPONENTS)
    central_position = sum(
        BODY_GM_KM3_S2[name] * states[name][0]
        for name in CENTRAL_COMPONENTS
    ) / central_gm
    central_velocity = sum(
        BODY_GM_KM3_S2[name] * states[name][1]
        for name in CENTRAL_COMPONENTS
    ) / central_gm
    solar_gm = BODY_GM_KM3_S2["sun"]

    particles = [{
        "name": "sun plus terrestrial barycenters",
        "components": list(CENTRAL_COMPONENTS),
        "gm_km3_s2": central_gm,
        "mass_solar": central_gm / solar_gm,
        "position_au": central_position.tolist(),
        "velocity_au_per_year": central_velocity.tolist(),
    }]
    for name in OUTER_BODIES:
        position, velocity = states[name]
        gm = BODY_GM_KM3_S2[name]
        particles.append({
            "name": name,
            "gm_km3_s2": gm,
            "mass_solar": gm / solar_gm,
            "position_au": position.tolist(),
            "velocity_au_per_year": velocity.tolist(),
        })

    source_states = []
    for name in names:
        position, velocity = states[name]
        source_states.append({
            "name": name,
            "gm_km3_s2": BODY_GM_KM3_S2[name],
            "position_au": position.tolist(),
            "velocity_au_per_year": velocity.tolist(),
        })

    provenance = {
        "kernel_path": str(kernel_path),
        "kernel_sha256": _sha256(kernel_path),
        "kernel_bytes": kernel_path.stat().st_size,
        "epoch_tt_jd": J2000_TT_JD,
        "frame": "Skyfield ICRF barycentric",
        "position_unit": "AU",
        "velocity_unit": "AU/Julian-year",
        "mass_source": "mini_ephemeris.ephem DE431 GM constants",
        "mass_unit": "GM/GM_sun",
        "central_folding": "GM-weighted barycenter of Sun through Mars barycenters",
        "gravitational_constant_au3_per_solar_mass_year2": (
            solar_gm * JULIAN_YEAR_SECONDS ** 2 / AU_KM ** 3
        ),
        "source_states": source_states,
        "simulation_particles_before_com_shift": particles,
    }
    fingerprint_payload = {
        "kernel_sha256": provenance["kernel_sha256"],
        "epoch_tt_jd": provenance["epoch_tt_jd"],
        "frame": provenance["frame"],
        "mass_source": provenance["mass_source"],
        "central_folding": provenance["central_folding"],
        "gravitational_constant": (
            provenance["gravitational_constant_au3_per_solar_mass_year2"]
        ),
        "particles": particles,
    }
    provenance["physical_configuration_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return particles, provenance


def build(
    dt: float,
    ephemeris: Path = DEFAULT_EPHEMERIS,
    *,
    megno_seed: int | None = None,
) -> tuple["rebound.Simulation", dict]:
    particles, provenance = load_ephemeris_initial_conditions(ephemeris)
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.G = provenance["gravitational_constant_au3_per_solar_mass_year2"]
    sim.integrator = "whfast"
    sim.dt = dt
    for particle in particles:
        sim.add(
            m=particle["mass_solar"],
            x=particle["position_au"][0],
            y=particle["position_au"][1],
            z=particle["position_au"][2],
            vx=particle["velocity_au_per_year"][0],
            vy=particle["velocity_au_per_year"][1],
            vz=particle["velocity_au_per_year"][2],
        )
    sim.move_to_com()
    if megno_seed is not None:
        sim.init_megno(seed=megno_seed)
    return sim, provenance


def minimum_circular_span_degrees(angles: list[float]) -> tuple[float, float]:
    """Return the shortest covering arc and its center for angular samples."""

    values = np.sort(np.mod(np.asarray(angles, dtype=float), 360.0))
    if values.size == 0:
        raise ValueError("at least one angular sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("angular samples must be finite")
    if values.size == 1:
        return 0.0, float(values[0])

    gaps = np.diff(np.concatenate((values, values[:1] + 360.0)))
    largest_gap_index = int(np.argmax(gaps))
    span = 360.0 - float(gaps[largest_gap_index])
    arc_start = float(values[(largest_gap_index + 1) % values.size])
    center = (arc_start + 0.5 * span) % 360.0
    return span, center


def check_pluto_resonance(
    dt: float,
    years: float = RESONANCE_CHECK_YEARS,
    samples: int = RESONANCE_CHECK_SAMPLES,
    ephemeris: Path = DEFAULT_EPHEMERIS,
) -> dict:
    """Does the 3:2 resonant argument librate? Precondition, not a diagnostic."""

    if years <= 0.0 or samples < 2:
        raise ValueError("resonance check requires positive years and at least 2 samples")
    sim, provenance = build(dt, ephemeris)
    phis: list[float] = []
    separations: list[float] = []
    semi_major: list[float] = []
    for t in np.linspace(years / samples, years, samples):
        sim.integrate(t, exact_finish_time=0)
        pluto = sim.particles[5].orbit(primary=sim.particles[0])
        neptune = sim.particles[4].orbit(primary=sim.particles[0])
        phi = (3.0 * pluto.l - 2.0 * neptune.l
               - (pluto.Omega + pluto.omega)) % (2.0 * math.pi)
        phis.append(math.degrees(phi))
        offset = sim.particles[5] - sim.particles[4]
        separations.append(math.sqrt(offset.x ** 2 + offset.y ** 2 + offset.z ** 2))
        semi_major.append(pluto.a)

    amplitude, center = minimum_circular_span_degrees(phis)
    return {
        "libration_amplitude_deg": amplitude,
        "libration_center_deg": center,
        "librating": amplitude < MAX_LIBRATION_AMPLITUDE_DEG,
        "min_pluto_neptune_separation_au": float(min(separations)),
        "pluto_a_range_au": [float(min(semi_major)), float(max(semi_major))],
        "checked_years": years,
        "sample_count": samples,
        "initial_condition_provenance": provenance,
    }


def log_tangent_norm(sim: "rebound.Simulation") -> float:
    total = 0.0
    for p in sim.particles[sim.N_real:]:
        total += p.x * p.x + p.y * p.y + p.z * p.z
        total += p.vx * p.vx + p.vy * p.vy + p.vz * p.vz
    return 0.5 * math.log(total)


def observation_targets(years: float, samples: int) -> np.ndarray:
    """Deterministic, non-periodic output times with fixed midpoint/final anchors."""

    if years <= 0.0 or samples < 20:
        raise ValueError("integration requires positive years and at least 20 samples")
    interval = years / samples
    indices = np.arange(1, samples + 1, dtype=float)
    golden_fraction = (math.sqrt(5.0) - 1.0) / 2.0
    dither = (((indices * golden_fraction) % 1.0) - 0.5) * 0.20 * interval
    targets = indices * interval + dither
    targets[-1] = years
    if math.isclose(years, 2.0 * CONVERGENCE_CHECKPOINT_YEARS):
        targets[samples // 2 - 1] = CONVERGENCE_CHECKPOINT_YEARS
    if not np.all(np.diff(targets) > 0.0):
        raise RuntimeError("internal error: observation targets are not increasing")
    return targets


def history_path_for_seed(path: Path | None, seed: int) -> Path | None:
    """Per-seed history file, so concurrent seeds cannot collide."""

    if path is None:
        return None
    return path.with_name(f"{path.stem}-seed{seed}{path.suffix or '.csv'}")


def run(
    years: float,
    dt: float,
    samples: int,
    progress: Path | None,
    seed: int,
    ephemeris: Path = DEFAULT_EPHEMERIS,
    history: Path | None = None,
) -> dict:
    sim, provenance = build(dt, ephemeris, megno_seed=seed)
    energy0 = sim.energy()
    log0 = log_tangent_norm(sim)

    requested_times = observation_targets(years, samples)
    times = np.empty(samples)
    growth = np.empty(samples)
    log_norms = np.empty(samples)
    mean_megno = np.empty(samples)
    energy_drifts = np.empty(samples)
    started = time.time()
    progress_interval = max(1, samples // 100)

    # Persist the sampled history as it is produced. The first attempt at this
    # rung saved only summary statistics, so every reanalysis since has had to
    # reconstruct S(t) from halving ratios, and two reproduction runs that died
    # mid-flight left nothing usable. A killed run should still leave data.
    history_file = None
    if history is not None:
        history.parent.mkdir(parents=True, exist_ok=True)
        history_file = history.open("w", encoding="utf-8")
        history_file.write(
            "time_years,cumulative_log_growth,mean_megno,relative_energy_error\n"
        )

    for i, target in enumerate(requested_times):
        sim.integrate(target, exact_finish_time=0)
        times[i] = sim.t
        log_norms[i] = log_tangent_norm(sim)
        growth[i] = log_norms[i] - log0
        mean_megno[i] = sim.megno()
        energy_drifts[i] = abs((sim.energy() - energy0) / energy0)
        if history_file is not None:
            history_file.write(
                f"{times[i]:.10e},{growth[i]:.10e},"
                f"{mean_megno[i]:.10e},{energy_drifts[i]:.6e}\n"
            )
            if i % progress_interval == 0:
                history_file.flush()
        if progress is not None and i % progress_interval == 0:
            elapsed = time.time() - started
            progress.write_text(
                f"seed={seed}  {i + 1}/{samples}  t={sim.t:.3e} yr  "
                f"S={growth[i]:.4f}  <Y>={mean_megno[i]:.4f}  "
                f"dE/E={energy_drifts[:i + 1].max():.2e}  "
                f"elapsed={elapsed / 60:.1f} min\n"
            )

    if history_file is not None:
        history_file.close()

    return {
        "requested_times": requested_times,
        "times": times,
        "growth": growth,
        "log_norms": log_norms,
        "mean_megno": mean_megno,
        "energy_drifts": energy_drifts,
        "initial_log_tangent_norm": log0,
        "wall_seconds": time.time() - started,
        "dt": dt,
        "years": years,
        "seed": seed,
        "initial_condition_provenance": provenance,
    }


def summarise(data: dict, stop_years: float | None = None) -> dict:
    count = len(data["times"])
    if stop_years is not None:
        index = int(np.argmin(np.abs(data["requested_times"] - stop_years)))
        if not math.isclose(
            float(data["requested_times"][index]),
            stop_years,
            rel_tol=0.0,
            abs_tol=max(data["dt"], 1.0e-12),
        ):
            raise ValueError("requested checkpoint is not represented in samples")
        count = index + 1

    times = data["times"][:count]
    growth = data["growth"][:count]
    log_norms = data["log_norms"][:count]
    mean_megno = data["mean_megno"][:count]
    energy_drifts = data["energy_drifts"][:count]
    max_energy_drift = float(np.max(energy_drifts))
    result = analyze_growth(
        times,
        growth,
        max_relative_energy_drift=max_energy_drift,
    )
    half = len(times) // 2
    megno_slope = float(np.polyfit(times[half:], mean_megno[half:], 1)[0])
    lambda_megno = megno_slope * MEGNO_MEAN_TO_LYAPUNOV
    lambda_benettin = result.lambda_running_final
    agreement = (
        abs(lambda_megno - lambda_benettin) / abs(lambda_benettin)
        if lambda_benettin > 0 else math.nan
    )
    finite_tangent = bool(np.all(np.isfinite(log_norms)))
    max_abs_log_norm = float(np.max(np.abs(log_norms)))
    saturation_excluded = (
        finite_tangent and max_abs_log_norm < SATURATION_LOG_NORM_LIMIT
    )
    lyapunov_time = (
        1.0 / lambda_benettin if lambda_benettin > 0 else math.inf
    )
    return {
        "seed": data["seed"],
        "lambda_benettin_1_per_year": lambda_benettin,
        "lambda_megno_1_per_year": lambda_megno,
        "lyapunov_time_years": lyapunov_time,
        "lyapunov_time_myr": lyapunov_time / 1.0e6,
        "halving_ratio": result.halving_ratio,
        "classification": result.classification,
        "final_mean_megno": float(mean_megno[-1]),
        "megno_slope_per_year": megno_slope,
        "estimator_relative_disagreement": agreement,
        "max_relative_energy_drift": max_energy_drift,
        "final_cumulative_log_growth": float(growth[-1]),
        "max_abs_log_tangent_norm": max_abs_log_norm,
        "saturation_excluded": saturation_excluded,
        "saturation_definition": (
            "linear variational tangent remains finite with "
            f"|log(norm)| < {SATURATION_LOG_NORM_LIMIT:g}"
        ),
        "dt_years": data["dt"],
        "duration_years": float(times[-1]),
        "sample_count": count,
        "wall_minutes": data["wall_seconds"] / 60.0,
    }


def aggregate_seed_summaries(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("at least one tangent seed is required")
    lambdas = np.asarray(
        [item["lambda_benettin_1_per_year"] for item in summaries],
        dtype=float,
    )
    lambda_megnos = np.asarray(
        [item["lambda_megno_1_per_year"] for item in summaries],
        dtype=float,
    )
    median_lambda = float(np.median(lambdas))
    median_lyapunov_time = (
        1.0 / median_lambda if median_lambda > 0.0 else math.inf
    )
    return {
        "seed_count": len(summaries),
        "seeds": [item["seed"] for item in summaries],
        "lambda_benettin_median_1_per_year": median_lambda,
        "lambda_benettin_min_1_per_year": float(np.min(lambdas)),
        "lambda_benettin_max_1_per_year": float(np.max(lambdas)),
        "lambda_benettin_relative_range": (
            float(np.ptp(lambdas) / abs(median_lambda))
            if median_lambda != 0.0 else math.inf
        ),
        "lambda_megno_median_1_per_year": float(np.median(lambda_megnos)),
        "lyapunov_time_years": median_lyapunov_time,
        "lyapunov_time_myr": median_lyapunov_time / 1.0e6,
        "halving_ratio_min": min(item["halving_ratio"] for item in summaries),
        "halving_ratio_max": max(item["halving_ratio"] for item in summaries),
        "estimator_relative_disagreement_max": max(
            item["estimator_relative_disagreement"] for item in summaries
        ),
        "max_relative_energy_drift": max(
            item["max_relative_energy_drift"] for item in summaries
        ),
        "all_classified_chaotic": all(
            item["classification"] == "chaotic_candidate"
            for item in summaries
        ),
        "all_saturation_excluded": all(
            item["saturation_excluded"] for item in summaries
        ),
        "dt_years": summaries[0]["dt_years"],
        "duration_years": summaries[0]["duration_years"],
        "wall_minutes_total": sum(item["wall_minutes"] for item in summaries),
        "seed_summaries": summaries,
    }


def parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("tangent seeds must be unique")
    if any(seed < 0 or seed > 0xFFFFFFFF for seed in seeds):
        raise argparse.ArgumentTypeError("tangent seeds must fit an unsigned 32-bit integer")
    return seeds


def _relative_change(first: float, second: float) -> float:
    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return 0.0
    return abs(first - second) / scale


def load_timestep_comparison(
    compare_to: Path | None,
    current_summary: dict,
    provenance: dict,
) -> dict:
    if compare_to is None:
        return {
            "available": False,
            "passed": False,
            "reason": "no coarse-lane JSON supplied with --compare-to",
            "relative_limit": TIMESTEP_RELATIVE_LIMIT,
        }

    source = compare_to.expanduser().resolve()
    payload = json.loads(source.read_text())
    evidence = payload.get("evidence", {})
    coarse = evidence.get("convergence_checkpoint_summary")
    coarse_provenance = evidence.get("initial_condition_provenance")
    if not isinstance(coarse, dict) or not isinstance(coarse_provenance, dict):
        raise ValueError("comparison JSON lacks rung-3 checkpoint/provenance evidence")
    if (
        coarse_provenance.get("physical_configuration_fingerprint")
        != provenance["physical_configuration_fingerprint"]
    ):
        raise ValueError("comparison JSON has an incompatible physical fingerprint")
    if not math.isclose(
        coarse["dt_years"],
        2.0 * current_summary["dt_years"],
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("comparison JSON timestep is not exactly twice the current dt")
    if not math.isclose(
        coarse["duration_years"],
        current_summary["duration_years"],
        rel_tol=0.0,
        abs_tol=max(coarse["dt_years"], current_summary["dt_years"]),
    ):
        raise ValueError("comparison JSON does not represent the same duration")
    if coarse["seeds"] != current_summary["seeds"]:
        raise ValueError("comparison JSON uses different tangent seeds")

    coarse_by_seed = {
        item["seed"]: item for item in coarse["seed_summaries"]
    }
    changes = []
    for current in current_summary["seed_summaries"]:
        prior = coarse_by_seed[current["seed"]]
        benettin_change = _relative_change(
            prior["lambda_benettin_1_per_year"],
            current["lambda_benettin_1_per_year"],
        )
        megno_change = _relative_change(
            prior["lambda_megno_1_per_year"],
            current["lambda_megno_1_per_year"],
        )
        changes.append({
            "seed": current["seed"],
            "benettin_relative_change": benettin_change,
            "megno_relative_change": megno_change,
            "benettin_passed": benettin_change < TIMESTEP_RELATIVE_LIMIT,
            "megno_passed": megno_change < TIMESTEP_RELATIVE_LIMIT,
        })

    return {
        "available": True,
        "passed": all(
            item["benettin_passed"] and item["megno_passed"]
            for item in changes
        ),
        "relative_limit": TIMESTEP_RELATIVE_LIMIT,
        "formula": "abs(lambda_fine-lambda_coarse)/max(abs(lambda_fine),abs(lambda_coarse))",
        "coarse_json_path": str(source),
        "coarse_json_sha256": _sha256(source),
        "coarse_dt_years": coarse["dt_years"],
        "fine_dt_years": current_summary["dt_years"],
        "common_duration_years": current_summary["duration_years"],
        "per_seed": changes,
        "benettin_relative_change_max": max(
            item["benettin_relative_change"] for item in changes
        ),
        "megno_relative_change_max": max(
            item["megno_relative_change"] for item in changes
        ),
    }


def progress_path_for_seed(path: Path | None, seed: int) -> Path | None:
    if path is None:
        return None
    return path.with_name(f"{path.stem}.seed{seed}{path.suffix}")


def write_json_atomic(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output collision: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def rung_conditions(
    years: float,
    summary: dict,
    seed_summaries: list[dict],
    resonance: dict,
    comparison: dict,
) -> tuple[tuple[str, bool], ...]:
    every_halving_ratio_passes = all(
        0.85 <= item["halving_ratio"] <= 1.15
        for item in seed_summaries
    )
    every_energy_gate_passes = all(
        item["max_relative_energy_drift"] < ENERGY_DRIFT_LIMIT
        for item in seed_summaries
    )
    every_estimator_gate_passes = all(
        item["estimator_relative_disagreement"] < ESTIMATOR_RELATIVE_LIMIT
        for item in seed_summaries
    )
    return (
        ("duration is at least 200 Myr", years >= CONVERGENCE_CHECKPOINT_YEARS),
        ("at least 5 independent tangent seeds", summary["seed_count"] >= 5),
        ("every seed is classified chaotic", summary["all_classified_chaotic"]),
        ("every seed has halving ratio in [0.85, 1.15]", every_halving_ratio_passes),
        (
            f"every seed has energy drift < {ENERGY_DRIFT_LIMIT:g}",
            every_energy_gate_passes,
        ),
        (
            "every seed has Benettin/MEGNO disagreement "
            f"< {ESTIMATOR_RELATIVE_LIMIT:g}",
            every_estimator_gate_passes,
        ),
        (
            "linear variational saturation is excluded for every seed",
            summary["all_saturation_excluded"],
        ),
        ("Pluto is protected by the 3:2 resonance", bool(resonance["librating"])),
        (
            "same-duration dt-halving changes both estimators by < 10% per seed",
            bool(comparison["passed"]),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=float, default=4.0e8)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--ephemeris", type=Path, default=DEFAULT_EPHEMERIS)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=DEFAULT_TANGENT_SEEDS,
        help="comma-separated unsigned 32-bit tangent seeds",
    )
    parser.add_argument("--compare-to", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--progress", type=Path, default=None)
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="write the per-sample history here; one file per seed",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--skip-resonance-check",
        action="store_true",
        help="diagnostic only; the rung cannot PASS without it",
    )
    args = parser.parse_args(argv)
    if args.dt <= 0.0:
        parser.error("--dt must be positive")

    print(
        f"REBOUND {rebound.__version__}   {args.years:.3e} yr at "
        f"dt = {args.dt} yr   seeds={args.seeds}"
    )

    if args.skip_resonance_check:
        _, provenance = load_ephemeris_initial_conditions(args.ephemeris)
        resonance = {
            "librating": False,
            "skipped": True,
            "note": "precondition skipped; this run cannot pass",
            "initial_condition_provenance": provenance,
        }
    else:
        print("  pre-flight: is Pluto in the 3:2 resonance?", flush=True)
        resonance = check_pluto_resonance(
            args.dt,
            ephemeris=args.ephemeris,
        )
    provenance = resonance["initial_condition_provenance"]
    for key, value in resonance.items():
        if key != "initial_condition_provenance":
            print(f"    {key}: {value}")
    print(f"    kernel_sha256: {provenance['kernel_sha256']}")
    print(
        "    physical_configuration_fingerprint: "
        f"{provenance['physical_configuration_fingerprint']}"
    )

    if not resonance["librating"]:
        print()
        print("  STOP. Pluto did not satisfy the preregistered 3:2-resonance")
        print("  precondition. No Lyapunov integration was launched.")
        if args.json is not None:
            write_json_atomic(
                args.json,
                {"status": "PREFLIGHT_FAILED", "evidence": resonance},
            )
        return 1

    if args.preflight_only:
        payload = {"status": "PREFLIGHT_PASSED", "evidence": resonance}
        if args.json is not None:
            write_json_atomic(args.json, payload)
            print(f"  wrote {args.json}")
        return 0

    full_seed_summaries = []
    checkpoint_seed_summaries = []
    for index, seed in enumerate(args.seeds, start=1):
        print(f"  tangent seed {index}/{len(args.seeds)}: {seed}", flush=True)
        data = run(
            args.years,
            args.dt,
            args.samples,
            progress_path_for_seed(args.progress, seed),
            seed,
            args.ephemeris,
            history_path_for_seed(args.history, seed),
        )
        if (
            data["initial_condition_provenance"]["physical_configuration_fingerprint"]
            != provenance["physical_configuration_fingerprint"]
        ):
            raise RuntimeError("physical configuration changed between tangent seeds")
        seed_summary = summarise(data)
        full_seed_summaries.append(seed_summary)
        if args.years + args.dt >= CONVERGENCE_CHECKPOINT_YEARS:
            checkpoint_seed_summaries.append(
                summarise(data, CONVERGENCE_CHECKPOINT_YEARS)
            )
        print(
            f"    T_L={seed_summary['lyapunov_time_myr']:.6g} Myr  "
            f"half={seed_summary['halving_ratio']:.6g}  "
            f"agreement={seed_summary['estimator_relative_disagreement']:.3e}  "
            f"dE/E={seed_summary['max_relative_energy_drift']:.3e}"
        )

    summary = aggregate_seed_summaries(full_seed_summaries)
    checkpoint_summary = (
        aggregate_seed_summaries(checkpoint_seed_summaries)
        if checkpoint_seed_summaries else None
    )
    comparison = (
        load_timestep_comparison(args.compare_to, checkpoint_summary, provenance)
        if checkpoint_summary is not None
        else {
            "available": False,
            "passed": False,
            "reason": "duration is shorter than the fixed 200 Myr checkpoint",
            "relative_limit": TIMESTEP_RELATIVE_LIMIT,
        }
    )

    conditions = rung_conditions(
        args.years,
        summary,
        full_seed_summaries,
        resonance,
        comparison,
    )
    evidence = {
        "ensemble_summary": summary,
        "convergence_checkpoint_summary": checkpoint_summary,
        "timestep_comparison": comparison,
        "resonance_precondition": {
            key: value
            for key, value in resonance.items()
            if key != "initial_condition_provenance"
        },
        "initial_condition_provenance": provenance,
        "thresholds": {
            "lyapunov_time_years": list(ACCEPTANCE_YEARS),
            "halving_ratio": [0.85, 1.15],
            "energy_drift_max": ENERGY_DRIFT_LIMIT,
            "estimator_relative_disagreement_max": ESTIMATOR_RELATIVE_LIMIT,
            "timestep_relative_change_max": TIMESTEP_RELATIVE_LIMIT,
            "minimum_tangent_seeds": 5,
            "saturation_log_norm_limit": SATURATION_LOG_NORM_LIMIT,
        },
        "sampling_strategy": (
            "golden-ratio-dithered targets; exact sim.t recorded; "
            "200 Myr and final targets anchored"
        ),
        "rebound_version": rebound.__version__,
    }
    result = evaluate_rung(
        "3",
        "Pluto Lyapunov time",
        measured=summary["lyapunov_time_years"],
        target=TARGET_LYAPUNOV_TIME_YEARS,
        acceptance=ACCEPTANCE_YEARS,
        unit="years",
        conditions=conditions,
        duration_seconds=summary["wall_minutes_total"] * 60.0,
        evidence=evidence,
    )

    print()
    print(result.one_line())
    print(f"    median T_L: {summary['lyapunov_time_myr']:.6g} Myr")
    print(
        "    tangent lambda range: "
        f"{summary['lambda_benettin_min_1_per_year']:.6e} .. "
        f"{summary['lambda_benettin_max_1_per_year']:.6e}"
    )
    print(
        "    max timestep changes: "
        f"Benettin={comparison.get('benettin_relative_change_max')}  "
        f"MEGNO={comparison.get('megno_relative_change_max')}"
    )
    for label, ok in result.conditions:
        print(f"    {'ok  ' if ok else 'BAD '} {label}")
    if args.json is not None:
        write_json_atomic(args.json, result.to_dict())
        print(f"  wrote {args.json}")
    return 0 if result.status.value == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
