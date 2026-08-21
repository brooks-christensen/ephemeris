#!/usr/bin/env python3
"""Adjudicate rung 3 from saved histories, against calibrated criteria.

Deliberately separate from the harness that generates the data. The harness
integrates; this decides. Keeping them apart means the verdict can be recomputed
without re-integrating, and means changing a criterion never means rerunning a
night of compute.

Every threshold below traces to a measurement on a system whose Lyapunov
exponent is known -- a massless body in Jupiter's resonance-overlap zone,
integrated for 417 Lyapunov times (scripts/calibrate_record_length.py). The
estimator's own scatter there:

    segment length     14 T_lyap   29 T_lyap   57 T_lyap   100 T_lyap
    1-sigma spread          33%         16%         17%           9%

Pluto at 400 Myr is 29 Lyapunov times and at 800 Myr is 57. Gates tighter than
about 16% are below the instrument's noise floor and cannot be met by any run.

    python3 scripts/analyze_rung3_history.py runs/rung3-400myr-seed*.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mini_ephemeris" / "src"))

from mini_ephemeris.chaos_estimator_diagnostics import (  # noqa: E402
    MEGNO_MEAN_TO_LYAPUNOV,
    analyze_growth,
    analyze_window_slopes,
    block_bootstrap_uncertainty,
    calibrated_uncertainty,
    segment_scatter,
)

ACCEPTANCE_MYR = (10.0, 40.0)
ESTIMATOR_AGREEMENT_LIMIT = 0.25       # quadrature of the two measured scatters
SEED_SPREAD_LIMIT = 0.10
MIN_SEEDS = 5


def analyse_one(path: Path) -> dict:
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    times, growth, megno, energy = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    windows = analyze_window_slopes(times, growth, n_windows=3)
    lam = float(np.median(windows.slopes_1_per_year))
    # The quoted uncertainty is externally calibrated. Within-record methods
    # bracket it but neither is right: the segment spread overestimates, the
    # block bootstrap underestimates, and both are reported so the bracket is
    # visible rather than hidden.
    record_lengths = float(times[-1]) * lam if lam > 0 else math.nan
    quoted = calibrated_uncertainty(record_lengths)
    upper = segment_scatter(times, growth, n_segments=4).relative_spread
    lower = block_bootstrap_uncertainty(times, growth, n_resamples=120)["relative_uncertainty"]
    late = times >= times[0] + 0.25 * (times[-1] - times[0])
    lam_megno = float(np.polyfit(times[late], megno[late], 1)[0]) * MEGNO_MEAN_TO_LYAPUNOV
    legacy = analyze_growth(times, growth, max_relative_energy_drift=float(energy.max()))
    half = len(times) // 2
    return {
        "file": path.name,
        "duration_years": float(times[-1]),
        "lambda_1_per_year": lam,
        "lyapunov_time_myr": (1.0 / lam) / 1.0e6 if lam > 0 else math.nan,
        "uncertainty_1_sigma": quoted,
        "uncertainty_upper_bound_segments": upper,
        "uncertainty_lower_bound_bootstrap": lower,
        "lambda_megno_1_per_year": lam_megno,
        "megno_disagreement": abs(lam_megno - lam) / abs(lam) if lam > 0 else math.nan,
        "record_lyapunov_times": record_lengths,
        "energy_max": float(energy.max()),
        "energy_at_half": float(energy[:half].max()),
        "energy_bounded": bool(float(energy.max()) < 2.0 * float(energy[:half].max())),
        "halving_ratio_diagnostic": legacy.halving_ratio,
        "classification_diagnostic": legacy.classification,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("histories", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--label", default="rung 3")
    args = parser.parse_args(argv)

    per_seed = [analyse_one(path) for path in args.histories]
    lambdas = np.array([item["lambda_1_per_year"] for item in per_seed])
    lam = float(np.median(lambdas))
    sigma = float(np.median([item["uncertainty_1_sigma"] for item in per_seed]))
    time_myr = (1.0 / lam) / 1.0e6
    lo, hi = time_myr * (1.0 - sigma), time_myr * (1.0 + sigma)
    seed_spread = float(np.std(lambdas) / abs(lam)) if lam > 0 else math.inf
    megno_gap = float(np.median([item["megno_disagreement"] for item in per_seed]))

    print()
    print(f"{args.label.upper()} -- {len(per_seed)} seeds, "
          f"{per_seed[0]['duration_years'] / 1e6:.0f} Myr, "
          f"{per_seed[0]['record_lyapunov_times']:.0f} Lyapunov times")
    print()
    for item in per_seed:
        print(f"  {item['file']:<34} {item['lyapunov_time_myr']:7.2f} Myr  "
              f"+/- {100 * item['uncertainty_1_sigma']:4.0f}%   "
              f"MEGNO {(1.0 / item['lambda_megno_1_per_year']) / 1e6:7.2f} Myr")
    print()

    checks = [
        (f"Lyapunov time {time_myr:.2f} Myr, 1-sigma band "
         f"{lo:.2f}-{hi:.2f} Myr, inside {ACCEPTANCE_MYR[0]:.0f}-{ACCEPTANCE_MYR[1]:.0f}",
         lo >= ACCEPTANCE_MYR[0] and hi <= ACCEPTANCE_MYR[1]),
        (f"windowed vs MEGNO {100 * megno_gap:.1f}% "
         f"< {100 * ESTIMATOR_AGREEMENT_LIMIT:.0f}% (calibrated)",
         megno_gap < ESTIMATOR_AGREEMENT_LIMIT),
        (f"seed spread {100 * seed_spread:.1f}% < {100 * SEED_SPREAD_LIMIT:.0f}%",
         seed_spread < SEED_SPREAD_LIMIT),
        (f"at least {MIN_SEEDS} seeds", len(per_seed) >= MIN_SEEDS),
        ("energy bounded, not secular",
         all(item["energy_bounded"] for item in per_seed)),
    ]
    for text, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    verdict = all(ok for _, ok in checks)
    print()
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}")
    print()
    print(f"  lambda = {lam:.5e} /yr   Lyapunov time = {time_myr:.2f} "
          f"(+{hi - time_myr:.2f}/-{time_myr - lo:.2f}) Myr")
    print("  Reported as a band. The estimator's floor at this record length is")
    print(f"  {100 * sigma:.0f}%, measured against a known answer, not assumed.")
    print()

    if args.json is not None:
        args.json.write_text(json.dumps({
            "label": args.label, "verdict": "PASS" if verdict else "FAIL",
            "lambda_1_per_year": lam, "lyapunov_time_myr": time_myr,
            "uncertainty_1_sigma": sigma, "band_myr": [lo, hi],
            "seed_spread": seed_spread, "megno_disagreement": megno_gap,
            "checks": [{"check": t, "passed": bool(o)} for t, o in checks],
            "per_seed": per_seed,
        }, indent=2))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
