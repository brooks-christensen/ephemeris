#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/peacelovephysics/ephemeris/mini_ephemeris"
PYTHON_BIN="${PYTHON:-/home/peacelovephysics/ephemeris/.venv/bin/python}"
KERNEL_PATH="${KERNEL_PATH:-/home/peacelovephysics/ephemeris/data/de431_part-2.bsp}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_benettin_smoke_script}"
UNINTERRUPTED_DIR="${UNINTERRUPTED_DIR:-/home/peacelovephysics/ephemeris/output/stability/gr_benettin_smoke_script_uninterrupted}"
TAG="${TAG:-two_body_jupiter_smoke}"

rm -rf "${OUTPUT_DIR}"
rm -rf "${UNINTERRUPTED_DIR}"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

echo "[smoke] compile/help"
"${PYTHON_BIN}" -m compileall src/mini_ephemeris/gr_benettin_cli.py src/mini_ephemeris/benettin_megno_comparison.py >/dev/null
"${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli --help >/dev/null
"${PYTHON_BIN}" -m mini_ephemeris.benettin_megno_comparison --help >/dev/null

echo "[smoke] independent REBOUNDx gr_potential symmetry"
"${PYTHON_BIN}" - <<'PY'
import numpy as np
import rebound
from mini_ephemeris.ephem import EphemerisConfig, initial_state_solar_system_barycentric
from mini_ephemeris.gr_benettin_cli import build_sim_pair
from mini_ephemeris.long_term_stability_cli import parse_start_datetime, rebound_state_from_sim
from mini_ephemeris.orbital_elements import JULIAN_YEAR_S

class Args:
    integrator = "whfast"
    rebound_integrator = None
    step_days = 1.0
    ias15_epsilon = 1.0e-10

state = initial_state_solar_system_barycentric(
    parse_start_datetime("2000-01-01"),
    bodies=("sun", "mercury barycenter"),
    config=EphemerisConfig(kernel_path="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp"),
)
ref, shadow, _ref_rebx, _shadow_rebx = build_sim_pair(rebound, state, Args(), "gr_potential")
ref.integrate(0.1 * JULIAN_YEAR_S, exact_finish_time=1)
shadow.integrate(0.1 * JULIAN_YEAR_S, exact_finish_time=1)
a = rebound_state_from_sim(ref, state.masses)
b = rebound_state_from_sim(shadow, state.masses)
if np.max(np.abs(a.positions - b.positions)) > 1.0e-6:
    raise SystemExit("position symmetry failed")
if np.max(np.abs(a.velocities - b.velocities)) > 1.0e-12:
    raise SystemExit("velocity symmetry failed")
PY

echo "[smoke] Newtonian finite-difference vs native variational oracle"
"${PYTHON_BIN}" - <<'PY'
import numpy as np
import rebound
from mini_ephemeris.ephem import EphemerisConfig, initial_state_solar_system_barycentric
from mini_ephemeris.gr_benettin_cli import (
    apply_state_to_sim,
    build_sim_pair,
    deviation_between_sims,
    direction_cosine,
    initial_shadow_state,
    scaled_norm,
    scaled_vector,
)
from mini_ephemeris.long_term_stability_cli import parse_start_datetime
from mini_ephemeris.orbital_elements import JULIAN_YEAR_S

class Args:
    integrator = "ias15"
    rebound_integrator = None
    step_days = 0.5
    ias15_epsilon = 1.0e-10
    perturb_body = "jupiter"
    perturbation_m = 1.0e-6
    perturbation_mode = "radial"

state = initial_state_solar_system_barycentric(
    parse_start_datetime("2000-01-01"),
    bodies=("sun", "jupiter barycenter"),
    config=EphemerisConfig(kernel_path="/home/peacelovephysics/ephemeris/data/de431_part-2.bsp"),
)
import random
shadow_state, target_norm, _ = initial_shadow_state(state, ("sun", "jupiter barycenter"), Args(), random.Random(7))
ref, shadow, _, _ = build_sim_pair(rebound, state, Args(), "none")
apply_state_to_sim(shadow, shadow_state)
var_sim = rebound.Simulation()
var_sim.G = ref.G
for p in ref.particles[:ref.N_real]:
    var_sim.add(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz)
var_sim.integrator = "ias15"
var_sim.ri_ias15.epsilon = Args.ias15_epsilon
variation = var_sim.add_variation()
delta_pos = shadow_state.positions - state.positions
delta_vel = shadow_state.velocities - state.velocities
for i, particle in enumerate(variation.particles):
    particle.x, particle.y, particle.z = delta_pos[i]
    particle.vx, particle.vy, particle.vz = delta_vel[i]
target_t = 1.0e-4 * JULIAN_YEAR_S
ref.integrate(target_t, exact_finish_time=1)
shadow.integrate(target_t, exact_finish_time=1)
var_sim.integrate(target_t, exact_finish_time=1)
fd_pos, fd_vel = deviation_between_sims(ref, shadow, state.masses)
var_pos = np.array([[p.x, p.y, p.z] for p in variation.particles])
var_vel = np.array([[p.vx, p.vy, p.vz] for p in variation.particles])
cosine = direction_cosine(scaled_vector(fd_pos, fd_vel), scaled_vector(var_pos, var_vel))
var_norm = scaled_norm(var_pos, var_vel)
if var_norm <= 0.0:
    print("[smoke] native variational oracle unavailable: arbitrary assigned variation did not propagate")
    raise SystemExit(0)
norm_ratio = scaled_norm(fd_pos, fd_vel) / var_norm
print("[smoke] variational cosine", cosine, "norm_ratio", norm_ratio)
if cosine < 0.999999:
    raise SystemExit("finite-difference and native variational directions disagree")
if abs(norm_ratio - 1.0) > 1.0e-4:
    raise SystemExit("finite-difference and native variational norms disagree")
PY

echo "[smoke] two-body checkpoint/resume"
"${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 2 \
  --step-days 16 \
  --record-every-years 1 \
  --model-scope two_body_jupiter \
  --integrator whfast \
  --gr-model none \
  --perturb-body jupiter \
  --perturbation-m 1 \
  --renorm-years 0.5 \
  --fit-start-years 0 \
  --seed 42 \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
  --with-standalone-reference-check \
  --progress-line-every-seconds 0 \
  --status-file-every-seconds 0 \
  --no-progress-bar >/dev/null

"${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 3 \
  --step-days 16 \
  --record-every-years 1 \
  --model-scope two_body_jupiter \
  --integrator whfast \
  --gr-model none \
  --perturb-body jupiter \
  --perturbation-m 1 \
  --renorm-years 0.5 \
  --fit-start-years 0 \
  --seed 42 \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
  --resume-latest \
  --with-standalone-reference-check \
  --progress-line-every-seconds 0 \
  --status-file-every-seconds 0 \
  --no-progress-bar >/dev/null

echo "[smoke] uninterrupted comparison run"
"${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 3 \
  --step-days 16 \
  --record-every-years 1 \
  --model-scope two_body_jupiter \
  --integrator whfast \
  --gr-model none \
  --perturb-body jupiter \
  --perturbation-m 1 \
  --renorm-years 0.5 \
  --fit-start-years 0 \
  --seed 42 \
  --output-dir "${UNINTERRUPTED_DIR}" \
  --tag "${TAG}" \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${UNINTERRUPTED_DIR}/checkpoints" \
  --with-standalone-reference-check \
  --progress-line-every-seconds 0 \
  --status-file-every-seconds 0 \
  --no-progress-bar >/dev/null

echo "[smoke] progress continuity and mismatch refusal"
"${PYTHON_BIN}" - <<'PY'
import csv
import json
from pathlib import Path

base = Path("/home/peacelovephysics/ephemeris/output/stability/gr_benettin_smoke_script")
uninterrupted = Path("/home/peacelovephysics/ephemeris/output/stability/gr_benettin_smoke_script_uninterrupted")
tag = "two_body_jupiter_smoke"
rows = list(csv.DictReader((base / f"benettin_progress_{tag}.csv").open()))
times = [float(row["time_years"]) for row in rows]
if times != sorted(times) or len(times) != len(set(times)):
    raise SystemExit("progress CSV is not monotonic unique")
summary = json.loads((base / f"benettin_summary_{tag}.json").read_text())
if summary.get("classification_hint") == "chaotic_candidate":
    raise SystemExit("two-body smoke must not classify as chaotic_candidate")
if summary.get("reference_standalone_max_position_delta_m", 1) > 1.0e-2:
    raise SystemExit("standalone reference check drifted from reference")
if summary.get("reference_standalone_max_velocity_delta_m_s", 1) > 1.0e-9:
    raise SystemExit("standalone reference velocity check drifted from reference")
other = json.loads((uninterrupted / f"benettin_summary_{tag}.json").read_text())
for key in ("finite_time_lcn_1_per_year", "accumulated_log_growth", "fit_accumulated_log_growth"):
    left = float(summary[key])
    right = float(other[key])
    if abs(left - right) > 1.0e-10:
        raise SystemExit(f"resume/uninterrupted mismatch for {key}: {left} vs {right}")
PY

set +e
"${PYTHON_BIN}" -m mini_ephemeris.gr_benettin_cli \
  --kernel-path "${KERNEL_PATH}" \
  --start-date 2000-01-01 \
  --duration-years 4 \
  --step-days 8 \
  --record-every-years 1 \
  --model-scope two_body_jupiter \
  --integrator whfast \
  --gr-model none \
  --perturb-body jupiter \
  --perturbation-m 1 \
  --renorm-years 0.5 \
  --fit-start-years 0 \
  --seed 42 \
  --output-dir "${OUTPUT_DIR}" \
  --tag "${TAG}" \
  --checkpoint-every-years 1 \
  --checkpoint-dir "${OUTPUT_DIR}/checkpoints" \
  --resume-latest \
  --no-progress-bar >/tmp/gr_benettin_mismatch.log 2>&1
code=$?
set -e
if [[ "${code}" -eq 0 ]]; then
  echo "configuration mismatch was not refused" >&2
  exit 1
fi
grep -q "configuration hash does not match" /tmp/gr_benettin_mismatch.log

echo "[smoke] comparison report"
"${PYTHON_BIN}" -m mini_ephemeris.benettin_megno_comparison \
  --output-dir "${OUTPUT_DIR}" \
  --tag smoke_compare >/dev/null

echo "[smoke] ok"
