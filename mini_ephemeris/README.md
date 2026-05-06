# mini-ephemeris

A small educational N-body sandbox for exploring Solar-System dynamics
with a symplectic (velocity-Verlet) integrator and comparisons against
JPL DE ephemerides via Skyfield.

Version 0.3.0 adds:

* Barycentric Sun + 8-planet Solar System initial conditions from JPL kernels
* Optional 1PN GR correction for Sun-planet interaction
* A higher-precision ephemeris experiment for 10^3–10^4 year spans
* A Lyapunov-visualization experiment using a perturbed copy of the Solar System
* Existing toy 3-body Euler / Lagrange experiments and Sun–Earth–Moon, etc.

## Install (editable)

```bash
cd mini_ephemeris
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

## Quick start

Basic Sun–Earth 2-year experiment with JPL comparison:

```bash
python -m mini_ephemeris.experiments --experiment sun_earth --years 2 --dt-days 1
```

High-accuracy Solar System ephemeris for 10,000 years (requires a long-span kernel,
e.g. DE431/DE441 that you have downloaded as 'de431.bsp'):

```bash
python -m mini_ephemeris.experiments \
  --experiment solar_system_ephem \
  --years 10000 \
  --dt-days 0.25 \
  --kernel-path de431.bsp
```

Lyapunov visualization (compares an unperturbed and a 1 m-perturbed Solar System):

```bash
python -m mini_ephemeris.experiments \
  --experiment solar_system_ephem \
  --years 2000000 \
  --dt-days 2.0 \
  --kernel-path de431.bsp \
  --with-lyapunov
```

The Solar-System experiments are barycentric and in SI units under the hood;
toy 3-body experiments use dimensionless units.