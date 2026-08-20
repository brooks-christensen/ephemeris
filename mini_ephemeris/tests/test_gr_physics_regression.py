"""Physics regression tests for the general-relativistic correction.

These lock in behaviour verified during an external review on 2026-08-19. The
central test is an end-to-end one: integrate a Mercury-like orbit under the
repository's own GR acceleration and measure the perihelion advance. A correct
1PN implementation gives 42.98 arcsec/century; any coefficient error gives a
clean multiple of it, so this single assertion catches the whole class of
prefactor mistakes.

No REBOUND, SciPy, or ephemeris kernel is required.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from mini_ephemeris.gr_potential_tangent import (
    C_M_PER_S,
    gr_potential_accelerations_and_tangent,
    gr_potential_pair_acceleration,
    gr_potential_pair_jacobian,
)

G_SI = 6.67430e-11
GM_SUN = 1.32712440018e20          # m^3 s^-2
M_SUN = GM_SUN / G_SI
MERCURY_A = 5.7909050e10           # m
MERCURY_E = 0.205630
ARCSEC_PER_RAD = 180.0 / math.pi * 3600.0
JULIAN_CENTURY_S = 36525.0 * 86400.0

# Textbook value for Mercury's relativistic perihelion advance.
EXPECTED_ARCSEC_PER_CENTURY = 42.98


def _analytic_precession_per_orbit(a: float, e: float, gm: float) -> float:
    """1PN apsidal advance, 6*pi*GM / (c^2 a (1 - e^2)), radians per orbit."""
    return 6.0 * math.pi * gm / (C_M_PER_S**2 * a * (1.0 - e * e))


def _lrl_argument(r: np.ndarray, v: np.ndarray, gm: float) -> tuple[float, float]:
    """Argument and magnitude of the Laplace-Runge-Lenz (eccentricity) vector."""
    h = np.cross(r, v)
    ecc = np.cross(v, h) / gm - r / np.linalg.norm(r)
    return math.atan2(ecc[1], ecc[0]), float(np.linalg.norm(ecc))


def _integrate_perihelion(
    n_orbits: int, steps_per_orbit: int, coefficient_scale: float = 1.0
) -> tuple[float, float, float]:
    """Return (precession rad/orbit, period s, eccentricity drift).

    ``coefficient_scale=0.0`` disables the GR term, which gives the pure
    Newtonian control run. Velocity-Verlet produces a spurious apsidal
    precession of its own, of order dt^2; differencing the two runs cancels it
    to first order and isolates the relativistic advance, which is what lets
    this test use a step size coarse enough to run in a couple of seconds.
    """
    a, e = MERCURY_A, MERCURY_E
    period = 2.0 * math.pi * math.sqrt(a**3 / GM_SUN)

    positions = np.zeros((2, 3))
    masses = np.array([M_SUN, 1.0])

    def acceleration(r: np.ndarray) -> np.ndarray:
        newtonian = -GM_SUN * r / np.linalg.norm(r) ** 3
        if coefficient_scale == 0.0:
            return newtonian
        positions[1] = r
        relativistic, _ = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            None,
            gravitational_constant=G_SI,
            central_index=0,
            coefficient_scale=coefficient_scale,
            include_central_response=False,
        )
        return newtonian + relativistic[1]

    r = np.array([a * (1.0 - e), 0.0, 0.0])
    v = np.array([0.0, math.sqrt(GM_SUN / a * (1.0 + e) / (1.0 - e)), 0.0])

    angle0, ecc0 = _lrl_argument(r, v, GM_SUN)
    dt = period / steps_per_orbit
    acc = acceleration(r)
    for _ in range(n_orbits * steps_per_orbit):
        v = v + 0.5 * dt * acc
        r = r + dt * v
        acc = acceleration(r)
        v = v + 0.5 * dt * acc
    angle1, ecc1 = _lrl_argument(r, v, GM_SUN)

    advance = (angle1 - angle0 + math.pi) % (2.0 * math.pi) - math.pi
    return advance / n_orbits, period, ecc1 - ecc0


class MercuryPerihelionPrecession(unittest.TestCase):
    """The decisive end-to-end check on the GR coefficient."""

    def test_analytic_formula_matches_textbook_value(self) -> None:
        per_orbit = _analytic_precession_per_orbit(MERCURY_A, MERCURY_E, GM_SUN)
        period = 2.0 * math.pi * math.sqrt(MERCURY_A**3 / GM_SUN)
        per_century = per_orbit * ARCSEC_PER_RAD * (JULIAN_CENTURY_S / period)
        self.assertAlmostEqual(per_century, EXPECTED_ARCSEC_PER_CENTURY, places=2)

    def test_integrated_precession_matches_general_relativity(self) -> None:
        """Integrate under the repo's own GR acceleration and measure.

        A wrong prefactor produces a clean multiple (1/2, 2, 3, 4x) of the
        expected advance, so a 3% tolerance is ample to catch one while
        tolerating the velocity-Verlet truncation at this step size.
        """
        with_gr, period, ecc_drift = _integrate_perihelion(
            n_orbits=4, steps_per_orbit=20_000, coefficient_scale=1.0
        )
        without_gr, _, _ = _integrate_perihelion(
            n_orbits=4, steps_per_orbit=20_000, coefficient_scale=0.0
        )
        measured_per_orbit = with_gr - without_gr
        per_century = (
            measured_per_orbit * ARCSEC_PER_RAD * (JULIAN_CENTURY_S / period)
        )
        self.assertLess(
            abs(per_century - EXPECTED_ARCSEC_PER_CENTURY)
            / EXPECTED_ARCSEC_PER_CENTURY,
            0.03,
            msg=f"measured {per_century:.4f} arcsec/century",
        )
        # The GR term is conservative: it must precess the orbit, not pump it.
        self.assertLess(abs(ecc_drift), 1.0e-9)

    def test_speed_of_light_is_the_defined_value(self) -> None:
        self.assertEqual(C_M_PER_S, 299_792_458.0)

    def test_zero_coefficient_scale_disables_the_correction(self) -> None:
        """A switch that silently fails to disable GR would be undetectable in
        the output, so assert the off state is genuinely off."""
        r = np.array([MERCURY_A, 0.0, 0.0])
        acceleration = gr_potential_pair_acceleration(
            r,
            gravitational_constant=G_SI,
            central_mass_kg=M_SUN,
            coefficient_scale=0.0,
        )
        self.assertTrue(np.array_equal(acceleration, np.zeros(3)))

    def test_correction_is_attractive_and_scales_as_inverse_r_cubed(self) -> None:
        for radius in (1.0e10, 1.0e11, 1.0e12):
            r = np.array([radius, 0.0, 0.0])
            a = gr_potential_pair_acceleration(
                r, gravitational_constant=G_SI, central_mass_kg=M_SUN
            )
            self.assertLess(a[0], 0.0)  # inward
            expected = 6.0 * GM_SUN**2 / (C_M_PER_S**2 * radius**3)
            self.assertLess(abs(abs(a[0]) - expected) / expected, 1.0e-12)


class GrTangentMatchesItsOwnAcceleration(unittest.TestCase):
    """A wrong GR tangent corrupts every Lyapunov and MEGNO result computed
    with the GR model, and produces no other symptom."""

    def test_pair_jacobian_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(7)
        for trial in range(4):
            with self.subTest(trial=trial):
                r = rng.normal(size=3) * 5.0e10
                analytic = gr_potential_pair_jacobian(
                    r, gravitational_constant=G_SI, central_mass_kg=M_SUN
                )
                best = math.inf
                for k in range(2, 14):
                    h = float(np.linalg.norm(r)) * 2.0**-k
                    numeric = np.zeros((3, 3))
                    for j in range(3):
                        step = np.zeros(3)
                        step[j] = h
                        plus = gr_potential_pair_acceleration(
                            r + step,
                            gravitational_constant=G_SI,
                            central_mass_kg=M_SUN,
                        )
                        minus = gr_potential_pair_acceleration(
                            r - step,
                            gravitational_constant=G_SI,
                            central_mass_kg=M_SUN,
                        )
                        numeric[:, j] = (plus - minus) / (2.0 * h)
                    best = min(
                        best,
                        float(
                            np.linalg.norm(numeric - analytic)
                            / np.linalg.norm(analytic)
                        ),
                    )
                self.assertLess(best, 1.0e-6)

    def test_multibody_tangent_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(11)
        n = 5
        positions = rng.normal(size=(n, 3)) * 8.0e10
        positions[0] = 0.0
        masses = np.array([M_SUN, 3.3e23, 4.9e24, 6.0e24, 6.4e23])
        delta = rng.normal(size=(n, 3))
        _, tangent = gr_potential_accelerations_and_tangent(
            positions, masses, delta, gravitational_constant=G_SI
        )
        best = math.inf
        for k in range(2, 16):
            h = 2.0**-k
            plus, _ = gr_potential_accelerations_and_tangent(
                positions + h * delta, masses, None, gravitational_constant=G_SI
            )
            minus, _ = gr_potential_accelerations_and_tangent(
                positions - h * delta, masses, None, gravitational_constant=G_SI
            )
            numeric = (plus - minus) / (2.0 * h)
            best = min(
                best,
                float(np.linalg.norm(numeric - tangent) / np.linalg.norm(tangent)),
            )
        self.assertLess(best, 1.0e-4)

    def test_third_law_holds_for_state_and_tangent(self) -> None:
        """Momentum conservation keeps the GR term out of the six neutral
        translation and boost directions, which the Benettin tangent relies on."""
        rng = np.random.default_rng(13)
        n = 5
        positions = rng.normal(size=(n, 3)) * 8.0e10
        positions[0] = 0.0
        masses = np.array([M_SUN, 3.3e23, 4.9e24, 6.0e24, 6.4e23])
        delta = rng.normal(size=(n, 3))
        acc, tangent = gr_potential_accelerations_and_tangent(
            positions,
            masses,
            delta,
            gravitational_constant=G_SI,
            include_central_response=True,
        )
        weighted = masses[:, None] * acc
        reference = float(np.linalg.norm(masses[1] * acc[1]))
        self.assertLess(float(np.linalg.norm(weighted.sum(axis=0))) / reference, 1.0e-12)

        weighted_tangent = masses[:, None] * tangent
        reference_tangent = float(np.linalg.norm(masses[1] * tangent[1]))
        self.assertLess(
            float(np.linalg.norm(weighted_tangent.sum(axis=0))) / reference_tangent,
            1.0e-11,
        )

    def test_central_response_can_be_switched_off(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [MERCURY_A, 0.0, 0.0]])
        masses = np.array([M_SUN, 3.3e23])
        without, _ = gr_potential_accelerations_and_tangent(
            positions, masses, None, gravitational_constant=G_SI,
            include_central_response=False,
        )
        self.assertTrue(np.array_equal(without[0], np.zeros(3)))
        with_response, _ = gr_potential_accelerations_and_tangent(
            positions, masses, None, gravitational_constant=G_SI,
            include_central_response=True,
        )
        self.assertGreater(float(np.linalg.norm(with_response[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
