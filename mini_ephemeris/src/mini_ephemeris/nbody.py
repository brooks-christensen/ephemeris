from __future__ import annotations
import numpy as np
from dataclasses import dataclass

G_SI = 6.67430e-11  # m^3 kg^-1 s^-2


@dataclass
class NBodyState:
    """State of an N-body system in Cartesian coordinates."""

    positions: np.ndarray  # shape (N, 3)
    velocities: np.ndarray  # shape (N, 3)
    masses: np.ndarray      # shape (N,)

    def copy(self) -> "NBodyState":
        return NBodyState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
        )


def accelerations(positions: np.ndarray,
                  masses: np.ndarray,
                  G: float = G_SI) -> np.ndarray:
    """Compute Newtonian gravitational accelerations on each body."""
    r = positions
    N = r.shape[0]

    diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    dist3 = dist2 * np.sqrt(dist2)

    factor = G * masses[np.newaxis, :] / dist3
    acc = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return acc


def velocity_verlet_step(state: NBodyState,
                         acc: np.ndarray,
                         dt: float,
                         G: float = G_SI) -> tuple[NBodyState, np.ndarray]:
    """Perform one velocity-Verlet step with Newtonian gravity."""
    r = state.positions
    v = state.velocities
    m = state.masses

    v_half = v + 0.5 * dt * acc
    r_new = r + dt * v_half
    acc_new = accelerations(r_new, m, G=G)
    v_new = v_half + 0.5 * dt * acc_new

    new_state = NBodyState(positions=r_new, velocities=v_new, masses=m)
    return new_state, acc_new


def integrate(state0: NBodyState,
              t_span: tuple[float, float],
              dt: float,
              G: float = G_SI,
              record_every: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate an N-body system using Newtonian velocity-Verlet."""
    t0, t1 = t_span
    n_steps = int(np.floor((t1 - t0) / dt))
    n_records = n_steps // record_every + 1

    state = state0.copy()
    acc = accelerations(state.positions, state.masses, G=G)

    N = state.positions.shape[0]
    positions = np.empty((n_records, N, 3), dtype=float)
    velocities = np.empty((n_records, N, 3), dtype=float)
    times = np.empty((n_records,), dtype=float)

    positions[0] = state.positions
    velocities[0] = state.velocities
    times[0] = 0.0

    rec_idx = 1
    t = t0

    for step in range(1, n_steps + 1):
        state, acc = velocity_verlet_step(state, acc, dt, G=G)
        t += dt

        if step % record_every == 0:
            positions[rec_idx] = state.positions
            velocities[rec_idx] = state.velocities
            times[rec_idx] = t - t0
            rec_idx += 1

    return times, positions, velocities


def total_energy_series(positions: np.ndarray,
                        velocities: np.ndarray,
                        masses: np.ndarray,
                        G: float = G_SI) -> np.ndarray:
    """Compute total (kinetic + Newtonian potential) energy vs time."""
    M, N, _ = positions.shape
    m = masses

    ke = 0.5 * np.sum(m[np.newaxis, :, np.newaxis] * velocities**2, axis=(1, 2))

    pe = np.zeros(M)
    for k in range(M):
        r = positions[k]
        diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]
        dist = np.linalg.norm(diff, axis=-1)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        rij = dist[mask]
        mi_mj = (m[:, np.newaxis] * m[np.newaxis, :])[mask]
        pe[k] = -G * np.sum(mi_mj / rij)

    return ke + pe