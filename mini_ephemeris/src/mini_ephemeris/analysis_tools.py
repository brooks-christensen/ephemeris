from __future__ import annotations
import numpy as np

from .nbody import NBodyState, G_SI
from .advanced_integrators import (
    velocity_verlet_step_generic,
    acceleration_newtonian,
)


def poincare_section_y0(times: np.ndarray,
                        positions: np.ndarray,
                        velocities: np.ndarray,
                        body_index: int = 1,
                        direction: str = "positive") -> tuple[np.ndarray, np.ndarray]:
    y = positions[:, body_index, 1]
    x = positions[:, body_index, 0]
    vx = velocities[:, body_index, 0]

    xs = []
    vxs = []

    for k in range(1, len(times)):
        y_prev, y_curr = y[k - 1], y[k]
        if direction == "positive":
            cond = (y_prev < 0.0) and (y_curr >= 0.0)
        else:
            cond = (y_prev > 0.0) and (y_curr <= 0.0)

        if not cond:
            continue

        dy = y_curr - y_prev
        if dy == 0.0:
            alpha = 0.0
        else:
            alpha = -y_prev / dy

        x_cross = x[k - 1] + alpha * (x[k] - x[k - 1])
        vx_cross = vx[k - 1] + alpha * (vx[k] - vx[k - 1])

        xs.append(x_cross)
        vxs.append(vx_cross)

    return np.array(xs), np.array(vxs)


def lyapunov_max(state0: NBodyState,
                 t_span: tuple[float, float],
                 dt: float,
                 G: float = G_SI,
                 delta0: float = 1e-6,
                 renorm_interval: int = 50) -> float:
    t0, t1 = t_span
    n_steps = int((t1 - t0) / dt)

    rng = np.random.default_rng(42)
    delta_pos = rng.normal(size=state0.positions.shape)
    norm = np.linalg.norm(delta_pos)
    delta_pos *= (delta0 / norm)
    delta_vel = np.zeros_like(delta_pos)

    state = state0.copy()
    state_pert = NBodyState(
        positions=state0.positions + delta_pos,
        velocities=state0.velocities + delta_vel,
        masses=state0.masses.copy(),
    )

    acc = acceleration_newtonian(state, G=G)
    acc_pert = acceleration_newtonian(state_pert, G=G)

    delta_norm0 = delta0
    sum_log = 0.0
    n_renorm = 0
    t = t0

    for step in range(1, n_steps + 1):
        state, acc = velocity_verlet_step_generic(
            state, acc, dt, acceleration_newtonian, accel_kwargs={}, G=G
        )
        state_pert, acc_pert = velocity_verlet_step_generic(
            state_pert, acc_pert, dt, acceleration_newtonian, accel_kwargs={}, G=G
        )
        t += dt

        if step % renorm_interval == 0:
            dpos = state_pert.positions - state.positions
            dvel = state_pert.velocities - state.velocities
            delta_vec = np.concatenate([dpos.ravel(), dvel.ravel()])
            delta_norm = float(np.linalg.norm(delta_vec))
            if delta_norm == 0.0:
                continue

            sum_log += np.log(delta_norm / delta_norm0)
            n_renorm += 1

            factor = delta_norm0 / delta_norm
            state_pert = NBodyState(
                positions=state.positions + dpos * factor,
                velocities=state.velocities + dvel * factor,
                masses=state.masses.copy(),
            )
            acc_pert = acceleration_newtonian(state_pert, G=G)

    if n_renorm == 0:
        return 0.0

    total_time = renorm_interval * dt * n_renorm
    return sum_log / total_time