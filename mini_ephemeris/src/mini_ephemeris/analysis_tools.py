from __future__ import annotations
import warnings

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
                 renorm_interval: int = 50,
                 return_growth: bool = False):
    """Finite-difference maximal Lyapunov estimate.

    This is a legacy demonstration path. It propagates two full trajectories
    and measures their separation, rather than integrating the variational
    equations as the production pipeline does, so its estimate is dominated by
    where in its oscillation the separation happens to be sampled until the run
    is long enough for genuine growth to dominate. On an integrable two-body
    system it returns small negative values that shrink with duration but do
    not reliably reach zero.

    ``lambda_max >= 0`` for a Hamiltonian flow, so a negative return value is
    never physical and always means the estimate has not converged. That case
    now emits a ``RuntimeWarning`` instead of being reported as a result.

    With ``return_growth=True`` this returns
    ``(estimate, times_seconds, cumulative_log_growth)`` so the caller can pass
    the history to ``chaos_estimator_diagnostics.analyze_growth`` and get a
    classification rather than a bare number.
    """
    t0, t1 = t_span
    n_steps = int((t1 - t0) / dt)

    rng = np.random.default_rng(42)
    delta_pos = rng.normal(size=state0.positions.shape)
    # Project out the barycenter displacement. A uniform shift of every body is
    # a neutral direction of the dynamics: it never grows, so any component of
    # the initial perturbation lying along it dilutes the estimate without
    # contributing signal. Previously the raw random vector was used, leaving a
    # large fraction of it in that neutral subspace.
    total_mass = float(np.sum(state0.masses))
    com_shift = (state0.masses[:, None] * delta_pos).sum(axis=0) / total_mass
    delta_pos = delta_pos - com_shift
    norm = float(np.linalg.norm(delta_pos))
    if norm == 0.0:
        raise ValueError("perturbation collapsed to zero after barycenter projection")
    delta_pos *= (delta0 / norm)
    delta_vel = np.zeros_like(delta_pos)

    # Characteristic timescale used to put velocities on the same footing as
    # positions in the phase-space norm. Mixing metres with metres/second, as
    # the previous norm did, is dimensionally meaningless.
    separations = np.linalg.norm(
        state0.positions - state0.positions.mean(axis=0), axis=1
    )
    length_scale = float(np.max(separations)) or 1.0
    speeds = np.linalg.norm(state0.velocities, axis=1)
    speed_scale = float(np.max(speeds))
    time_scale = length_scale / speed_scale if speed_scale > 0.0 else 1.0

    def phase_norm(dpos: np.ndarray, dvel: np.ndarray) -> float:
        return float(
            np.linalg.norm(
                np.concatenate([dpos.ravel(), (dvel * time_scale).ravel()])
            )
        )

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
    growth_times: list[float] = []
    growth_values: list[float] = []

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
            delta_norm = phase_norm(dpos, dvel)
            if delta_norm == 0.0:
                continue

            sum_log += np.log(delta_norm / delta_norm0)
            n_renorm += 1
            growth_times.append(t - t0)
            growth_values.append(sum_log)

            factor = delta_norm0 / delta_norm
            state_pert = NBodyState(
                positions=state.positions + dpos * factor,
                velocities=state.velocities + dvel * factor,
                masses=state.masses.copy(),
            )
            acc_pert = acceleration_newtonian(state_pert, G=G)

    if n_renorm == 0:
        return (0.0, [], []) if return_growth else 0.0

    total_time = renorm_interval * dt * n_renorm
    estimate = sum_log / total_time
    if estimate < 0.0:
        # lambda_max >= 0 for a Hamiltonian flow, so a negative value is never
        # physical -- it means the accumulated log is dominated by where in its
        # oscillation the separation happened to be sampled, not by growth. The
        # run is too short or too coarsely renormalized to have converged.
        warnings.warn(
            f"lyapunov_max returned a negative value ({estimate:.6e} 1/s). "
            "The maximal Lyapunov exponent of a Hamiltonian flow is "
            "non-negative; this indicates the estimate has not converged. "
            "Lengthen the run or shorten renorm_interval before using it.",
            RuntimeWarning,
            stacklevel=2,
        )
    if return_growth:
        return estimate, growth_times, growth_values
    return estimate