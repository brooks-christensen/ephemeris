from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import math
from tqdm.auto import tqdm

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .nbody import NBodyState, accelerations, G_SI


R_EARTH_SI = 6_378_136.3
J2_EARTH = 1.08262545e-3
C_LIGHT = 299792458.0  # m/s


def acceleration_newtonian(state: NBodyState,
                           G: float = G_SI) -> np.ndarray:
    return accelerations(state.positions, state.masses, G=G)


def acceleration_newtonian_gr_sun(state: NBodyState,
                                  sun_index: int = 0,
                                  G: float = G_SI,
                                  c: float = C_LIGHT) -> np.ndarray:
    """Newtonian acceleration plus 1PN GR correction for Sun-planet pairs."""
    a = accelerations(state.positions, state.masses, G=G)

    rs = state.positions[sun_index]
    vs = state.velocities[sun_index]
    Msun = state.masses[sun_index]
    GM = G * Msun

    for i in range(state.positions.shape[0]):
        if i == sun_index:
            continue
        r_vec = state.positions[i] - rs
        v_vec = state.velocities[i] - vs
        r = np.linalg.norm(r_vec)
        if r == 0.0:
            continue

        v2 = float(np.dot(v_vec, v_vec))
        rv = float(np.dot(r_vec, v_vec))

        factor = GM / (c**2 * r**3)
        a_gr = factor * ((4.0 * GM / r - v2) * r_vec + 4.0 * rv * v_vec)
        a[i] += a_gr

    return a


def velocity_verlet_step_generic(state: NBodyState,
                                 acc: np.ndarray,
                                 dt: float,
                                 accel_func,
                                 accel_kwargs: dict | None = None,
                                 G: float = G_SI) -> tuple[NBodyState, np.ndarray]:
    if accel_kwargs is None:
        accel_kwargs = {}

    r = state.positions
    v = state.velocities
    m = state.masses

    v_half = v + 0.5 * dt * acc
    r_new = r + dt * v_half

    tmp_state = NBodyState(
        positions=r_new,
        velocities=v_half,
        masses=m,
    )
    acc_new = accel_func(tmp_state, G=G, **accel_kwargs)

    v_new = v_half + 0.5 * dt * acc_new

    new_state = NBodyState(positions=r_new, velocities=v_new, masses=m)
    return new_state, acc_new


def j2_acceleration_relative(
    r_rel: np.ndarray,
    mu: float,
    radius: float,
    j2: float,
    axis: np.ndarray | None = None,
) -> np.ndarray:
    """
    J2 acceleration correction on a satellite/body at r_rel relative to
    an oblate primary.

    r_rel points from primary -> secondary.
    axis is the primary spin axis in the same inertial frame.

    For now, axis defaults to J2000 +z, a reasonable first approximation
    for Earth's mean equator in this project.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

    r2 = float(np.dot(r_rel, r_rel))
    r = np.sqrt(r2)

    if r == 0.0:
        return np.zeros(3)

    s = float(np.dot(r_rel, axis))
    s2_over_r2 = (s * s) / r2

    factor = 1.5 * j2 * mu * radius * radius / (r ** 5)

    return factor * ((5.0 * s2_over_r2 - 1.0) * r_rel - 2.0 * s * axis)


def add_earth_moon_j2_correction(
    acc: np.ndarray,
    state: NBodyState,
    G: float = G_SI,
    earth_index: int = 3,
    moon_index: int = 4,
    radius_earth: float = R_EARTH_SI,
    j2_earth: float = J2_EARTH,
) -> np.ndarray:
    """
    Add Earth's J2 correction to the Earth-Moon pair.

    Applies the J2 acceleration to the Moon and an equal/opposite reaction
    to Earth to preserve linear momentum approximately.
    """
    acc = acc.copy()

    r_earth = state.positions[earth_index]
    r_moon = state.positions[moon_index]

    m_earth = state.masses[earth_index]
    m_moon = state.masses[moon_index]

    r_rel = r_moon - r_earth
    mu_earth = G * m_earth

    a_moon_j2 = j2_acceleration_relative(
        r_rel,
        mu=mu_earth,
        radius=radius_earth,
        j2=j2_earth,
    )

    acc[moon_index] += a_moon_j2
    acc[earth_index] -= (m_moon / m_earth) * a_moon_j2

    return acc


def acceleration_newtonian_earth_j2(
    state: NBodyState,
    G: float = G_SI,
    earth_index: int = 3,
    moon_index: int = 4,
) -> np.ndarray:
    acc = acceleration_newtonian_vectorized(state, G=G)
    return add_earth_moon_j2_correction(
        acc,
        state,
        G=G,
        earth_index=earth_index,
        moon_index=moon_index,
    )


def acceleration_newtonian_gr_sun_earth_j2(
    state: NBodyState,
    G: float = G_SI,
    sun_index: int = 0,
    earth_index: int = 3,
    moon_index: int = 4,
) -> np.ndarray:
    acc = acceleration_newtonian_gr_sun(
        state,
        G=G,
        sun_index=sun_index,
    )
    return add_earth_moon_j2_correction(
        acc,
        state,
        G=G,
        earth_index=earth_index,
        moon_index=moon_index,
    )


def _unit_vector_for_lunar_accel(vec: np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Return vec / |vec| with a useful error if the norm is degenerate."""
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Cannot normalize degenerate {name}.")
    return vec / norm


def lunar_geocentric_basis_from_state(
    state: NBodyState,
    earth_index: int,
    moon_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the Moon's instantaneous geocentric orbital basis.

    The basis is the same one used for the initial lunar velocity correction,
    but computed at each integration RHS evaluation from the current
    Earth-Moon state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(r_hat, t_hat, h_hat)`` where ``r_hat`` points from Earth to Moon,
        ``t_hat`` is the prograde tangential direction in the instantaneous
        orbital plane, and ``h_hat`` points along geocentric angular momentum.
    """
    r_geo = state.positions[moon_index] - state.positions[earth_index]
    v_geo = state.velocities[moon_index] - state.velocities[earth_index]

    r_hat = _unit_vector_for_lunar_accel(
        r_geo,
        name="geocentric lunar radius vector",
    )
    h_hat = _unit_vector_for_lunar_accel(
        np.cross(r_geo, v_geo),
        name="geocentric lunar angular momentum",
    )
    t_hat = _unit_vector_for_lunar_accel(
        np.cross(h_hat, r_hat),
        name="geocentric lunar tangential direction",
    )
    return r_hat, t_hat, h_hat


def lunar_geocentric_tangential_direction_from_state(
    state: NBodyState,
    earth_index: int,
    moon_index: int,
) -> np.ndarray:
    """
    Return the Moon's instantaneous prograde geocentric tangential direction.

    Kept as a compatibility helper for the earlier tangential-only empirical
    acceleration term.
    """
    _, t_hat, _ = lunar_geocentric_basis_from_state(
        state,
        earth_index=earth_index,
        moon_index=moon_index,
    )
    return t_hat


def add_earth_moon_empirical_acceleration(
    acc: np.ndarray,
    state: NBodyState,
    *,
    a_r_m_s2: float = 0.0,
    a_t_m_s2: float = 0.0,
    a_h_m_s2: float = 0.0,
    earth_index: int,
    moon_index: int,
) -> np.ndarray:
    """
    Add tiny empirical lunar accelerations in the geocentric orbital basis.

    The parameters are the desired change in relative Earth->Moon acceleration:

        a_rel = a_r * r_hat + a_t * t_hat + a_h * h_hat

    The acceleration is split between Earth and Moon to preserve Earth-Moon
    barycenter linear momentum:

        a_moon  += m_earth / (m_earth + m_moon) * a_rel
        a_earth -= m_moon  / (m_earth + m_moon) * a_rel

    Positive ``a_r`` points from Earth to Moon, positive ``a_t`` is prograde,
    and positive ``a_h`` points along geocentric angular momentum.
    """
    acc = acc.copy()

    r_hat, t_hat, h_hat = lunar_geocentric_basis_from_state(
        state,
        earth_index=earth_index,
        moon_index=moon_index,
    )

    a_rel = (
        float(a_r_m_s2) * r_hat
        + float(a_t_m_s2) * t_hat
        + float(a_h_m_s2) * h_hat
    )

    m_earth = float(state.masses[earth_index])
    m_moon = float(state.masses[moon_index])
    m_total = m_earth + m_moon

    acc[moon_index] += (m_earth / m_total) * a_rel
    acc[earth_index] -= (m_moon / m_total) * a_rel

    return acc


def add_earth_moon_tangential_acceleration(
    acc: np.ndarray,
    state: NBodyState,
    a_t_m_s2: float,
    *,
    earth_index: int,
    moon_index: int,
) -> np.ndarray:
    """
    Add a tiny empirical geocentric lunar along-track acceleration.

    Compatibility wrapper around ``add_earth_moon_empirical_acceleration`` for
    the original one-dimensional acceleration calibration.
    """
    return add_earth_moon_empirical_acceleration(
        acc,
        state,
        a_t_m_s2=a_t_m_s2,
        earth_index=earth_index,
        moon_index=moon_index,
    )


def make_acceleration_with_earth_moon_empirical_term(
    base_accel_func,
    *,
    earth_index: int,
    moon_index: int,
    a_r_m_s2: float = 0.0,
    a_t_m_s2: float = 0.0,
    a_h_m_s2: float = 0.0,
    base_accel_kwargs: dict | None = None,
):
    """
    Wrap an acceleration model with empirical Earth-Moon basis accelerations.

    This leaves the Newtonian / GR / J2 machinery untouched and adds only the
    requested short-range lunar calibration term.
    """
    base_accel_kwargs = dict(base_accel_kwargs or {})

    def accel_with_lunar_empirical_term(
        state: NBodyState,
        G: float = G_SI,
        **_ignored_kwargs,
    ) -> np.ndarray:
        acc = base_accel_func(state, G=G, **base_accel_kwargs)
        return add_earth_moon_empirical_acceleration(
            acc,
            state,
            a_r_m_s2=a_r_m_s2,
            a_t_m_s2=a_t_m_s2,
            a_h_m_s2=a_h_m_s2,
            earth_index=earth_index,
            moon_index=moon_index,
        )

    return accel_with_lunar_empirical_term


def make_acceleration_with_earth_moon_tangential_term(
    base_accel_func,
    *,
    earth_index: int,
    moon_index: int,
    a_t_m_s2: float,
    base_accel_kwargs: dict | None = None,
):
    """
    Wrap an existing acceleration model with a tiny empirical Earth-Moon
    tangential acceleration term.

    Compatibility wrapper for the original one-dimensional acceleration
    calibration.
    """
    return make_acceleration_with_earth_moon_empirical_term(
        base_accel_func,
        earth_index=earth_index,
        moon_index=moon_index,
        a_t_m_s2=a_t_m_s2,
        base_accel_kwargs=base_accel_kwargs,
    )


def integrate_with_accel(state0: NBodyState,
                         t_span: tuple[float, float],
                         dt: float,
                         accel_func,
                         accel_kwargs: dict | None = None,
                         G: float = G_SI,
                         record_every: int = 1,
                         progress=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic integration routine using velocity_verlet_step_generic.

    progress: optional callable(progress_fraction, i, t)
        Called occasionally with:
            progress_fraction: float in [0, 1]
            i: current step index (1-based)
            t: current model time (seconds)
    """
    t0, t1 = t_span
    n_steps = int(np.floor((t1 - t0) / dt))
    n_records = n_steps // record_every + 1

    state = state0.copy()
    acc = accel_func(state, **(accel_kwargs or {}), G=G)

    positions = np.empty((n_records, state0.positions.shape[0], 3), dtype=float)
    velocities = np.empty_like(positions)
    times = np.empty((n_records,), dtype=float)

    positions[0] = state.positions
    velocities[0] = state.velocities
    times[0] = 0.0

    rec_idx = 1
    t = t0

    progress_stride = max(1, n_steps // 100)

    for step in range(1, n_steps + 1):
        # Progress / ETA reporting
        if progress is not None and (step % progress_stride == 0 or step == n_steps):
            frac = step / n_steps if n_steps > 0 else 1.0
            progress(frac, step, t)

        # One velocity–Verlet step with the provided acceleration function
        state, acc = velocity_verlet_step_generic(
            state, acc, dt, accel_func, accel_kwargs=accel_kwargs, G=G
        )
        t += dt

        if step % record_every == 0:
            positions[rec_idx] = state.positions
            velocities[rec_idx] = state.velocities
            times[rec_idx] = t - t0
            rec_idx += 1

    return times, positions, velocities


def state_derivative(state: NBodyState,
                     accel_func,
                     accel_kwargs: dict | None = None,
                     G: float = G_SI) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (dr/dt, dv/dt) for the given state.

    dr/dt = v
    dv/dt = a(state)
    """
    if accel_kwargs is None:
        accel_kwargs = {}

    drdt = state.velocities
    dvdt = accel_func(state, G=G, **accel_kwargs)
    return drdt, dvdt


def rk4_step_generic(state: NBodyState,
                     dt: float,
                     accel_func,
                     accel_kwargs: dict | None = None,
                     G: float = G_SI) -> NBodyState:
    """
    One classical RK4 step for an N-body state.
    """
    if accel_kwargs is None:
        accel_kwargs = {}

    r0 = state.positions
    v0 = state.velocities
    m = state.masses

    # k1
    k1_r, k1_v = state_derivative(state, accel_func, accel_kwargs=accel_kwargs, G=G)

    # k2
    state_k2 = NBodyState(
        positions=r0 + 0.5 * dt * k1_r,
        velocities=v0 + 0.5 * dt * k1_v,
        masses=m,
    )
    k2_r, k2_v = state_derivative(state_k2, accel_func, accel_kwargs=accel_kwargs, G=G)

    # k3
    state_k3 = NBodyState(
        positions=r0 + 0.5 * dt * k2_r,
        velocities=v0 + 0.5 * dt * k2_v,
        masses=m,
    )
    k3_r, k3_v = state_derivative(state_k3, accel_func, accel_kwargs=accel_kwargs, G=G)

    # k4
    state_k4 = NBodyState(
        positions=r0 + dt * k3_r,
        velocities=v0 + dt * k3_v,
        masses=m,
    )
    k4_r, k4_v = state_derivative(state_k4, accel_func, accel_kwargs=accel_kwargs, G=G)

    r_new = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_new = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return NBodyState(
        positions=r_new,
        velocities=v_new,
        masses=m,
    )


def integrate_rk4_with_accel(state0: NBodyState,
                             t_span: tuple[float, float],
                             dt: float,
                             accel_func,
                             accel_kwargs: dict | None = None,
                             G: float = G_SI,
                             record_every: int = 1,
                             progress=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate using classical RK4 and a generic acceleration callback.
    """
    t0, t1 = t_span
    n_steps = int(np.floor((t1 - t0) / dt))
    n_records = n_steps // record_every + 1

    if accel_kwargs is None:
        accel_kwargs = {}

    state = state0.copy()

    N = state.positions.shape[0]
    positions = np.empty((n_records, N, 3), dtype=float)
    velocities = np.empty((n_records, N, 3), dtype=float)
    times = np.empty((n_records,), dtype=float)

    positions[0] = state.positions
    velocities[0] = state.velocities
    times[0] = 0.0

    rec_idx = 1
    t = t0

    progress_stride = max(1, n_steps // 100)

    for step in range(1, n_steps + 1):
        if progress is not None and (step % progress_stride == 0 or step == n_steps):
            frac = step / n_steps if n_steps > 0 else 1.0
            progress(frac, step, t)

        state = rk4_step_generic(
            state,
            dt,
            accel_func,
            accel_kwargs=accel_kwargs,
            G=G,
        )
        t += dt

        if step % record_every == 0:
            positions[rec_idx] = state.positions
            velocities[rec_idx] = state.velocities
            times[rec_idx] = t - t0
            rec_idx += 1

    return times, positions, velocities


def pack_state(state: NBodyState) -> np.ndarray:
    """
    Flatten positions and velocities into a 1D vector for solve_ivp.

    Layout:
        y = [r_0x, r_0y, r_0z, ..., r_Nz, v_0x, v_0y, v_0z, ..., v_Nz]
    """
    return np.concatenate([
        state.positions.reshape(-1),
        state.velocities.reshape(-1),
    ])


def unpack_state(y: np.ndarray, masses: np.ndarray) -> NBodyState:
    """
    Reconstruct NBodyState from a flattened solve_ivp vector.
    """
    n_bodies = len(masses)
    n_coords = 3 * n_bodies

    positions = y[:n_coords].reshape((n_bodies, 3))
    velocities = y[n_coords:].reshape((n_bodies, 3))

    return NBodyState(
        positions=positions,
        velocities=velocities,
        masses=masses,
    )


def rhs_solve_ivp(t: float,
                  y: np.ndarray,
                  masses: np.ndarray,
                  accel_func,
                  accel_kwargs: dict | None = None,
                  G: float = G_SI) -> np.ndarray:
    """
    Right-hand side for solve_ivp:
        dy/dt = [v, a]
    """
    if accel_kwargs is None:
        accel_kwargs = {}

    state = unpack_state(y, masses)
    acc = accel_func(state, G=G, **accel_kwargs)

    dydt = np.concatenate([
        state.velocities.reshape(-1),
        acc.reshape(-1),
    ])
    return dydt


def integrate_dop853_with_accel(state0: NBodyState,
                                t_span: tuple[float, float],
                                dt: float,
                                accel_func,
                                accel_kwargs: dict | None = None,
                                G: float = G_SI,
                                record_every: int = 1,
                                progress=None,
                                rtol: float = 1e-12,
                                atol: float = 1e-15,
                                chunk_duration: float | None = None,
                                max_step: float | None = None,
                                show_progress: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate using SciPy's adaptive DOP853 solver, but in chunks so that
    long runs can show progress and recover more gracefully.

    Parameters
    ----------
    dt : float
        Base output cadence in seconds.
    chunk_duration : float | None
        Duration of each integration chunk in seconds. If None, integrate the
        whole span in one chunk.
    """
    if accel_kwargs is None:
        accel_kwargs = {}

    t0, t1 = t_span
    total_duration = t1 - t0

    if chunk_duration is None or chunk_duration <= 0:
        chunk_duration = total_duration

    if max_step is None or max_step <= 0:
        max_step_solver = np.inf
    else:
        max_step_solver = max_step

    # Fixed nominal output grid, same convention as the rest of the package.
    n_steps = int(np.floor(total_duration / dt))
    full_times = t0 + np.arange(n_steps + 1, dtype=float) * dt
    output_idx = np.arange(0, len(full_times), record_every, dtype=int)
    t_eval_all = full_times[output_idx]

    # Chunk boundaries
    n_chunks = int(math.ceil(total_duration / chunk_duration))
    chunk_edges = [t0 + i * chunk_duration for i in range(n_chunks + 1)]
    chunk_edges[-1] = t1

    y_current = pack_state(state0)

    all_t = []
    all_y = []

    pbar = tqdm(total=n_chunks, disable=not show_progress, desc="DOP853 chunks")

    for chunk_idx in range(n_chunks):
        tc0 = chunk_edges[chunk_idx]
        tc1 = chunk_edges[chunk_idx + 1]

        # Select t_eval points that fall in this chunk
        mask = (t_eval_all >= tc0) & (t_eval_all <= tc1)
        t_eval_chunk = t_eval_all[mask]

        # Avoid duplicate boundary output except for first chunk
        if chunk_idx > 0 and len(t_eval_chunk) > 0 and np.isclose(t_eval_chunk[0], tc0):
            t_eval_chunk = t_eval_chunk[1:]

        sol = solve_ivp(
            fun=lambda t, y: rhs_solve_ivp(
                t, y,
                masses=state0.masses,
                accel_func=accel_func,
                accel_kwargs=accel_kwargs,
                G=G,
            ),
            t_span=(tc0, tc1),
            y0=y_current,
            method="DOP853",
            t_eval=t_eval_chunk if len(t_eval_chunk) > 0 else None,
            rtol=rtol,
            atol=atol,
            vectorized=False,
            dense_output=True,
            max_step=max_step_solver,
        )

        if not sol.success:
            raise RuntimeError(f"DOP853 integration failed in chunk {chunk_idx}: {sol.message}")

        # Store requested outputs
        if sol.t.size > 0:
            all_t.append(sol.t.copy())
            all_y.append(sol.y.T.copy())

        # IMPORTANT:
        # Advance to the exact chunk end, not just the last sampled t_eval point.
        y_current = sol.sol(tc1).copy()

        pbar.update(1)

        if progress is not None:
            frac = (tc1 - t0) / total_duration if total_duration > 0 else 1.0
            progress(frac, min(int(frac * n_steps), n_steps), tc1)

    pbar.close()

    # Concatenate chunk outputs
    if len(all_t) == 0:
        raise RuntimeError("No output samples were produced by DOP853.")

    t_out = np.concatenate(all_t)
    y_out = np.concatenate(all_y, axis=0)

    n_bodies = len(state0.masses)
    n_coords = 3 * n_bodies

    positions = y_out[:, :n_coords].reshape((-1, n_bodies, 3))
    velocities = y_out[:, n_coords:].reshape((-1, n_bodies, 3))
    times = t_out - t0

    return times, positions, velocities


def acceleration_newtonian_vectorized(state: NBodyState,
                                      G: float = G_SI) -> np.ndarray:
    """
    Vectorized Newtonian N-body acceleration.
    """
    r = state.positions                      # (N, 3)
    m = state.masses                        # (N,)
    dr = r[:, None, :] - r[None, :, :]      # (N, N, 3)

    d2 = np.sum(dr * dr, axis=-1)           # (N, N)
    np.fill_diagonal(d2, np.inf)

    inv_r3 = 1.0 / (d2 * np.sqrt(d2))       # (N, N)
    weighted = dr * (m[None, :, None] * inv_r3[:, :, None])

    acc = -G * np.sum(weighted, axis=1)     # (N, 3)
    return acc


def acceleration_newtonian_eih_1pn(state: NBodyState,
                                   G: float = G_SI,
                                   c: float = C_LIGHT) -> np.ndarray:
    """
    Approximate barycentric Einstein-Infeld-Hoffmann (EIH-style) 1PN correction
    for point masses, specialized to beta=gamma=1.

    This is a practical point-mass 1PN model intended to improve barycentric
    consistency relative to a purely Sun-centered correction.

    Notes
    -----
    - Returns Newtonian acceleration + 1PN correction.
    - Intended for Solar-System-style weak-field, slow-motion dynamics.
    - This is not a full finite-size / multipole / J2 / lunar-tidal model.
    """
    r = state.positions
    v = state.velocities
    m = state.masses

    n = len(m)
    acc_newton = acceleration_newtonian_vectorized(state, G=G)

    # Start from Newtonian term
    acc = acc_newton.copy()

    # 1PN correction
    # This is a practical EIH-style implementation with pairwise sums.
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            rij = r[i] - r[j]
            vij = v[i] - v[j]

            r2 = np.dot(rij, rij)
            rmag = np.sqrt(r2)
            rhat = rij / rmag
            inv_r = 1.0 / rmag
            inv_r2 = inv_r * inv_r

            vi2 = np.dot(v[i], v[i])
            vj2 = np.dot(v[j], v[j])
            vidotvj = np.dot(v[i], v[j])
            n_dot_vi = np.dot(rhat, v[i])
            n_dot_vj = np.dot(rhat, v[j])

            # Potential sums over third bodies
            Ui = 0.0
            Uj = 0.0
            for k in range(n):
                if k != i:
                    rik = r[i] - r[k]
                    Ui += G * m[k] / np.linalg.norm(rik)
                if k != j:
                    rjk = r[j] - r[k]
                    Uj += G * m[k] / np.linalg.norm(rjk)

            # EIH-style scalar coefficient along rhat
            A = (
                4.0 * Ui
                + Uj
                - vi2
                - 2.0 * vj2
                + 4.0 * vidotvj
                + 1.5 * (n_dot_vj ** 2)
            )

            # Velocity-dependent coefficient
            B = 4.0 * n_dot_vi - 3.0 * n_dot_vj

            # 1PN pair contribution
            a_1pn_ij = (
                -G * m[j] * inv_r2 / (c * c)
            ) * (A * rhat + B * vij)

            acc[i] += a_1pn_ij

    return acc
