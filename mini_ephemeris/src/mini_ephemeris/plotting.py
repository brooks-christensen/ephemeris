from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_xy(positions: np.ndarray,
                       labels: list[str] | None = None,
                       title: str = "Trajectories (xy-plane)",
                       filename: str | None = None,
                       show: bool = True) -> None:
    M, N, _ = positions.shape
    if labels is None:
        labels = [f"body {i}" for i in range(N)]

    fig, ax = plt.subplots()
    for i in range(N):
        x = positions[:, i, 0] / 1.496e11
        y = positions[:, i, 1] / 1.496e11
        ax.plot(x, y, label=labels[i])

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", "box")

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_xy_raw(positions: np.ndarray,
                           labels: list[str] | None = None,
                           title: str = "Trajectories (xy-plane, raw units)",
                           filename: str | None = None,
                           show: bool = True) -> None:
    M, N, _ = positions.shape
    if labels is None:
        labels = [f"body {i}" for i in range(N)]

    fig, ax = plt.subplots()
    for i in range(N):
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        ax.plot(x, y, label=labels[i])

    ax.set_xlabel("x [units]")
    ax.set_ylabel("y [units]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", "box")

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_position_error(times_s: np.ndarray,
                        errors_m: np.ndarray,
                        title: str = "Position error vs time",
                        filename: str | None = None,
                        show: bool = True) -> None:
    days = times_s / 86400.0
    errors_km = errors_m / 1e3

    fig, ax = plt.subplots()
    ax.plot(days, errors_km)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("position error [km]")
    ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_log_error(times_s: np.ndarray,
                   errors_m: np.ndarray,
                   title: str = "Log position separation vs time",
                   filename: str | None = None,
                   show: bool = True) -> None:
    years = times_s / (86400.0 * 365.25)
    err = np.maximum(errors_m, 1e-9)
    log10_err = np.log10(err)

    fig, ax = plt.subplots()
    ax.plot(years, log10_err)
    ax.set_xlabel("time [years]")
    ax.set_ylabel("log10 separation [m]")
    ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_energy(times_s: np.ndarray,
                energies: np.ndarray,
                title: str = "Total energy vs time",
                filename: str | None = None,
                show: bool = True) -> None:
    days = times_s / 86400.0
    E0 = energies[0]
    rel = (energies - E0) / abs(E0)

    fig, ax = plt.subplots()
    ax.plot(days, rel)
    ax.set_xlabel("time [days]")
    ax.set_ylabel("(E - E0) / |E0|")
    ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_poincare_section(x: np.ndarray,
                          vx: np.ndarray,
                          title: str = "Poincaré section",
                          filename: str | None = None,
                          show: bool = True) -> None:
    fig, ax = plt.subplots()
    ax.scatter(x, vx, s=5)
    ax.set_xlabel("x")
    ax.set_ylabel("v_x")
    ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)