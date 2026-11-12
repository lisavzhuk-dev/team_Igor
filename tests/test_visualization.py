import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # безголовый backend — чтобы тесты работали без окна
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization import plot_all, plot_particles, plot_energy


def test_plot_particles_runs():
    pos = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
    ax = plot_particles(pos, show=False)
    assert ax is not None
    plt.close("all")


def test_plot_energy_runs():
    t = np.linspace(0, 1, 10)
    e = np.ones_like(t) * 3.0
    ax = plot_energy(t, e, show=False)
    assert ax is not None
    plt.close("all")


def test_plot_particles_with_trajectories():
    pos = np.array([[0.0, 0.0], [1.5, 0.0]])
    traj = np.stack([pos + 0.01 * k for k in range(5)])  # (5, 2, 2)
    ax = plot_particles(pos, trajectories=traj, show=False)
    assert ax is not None
    plt.close("all")


def test_plot_particles_rejects_mismatched_trajectory_shape():
    pos = np.array([[0.0, 0.0], [1.5, 0.0]])
    traj = np.zeros((5, 3, 2))

    with np.testing.assert_raises(ValueError):
        plot_particles(pos, trajectories=traj, show=False)


def test_plot_energy_rejects_different_series_lengths():
    with np.testing.assert_raises(ValueError):
        plot_energy(np.array([0.0, 1.0]), np.array([3.0]), show=False)


def test_plot_all_returns_and_saves_two_panel_figure(tmp_path):
    positions = np.array([[0.0, 0.0], [1.5, 0.0]])
    time = np.linspace(0, 1, 3)
    energy = np.ones_like(time)
    filename = tmp_path / "visualization.png"

    fig = plot_all(positions, time, energy, filename=filename, show=False)

    assert len(fig.axes) == 2
    assert filename.is_file()
    plt.close(fig)