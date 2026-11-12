import matplotlib.pyplot as plt
import numpy as np


def _validate_particle_data(positions, trajectories=None):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")

    if trajectories is not None:
        trajectories = np.asarray(trajectories, dtype=float)
        if (trajectories.ndim != 3
                or trajectories.shape[1:] != positions.shape):
            raise ValueError("trajectories must have shape (T, N, 2)")

    return positions, trajectories


def _validate_energy_data(time, energy):
    time = np.asarray(time, dtype=float)
    energy = np.asarray(energy, dtype=float)
    if time.ndim != 1 or energy.ndim != 1 or time.shape != energy.shape:
        raise ValueError("time and energy must be 1D arrays of equal length")

    return time, energy


def plot_particles(positions, trajectories=None, ax=None, show=True,
                   title="Частицы системы", color="tab:blue", size=80):
    """
    Отображение частиц на плоскости (2D).

    Параметры
    ---------
    positions : np.ndarray, shape (N, 2)
        Текущие координаты частиц.
    trajectories : np.ndarray, shape (T, N, 2), optional
        История координат (для отрисовки траекторий). Если None — только точки.
    ax : matplotlib.axes.Axes, optional
        Оси для рисования. Если None — создаются новые.
    show : bool
        Вызвать ли plt.show() в конце.
    title : str
        Заголовок графика.
    color : str
        Цвет точек.
    size : float
        Размер маркера.

    Возвращает
    ----------
    ax : matplotlib.axes.Axes
        Оси с нанесённым графиком.
    """
    positions, trajectories = _validate_particle_data(positions, trajectories)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # --- Траектории (если переданы) ---
    if trajectories is not None:
        # trajectories имеет форму (T, N, 2): T — число кадров, N — число частиц
        # Рисуем траекторию каждой частицы отдельной линией
        for i in range(trajectories.shape[1]):
            ax.plot(trajectories[:, i, 0],
                    trajectories[:, i, 1],
                    lw=0.8, alpha=0.6)

    # --- Текущие положения частиц ---
    ax.scatter(positions[:, 0], positions[:, 1],
               s=size, c=color, edgecolors="black", zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax


def plot_energy(time, energy, ax=None, show=True,
                title="Полная энергия системы"):
    """
    График полной энергии от времени.

    Параметры
    ---------
    time : np.ndarray, shape (T,)
        Массив моментов времени.
    energy : np.ndarray, shape (T,)
        Значения полной энергии в эти моменты.
    ax : matplotlib.axes.Axes, optional
        Оси для рисования.
    show : bool
        Вызвать ли plt.show() в конце.
    title : str
        Заголовок графика.

    Возвращает
    ----------
    ax : matplotlib.axes.Axes
        Оси с нанесённым графиком.
    """
    time, energy = _validate_energy_data(time, energy)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(time, energy, color="tab:red", lw=1.5)

    # Полезно видеть, насколько энергия "плывёт": подписываем отклонение
    if energy.size > 0:
        drift = energy[-1] - energy[0]
        ax.set_title(f"{title}\nΔE = {drift:+.3e}")
    else:
        ax.set_title(title)

    ax.set_xlabel("Время, t")
    ax.set_ylabel("Энергия, E")
    ax.grid(True, alpha=0.3)

    if show:
        plt.show()

    return ax


def plot_all(positions, time, energy, trajectories=None,
             filename=None, show=True):
    """
    Удобная обёртка: рисует частицы с траекториями и энергию
    на двух панелях одной фигуры.

    Параметры
    ---------
    positions : np.ndarray, (N, 2)
        Текущие координаты.
    time : np.ndarray, (T,)
        Времена.
    energy : np.ndarray, (T,)
        Полная энергия в эти моменты.
    trajectories : np.ndarray, (T, N, 2), optional
        История координат.
    filename : str, optional
        Если задано — сохраняет фигуру в файл.
    show : bool
        Вызвать ли plt.show().

    Возвращает
    ----------
    fig : matplotlib.figure.Figure
        Фигура с графиками.
    """
    positions, trajectories = _validate_particle_data(positions, trajectories)
    time, energy = _validate_energy_data(time, energy)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_particles(positions, trajectories=trajectories,
                   ax=axes[0], show=False,
                   title="Частицы и траектории")
    plot_energy(time, energy, ax=axes[1], show=False)

    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ============================================================
# Блок для проверки/демонстрации работы модуля
# ============================================================
if __name__ == "__main__":
    # Простая демонстрация: 4 частицы + фиктивная история
    positions = np.array([
        [0.0, 0.0],
        [1.2, 0.0],
        [0.0, 1.2],
        [1.2, 1.2],
    ])

    # Фиктивные траектории: T=50 кадров, N=4, 2 координаты
    rng = np.random.default_rng(0)
    trajectories = positions[None, :, :] + np.cumsum(
        rng.normal(scale=0.02, size=(50, 4, 2)), axis=0
    )

    time = np.linspace(0, 1, 50)
    # Энергия сохраняется с точностью до численной погрешности
    energy = np.full_like(time, 3.14) + rng.normal(scale=1e-6, size=time.size)

    plot_all(positions, time, energy, trajectories=trajectories)