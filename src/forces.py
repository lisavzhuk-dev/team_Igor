import numpy as np


def compute_forces(positions, epsilon=1.0, sigma=1.0):
    """
    Вычисление сил Леннарда-Джонса и потенциальной энергии системы.

    Параметры
    ---------
    positions : np.ndarray, shape (N, 2)
        Координаты частиц (x, y).
    epsilon : float
        Глубина потенциальной ямы (по умолчанию 1.0).
    sigma : float
        Характерный параметр длины (по умолчанию 1.0).

    Возвращает
    ----------
    forces : np.ndarray, shape (N, 2)
        Результирующие силы, действующие на каждую частицу.
    potential_energy : float
        Полная потенциальная энергия системы.
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]

    forces = np.zeros_like(positions)
    potential_energy = 0.0

    # Перебор всех пар частиц (i < j), чтобы не считать дважды
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Вектор между частицами: r_ij = r_i - r_j
            r_ij = positions[i] - positions[j]
            r = np.linalg.norm(r_ij)

            # Защита от деления на ноль (совпадающие координаты)
            if r < 1e-12:
                continue

            # Отношение sigma / r
            sr = sigma / r
            sr6 = sr ** 6
            sr12 = sr6 ** 2

            # Потенциал Леннарда-Джонса: U(r) = 4*eps*(sr^12 - sr^6)
            potential_energy += 4.0 * epsilon * (sr12 - sr6)

            # Сила: F_ij = (24*eps / r^2) * (2*sr^12 - sr^6) * r_ij
            # (согласно формуле (6) из методички)
            f_scalar = (24.0 * epsilon / (r ** 2)) * (2.0 * sr12 - sr6)
            f_vec = f_scalar * r_ij

            # Третий закон Ньютона: F_ij = -F_ji
            forces[i] += f_vec
            forces[j] -= f_vec

    return forces, potential_energy


def check_newton_third_law(positions, epsilon=1.0, sigma=1.0, tol=1e-10):
    """
    Проверка третьего закона Ньютона: сумма всех внутренних сил должна быть
    равна нулю (импульс системы сохраняется).

    Возвращает True, если проверка пройдена, иначе False.
    """
    forces, _ = compute_forces(positions, epsilon, sigma)
    total_force = np.sum(forces, axis=0)
    return np.linalg.norm(total_force) < tol


# ============================================================
# Блок для проверки/демонстрации работы модуля
# ============================================================
if __name__ == "__main__":
    # Тестовая система из 4 частиц (для наглядности)
    test_positions = np.array([
        [0.0, 0.0],
        [1.2, 0.0],
        [0.0, 1.2],
        [1.2, 1.2],
    ])

    forces, u = compute_forces(test_positions)
    print("Координаты частиц:")
    print(test_positions)
    print("\nСилы, действующие на частицы:")
    print(forces)
    print(f"\nПотенциальная энергия системы: {u:.6f}")
    print(f"Сумма всех сил (должна быть ~0): {np.sum(forces, axis=0)}")
    print(f"Третий закон Ньютона выполняется: "
          f"{check_newton_third_law(test_positions)}")