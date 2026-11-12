import numpy as np

def compute_forces(positions, epsilon=1.0, sigma=1.0, box_size=5.0):
    """
    Вычисление сил Леннарда-Джонса и потенциальной энергии системы.
    Добавлены стенки коробки (отталкивание от границ 0..box_size).
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]

    forces = np.zeros_like(positions)
    potential_energy = 0.0

    # --- 1. Силы взаимодействия между частицами ---
    for i in range(n - 1):
        for j in range(i + 1, n):
            r_ij = positions[i] - positions[j]
            r = np.linalg.norm(r_ij)

            if r < 1e-12:
                continue

            sr = sigma / r
            sr6 = sr ** 6
            sr12 = sr6 ** 2

            potential_energy += 4.0 * epsilon * (sr12 - sr6)

            f_scalar = (24.0 * epsilon / (r ** 2)) * (2.0 * sr12 - sr6)
            f_vec = f_scalar * r_ij

            forces[i] += f_vec
            forces[j] -= f_vec

    # --- 2. Силы отталкивания от стенок коробки ---
    wall_stiffness = 50.0 # Жесткость стенок (чем больше, тем сильнее отталкивают)
    wall_radius = 0.5     # Расстояние, на котором начинает действовать стенка

    for i in range(n):
        x, y = positions[i]
        
        # Левая стенка (x=0)
        if x < wall_radius:
            forces[i, 0] += wall_stiffness * (wall_radius - x)
        # Правая стенка (x=box_size)
        if x > box_size - wall_radius:
            forces[i, 0] -= wall_stiffness * (x - (box_size - wall_radius))
            
        # Нижняя стенка (y=0)
        if y < wall_radius:
            forces[i, 1] += wall_stiffness * (wall_radius - y)
        # Верхняя стенка (y=box_size)
        if y > box_size - wall_radius:
            forces[i, 1] -= wall_stiffness * (y - (box_size - wall_radius))

    return forces, potential_energy