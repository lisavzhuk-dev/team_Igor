import numpy as np

def save_for_ovito(filename, trajectories):
    """
    Сохраняет траектории частиц в формате XYZ для программы OVITO.
    
    Параметры:
    -----------
    filename : str
        Имя файла для сохранения (например, "results/trajectory.xyz")
    trajectories : np.ndarray, shape (T, N, 2)
        История координат частиц (T - шагов, N - частиц, 2 - координаты x, y)
    """
    # Проверяем, что нам передали правильный массив
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError("trajectories must have shape (T, N, 2)")

    T, N, _ = trajectories.shape

    with open(filename, "w") as f:
        for t in range(T):
            # Первая строка кадра - количество частиц
            f.write(f"{N}\n")
            # Вторая строка кадра - комментарий (можно указать номер шага)
            f.write(f"Step {t}\n")
            
            # Записываем координаты каждой частицы
            for i in range(N):
                x = trajectories[t, i, 0]
                y = trajectories[t, i, 1]
                # OVITO ожидает минимум 3 координаты (x, y, z), поэтому z ставим 0
                f.write(f"Type1 {x:.6f} {y:.6f} 0.0\n")
                
    print(f"Данные успешно сохранены в файл: {filename}")