import numpy as np

def initialize_particles(n=16, spacing=1.5):
    # Считаем, сколько частиц поместится в ряд (корень из 16 = 4)
    side = int(np.ceil(np.sqrt(n)))
    
    positions_list = []
    
    # Расставляем частицы в виде сетки (решетки)
    for i in range(side):
        for j in range(side):
            if len(positions_list) < n:
                positions_list.append([i * spacing, j * spacing])
    
    positions = np.array(positions_list, dtype=float)
    
    # --- ИЗМЕНЕНИЕ: Генерируем случайные скорости ---
    # np.random.normal(среднее, разброс, размер)
    # Это создаст скорости, которые в среднем близки к нулю, но у каждой частицы будут свои.
    velocities = np.random.normal(0, 0.5, (n, 2))
    
    masses = np.ones(n)
    
    return positions, velocities, masses