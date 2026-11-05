import numpy as np

def initialize_particles(n=16, spacing=1.5):
    # Считаем, сколько частиц поместится в ряд (корень из 16 = 4)
    side = int(np.ceil(np.sqrt(n)))
    
    positions_list = []
    
    # Расставляем частицы в виде сетки (решетки)
    for i in range(side):
        for j in range(side):
            if len(positions_list) < n:
                # Координаты x и y для каждой частицы
                positions_list.append([i * spacing, j * spacing])
    
    # Превращаем список в массив NumPy (как требует задание)
    positions = np.array(positions_list, dtype=float)
    
    # Начальные скорости: все нули (частицы сначала покоятся)
    velocities = np.zeros((n, 2))
    
    # Массы: все равны 1 (как сказано в задании)
    masses = np.ones(n)
    
    return positions, velocities, masses