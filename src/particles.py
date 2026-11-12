import numpy as np

def initialize_particles(n=16, spacing=1.2):
    side = int(np.ceil(np.sqrt(n)))
    positions_list = []
    
    # Сдвигаем сетку на 0.5 от левого нижнего угла, чтобы частицы не были на стенках
    offset = 0.5 
    
    for i in range(side):
        for j in range(side):
            if len(positions_list) < n:
                positions_list.append([i * spacing + offset, j * spacing + offset])
    
    positions = np.array(positions_list, dtype=float)
    velocities = np.random.normal(0, 0.5, (n, 2))
    masses = np.ones(n)
    
    return positions, velocities, masses