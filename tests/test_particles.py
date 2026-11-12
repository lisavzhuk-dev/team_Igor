import numpy as np
from src.particles import initialize_particles

def test_initialize_particles():
    # Запускаем функцию с параметрами по умолчанию
    positions, velocities, masses = initialize_particles()
    
    # Проверяем, что создалось ровно 16 частиц
    assert len(positions) == 16
    assert len(velocities) == 16
    assert len(masses) == 16
    
    # Проверяем размерность массивов (как требует задание: N на 2 и N)
    assert positions.shape == (16, 2)
    assert velocities.shape == (16, 2)
    assert masses.shape == (16,)
    
    # Проверяем, что начальные скорости равны нулю
    assert np.all(velocities == 0)
    
    # Проверяем, что массы равны 1
    assert np.all(masses == 1)
    
    print("Все тесты для particles.py пройдены успешно!")