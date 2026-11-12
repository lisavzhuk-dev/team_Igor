import numpy as np
# Импортируем ваши готовые модули (Участники 1, 2, 3)
from src.particles import initialize_particles
from src.forces import compute_forces
from src.integrator import velocity_verlet

# Попробуем импортировать модуль визуализации (Участник 4). 
# Если его код еще не готов или сломан, программа не вылетит, а пойдет дальше.
try:
    from src.visualization import plot_particles, plot_energy
    HAS_VISUALIZATION = True
except ImportError:
    print("Внимание: модуль визуализации еще не готов. Графики строиться не будут.")
    HAS_VISUALIZATION = False

def main():
    print("Запуск моделирования...")
    
    # 1. Создаем систему (16 частиц)
    n_particles = 16
    positions, velocities, masses = initialize_particles(n=n_particles)
    
    # Параметры из задания
    dt = 0.001
    steps = 1000
    
    # Массивы для сохранения энергии на каждом шаге
    times = []
    energies = []
    
    print(f"Начинаем расчет {steps} шагов...")
    
    for step in range(steps):
        # 2. Выполняем шаг интегратором (считаем новые координаты и скорости)
        positions, velocities = velocity_verlet(positions, velocities, masses, dt, compute_forces)
        
        # Считаем энергию для проверки закона сохранения
        _, potential_energy = compute_forces(positions)
        kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        total_energy = kinetic_energy + potential_energy
        
        # Сохраняем в массивы
        times.append(step * dt)
        energies.append(total_energy)
        
    print("Моделирование завершено!")
    print(f"Полная энергия в начале: {energies[0]:.4f}")
    print(f"Полная энергия в конце:  {energies[-1]:.4f}")
    
    # 3. Визуализация (если модуль готов)
    if HAS_VISUALIZATION:
        print("Строим графики...")
        plot_particles(positions)
        plot_energy(times, energies)
    else:
        print("Графики не построены, так как модуль visualization.py в разработке.")

if __name__ == "__main__":
    main()