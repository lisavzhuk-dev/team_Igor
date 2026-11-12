import numpy as np
import os
from src.particles import initialize_particles
from src.forces import compute_forces
from src.integrator import velocity_verlet
from src.visualization import save_for_ovito

def main():
    print("Запуск моделирования...")
    
    # 1. Создаем систему (16 частиц)
    n_particles = 16
    positions, velocities, masses = initialize_particles(n=n_particles)
    
    # Параметры из задания
    dt = 0.001
    steps = 5000
    
    times = []
    energies = []
    trajectories = [] 
    
    print(f"Начинаем расчет {steps} шагов...")
    
    for step in range(steps):
        # 2. Шаг интегратора
        positions, velocities = velocity_verlet(positions, velocities, masses, dt, compute_forces)
        
        # Считаем энергию
        _, potential_energy = compute_forces(positions)
        kinetic_energy = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        total_energy = kinetic_energy + potential_energy
        
        times.append(step * dt)
        energies.append(total_energy)
        
        # Сохраняем координаты (можно каждый шаг или каждый 10-й)
        if step % 10 == 0:
            trajectories.append(positions.copy())
        
    print("Моделирование завершено!")
    print(f"Полная энергия в начале: {energies[0]:.4f}")
    print(f"Полная энергия в конце:  {energies[-1]:.4f}")
    
    # 3. Сохранение для OVITO
    # Создаем папку results, если ее нет
    if not os.path.exists("results"):
        os.makedirs("results")
        
    trajectories = np.array(trajectories)
    save_for_ovito("results/trajectory.xyz", trajectories)

if __name__ == "__main__":
    main()