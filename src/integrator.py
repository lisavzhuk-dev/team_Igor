import numpy as np

def velocity_verlet(positions, velocities, masses, dt, compute_forces):
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities,dtype=float)
    masses = np.asarray(masses, dtype=float)

    n = positions.shape[0]
    assert positions.shape == (n,2),
    assert velocities.shape == (n,2),
    assert masses.shape == (n,),

    forces, _ = compute_forces(positions)
    forces =np.asarray(forces, dtype=float)
    assert forces.shape == (n,2),
    accelerations = forces / masses[:,np.newaxis]

    new_positions = (
        positions
        + velocities * dt
        +0.5 * accelerations * dt * dt
    )

    new_forces, _ = compute_forces(new_positions)
    new_forces = np.asarray(new_forces, dtype=float)
    assert new_forces.shape == (n,2),

    new_accelerations = new_forces/ masses[:,np.newaxis]

    new_velocities = (
        velocities
        +0.5 * (accelerations+new_accelerations)*dt
    )

    return new_positions, new_velocities