import numpy as np
import pytest

from src.integrator import velocity_verlet


def _zero_forces(positions):
    n = positions.shape[0]
    return np.zeros((n, 2)), 0.0


def _constant_forces(positions):
    n = positions.shape[0]
    return np.tile([1.0, 0.0], (n, 1)), 0.0


def test_no_forces_linear_motion():
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    velocities = np.array([[1.0, 2.0], [-1.0, 0.5]])
    masses = np.array([1.0, 1.0])
    dt = 0.1

    new_pos, new_vel = velocity_verlet(
        positions, velocities, masses, dt, _zero_forces
    )

    np.testing.assert_allclose(new_vel, velocities, atol=1e-12)
    expected_pos = positions + velocities * dt
    np.testing.assert_allclose(new_pos, expected_pos, atol=1e-12)


def test_constant_force_uniform_acceleration():
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    masses = np.array([1.0])
    dt = 0.1

    new_pos, new_vel = velocity_verlet(
        positions, velocities, masses, dt, _constant_forces
    )

    np.testing.assert_allclose(new_pos, [[0.005, 0.0]], atol=1e-12)
    np.testing.assert_allclose(new_vel, [[0.1, 0.0]], atol=1e-12)


def test_returns_shapes():
    rng = np.random.default_rng(0)
    n = 8
    positions = rng.random((n, 2))
    velocities = rng.random((n, 2))
    masses = np.ones(n)

    new_pos, new_vel = velocity_verlet(
        positions, velocities, masses, 1e-3, _zero_forces
    )

    assert new_pos.shape == (n, 2)
    assert new_vel.shape == (n, 2)


def test_mass_dependence():
    positions = np.zeros((2, 2))
    velocities = np.zeros((2, 2))
    masses = np.array([1.0, 2.0])
    dt = 0.1

    new_pos, _ = velocity_verlet(
        positions, velocities, masses, dt, _constant_forces
    )

    np.testing.assert_allclose(new_pos[0], [0.005, 0.0], atol=1e-12)
    np.testing.assert_allclose(new_pos[1], [0.0025, 0.0], atol=1e-12)


def test_energy_conservation_with_harmonic_force():
    k = 1.0

    def harmonic_forces(positions):
        forces = -k * positions
        ep = 0.5 * k * np.sum(positions ** 2)
        return forces, ep

    masses = np.array([1.0, 1.0])
    positions = np.array([[0.1, 0.0], [-0.1, 0.0]])
    velocities = np.zeros((2, 2))
    dt = 1e-3

    def total_energy(r, v):
        _, ep = harmonic_forces(r)
        ek = 0.5 * np.sum(masses[:, np.newaxis] * v ** 2)
        return ek + ep

    e0 = total_energy(positions, velocities)
    for _ in range(1000):
        positions, velocities = velocity_verlet(
            positions, velocities, masses, dt, harmonic_forces
        )
    e1 = total_energy(positions, velocities)

    assert abs(e1 - e0) / abs(e0) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])