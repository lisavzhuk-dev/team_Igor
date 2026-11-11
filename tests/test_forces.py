import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forces import compute_forces, check_newton_third_law


def test_forces_shape():
    pos = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
    forces, u = compute_forces(pos)
    assert forces.shape == (3, 2)
    assert isinstance(u, float)


def test_newton_third_law():
    pos = np.array([[0.0, 0.0], [1.2, 0.0], [0.0, 1.2], [1.2, 1.2]])
    assert check_newton_third_law(pos)


def test_repulsion_at_short_distance():
    # При очень малом расстоянии частицы должны отталкиваться
    pos = np.array([[0.0, 0.0], [0.5, 0.0]])
    forces, _ = compute_forces(pos)
    # Сила на первую частицу должна быть направлена от второй (вдоль -x)
    assert forces[0, 0] < 0
    assert forces[1, 0] > 0