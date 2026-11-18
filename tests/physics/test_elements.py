"""Проверка форм и апертуры по аналитическим площадям.

Допуск 1%: бинарная маска даёт ступенчатый край, ошибка площади ~ периметр·dx.
"""

import dataclasses
import math

import pytest

from optsim.core import Grid, Simulation
from optsim.physics.elements import Aperture, Circle, Rect, Ring

R = 1e-3


@pytest.fixture
def sim():
    return Simulation(Grid(n=1024, dx=2.5e-6))  # окно 2.56 мм


@pytest.mark.parametrize(
    "shape, area",
    [
        (Circle(r=R), math.pi * R**2),
        (Ring(r_in=R / 2, r_out=R), math.pi * (R**2 - (R / 2) ** 2)),
        (Rect(width=2 * R, height=R), 2 * R**2),
    ],
)
def test_shape_area(sim, shape, area):
    x, y = sim.coords()
    measured = shape.mask(x, y).sum().item() * sim.grid.dx**2
    assert measured == pytest.approx(area, rel=1e-2)


def test_aperture_energy(sim):
    a = 3.0
    field = sim.field(a, [450e-9, 660e-9])
    x, y = sim.coords()
    t = Aperture(shape=Circle(r=R)).transmission(x, y, None)
    out = dataclasses.replace(field, data=field.data * t)
    expected = a**2 * math.pi * R**2
    assert out.energy().tolist() == pytest.approx([expected, expected], rel=1e-2)
