"""Радиус кольца после собирающего аксикона.

Луч с радиуса r после аксикона отклоняется к оси на угол λ/p и в плоскости z
приходит на радиус λz/p - r (при λz/p > r). Энергия переносится вдоль лучей,
поэтому энергетический центроид радиуса R_c = Σ r·I / Σ I на выходе равен
λz/p - R_c на входе. Отличие даёт только дифракция. Допуск 1% (тр. 7).

z = 30 мм выбрано внутри критерия дискретизации ядра (≈ 39 мм при 660 нм).
"""

import warnings

import pytest
import torch

from optsim.core import Grid, Simulation
from optsim.physics.elements import Axicon, Ring
from optsim.physics.sources import PlaneWave
from optsim.physics.system import System

P = 10e-6
Z = 30e-3
LAMS = (450e-9, 660e-9)


def centroid_radius(sim, field):
    """Энергетический центроид радиуса для каждой λ, м."""
    x, y = sim.coords()
    r = torch.sqrt(x**2 + y**2)
    intensity = field.intensity().double()
    return ((intensity * r).sum(dim=(-2, -1)) / intensity.sum(dim=(-2, -1))).tolist()


def test_ring_radius():
    sim = Simulation(Grid(n=4096, dx=2.5e-6))
    source = PlaneWave(wavelengths=LAMS)
    system = System([(0.0, Axicon(period=P, shape=Ring(r_in=0.4e-3, r_out=0.6e-3)))])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r_in = centroid_radius(sim, system.run(source, 0.0, sim))
        r_out = centroid_radius(sim, system.run(source, Z, sim))

    expected = [lam * Z / P - r for lam, r in zip(LAMS, r_in)]
    assert r_out == pytest.approx(expected, rel=1e-2)
