"""Проверка ядра по аналитическим результатам (тр. 7).

Гауссов пучок: w(z) = w0·√(1 + (z/zR)²), zR = πw0²/λ
(Saleh, Teich, «Fundamentals of Photonics», гл. 3).
"""

import math
import warnings

import pytest
import torch

from optsim.core import Grid, Simulation, angular_spectrum, fresnel

LAM = 633e-9
W0 = 50e-6
Z_R = math.pi * W0**2 / LAM  # ≈ 12.4 мм


def gaussian(sim, fx=0.0):
    """Гауссов пучок в z = 0; fx — наклон (пространственная частота), 1/м."""
    x, y = sim.coords()
    amplitude = torch.exp(-(x**2 + y**2) / W0**2) * torch.exp(2j * math.pi * fx * x)
    return sim.field(amplitude, [LAM])


def beam_width(sim, field):
    """Ширина по второму моменту: для гауссова пучка w = 2·√<x²>."""
    x, _ = sim.coords()
    intensity = field.intensity()[0].double()
    return 2 * math.sqrt(((intensity * x**2).sum() / intensity.sum()).item())


# Окно 4 мм; на z = zR критерий дискретизации выполнен (z_max ≈ 12.9 мм)
def make_sim(propagator):
    return Simulation(Grid(n=2048, dx=2e-6), propagator)


@pytest.mark.parametrize("propagator", [fresnel, angular_spectrum])
def test_gaussian_width(propagator):
    sim = make_sim(propagator)
    out = sim.propagate(gaussian(sim), Z_R)
    expected = W0 * math.sqrt(2)
    assert abs(beam_width(sim, out) - expected) / expected < 0.01


@pytest.mark.parametrize("propagator", [fresnel, angular_spectrum])
def test_energy_conserved(propagator):
    sim = make_sim(propagator)
    field = gaussian(sim)
    out = sim.propagate(field, Z_R)
    assert abs(out.energy() / field.energy() - 1).item() < 1e-4


def test_no_warnings_for_valid_setup():
    sim = make_sim(fresnel)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sim.propagate(gaussian(sim), Z_R)


def test_sampling_warning():
    sim = make_sim(fresnel)
    with pytest.warns(UserWarning, match="больше допустимого"):
        sim.propagate(gaussian(sim), 2 * Z_R)


def test_edge_warning():
    # Наклонный пучок смещается на λ·z·fx ≈ 0.8 мм при половине окна ≈ 1 мм
    sim = Simulation(Grid(n=512, dx=4e-6))
    with pytest.warns(UserWarning, match="выходит за границы"):
        sim.propagate(gaussian(sim, fx=31250), 0.04)
