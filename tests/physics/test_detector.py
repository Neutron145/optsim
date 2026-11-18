"""Проверка детектора: сигнал пикселя = интеграл интенсивности по его площади, сумма по λ."""

import pytest
import torch

from optsim.core import Grid, Simulation
from optsim.physics.detector import Detector
from optsim.physics.sources import GaussianBeam

LAMS = (450e-9, 660e-9)


@pytest.fixture
def sim():
    return Simulation(Grid(n=1024, dx=2.5e-6))  # окно 2.56 мм


def test_uniform_field(sim):
    """Равномерное поле a: каждый пиксель = Nλ·a²·pixel²."""
    a = 3.0
    detector = Detector(z=0.0, pixel=10e-6, n_pixels=100)
    signal = detector.measure(sim.field(a, LAMS))
    expected = torch.full_like(signal, len(LAMS) * a**2 * detector.pixel**2)
    torch.testing.assert_close(signal, expected, rtol=1e-6, atol=0)


def test_total_energy(sim):
    """Гауссов пучок целиком внутри матрицы (полуширина 1 мм = 5·w0): сумма пикселей = энергия."""
    field = GaussianBeam(wavelengths=LAMS, w0=0.2e-3).field(sim)
    detector = Detector(z=0.0, pixel=10e-6, n_pixels=200)
    signal = detector.measure(field)
    assert signal.sum().item() == pytest.approx(field.energy().sum().item(), rel=1e-4)
