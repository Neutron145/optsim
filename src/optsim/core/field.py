from dataclasses import dataclass

import torch

from .grid import Grid


@dataclass(frozen=True)
class Field:
    """Световое поле в плоскости z.

    data        — комплексная амплитуда [Nλ, n, n], complex64
    wavelengths — длины волн, м (по одной на каждый слой data)
    z           — положение плоскости на оптической оси, м
    """

    data: torch.Tensor
    grid: Grid
    wavelengths: tuple
    z: float

    def intensity(self):
        """|U|², [Nλ, n, n]."""
        return self.data.abs() ** 2

    def energy(self):
        """Σ|U|²·dx² для каждой длины волны, [Nλ]."""
        return self.intensity().sum(dim=(-2, -1)) * self.grid.dx**2
