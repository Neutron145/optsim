from dataclasses import dataclass

import torch

from .grid import Grid


@dataclass(frozen=True)
class Field:
    """Описание светового поля в плоскости z.

    data        - комплексная амплитуда, complex64
    wavelengths - длины волн, м
    z           - положение плоскости на оптической оси, м
    """

    data: torch.Tensor
    grid: Grid
    wavelengths: tuple[float]
    z: float

    def intensity(self):
        return self.data.abs() ** 2

    def energy(self):
        return self.intensity().sum(dim=(-2, -1)) * self.grid.dx**2
