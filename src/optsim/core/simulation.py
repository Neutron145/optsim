import warnings

import torch

from .field import Field
from .propagation import fresnel


class Simulation:
    """Описание численной модели: сетка, метод распространения, устройство.

    grid        - описание сетки для проведения расчетов, Grid
    propagator  - метод расчёта распространения световой волны в свободном пространстве
    device      - используемой устройство для расчётов, cpu или gpu
    edge_band   - ширина полосы на краю окна, доля
    edge_tol    - допустимая доля энергии в полосе на краю окна
    """

    def __init__(self, grid, propagator=fresnel, device=None, edge_band=0.05, edge_tol=1e-3):
        self.grid = grid
        self.propagator = propagator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_band = edge_band
        self.edge_tol = edge_tol

    def coords(self):
        """Координаты в пространстве для каждого отсчёта сетки."""
        a = self.grid.axis(self.device)
        return a[None, :], a[:, None]

    def field(self, amplitude, wavelengths, z=0.0):
        """Описание светового поля."""
        wavelengths = tuple(wavelengths)
        data = torch.as_tensor(amplitude, device=self.device).to(torch.complex64)
        data = data.expand(len(wavelengths), self.grid.n, self.grid.n).clone()
        return Field(data, self.grid, wavelengths, z)

    def propagate(self, field, z):
        """Расчет распространения поля на расстояние dz с помощью пропагатора"""

        dz = z - field.z
        self._check_sampling(field.wavelengths, dz)
        data = self.propagator(field.data, self.grid, field.wavelengths, dz)
        out = Field(data, self.grid, field.wavelengths, z)
        self._check_edges(out)
        return out

    def _check_sampling(self, wavelengths, dz):
        for lam in wavelengths:
            z_max = self.grid.dx * self.grid.size / lam
            if abs(dz) > z_max:
                warnings.warn(f"lambda = {lam:.3e} м: dz = {dz:.3e} м больше допустимого {z_max:.3e} м для этой сетки")

    def _check_edges(self, field):
        b = int(self.edge_band * self.grid.n)
        intensity = field.intensity().detach()
        total = intensity.sum(dim=(-2, -1))
        inner = intensity[:, b:-b, b:-b].sum(dim=(-2, -1))
        for lam, frac in zip(field.wavelengths, (1 - inner / total).tolist()):
            if frac > self.edge_tol:
                warnings.warn(f"lambda = {lam:.3e} м: у края окна {frac:.1%} энергии, поле выходит за границы")
