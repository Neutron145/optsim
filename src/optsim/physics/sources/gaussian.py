from dataclasses import dataclass

import torch

from .base import Source


@dataclass(frozen=True, kw_only=True)
class GaussianBeam(Source):
    """Гауссов пучок.

    w0         - радиус пучка, м
    amplitude0 - амплитуда на оси
    """

    w0: float
    amplitude0: float = 1.0

    def amplitude(self, x, y):
        return self.amplitude0 * torch.exp(-(x**2 + y**2) / self.w0**2)
