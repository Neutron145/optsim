import math
from dataclasses import dataclass

import torch

from optsim.core import expi

from .base import Element


@dataclass(frozen=True, kw_only=True)
class Axicon(Element):
    """Дифрационный аксикон

    period     - период, м
    converging - True: собирающий, False: рассеивающий
    """

    period: float
    converging: bool = True

    def profile(self, x, y, lam):
        sign = -1 if self.converging else 1
        return expi(sign * 2 * math.pi * torch.sqrt(x**2 + y**2) / self.period)
