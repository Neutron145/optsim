from dataclasses import dataclass

from .base import Source


@dataclass(frozen=True, kw_only=True)
class PlaneWave(Source):
    """Плоская волна, падающая вдоль оси.

    amplitude0 - амплитуда
    """

    amplitude0: float = 1.0

    def amplitude(self, x, y):
        return self.amplitude0
