from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Grid:
    """Квадратная сетка n×n с шагом dx (м). Центр — в пикселе n // 2."""

    n: int
    dx: float

    @property
    def size(self):
        """Сторона расчётной области, м."""
        return self.n * self.dx

    def axis(self, device="cpu"):
        """Координаты узлов вдоль одной оси, м."""
        return (torch.arange(self.n, dtype=torch.float64, device=device) - self.n // 2) * self.dx

    def freqs(self, device="cpu"):
        """Пространственные частоты вдоль одной оси в порядке FFT, 1/м."""
        return torch.fft.fftfreq(self.n, d=self.dx, dtype=torch.float64, device=device)
