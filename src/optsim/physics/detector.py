from dataclasses import dataclass

from optsim.core import integrate_pixels


@dataclass(frozen=True, kw_only=True)
class Detector:
    """Детектор квадратная матрица пикселей с центром на оптической оси.

    z        - положение от плоскости источника, м
    pixel    - размер пикселя, м; должен быть кратен шагу дискретизации сетки
    n_pixels - число пикселей по одной оси
    """

    z: float
    pixel: float
    n_pixels: int

    def measure(self, field):
        """Сигнал [n_pixels, n_pixels]: интеграл интенсивности по пикселю, сумма интенсивностей по lambda."""
        return integrate_pixels(field.intensity(), field.grid, self.pixel, self.n_pixels).sum(dim=0)
