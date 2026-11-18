from dataclasses import dataclass

from .shapes import Shape


@dataclass(frozen=True, kw_only=True)
class Element:
    """Тонкий оптический элемент

    shape - форма элемента, Shape
    """

    shape: Shape | None = None

    def profile(self, x, y, lam):
        """Формирование передаточной функции элемента

        x [1, n], y [n, 1]  - координаты, м
        lam [Nλ, 1, 1]      - длины волн, м.
        """
        raise NotImplementedError

    def transmission(self, x, y, lam):
        t = self.profile(x, y, lam)
        if self.shape is not None:
            t = t * self.shape.mask(x, y)
        return t
