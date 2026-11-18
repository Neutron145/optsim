from dataclasses import dataclass


class Shape:
    """Форма элемента."""

    def mask(self, x, y):
        """Булева маска [n, n]"""
        raise NotImplementedError


@dataclass(frozen=True)
class Circle(Shape):
    """r - радиус, м"""

    r: float

    def mask(self, x, y):
        return x**2 + y**2 <= self.r**2


@dataclass(frozen=True)
class Ring(Shape):
    """r_in, r_out - внутренний и внешний радиусы, м"""

    r_in: float
    r_out: float

    def mask(self, x, y):
        r2 = x**2 + y**2
        return (r2 >= self.r_in**2) & (r2 <= self.r_out**2)


@dataclass(frozen=True)
class Rect(Shape):
    """width, height - размеры по x и по y, м"""

    width: float
    height: float

    def mask(self, x, y):
        return (x.abs() <= self.width / 2) & (y.abs() <= self.height / 2)
