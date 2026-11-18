from dataclasses import dataclass

from .base import Element
from .shapes import Shape


@dataclass(frozen=True, kw_only=True)
class Aperture(Element):

    shape: Shape

    def profile(self, x, y, lam):
        return 1.0
