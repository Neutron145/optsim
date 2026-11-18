from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Source:
    """Источник света в плоскости z = 0.

    wavelengths - длины волн, м, кортеж
    """

    wavelengths: tuple[float, ...]

    def amplitude(self, x, y):
        """Комплексная амплитуда: [n, n] или число."""
        raise NotImplementedError

    def field(self, sim):
        x, y = sim.coords()
        return sim.field(self.amplitude(x, y), self.wavelengths, z=0.0)
