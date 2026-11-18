import dataclasses

import torch


class System:
    """Оптическая система.

    elements - список пар (z, элемент), где z - положение от плоскости источника, м
    """

    def __init__(self, elements):
        self.elements = sorted(elements, key=lambda e: e[0])
        zs = [z for z, _ in self.elements]
        if len(set(zs)) != len(zs):
            raise ValueError(f"несколько элементов в одной плоскости: z = {zs}")

    def run(self, source, z_out, sim):
        """Расчёт поля в плоскости z_out."""
        field = source.field(sim)
        x, y = sim.coords()
        lam = torch.tensor(field.wavelengths, dtype=torch.float64, device=field.data.device)[:, None, None]
        for z, element in self.elements:
            if z != field.z:
                field = sim.propagate(field, z)
            field = dataclasses.replace(field, data=field.data * element.transmission(x, y, lam))
        if z_out != field.z:
            field = sim.propagate(field, z_out)
        return field
