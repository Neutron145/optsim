import warnings

import torch.nn.functional as F


def integrate_pixels(intensity, grid, pixel, m):
    """Интеграл интенсивности по пикселям матрицы m×m с центром на оптической оси."""
    k = round(pixel / grid.dx)
    if k < 1 or abs(k * grid.dx - pixel) > 1e-6 * pixel:
        raise ValueError(f"размер пикселя {pixel:.3e} м не кратен шагу сетки {grid.dx:.3e} м")

    s = m * k
    start = grid.n // 2 - s // 2
    lo, hi = max(0, -start), max(0, start + s - grid.n)
    if lo or hi:
        warnings.warn(f"матрица {s * grid.dx:.3e} м больше окна {grid.size:.3e} м, вне окна поле считается нулевым")
        intensity = F.pad(intensity, (lo, hi, lo, hi))
        start += lo

    crop = intensity[..., start : start + s, start : start + s]
    return crop.unflatten(-1, (m, k)).unflatten(-3, (m, k)).sum(dim=(-3, -1)) * grid.dx**2
