import math

import torch


def expi(phase):
    """exp(i*phase) в complex64."""
    phase = torch.remainder(phase, 2 * math.pi).float()
    return torch.polar(torch.ones_like(phase), phase)


def fresnel(data, grid, wavelengths, dz):
    """Преобразование Френеля"""
    f = grid.freqs(data.device)
    lam = torch.tensor(wavelengths, dtype=torch.float64, device=data.device)[:, None, None]
    a = -math.pi * lam * dz
    hx = expi(a * f**2)
    hy = expi(a * f[:, None] ** 2)
    return torch.fft.ifft2(torch.fft.fft2(data) * hx * hy)


def angular_spectrum(data, grid, wavelengths, dz):
    """Угловой спектр"""
    f = grid.freqs(data.device)
    lam = torch.tensor(wavelengths, dtype=torch.float64, device=data.device)[:, None, None]
    arg = 1 / lam**2 - (f**2 + f[:, None] ** 2)  # [Nλ, n, n]
    h = expi(2 * math.pi * dz * (torch.sqrt(arg.clamp(min=0)) - 1 / lam)) * (arg > 0)
    return torch.fft.ifft2(torch.fft.fft2(data) * h)
