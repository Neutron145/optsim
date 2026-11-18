"""Распространение поля в свободном пространстве.

Свёртка считается через FFT, поэтому она циклическая: свет, ушедший
за край окна, появляется с другой стороны. Окно должно быть с запасом.

Постоянный множитель exp(ikz) опущен: на интенсивность он не влияет.
"""

import math

import torch


def expi(phase):
    """exp(i·phase) в complex64.

    Фаза бывает до 10⁴ рад: во float32 её дробная часть теряется.
    Поэтому сначала приводим её к [0, 2π) во float64.
    """
    phase = torch.remainder(phase, 2 * math.pi).float()
    return torch.polar(torch.ones_like(phase), phase)


def fresnel(data, grid, wavelengths, dz):
    """Френель: H = exp(-iπλz(fx² + fy²)). Goodman, «Introduction to Fourier Optics», гл. 4.

    H раскладывается в произведение множителей по x и по y.
    """
    f = grid.freqs(data.device)
    lam = torch.tensor(wavelengths, dtype=torch.float64, device=data.device)[:, None, None]
    a = -math.pi * lam * dz
    hx = expi(a * f**2)  # [Nλ, 1, n]
    hy = expi(a * f[:, None] ** 2)  # [Nλ, n, 1]
    return torch.fft.ifft2(torch.fft.fft2(data) * hx * hy)


def angular_spectrum(data, grid, wavelengths, dz):
    """Угловой спектр: H = exp(i2πz(√(1/λ² − f²) − 1/λ)). Goodman, гл. 3.

    Затухающие волны (1/λ² < f²) отбрасываются.
    """
    f = grid.freqs(data.device)
    lam = torch.tensor(wavelengths, dtype=torch.float64, device=data.device)[:, None, None]
    arg = 1 / lam**2 - (f**2 + f[:, None] ** 2)  # [Nλ, n, n]
    h = expi(2 * math.pi * dz * (torch.sqrt(arg.clamp(min=0)) - 1 / lam)) * (arg > 0)
    return torch.fft.ifft2(torch.fft.fft2(data) * h)
