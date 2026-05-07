from __future__ import annotations

import torch
import torch.fft as fft


def fftshift2(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[-2], values.shape[-1]
    shifted = torch.roll(values, shifts=(height // 2,), dims=(-2,))
    return torch.roll(shifted, shifts=(width // 2,), dims=(-1,))


def compute_intensities(phdx: torch.Tensor, phdy: torch.Tensor, pair_mat: torch.Tensor) -> torch.Tensor:
    if phdx.shape != phdy.shape:
        raise ValueError("phdx and phdy must have the same shape")
    channels = []
    for channel in range(pair_mat.shape[0]):
        m = pair_mat[channel, 0]
        n = pair_mat[channel, 1]
        phase = m * phdx + n * phdy
        field = torch.exp(1j * phase)
        spectrum = fftshift2(fft.fft2(field))
        channels.append(torch.abs(spectrum) ** 2)
    return torch.stack(channels, dim=0)


def normalize_intensities(intensities: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    max_values = intensities.amax(dim=(-2, -1), keepdim=True)
    nonzero_channels = max_values > 0
    safe_max_values = torch.where(nonzero_channels, max_values, torch.ones_like(max_values))
    return torch.where(nonzero_channels, intensities / safe_max_values, torch.zeros_like(intensities))


def training_loss(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    intensities = compute_intensities(phdx, phdy, pair_mat)
    normalized_intensity = normalize_intensities(intensities, epsilon=epsilon)
    target_max = targets.amax(dim=(-2, -1), keepdim=True)
    normalized_target = targets / (target_max + epsilon)
    per_channel = ((normalized_intensity - normalized_target) ** 2).mean(dim=(-2, -1))
    return torch.sum(weights * per_channel)
