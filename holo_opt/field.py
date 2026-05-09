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


def _as_channel_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim == 2:
        return values.unsqueeze(0)
    if values.ndim != 3:
        raise ValueError(f"{name} must be 2D or 3D")
    return values


def _wrapped_phase_delta(delta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def phase_smoothness_loss(phdx: torch.Tensor, phdy: torch.Tensor) -> torch.Tensor:
    terms = []
    for phase in (phdx, phdy):
        if phase.shape[-2] > 1:
            terms.append(_wrapped_phase_delta(phase[..., 1:, :] - phase[..., :-1, :]).square().mean())
        if phase.shape[-1] > 1:
            terms.append(_wrapped_phase_delta(phase[..., :, 1:] - phase[..., :, :-1]).square().mean())
    if not terms:
        return phdx.new_tensor(0.0)
    return torch.stack(terms).sum()


def gray_monotonic_loss(reconstruction: torch.Tensor, targets: torch.Tensor, levels: int = 16) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be at least 2")
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")

    penalties = []
    level_indices = torch.round(torch.clamp(targets, 0.0, 1.0) * float(levels - 1)).long()
    for channel in range(targets.shape[0]):
        means = []
        for level in range(levels):
            mask = level_indices[channel] == level
            if torch.any(mask):
                means.append(reconstruction[channel][mask].mean())
        if len(means) >= 2:
            means_tensor = torch.stack(means)
            penalties.append(torch.relu(-(means_tensor[1:] - means_tensor[:-1])).mean())
    if not penalties:
        return reconstruction.new_tensor(0.0)
    return torch.stack(penalties).mean()


def channel_energy_balance_loss(
    intensities: torch.Tensor,
    targets: torch.Tensor | float | None = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    if isinstance(targets, (float, int)):
        epsilon = float(targets)
        targets = None

    intensities = _as_channel_tensor(intensities, "intensities")
    energy = intensities.sum(dim=(-2, -1))
    if targets is None:
        mean_energy = energy.mean()
        return energy.std(unbiased=False) / (mean_energy + epsilon)

    targets = _as_channel_tensor(targets, "targets")
    if intensities.shape != targets.shape:
        raise ValueError("intensities and targets must have the same shape")

    useful_mask = targets > epsilon
    useful_energy = torch.where(useful_mask, intensities, torch.zeros_like(intensities)).sum(dim=(-2, -1))
    efficiencies = useful_energy / (energy + epsilon)
    mean_efficiency = efficiencies.mean()
    return efficiencies.std(unbiased=False) / (mean_efficiency + epsilon)


def background_loss(reconstruction: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")

    dark_mask = targets <= epsilon
    if not torch.any(dark_mask):
        return reconstruction.new_tensor(0.0)
    return reconstruction[dark_mask].mean()


def compute_loss_terms(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if loss_weights is None:
        loss_weights = {}

    intensities = compute_intensities(phdx, phdy, pair_mat)
    normalized_intensity = normalize_intensities(intensities, epsilon=epsilon)
    target_max = targets.amax(dim=(-2, -1), keepdim=True)
    normalized_target = targets / (target_max + epsilon)
    per_channel = ((normalized_intensity - normalized_target) ** 2).mean(dim=(-2, -1))
    image_mse = torch.sum(weights * per_channel)
    eta_balance = channel_energy_balance_loss(intensities, normalized_target, epsilon=epsilon)
    gray_monotonic = gray_monotonic_loss(normalized_intensity, normalized_target)
    smoothness = phase_smoothness_loss(phdx, phdy)
    background = background_loss(normalized_intensity, normalized_target, epsilon=epsilon)
    total = (
        float(loss_weights.get("image_weight", 1.0)) * image_mse
        + float(loss_weights.get("eta_balance_weight", 0.0)) * eta_balance
        + float(loss_weights.get("gray_monotonic_weight", 0.0)) * gray_monotonic
        + float(loss_weights.get("phase_smoothness_weight", 0.0)) * smoothness
        + float(loss_weights.get("background_weight", 0.0)) * background
    )
    return {
        "total": total,
        "image_mse": image_mse,
        "eta_balance": eta_balance,
        "gray_monotonic": gray_monotonic,
        "phase_smoothness": smoothness,
        "background": background,
    }


def training_loss(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float | dict[str, float] = 1e-8,
    loss_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    if isinstance(epsilon, dict):
        loss_weights = epsilon
        epsilon = 1e-8
    return compute_loss_terms(phdx, phdy, pair_mat, targets, weights, loss_weights, epsilon=epsilon)["total"]
