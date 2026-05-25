"""FFT-based field simulation and differentiable loss terms for image optimization."""

from __future__ import annotations

import torch
import torch.fft as fft
import torch.nn.functional as F


def fftshift2(values: torch.Tensor) -> torch.Tensor:
    height, width = values.shape[-2], values.shape[-1]
    shifted = torch.roll(values, shifts=(height // 2,), dims=(-2,))
    return torch.roll(shifted, shifts=(width // 2,), dims=(-1,))


def compute_intensities(phdx: torch.Tensor, phdy: torch.Tensor, pair_mat: torch.Tensor) -> torch.Tensor:
    """Simulate one far-field intensity image per diffraction-channel pair."""
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
    return gray_monotonic_loss_masked(reconstruction, targets, None, levels=levels)


def gray_monotonic_loss_masked(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    spatial_weights: torch.Tensor | None,
    levels: int = 16,
) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be at least 2")
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")
    if spatial_weights is not None:
        spatial_weights = _as_channel_tensor(spatial_weights, "spatial_weights")
        if spatial_weights.shape != targets.shape:
            raise ValueError("spatial_weights and targets must have the same shape")

    penalties = []
    level_indices = torch.round(torch.clamp(targets, 0.0, 1.0) * float(levels - 1)).long()
    for channel in range(targets.shape[0]):
        means = []
        for level in range(levels):
            mask = level_indices[channel] == level
            if spatial_weights is not None:
                mask = mask & (spatial_weights[channel] > 0)
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
    return background_loss_masked(reconstruction, targets, None, epsilon=epsilon)


def background_loss_masked(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    spatial_weights: torch.Tensor | None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")
    if spatial_weights is not None:
        spatial_weights = _as_channel_tensor(spatial_weights, "spatial_weights")
        if spatial_weights.shape != targets.shape:
            raise ValueError("spatial_weights and targets must have the same shape")

    dark_mask = targets <= epsilon
    if spatial_weights is not None:
        dark_mask = dark_mask & (spatial_weights > 0)
    if not torch.any(dark_mask):
        return reconstruction.new_tensor(0.0)
    return reconstruction[dark_mask].mean()


def _target_edge_weight(targets: torch.Tensor) -> torch.Tensor:
    gradient_y = torch.zeros_like(targets)
    gradient_x = torch.zeros_like(targets)
    gradient_y[..., 1:, :] = torch.abs(targets[..., 1:, :] - targets[..., :-1, :])
    gradient_x[..., :, 1:] = torch.abs(targets[..., :, 1:] - targets[..., :, :-1])
    edge_strength = torch.maximum(gradient_x, gradient_y)
    return 1.0 / (1.0 + 4.0 * edge_strength)


def local_uniformity_loss(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    kernel_size: int = 3,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    object_weight = torch.where(targets > epsilon, torch.clamp(targets, 0.0, 1.0), torch.zeros_like(targets))
    if not torch.any(object_weight > 0):
        return reconstruction.new_tensor(0.0)

    padding = kernel_size // 2
    object_weight_4d = object_weight.unsqueeze(1)
    weighted_sum = F.avg_pool2d(
        (reconstruction * object_weight).unsqueeze(1),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    weight_sum = F.avg_pool2d(object_weight_4d, kernel_size=kernel_size, stride=1, padding=padding)
    local_mean = (weighted_sum / weight_sum.clamp_min(epsilon)).squeeze(1)
    edge_weight = _target_edge_weight(torch.clamp(targets, 0.0, 1.0))
    combined_weight = object_weight * edge_weight
    squared_deviation = (reconstruction - local_mean).square()
    return (combined_weight * squared_deviation).sum() / combined_weight.sum().clamp_min(epsilon)


def high_frequency_loss(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    reconstruction = _as_channel_tensor(reconstruction, "reconstruction")
    targets = _as_channel_tensor(targets, "targets")
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have the same shape")

    object_weight = torch.where(targets > epsilon, torch.clamp(targets, 0.0, 1.0), torch.zeros_like(targets))
    if not torch.any(object_weight > 0):
        return reconstruction.new_tensor(0.0)

    edge_weight = _target_edge_weight(torch.clamp(targets, 0.0, 1.0))
    combined_weight = object_weight * edge_weight
    kernel = reconstruction.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    padded = F.pad(reconstruction.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    response = F.conv2d(padded, kernel).squeeze(1)
    return (combined_weight * response.square()).sum() / combined_weight.sum().clamp_min(epsilon)


def compute_loss_terms(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
    epsilon: float = 1e-8,
    spatial_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare simulated far-field channels against target channels and return loss terms."""
    if loss_weights is None:
        loss_weights = {}

    intensities = compute_intensities(phdx, phdy, pair_mat)
    normalized_intensity = normalize_intensities(intensities, epsilon=epsilon)
    target_max = targets.amax(dim=(-2, -1), keepdim=True)
    normalized_target = targets / (target_max + epsilon)
    spatial_mask = None
    if spatial_weights is not None:
        spatial_mask = _as_channel_tensor(spatial_weights, "spatial_weights")
        if spatial_mask.shape != normalized_target.shape:
            raise ValueError("spatial_weights and targets must have the same shape")
    if spatial_mask is None:
        per_channel = ((normalized_intensity - normalized_target) ** 2).mean(dim=(-2, -1))
    else:
        weighted_error = spatial_mask * (normalized_intensity - normalized_target).square()
        per_channel = weighted_error.sum(dim=(-2, -1)) / spatial_mask.sum(dim=(-2, -1)).clamp_min(epsilon)
    image_mse = torch.sum(weights * per_channel)
    eta_balance = channel_energy_balance_loss(intensities, normalized_target, epsilon=epsilon)
    gray_monotonic = gray_monotonic_loss_masked(normalized_intensity, normalized_target, spatial_mask)
    smoothness = phase_smoothness_loss(phdx, phdy)
    background = background_loss_masked(normalized_intensity, normalized_target, spatial_mask, epsilon=epsilon)
    if spatial_mask is None:
        local_uniformity = local_uniformity_loss(normalized_intensity, normalized_target, epsilon=epsilon)
        high_frequency = high_frequency_loss(normalized_intensity, normalized_target, epsilon=epsilon)
    else:
        local_uniformity = local_uniformity_loss(normalized_intensity, normalized_target * spatial_mask, epsilon=epsilon)
        high_frequency = high_frequency_loss(normalized_intensity, normalized_target * spatial_mask, epsilon=epsilon)
    total = (
        float(loss_weights.get("image_weight", 1.0)) * image_mse
        + float(loss_weights.get("eta_balance_weight", 0.0)) * eta_balance
        + float(loss_weights.get("gray_monotonic_weight", 0.0)) * gray_monotonic
        + float(loss_weights.get("phase_smoothness_weight", 0.0)) * smoothness
        + float(loss_weights.get("background_weight", 0.0)) * background
        + float(loss_weights.get("local_uniformity_weight", 0.0)) * local_uniformity
        + float(loss_weights.get("high_frequency_weight", 0.0)) * high_frequency
    )
    return {
        "total": total,
        "image_mse": image_mse,
        "eta_balance": eta_balance,
        "gray_monotonic": gray_monotonic,
        "phase_smoothness": smoothness,
        "background": background,
        "local_uniformity": local_uniformity,
        "high_frequency": high_frequency,
    }


def training_loss(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float | dict[str, float] = 1e-8,
    loss_weights: dict[str, float] | None = None,
    spatial_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(epsilon, dict):
        loss_weights = epsilon
        epsilon = 1e-8
    return compute_loss_terms(
        phdx,
        phdy,
        pair_mat,
        targets,
        weights,
        loss_weights,
        epsilon=epsilon,
        spatial_weights=spatial_weights,
    )["total"]
