"""Evaluation metrics for reconstruction quality, grayscale behavior, and efficiency."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _validate_metric_inputs(intensities: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr_i = np.asarray(intensities, dtype=np.float32)
    arr_t = np.asarray(targets, dtype=np.float32)
    if arr_i.ndim != 3 or arr_t.ndim != 3:
        raise ValueError("intensities and targets must be 3D arrays with the same shape (channels, height, width)")
    if arr_i.shape != arr_t.shape:
        raise ValueError("intensities and targets must have the same shape (channels, height, width)")
    return arr_i, arr_t


def normalize_per_channel(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    max_values = np.max(arr, axis=(-2, -1), keepdims=True)
    return arr / (max_values + epsilon)


def _target_edge_weight(target: np.ndarray) -> np.ndarray:
    gradient_y = np.zeros_like(target, dtype=np.float32)
    gradient_x = np.zeros_like(target, dtype=np.float32)
    gradient_y[1:, :] = np.abs(target[1:, :] - target[:-1, :])
    gradient_x[:, 1:] = np.abs(target[:, 1:] - target[:, :-1])
    edge_strength = np.maximum(gradient_x, gradient_y)
    return (1.0 / (1.0 + 4.0 * edge_strength)).astype(np.float32)


def _object_flat_weight(target: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    clipped = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    object_weight = np.where(clipped > epsilon, clipped, 0.0).astype(np.float32)
    return object_weight * _target_edge_weight(clipped)


def _local_mean_2d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    pad = kernel_size // 2
    padded = np.pad(np.asarray(values, dtype=np.float32), pad_width=pad, mode="reflect")
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows.mean(axis=(-2, -1), dtype=np.float32).astype(np.float32)


def _laplacian_response_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    padded = np.pad(arr, pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return (4.0 * center - up - down - left - right).astype(np.float32)


def channel_mse(intensities: np.ndarray, targets: np.ndarray, spatial_weights: np.ndarray | None = None) -> np.ndarray:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    norm_i = normalize_per_channel(arr_i)
    norm_t = normalize_per_channel(arr_t)
    if spatial_weights is None:
        return np.mean((norm_i - norm_t) ** 2, axis=(-2, -1))
    mask = np.asarray(spatial_weights, dtype=np.float32)
    if mask.shape != arr_t.shape:
        raise ValueError("spatial_weights and targets must have the same shape")
    weighted_error = mask * np.square(norm_i - norm_t)
    return np.sum(weighted_error, axis=(-2, -1)) / np.maximum(np.sum(mask, axis=(-2, -1)), 1e-8)


def compute_eta(intensities: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    eta_values = []
    for channel in range(arr_i.shape[0]):
        mask = arr_t[channel] > 0
        numerator = float(arr_i[channel][mask].sum()) if np.any(mask) else 0.0
        denominator = float(arr_i[channel].sum()) + epsilon
        eta_values.append(numerator / denominator)
    return np.asarray(eta_values, dtype=np.float32)


def gray_level_stats(
    reconstruction: np.ndarray,
    target: np.ndarray,
    levels: int = 16,
    spatial_weights: np.ndarray | None = None,
) -> dict[str, object]:
    recon = np.asarray(reconstruction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if recon.shape != tgt.shape:
        raise ValueError("reconstruction and target must have the same shape")
    valid_mask = None
    if spatial_weights is not None:
        valid_mask = np.asarray(spatial_weights, dtype=np.float32)
        if valid_mask.shape != tgt.shape:
            raise ValueError("spatial_weights and target must have the same shape")
        valid_mask = valid_mask > 0
    indices = np.rint(np.clip(tgt, 0.0, 1.0) * (levels - 1)).astype(np.int32)
    means: list[float] = []
    variances: list[float] = []
    for level in range(levels):
        mask = indices == level
        if valid_mask is not None:
            mask &= valid_mask
        if np.any(mask):
            pixels = recon[mask]
            means.append(float(np.mean(pixels)))
            variances.append(float(np.var(pixels)))
        else:
            means.append(float("nan"))
            variances.append(float("nan"))
    valid_means = np.asarray([value for value in means if np.isfinite(value)], dtype=np.float32)
    if valid_means.size >= 2:
        spacing = np.diff(valid_means)
        inversions = int(np.sum(spacing < 0))
        ideal_gap = 1.0 / float(levels - 1)
        spacing_penalty = float(np.sum(np.maximum(0.0, ideal_gap - spacing)))
        inversion_penalty = float(np.sum(np.maximum(0.0, -spacing)))
        dynamic_range = float(np.max(valid_means) - np.min(valid_means))
        range_penalty = float(max(0.0, 1.0 - dynamic_range))
    else:
        inversions = 0
        spacing_penalty = 1.0
        inversion_penalty = 1.0
        range_penalty = 1.0
    valid_vars = np.asarray([value for value in variances if np.isfinite(value)], dtype=np.float32)
    uniformity = float(np.mean(valid_vars)) if valid_vars.size else 1.0
    gray_error = spacing_penalty + inversion_penalty + range_penalty + uniformity
    return {
        "means": means,
        "variances": variances,
        "inversions": inversions,
        "spacing_penalty": spacing_penalty,
        "inversion_penalty": inversion_penalty,
        "range_penalty": range_penalty,
        "uniformity": uniformity,
        "gray_level_error": float(gray_error),
    }


def object_noise_stats(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    kernel_size: int = 5,
    epsilon: float = 1e-8,
    spatial_weights: np.ndarray | None = None,
) -> dict[str, float]:
    recon = np.asarray(reconstruction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if recon.shape != tgt.shape:
        raise ValueError("reconstruction and target must have the same shape")
    weights = _object_flat_weight(tgt, epsilon=epsilon)
    if spatial_weights is not None:
        mask = np.asarray(spatial_weights, dtype=np.float32)
        if mask.shape != tgt.shape:
            raise ValueError("spatial_weights and target must have the same shape")
        weights = weights * np.clip(mask, 0.0, 1.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= epsilon:
        return {"object_local_variance": 0.0, "object_high_frequency_energy": 0.0}

    local_mean = _local_mean_2d(recon, kernel_size=kernel_size)
    local_variance = float(np.sum(weights * np.square(recon - local_mean)) / (weight_sum + epsilon))
    laplacian = _laplacian_response_2d(recon)
    high_frequency_energy = float(np.sum(weights * np.square(laplacian)) / (weight_sum + epsilon))
    return {
        "object_local_variance": local_variance,
        "object_high_frequency_energy": high_frequency_energy,
    }


def evaluate_metrics(intensities: np.ndarray, targets: np.ndarray, spatial_weights: np.ndarray | None = None) -> dict[str, object]:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    norm_i = normalize_per_channel(arr_i)
    if spatial_weights is not None:
        mask = np.asarray(spatial_weights, dtype=np.float32)
        if mask.shape != arr_t.shape:
            raise ValueError("spatial_weights and targets must have the same shape")
    else:
        mask = None
    mse = channel_mse(arr_i, arr_t, spatial_weights=mask)
    eta = compute_eta(arr_i, arr_t)
    rows = []
    gray_errors = []
    object_local_variances = []
    object_high_frequency_energies = []
    for channel in range(arr_t.shape[0]):
        channel_mask = None if mask is None else mask[channel]
        gray = gray_level_stats(norm_i[channel], arr_t[channel], levels=16, spatial_weights=channel_mask)
        noise = object_noise_stats(norm_i[channel], arr_t[channel], spatial_weights=channel_mask)
        gray_errors.append(float(gray["gray_level_error"]))
        object_local_variances.append(float(noise["object_local_variance"]))
        object_high_frequency_energies.append(float(noise["object_high_frequency_energy"]))
        rows.append({
            "channel": channel + 1,
            "mse": float(mse[channel]),
            "eta": float(eta[channel]),
            "gray_level_error": float(gray["gray_level_error"]),
            "gray_means": gray["means"],
            "object_local_variance": float(noise["object_local_variance"]),
            "object_high_frequency_energy": float(noise["object_high_frequency_energy"]),
        })
    image_error = float(np.mean(mse))
    gray_level_error = float(np.mean(gray_errors))
    mean_eta = float(np.mean(eta))
    eta_balance = float(np.std(eta) / (mean_eta + 1e-8))
    object_local_variance = float(np.mean(object_local_variances))
    object_high_frequency_energy = float(np.mean(object_high_frequency_energies))
    score = image_error + gray_level_error + eta_balance - mean_eta
    return {
        "rows": rows,
        "summary": {
            "image_error": image_error,
            "gray_level_error": gray_level_error,
            "efficiency_balance_penalty": eta_balance,
            "mean_eta": mean_eta,
            "object_local_variance": object_local_variance,
            "object_high_frequency_energy": object_high_frequency_energy,
            "score": float(score),
        },
    }
