"""Evaluation metrics for reconstruction quality, grayscale behavior, and efficiency."""

from __future__ import annotations

import numpy as np


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


def channel_mse(intensities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    norm_i = normalize_per_channel(arr_i)
    norm_t = normalize_per_channel(arr_t)
    return np.mean((norm_i - norm_t) ** 2, axis=(-2, -1))


def compute_eta(intensities: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    eta_values = []
    for channel in range(arr_i.shape[0]):
        mask = arr_t[channel] > 0
        numerator = float(arr_i[channel][mask].sum()) if np.any(mask) else 0.0
        denominator = float(arr_i[channel].sum()) + epsilon
        eta_values.append(numerator / denominator)
    return np.asarray(eta_values, dtype=np.float32)


def gray_level_stats(reconstruction: np.ndarray, target: np.ndarray, levels: int = 16) -> dict[str, object]:
    recon = np.asarray(reconstruction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if recon.shape != tgt.shape:
        raise ValueError("reconstruction and target must have the same shape")
    indices = np.rint(np.clip(tgt, 0.0, 1.0) * (levels - 1)).astype(np.int32)
    means: list[float] = []
    variances: list[float] = []
    for level in range(levels):
        mask = indices == level
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


def evaluate_metrics(intensities: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    arr_i, arr_t = _validate_metric_inputs(intensities, targets)
    norm_i = normalize_per_channel(arr_i)
    mse = channel_mse(arr_i, arr_t)
    eta = compute_eta(arr_i, arr_t)
    rows = []
    gray_errors = []
    for channel in range(arr_t.shape[0]):
        gray = gray_level_stats(norm_i[channel], arr_t[channel], levels=16)
        gray_errors.append(float(gray["gray_level_error"]))
        rows.append({
            "channel": channel + 1,
            "mse": float(mse[channel]),
            "eta": float(eta[channel]),
            "gray_level_error": float(gray["gray_level_error"]),
            "gray_means": gray["means"],
        })
    image_error = float(np.mean(mse))
    gray_level_error = float(np.mean(gray_errors))
    mean_eta = float(np.mean(eta))
    eta_balance = float(np.std(eta) / (mean_eta + 1e-8))
    score = image_error + gray_level_error + eta_balance - mean_eta
    return {
        "rows": rows,
        "summary": {
            "image_error": image_error,
            "gray_level_error": gray_level_error,
            "efficiency_balance_penalty": eta_balance,
            "mean_eta": mean_eta,
            "score": float(score),
        },
    }
