"""Adaptive channel-weight updates driven by efficiency and reconstruction error."""

from __future__ import annotations

import numpy as np


def _as_channel_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D per-channel array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_nonnegative_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return scalar


def _normalize_mean_with_clip(values: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("updated weights contain NaN or inf")
    if not np.any(arr > 0.0):
        raise ValueError("updated weights must include at least one positive value")

    target_sum = float(arr.size)

    def clipped_sum(scale: float) -> float:
        return float(np.sum(np.clip(arr * scale, clip_min, clip_max)))

    low = 0.0
    high = 1.0
    while clipped_sum(high) < target_sum:
        high *= 2.0
        if not np.isfinite(high):
            raise ValueError("updated weights could not be normalized")

    for _ in range(100):
        mid = (low + high) / 2.0
        if clipped_sum(mid) < target_sum:
            low = mid
        else:
            high = mid

    normalized = np.clip(arr * high, clip_min, clip_max)
    return normalized.astype(np.float32)


def update_weights(
    old_weights: np.ndarray,
    eta: np.ndarray,
    channel_error: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    clip_min: float = 0.5,
    clip_max: float = 5.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Update per-channel optimization weights from efficiency and error terms."""
    old = _as_channel_array("old_weights", old_weights)
    eta_arr = _as_channel_array("eta", eta)
    error_arr = _as_channel_array("channel_error", channel_error)
    if old.shape != eta_arr.shape or old.shape != error_arr.shape:
        raise ValueError("old_weights, eta, and channel_error must have the same shape")

    alpha_value = _as_nonnegative_scalar("alpha", alpha)
    beta_value = _as_nonnegative_scalar("beta", beta)
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be a positive finite scalar")

    min_value = float(clip_min)
    max_value = float(clip_max)
    if not np.isfinite(min_value) or not np.isfinite(max_value) or min_value <= 0.0 or min_value >= max_value:
        raise ValueError("clip_min and clip_max must be finite values with 0 < clip_min < clip_max")
    if min_value > 1.0 or max_value < 1.0:
        raise ValueError("clip_min and clip_max must bracket 1.0 to preserve mean normalization")

    if np.any(old < 0.0):
        raise ValueError("old_weights must be nonnegative")
    if np.any(eta_arr < 0.0):
        raise ValueError("eta must be nonnegative")
    if np.any(error_arr < 0.0):
        raise ValueError("channel_error must be nonnegative")

    mean_eta = float(np.mean(eta_arr))
    mean_error = float(np.mean(error_arr))
    eta_factor = ((mean_eta + eps) / (eta_arr.astype(np.float64) + eps)) ** alpha_value
    error_factor = ((error_arr.astype(np.float64) + eps) / (mean_error + eps)) ** beta_value

    updated = old.astype(np.float64) * eta_factor * error_factor
    updated = _normalize_mean_with_clip(updated, min_value, max_value)
    if not np.isfinite(updated).all():
        raise ValueError("updated weights contain NaN or inf")
    return updated.astype(np.float32)
