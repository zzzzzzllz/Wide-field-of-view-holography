"""Target-derived region masks for signal-window holography losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from holo_opt.config import RegionMaskConfig


@dataclass
class RegionMasks:
    edge: np.ndarray
    signal: np.ndarray
    flat: np.ndarray
    dark: np.ndarray
    relaxed: np.ndarray
    report_rows: list[dict[str, float]]


def _validate_targets(targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(targets, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("targets must have shape (channels, height, width)")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("targets must have positive channel, height, and width dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("targets contain NaN or inf")
    return arr


def _normalize_channel(target: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    max_value = float(np.max(target))
    if max_value <= epsilon:
        return np.zeros_like(target, dtype=np.float32)
    return (target / (max_value + epsilon)).astype(np.float32)


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(values.astype(np.float32))
    return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for y_offset in range(size):
        for x_offset in range(size):
            output |= padded[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]]
    return output


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(np.float32)))


def _quantile_or_default(values: np.ndarray, quantile: float, default: float) -> float:
    nonzero = values[values > 0.0]
    if not nonzero.size:
        return default
    return float(np.quantile(nonzero, quantile))


def generate_region_masks(targets: np.ndarray, config: RegionMaskConfig) -> RegionMasks:
    arr = _validate_targets(targets)
    edge_masks = []
    signal_masks = []
    flat_masks = []
    dark_masks = []
    relaxed_masks = []
    rows: list[dict[str, float]] = []

    for channel_index, target in enumerate(arr):
        normalized = _normalize_channel(target)
        gradient = _gradient_magnitude(normalized)
        dark = normalized <= float(config.dark_threshold)
        raw_signal = normalized > float(config.signal_threshold)

        edge_threshold = _quantile_or_default(gradient, config.edge_quantile, float("inf"))
        raw_edge = gradient >= edge_threshold if np.isfinite(edge_threshold) else np.zeros_like(dark, dtype=bool)
        edge = _dilate(raw_edge, int(config.edge_dilation)) & ~dark

        signal_without_edge = raw_signal & ~edge & ~dark
        flat_threshold = _quantile_or_default(gradient, config.flat_gradient_quantile, 0.0)
        flat_candidates = gradient <= flat_threshold
        flat = signal_without_edge & flat_candidates
        signal = signal_without_edge & ~flat
        relaxed = ~(edge | signal | flat | dark)

        edge_masks.append(edge.astype(np.float32))
        signal_masks.append(signal.astype(np.float32))
        flat_masks.append(flat.astype(np.float32))
        dark_masks.append(dark.astype(np.float32))
        relaxed_masks.append(relaxed.astype(np.float32))
        rows.append({
            "channel": float(channel_index + 1),
            "edge_fraction": _fraction(edge),
            "signal_fraction": _fraction(signal),
            "flat_fraction": _fraction(flat),
            "dark_fraction": _fraction(dark),
            "relaxed_fraction": _fraction(relaxed),
            "target_mean": float(np.mean(target)),
            "target_max": float(np.max(target)),
            "edge_threshold": edge_threshold if np.isfinite(edge_threshold) else 0.0,
        })

    return RegionMasks(
        edge=np.stack(edge_masks, axis=0).astype(np.float32),
        signal=np.stack(signal_masks, axis=0).astype(np.float32),
        flat=np.stack(flat_masks, axis=0).astype(np.float32),
        dark=np.stack(dark_masks, axis=0).astype(np.float32),
        relaxed=np.stack(relaxed_masks, axis=0).astype(np.float32),
        report_rows=rows,
    )
