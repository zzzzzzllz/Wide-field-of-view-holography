"""Target generation and MAT-file loading for multi-channel far-field objectives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def normalize_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        raise ValueError("target array contains NaN or inf")
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val <= min_val:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


def generate_gray_step_targets(n_channels: int = 9, size: int = 128, levels: int = 16) -> np.ndarray:
    """Generate the built-in multi-channel grayscale target stack used for diagnostics."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    if levels != 16:
        raise ValueError("standard gray-step target currently expects 16 levels")
    tile = int(np.ceil(size / 4))
    level_grid = np.arange(levels, dtype=np.float32).reshape(4, 4) / float(levels - 1)
    base = np.kron(level_grid, np.ones((tile, tile), dtype=np.float32))[:size, :size]
    targets = []
    for channel in range(n_channels):
        shift_y = (channel % 3) * max(1, size // 16)
        shift_x = (channel // 3) * max(1, size // 16)
        targets.append(np.roll(np.roll(base, shift_y, axis=0), shift_x, axis=1))
    return np.stack(targets, axis=0).astype(np.float32)


def validate_targets(targets: np.ndarray, expected_channels: int) -> np.ndarray:
    """Validate that targets match the expected (channels, height, width) layout."""
    arr = np.asarray(targets, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("targets must have shape (channels, height, width)")
    if arr.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {arr.shape[0]}")
    if arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("target height and width must be positive")
    if not np.isfinite(arr).all():
        raise ValueError("targets contain NaN or inf")
    return arr.astype(np.float32)


def load_mat_targets(path: str | Path, variable: str = "bw_all", expected_channels: int = 9) -> np.ndarray:
    """Load a user-provided multi-channel target stack from a MAT file."""
    mat_path = Path(path)
    if not mat_path.exists():
        raise FileNotFoundError(str(mat_path))
    data = loadmat(mat_path)
    if variable not in data:
        public_keys = sorted(key for key in data.keys() if not key.startswith("__"))
        raise KeyError(f"Variable {variable!r} not found. Available variables: {public_keys}")
    arr = np.asarray(data[variable])
    if arr.ndim != 3:
        raise ValueError("MAT target variable must be a 3D array")
    first_axis_matches = arr.shape[0] == expected_channels
    last_axis_matches = arr.shape[-1] == expected_channels
    if first_axis_matches and last_axis_matches:
        raise ValueError(f"ambiguous MAT target channel axis for {expected_channels} channels")
    if first_axis_matches:
        channel_first = arr
    elif last_axis_matches:
        channel_first = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected {expected_channels} channels in first or last axis")
    return validate_targets(normalize_array(channel_first), expected_channels)
