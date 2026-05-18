"""Generate image-derived grayscale targets from RGB images for holography runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, gaussian_filter, sobel


DEFAULT_EDGE_PERCENTILE = 75.0
DEFAULT_BLUR_SIGMA = 1.0
DEFAULT_GRAYSCALE_MAX_INTENSITY = 0.65
DEFAULT_GRAYSCALE_GAMMA = 1.6
DEFAULT_FLAT_REGION_DARKENING = 0.55


def _resampling_lanczos() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _validate_size(size: int) -> None:
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")


def _line_radius_for_size(size: int) -> int:
    return max(2, size // 64)


def load_rgb_image_as_square_grayscale(path: str | Path, size: int) -> np.ndarray:
    """Load an RGB image, preserve aspect ratio, and pad it into a square grayscale canvas."""
    canvas = load_rgb_image_as_square_canvas(path, size)
    grayscale = ImageOps.autocontrast(canvas.convert("L"))
    return np.asarray(grayscale, dtype=np.float32) / 255.0


def load_rgb_image_as_square_canvas(path: str | Path, size: int) -> Image.Image:
    """Load an RGB image, preserve aspect ratio, and pad it into a square RGB canvas."""
    _validate_size(size)
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    scale = min(size / image.width, size / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), resample=_resampling_lanczos())

    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def load_rgb_image_as_dimmed_square_grayscale(path: str | Path, size: int) -> np.ndarray:
    """Load an RGB image as grayscale without stretching broad color blocks to white."""
    canvas = load_rgb_image_as_square_canvas(path, size)
    grayscale = np.asarray(canvas.convert("L"), dtype=np.float32) / 255.0
    return build_dimmed_grayscale_image(grayscale)


def build_dimmed_grayscale_image(
    grayscale: np.ndarray,
    *,
    max_intensity: float = DEFAULT_GRAYSCALE_MAX_INTENSITY,
    gamma: float = DEFAULT_GRAYSCALE_GAMMA,
    flat_region_darkening: float = DEFAULT_FLAT_REGION_DARKENING,
) -> np.ndarray:
    """Compress grayscale brightness and darken broad low-detail regions."""
    image = np.asarray(grayscale, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("grayscale image must be 2D")
    if not np.isfinite(image).all():
        raise ValueError("grayscale image contains NaN or inf")
    if not (0.0 < float(max_intensity) <= 1.0):
        raise ValueError("max_intensity must be in the range (0, 1]")
    if not (float(gamma) > 0.0):
        raise ValueError("gamma must be positive")
    if not (0.0 < float(flat_region_darkening) <= 1.0):
        raise ValueError("flat_region_darkening must be in the range (0, 1]")

    clipped = np.clip(image, 0.0, 1.0)
    compressed = np.power(clipped, float(gamma)) * float(max_intensity)

    smoothed = gaussian_filter(clipped, sigma=1.0)
    gradient_x = sobel(smoothed, axis=1, mode="reflect")
    gradient_y = sobel(smoothed, axis=0, mode="reflect")
    gradient = np.hypot(gradient_x, gradient_y)
    detail = np.zeros_like(clipped, dtype=np.float32)
    max_gradient = float(gradient.max())
    if max_gradient > 0.0:
        detail = gaussian_filter(gradient / max_gradient, sigma=1.0).astype(np.float32)

    flat_scale = float(flat_region_darkening)
    region_scale = flat_scale + (1.0 - flat_scale) * np.clip(detail * 3.0, 0.0, 1.0)
    return np.clip(compressed * region_scale, 0.0, float(max_intensity)).astype(np.float32)


def _grid_size_for_channels(expected_channels: int) -> int:
    if type(expected_channels) is not int or expected_channels <= 0:
        raise ValueError("expected_channels must be a positive integer")
    grid_size = int(round(expected_channels ** 0.5))
    if grid_size * grid_size != expected_channels:
        raise ValueError("grayscale split target requires a square channel count")
    return grid_size


def split_grayscale_image_into_channel_tiles(grayscale: np.ndarray, expected_channels: int = 9) -> np.ndarray:
    """Split one grayscale image into row-major tiles and resize each tile per channel."""
    image = np.asarray(grayscale, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("grayscale image must be 2D")
    if not np.isfinite(image).all():
        raise ValueError("grayscale image contains NaN or inf")

    grid_size = _grid_size_for_channels(expected_channels)
    height, width = image.shape
    max_intensity = float(np.clip(image.max(), 0.0, 1.0))
    row_indices = np.array_split(np.arange(height), grid_size)
    col_indices = np.array_split(np.arange(width), grid_size)
    tiles: list[np.ndarray] = []

    for rows in row_indices:
        for cols in col_indices:
            tile = image[int(rows[0]): int(rows[-1]) + 1, int(cols[0]): int(cols[-1]) + 1]
            tile_image = Image.fromarray(np.uint8(np.clip(tile, 0.0, 1.0) * 255.0), mode="L")
            resized = tile_image.resize((width, height), resample=_resampling_lanczos())
            resized_tile = np.asarray(resized, dtype=np.float32) / 255.0
            tiles.append(np.clip(resized_tile, 0.0, max_intensity))

    return np.stack(tiles, axis=0).astype(np.float32)


def extract_edge_mask(
    grayscale: np.ndarray,
    *,
    edge_percentile: float = DEFAULT_EDGE_PERCENTILE,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
) -> np.ndarray:
    """Extract a binary edge mask from a grayscale image using Sobel gradients."""
    image = np.asarray(grayscale, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("grayscale image must be 2D")
    if not np.isfinite(image).all():
        raise ValueError("grayscale image contains NaN or inf")

    smoothed = gaussian_filter(image, sigma=float(blur_sigma))
    gradient_x = sobel(smoothed, axis=1, mode="reflect")
    gradient_y = sobel(smoothed, axis=0, mode="reflect")
    magnitude = np.hypot(gradient_x, gradient_y)

    positive = magnitude[magnitude > 0.0]
    if positive.size == 0:
        return np.zeros_like(image, dtype=bool)

    percentile_value = float(np.clip(edge_percentile, 0.0, 100.0))
    threshold = float(np.percentile(positive, percentile_value))
    mask = magnitude >= threshold
    if np.any(mask):
        mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return mask.astype(bool)


def build_center_weighted_line_image(edge_mask: np.ndarray, line_radius: int) -> np.ndarray:
    """Expand a binary line mask and brighten line centers more than line borders."""
    mask = np.asarray(edge_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("edge mask must be 2D")
    if type(line_radius) is not int or line_radius <= 0:
        raise ValueError("line_radius must be a positive integer")
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)

    expanded = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=line_radius)
    distances = distance_transform_edt(expanded)
    max_distance = float(distances.max())
    result = np.zeros(mask.shape, dtype=np.float32)

    if max_distance <= 1.0:
        result[expanded] = 1.0
        return result

    weighted = np.clip((distances - 1.0) / (max_distance - 1.0), 0.0, 1.0)
    result[expanded] = weighted[expanded].astype(np.float32)
    return result


def generate_line_art_targets(
    path: str | Path,
    *,
    expected_channels: int = 9,
    size: int = 128,
) -> np.ndarray:
    """Create one line-art grayscale image and repeat it across all target channels."""
    if type(expected_channels) is not int or expected_channels <= 0:
        raise ValueError("expected_channels must be a positive integer")

    grayscale = load_rgb_image_as_square_grayscale(path, size)
    edge_mask = extract_edge_mask(grayscale)
    line_image = build_center_weighted_line_image(edge_mask, line_radius=_line_radius_for_size(size))
    if not np.any(line_image > 0.0):
        raise ValueError("line-art target generation produced an empty line image")

    target_stack = np.repeat(line_image[np.newaxis, :, :], expected_channels, axis=0)
    return target_stack.astype(np.float32)


def generate_grayscale_image_targets(
    path: str | Path,
    *,
    expected_channels: int = 9,
    size: int = 128,
) -> np.ndarray:
    """Create dimmed grayscale image tiles and assign one tile to each channel."""
    _grid_size_for_channels(expected_channels)
    grayscale = load_rgb_image_as_dimmed_square_grayscale(path, size)
    if not np.any(grayscale > 0.0):
        raise ValueError("grayscale target generation produced an empty image")

    return split_grayscale_image_into_channel_tiles(grayscale, expected_channels=expected_channels)
