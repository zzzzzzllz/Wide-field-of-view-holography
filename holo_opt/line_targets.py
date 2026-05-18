"""Generate image-derived grayscale targets from RGB images for holography runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, gaussian_filter, sobel

from holo_opt.config import GrayscalePreprocessConfig


DEFAULT_EDGE_PERCENTILE = 75.0
DEFAULT_BLUR_SIGMA = 1.0
DEFAULT_GRAYSCALE_MAX_INTENSITY = 0.65
DEFAULT_GRAYSCALE_GAMMA = 1.6
DEFAULT_FLAT_REGION_DARKENING = 0.55
DEFAULT_GRAYSCALE_DETAIL_BOOST = 0.2
DEFAULT_GRAYSCALE_TILE_BALANCE_STRENGTH = 0.35
DEFAULT_GRAYSCALE_TILE_BALANCE_CLIP = 1.35
DEFAULT_GRAYSCALE_PERCENTILE_LOW = 2.0
DEFAULT_GRAYSCALE_PERCENTILE_HIGH = 98.0


@dataclass(frozen=True)
class GrayscaleTargetArtifacts:
    source_grayscale: np.ndarray
    processed_grayscale: np.ndarray
    targets: np.ndarray
    stitched_target: np.ndarray
    report_rows: list[dict[str, float | int | str]]


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
    detail_boost: float = DEFAULT_GRAYSCALE_DETAIL_BOOST,
) -> np.ndarray:
    """Compress grayscale brightness while preserving local detail."""
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
    if not (float(detail_boost) >= 0.0):
        raise ValueError("detail_boost must be nonnegative")

    clipped = np.clip(image, 0.0, 1.0)
    normalized = _normalize_foreground_luminance(clipped)
    tone_input = 0.7 * clipped + 0.3 * normalized
    compressed = np.power(np.clip(tone_input, 0.0, 1.0), float(gamma)) * float(max_intensity)

    smoothed = gaussian_filter(normalized, sigma=1.0)
    gradient_x = sobel(smoothed, axis=1, mode="reflect")
    gradient_y = sobel(smoothed, axis=0, mode="reflect")
    gradient = np.hypot(gradient_x, gradient_y)
    detail = np.zeros_like(clipped, dtype=np.float32)
    max_gradient = float(gradient.max())
    if max_gradient > 0.0:
        detail = gaussian_filter(gradient / max_gradient, sigma=1.0).astype(np.float32)

    flat_scale = float(flat_region_darkening)
    region_scale = flat_scale + (1.0 - flat_scale) * np.clip(detail * 3.0, 0.0, 1.0)
    dog_detail = gaussian_filter(normalized, sigma=0.8) - gaussian_filter(normalized, sigma=2.2)
    detail_scale = _percentile_scale(np.abs(dog_detail), percentile=95.0)
    boosted = compressed * region_scale
    if detail_scale > 0.0:
        boosted = boosted + float(detail_boost) * 0.12 * float(max_intensity) * (dog_detail / detail_scale)
    return np.clip(boosted, 0.0, float(max_intensity)).astype(np.float32)


def _normalize_foreground_luminance(
    grayscale: np.ndarray,
    *,
    low_percentile: float = DEFAULT_GRAYSCALE_PERCENTILE_LOW,
    high_percentile: float = DEFAULT_GRAYSCALE_PERCENTILE_HIGH,
) -> np.ndarray:
    image = np.asarray(grayscale, dtype=np.float32)
    foreground = image[image > 1e-3]
    if foreground.size < 16:
        return image.astype(np.float32)

    low = float(np.percentile(foreground, low_percentile))
    high = float(np.percentile(foreground, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-6:
        return image.astype(np.float32)
    normalized = (image - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _percentile_scale(values: np.ndarray, *, percentile: float) -> float:
    flattened = np.asarray(values, dtype=np.float32)
    positive = flattened[np.isfinite(flattened) & (flattened > 0.0)]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive, percentile))


def _grid_size_for_channels(expected_channels: int) -> int:
    if type(expected_channels) is not int or expected_channels <= 0:
        raise ValueError("expected_channels must be a positive integer")
    grid_size = int(round(expected_channels ** 0.5))
    if grid_size * grid_size != expected_channels:
        raise ValueError("grayscale split target requires a square channel count")
    return grid_size


def split_grayscale_image_into_channel_tiles(grayscale: np.ndarray, expected_channels: int = 9) -> np.ndarray:
    """Split one grayscale image into row-major tiles and resize each tile per channel."""
    return split_grayscale_image_into_channel_tiles_with_report(
        grayscale,
        expected_channels=expected_channels,
        tile_balance_strength=0.0,
    )[0]


def split_grayscale_image_into_channel_tiles_with_report(
    grayscale: np.ndarray,
    expected_channels: int = 9,
    *,
    tile_balance_strength: float = DEFAULT_GRAYSCALE_TILE_BALANCE_STRENGTH,
    tile_balance_clip: float = DEFAULT_GRAYSCALE_TILE_BALANCE_CLIP,
) -> tuple[np.ndarray, list[dict[str, float | int | str]]]:
    """Split one grayscale image into row-major tiles, gently equalize tile energy, and report it."""
    image = np.asarray(grayscale, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("grayscale image must be 2D")
    if not np.isfinite(image).all():
        raise ValueError("grayscale image contains NaN or inf")
    if not (float(tile_balance_strength) >= 0.0):
        raise ValueError("tile_balance_strength must be nonnegative")
    if not (float(tile_balance_clip) >= 1.0):
        raise ValueError("tile_balance_clip must be at least 1")

    grid_size = _grid_size_for_channels(expected_channels)
    height, width = image.shape
    max_intensity = float(np.clip(image.max(), 0.0, 1.0))
    row_indices = np.array_split(np.arange(height), grid_size)
    col_indices = np.array_split(np.arange(width), grid_size)
    tile_slices: list[tuple[int, int, np.ndarray]] = []
    tile_means: list[float] = []

    for rows in row_indices:
        for cols in col_indices:
            tile = image[int(rows[0]): int(rows[-1]) + 1, int(cols[0]): int(cols[-1]) + 1]
            tile_slices.append((int(rows[0]), int(cols[0]), tile))
            tile_means.append(float(tile.mean()))

    nonempty_means = [value for value in tile_means if value > 1e-4]
    reference_mean = float(np.median(nonempty_means)) if nonempty_means else 0.0
    tiles: list[np.ndarray] = []
    report_rows: list[dict[str, float | int | str]] = []

    for tile_index, (row_start, col_start, tile) in enumerate(tile_slices, start=1):
        tile_mean = float(tile.mean())
        if tile_mean > 1e-4 and reference_mean > 0.0:
            ratio = reference_mean / tile_mean
            scale = float(np.clip(ratio ** float(tile_balance_strength), 1.0 / tile_balance_clip, tile_balance_clip))
        else:
            scale = 1.0
        tile_image = Image.fromarray(np.uint8(np.clip(tile, 0.0, 1.0) * 255.0), mode="L")
        resized = tile_image.resize((width, height), resample=_resampling_lanczos())
        resized_tile = (np.asarray(resized, dtype=np.float32) / 255.0) * scale
        clipped_tile = np.clip(resized_tile, 0.0, max_intensity).astype(np.float32)
        tiles.append(clipped_tile)
        report_rows.append(
            {
                "stage": "tile",
                "tile": tile_index,
                "grid_row": 1 + ((tile_index - 1) // grid_size),
                "grid_col": 1 + ((tile_index - 1) % grid_size),
                "source_row": row_start,
                "source_col": col_start,
                "mean_intensity": float(clipped_tile.mean()),
                "peak_intensity": float(clipped_tile.max()),
                "nonzero_ratio": float(np.mean(clipped_tile > 1e-3)),
                "energy_share": float(clipped_tile.sum() / max(np.sum(image), 1e-6)),
                "budget_scale": scale,
            }
        )

    return np.stack(tiles, axis=0).astype(np.float32), report_rows


def summarize_grayscale_image(image: np.ndarray, *, stage: str) -> dict[str, float | int | str]:
    array = np.asarray(image, dtype=np.float32)
    smoothed = gaussian_filter(array, sigma=1.0)
    gradient_x = sobel(smoothed, axis=1, mode="reflect")
    gradient_y = sobel(smoothed, axis=0, mode="reflect")
    gradient = np.hypot(gradient_x, gradient_y)
    positive = gradient[gradient > 0.0]
    edge_threshold = float(np.percentile(positive, 75.0)) if positive.size else 0.0
    flat_threshold = float(np.percentile(positive, 35.0)) if positive.size else 0.0
    return {
        "stage": stage,
        "tile": 0,
        "grid_row": 0,
        "grid_col": 0,
        "source_row": 0,
        "source_col": 0,
        "mean_intensity": float(array.mean()),
        "peak_intensity": float(array.max()),
        "nonzero_ratio": float(np.mean(array > 1e-3)),
        "energy_share": float(array.sum()),
        "budget_scale": 1.0,
        "edge_density": float(np.mean(gradient >= edge_threshold)) if edge_threshold > 0.0 else 0.0,
        "flat_region_ratio": float(np.mean(gradient <= flat_threshold)) if flat_threshold > 0.0 else 1.0,
    }


def _stitch_channel_tiles(tiles: np.ndarray) -> np.ndarray:
    array = np.asarray(tiles, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("tiles must have shape (channels, height, width)")
    channels, height, width = array.shape
    grid_size = _grid_size_for_channels(channels)
    row_indices = np.array_split(np.arange(height), grid_size)
    col_indices = np.array_split(np.arange(width), grid_size)
    rows: list[np.ndarray] = []
    for row_index in range(grid_size):
        row_tiles = []
        for col_index in range(grid_size):
            tile = array[row_index * grid_size + col_index]
            tile_height = len(row_indices[row_index])
            tile_width = len(col_indices[col_index])
            row_tiles.append(tile[:tile_height, :tile_width])
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32)


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
    return generate_grayscale_target_artifacts(
        path,
        expected_channels=expected_channels,
        size=size,
    ).targets


def generate_grayscale_target_artifacts(
    path: str | Path,
    *,
    expected_channels: int = 9,
    size: int = 128,
    preprocess: GrayscalePreprocessConfig | None = None,
) -> GrayscaleTargetArtifacts:
    """Create grayscale targets and export-ready diagnostics for image adaptation."""
    _grid_size_for_channels(expected_channels)
    preprocess_config = preprocess or GrayscalePreprocessConfig()
    source_grayscale = load_rgb_image_as_square_canvas(path, size).convert("L")
    source_array = np.asarray(source_grayscale, dtype=np.float32) / 255.0
    processed = build_dimmed_grayscale_image(
        source_array,
        max_intensity=preprocess_config.max_intensity,
        gamma=preprocess_config.gamma,
        flat_region_darkening=preprocess_config.flat_region_darkening,
        detail_boost=preprocess_config.detail_boost,
    )
    if not np.any(processed > 0.0):
        raise ValueError("grayscale target generation produced an empty image")
    targets, tile_rows = split_grayscale_image_into_channel_tiles_with_report(
        processed,
        expected_channels=expected_channels,
        tile_balance_strength=preprocess_config.tile_balance_strength,
        tile_balance_clip=preprocess_config.tile_balance_clip,
    )
    report_rows = [
        summarize_grayscale_image(source_array, stage="source"),
        summarize_grayscale_image(processed, stage="processed"),
        *tile_rows,
    ]
    return GrayscaleTargetArtifacts(
        source_grayscale=source_array.astype(np.float32),
        processed_grayscale=processed.astype(np.float32),
        targets=targets.astype(np.float32),
        stitched_target=_stitch_channel_tiles(targets),
        report_rows=report_rows,
    )
