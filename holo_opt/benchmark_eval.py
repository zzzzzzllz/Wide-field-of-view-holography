"""Standalone benchmark evaluator for flat-region noise and image similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from holo_opt.metrics import evaluate_metrics, normalize_per_channel


DEFAULT_BENCHMARK_PATH = "inputs/lineart_sources/benchmarks/benchmark_geometric_512.png"
DEFAULT_GRADIENT_THRESHOLD = 0.01
DEFAULT_TARGET_MIN = 0.05
DEFAULT_KERNEL_SIZE = 9


def _grid_size_for_channels(channels: int) -> int:
    grid_size = int(round(channels ** 0.5))
    if grid_size * grid_size != channels:
        raise ValueError("stitched evaluation requires a square channel count")
    return grid_size


def stitch_channel_grid(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("channel grid values must have shape (channels, height, width)")
    channels, _, _ = array.shape
    grid_size = _grid_size_for_channels(channels)
    rows = []
    for row_index in range(grid_size):
        start = row_index * grid_size
        rows.append(np.concatenate([array[start + col_index] for col_index in range(grid_size)], axis=1))
    return np.concatenate(rows, axis=0)


def forward_gradient_max(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("image must be 2D")
    padded = np.pad(arr, ((0, 1), (0, 1)), mode="edge")
    gradient_y = np.abs(padded[1:, :-1] - padded[:-1, :-1])
    gradient_x = np.abs(padded[:-1, 1:] - padded[:-1, :-1])
    return np.maximum(gradient_x, gradient_y).astype(np.float32)


def flat_region_mask(
    target: np.ndarray,
    *,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    target_min: float = DEFAULT_TARGET_MIN,
) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("target must be 2D")
    if gradient_threshold < 0.0:
        raise ValueError("gradient_threshold must be nonnegative")
    if target_min < 0.0:
        raise ValueError("target_min must be nonnegative")
    gradient = forward_gradient_max(arr)
    return ((gradient <= float(gradient_threshold)) & (arr > float(target_min))).astype(bool)


def _local_mean(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    padded = np.pad(np.asarray(values, dtype=np.float32), ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window_sum = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    return (window_sum / float(kernel_size * kernel_size)).astype(np.float32)


def local_variance_map(values: np.ndarray, kernel_size: int = DEFAULT_KERNEL_SIZE) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("values must be 2D")
    local_mean = _local_mean(arr, kernel_size)
    local_mean_sq = _local_mean(arr * arr, kernel_size)
    return np.maximum(0.0, local_mean_sq - local_mean * local_mean).astype(np.float32)


def evaluate_flat_region_noise(
    reconstruction: np.ndarray,
    target: np.ndarray,
    *,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    target_min: float = DEFAULT_TARGET_MIN,
) -> dict[str, float]:
    recon = np.asarray(reconstruction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    if recon.shape != tgt.shape:
        raise ValueError("reconstruction and target must have the same shape")
    if recon.ndim != 2:
        raise ValueError("reconstruction and target must be 2D")

    mask = flat_region_mask(tgt, gradient_threshold=gradient_threshold, target_min=target_min)
    pixel_fraction = float(np.mean(mask))
    if not np.any(mask):
        return {"local_variance": 0.0, "local_std": 0.0, "pixel_fraction": pixel_fraction}

    residual = recon - tgt
    variance = local_variance_map(residual, kernel_size=kernel_size)
    local_variance = float(np.mean(variance[mask]))
    return {
        "local_variance": local_variance,
        "local_std": float(np.sqrt(max(local_variance, 0.0))),
        "pixel_fraction": pixel_fraction,
    }


def evaluate_reconstruction(
    intensities: np.ndarray,
    targets: np.ndarray,
    *,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    target_min: float = DEFAULT_TARGET_MIN,
) -> dict[str, Any]:
    arr_i = np.asarray(intensities, dtype=np.float32)
    arr_t = np.asarray(targets, dtype=np.float32)
    if arr_i.ndim != 3 or arr_t.ndim != 3 or arr_i.shape != arr_t.shape:
        raise ValueError("intensities and targets must have the same 3D shape")

    base_metrics = evaluate_metrics(arr_i, arr_t)
    norm_i = normalize_per_channel(arr_i)
    norm_t = normalize_per_channel(arr_t)

    per_channel = []
    for channel in range(arr_i.shape[0]):
        channel_eval = evaluate_flat_region_noise(
            norm_i[channel],
            norm_t[channel],
            kernel_size=kernel_size,
            gradient_threshold=gradient_threshold,
            target_min=target_min,
        )
        per_channel.append({
            "channel": channel + 1,
            "local_variance": channel_eval["local_variance"],
            "local_std": channel_eval["local_std"],
            "pixel_fraction": channel_eval["pixel_fraction"],
        })

    stitched_eval = evaluate_flat_region_noise(
        stitch_channel_grid(norm_i),
        stitch_channel_grid(norm_t),
        kernel_size=kernel_size,
        gradient_threshold=gradient_threshold,
        target_min=target_min,
    )
    mean_channel_noise = float(np.mean([row["local_variance"] for row in per_channel])) if per_channel else 0.0
    mean_channel_fraction = float(np.mean([row["pixel_fraction"] for row in per_channel])) if per_channel else 0.0

    return {
        "definition": {
            "normalization": "per_channel_then_stitched",
            "residual": "reconstruction_minus_target",
            "flat_region_rule": {
                "gradient_threshold": float(gradient_threshold),
                "target_min": float(target_min),
            },
            "kernel_size": int(kernel_size),
        },
        "base_metrics": base_metrics,
        "flat_region_noise": {
            "stitched": stitched_eval,
            "mean_channel_local_variance": mean_channel_noise,
            "mean_channel_pixel_fraction": mean_channel_fraction,
            "channels": per_channel,
        },
    }


def _load_run_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = run_dir / "optimized_results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(str(npz_path))
    with np.load(npz_path) as data:
        return np.asarray(data["intensities"], dtype=np.float32), np.asarray(data["targets"], dtype=np.float32)


def _load_config_target_path(run_dir: Path) -> str | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    target_path = config.get("target_path")
    return str(target_path) if target_path else None


def evaluate_run_dir(
    run_dir: str | Path,
    *,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    target_min: float = DEFAULT_TARGET_MIN,
    benchmark_path: str = DEFAULT_BENCHMARK_PATH,
) -> dict[str, Any]:
    path = Path(run_dir)
    intensities, targets = _load_run_arrays(path)
    result = evaluate_reconstruction(
        intensities,
        targets,
        kernel_size=kernel_size,
        gradient_threshold=gradient_threshold,
        target_min=target_min,
    )
    configured_target_path = _load_config_target_path(path)
    benchmark_match = configured_target_path is None or Path(configured_target_path) == Path(benchmark_path)
    result["run_dir"] = str(path)
    result["benchmark_image"] = benchmark_path
    result["configured_target_path"] = configured_target_path
    result["benchmark_match"] = bool(benchmark_match)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate benchmark flat-region noise from an exported run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE)
    parser.add_argument("--gradient-threshold", type=float, default=DEFAULT_GRADIENT_THRESHOLD)
    parser.add_argument("--target-min", type=float, default=DEFAULT_TARGET_MIN)
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = evaluate_run_dir(
        args.run_dir,
        kernel_size=args.kernel_size,
        gradient_threshold=args.gradient_threshold,
        target_min=args.target_min,
        benchmark_path=args.benchmark_path,
    )
    output_path = Path(args.output_json) if args.output_json else Path(args.run_dir) / "benchmark_flat_region_eval.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, ensure_ascii=False, indent=2)

    stitched = evaluation["flat_region_noise"]["stitched"]
    summary = evaluation["base_metrics"]["summary"]
    print(f"Saved evaluation to {output_path}")
    print(f"benchmark_match={evaluation['benchmark_match']}")
    print(f"stitched_flat_region_local_variance={stitched['local_variance']:.6f}")
    print(f"stitched_flat_region_local_std={stitched['local_std']:.6f}")
    print(f"stitched_flat_region_pixel_fraction={stitched['pixel_fraction']:.6f}")
    print(f"image_error={summary['image_error']:.6f}")
    print(f"gray_level_error={summary['gray_level_error']:.6f}")
    print(f"mean_eta={summary['mean_eta']:.6f}")
    print(f"score={summary['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
