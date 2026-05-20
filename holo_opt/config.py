"""Configuration models and validation rules for the holography pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any


DEFAULT_PAIR_MAT: list[list[int]] = [
    [-2, 2], [-2, 1], [-2, 0],
    [-3, 2], [-3, 1], [-3, 0],
    [-4, 2], [-4, 1], [-4, 0],
]


@dataclass(frozen=True)
class PhysicalConfig:
    lambda_nm: float = 532.0
    px_nm: float = 830.0
    py_nm: float = 830.0


@dataclass(frozen=True)
class GuidedModeConfig:
    enabled: bool = True
    neff: float = 2.05
    alpha_deg: float = -16.7


@dataclass
class WeightUpdateConfig:
    alpha: float = 0.5
    beta: float = 0.5
    clip_min: float = 0.5
    clip_max: float = 5.0
    epsilon: float = 1e-8


@dataclass
class ScoreConfig:
    image_weight: float = 1.0
    gray_level_weight: float = 1.0
    balance_weight: float = 1.0
    total_efficiency_weight: float = 1.0


@dataclass
class LossConfig:
    image_weight: float = 1.0
    eta_balance_weight: float = 0.05
    gray_monotonic_weight: float = 0.1
    phase_smoothness_weight: float = 1e-4
    background_weight: float = 0.0


@dataclass
class GrayscalePreprocessConfig:
    max_intensity: float = 0.65
    gamma: float = 1.6
    flat_region_darkening: float = 0.55
    detail_boost: float = 0.2
    tile_balance_strength: float = 0.35
    tile_balance_clip: float = 1.35


@dataclass
class ExperimentConfig:
    n_channels: int = 9
    size: int = 128
    levels: int = 16
    epochs_per_chunk: int = 300
    outer_loops: int = 3
    lr: float = 5e-4
    seed: int = 42
    device: str = "auto"
    target_mode: str = "standard"
    target_path: str | None = None
    mat_variable: str = "bw_all"
    output_root: str = "outputs/holo_experiments"
    label: str = "quick9"
    diagnostic_interval: int = 1
    pair_mat: list[list[int]] = field(default_factory=lambda: [row[:] for row in DEFAULT_PAIR_MAT])
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    guided_mode: GuidedModeConfig = field(default_factory=GuidedModeConfig)
    weight_update: WeightUpdateConfig = field(default_factory=WeightUpdateConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    grayscale_preprocess: GrayscalePreprocessConfig = field(default_factory=GrayscalePreprocessConfig)


def _all_positive(values: list[float]) -> bool:
    try:
        return all(isfinite(value) and value > 0 for value in values)
    except (TypeError, ValueError):
        return False


def _all_nonnegative(values: list[float]) -> bool:
    try:
        return all(isfinite(value) and value >= 0 for value in values)
    except (TypeError, ValueError):
        return False


def _all_finite(values: list[float]) -> bool:
    try:
        return all(isfinite(value) for value in values)
    except (TypeError, ValueError):
        return False


def _is_positive_number(value: object) -> bool:
    return isinstance(value, Real) and isfinite(value) and value > 0


def _is_positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def validate_config(config: ExperimentConfig) -> None:
    if not _is_positive_integer(config.n_channels):
        raise ValueError("n_channels must be positive")
    if not _is_positive_integer(config.size):
        raise ValueError("size must be positive")
    if type(config.levels) is not int or config.levels < 2:
        raise ValueError("levels must be at least 2")
    if not _is_positive_integer(config.epochs_per_chunk):
        raise ValueError("epochs_per_chunk must be positive")
    if not _is_positive_integer(config.outer_loops):
        raise ValueError("outer_loops must be positive")
    if not _is_positive_number(config.lr):
        raise ValueError("lr must be positive")
    if not _is_positive_integer(config.diagnostic_interval):
        raise ValueError("diagnostic_interval must be positive")
    if not isinstance(config.pair_mat, (list, tuple)):
        raise ValueError("pair_mat length must equal n_channels")
    if len(config.pair_mat) != config.n_channels:
        raise ValueError("pair_mat length must equal n_channels")
    for row in config.pair_mat:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each pair_mat row must contain two integers")
        if not all(type(value) is int for value in row):
            raise ValueError("pair_mat values must be integers")
    if config.target_mode not in {"standard", "mat", "lineart", "grayscale", "image"}:
        raise ValueError("target_mode must be standard, mat, lineart, grayscale, or image")
    if config.target_mode in {"mat", "lineart", "grayscale", "image"} and not config.target_path:
        raise ValueError("target_path is required when target_mode is mat, lineart, grayscale, or image")
    if not _all_positive([config.physical.lambda_nm, config.physical.px_nm, config.physical.py_nm]):
        raise ValueError("physical values must be positive and finite")
    if not _all_positive([config.guided_mode.neff]):
        raise ValueError("guided neff must be positive and finite")
    if not _all_finite([config.guided_mode.alpha_deg]):
        raise ValueError("guided alpha_deg must be finite")
    if not _all_nonnegative([config.weight_update.alpha, config.weight_update.beta]):
        raise ValueError("weight_update alpha and beta must be nonnegative and finite")
    if not _all_positive([config.weight_update.epsilon]):
        raise ValueError("weight_update epsilon must be positive and finite")
    try:
        clip_range_valid = (
            isfinite(config.weight_update.clip_min)
            and isfinite(config.weight_update.clip_max)
            and 0 < config.weight_update.clip_min < config.weight_update.clip_max
        )
    except TypeError:
        clip_range_valid = False
    if not clip_range_valid:
        raise ValueError("weight_update clip range must satisfy 0 < clip_min < clip_max")
    score_values = [
        config.score.image_weight,
        config.score.gray_level_weight,
        config.score.balance_weight,
        config.score.total_efficiency_weight,
    ]
    if not _all_nonnegative(score_values):
        raise ValueError("score weights must be nonnegative and finite")
    loss_values = [
        config.loss.image_weight,
        config.loss.eta_balance_weight,
        config.loss.gray_monotonic_weight,
        config.loss.phase_smoothness_weight,
        config.loss.background_weight,
    ]
    if not _all_nonnegative(loss_values):
        raise ValueError("loss weights must be nonnegative and finite")
    preprocess = config.grayscale_preprocess
    if not _all_positive(
        [
            preprocess.max_intensity,
            preprocess.gamma,
            preprocess.flat_region_darkening,
            preprocess.tile_balance_clip,
        ]
    ):
        raise ValueError("grayscale preprocess positive values must be positive and finite")
    if not _all_nonnegative([preprocess.detail_boost, preprocess.tile_balance_strength]):
        raise ValueError("grayscale preprocess boost values must be nonnegative and finite")
    if preprocess.max_intensity > 1.0:
        raise ValueError("grayscale preprocess max_intensity must be at most 1")
    if preprocess.flat_region_darkening > 1.0:
        raise ValueError("grayscale preprocess flat_region_darkening must be at most 1")
    if preprocess.tile_balance_clip < 1.0:
        raise ValueError("grayscale preprocess tile_balance_clip must be at least 1")


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    if config.target_path is not None:
        data["target_path"] = str(Path(config.target_path))
    return data
