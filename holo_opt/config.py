from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    pair_mat: list[list[int]] = field(default_factory=lambda: [row[:] for row in DEFAULT_PAIR_MAT])
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    guided_mode: GuidedModeConfig = field(default_factory=GuidedModeConfig)
    weight_update: WeightUpdateConfig = field(default_factory=WeightUpdateConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)


def validate_config(config: ExperimentConfig) -> None:
    if config.n_channels <= 0:
        raise ValueError("n_channels must be positive")
    if config.size <= 0:
        raise ValueError("size must be positive")
    if config.levels < 2:
        raise ValueError("levels must be at least 2")
    if config.epochs_per_chunk <= 0:
        raise ValueError("epochs_per_chunk must be positive")
    if config.outer_loops <= 0:
        raise ValueError("outer_loops must be positive")
    if config.lr <= 0:
        raise ValueError("lr must be positive")
    if len(config.pair_mat) != config.n_channels:
        raise ValueError("pair_mat length must equal n_channels")
    for row in config.pair_mat:
        if len(row) != 2:
            raise ValueError("each pair_mat row must contain two integers")
        if not all(type(value) is int for value in row):
            raise ValueError("pair_mat values must be integers")
    if config.target_mode not in {"standard", "mat"}:
        raise ValueError("target_mode must be standard or mat")
    if config.target_mode == "mat" and not config.target_path:
        raise ValueError("target_path is required when target_mode is mat")


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    if config.target_path is not None:
        data["target_path"] = str(Path(config.target_path))
    return data
