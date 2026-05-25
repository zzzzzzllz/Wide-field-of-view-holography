"""Multi-seed experiment runner for comparing random initializations."""

from __future__ import annotations

import argparse
import csv
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from holo_opt.cli import build_parser as build_single_parser
from holo_opt.cli import config_from_args
from holo_opt.config import ExperimentConfig, validate_config
from holo_opt.runner import ExperimentResult, run_experiment


@dataclass
class SeedBatchResult:
    batch_dir: Path
    summary_path: Path
    rows: list[dict[str, object]]
    best_seed: int
    best_run_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = build_single_parser()
    parser.description = "Run holography optimization across multiple random seeds."
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more integer seeds to run. Defaults to the single --seed value.",
    )
    return parser


def config_from_batch_args(args: argparse.Namespace) -> tuple[ExperimentConfig, list[int]]:
    config = config_from_args(args)
    seeds = list(args.seeds) if args.seeds is not None else [config.seed]
    if not seeds:
        raise ValueError("at least one seed is required")
    return config, seeds


def _safe_path_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")
    return safe if safe else "batch"


def _create_batch_dir(output_root: str | Path, label: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = root / f"{_safe_path_label(label)}_seeds_{timestamp}"
    root_resolved = root.resolve()
    batch_resolved = batch_dir.resolve()
    if not batch_resolved.is_relative_to(root_resolved):
        raise ValueError("batch directory must stay inside output_root")
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def _config_for_seed(base_config: ExperimentConfig, seed: int, runs_root: Path) -> ExperimentConfig:
    config = deepcopy(base_config)
    config.seed = int(seed)
    config.output_root = str(runs_root)
    config.label = f"{base_config.label}_seed{seed}"
    validate_config(config)
    return config


def _summary_float(summary: dict[str, object], key: str) -> float:
    return float(summary.get(key, 0.0))


def _row_for_result(seed: int, result: ExperimentResult) -> dict[str, object]:
    summary = result.final_metrics.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("final metrics summary must be a dictionary")
    return {
        "seed": int(seed),
        "run_dir": str(result.run_dir),
        "score": _summary_float(summary, "score"),
        "image_error": _summary_float(summary, "image_error"),
        "gray_level_error": _summary_float(summary, "gray_level_error"),
        "efficiency_balance_penalty": _summary_float(summary, "efficiency_balance_penalty"),
        "mean_eta": _summary_float(summary, "mean_eta"),
        "best": "",
    }


def _write_seed_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "run_dir",
        "score",
        "image_error",
        "gray_level_error",
        "efficiency_balance_penalty",
        "mean_eta",
        "best",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_seed_batch(config: ExperimentConfig, seeds: list[int]) -> SeedBatchResult:
    validate_config(config)
    batch_dir = _create_batch_dir(config.output_root, config.label)
    runs_root = batch_dir / "runs"
    rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_config = _config_for_seed(config, seed, runs_root)
        result = run_experiment(seed_config)
        rows.append(_row_for_result(seed, result))

    best_index = min(range(len(rows)), key=lambda index: float(rows[index]["score"]))
    rows[best_index]["best"] = "1"
    summary_path = batch_dir / "seed_summary.csv"
    _write_seed_summary(summary_path, rows)
    return SeedBatchResult(
        batch_dir=batch_dir,
        summary_path=summary_path,
        rows=rows,
        best_seed=int(rows[best_index]["seed"]),
        best_run_dir=Path(str(rows[best_index]["run_dir"])),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, seeds = config_from_batch_args(args)
    result = run_seed_batch(config, seeds)
    print(f"Saved seed summary to {result.summary_path}")
    print(f"Best seed: {result.best_seed}")
    print(f"Best run: {result.best_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
