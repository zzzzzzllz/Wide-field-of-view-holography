"""Export helpers for saving run artifacts, metrics tables, and diagnostic plots."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from holo_opt.config import ExperimentConfig, config_to_dict
from holo_opt.metrics import normalize_per_channel


def export_results(
    config: ExperimentConfig,
    targets: np.ndarray,
    intensities: np.ndarray,
    phdx: np.ndarray,
    phdy: np.ndarray,
    losses: list[float] | np.ndarray,
    eta_history: list[list[float]] | np.ndarray,
    weights_history: list[list[float]] | np.ndarray,
    metrics: dict[str, Any],
    diagnostics: list[dict[str, float]] | None = None,
    loss_terms_history: list[dict[str, float]] | None = None,
    outer_summaries: list[tuple[int, np.ndarray]] | None = None,
) -> Path:
    """Write one complete experiment folder containing images, tables, and raw arrays."""
    run_dir = create_run_dir(config)
    _write_config(run_dir / "config.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    _write_npz(
        run_dir / "optimized_results.npz",
        config,
        targets,
        intensities,
        phdx,
        phdy,
        losses,
        eta_history,
        weights_history,
    )
    _write_metrics_csv(run_dir / "metrics.csv", metrics)
    np.savetxt(run_dir / "phdx.csv", np.asarray(phdx), delimiter=",")
    np.savetxt(run_dir / "phdy.csv", np.asarray(phdy), delimiter=",")

    _plot_summary(run_dir / "summary.png", targets, intensities)
    _plot_loss_curve(run_dir / "loss_curve.png", losses)
    _plot_eta_curve(run_dir / "eta_curve.png", eta_history)
    _plot_gray_levels(run_dir / "gray_levels.png", metrics)
    if diagnostics:
        _write_rows_csv(run_dir / "diagnostics.csv", diagnostics)
    if loss_terms_history:
        _write_rows_csv(run_dir / "loss_terms.csv", loss_terms_history)
        _plot_loss_terms(run_dir / "loss_terms.png", loss_terms_history)
    if outer_summaries:
        for outer_index, outer_intensities in outer_summaries:
            _plot_summary(run_dir / f"outer_{outer_index:03d}_summary.png", targets, outer_intensities)

    return run_dir


def create_run_dir(config: ExperimentConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    label = _safe_path_label(config.label)
    run_dir = root / f"{label}_{config.n_channels}ch_{config.size}_{timestamp}"
    root_resolved = root.resolve()
    run_resolved = run_dir.resolve()
    if not run_resolved.is_relative_to(root_resolved):
        raise ValueError("run directory must stay inside output_root")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_config(path: Path, config: ExperimentConfig) -> None:
    _write_json(path, config_to_dict(config))


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _safe_path_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_")
    if not safe:
        safe = "run"
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{idx}" for idx in range(1, 10)), *(f"LPT{idx}" for idx in range(1, 10))}
    if safe.upper() in reserved:
        safe = f"run_{safe}"
    return safe


def _write_npz(
    path: Path,
    config: ExperimentConfig,
    targets: np.ndarray,
    intensities: np.ndarray,
    phdx: np.ndarray,
    phdy: np.ndarray,
    losses: list[float] | np.ndarray,
    eta_history: list[list[float]] | np.ndarray,
    weights_history: list[list[float]] | np.ndarray,
) -> None:
    np.savez(
        path,
        phdx=np.asarray(phdx),
        phdy=np.asarray(phdy),
        targets=np.asarray(targets),
        intensities=np.asarray(intensities),
        pairMat=np.asarray(config.pair_mat, dtype=np.int32),
        loss=np.asarray(losses, dtype=np.float32),
        eta_history=np.asarray(eta_history, dtype=np.float32),
        weights_history=np.asarray(weights_history, dtype=np.float32),
    )


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    score = metrics["summary"]["score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "mse", "eta", "gray_level_error", "score"])
        for row in metrics["rows"]:
            writer.writerow([row["channel"], row["mse"], row["eta"], row["gray_level_error"], score])


def _write_rows_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(path: Path, targets: np.ndarray, intensities: np.ndarray) -> None:
    target_array = np.asarray(targets)
    reconstruction = normalize_per_channel(intensities)
    channels = target_array.shape[0]
    fig, axes = plt.subplots(2, channels, figsize=(max(9.0, channels * 1.5), 3.0), squeeze=False)
    try:
        for channel in range(channels):
            axes[0, channel].imshow(target_array[channel], cmap="gray", vmin=0.0, vmax=1.0)
            axes[0, channel].set_title(f"T{channel + 1}")
            axes[0, channel].axis("off")
            axes[1, channel].imshow(reconstruction[channel], cmap="gray", vmin=0.0, vmax=1.0)
            axes[1, channel].set_title(f"R{channel + 1}")
            axes[1, channel].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_loss_curve(path: Path, losses: list[float] | np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    try:
        ax.plot(np.asarray(losses, dtype=np.float32), marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_loss_terms(path: Path, loss_terms_history: list[dict[str, float]]) -> None:
    terms = ("image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background")
    steps = np.asarray([row["step"] for row in loss_terms_history], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    try:
        for term in terms:
            values = np.asarray([row[term] for row in loss_terms_history], dtype=np.float32)
            ax.plot(steps, values, linewidth=1.5, label=term)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss term")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_eta_curve(path: Path, eta_history: list[list[float]] | np.ndarray) -> None:
    eta = np.asarray(eta_history, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    try:
        if eta.ndim == 1:
            ax.plot(eta, marker="o", label="eta")
        elif eta.size:
            for channel in range(eta.shape[1]):
                ax.plot(eta[:, channel], alpha=0.5, linewidth=1.0)
            ax.plot(np.mean(eta, axis=1), color="black", linewidth=2.0, label="mean")
            ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eta")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_gray_levels(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics["rows"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    try:
        gray_means = [row.get("gray_means") for row in rows if row.get("gray_means") is not None]
        if gray_means:
            for index, means in enumerate(gray_means, start=1):
                ax.plot(means, alpha=0.5, linewidth=1.0, label=f"ch {index}")
            ax.set_xlabel("Gray level")
            ax.set_ylabel("Mean reconstruction")
        else:
            channels = [row["channel"] for row in rows]
            errors = [row["gray_level_error"] for row in rows]
            ax.bar(channels, errors)
            ax.set_xlabel("Channel")
            ax.set_ylabel("Gray level error")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
