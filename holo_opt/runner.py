"""Main experiment loop that optimizes phase maps and records diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from holo_opt.config import ExperimentConfig, ScoreConfig, validate_config
from holo_opt.export import export_results
from holo_opt.field import compute_intensities, compute_loss_terms
from holo_opt.line_targets import (
    GrayscaleTargetArtifacts,
    generate_grayscale_target_artifacts,
    generate_line_art_targets,
)
from holo_opt.metrics import evaluate_metrics
from holo_opt.region_masks import RegionMasks, generate_region_masks
from holo_opt.targets import generate_gray_step_targets, load_mat_targets, validate_targets
from holo_opt.weights import update_weights


PROGRESS_INTERVAL_STEPS = 500


@dataclass
class ExperimentResult:
    run_dir: Path
    final_intensities: np.ndarray
    final_metrics: dict[str, object]


@dataclass
class TargetLoadResult:
    targets: np.ndarray
    grayscale_artifacts: GrayscaleTargetArtifacts | None = None


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def load_targets_for_config(config: ExperimentConfig) -> np.ndarray:
    return load_targets_bundle_for_config(config).targets


def load_targets_bundle_for_config(config: ExperimentConfig) -> TargetLoadResult:
    if config.target_mode == "standard":
        targets = generate_gray_step_targets(config.n_channels, config.size, config.levels)
        return TargetLoadResult(targets=validate_targets(targets, expected_channels=config.n_channels))
    elif config.target_mode == "mat":
        targets = load_mat_targets(
            config.target_path,
            variable=config.mat_variable,
            expected_channels=config.n_channels,
        )
        return TargetLoadResult(targets=validate_targets(targets, expected_channels=config.n_channels))
    elif config.target_mode == "lineart":
        targets = generate_line_art_targets(
            config.target_path,
            expected_channels=config.n_channels,
            size=config.size,
        )
        return TargetLoadResult(targets=validate_targets(targets, expected_channels=config.n_channels))
    elif config.target_mode == "grayscale":
        artifacts = generate_grayscale_target_artifacts(
            config.target_path,
            expected_channels=config.n_channels,
            size=config.size,
            preprocess=config.grayscale_preprocess,
        )
        return TargetLoadResult(
            targets=validate_targets(artifacts.targets, expected_channels=config.n_channels),
            grayscale_artifacts=artifacts,
        )
    else:
        raise ValueError("target_mode must be standard, mat, lineart, or grayscale")


def compute_score(summary: dict[str, object], score_config: ScoreConfig) -> float:
    return float(
        score_config.image_weight * float(summary["image_error"])
        + score_config.gray_level_weight * float(summary["gray_level_error"])
        + score_config.balance_weight * float(summary["efficiency_balance_penalty"])
        - score_config.total_efficiency_weight * float(summary["mean_eta"])
    )


def apply_score_config(metrics: dict[str, object], score_config: ScoreConfig) -> dict[str, object]:
    summary = metrics["summary"]
    if not isinstance(summary, dict):
        raise ValueError("metrics summary must be a dictionary")
    summary["score"] = compute_score(summary, score_config)
    return metrics


def selection_metric_value(metrics: dict[str, object], metric_name: str) -> float:
    summary = metrics["summary"]
    if not isinstance(summary, dict):
        raise ValueError("metrics summary must be a dictionary")
    if metric_name not in summary:
        raise ValueError(f"selection metric not found: {metric_name}")
    return float(summary[metric_name])


def loss_config_to_dict(config: ExperimentConfig) -> dict[str, float]:
    return {
        "image_weight": config.loss.image_weight,
        "eta_balance_weight": config.loss.eta_balance_weight,
        "gray_monotonic_weight": config.loss.gray_monotonic_weight,
        "phase_smoothness_weight": config.loss.phase_smoothness_weight,
        "background_weight": config.loss.background_weight,
    }


def format_progress_message(step: int, total_steps: int, loss_value: float) -> str | None:
    if step % PROGRESS_INTERVAL_STEPS != 0:
        return None
    return f"step {step}/{total_steps} loss={loss_value:.6f}"


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    validate_config(config)
    device = resolve_device(config.device)
    print(f"Device: {device.type}", flush=True)
    started_at = time.perf_counter()
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Stage 1: prepare the multi-channel target images that each diffraction
    # channel should reproduce in the far field.
    target_bundle = load_targets_bundle_for_config(config)
    targets_np = target_bundle.targets
    region_masks_np: RegionMasks | None = None
    if config.region_mask.enabled or config.signal_window.image_loss_mode in {"signal_window", "hybrid"}:
        region_masks_np = generate_region_masks(targets_np, config.region_mask)
    height, width = targets_np.shape[-2], targets_np.shape[-1]
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    region_masks_torch: dict[str, torch.Tensor] | None = None
    if region_masks_np is not None:
        region_masks_torch = {
            "edge": torch.as_tensor(region_masks_np.edge, dtype=torch.float32, device=device),
            "signal": torch.as_tensor(region_masks_np.signal, dtype=torch.float32, device=device),
            "flat": torch.as_tensor(region_masks_np.flat, dtype=torch.float32, device=device),
            "dark": torch.as_tensor(region_masks_np.dark, dtype=torch.float32, device=device),
            "relaxed": torch.as_tensor(region_masks_np.relaxed, dtype=torch.float32, device=device),
        }
    pair_mat = torch.as_tensor(config.pair_mat, dtype=torch.float32, device=device)

    # Stage 2: initialize the proxy on-chip structure parameters. The optimizer
    # does not directly update nanostructure geometry; it updates the effective
    # phase maps phdx/phdy used by the FFT model.
    phdx = torch.tensor(
        rng.uniform(0.0, 2.0 * np.pi, size=(height, width)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    phdy = torch.tensor(
        rng.uniform(0.0, 2.0 * np.pi, size=(height, width)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    weights_np = np.ones(config.n_channels, dtype=np.float32)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([phdx, phdy], lr=config.lr)

    losses: list[float] = []
    loss_terms_history: list[dict[str, float]] = []
    eta_history: list[list[float]] = []
    weights_history: list[list[float]] = [weights_np.astype(float).tolist()]
    diagnostics: list[dict[str, float]] = []
    outer_summaries: list[tuple[int, np.ndarray]] = []
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    total_steps = config.outer_loops * config.epochs_per_chunk
    loss_weights = loss_config_to_dict(config)
    loss_term_names = ["total", "image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background"]
    if config.signal_window.image_loss_mode in {"signal_window", "hybrid"}:
        loss_term_names.extend([
            "signal_window",
            "edge_mse",
            "signal_mse",
            "flat_lowpass_mse",
            "relaxed_lowpass_mse",
            "dark_leakage",
        ])

    # Stage 3: optimize the proxy structure so that every configured channel
    # produces a far-field image close to its own target image.
    for _outer_index in range(config.outer_loops):
        for _epoch_index in range(config.epochs_per_chunk):
            optimizer.zero_grad(set_to_none=True)
            terms = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                loss_weights,
                region_masks=region_masks_torch,
                signal_window_config=config.signal_window,
            )
            loss = terms["total"]
            if not torch.isfinite(loss).item():
                raise RuntimeError("non-finite loss encountered")
            loss.backward()
            optimizer.step()
            term_values = torch.stack([terms[name].detach() for name in loss_term_names]).cpu().tolist()
            loss_value = float(term_values[0])
            losses.append(loss_value)
            progress_message = format_progress_message(len(losses), total_steps, loss_value)
            if progress_message is not None:
                print(progress_message, flush=True)
            loss_terms_history.append({
                "step": float(len(losses)),
                **{name: float(term_values[index]) for index, name in enumerate(loss_term_names)},
            })

        with torch.no_grad():
            intensities_np = compute_intensities(phdx, phdy, pair_mat).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(intensities_np).all():
            raise RuntimeError("non-finite intensities encountered")
        metrics = apply_score_config(evaluate_metrics(intensities_np, targets_np), config.score)
        score = float(metrics["summary"]["score"])  # type: ignore[index]
        selection_value = selection_metric_value(metrics, config.selection_metric)
        outer_number = _outer_index + 1
        summary = metrics["summary"]  # type: ignore[assignment]
        diagnostics.append({
            "outer": float(outer_number),
            "loss": losses[-1],
            "score": score,
            "mean_eta": float(summary["mean_eta"]),  # type: ignore[index]
            "eta_balance": float(summary["efficiency_balance_penalty"]),  # type: ignore[index]
            "image_error": float(summary["image_error"]),  # type: ignore[index]
            "gray_level_error": float(summary["gray_level_error"]),  # type: ignore[index]
            "weight_min": float(np.min(weights_np)),
            "weight_max": float(np.max(weights_np)),
        })
        if outer_number % config.diagnostic_interval == 0:
            outer_summaries.append((outer_number, intensities_np.copy()))
        if np.isfinite(selection_value) and selection_value < best_score:
            best_score = selection_value
            best_state = {
                "intensities": intensities_np.copy(),
                "phdx": phdx.detach().cpu().numpy().astype(np.float32).copy(),
                "phdy": phdy.detach().cpu().numpy().astype(np.float32).copy(),
                "metrics": metrics,
            }

        eta = np.asarray([row["eta"] for row in metrics["rows"]], dtype=np.float32)  # type: ignore[index]
        channel_error = np.asarray(
            [float(row["mse"]) + float(row["gray_level_error"]) for row in metrics["rows"]],  # type: ignore[index]
            dtype=np.float32,
        )
        eta_history.append(eta.astype(float).tolist())
        weights_np = update_weights(
            weights_np,
            eta,
            channel_error,
            alpha=config.weight_update.alpha,
            beta=config.weight_update.beta,
            clip_min=config.weight_update.clip_min,
            clip_max=config.weight_update.clip_max,
            epsilon=config.weight_update.epsilon,
        )
        weights_history.append(weights_np.astype(float).tolist())
        weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

    if best_state is None:
        raise RuntimeError("no valid optimization state produced")

    # Stage 4: export the best optimization state and all diagnostics for later
    # visual inspection under outputs/holo_experiments.
    run_dir = export_results(
        config,
        targets_np,
        best_state["intensities"],
        best_state["phdx"],
        best_state["phdy"],
        losses,
        eta_history,
        weights_history,
        best_state["metrics"],
        diagnostics=diagnostics,
        loss_terms_history=loss_terms_history,
        outer_summaries=outer_summaries,
        grayscale_artifacts=target_bundle.grayscale_artifacts,
        region_masks=region_masks_np,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at
    print(f"Run finished in {elapsed_seconds:.1f} s", flush=True)
    return ExperimentResult(
        run_dir=run_dir,
        final_intensities=best_state["intensities"],
        final_metrics=best_state["metrics"],
    )
