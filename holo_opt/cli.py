"""Command-line entrypoint for configuring and launching a holography run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from holo_opt.config import (
    ExperimentConfig,
    GrayscalePreprocessConfig,
    GuidedModeConfig,
    LossConfig,
    PhysicalConfig,
    WeightUpdateConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 9-channel grayscale holography optimization."
    )
    parser.add_argument("--target-mode", choices=["standard", "mat", "lineart", "grayscale"], default="standard")
    parser.add_argument("--target-path", default=None)
    parser.add_argument("--mat-variable", default="bw_all")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs-per-chunk", type=int, default=300)
    parser.add_argument("--outer-loops", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-root", default="outputs/holo_experiments")
    parser.add_argument("--label", default="quick9")
    parser.add_argument("--diagnostic-interval", type=int, default=1)
    parser.add_argument("--lambda-nm", type=float, default=532.0)
    parser.add_argument("--px-nm", type=float, default=830.0)
    parser.add_argument("--py-nm", type=float, default=830.0)
    parser.add_argument("--guided-enabled", action="store_true", dest="guided_enabled", default=True)
    parser.add_argument("--guided-disabled", action="store_false", dest="guided_enabled")
    parser.add_argument("--neff", type=float, default=2.05)
    parser.add_argument("--alpha-deg", type=float, default=-16.7)
    parser.add_argument("--weight-alpha", type=float, default=0.5)
    parser.add_argument("--weight-beta", type=float, default=0.5)
    parser.add_argument("--eta-balance-weight", type=float, default=0.05)
    parser.add_argument("--gray-monotonic-weight", type=float, default=0.1)
    parser.add_argument("--phase-smoothness-weight", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.0)
    parser.add_argument("--grayscale-max-intensity", type=float, default=0.65)
    parser.add_argument("--grayscale-gamma", type=float, default=1.6)
    parser.add_argument("--grayscale-flat-darkening", type=float, default=0.55)
    parser.add_argument("--grayscale-detail-boost", type=float, default=0.2)
    parser.add_argument("--grayscale-tile-balance-strength", type=float, default=0.35)
    parser.add_argument("--grayscale-tile-balance-clip", type=float, default=1.35)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        size=args.size,
        epochs_per_chunk=args.epochs_per_chunk,
        outer_loops=args.outer_loops,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        target_mode=args.target_mode,
        target_path=args.target_path,
        mat_variable=args.mat_variable,
        output_root=args.output_root,
        label=args.label,
        diagnostic_interval=args.diagnostic_interval,
        physical=PhysicalConfig(
            lambda_nm=args.lambda_nm,
            px_nm=args.px_nm,
            py_nm=args.py_nm,
        ),
        guided_mode=GuidedModeConfig(
            enabled=args.guided_enabled,
            neff=args.neff,
            alpha_deg=args.alpha_deg,
        ),
        weight_update=WeightUpdateConfig(
            alpha=args.weight_alpha,
            beta=args.weight_beta,
        ),
        loss=LossConfig(
            eta_balance_weight=args.eta_balance_weight,
            gray_monotonic_weight=args.gray_monotonic_weight,
            phase_smoothness_weight=args.phase_smoothness_weight,
            background_weight=args.background_weight,
        ),
        grayscale_preprocess=GrayscalePreprocessConfig(
            max_intensity=args.grayscale_max_intensity,
            gamma=args.grayscale_gamma,
            flat_region_darkening=args.grayscale_flat_darkening,
            detail_boost=args.grayscale_detail_boost,
            tile_balance_strength=args.grayscale_tile_balance_strength,
            tile_balance_clip=args.grayscale_tile_balance_clip,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    from holo_opt.runner import run_experiment

    result = run_experiment(config)
    score = float(result.final_metrics["summary"]["score"])  # type: ignore[index]
    print(f"Saved results to {result.run_dir}")
    print(f"Final score: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
