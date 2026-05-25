# Workflow

This file explains how to run, inspect, and validate the current project.

## Platform

- Primary user-facing platform: Windows.
- Prefer PowerShell examples.
- Prefer `py` for Python commands.
- macOS is acceptable for reading, documentation, and lightweight checks, but Windows should be used for final runtime confirmation.

## Install

```powershell
py -m pip install -r requirements.txt
```

Main dependencies:

- `numpy`
- `torch`
- `scipy`
- `matplotlib`
- `pillow`

## Main Commands

Run the full test suite:

```powershell
py -m unittest discover -s tests -q
```

Run a small smoke optimization:

```powershell
py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 2 --device cpu --output-root outputs/holo_experiments --label smoke
```

Run a diagnostic standard-target optimization:

```powershell
py -m holo_opt.cli --target-mode standard --size 64 --epochs-per-chunk 300 --outer-loops 5 --device cpu --output-root outputs/holo_experiments --label diagnostic --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001 --background-weight 0.0 --local-uniformity-weight 0.02
```

Run the main visual-quality benchmark with the geometric grayscale image:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/benchmarks/benchmark_geometric_512.png --size 128 --epochs-per-chunk 1000 --outer-loops 10 --device cpu --output-root outputs/holo_experiments --label benchmark_geometric_uniformity --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001 --background-weight 0.0 --local-uniformity-weight 0.02
```

Evaluate one finished run with the standalone flat-region benchmark metric:

```powershell
py -m holo_opt.benchmark_eval --run-dir outputs/holo_experiments/<run_folder>
```

Run current grayscale target mode:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/demo.png --size 128 --device cpu --output-root outputs/holo_experiments --label grayscale
```

Run direct grayscale target mode without the extra grayscale shaping used by `grayscale`:

```powershell
py -m holo_opt.cli --target-mode grayscale_direct --target-path inputs/lineart_sources/demo.png --size 128 --device cpu --output-root outputs/holo_experiments --label grayscale_direct
```

To push harder against speckle inside gray blocks, raise the direct-image noise losses:

```powershell
py -m holo_opt.cli --target-mode grayscale_direct --target-path inputs/lineart_sources/demo.png --size 128 --device cpu --output-root outputs/holo_experiments --label grayscale_direct_denoise --local-uniformity-weight 0.02 --high-frequency-weight 0.05
```

Preview current grayscale conversion:

```powershell
py -m holo_opt.grayscale_preview --input demo.png --size 64
```

## Input Directories

- `inputs/lineart_sources/`
  - Put user target images here for `lineart`, `grayscale`, and preview commands.

## Output Directories

- `outputs/holo_experiments/`
  - Main optimization runs.
- `outputs/lineart_preview/`
  - Line-art preview output.
- `outputs/grayscale_preview/`
  - Grayscale preview output.

Do not delete output directories unless the user explicitly asks.

## Result Reading Order

For image quality, inspect:

1. `stitched_comparison.png`
2. `summary.png`
3. latest `outer_###_stitched_comparison.png`
4. latest `outer_###_summary.png`
5. `loss_terms.png`
6. `diagnostics.csv`
7. `eta_curve.png`
8. `gray_levels.png`
9. `metrics.json` and `metrics.csv`
10. `benchmark_flat_region_eval.json`

## Important Output Files

- `stitched_comparison.png`
  - Primary final target-vs-reconstruction file.
  - Left side: target channels stitched into a 3x3 image.
  - Right side: reconstructed channels stitched into a 3x3 image.
- `summary.png`
  - Per-channel target/reconstruction grid.
  - Useful for finding which channel failed.
- `outer_###_stitched_comparison.png`
  - Intermediate stitched comparison at diagnostic outer loops.
- `diagnostics.csv`
  - Outer-loop score, image error, gray-level error, mean eta, and weight range.
- `loss_terms.csv` / `loss_terms.png`
  - Structured loss history.
  - Includes the local uniformity term used to reduce speckle inside non-black regions.
- `benchmark_flat_region_eval.json`
  - Standalone benchmark-evaluation output written by `py -m holo_opt.benchmark_eval`.
  - Reports stitched flat-region local variance and local standard deviation on the fixed geometric benchmark.
- `phdx.csv` and `phdy.csv`
  - Optimized proxy phase maps.

## Validation Rules

- Documentation-only changes do not require a full optimization run.
- Algorithm, runner, target-generation, export, CLI, or metric changes should run:

```powershell
py -m unittest discover -s tests -q
```

- For behavior changes, also run a small smoke optimization and inspect exported images.
- For visual-quality judgment, prefer the geometric benchmark command above over the built-in `standard` target.
- A smoke run is only for checking the pipeline. For image-quality comparison, use about 10,000 optimization steps.
- When only macOS checks were performed, say so clearly and provide Windows commands for final verification.
