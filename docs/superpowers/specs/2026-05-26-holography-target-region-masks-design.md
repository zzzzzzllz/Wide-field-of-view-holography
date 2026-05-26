# Holography Target Region Masks Design

## Goal

Design an automatic target-region partitioning system for the current FFT-based holography optimizer, so later MRAF / signal-window losses can improve visual quality while preserving the physically meaningful `phdx` / `phdy` coupled phase model.

The first deliverable should make the masks visible and auditable before changing optimization behavior. The second deliverable should use those masks to replace or augment the current global image MSE with region-aware losses.

## Context

The project currently optimizes two shared phase proxy maps, `phdx` and `phdy`, for 9 diffraction channels. This keeps the model closer to a real metasurface device than per-channel independent phase maps, but it also makes exact full-image pixel fitting difficult.

Recent experiments show that the optimizer can reduce loss while reconstructed images still contain heavy speckle, especially in large flat grayscale regions. A global full-image MSE forces the phase-only model to fit every pixel equally, including regions that are physically hard to make smooth. This often spends optimization capacity chasing flat-region high-frequency errors instead of preserving visible structure.

The proposed design keeps the current physical model and adds target-derived region masks. These masks distinguish strict signal regions, edges, flat grayscale regions, dark background, and relaxed energy-release regions.

## Non-Goals

- Do not introduce per-channel independent phase maps as the final physical model.
- Do not add RCWA, FDTD, unit-cell library lookup, GDS export, or material modeling in this change.
- Do not require manual semantic segmentation or object labels.
- Do not depend on external vision models.
- Do not remove the existing global loss path; default behavior must remain backward compatible.

## Design Options Considered

### Option A: Target-Derived Region Masks

Generate masks from the existing target tensor using intensity thresholds, gradient magnitude, edge dilation, and local smoothness. This is deterministic, lightweight, and works for `standard`, `lineart`, `grayscale`, and `mat` targets.

This is the recommended approach.

### Option B: Classical Image Segmentation

Use watershed, connected components, or contour extraction to find objects. This can help when images contain clear separated objects, but it is less stable for grayscale holography targets and introduces more parameters.

This can be added later if target-derived masks are not expressive enough.

### Option C: Semantic Segmentation

Use a neural network to segment foreground and background. This is not appropriate for the first version because it adds a dependency, is hard to reproduce in batch experiments, and does not align with holography-specific energy constraints.

## Region Definitions

Masks are generated per channel after the final target stack has been produced. For `grayscale`, this means after the 3x3 tile split and resize, not on the original full image. This keeps the masks aligned with the channel energy budget seen by the optimizer.

Input tensor shape:

```text
targets: (channels, height, width), float32, finite
```

For each channel, compute:

```text
target_norm = target / (max(target) + epsilon)
gradient = sqrt(dx(target_norm)^2 + dy(target_norm)^2)
```

The masks are:

```text
dark_mask
  target_norm <= dark_threshold

raw_signal_mask
  target_norm > signal_threshold

raw_edge_mask
  gradient >= per-channel edge threshold

edge_mask
  dilate(raw_edge_mask, edge_dilation) and not dark_mask

signal_mask
  raw_signal_mask and not edge_mask and not dark_mask

flat_mask
  raw_signal_mask and not edge_mask and local_gradient <= flat_gradient_threshold

relaxed_mask
  not edge_mask and not signal_mask and not flat_mask and not dark_mask
```

The masks should be stored as float tensors or arrays with values `0.0` and `1.0`. During loss computation, overlapping masks should be resolved by priority:

```text
edge > signal > flat > dark > relaxed
```

This priority keeps edge pixels from being diluted by flat-region logic.

## Default Parameters

Add a region-mask configuration object:

```python
RegionMaskConfig(
    enabled=False,
    signal_threshold=0.04,
    dark_threshold=0.02,
    edge_quantile=0.85,
    edge_dilation=2,
    flat_gradient_quantile=0.35,
    min_region_fraction=0.002,
)
```

Parameter meanings:

- `enabled`: controls whether masks are generated and exported in normal runs.
- `signal_threshold`: target intensity above this is meaningful signal.
- `dark_threshold`: target intensity at or below this is background-like dark region.
- `edge_quantile`: per-channel gradient quantile used to select strong edges.
- `edge_dilation`: pixel radius for edge dilation, implemented with max pooling.
- `flat_gradient_quantile`: low-gradient cutoff inside signal regions.
- `min_region_fraction`: if a region is too small, its loss term is skipped to avoid unstable normalization.

The thresholds should be per-channel where possible. Per-channel quantiles are more robust than one global threshold because channel targets can have different brightness budgets.

## Region-Aware Loss Design

The existing `image_mse` uses energy-matched intensity:

```text
energy_matched = match_target_energy(intensities, targets)
```

The region-aware loss should use the same energy-matched image for signal fitting, so it remains consistent with the current fixed-energy interpretation.

Define a helper:

```text
masked_mean(values, mask) =
    sum(values * mask) / max(sum(mask), epsilon)
```

The first signal-window loss should be:

```text
edge_mse =
    masked_mean((energy_matched - targets)^2, edge_mask)

signal_mse =
    masked_mean((energy_matched - targets)^2, signal_mask)

flat_lowpass_mse =
    masked_mean((blur(energy_matched) - blur(targets))^2, flat_mask)

relaxed_lowpass_mse =
    masked_mean((blur(energy_matched) - blur(targets))^2, relaxed_mask)

dark_leakage =
    masked_mean(relu(normalized_intensity - dark_limit)^2, dark_mask)
```

Then:

```text
signal_window =
    edge_weight * edge_mse
    + signal_weight * signal_mse
    + flat_weight * flat_lowpass_mse
    + relaxed_weight * relaxed_lowpass_mse
    + dark_weight * dark_leakage
```

Recommended initial loss weights:

```python
SignalWindowLossConfig(
    image_loss_mode="global",
    edge_weight=2.0,
    signal_weight=1.0,
    flat_weight=0.15,
    relaxed_weight=0.03,
    dark_weight=0.2,
    dark_limit=0.03,
    lowpass_sigma=1.0,
)
```

`image_loss_mode` controls how this interacts with the existing loss:

```text
global
  Current behavior. Use existing image_mse only.

signal_window
  Replace image_mse with signal_window.

hybrid
  Use both: image_mse * image_weight + signal_window * signal_window_weight.
```

Default must be `global` so existing commands and tests keep the same behavior unless the user explicitly opts in.

## Data Flow

### Preview-Only Flow

```text
CLI args
  -> load target stack using existing target-mode logic
  -> generate region masks per channel
  -> export mask_summary.png
  -> export region_mask_report.csv
```

Recommended command:

```powershell
py -m holo_opt.mask_preview --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 128 --output-dir outputs/mask_preview
```

### Optimization Flow

```text
run_experiment()
  -> load target stack
  -> generate masks if region masks or signal-window loss are enabled
  -> pass masks into compute_loss_terms()
  -> record signal-window loss terms when enabled
  -> export masks and region report when masks are available
```

Recommended experimental command:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 128 --epochs-per-chunk 10000 --outer-loops 1 --device cuda --selection-metric image_error --region-mask-enabled --image-loss-mode signal_window --edge-weight 2.0 --signal-weight 1.0 --flat-weight 0.15 --relaxed-weight 0.03 --dark-weight 0.2 --dark-limit 0.03
```

## File and Module Boundaries

### New Module: `holo_opt/region_masks.py`

Responsibilities:

- Validate target tensor shape.
- Generate per-channel region masks.
- Resolve overlap priority.
- Compute coverage statistics.
- Provide a small dataclass such as `RegionMasks`.

Proposed public API:

```python
@dataclass
class RegionMasks:
    edge: np.ndarray
    signal: np.ndarray
    flat: np.ndarray
    dark: np.ndarray
    relaxed: np.ndarray
    report_rows: list[dict[str, float]]


def generate_region_masks(
    targets: np.ndarray,
    config: RegionMaskConfig,
) -> RegionMasks:
    ...
```

### New CLI Module: `holo_opt/mask_preview.py`

Responsibilities:

- Reuse the existing CLI target-loading options where practical.
- Generate masks without running optimization.
- Export `mask_summary.png` and `region_mask_report.csv`.

### Modify: `holo_opt/config.py`

Add:

- `RegionMaskConfig`
- `SignalWindowLossConfig`
- fields on `ExperimentConfig`
- validation rules for thresholds, quantiles, weights, dilation, and mode choices

### Modify: `holo_opt/cli.py`

Add arguments:

```text
--region-mask-enabled
--region-signal-threshold
--region-dark-threshold
--region-edge-quantile
--region-edge-dilation
--region-flat-gradient-quantile
--image-loss-mode global|signal_window|hybrid
--edge-weight
--signal-weight
--flat-weight
--relaxed-weight
--dark-weight
--dark-limit
--lowpass-sigma
```

### Modify: `holo_opt/field.py`

Add:

- `masked_mean()`
- `lowpass_blur()` or a small Gaussian/box blur helper
- `compute_signal_window_loss_terms()`
- optional `region_masks` parameter in `compute_loss_terms()`

The existing `training_loss()` compatibility wrapper should remain valid.

### Modify: `holo_opt/runner.py`

Add:

- region-mask generation after target loading
- mask passing into `compute_loss_terms()`
- loss history rows for signal-window subterms when enabled
- mask artifacts passed into export

### Modify: `holo_opt/export.py`

Add:

- `_plot_region_masks()` for `mask_summary.png`
- writing `region_mask_report.csv`
- optional plotting of signal-window loss terms in `loss_terms.png`

## Export Artifacts

When region masks are generated, export:

```text
mask_summary.png
region_mask_report.csv
```

`mask_summary.png` should show 9 channel columns or a stitched mosaic. The minimum useful layout is:

```text
target
edge_mask
signal_mask
flat_mask
dark_mask
relaxed_mask
```

`region_mask_report.csv` should include one row per channel:

```text
channel
edge_fraction
signal_fraction
flat_fraction
dark_fraction
relaxed_fraction
target_mean
target_max
edge_threshold
```

These files are required because mask quality must be visually inspected before trusting signal-window loss results.

## Metrics and Diagnostics

When `image_loss_mode` is `signal_window` or `hybrid`, `loss_terms.csv` should include:

```text
signal_window
edge_mse
signal_mse
flat_lowpass_mse
relaxed_lowpass_mse
dark_leakage
```

The existing metrics should remain:

```text
image_error
gray_level_error
efficiency_balance_penalty
mean_eta
score
```

Do not rename existing output files or remove existing columns. New columns should be appended where possible.

## Validation Strategy

### Unit Tests

Add `tests/test_region_masks.py`:

- mask generation returns all expected masks with shape `(channels, height, width)`
- masks are finite and in `[0, 1]`
- edge dilation increases or preserves edge coverage
- dark pixels are not assigned to edge/signal/flat after priority resolution
- report rows contain all required columns

Extend `tests/test_config.py`:

- valid region-mask and signal-window configs pass validation
- invalid thresholds, quantiles, negative weights, and invalid modes fail validation

Extend `tests/test_cli.py`:

- parser exposes region-mask and signal-window options
- config assembly stores them correctly

Extend `tests/test_field.py`:

- signal-window loss returns finite scalar terms
- a better reconstruction has lower edge/signal loss than a worse reconstruction
- dark leakage is zero below `dark_limit` and positive above it
- default `global` mode produces the existing `image_mse` behavior

Extend `tests/test_export.py`:

- mask artifacts write non-empty `mask_summary.png`
- `region_mask_report.csv` contains expected headers
- loss terms with signal-window keys are exported and plotted

Extend `tests/test_runner.py`:

- optimization smoke with region masks enabled exports mask artifacts
- default run without region masks still behaves like current baseline

### Manual Smoke Commands

Mask preview:

```powershell
py -m holo_opt.mask_preview --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 64 --output-dir outputs/mask_preview
```

Small CPU optimization smoke:

```powershell
py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 1 --device cpu --output-root outputs/holo_experiments --label mask_smoke --region-mask-enabled --image-loss-mode signal_window
```

Full tests:

```powershell
py -m unittest discover -s tests -q
```

CUDA quality probe:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 128 --epochs-per-chunk 10000 --outer-loops 1 --device cuda --output-root outputs/holo_experiments --label signal_window_probe --selection-metric image_error --region-mask-enabled --image-loss-mode signal_window
```

## Success Criteria

The first implementation phase is successful if:

- `mask_summary.png` clearly shows edge, signal, flat, dark, and relaxed regions for all 9 channels.
- `region_mask_report.csv` makes region coverage auditable.
- Default optimization commands produce the same output structure and behavior as before.
- All unit tests pass.

The second implementation phase is successful if:

- `signal_window` loss terms are visible in `loss_terms.csv` and `loss_terms.png`.
- On the existing grayscale test image, `summary.png` shows less visually distracting speckle in flat and dark regions while preserving main contours.
- `diagnostics.csv` and `metrics.json` do not show severe efficiency collapse.
- The approach remains compatible with the `phdx` / `phdy` coupled phase model.

The expected outcome is improved visual quality and better failure diagnosis under physical constraints. It is not expected to reach the per-channel independent-phase `1e-4` MSE upper bound.

## Risks and Mitigations

### Risk: Masks Select the Wrong Regions

If edge or signal masks are too broad, the method becomes close to global MSE again. If they are too narrow, important structure may be ignored.

Mitigation: make mask preview the first phase, export `mask_summary.png`, and use per-channel coverage reports.

### Risk: Dark Penalty Suppresses Useful Energy

Aggressive dark leakage penalties can reduce diffraction efficiency or make bright structure weaker.

Mitigation: keep `dark_weight` modest, default `image_loss_mode` to `global`, and compare `eta_curve.png` and `metrics.csv`.

### Risk: Flat Lowpass Loss Over-Smooths Details

If `flat_mask` covers details, the optimizer may blur important features.

Mitigation: edge priority is higher than flat priority, and edge dilation should protect contours.

### Risk: Too Many CLI Parameters

The design adds many knobs.

Mitigation: provide conservative defaults and document only two recommended command presets: preview and signal-window probe.

## Rollout Plan

Phase 1: mask generation and preview only.

- Add config and mask generator.
- Add `mask_preview` command.
- Add export of `mask_summary.png` and `region_mask_report.csv`.
- Do not change optimizer behavior.

Phase 2: signal-window loss integration.

- Add region-aware loss helpers.
- Add `image_loss_mode`.
- Record new loss terms.
- Run small smoke tests and one CUDA probe.

Phase 3: tuning and documentation.

- Compare `global`, `signal_window`, and `hybrid` modes.
- Update `README.md`, `AGENT.MD`, and `docs/ROADMAP.md` after implementation.
- Keep experiment outputs local and do not commit `inputs/` or `outputs/`.

## Open Decision

The main decision before implementation is whether Phase 1 should include a standalone `holo_opt.mask_preview` command, or only export masks during normal optimization runs.

Recommendation: add the standalone preview command. It lets the user validate segmentation quality without spending CUDA time on a full optimization run.
