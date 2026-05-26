# Holography Target Region Masks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add automatic target-derived region masks and then use them for signal-window holography loss while preserving the current `phdx` / `phdy` coupled physical model.

**Architecture:** Implement mask generation as a small independent module that operates on the final `(channels, height, width)` target stack. Wire masks into preview/export first, then pass them into `compute_loss_terms()` for opt-in `global`, `signal_window`, or `hybrid` image loss modes. Defaults preserve existing behavior.

**Tech Stack:** Python 3, NumPy, PyTorch, Matplotlib, unittest, existing `holo_opt` CLI/export/runner patterns.

---

## Scope and Sequence

This plan implements the design in `docs/superpowers/specs/2026-05-26-holography-target-region-masks-design.md`.

The implementation is intentionally staged:

1. Add configs and deterministic region mask generation.
2. Add export/preview so masks can be visually audited without training.
3. Integrate masks into normal experiment exports while keeping optimizer behavior unchanged by default.
4. Add signal-window loss as an explicit opt-in mode.
5. Update docs and run verification.

Do not stage or commit existing unrelated local changes. Current known unrelated changes may exist in:

```text
holo_opt/line_targets.py
tests/test_export.py
tests/test_targets.py
```

Read them before editing if a task touches those files.

## File Structure

Create:

```text
holo_opt/region_masks.py
holo_opt/mask_preview.py
tests/test_region_masks.py
```

Modify:

```text
holo_opt/config.py
holo_opt/cli.py
holo_opt/field.py
holo_opt/runner.py
holo_opt/export.py
tests/test_config.py
tests/test_cli.py
tests/test_field.py
tests/test_export.py
tests/test_runner.py
README.md
AGENT.MD
docs/ROADMAP.md
```

Keep `holo_opt/region_masks.py` focused on target partitioning. Keep `holo_opt/field.py` focused on differentiable loss. Keep export plotting in `holo_opt/export.py`.

## Task 1: Add Region Mask Config

**Files:**
- Modify: `holo_opt/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/test_config.py`:

```python
from holo_opt.config import RegionMaskConfig, SignalWindowLossConfig


def test_validate_config_accepts_region_mask_and_signal_window_defaults(self):
    config = ExperimentConfig()

    validate_config(config)

    self.assertFalse(config.region_mask.enabled)
    self.assertEqual(config.signal_window.image_loss_mode, "global")


def test_validate_config_rejects_invalid_region_mask_values(self):
    cases = [
        RegionMaskConfig(signal_threshold=-0.1),
        RegionMaskConfig(signal_threshold=1.1),
        RegionMaskConfig(dark_threshold=-0.1),
        RegionMaskConfig(dark_threshold=1.1),
        RegionMaskConfig(edge_quantile=0.0),
        RegionMaskConfig(edge_quantile=1.0),
        RegionMaskConfig(edge_dilation=-1),
        RegionMaskConfig(flat_gradient_quantile=0.0),
        RegionMaskConfig(flat_gradient_quantile=1.0),
        RegionMaskConfig(min_region_fraction=-0.1),
    ]
    for region_mask in cases:
        with self.subTest(region_mask=region_mask):
            config = ExperimentConfig(region_mask=region_mask)
            with self.assertRaisesRegex(ValueError, "region mask"):
                validate_config(config)


def test_validate_config_rejects_invalid_signal_window_values(self):
    cases = [
        SignalWindowLossConfig(image_loss_mode="invalid"),
        SignalWindowLossConfig(edge_weight=-1.0),
        SignalWindowLossConfig(signal_weight=-1.0),
        SignalWindowLossConfig(flat_weight=-1.0),
        SignalWindowLossConfig(relaxed_weight=-1.0),
        SignalWindowLossConfig(dark_weight=-1.0),
        SignalWindowLossConfig(dark_limit=-0.1),
        SignalWindowLossConfig(dark_limit=1.1),
        SignalWindowLossConfig(lowpass_sigma=0.0),
        SignalWindowLossConfig(signal_window_weight=-1.0),
    ]
    for signal_window in cases:
        with self.subTest(signal_window=signal_window):
            config = ExperimentConfig(signal_window=signal_window)
            with self.assertRaisesRegex(ValueError, "signal window"):
                validate_config(config)
```

- [ ] **Step 2: Run config tests to verify failure**

Run:

```powershell
py -m unittest tests.test_config -q
```

Expected: fail because `RegionMaskConfig`, `SignalWindowLossConfig`, `ExperimentConfig.region_mask`, and `ExperimentConfig.signal_window` do not exist.

- [ ] **Step 3: Implement config dataclasses and validation**

Add to `holo_opt/config.py` near `LossConfig`:

```python
@dataclass
class RegionMaskConfig:
    enabled: bool = False
    signal_threshold: float = 0.04
    dark_threshold: float = 0.02
    edge_quantile: float = 0.85
    edge_dilation: int = 2
    flat_gradient_quantile: float = 0.35
    min_region_fraction: float = 0.002


@dataclass
class SignalWindowLossConfig:
    image_loss_mode: str = "global"
    signal_window_weight: float = 1.0
    edge_weight: float = 2.0
    signal_weight: float = 1.0
    flat_weight: float = 0.15
    relaxed_weight: float = 0.03
    dark_weight: float = 0.2
    dark_limit: float = 0.03
    lowpass_sigma: float = 1.0
```

Add fields to `ExperimentConfig`:

```python
region_mask: RegionMaskConfig = field(default_factory=RegionMaskConfig)
signal_window: SignalWindowLossConfig = field(default_factory=SignalWindowLossConfig)
```

Add validation helpers near existing helpers:

```python
def _is_unit_interval(value: object, *, include_zero: bool = True, include_one: bool = True) -> bool:
    if not isinstance(value, Real) or not isfinite(value):
        return False
    lower_ok = value >= 0.0 if include_zero else value > 0.0
    upper_ok = value <= 1.0 if include_one else value < 1.0
    return bool(lower_ok and upper_ok)
```

Add inside `validate_config()` after loss validation:

```python
    region = config.region_mask
    if not isinstance(region.enabled, bool):
        raise ValueError("region mask enabled must be boolean")
    if not _is_unit_interval(region.signal_threshold):
        raise ValueError("region mask signal_threshold must be in [0, 1]")
    if not _is_unit_interval(region.dark_threshold):
        raise ValueError("region mask dark_threshold must be in [0, 1]")
    if not _is_unit_interval(region.edge_quantile, include_zero=False, include_one=False):
        raise ValueError("region mask edge_quantile must be in (0, 1)")
    if type(region.edge_dilation) is not int or region.edge_dilation < 0:
        raise ValueError("region mask edge_dilation must be a nonnegative integer")
    if not _is_unit_interval(region.flat_gradient_quantile, include_zero=False, include_one=False):
        raise ValueError("region mask flat_gradient_quantile must be in (0, 1)")
    if not _is_unit_interval(region.min_region_fraction):
        raise ValueError("region mask min_region_fraction must be in [0, 1]")

    signal_window = config.signal_window
    if signal_window.image_loss_mode not in {"global", "signal_window", "hybrid"}:
        raise ValueError("signal window image_loss_mode must be global, signal_window, or hybrid")
    signal_window_values = [
        signal_window.signal_window_weight,
        signal_window.edge_weight,
        signal_window.signal_weight,
        signal_window.flat_weight,
        signal_window.relaxed_weight,
        signal_window.dark_weight,
    ]
    if not _all_nonnegative(signal_window_values):
        raise ValueError("signal window weights must be nonnegative and finite")
    if not _is_unit_interval(signal_window.dark_limit):
        raise ValueError("signal window dark_limit must be in [0, 1]")
    if not _is_positive_number(signal_window.lowpass_sigma):
        raise ValueError("signal window lowpass_sigma must be positive")
```

- [ ] **Step 4: Run config tests to verify pass**

Run:

```powershell
py -m unittest tests.test_config -q
```

Expected: pass.

## Task 2: Add CLI Arguments

**Files:**
- Modify: `holo_opt/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/test_cli.py`:

```python
def test_parser_exposes_region_mask_options(self):
    args = build_parser().parse_args([
        "--region-mask-enabled",
        "--region-signal-threshold", "0.08",
        "--region-dark-threshold", "0.01",
        "--region-edge-quantile", "0.9",
        "--region-edge-dilation", "3",
        "--region-flat-gradient-quantile", "0.4",
        "--region-min-fraction", "0.01",
    ])
    config = config_from_args(args)

    self.assertTrue(config.region_mask.enabled)
    self.assertEqual(config.region_mask.signal_threshold, 0.08)
    self.assertEqual(config.region_mask.dark_threshold, 0.01)
    self.assertEqual(config.region_mask.edge_quantile, 0.9)
    self.assertEqual(config.region_mask.edge_dilation, 3)
    self.assertEqual(config.region_mask.flat_gradient_quantile, 0.4)
    self.assertEqual(config.region_mask.min_region_fraction, 0.01)


def test_parser_exposes_signal_window_options(self):
    args = build_parser().parse_args([
        "--image-loss-mode", "signal_window",
        "--signal-window-weight", "0.7",
        "--edge-weight", "2.5",
        "--signal-weight", "1.2",
        "--flat-weight", "0.2",
        "--relaxed-weight", "0.04",
        "--dark-weight", "0.3",
        "--dark-limit", "0.05",
        "--lowpass-sigma", "1.5",
    ])
    config = config_from_args(args)

    self.assertEqual(config.signal_window.image_loss_mode, "signal_window")
    self.assertEqual(config.signal_window.signal_window_weight, 0.7)
    self.assertEqual(config.signal_window.edge_weight, 2.5)
    self.assertEqual(config.signal_window.signal_weight, 1.2)
    self.assertEqual(config.signal_window.flat_weight, 0.2)
    self.assertEqual(config.signal_window.relaxed_weight, 0.04)
    self.assertEqual(config.signal_window.dark_weight, 0.3)
    self.assertEqual(config.signal_window.dark_limit, 0.05)
    self.assertEqual(config.signal_window.lowpass_sigma, 1.5)
```

- [ ] **Step 2: Run CLI tests to verify failure**

Run:

```powershell
py -m unittest tests.test_cli -q
```

Expected: fail because parser options are missing.

- [ ] **Step 3: Implement parser and config assembly**

Modify imports in `holo_opt/cli.py`:

```python
    RegionMaskConfig,
    SignalWindowLossConfig,
```

Add parser args:

```python
    parser.add_argument("--region-mask-enabled", action="store_true", default=False)
    parser.add_argument("--region-signal-threshold", type=float, default=0.04)
    parser.add_argument("--region-dark-threshold", type=float, default=0.02)
    parser.add_argument("--region-edge-quantile", type=float, default=0.85)
    parser.add_argument("--region-edge-dilation", type=int, default=2)
    parser.add_argument("--region-flat-gradient-quantile", type=float, default=0.35)
    parser.add_argument("--region-min-fraction", type=float, default=0.002)
    parser.add_argument("--image-loss-mode", choices=["global", "signal_window", "hybrid"], default="global")
    parser.add_argument("--signal-window-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=2.0)
    parser.add_argument("--signal-weight", type=float, default=1.0)
    parser.add_argument("--flat-weight", type=float, default=0.15)
    parser.add_argument("--relaxed-weight", type=float, default=0.03)
    parser.add_argument("--dark-weight", type=float, default=0.2)
    parser.add_argument("--dark-limit", type=float, default=0.03)
    parser.add_argument("--lowpass-sigma", type=float, default=1.0)
```

Add to `ExperimentConfig(...)` in `config_from_args()`:

```python
        region_mask=RegionMaskConfig(
            enabled=args.region_mask_enabled,
            signal_threshold=args.region_signal_threshold,
            dark_threshold=args.region_dark_threshold,
            edge_quantile=args.region_edge_quantile,
            edge_dilation=args.region_edge_dilation,
            flat_gradient_quantile=args.region_flat_gradient_quantile,
            min_region_fraction=args.region_min_fraction,
        ),
        signal_window=SignalWindowLossConfig(
            image_loss_mode=args.image_loss_mode,
            signal_window_weight=args.signal_window_weight,
            edge_weight=args.edge_weight,
            signal_weight=args.signal_weight,
            flat_weight=args.flat_weight,
            relaxed_weight=args.relaxed_weight,
            dark_weight=args.dark_weight,
            dark_limit=args.dark_limit,
            lowpass_sigma=args.lowpass_sigma,
        ),
```

- [ ] **Step 4: Run CLI tests to verify pass**

Run:

```powershell
py -m unittest tests.test_cli -q
```

Expected: pass.

## Task 3: Implement Region Mask Generation

**Files:**
- Create: `holo_opt/region_masks.py`
- Test: `tests/test_region_masks.py`

- [ ] **Step 1: Write failing region mask tests**

Create `tests/test_region_masks.py`:

```python
import unittest

import numpy as np

from holo_opt.config import RegionMaskConfig
from holo_opt.region_masks import generate_region_masks


class RegionMasksTest(unittest.TestCase):
    def test_generate_region_masks_returns_expected_shapes_and_rows(self):
        targets = np.zeros((2, 8, 8), dtype=np.float32)
        targets[0, 2:6, 2:6] = 0.5
        targets[1, 1:7, 1:7] = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)

        masks = generate_region_masks(targets, RegionMaskConfig(enabled=True))

        for values in (masks.edge, masks.signal, masks.flat, masks.dark, masks.relaxed):
            self.assertEqual(values.shape, targets.shape)
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)
        self.assertEqual(len(masks.report_rows), 2)
        self.assertIn("edge_fraction", masks.report_rows[0])
        self.assertIn("edge_threshold", masks.report_rows[0])

    def test_dark_pixels_do_not_overlap_signal_edge_or_flat_masks(self):
        targets = np.zeros((1, 8, 8), dtype=np.float32)
        targets[0, 3:5, 3:5] = 1.0

        masks = generate_region_masks(
            targets,
            RegionMaskConfig(enabled=True, dark_threshold=0.1, signal_threshold=0.2, edge_dilation=1),
        )

        dark = masks.dark.astype(bool)
        self.assertFalse(np.any(masks.edge.astype(bool) & dark))
        self.assertFalse(np.any(masks.signal.astype(bool) & dark))
        self.assertFalse(np.any(masks.flat.astype(bool) & dark))

    def test_edge_dilation_increases_or_preserves_edge_fraction(self):
        targets = np.zeros((1, 12, 12), dtype=np.float32)
        targets[0, :, 6:] = 1.0

        no_dilation = generate_region_masks(targets, RegionMaskConfig(enabled=True, edge_dilation=0))
        dilation = generate_region_masks(targets, RegionMaskConfig(enabled=True, edge_dilation=2))

        self.assertGreaterEqual(float(dilation.edge.sum()), float(no_dilation.edge.sum()))

    def test_generate_region_masks_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "targets must have shape"):
            generate_region_masks(np.zeros((8, 8), dtype=np.float32), RegionMaskConfig(enabled=True))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run region mask tests to verify failure**

Run:

```powershell
py -m unittest tests.test_region_masks -q
```

Expected: fail because `holo_opt.region_masks` does not exist.

- [ ] **Step 3: Implement region mask module**

Create `holo_opt/region_masks.py`:

```python
"""Target-derived region masks for signal-window holography losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from holo_opt.config import RegionMaskConfig


@dataclass
class RegionMasks:
    edge: np.ndarray
    signal: np.ndarray
    flat: np.ndarray
    dark: np.ndarray
    relaxed: np.ndarray
    report_rows: list[dict[str, float]]


def _validate_targets(targets: np.ndarray) -> np.ndarray:
    arr = np.asarray(targets, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("targets must have shape (channels, height, width)")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0 or arr.shape[2] <= 0:
        raise ValueError("targets must have positive channel, height, and width dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("targets contain NaN or inf")
    return arr


def _normalize_channel(target: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    max_value = float(np.max(target))
    if max_value <= epsilon:
        return np.zeros_like(target, dtype=np.float32)
    return (target / (max_value + epsilon)).astype(np.float32)


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(values.astype(np.float32))
    return np.sqrt(grad_x * grad_x + grad_y * grad_y).astype(np.float32)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for y_offset in range(size):
        for x_offset in range(size):
            output |= padded[y_offset:y_offset + mask.shape[0], x_offset:x_offset + mask.shape[1]]
    return output


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(mask.astype(np.float32)))


def generate_region_masks(targets: np.ndarray, config: RegionMaskConfig) -> RegionMasks:
    arr = _validate_targets(targets)
    edge_masks = []
    signal_masks = []
    flat_masks = []
    dark_masks = []
    relaxed_masks = []
    rows: list[dict[str, float]] = []

    for channel_index, target in enumerate(arr):
        normalized = _normalize_channel(target)
        gradient = _gradient_magnitude(normalized)
        dark = normalized <= float(config.dark_threshold)
        raw_signal = normalized > float(config.signal_threshold)

        nonzero_gradient = gradient[gradient > 0.0]
        edge_threshold = (
            float(np.quantile(nonzero_gradient, config.edge_quantile))
            if nonzero_gradient.size
            else float("inf")
        )
        raw_edge = gradient >= edge_threshold if np.isfinite(edge_threshold) else np.zeros_like(dark, dtype=bool)
        edge = _dilate(raw_edge, int(config.edge_dilation)) & ~dark

        signal_without_edge = raw_signal & ~edge & ~dark
        flat_candidates = gradient <= (
            float(np.quantile(nonzero_gradient, config.flat_gradient_quantile))
            if nonzero_gradient.size
            else 0.0
        )
        flat = signal_without_edge & flat_candidates
        signal = signal_without_edge & ~flat
        relaxed = ~(edge | signal | flat | dark)

        edge_masks.append(edge.astype(np.float32))
        signal_masks.append(signal.astype(np.float32))
        flat_masks.append(flat.astype(np.float32))
        dark_masks.append(dark.astype(np.float32))
        relaxed_masks.append(relaxed.astype(np.float32))
        rows.append({
            "channel": float(channel_index + 1),
            "edge_fraction": _fraction(edge),
            "signal_fraction": _fraction(signal),
            "flat_fraction": _fraction(flat),
            "dark_fraction": _fraction(dark),
            "relaxed_fraction": _fraction(relaxed),
            "target_mean": float(np.mean(target)),
            "target_max": float(np.max(target)),
            "edge_threshold": edge_threshold if np.isfinite(edge_threshold) else 0.0,
        })

    return RegionMasks(
        edge=np.stack(edge_masks, axis=0).astype(np.float32),
        signal=np.stack(signal_masks, axis=0).astype(np.float32),
        flat=np.stack(flat_masks, axis=0).astype(np.float32),
        dark=np.stack(dark_masks, axis=0).astype(np.float32),
        relaxed=np.stack(relaxed_masks, axis=0).astype(np.float32),
        report_rows=rows,
    )
```

- [ ] **Step 4: Run region mask tests to verify pass**

Run:

```powershell
py -m unittest tests.test_region_masks -q
```

Expected: pass.

## Task 4: Add Mask Export Helpers

**Files:**
- Modify: `holo_opt/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Read user changes in export tests**

Run:

```powershell
git diff -- tests/test_export.py
```

Expected: inspect any existing user edits before adding new tests.

- [ ] **Step 2: Write failing export tests**

Add imports in `tests/test_export.py`:

```python
from holo_opt.region_masks import RegionMasks
```

Add a test:

```python
def test_export_results_writes_region_mask_artifacts(self):
    with tempfile.TemporaryDirectory(dir=Path.cwd() / "outputs" / "test_export") as tmp:
        config = ExperimentConfig(size=4, output_root=tmp, label="masks")
        targets = np.zeros((9, 4, 4), dtype=np.float32)
        targets[:, 1:3, 1:3] = 1.0
        intensities = targets.copy()
        phases = np.zeros((4, 4), dtype=np.float32)
        metrics = {
            "rows": [
                {"channel": index + 1, "mse": 0.0, "eta": 1.0, "gray_level_error": 0.0, "gray_means": [0.0] * 16}
                for index in range(9)
            ],
            "summary": {
                "score": 0.0,
                "image_error": 0.0,
                "gray_level_error": 0.0,
                "efficiency_balance_penalty": 0.0,
                "mean_eta": 1.0,
            },
        }
        masks = RegionMasks(
            edge=np.ones_like(targets),
            signal=np.zeros_like(targets),
            flat=np.zeros_like(targets),
            dark=np.zeros_like(targets),
            relaxed=np.zeros_like(targets),
            report_rows=[{
                "channel": float(index + 1),
                "edge_fraction": 1.0,
                "signal_fraction": 0.0,
                "flat_fraction": 0.0,
                "dark_fraction": 0.0,
                "relaxed_fraction": 0.0,
                "target_mean": 0.25,
                "target_max": 1.0,
                "edge_threshold": 0.1,
            } for index in range(9)],
        )

        run_dir = export_results(
            config,
            targets,
            intensities,
            phases,
            phases,
            [1.0],
            [[1.0] * 9],
            [[1.0] * 9],
            metrics,
            region_masks=masks,
        )

        self.assertTrue((run_dir / "mask_summary.png").exists())
        self.assertGreater((run_dir / "mask_summary.png").stat().st_size, 0)
        with (run_dir / "region_mask_report.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0][0], "channel")
        self.assertEqual(len(rows), 10)
```

- [ ] **Step 3: Run export tests to verify failure**

Run:

```powershell
py -m unittest tests.test_export -q
```

Expected: fail because `export_results()` has no `region_masks` parameter or mask plot.

- [ ] **Step 4: Implement export support**

Modify imports in `holo_opt/export.py`:

```python
from holo_opt.region_masks import RegionMasks
```

Add parameter to `export_results()`:

```python
    region_masks: RegionMasks | None = None,
```

Add after grayscale artifacts export:

```python
    if region_masks is not None:
        _write_rows_csv(run_dir / "region_mask_report.csv", region_masks.report_rows)
        _plot_region_masks(run_dir / "mask_summary.png", targets, region_masks)
```

Add helper:

```python
def _plot_region_masks(path: Path, targets: np.ndarray, masks: RegionMasks) -> None:
    panels = (
        ("target", np.asarray(targets, dtype=np.float32)),
        ("edge", masks.edge),
        ("signal", masks.signal),
        ("flat", masks.flat),
        ("dark", masks.dark),
        ("relaxed", masks.relaxed),
    )
    channels = np.asarray(targets).shape[0]
    fig, axes = plt.subplots(len(panels), channels, figsize=(max(9.0, channels * 1.3), 8.0), squeeze=False)
    try:
        for row_index, (name, values) in enumerate(panels):
            for channel in range(channels):
                axes[row_index, channel].imshow(values[channel], cmap="gray", vmin=0.0, vmax=1.0)
                if row_index == 0:
                    axes[row_index, channel].set_title(f"ch {channel + 1}")
                if channel == 0:
                    axes[row_index, channel].set_ylabel(name)
                axes[row_index, channel].set_xticks([])
                axes[row_index, channel].set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
```

- [ ] **Step 5: Run export tests to verify pass**

Run:

```powershell
py -m unittest tests.test_export -q
```

Expected: pass.

## Task 5: Add Mask Preview CLI

**Files:**
- Create: `holo_opt/mask_preview.py`
- Test: add coverage in `tests/test_region_masks.py` or create `tests/test_mask_preview.py`

- [ ] **Step 1: Write failing mask preview test**

Create `tests/test_mask_preview.py`:

```python
import shutil
import uuid
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from holo_opt.mask_preview import main


class MaskPreviewTest(unittest.TestCase):
    def test_mask_preview_writes_summary_and_report(self):
        output_root = Path.cwd() / "outputs" / "test_mask_preview" / uuid.uuid4().hex
        output_root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))
        image_path = output_root / "input.png"
        image = Image.new("RGB", (24, 24), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 20, 20), fill=(180, 180, 180))
        draw.line((4, 12, 20, 12), fill=(255, 255, 255), width=2)
        image.save(image_path)

        exit_code = main([
            "--target-mode", "grayscale",
            "--target-path", str(image_path),
            "--size", "8",
            "--output-dir", str(output_root / "preview"),
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_root / "preview" / "mask_summary.png").exists())
        self.assertTrue((output_root / "preview" / "region_mask_report.csv").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run preview test to verify failure**

Run:

```powershell
py -m unittest tests.test_mask_preview -q
```

Expected: fail because `holo_opt.mask_preview` does not exist.

- [ ] **Step 3: Implement preview command**

Create `holo_opt/mask_preview.py`:

```python
"""Preview target-derived region masks without running optimization."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from holo_opt.cli import build_parser, config_from_args
from holo_opt.config import ExperimentConfig, validate_config
from holo_opt.export import _plot_region_masks, _write_rows_csv
from holo_opt.region_masks import generate_region_masks
from holo_opt.runner import load_targets_for_config


def _build_preview_parser() -> argparse.ArgumentParser:
    base = build_parser()
    parser = argparse.ArgumentParser(description="Preview target-derived region masks.")
    for action in base._actions:
        if action.dest in {"help", "epochs_per_chunk", "outer_loops", "lr", "device", "output_root", "label"}:
            continue
        parser._add_action(action)
    parser.add_argument("--output-dir", default="outputs/mask_preview")
    return parser


def _preview_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    namespace = argparse.Namespace(**vars(args))
    namespace.epochs_per_chunk = 1
    namespace.outer_loops = 1
    namespace.lr = 5e-4
    namespace.device = "cpu"
    namespace.output_root = "outputs/holo_experiments"
    namespace.label = "mask_preview"
    namespace.region_mask_enabled = True
    config = config_from_args(namespace)
    config.region_mask.enabled = True
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_preview_parser().parse_args(argv)
    config = _preview_config_from_args(args)
    validate_config(config)
    targets = load_targets_for_config(config)
    masks = generate_region_masks(targets, config.region_mask)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(output_dir / "region_mask_report.csv", masks.report_rows)
    _plot_region_masks(output_dir / "mask_summary.png", np.asarray(targets, dtype=np.float32), masks)
    print(f"Saved mask preview to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run preview test to verify pass**

Run:

```powershell
py -m unittest tests.test_mask_preview -q
```

Expected: pass.

## Task 6: Integrate Region Masks Into Runner Exports Without Changing Loss

**Files:**
- Modify: `holo_opt/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing runner export test**

Add to `tests/test_runner.py`:

```python
def test_run_experiment_exports_region_masks_when_enabled(self):
    output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
    output_root.mkdir(parents=True)
    self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))

    config = ExperimentConfig(
        size=8,
        epochs_per_chunk=1,
        outer_loops=1,
        output_root=str(output_root),
        label="region_masks",
        device="cpu",
    )
    config.region_mask.enabled = True

    result = run_experiment(config)

    self.assertTrue((result.run_dir / "mask_summary.png").exists())
    self.assertTrue((result.run_dir / "region_mask_report.csv").exists())
```

- [ ] **Step 2: Run runner test to verify failure**

Run:

```powershell
py -m unittest tests.test_runner -q
```

Expected: fail because runner does not generate/export masks.

- [ ] **Step 3: Implement runner mask generation and export**

Modify imports in `holo_opt/runner.py`:

```python
from holo_opt.region_masks import RegionMasks, generate_region_masks
```

After `targets_np = target_bundle.targets`, add:

```python
    region_masks_np: RegionMasks | None = None
    if config.region_mask.enabled or config.signal_window.image_loss_mode in {"signal_window", "hybrid"}:
        region_masks_np = generate_region_masks(targets_np, config.region_mask)
```

Pass to `export_results()`:

```python
        region_masks=region_masks_np,
```

- [ ] **Step 4: Run runner test to verify pass**

Run:

```powershell
py -m unittest tests.test_runner -q
```

Expected: pass.

## Task 7: Add Signal-Window Loss Helpers

**Files:**
- Modify: `holo_opt/field.py`
- Test: `tests/test_field.py`

- [ ] **Step 1: Write failing field tests**

Add to `tests/test_field.py`:

```python
from holo_opt.config import SignalWindowLossConfig
from holo_opt.field import compute_signal_window_loss_terms, masked_mean


def test_masked_mean_uses_only_masked_pixels(self):
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    self.assertAlmostEqual(float(masked_mean(values, mask).item()), 2.0)


def test_signal_window_loss_terms_penalize_bad_signal_region_more_than_good(self):
    targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    good = targets.clone()
    bad = 1.0 - targets
    masks = {
        "edge": torch.zeros_like(targets),
        "signal": targets.clone(),
        "flat": torch.zeros_like(targets),
        "dark": 1.0 - targets,
        "relaxed": torch.zeros_like(targets),
    }
    config = SignalWindowLossConfig(image_loss_mode="signal_window")

    good_terms = compute_signal_window_loss_terms(good, targets, good, masks, config)
    bad_terms = compute_signal_window_loss_terms(bad, targets, bad, masks, config)

    self.assertLess(float(good_terms["signal_window"].item()), float(bad_terms["signal_window"].item()))


def test_signal_window_dark_leakage_respects_dark_limit(self):
    targets = torch.zeros((1, 2, 2), dtype=torch.float32)
    masks = {
        "edge": torch.zeros_like(targets),
        "signal": torch.zeros_like(targets),
        "flat": torch.zeros_like(targets),
        "dark": torch.ones_like(targets),
        "relaxed": torch.zeros_like(targets),
    }
    config = SignalWindowLossConfig(image_loss_mode="signal_window", dark_limit=0.2)
    low = torch.full_like(targets, 0.1)
    high = torch.full_like(targets, 0.5)

    low_terms = compute_signal_window_loss_terms(low, targets, low, masks, config)
    high_terms = compute_signal_window_loss_terms(high, targets, high, masks, config)

    self.assertEqual(float(low_terms["dark_leakage"].item()), 0.0)
    self.assertGreater(float(high_terms["dark_leakage"].item()), 0.0)
```

- [ ] **Step 2: Run field tests to verify failure**

Run:

```powershell
py -m unittest tests.test_field -q
```

Expected: fail because helpers do not exist.

- [ ] **Step 3: Implement field helpers**

Modify imports in `holo_opt/field.py`:

```python
import torch.nn.functional as F

from holo_opt.config import SignalWindowLossConfig
```

Add helpers:

```python
def masked_mean(values: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    numerator = torch.sum(values * mask)
    denominator = torch.sum(mask).clamp_min(epsilon)
    return numerator / denominator


def _box_blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(float(sigma) * 2.0)))
    kernel_size = 2 * radius + 1
    channels = values.shape[0]
    kernel = values.new_ones((channels, 1, kernel_size, kernel_size)) / float(kernel_size * kernel_size)
    padded = F.pad(values.unsqueeze(0), (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels).squeeze(0)


def compute_signal_window_loss_terms(
    energy_matched_intensity: torch.Tensor,
    targets: torch.Tensor,
    normalized_intensity: torch.Tensor,
    region_masks: dict[str, torch.Tensor],
    config: SignalWindowLossConfig,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    squared_error = (energy_matched_intensity - targets) ** 2
    blurred_reconstruction = _box_blur(energy_matched_intensity, config.lowpass_sigma)
    blurred_targets = _box_blur(targets, config.lowpass_sigma)
    lowpass_error = (blurred_reconstruction - blurred_targets) ** 2
    dark_error = torch.relu(normalized_intensity - float(config.dark_limit)) ** 2

    edge_mse = masked_mean(squared_error, region_masks["edge"], epsilon)
    signal_mse = masked_mean(squared_error, region_masks["signal"], epsilon)
    flat_lowpass_mse = masked_mean(lowpass_error, region_masks["flat"], epsilon)
    relaxed_lowpass_mse = masked_mean(lowpass_error, region_masks["relaxed"], epsilon)
    dark_leakage = masked_mean(dark_error, region_masks["dark"], epsilon)
    signal_window = (
        float(config.edge_weight) * edge_mse
        + float(config.signal_weight) * signal_mse
        + float(config.flat_weight) * flat_lowpass_mse
        + float(config.relaxed_weight) * relaxed_lowpass_mse
        + float(config.dark_weight) * dark_leakage
    )
    return {
        "signal_window": signal_window,
        "edge_mse": edge_mse,
        "signal_mse": signal_mse,
        "flat_lowpass_mse": flat_lowpass_mse,
        "relaxed_lowpass_mse": relaxed_lowpass_mse,
        "dark_leakage": dark_leakage,
    }
```

- [ ] **Step 4: Run field tests to verify pass**

Run:

```powershell
py -m unittest tests.test_field -q
```

Expected: pass.

## Task 8: Wire Signal-Window Loss Into compute_loss_terms

**Files:**
- Modify: `holo_opt/field.py`
- Modify: `holo_opt/runner.py`
- Test: `tests/test_field.py`, `tests/test_runner.py`

- [ ] **Step 1: Write failing integration tests**

Add to `tests/test_field.py`:

```python
def test_compute_loss_terms_signal_window_mode_replaces_image_mse_in_total(self):
    phdx = torch.zeros((2, 2), dtype=torch.float32)
    phdy = torch.zeros((2, 2), dtype=torch.float32)
    pair_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    weights = torch.ones(1, dtype=torch.float32)
    intensities = torch.ones((1, 2, 2), dtype=torch.float32)
    masks = {
        "edge": torch.zeros_like(targets),
        "signal": targets.clone(),
        "flat": torch.zeros_like(targets),
        "dark": 1.0 - targets,
        "relaxed": torch.zeros_like(targets),
    }
    config = SignalWindowLossConfig(image_loss_mode="signal_window", signal_weight=1.0, edge_weight=0.0, dark_weight=0.0)

    with patch("holo_opt.field.compute_intensities", return_value=intensities):
        terms = compute_loss_terms(
            phdx,
            phdy,
            pair_mat,
            targets,
            weights,
            {"image_weight": 1.0},
            region_masks=masks,
            signal_window_config=config,
        )

    self.assertIn("signal_window", terms)
    self.assertAlmostEqual(float(terms["total"].item()), float(terms["signal_window"].item()))
```

Add to `tests/test_runner.py`:

```python
def test_run_experiment_records_signal_window_loss_terms(self):
    output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
    output_root.mkdir(parents=True)
    self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))

    config = ExperimentConfig(
        size=8,
        epochs_per_chunk=1,
        outer_loops=1,
        output_root=str(output_root),
        label="signal_window",
        device="cpu",
    )
    config.region_mask.enabled = True
    config.signal_window.image_loss_mode = "signal_window"

    result = run_experiment(config)

    with (result.run_dir / "loss_terms.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    self.assertIn("signal_window", header)
    self.assertIn("edge_mse", header)
```

- [ ] **Step 2: Run field and runner tests to verify failure**

Run:

```powershell
py -m unittest tests.test_field tests.test_runner -q
```

Expected: fail because `compute_loss_terms()` does not accept mask/config arguments and runner does not record new keys.

- [ ] **Step 3: Update compute_loss_terms signature and total loss logic**

Modify `compute_loss_terms()` signature in `holo_opt/field.py`:

```python
    region_masks: dict[str, torch.Tensor] | None = None,
    signal_window_config: SignalWindowLossConfig | None = None,
```

After `image_mse`:

```python
    signal_terms: dict[str, torch.Tensor] = {}
    image_component = image_mse
    if signal_window_config is not None and signal_window_config.image_loss_mode in {"signal_window", "hybrid"}:
        if region_masks is None:
            raise ValueError("region_masks are required for signal_window image loss mode")
        signal_terms = compute_signal_window_loss_terms(
            energy_matched_intensity,
            targets,
            normalized_intensity,
            region_masks,
            signal_window_config,
            epsilon=epsilon,
        )
        if signal_window_config.image_loss_mode == "signal_window":
            image_component = signal_terms["signal_window"]
        else:
            image_component = image_mse + float(signal_window_config.signal_window_weight) * signal_terms["signal_window"]
```

Change total image part:

```python
        float(loss_weights.get("image_weight", 1.0)) * image_component
```

Before return:

```python
    result = {
        "total": total,
        "image_mse": image_mse,
        "eta_balance": eta_balance,
        "gray_monotonic": gray_monotonic,
        "phase_smoothness": smoothness,
        "background": background,
    }
    result.update(signal_terms)
    return result
```

- [ ] **Step 4: Add runner conversion from NumPy masks to torch masks**

In `holo_opt/runner.py`, after targets tensor creation:

```python
    region_masks_torch: dict[str, torch.Tensor] | None = None
    if region_masks_np is not None:
        region_masks_torch = {
            "edge": torch.as_tensor(region_masks_np.edge, dtype=torch.float32, device=device),
            "signal": torch.as_tensor(region_masks_np.signal, dtype=torch.float32, device=device),
            "flat": torch.as_tensor(region_masks_np.flat, dtype=torch.float32, device=device),
            "dark": torch.as_tensor(region_masks_np.dark, dtype=torch.float32, device=device),
            "relaxed": torch.as_tensor(region_masks_np.relaxed, dtype=torch.float32, device=device),
        }
```

Update `loss_term_names`:

```python
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
```

Update `compute_loss_terms()` call:

```python
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
```

Update loss history append to be dynamic:

```python
            loss_terms_history.append({
                "step": float(len(losses)),
                **{name: float(term_values[index]) for index, name in enumerate(loss_term_names)},
            })
```

- [ ] **Step 5: Run field and runner tests to verify pass**

Run:

```powershell
py -m unittest tests.test_field tests.test_runner -q
```

Expected: pass.

## Task 9: Update Loss Plotting For New Terms

**Files:**
- Modify: `holo_opt/export.py`
- Test: `tests/test_export.py`

- [ ] **Step 1: Write failing export assertion for signal terms**

Extend the existing diagnostics/loss terms export test in `tests/test_export.py` so `loss_terms_history` includes:

```python
{
    "step": 1,
    "total": 2.0,
    "image_mse": 1.0,
    "eta_balance": 0.2,
    "gray_monotonic": 0.3,
    "phase_smoothness": 0.4,
    "background": 0.0,
    "signal_window": 0.5,
    "edge_mse": 0.1,
    "signal_mse": 0.2,
    "flat_lowpass_mse": 0.03,
    "relaxed_lowpass_mse": 0.01,
    "dark_leakage": 0.04,
}
```

Assert header contains:

```python
self.assertIn("signal_window", loss_terms_rows[0])
self.assertIn("edge_mse", loss_terms_rows[0])
```

- [ ] **Step 2: Run export tests to verify plot failure or missing plotted terms**

Run:

```powershell
py -m unittest tests.test_export -q
```

Expected: CSV may pass because `_write_rows_csv` is dynamic, but plotting currently ignores signal-window terms. Adjust test if needed to verify plot is non-empty and no exception is raised.

- [ ] **Step 3: Update `_plot_loss_terms()` term list**

In `holo_opt/export.py`, change:

```python
    terms = ("image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background")
```

to:

```python
    terms = (
        "image_mse",
        "signal_window",
        "edge_mse",
        "signal_mse",
        "flat_lowpass_mse",
        "relaxed_lowpass_mse",
        "dark_leakage",
        "eta_balance",
        "gray_monotonic",
        "phase_smoothness",
        "background",
    )
```

- [ ] **Step 4: Run export tests to verify pass**

Run:

```powershell
py -m unittest tests.test_export -q
```

Expected: pass.

## Task 10: Documentation Updates

**Files:**
- Modify: `README.md`
- Modify: `AGENT.MD`
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Update README quick-start text**

Add a section after grayscale input:

```markdown
## Target 区域分区与 signal-window 损失

如果重建图主体已经出现但平坦灰度区雪花噪声很重，可以先预览 target 自动分区：

```powershell
py -m holo_opt.mask_preview --target-mode grayscale --target-path inputs/lineart_sources/demo_preview.png --size 128 --output-dir outputs/mask_preview
```

优先检查：

- `mask_summary.png`
- `region_mask_report.csv`

确认 `edge / signal / flat / dark / relaxed` 分区合理后，再尝试 signal-window 优化：

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/demo_preview.png --size 128 --epochs-per-chunk 10000 --outer-loops 1 --device cuda --selection-metric image_error --region-mask-enabled --image-loss-mode signal_window
```

该模式保留 `phdx/phdy` 耦合物理模型，不追求每通道独立相位图的理论最低 MSE；目标是降低平坦区和暗背景的视觉噪声，同时保护主体边缘。
```

- [ ] **Step 2: Update AGENT.MD algorithm history**

Append a section:

```markdown
## 2026-05-26 target 区域分区与 signal-window 损失

- 改动动机：全图 MSE 会让 phase-only 或近似 phase-only 模型在大面积平坦灰度区域硬拟合高频误差，导致主体结构出现但平坦区仍有明显雪花噪声。
- 影响范围：新增 `holo_opt/region_masks.py` 和 `holo_opt/mask_preview.py`；`config.py`、`cli.py`、`runner.py`、`field.py`、`export.py` 接入区域 mask 和 opt-in 的 `signal_window` 图像损失。
- 用户可见变化：新增 `py -m holo_opt.mask_preview ...` 预览命令；正式运行可用 `--region-mask-enabled --image-loss-mode signal_window`；导出新增 `mask_summary.png` 和 `region_mask_report.csv`，启用 signal-window 时 `loss_terms.csv` 增加 `signal_window`、`edge_mse`、`signal_mse`、`flat_lowpass_mse`、`relaxed_lowpass_mse`、`dark_leakage`。
- 算法细节：分区在最终 channel target 栈上逐通道生成，优先级为 `edge > signal > flat > dark > relaxed`；边缘和主体区强拟合，平坦和 relaxed 区只做低频拟合，暗区只惩罚超过阈值的亮泄漏。
- 验证方式：需要运行 `py -m unittest discover -s tests -q`，并建议运行 mask preview 和小尺寸 CPU smoke；质量实验建议在 CUDA 上对比 `global`、`signal_window` 和 `hybrid`。
```

- [ ] **Step 3: Update ROADMAP**

In `docs/ROADMAP.md`, under grayscale quality / image adaptation, add:

```markdown
- target 区域分区和 signal-window 损失已成为下一步图像质量主线：先用 `mask_preview` 审查自动分区，再用 opt-in 的 `signal_window` 模式减少平坦区散斑，保持 `phdx/phdy` 耦合模型不变。
```

- [ ] **Step 4: Documentation self-check**

Run:

```powershell
rg -n "mask_preview|signal-window|signal_window|region_mask" README.md AGENT.MD docs\\ROADMAP.md
```

Expected: new docs mention preview, signal-window mode, and exported mask artifacts.

## Task 11: Full Verification

**Files:**
- No new file edits unless verification exposes a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
py -m unittest tests.test_config tests.test_cli tests.test_region_masks tests.test_mask_preview tests.test_field tests.test_export tests.test_runner -q
```

Expected: pass.

- [ ] **Step 2: Run full suite**

Run:

```powershell
py -m unittest discover -s tests -q
```

Expected: pass.

- [ ] **Step 3: Run mask preview smoke**

Run:

```powershell
py -m holo_opt.mask_preview --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 64 --output-dir outputs/mask_preview
```

Expected:

```text
outputs/mask_preview/mask_summary.png
outputs/mask_preview/region_mask_report.csv
```

exist and are non-empty.

- [ ] **Step 4: Run CPU optimization smoke**

Run:

```powershell
py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 1 --device cpu --output-root outputs/holo_experiments --label mask_smoke --region-mask-enabled --image-loss-mode signal_window
```

Expected: run completes, prints saved run directory, and that directory contains `mask_summary.png`, `region_mask_report.csv`, `loss_terms.csv`, and `summary.png`.

- [ ] **Step 5: Inspect git status**

Run:

```powershell
git status --short
```

Expected: only planned source, test, and documentation files are modified or added. Do not delete or stage unrelated user changes.

## Task 12: Optional CUDA Quality Probe

**Files:**
- No source edits expected.

- [ ] **Step 1: Run baseline global mode**

Run:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 128 --epochs-per-chunk 3000 --outer-loops 1 --device cuda --output-root outputs/holo_experiments --label global_probe --selection-metric image_error --image-loss-mode global
```

Expected: CUDA run completes.

- [ ] **Step 2: Run signal-window mode**

Run:

```powershell
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/test_input.png --size 128 --epochs-per-chunk 3000 --outer-loops 1 --device cuda --output-root outputs/holo_experiments --label signal_window_probe --selection-metric image_error --region-mask-enabled --image-loss-mode signal_window
```

Expected: CUDA run completes.

- [ ] **Step 3: Compare outputs**

Inspect:

```text
summary.png
stitched_comparison.png
mask_summary.png
loss_terms.png
diagnostics.csv
metrics.json
```

Expected: signal-window mode should reduce visually distracting flat/dark speckle or make failure modes clearer. Do not claim quality improvement if images or metrics do not support it.

## Self-Review Checklist

- Spec coverage: config, mask generation, preview, export, runner integration, signal-window loss, docs, and verification are covered.
- Placeholder scan: this plan intentionally contains no `TBD`, no empty "write tests" step, and no unspecified file names.
- Type consistency: `RegionMaskConfig`, `SignalWindowLossConfig`, `RegionMasks`, `generate_region_masks()`, `compute_signal_window_loss_terms()`, `region_masks`, and `signal_window_config` are named consistently across tasks.
- Backward compatibility: `image_loss_mode="global"` remains default and should keep existing behavior unless mask preview or signal-window options are explicitly enabled.
