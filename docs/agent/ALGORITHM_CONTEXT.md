# Algorithm Context

This file records the current algorithm structure, known failure modes, and the intended direction for future improvements.

## Current Model

The project uses an FFT-based proxy holography optimizer.

It does not currently solve the full physical metasurface design chain. It does not do RCWA/FDTD, unit-cell library lookup, GDS export, or material/fabrication modeling.

The core proxy variables are:

- `phdx`: x-direction phase proxy map.
- `phdy`: y-direction phase proxy map.

For each diffraction-channel pair `(m, n)`, the model computes:

```text
phase = m * phdx + n * phdy
field = exp(i * phase)
intensity = |FFT(field)|^2
```

The optimizer updates `phdx` and `phdy` so the simulated far-field intensity channels resemble the target channels.

## Default Channels

Default diffraction orders:

```text
(-2, 2), (-2, 1), (-2, 0),
(-3, 2), (-3, 1), (-3, 0),
(-4, 2), (-4, 1), (-4, 0)
```

These 9 channels are treated as a 3x3 grid when producing `stitched_comparison.png`.

## Important Physical Parameters

Current defaults:

- `lambda_nm = 532`
- `px_nm = 830`
- `py_nm = 830`
- guided mode enabled
- `neff = 2.05`
- `alpha_deg = -16.7`

These values are part of the current design context. Do not silently change them while doing unrelated work.

## Target Modes

- `standard`
  - Built-in 16-level gray-step diagnostic target.
  - Useful for checking grayscale behavior and optimization health.
- `grayscale`
  - Current closest match to the desired main workflow.
  - Converts one RGB image into grayscale targets split across 9 channels.
  - Current implementation still uses stronger grayscale shaping for compatibility experiments.
- `grayscale_direct`
  - Converts one RGB image to grayscale, applies only a global brightness cap, then splits it into 9 channel tiles.
  - Use this when the user wants a more direct image-to-target path without the extra grayscale shaping used by `grayscale`.
- `lineart`
  - Converts one RGB image into edge/line targets and repeats them across channels.
  - Useful for contour experiments, not the main final-image workflow.
- `mat`
  - Loads externally prepared 9-channel target stacks from `.mat`.
  - Compatibility path only; do not make it the main user workflow.

## Current Loss Terms

Implemented structured loss terms:

- `image_mse`
  - Per-channel normalized image mismatch.
- `eta_balance`
  - Penalizes poor useful-energy balance across channels.
- `gray_monotonic`
  - Penalizes inverted gray-level ordering.
- `phase_smoothness`
  - Penalizes high-frequency phase changes.
- `background`
  - Penalizes intensity in dark target regions when enabled.
- `local_uniformity`
  - Penalizes reconstruction pixels that deviate from their small local mean inside non-black target regions.
  - Uses lower weight near target edges so contour clarity is less likely to be over-smoothed.
- `high_frequency`
  - Penalizes Laplacian-style high-frequency energy inside non-black target regions.
  - Uses lower weight near target edges so the term focuses more on block-internal speckle than on outline suppression.

The total loss is a weighted sum controlled by `LossConfig` and CLI options.

## Metrics And Score

Evaluation includes:

- per-channel MSE
- useful energy ratio `eta`
- gray-level error
- efficiency balance penalty
- mean eta
- score
- standalone benchmark flat-region noise evaluation
  - implemented in `holo_opt/benchmark_eval.py`
  - uses per-channel normalization, stitched 3x3 comparison, low-gradient target masking, `target > 0.05`, and `9x9` local residual variance

Use metrics to diagnose failure, but do not let a scalar score replace visual inspection of `stitched_comparison.png`.

## Known Failure Modes

- Final reconstruction is blurry.
- Reconstruction looks noisy or speckled.
- Broad gray regions are not uniform; intensity jumps inside areas that should be smooth.
- Smooth grayscale ramps can fail badly, reconstructing as random dot-like intensity jumps instead of a gradual black-to-white transition.
- Some channels are much darker than others.
- Large bright regions consume the limited brightness budget.
- Loss decreases while the visible image does not improve.
- Gray levels are not monotonic.
- Random phase initialization can make runs unstable.
- Using black areas as noise sinks may hide noise but can reduce useful target area and channel efficiency.
- The user prefers darker but cleaner reconstructions over brighter noisy reconstructions.

## Current Best Next Algorithm Directions

After the automatic image workflow is smooth, likely algorithm improvements are:

1. Better initialization for `phdx` and `phdy`.
2. Cleaner grayscale target generation with brightness cap but less distortion.
3. More direct stitched-image similarity diagnostics.
4. Better handling of channel energy limits.
5. Early-stop or best-outer selection based on visible-quality-related metrics.
6. Object-region uniformity losses for flat gray regions.
7. High-frequency residual penalties inside non-black target regions.
8. Grayscale-ramp monotonicity/smoothness diagnostics that measure whether gradual tones remain gradual.

Keep these as incremental changes. Do not replace the FFT prototype with a heavier physical simulation unless the user explicitly asks.

## Algorithm Design Constraint

New algorithm code should stay compact and general.

- Prefer simple losses or metrics that apply to ordinary grayscale targets.
- Avoid brittle target-type special cases.
- Avoid large nested condition trees for "flat block" vs "gradient" vs "line" targets.
- Each added term must have a clear reason, a clear formula, and an output that proves whether it helped.
