# Roadmap

This file splits the large project into smaller future tasks.

## Priority 1: Automatic Image Main Line

Goal:

```text
one user target image -> automatic grayscale target stack -> FFT optimization -> stitched_comparison.png
```

Required behavior:

- User gives one target image.
- The program converts it into a grayscale target stack automatically.
- The conversion preserves the original grayscale relationship.
- The conversion applies a maximum brightness cap.
- The user does not need MATLAB `.mat` generation.
- The main result to inspect is `stitched_comparison.png`.

Current gap:

- Existing `grayscale` mode works, but uses stronger gamma compression and flat-region darkening.
- Future work should align it with the chosen policy: preserve grayscale + brightness cap.

Acceptance check:

- A single command can run from image input to exported stitched comparison.
- Documentation tells the user exactly where to put the image and which output to inspect.

## Priority 2: Improve Clarity And Reduce Noise

Goal:

- Make the stitched reconstruction less blurry and less noisy.
- Improve visible similarity to the target image.
- Make broad gray regions visually uniform instead of speckled.
- Make grayscale ramps smooth instead of random dot-like jumps.
- Preserve boundary/shape clarity first, then improve gray-region uniformity, then exact gray brightness.

Likely work:

- First evaluate whether existing `gray_level_stats` is enough for block uniformity; if not, add one compact metric rather than a large special-case evaluator.
- Improve phase initialization.
- Tune or redesign loss weights for image similarity.
- Add or refine smoothness/noise regularization.
- Add object-region uniformity loss for areas that should be flat gray.
- Add high-frequency residual or local-variance diagnostics.
- Add a grayscale-ramp benchmark or metric for black-to-white smooth transitions.
- Evaluate whether black sacrificial regions help enough to justify losing useful image area.
- Compare best outer-loop outputs instead of only final output.

Acceptance check:

- `stitched_comparison.png` visually improves on the same input image and seed.
- Diagnostics explain whether the improvement came from lower image error, better energy balance, or less noise.
- On a benchmark with flat gray blocks, the reconstructed gray block becomes smoother without destroying the outline.
- On a black-to-white ramp, the reconstruction should preserve a gradual tone trend rather than turning into strong local jumps.

## Priority 3: Improve Diagnostics

Goal:

- Make failed runs easier to understand.

Likely work:

- Add stitched-level image similarity metrics.
- Add channel brightness budget diagnostics.
- Add noise/speckle indicators.
- Improve visual reports that compare target, reconstruction, difference, and channel errors.

Acceptance check:

- When a run looks bad, the exported files make the main reason clear.

## Priority 4: Research-Informed Physical Refinement

Goal:

- Use supporting papers to guide more realistic design choices without losing the working FFT prototype.

Likely work:

- Clarify how the proxy phase maps relate to on-chip detour phase ideas.
- Decide whether brightness modulation should be modeled separately from phase.
- Keep physical assumptions documented before implementing heavier simulation.

Acceptance check:

- Any added physical feature has a clear reason, a small implementation scope, and a diagnostic proving it helps image quality.

## Out Of Scope For Now

- Full RCWA/FDTD simulation.
- Nanostructure geometry library mapping.
- GDS export.
- Fabrication tolerance modeling.
- Replacing the FFT optimization pipeline.
