# Project Memory

This file stores long-lived project memory from user conversations. Future agents should read it before planning or editing.

## User Intent

- The user wants the final generated holography image to look more like the original target image.
- The user does not care about unrelated physical completeness at this stage if it does not help the image become more similar.
- The current pain points are blurry reconstruction, noisy output, channel brightness limits, and poor final visual quality.
- The current main quality pain point is no longer only "can the outline appear"; outlines can already appear in some cases, but large gray regions are noisy and non-uniform.
- For gray blocks or broad smooth areas, the desired result is a stable, even gray tone rather than jumping speckle-like intensity.
- Smooth grayscale gradients are also a major failure case. A target that should transition gradually from black to gray to white may reconstruct as random dot-like jumps instead of a smooth ramp.
- The user wants the large project split into smaller, clear subprojects so future work can continue across conversations.
- The user treats gray-block smoothness and noise suppression as a long-term project goal, so solutions should be judged not only by quick local tweaks but also by whether they support a durable long-range path.

## Persistent Preferences

- Reply in Chinese by default unless the user asks otherwise.
- Be concise, clear, and direct.
- Do not proactively use Skills unless the user explicitly requests one with `/skill-name`, except for basic/general obvious capabilities.
- If a user reminder or project decision is meaningful for future work, write it into this memory/documentation structure.
- Ask when the goal or tradeoff is unclear, but first inspect the repository if the answer is discoverable from files.
- Keep code as simple and minimal as possible, especially for algorithm changes.
- Do not add long chains of special-case `if` logic to detect every target type and optimize each one separately.
- When adding an algorithmic component, explain exactly why it is added, what logic it follows, and how to judge whether it helped.

## Final Result Definition

- Primary final-quality file: `stitched_comparison.png`.
- `stitched_comparison.png` stitches the 9 target channels into one 3x3 target image and stitches the 9 reconstructed channels into one 3x3 reconstruction image.
- The final question is: does the stitched reconstruction look more like the stitched target?
- `summary.png` is a diagnostic file for per-channel comparison, not the primary final result.
- Current visual priority: preserve overall contour and spatial position first, then reduce speckle/noise inside broad gray regions, then improve grayscale tone accuracy and fine texture.
- Refined visual priority: boundary/shape clarity is slightly more important than gray-block uniformity, and both are more important than exact brightness.
- The user specifically wants broad gray regions to look uniform; noisy gray-region texture should be treated as a major failure mode.
- Clean and stable tone is more important than brightness. A darker but cleaner reconstruction is preferred over a brighter reconstruction with strong noise.
- For a flat gray block, it is acceptable if the reconstructed gray is dimmer than the target, as long as the block is internally close to one consistent tone.
- Avoid obvious local jumps inside a region even if exact target brightness cannot be reached.
- For gray-block cleanup, prefer a general local denoising/uniformity loss rather than target-type-specific rules.
- This local denoising loss should focus on non-black target regions and can ignore fully black regions.
- Within that general local denoising loss, target edges should receive lower weight than flat interior regions so contour clarity is not smoothed away.

## Noise Reduction Discussion

- A previous suggested strategy was to leave black regions that can absorb noise.
- This is a real tradeoff: dark sacrificial regions may reduce visible noise in the object, but they also reduce the useful image area and can waste channel capacity.
- The user agrees black sacrificial/noise-sink regions should be treated as a comparison experiment, not as a default assumption.
- Future algorithm work should compare this against direct object-region denoising, local uniformity losses, frequency-domain noise penalties, and better channel-energy constraints.

## Benchmark Decision

- Use `inputs/lineart_sources/benchmarks/benchmark_geometric_512.png` as the first long-term benchmark image.
- This benchmark is useful because it includes flat gray regions, thin lines, circular shapes, dots, and grayscale steps.
- The benchmark should be used to check both outline preservation and gray-region smoothness.
- Solve discrete gray blocks before continuous gradients. Continuous black-to-white ramp behavior is important, but it should come after flat block uniformity is improved.
- For visual-quality validation, do not treat the built-in `standard` target as the main evaluation image when the benchmark image is available.
- For a meaningful quality run, keep total optimization steps around 10,000 (`epochs_per_chunk * outer_loops ≈ 10000`) rather than very short smoke-style runs.
- Add a standalone benchmark evaluator instead of changing the main optimizer/export path first.
- The current flat-region noise metric definition is:
  - normalize each channel separately,
  - stitch the 9 channels into one 3x3 image,
  - keep only target pixels with low local gradient and target brightness above `0.05`,
  - compute residual `reconstruction - target`,
  - measure `9x9` local residual variance inside that flat-region mask.
- Treat this as a diagnostic metric for "how messy broad flat regions are" and do not fold it into the optimization score yet.

## Input Workflow Decision

- The desired main path is not MATLAB `.mat` preparation.
- The desired main path is: user gives one target image, the project automatically generates a grayscale 9-channel target stack, then runs the core FFT optimization.
- `mat` mode should remain available only as a compatibility path for externally prepared targets.
- `lineart` mode is useful for early contour experiments, but it is not the main target-image workflow.
- `standard` mode is a diagnostic target for algorithm checks, not the user's final target.

## Grayscale Target Decision

- The target image should preserve the original grayscale relationship.
- Add a maximum brightness cap so large bright regions do not consume the limited channel brightness budget.
- Avoid requiring the user to manually pre-process the image.
- Do not silently redefine the existing `grayscale` path when testing a different target-preprocessing idea.
- If a new direct-image path is needed, add it as a separate mode rather than changing the meaning of `grayscale`.

## Documentation Architecture Decision

- Do not put all project memory into one huge `AGENTS.md`.
- `AGENTS.md` is the entry point and reading map.
- Detailed long-term memory lives in `docs/agent/PROJECT_MEMORY.md`.
- Workflow, algorithm context, roadmap, and research notes each live in their own files.

## Current Project State

- The project is not an empty prototype.
- It already has an FFT-based 9-channel optimization pipeline.
- It optimizes proxy phase maps `phdx` and `phdy`, not real nanostructure geometry.
- It exports diagnostics and visual files that can be used to understand why a run failed.
- The next useful step is to make the automatic single-image-to-stitched-result workflow smooth before doing deeper algorithm changes.
