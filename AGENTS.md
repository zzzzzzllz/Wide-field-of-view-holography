# Agent Entry Point

This repository is a wide-field-of-view holography optimization project. Future agents must treat this file as the entry point, not as the full knowledge base.

## Required Reading Order

Before making non-trivial changes, read:

1. `docs/agent/PROJECT_MEMORY.md`
2. `docs/agent/ALGORITHM_CONTEXT.md`
3. `docs/agent/WORKFLOW.md`
4. `docs/agent/ROADMAP.md`
5. `README.md`

For research or physics questions, also read:

1. `docs/RESEARCH_NOTES.md`
2. Relevant PDFs under `docs/supporting_data_pdf/`

## Skill Policy

- Do not proactively use Skills by default.
- Use a Skill only when it is a basic/general capability that is obviously appropriate, or when the user explicitly requests it with `/skill-name` in the current conversation.
- Keep `skills/holography-workflow/SKILL.md`; it is a project resource, but it is not an automatic instruction to use Skills.

## Project Goal

The practical goal is simple: make the reconstructed holography image look more like the target image.

The primary visual judge is:

- `stitched_comparison.png`: final target-vs-reconstruction comparison after stitching the 9 channels into one 3x3 image.

Diagnostic helpers are:

- `summary.png`: per-channel target/reconstruction comparison.
- `loss_terms.png`: loss component trends.
- `diagnostics.csv`: outer-loop score, image error, efficiency, and gray-level diagnostics.
- `eta_curve.png`: channel efficiency balance.

## Current Main Line

The intended user workflow is:

1. User provides one target image.
2. The project automatically converts it into a grayscale 9-channel target stack.
3. The FFT proxy optimizer updates `phdx` and `phdy`.
4. The run exports `stitched_comparison.png`.
5. Success means the stitched reconstruction is visually closer to the stitched target.

Do not make MATLAB `.mat` target preparation the main path. Keep `mat` mode as a compatibility path only.

## Documentation Rule

When the user gives a meaningful reminder, preference, or decision that should persist across future conversations, add it to `docs/agent/PROJECT_MEMORY.md`. Do not leave important project memory only in chat.

When algorithm behavior, target generation, exports, CLI usage, or validation strategy changes, update the relevant file under `docs/agent/`.

## Operating Constraints

- Main user-facing commands should be Windows PowerShell friendly and use `py`.
- macOS checks are useful for reading and lightweight validation, but do not describe them as full Windows validation.
- Do not delete experiment outputs unless the user explicitly asks.
- Keep changes scoped. Do not refactor unrelated code while doing documentation or algorithm work.
