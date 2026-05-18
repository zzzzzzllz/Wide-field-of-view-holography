---
name: holography-workflow
description: Shared workflow for the Wide-field-of-view-holography project. Use when Codex works on this repository to modify holography algorithms, target preprocessing, spherical field-of-view mapping, grayscale or energy-budget optimization, experiment exports, tests, Git commits, PRs, or collaboration hygiene.
---

# Holography Workflow

Use this skill for every non-trivial change in this repository. Keep changes scoped, reproducible, and easy for collaborators to verify on Windows.

## Start Every Task

1. Read `AGENT.MD` first, then inspect the files directly related to the requested change.
2. Run `git status --short` before editing.
3. Treat existing uncommitted changes as user work. Do not revert or stage unrelated files.
4. Prefer Windows PowerShell commands and `py` in user-facing instructions.
5. Do not commit experiment inputs or outputs. `inputs/` and `outputs/` are local working directories; only `.gitkeep` placeholders belong in Git.

## Project Reading Order

Use this order when context is missing:

1. `AGENT.MD`
2. `README.md`
3. `holo_opt/runner.py`
4. `holo_opt/line_targets.py`
5. `holo_opt/field.py`
6. `holo_opt/metrics.py`
7. `holo_opt/export.py`
8. Relevant tests under `tests/`

## Algorithm Change Rules

- Keep the FFT proxy architecture unless the user explicitly asks for RCWA/FDTD, metasurface unit-cell lookup, GDS export, or material modeling.
- For spherical field-of-view work, add a geometry/mapping layer first; keep planar mode available for comparison.
- For grayscale work, validate both visual output and metrics: `summary.png`, `stitched_comparison.png`, `gray_levels.png`, `loss_terms.png`, and `diagnostics.csv`.
- For image preprocessing work, export comparisons of original image, processed target, stitched target, and stitched reconstruction.
- For energy-budget work, check `eta_curve.png`, `metrics.csv`, and channel-level MSE/eta balance before claiming improvement.
- Any change to target generation, loss terms, runner behavior, exports, CLI options, or test strategy must update `AGENT.MD`.

## Validation Commands

Run the full test suite before commit:

```powershell
py -m unittest discover -s tests -q
```

For a fast runtime smoke test after algorithm, runner, CLI, or export changes:

```powershell
py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 2 --device cpu --output-root outputs/holo_experiments --label smoke
```

For grayscale input checks:

```powershell
py -m holo_opt.grayscale_preview --input demo.png --size 64
py -m holo_opt.cli --target-mode grayscale --target-path inputs/lineart_sources/demo.png --size 64 --epochs-per-chunk 20 --outer-loops 1 --device cpu --output-root outputs/holo_experiments --label grayscale_probe
```

## Inspecting Results

After an experiment, inspect files in this order:

1. `summary.png`
2. `stitched_comparison.png`
3. latest `outer_###_summary.png`
4. `loss_terms.png`
5. `diagnostics.csv`
6. `gray_levels.png`
7. `eta_curve.png`
8. `metrics.json` and `metrics.csv`

Report exact output paths and explain which file proves the change worked.

## Git Hygiene

Use one branch per feature:

```text
codex/<short-topic>
```

Commit message prefixes:

- `feat:` new capability
- `fix:` bug fix
- `docs:` documentation or workflow text
- `test:` tests only
- `chore:` cleanup, ignore rules, repository hygiene
- `refactor:` behavior-preserving code structure change

Before staging:

```powershell
git status --short
git ls-files inputs outputs
```

Expected tracked data directories:

```text
inputs/lineart_sources/.gitkeep
outputs/.gitkeep
```

Stage only files related to the current task. Do not use `git add .` when unrelated user edits are present.

## PR Template

Use this structure for PR descriptions or final delivery notes:

```markdown
## Summary
- What changed.
- Why it changed.

## Verification
- `py -m unittest discover -s tests -q`
- Any smoke or experiment command used.

## Result Review
- Key output images or CSV files to inspect.
- What improvement or behavior should be visible.

## Risks
- Known limitations.
- What has not been verified yet.
```

## Collaboration Defaults

- Keep `README.md` as the quick-start document.
- Keep `AGENT.MD` as the full project maintenance context.
- Keep this skill as the operational checklist for Codex and collaborators.
- Do not delete historical local experiment output unless the user explicitly asks; ignore it in Git instead.
