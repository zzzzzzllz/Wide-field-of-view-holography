"""Core package for 9-channel grayscale holography optimization experiments.

Recommended way to read this package:

1. Target-image preparation
   - ``config.py``: run configuration, default channel orders, validation.
   - ``targets.py``: standard grayscale targets and MAT target loading.
   - ``line_targets.py``: RGB-image to line-art grayscale target generation.

2. On-chip structure proxy optimization
   - ``field.py``: FFT-based far-field simulation and differentiable loss terms.
   - ``weights.py``: adaptive per-channel reweighting during optimization.
   - ``runner.py``: full optimization loop that updates ``phdx`` and ``phdy``.

3. Evaluation and outputs
   - ``metrics.py``: reconstruction quality, grayscale, and efficiency metrics.
   - ``export.py``: result folders, CSV/JSON/NPZ exports, and summary figures.
   - ``cli.py``: command-line entrypoint that wires everything together.

Output-path convention:

- Real experiment runs are written under ``outputs/holo_experiments``.
- Test-only artifacts are written under ``outputs/test_export`` and
  ``outputs/test_runner``.
"""

__version__ = "0.1.0"
