# Holography Quality Optimization Design

## Goal

Improve reconstruction quality when `summary.png` looks like speckle or snow. The next iteration should make failures diagnosable and add loss terms that directly target the common failure modes:

- noisy reconstructions even when loss decreases
- poor grayscale monotonicity in `gray_levels.png`
- energy collapse into only a few channels in `eta_curve.png`
- unstable or ineffective optimization from random phase initialization

This is an algorithm-quality upgrade for the existing Fourier holography prototype. It should not add RCWA/FDTD, metasurface unit-cell lookup, GDS export, or material modeling.

## User Workflow

The user should still run experiments from VS Code using launch configurations. The first useful diagnostic run should be:

- `Holo diagnostic: standard 64`
- `size=64`
- `epochs_per_chunk=300`
- `outer_loops=5`
- `device=cpu`

For quality runs, the user should run:

- `Holo quality: standard 128`
- `size=128`
- `epochs_per_chunk=1000`
- `outer_loops=5`

Each run should export intermediate diagnostics so the user can see whether the algorithm is improving between outer loops, not just at the final state.

## Proposed Approach

Use a staged improvement rather than a large rewrite.

1. Add diagnostic exports.
   - Save per-outer-loop summaries.
   - Save `diagnostics.csv` containing loss, score, mean eta, eta balance, image error, gray error, and weight range.
   - Make it easy to compare whether outer loop 1, 2, 3, etc. improves.

2. Add configurable loss components.
   - Keep the existing image MSE as the base loss.
   - Add channel efficiency balance loss to prevent energy collapse.
   - Add grayscale monotonicity loss for 16-level targets.
   - Add phase smoothness regularization to reduce high-frequency phase noise.
   - Keep all new terms controlled by config weights so they can be disabled.

3. Add quality presets.
   - `smoke`: very small and fast, for checking the program runs.
   - `diagnostic`: medium runtime, useful for debugging quality.
   - `quality`: slower but better reconstruction.

This keeps the current architecture while making the optimization behavior visible and adjustable.

## Components

### Config

Add a new config group, `LossConfig`, with defaults that keep behavior close to the current implementation:

- `image_weight=1.0`
- `eta_balance_weight=0.05`
- `gray_monotonic_weight=0.1`
- `phase_smoothness_weight=1e-4`
- `background_weight=0.0`

Add a run option for `diagnostic_interval`, defaulting to one export per outer loop.

### Field Loss

Replace the single scalar `training_loss(...)` output with a structured loss helper:

- `compute_loss_terms(...) -> dict[str, torch.Tensor]`
- `training_loss(...)` remains as a compatibility wrapper returning the total loss.

Loss terms:

- `image_mse`: weighted per-channel MSE between normalized reconstruction and normalized target.
- `eta_balance`: penalize high standard deviation of channel total energy.
- `gray_monotonic`: for standard 16-level targets, penalize negative adjacent differences in reconstructed mean intensity per gray level.
- `phase_smoothness`: penalize adjacent finite differences in `phdx` and `phdy`.

The total loss is the weighted sum of these terms.

### Runner

Update `run_experiment` to:

- collect `loss_terms_history`
- export diagnostics at each outer loop
- select best state using the configured score
- keep adaptive channel weights, but use the enhanced error signal

The runner should remain deterministic for a fixed seed.

### Export

Add optional diagnostic output files:

- `diagnostics.csv`
- `outer_001_summary.png`, `outer_002_summary.png`, etc.
- `loss_terms.csv`

Existing output files remain unchanged.

### CLI And VS Code

Add CLI arguments for the new loss weights:

- `--eta-balance-weight`
- `--gray-monotonic-weight`
- `--phase-smoothness-weight`
- `--background-weight`

Add launch configurations:

- `Holo diagnostic: standard 64`
- `Holo quality: standard 128`

## Diagnosis Rules

The documentation and launch names should guide the user this way:

- If `loss_curve.png` is flat, adjust learning rate or initialization.
- If loss decreases but `summary.png` stays noisy, increase gray monotonic and smoothness weights.
- If some channels are blank or very dim, increase eta balance and adaptive weight update strength.
- If grayscale curves are not monotonic, increase gray monotonic weight.
- If output has high-frequency snow everywhere, increase phase smoothness weight moderately.
- If the image is overly blurred, reduce phase smoothness.

## Testing

Add focused unit tests for:

- loss term keys and finite scalar outputs
- zero phase smoothness for constant phase
- positive phase smoothness for varying phase
- gray monotonic penalty increasing when grayscale means are inverted
- runner exports diagnostics files in a tiny run
- CLI parser exposes new loss-weight options

Keep tests small enough to run on CPU.

## Out Of Scope

- RCWA/FDTD validation
- actual metasurface cell-library mapping
- phase quantization
- multi-seed batch optimization
- CUDA-specific tests
- replacing the FFT model

## Success Criteria

The upgrade is successful when:

- full unittest suite passes
- a diagnostic run exports intermediate summaries and diagnostics CSV files
- loss terms are inspectable in output files
- user can choose diagnostic or quality launch configurations from VS Code
- the framework gives clear evidence about whether poor images are caused by non-convergence, energy imbalance, grayscale failure, or excessive phase noise
