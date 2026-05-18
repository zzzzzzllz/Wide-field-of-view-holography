# Historical Plan: Holography Quality Optimization

> Status: implemented. This file is retained as historical implementation context only.
> Do not use it as the current project roadmap. For current next-step directions, read `docs/ROADMAP.md`.

The work described here has already been integrated into the project baseline:

- `LossConfig` and `diagnostic_interval`
- structured loss terms
- diagnostic CSV and per-outer-loop exports
- runner integration
- CLI loss-weight options
- VS Code diagnostic and quality presets

The next phase is tracked in `docs/ROADMAP.md`.

---

## Original Plan

**Goal:** Add diagnostic exports and enhanced loss terms so noisy holography reconstructions can be diagnosed and improved.

**Architecture:** Keep the existing FFT-based optimization pipeline. Add a `LossConfig`, structured differentiable loss terms, optional diagnostic exports, and VS Code launch presets without replacing the current runner/export interfaces.

**Tech Stack:** Python, NumPy, PyTorch, matplotlib Agg, unittest, VS Code debug launch configurations.

---

## File Structure

- `holo_opt/config.py`: add `LossConfig` and `diagnostic_interval`; keep existing config defaults stable.
- `holo_opt/field.py`: add differentiable loss term helpers while keeping `training_loss(...)` compatible.
- `holo_opt/export.py`: add diagnostic CSV and per-outer-loop summary export helpers.
- `holo_opt/runner.py`: collect loss term history, use enhanced loss config, and export diagnostics.
- `holo_opt/cli.py`: expose loss-weight and diagnostic options.
- `.vscode/launch.json`: add diagnostic and quality launch presets.
- `tests/test_config.py`: cover new config defaults and validation.
- `tests/test_field.py`: cover enhanced loss terms.
- `tests/test_export.py`: cover diagnostic export files.
- `tests/test_runner.py`: cover tiny diagnostic run exports.
- `tests/test_cli.py`: cover new CLI options.

---

## Task 1: Loss Config Defaults And Validation

**Files:**
- Modify: `holo_opt/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Append tests like this to `tests/test_config.py`:

```python
def test_loss_config_defaults_are_quality_safe(self):
    config = ExperimentConfig()
    self.assertEqual(config.loss.image_weight, 1.0)
    self.assertEqual(config.loss.eta_balance_weight, 0.05)
    self.assertEqual(config.loss.gray_monotonic_weight, 0.1)
    self.assertEqual(config.loss.phase_smoothness_weight, 1e-4)
    self.assertEqual(config.loss.background_weight, 0.0)
    self.assertEqual(config.diagnostic_interval, 1)

def test_validate_config_rejects_invalid_loss_weights(self):
    config = ExperimentConfig()
    config.loss.image_weight = -1.0
    with self.assertRaisesRegex(ValueError, "loss weights"):
        validate_config(config)

def test_validate_config_rejects_invalid_diagnostic_interval(self):
    config = ExperimentConfig(diagnostic_interval=0)
    with self.assertRaisesRegex(ValueError, "diagnostic_interval"):
        validate_config(config)
```

- [ ] **Step 2: Run config tests to verify failure**

Run:

```powershell
py -m unittest tests.test_config -q
```

Expected: FAIL because `ExperimentConfig.loss` and `diagnostic_interval` do not exist.

- [ ] **Step 3: Implement config changes**

In `holo_opt/config.py`, add:

```python
@dataclass
class LossConfig:
    image_weight: float = 1.0
    eta_balance_weight: float = 0.05
    gray_monotonic_weight: float = 0.1
    phase_smoothness_weight: float = 1e-4
    background_weight: float = 0.0
```

Add fields to `ExperimentConfig`:

```python
diagnostic_interval: int = 1
loss: LossConfig = field(default_factory=LossConfig)
```

Add validation:

```python
    if config.diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive")
    loss_values = [
        config.loss.image_weight,
        config.loss.eta_balance_weight,
        config.loss.gray_monotonic_weight,
        config.loss.phase_smoothness_weight,
        config.loss.background_weight,
    ]
    if any(value < 0 for value in loss_values):
        raise ValueError("loss weights must be nonnegative")
```

- [ ] **Step 4: Run tests**

Run:

```powershell
py -m unittest tests.test_config -q
py -m unittest -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add holo_opt/config.py tests/test_config.py
git commit -m "feat: add holography loss config"
```

---

## Task 2: Structured Loss Terms

**Files:**
- Modify: `holo_opt/field.py`
- Modify: `tests/test_field.py`

- [ ] **Step 1: Write failing field loss tests**

Append tests like this to `tests/test_field.py`:

```python
def test_compute_loss_terms_returns_finite_terms(self):
    phdx = torch.zeros((4, 4), dtype=torch.float32)
    phdy = torch.zeros((4, 4), dtype=torch.float32)
    pair_mat = torch.tensor([[-2, 2], [-2, 1]], dtype=torch.float32)
    targets = torch.ones((2, 4, 4), dtype=torch.float32)
    weights = torch.ones(2, dtype=torch.float32)
    loss_weights = {
        "image_weight": 1.0,
        "eta_balance_weight": 0.05,
        "gray_monotonic_weight": 0.1,
        "phase_smoothness_weight": 1e-4,
        "background_weight": 0.0,
    }

    terms = compute_loss_terms(phdx, phdy, pair_mat, targets, weights, loss_weights)

    self.assertEqual(
        set(terms),
        {"total", "image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background"},
    )
    for value in terms.values():
        self.assertTrue(torch.isfinite(value).item())
        self.assertEqual(value.ndim, 0)

def test_phase_smoothness_zero_for_constant_phase(self):
    phase = torch.ones((4, 4), dtype=torch.float32)
    self.assertEqual(float(phase_smoothness_loss(phase, phase)), 0.0)

def test_phase_smoothness_positive_for_varying_phase(self):
    phase = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    self.assertGreater(float(phase_smoothness_loss(phase, torch.zeros_like(phase))), 0.0)

def test_gray_monotonic_loss_penalizes_inversions(self):
    target = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).reshape(1, 4, 4)
    good = target.clone()
    bad = 1.0 - target
    self.assertGreater(float(gray_monotonic_loss(bad, target)), float(gray_monotonic_loss(good, target)))
```

Also import:

```python
from holo_opt.field import compute_loss_terms, gray_monotonic_loss, phase_smoothness_loss
```

- [ ] **Step 2: Run field tests to verify failure**

```powershell
py -m unittest tests.test_field -q
```

Expected: FAIL because the new helpers do not exist.

- [ ] **Step 3: Implement loss helpers**

Add to `holo_opt/field.py`:

```python
def phase_smoothness_loss(phdx: torch.Tensor, phdy: torch.Tensor) -> torch.Tensor:
    dx_y = phdx[1:, :] - phdx[:-1, :]
    dx_x = phdx[:, 1:] - phdx[:, :-1]
    dy_y = phdy[1:, :] - phdy[:-1, :]
    dy_x = phdy[:, 1:] - phdy[:, :-1]
    return dx_y.square().mean() + dx_x.square().mean() + dy_y.square().mean() + dy_x.square().mean()


def gray_monotonic_loss(reconstruction: torch.Tensor, targets: torch.Tensor, levels: int = 16) -> torch.Tensor:
    if reconstruction.ndim == 2:
        reconstruction = reconstruction.unsqueeze(0)
    if targets.ndim == 2:
        targets = targets.unsqueeze(0)
    penalties = []
    level_indices = torch.round(torch.clamp(targets, 0.0, 1.0) * float(levels - 1)).long()
    for channel in range(targets.shape[0]):
        means = []
        for level in range(levels):
            mask = level_indices[channel] == level
            if torch.any(mask):
                means.append(reconstruction[channel][mask].mean())
        if len(means) >= 2:
            means_tensor = torch.stack(means)
            penalties.append(torch.relu(-(means_tensor[1:] - means_tensor[:-1])).mean())
    if not penalties:
        return reconstruction.new_tensor(0.0)
    return torch.stack(penalties).mean()


def channel_energy_balance_loss(intensities: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    energy = intensities.sum(dim=(-2, -1))
    mean_energy = energy.mean()
    return energy.std(unbiased=False) / (mean_energy + epsilon)
```

Add structured loss:

```python
def compute_loss_terms(
    phdx: torch.Tensor,
    phdy: torch.Tensor,
    pair_mat: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    loss_weights: dict[str, float] | None = None,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if loss_weights is None:
        loss_weights = {}
    intensities = compute_intensities(phdx, phdy, pair_mat)
    normalized_intensity = normalize_intensities(intensities, epsilon=epsilon)
    target_max = targets.amax(dim=(-2, -1), keepdim=True)
    normalized_target = targets / (target_max + epsilon)
    per_channel = ((normalized_intensity - normalized_target) ** 2).mean(dim=(-2, -1))
    image_mse = torch.sum(weights * per_channel)
    eta_balance = channel_energy_balance_loss(intensities, epsilon=epsilon)
    gray_monotonic = gray_monotonic_loss(normalized_intensity, normalized_target)
    smoothness = phase_smoothness_loss(phdx, phdy)
    background = normalized_intensity.new_tensor(0.0)
    total = (
        float(loss_weights.get("image_weight", 1.0)) * image_mse
        + float(loss_weights.get("eta_balance_weight", 0.0)) * eta_balance
        + float(loss_weights.get("gray_monotonic_weight", 0.0)) * gray_monotonic
        + float(loss_weights.get("phase_smoothness_weight", 0.0)) * smoothness
        + float(loss_weights.get("background_weight", 0.0)) * background
    )
    return {
        "total": total,
        "image_mse": image_mse,
        "eta_balance": eta_balance,
        "gray_monotonic": gray_monotonic,
        "phase_smoothness": smoothness,
        "background": background,
    }
```

Update `training_loss(...)` to call `compute_loss_terms(...)["total"]`, adding an optional `loss_weights` argument while keeping existing calls valid.

- [ ] **Step 4: Run tests**

```powershell
py -m unittest tests.test_field -q
py -m unittest -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add holo_opt/field.py tests/test_field.py
git commit -m "feat: add structured holography loss terms"
```

---

## Task 3: Diagnostic Export Files

**Files:**
- Modify: `holo_opt/export.py`
- Modify: `tests/test_export.py`

- [ ] **Step 1: Write failing export diagnostics tests**

Append to `tests/test_export.py`:

```python
def test_export_results_writes_diagnostics_when_provided(self):
    tmp_path = Path.cwd() / "outputs" / "test_export" / uuid.uuid4().hex
    tmp_path.mkdir(parents=True)
    self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    tmp = str(tmp_path)
    try:
        config = ExperimentConfig(size=4, output_root=tmp, label="diag")
        targets = np.ones((9, 4, 4), dtype=np.float32)
        intensities = np.ones((9, 4, 4), dtype=np.float32)
        phdx = np.zeros((4, 4), dtype=np.float32)
        phdy = np.zeros((4, 4), dtype=np.float32)
        metrics = {
            "rows": [
                {"channel": index + 1, "mse": 0.1, "eta": 0.5, "gray_level_error": 0.2, "gray_means": [0.0] * 16}
                for index in range(9)
            ],
            "summary": {"score": 1.0, "image_error": 0.1, "gray_level_error": 0.2, "efficiency_balance_penalty": 0.3, "mean_eta": 0.5},
        }
        diagnostics = [
            {"outer": 1, "loss": 2.0, "score": 1.0, "mean_eta": 0.5, "eta_balance": 0.3, "image_error": 0.1, "gray_level_error": 0.2, "weight_min": 1.0, "weight_max": 1.0}
        ]
        loss_terms_history = [
            {"step": 1, "total": 2.0, "image_mse": 1.0, "eta_balance": 0.2, "gray_monotonic": 0.3, "phase_smoothness": 0.4, "background": 0.0}
        ]
        run_dir = export_results(
            config,
            targets,
            intensities,
            phdx,
            phdy,
            [2.0],
            [[0.5] * 9],
            [[1.0] * 9],
            metrics,
            diagnostics=diagnostics,
            loss_terms_history=loss_terms_history,
            outer_summaries=[(1, intensities)],
        )
        self.assertTrue((run_dir / "diagnostics.csv").exists())
        self.assertTrue((run_dir / "loss_terms.csv").exists())
        self.assertTrue((run_dir / "outer_001_summary.png").exists())
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
```

- [ ] **Step 2: Run export tests to verify failure**

```powershell
py -m unittest tests.test_export -q
```

Expected: FAIL because `export_results` does not accept diagnostic arguments.

- [ ] **Step 3: Implement diagnostic export**

Extend `export_results(...)` signature:

```python
    diagnostics: list[dict[str, float]] | None = None,
    loss_terms_history: list[dict[str, float]] | None = None,
    outer_summaries: list[tuple[int, np.ndarray]] | None = None,
```

Add CSV writer:

```python
def _write_rows_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

In `export_results(...)`:

```python
    if diagnostics:
        _write_rows_csv(run_dir / "diagnostics.csv", diagnostics)
    if loss_terms_history:
        _write_rows_csv(run_dir / "loss_terms.csv", loss_terms_history)
    if outer_summaries:
        for outer_index, outer_intensities in outer_summaries:
            _plot_summary(run_dir / f"outer_{outer_index:03d}_summary.png", targets, outer_intensities)
```

- [ ] **Step 4: Run tests**

```powershell
py -m unittest tests.test_export -q
py -m unittest -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add holo_opt/export.py tests/test_export.py
git commit -m "feat: export holography diagnostics"
```

---

## Task 4: Runner Integration For Enhanced Loss And Diagnostics

**Files:**
- Modify: `holo_opt/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write failing runner diagnostics test**

Append to `tests/test_runner.py`:

```python
def test_run_experiment_exports_diagnostics(self):
    output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
    output_root.mkdir(parents=True)
    self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))
    config = ExperimentConfig(
        size=8,
        epochs_per_chunk=1,
        outer_loops=1,
        output_root=str(output_root),
        label="diag",
        device="cpu",
    )
    result = run_experiment(config)
    self.assertTrue((result.run_dir / "diagnostics.csv").exists())
    self.assertTrue((result.run_dir / "loss_terms.csv").exists())
    self.assertTrue((result.run_dir / "outer_001_summary.png").exists())
```

- [ ] **Step 2: Run runner tests to verify failure**

```powershell
py -m unittest tests.test_runner -q
```

Expected: FAIL because runner does not pass diagnostics into `export_results`.

- [ ] **Step 3: Integrate enhanced loss**

In `holo_opt/runner.py`, import `compute_loss_terms`:

```python
from holo_opt.field import compute_intensities, compute_loss_terms
```

Add helper:

```python
def loss_config_to_dict(config: ExperimentConfig) -> dict[str, float]:
    return {
        "image_weight": config.loss.image_weight,
        "eta_balance_weight": config.loss.eta_balance_weight,
        "gray_monotonic_weight": config.loss.gray_monotonic_weight,
        "phase_smoothness_weight": config.loss.phase_smoothness_weight,
        "background_weight": config.loss.background_weight,
    }
```

In the training loop, replace:

```python
loss = training_loss(phdx, phdy, pair_mat, targets, weights)
```

with:

```python
terms = compute_loss_terms(phdx, phdy, pair_mat, targets, weights, loss_config_to_dict(config))
loss = terms["total"]
loss_terms_history.append({
    "step": len(losses) + 1,
    "total": float(terms["total"].detach().cpu().item()),
    "image_mse": float(terms["image_mse"].detach().cpu().item()),
    "eta_balance": float(terms["eta_balance"].detach().cpu().item()),
    "gray_monotonic": float(terms["gray_monotonic"].detach().cpu().item()),
    "phase_smoothness": float(terms["phase_smoothness"].detach().cpu().item()),
    "background": float(terms["background"].detach().cpu().item()),
})
```

Initialize before loops:

```python
loss_terms_history: list[dict[str, float]] = []
diagnostics: list[dict[str, float]] = []
outer_summaries: list[tuple[int, np.ndarray]] = []
```

After metrics are computed each outer loop:

```python
outer_number = _outer_index + 1
diagnostics.append({
    "outer": float(outer_number),
    "loss": losses[-1],
    "score": score,
    "mean_eta": float(metrics["summary"]["mean_eta"]),
    "eta_balance": float(metrics["summary"]["efficiency_balance_penalty"]),
    "image_error": float(metrics["summary"]["image_error"]),
    "gray_level_error": float(metrics["summary"]["gray_level_error"]),
    "weight_min": float(np.min(weights_np)),
    "weight_max": float(np.max(weights_np)),
})
if outer_number % config.diagnostic_interval == 0:
    outer_summaries.append((outer_number, intensities_np.copy()))
```

Pass to `export_results(...)`:

```python
        diagnostics=diagnostics,
        loss_terms_history=loss_terms_history,
        outer_summaries=outer_summaries,
```

- [ ] **Step 4: Run tests**

```powershell
py -m unittest tests.test_runner -q
py -m unittest -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add holo_opt/runner.py tests/test_runner.py
git commit -m "feat: add runner diagnostics"
```

---

## Task 5: CLI And VS Code Quality Presets

**Files:**
- Modify: `holo_opt/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `.vscode/launch.json`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
def test_parser_exposes_loss_weight_options(self):
    args = build_parser().parse_args([
        "--eta-balance-weight", "0.2",
        "--gray-monotonic-weight", "0.3",
        "--phase-smoothness-weight", "0.004",
        "--background-weight", "0.1",
        "--diagnostic-interval", "2",
    ])
    config = config_from_args(args)
    self.assertEqual(config.loss.eta_balance_weight, 0.2)
    self.assertEqual(config.loss.gray_monotonic_weight, 0.3)
    self.assertEqual(config.loss.phase_smoothness_weight, 0.004)
    self.assertEqual(config.loss.background_weight, 0.1)
    self.assertEqual(config.diagnostic_interval, 2)
```

- [ ] **Step 2: Run CLI tests to verify failure**

```powershell
py -m unittest tests.test_cli -q
```

Expected: FAIL because the parser does not expose the new arguments.

- [ ] **Step 3: Implement CLI options**

In `holo_opt/cli.py`, import `LossConfig`:

```python
from holo_opt.config import ExperimentConfig, GuidedModeConfig, LossConfig, PhysicalConfig, WeightUpdateConfig
```

Add parser args:

```python
    parser.add_argument("--diagnostic-interval", type=int, default=1)
    parser.add_argument("--eta-balance-weight", type=float, default=0.05)
    parser.add_argument("--gray-monotonic-weight", type=float, default=0.1)
    parser.add_argument("--phase-smoothness-weight", type=float, default=1e-4)
    parser.add_argument("--background-weight", type=float, default=0.0)
```

Pass into `ExperimentConfig`:

```python
        diagnostic_interval=args.diagnostic_interval,
        loss=LossConfig(
            eta_balance_weight=args.eta_balance_weight,
            gray_monotonic_weight=args.gray_monotonic_weight,
            phase_smoothness_weight=args.phase_smoothness_weight,
            background_weight=args.background_weight,
        ),
```

- [ ] **Step 4: Update VS Code launch configs**

Modify `.vscode/launch.json` to include:

```json
{
  "name": "Holo diagnostic: standard 64",
  "type": "debugpy",
  "request": "launch",
  "module": "holo_opt.cli",
  "console": "integratedTerminal",
  "args": [
    "--target-mode", "standard",
    "--size", "64",
    "--epochs-per-chunk", "300",
    "--outer-loops", "5",
    "--device", "cpu",
    "--output-root", "outputs/holo_experiments",
    "--label", "diagnostic",
    "--eta-balance-weight", "0.05",
    "--gray-monotonic-weight", "0.1",
    "--phase-smoothness-weight", "0.0001"
  ]
}
```

Add:

```json
{
  "name": "Holo quality: standard 128",
  "type": "debugpy",
  "request": "launch",
  "module": "holo_opt.cli",
  "console": "integratedTerminal",
  "args": [
    "--target-mode", "standard",
    "--size", "128",
    "--epochs-per-chunk", "1000",
    "--outer-loops", "5",
    "--device", "cpu",
    "--output-root", "outputs/holo_experiments",
    "--label", "quality",
    "--eta-balance-weight", "0.05",
    "--gray-monotonic-weight", "0.1",
    "--phase-smoothness-weight", "0.0001"
  ]
}
```

- [ ] **Step 5: Run tests**

```powershell
py -m unittest tests.test_cli -q
py -m unittest -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```powershell
git add holo_opt/cli.py tests/test_cli.py .vscode/launch.json
git commit -m "feat: add quality optimization presets"
```

---

## Task 6: Full Verification And Diagnostic Smoke

**Files:**
- Modify: none beyond ignored output directories

- [ ] **Step 1: Run full tests**

```powershell
py -m unittest discover -s tests -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run diagnostic smoke**

```powershell
py -m holo_opt.cli --target-mode standard --size 16 --epochs-per-chunk 2 --outer-loops 2 --device cpu --output-root outputs/holo_experiments --label diag_smoke --eta-balance-weight 0.05 --gray-monotonic-weight 0.1 --phase-smoothness-weight 0.0001
```

Expected: command exits 0 and prints a directory beginning with `outputs\holo_experiments\diag_smoke_9ch_16_`.

- [ ] **Step 3: Verify diagnostic output files**

```powershell
$latest = Get-ChildItem -LiteralPath 'outputs\holo_experiments' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem -LiteralPath $latest.FullName | Select-Object -ExpandProperty Name
```

Expected names include:

```text
diagnostics.csv
loss_terms.csv
outer_001_summary.png
outer_002_summary.png
summary.png
config.json
metrics.csv
metrics.json
optimized_results.npz
phdx.csv
phdy.csv
```

- [ ] **Step 4: Check git status**

```powershell
git status --short
```

Expected: only unrelated untracked local files remain, such as `docs/superpowers/plans/2026-05-07-wide-fov-grayscale-holography.md` or `大创资料/`, if still untracked.

- [ ] **Step 5: Push branch**

```powershell
git push
```

Expected: branch updates on GitHub.
