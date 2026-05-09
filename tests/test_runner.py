import csv
import contextlib
import io
import shutil
import uuid
import unittest
from pathlib import Path

import numpy as np
import torch

import holo_opt.runner as runner
from holo_opt.config import ExperimentConfig, ScoreConfig
from holo_opt.runner import (
    compute_score,
    format_progress_message,
    load_targets_for_config,
    loss_config_to_dict,
    resolve_device,
    run_experiment,
)


class RunnerTest(unittest.TestCase):
    def test_resolve_device_accepts_cpu(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_load_targets_for_standard_config_returns_valid_shape(self):
        config = ExperimentConfig(size=8, target_mode="standard")
        targets = load_targets_for_config(config)
        self.assertEqual(targets.shape, (9, 8, 8))
        self.assertEqual(targets.dtype, np.float32)

    def test_compute_score_uses_score_config_weights(self):
        summary = {
            "image_error": 1.0,
            "gray_level_error": 2.0,
            "efficiency_balance_penalty": 3.0,
            "mean_eta": 0.5,
        }
        score_config = ScoreConfig(
            image_weight=10.0,
            gray_level_weight=1.0,
            balance_weight=0.1,
            total_efficiency_weight=4.0,
        )

        self.assertAlmostEqual(compute_score(summary, score_config), 10.3)

    def test_loss_config_to_dict_exposes_training_weights(self):
        config = ExperimentConfig()

        self.assertEqual(
            loss_config_to_dict(config),
            {
                "image_weight": 1.0,
                "eta_balance_weight": 0.05,
                "gray_monotonic_weight": 0.1,
                "phase_smoothness_weight": 1e-4,
                "background_weight": 0.0,
            },
        )

    def test_format_progress_message_reports_every_500_steps(self):
        self.assertEqual(
            format_progress_message(500, 900, 0.125),
            "step 500/900 loss=0.125000",
        )
        self.assertIsNone(format_progress_message(499, 900, 0.125))

    def test_run_experiment_smoke_exports_results(self):
        output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
        output_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))

        config = ExperimentConfig(
            size=8,
            epochs_per_chunk=1,
            outer_loops=1,
            output_root=str(output_root),
            label="smoke",
            device="cpu",
        )

        result = run_experiment(config)

        self.assertTrue(result.run_dir.name.startswith("smoke_9ch_8_"))
        self.assertEqual(result.final_intensities.shape, (9, 8, 8))
        self.assertTrue(result.run_dir.exists())
        self.assertIn("summary", result.final_metrics)
        self.assertIn("score", result.final_metrics["summary"])
        for name in (
            "config.json",
            "optimized_results.npz",
            "metrics.csv",
            "metrics.json",
            "summary.png",
            "eta_curve.png",
            "loss_curve.png",
            "gray_levels.png",
            "phdx.csv",
            "phdy.csv",
        ):
            self.assertTrue((result.run_dir / name).exists(), name)

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

    def test_run_experiment_prints_progress_at_interval(self):
        output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
        output_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))
        original_interval = runner.PROGRESS_INTERVAL_STEPS
        self.addCleanup(lambda: setattr(runner, "PROGRESS_INTERVAL_STEPS", original_interval))
        runner.PROGRESS_INTERVAL_STEPS = 2

        config = ExperimentConfig(
            size=8,
            epochs_per_chunk=2,
            outer_loops=1,
            output_root=str(output_root),
            label="progress",
            device="cpu",
        )
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            run_experiment(config)

        self.assertIn("step 2/2 loss=", output.getvalue())

    def test_run_experiment_diagnostics_record_rows_and_interval(self):
        output_root = Path.cwd() / "outputs" / "test_runner" / uuid.uuid4().hex
        output_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))

        config = ExperimentConfig(
            size=8,
            epochs_per_chunk=2,
            outer_loops=2,
            diagnostic_interval=2,
            output_root=str(output_root),
            label="diag_interval",
            device="cpu",
        )

        result = run_experiment(config)

        with (result.run_dir / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            diagnostics_rows = list(csv.DictReader(handle))
        self.assertEqual(len(diagnostics_rows), 2)
        self.assertEqual(diagnostics_rows[0]["outer"], "1.0")
        self.assertEqual(diagnostics_rows[1]["outer"], "2.0")
        with (result.run_dir / "loss_terms.csv").open(newline="", encoding="utf-8") as handle:
            loss_rows = list(csv.DictReader(handle))
        self.assertEqual(len(loss_rows), 4)
        self.assertFalse((result.run_dir / "outer_001_summary.png").exists())
        self.assertTrue((result.run_dir / "outer_002_summary.png").exists())


if __name__ == "__main__":
    unittest.main()
