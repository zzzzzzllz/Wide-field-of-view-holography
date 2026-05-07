import shutil
import uuid
import unittest
from pathlib import Path

import numpy as np
import torch

from holo_opt.config import ExperimentConfig, ScoreConfig
from holo_opt.runner import compute_score, load_targets_for_config, resolve_device, run_experiment


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


if __name__ == "__main__":
    unittest.main()
