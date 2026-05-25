import json
import shutil
import uuid
import unittest
from pathlib import Path

import numpy as np

from holo_opt.benchmark_eval import (
    evaluate_flat_region_noise,
    evaluate_reconstruction,
    evaluate_run_dir,
    flat_region_mask,
    stitch_channel_grid,
)


class BenchmarkEvalTest(unittest.TestCase):
    def test_stitch_channel_grid_requires_square_channel_count(self):
        with self.assertRaisesRegex(ValueError, "square channel count"):
            stitch_channel_grid(np.ones((2, 4, 4), dtype=np.float32))

    def test_flat_region_mask_excludes_black_background(self):
        target = np.zeros((6, 6), dtype=np.float32)
        target[1:5, 1:5] = 0.4
        mask = flat_region_mask(target, gradient_threshold=0.01, target_min=0.05)
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])

    def test_flat_region_noise_zero_for_identical_images(self):
        target = np.ones((16, 16), dtype=np.float32) * 0.4
        result = evaluate_flat_region_noise(target, target)
        self.assertAlmostEqual(result["local_variance"], 0.0, places=7)
        self.assertAlmostEqual(result["local_std"], 0.0, places=7)
        self.assertGreater(result["pixel_fraction"], 0.5)

    def test_flat_region_noise_increases_when_flat_region_is_speckled(self):
        target = np.ones((16, 16), dtype=np.float32) * 0.4
        reconstruction = target.copy()
        reconstruction[::2, ::2] += 0.2
        result = evaluate_flat_region_noise(reconstruction, target)
        self.assertGreater(result["local_variance"], 0.0)

    def test_evaluate_reconstruction_returns_stitched_and_channel_metrics(self):
        targets = np.ones((9, 8, 8), dtype=np.float32) * 0.3
        intensities = targets.copy()
        result = evaluate_reconstruction(intensities, targets)
        self.assertIn("base_metrics", result)
        self.assertIn("flat_region_noise", result)
        self.assertEqual(len(result["flat_region_noise"]["channels"]), 9)
        self.assertAlmostEqual(result["flat_region_noise"]["stitched"]["local_variance"], 0.0, places=7)
        self.assertAlmostEqual(result["flat_region_noise"]["stitched"]["local_std"], 0.0, places=7)

    def test_evaluate_run_dir_reads_exported_arrays(self):
        run_dir = Path.cwd() / "outputs" / "test_benchmark_eval" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(run_dir.parent, ignore_errors=True))
        np.savez(
            run_dir / "optimized_results.npz",
            intensities=np.ones((9, 4, 4), dtype=np.float32),
            targets=np.ones((9, 4, 4), dtype=np.float32),
        )
        with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({"target_path": "inputs/lineart_sources/benchmarks/benchmark_geometric_512.png"}, handle)

        result = evaluate_run_dir(run_dir)
        self.assertTrue(result["benchmark_match"])
        self.assertEqual(result["configured_target_path"], "inputs/lineart_sources/benchmarks/benchmark_geometric_512.png")


if __name__ == "__main__":
    unittest.main()
