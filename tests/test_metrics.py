import unittest

import numpy as np

from holo_opt.metrics import channel_mse, compute_eta, evaluate_metrics, gray_level_stats


class MetricsTest(unittest.TestCase):
    def test_channel_mse_zero_for_identical_inputs(self):
        values = np.ones((2, 4, 4), dtype=np.float32)
        mse = channel_mse(values, values)
        self.assertTrue(np.allclose(mse, [0.0, 0.0]))

    def test_compute_eta_uses_positive_target_region(self):
        intensity = np.ones((1, 4, 4), dtype=np.float32)
        target = np.zeros((1, 4, 4), dtype=np.float32)
        target[:, :2, :] = 1.0
        eta = compute_eta(intensity, target)
        self.assertAlmostEqual(float(eta[0]), 0.5)

    def test_compute_eta_zero_total_intensity_returns_zero(self):
        intensity = np.zeros((1, 4, 4), dtype=np.float32)
        target = np.ones((1, 4, 4), dtype=np.float32)
        eta = compute_eta(intensity, target)
        self.assertAlmostEqual(float(eta[0]), 0.0)

    def test_gray_level_stats_reports_sixteen_levels(self):
        target = np.kron(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0, np.ones((2, 2), dtype=np.float32))
        stats = gray_level_stats(target, target, levels=16)
        self.assertEqual(len(stats["means"]), 16)
        self.assertEqual(stats["inversions"], 0)
        self.assertIn("inversion_penalty", stats)
        self.assertLess(stats["gray_level_error"], 0.1)

    def test_gray_level_stats_penalizes_collapsed_dynamic_range(self):
        target = np.kron(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0, np.ones((2, 2), dtype=np.float32))
        reconstruction = np.zeros_like(target)
        stats = gray_level_stats(reconstruction, target, levels=16)
        self.assertGreater(stats["gray_level_error"], 0.5)

    def test_gray_level_stats_requires_same_shape(self):
        reconstruction = np.ones((4, 4), dtype=np.float32)
        target = np.ones((2, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "same shape"):
            gray_level_stats(reconstruction, target, levels=16)

    def test_evaluate_metrics_returns_summary_and_rows(self):
        target = np.ones((2, 4, 4), dtype=np.float32)
        intensity = np.ones((2, 4, 4), dtype=np.float32)
        result = evaluate_metrics(intensity, target)
        self.assertEqual(len(result["rows"]), 2)
        self.assertIn("score", result["summary"])
        self.assertTrue(np.isfinite(result["summary"]["score"]))

    def test_metric_functions_require_same_3d_shape(self):
        intensities = np.ones((1, 4, 4), dtype=np.float32)
        mismatched = np.ones((2, 4, 4), dtype=np.float32)
        not_3d = np.ones((4, 4), dtype=np.float32)

        for func in (channel_mse, compute_eta, evaluate_metrics):
            with self.subTest(func=func.__name__, case="mismatched"):
                with self.assertRaisesRegex(ValueError, "same shape"):
                    func(intensities, mismatched)
            with self.subTest(func=func.__name__, case="not_3d"):
                with self.assertRaisesRegex(ValueError, "3D"):
                    func(not_3d, not_3d)
