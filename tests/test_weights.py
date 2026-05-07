import unittest

import numpy as np

from holo_opt.weights import update_weights


class WeightsTest(unittest.TestCase):
    def test_update_weights_preserves_shape_finite_and_normalized_mean(self):
        old = np.ones(3, dtype=np.float32)
        eta = np.asarray([0.1, 0.2, 0.4], dtype=np.float32)
        error = np.asarray([0.3, 0.2, 0.1], dtype=np.float32)

        updated = update_weights(old, eta, error, alpha=0.5, beta=0.5)

        self.assertEqual(updated.shape, old.shape)
        self.assertEqual(updated.dtype, np.float32)
        self.assertTrue(np.isfinite(updated).all())
        self.assertAlmostEqual(float(np.mean(updated)), 1.0, places=6)

    def test_low_eta_high_error_channel_receives_larger_weight(self):
        old = np.ones(3, dtype=np.float32)
        eta = np.asarray([0.1, 0.3, 0.4], dtype=np.float32)
        error = np.asarray([0.4, 0.2, 0.1], dtype=np.float32)

        updated = update_weights(old, eta, error, alpha=1.0, beta=1.0)

        self.assertGreater(float(updated[0]), float(updated[1]))
        self.assertGreater(float(updated[0]), float(updated[2]))

    def test_extreme_low_eta_high_error_is_clipped(self):
        old = np.ones(3, dtype=np.float32)
        eta = np.asarray([1e-9, 1.0, 1.0], dtype=np.float32)
        error = np.asarray([1000.0, 0.001, 0.001], dtype=np.float32)

        updated = update_weights(old, eta, error, alpha=2.0, beta=2.0, clip_min=0.5, clip_max=5.0)

        self.assertLessEqual(float(np.max(updated)), 5.0)
        self.assertGreaterEqual(float(np.min(updated)), 0.5)
        self.assertTrue(np.isfinite(updated).all())

    def test_final_normalization_keeps_upper_clip_bound(self):
        old = np.ones(10, dtype=np.float32)
        eta = np.asarray([1e-9] + [1.0] * 9, dtype=np.float32)
        error = np.asarray([1000.0] + [0.001] * 9, dtype=np.float32)

        updated = update_weights(old, eta, error, alpha=2.0, beta=2.0, clip_min=0.5, clip_max=5.0)

        self.assertLessEqual(float(np.max(updated)), 5.0)
        self.assertAlmostEqual(float(np.mean(updated)), 1.0, places=6)

    def test_update_weights_validates_inputs(self):
        old = np.ones(3, dtype=np.float32)
        eta = np.ones(2, dtype=np.float32)
        error = np.ones(3, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "same shape"):
            update_weights(old, eta, error)
        with self.assertRaisesRegex(ValueError, "1D"):
            update_weights(np.ones((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            update_weights(old, old, error, alpha=-0.1)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            update_weights(old, old, error, beta=-0.1)
        with self.assertRaisesRegex(ValueError, "clip_min"):
            update_weights(old, old, error, clip_min=0.0)
        with self.assertRaisesRegex(ValueError, "clip_min"):
            update_weights(old, old, error, clip_min=2.0, clip_max=1.0)


if __name__ == "__main__":
    unittest.main()
