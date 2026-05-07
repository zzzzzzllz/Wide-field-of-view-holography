import unittest
from unittest.mock import patch

import numpy as np

from holo_opt.targets import generate_gray_step_targets, load_mat_targets, normalize_array, validate_targets


class TargetsTest(unittest.TestCase):
    def test_generate_standard_targets_shape_and_levels(self):
        targets = generate_gray_step_targets(n_channels=9, size=16, levels=16)
        self.assertEqual(targets.shape, (9, 16, 16))
        self.assertEqual(targets.dtype, np.float32)
        self.assertGreaterEqual(targets.min(), 0.0)
        self.assertLessEqual(targets.max(), 1.0)
        self.assertEqual(len(np.unique(targets[0])), 16)

    def test_validate_targets_rejects_wrong_channel_count(self):
        targets = np.zeros((8, 16, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Expected 9 channels"):
            validate_targets(targets, expected_channels=9)

    def test_normalize_array_returns_zeros_for_constant_input(self):
        values = np.full((2, 3), 7.0, dtype=np.float32)
        normalized = normalize_array(values)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized, np.zeros((2, 3), dtype=np.float32))

    def test_load_mat_targets_accepts_channel_first_stack(self):
        arr = np.arange(9 * 4 * 4, dtype=np.float32).reshape(9, 4, 4)
        with patch("holo_opt.targets.Path.exists", return_value=True), patch(
            "holo_opt.targets.loadmat", return_value={"bw_all": arr}
        ):
            targets = load_mat_targets("targets.mat", variable="bw_all", expected_channels=9)
        self.assertEqual(targets.shape, (9, 4, 4))
        self.assertEqual(targets.dtype, np.float32)
        self.assertAlmostEqual(float(targets.min()), 0.0)
        self.assertAlmostEqual(float(targets.max()), 1.0)

    def test_load_mat_targets_accepts_channel_last_stack(self):
        arr = np.ones((4, 4, 9), dtype=np.float32)
        with patch("holo_opt.targets.Path.exists", return_value=True), patch(
            "holo_opt.targets.loadmat", return_value={"bw_all": arr}
        ):
            targets = load_mat_targets("targets_last.mat", variable="bw_all", expected_channels=9)
        self.assertEqual(targets.shape, (9, 4, 4))

    def test_load_mat_targets_rejects_ambiguous_channel_axes(self):
        arr = np.ones((9, 4, 9), dtype=np.float32)
        with patch("holo_opt.targets.Path.exists", return_value=True), patch(
            "holo_opt.targets.loadmat", return_value={"bw_all": arr}
        ):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                load_mat_targets("targets_ambiguous.mat", variable="bw_all", expected_channels=9)
