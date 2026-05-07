import unittest
from unittest.mock import patch

import numpy as np
import torch

from holo_opt.field import compute_intensities, fftshift2, normalize_intensities, training_loss


class FieldTest(unittest.TestCase):
    def test_fftshift2_matches_torch_fftshift_for_2d_tensor(self):
        values = torch.arange(16).reshape(4, 4)
        torch.testing.assert_close(fftshift2(values), torch.fft.fftshift(values))

    def test_compute_intensities_shape_and_finite_values(self):
        phdx = torch.zeros((8, 8), dtype=torch.float32)
        phdy = torch.zeros((8, 8), dtype=torch.float32)
        pair_mat = torch.tensor([[-2.0, 2.0], [-2.0, 1.0]], dtype=torch.float32)
        intensities = compute_intensities(phdx, phdy, pair_mat)
        self.assertEqual(tuple(intensities.shape), (2, 8, 8))
        self.assertTrue(torch.isfinite(intensities).all().item())
        self.assertGreater(float(intensities.sum().item()), 0.0)

    def test_normalize_intensities_scales_each_channel(self):
        values = torch.tensor([[[0.0, 2.0], [1.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
        normalized = normalize_intensities(values)
        self.assertAlmostEqual(float(normalized[0].max().item()), 1.0)
        self.assertAlmostEqual(float(normalized[1].max().item()), 0.0)

    def test_normalize_intensities_scales_tiny_nonzero_channel_to_one(self):
        values = torch.tensor([[[0.0, 1e-9]], [[0.0, 0.0]]], dtype=torch.float32)
        normalized = normalize_intensities(values)
        self.assertAlmostEqual(float(normalized[0].max().item()), 1.0)
        self.assertEqual(float(normalized[1].max().item()), 0.0)

    def test_training_loss_returns_finite_scalar(self):
        phdx = torch.rand((8, 8), dtype=torch.float32)
        phdy = torch.rand((8, 8), dtype=torch.float32)
        pair_mat = torch.tensor([[-2.0, 2.0], [-2.0, 1.0]], dtype=torch.float32)
        targets = torch.tensor(np.zeros((2, 8, 8), dtype=np.float32))
        weights = torch.ones(2, dtype=torch.float32)
        loss = training_loss(phdx, phdy, pair_mat, targets, weights)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_training_loss_uses_weighted_mean_squared_error_per_channel(self):
        phdx = torch.zeros((2, 2), dtype=torch.float32)
        phdy = torch.zeros((2, 2), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.ones((2, 2, 2), dtype=torch.float32)
        weights = torch.tensor([2.0, 3.0], dtype=torch.float32)

        with patch("holo_opt.field.compute_intensities", return_value=torch.zeros((2, 2, 2), dtype=torch.float32)):
            loss = training_loss(phdx, phdy, pair_mat, targets, weights)

        self.assertAlmostEqual(float(loss.item()), 5.0)
