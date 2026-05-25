import unittest
from unittest.mock import patch

import numpy as np
import torch

from holo_opt.field import (
    background_loss,
    channel_energy_balance_loss,
    compute_intensities,
    compute_loss_terms,
    fftshift2,
    gray_monotonic_loss,
    normalize_intensities,
    phase_smoothness_loss,
    training_loss,
)


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

    def test_compute_loss_terms_matches_reconstruction_energy_to_target_energy_for_image_mse(self):
        phdx = torch.zeros((2, 2), dtype=torch.float32)
        phdy = torch.zeros((2, 2), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
        intensities = torch.tensor([[[0.0, 2.0], [6.0, 2.0]]], dtype=torch.float32)
        weights = torch.ones(1, dtype=torch.float32)

        with patch("holo_opt.field.compute_intensities", return_value=intensities):
            terms = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "phase_smoothness_weight": 0.0},
            )

        expected_reconstruction = intensities / intensities.sum(dim=(-2, -1), keepdim=True) * targets.sum()
        expected_mse = torch.mean((expected_reconstruction - targets) ** 2)
        self.assertAlmostEqual(float(terms["image_mse"].item()), float(expected_mse.item()))

    def test_compute_loss_terms_returns_finite_terms(self):
        phdx = torch.zeros((4, 4), dtype=torch.float32)
        phdy = torch.zeros((4, 4), dtype=torch.float32)
        pair_mat = torch.tensor([[-2.0, 2.0], [-2.0, 1.0]], dtype=torch.float32)
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

    def test_phase_smoothness_wraps_periodic_boundaries(self):
        phase = torch.tensor([[0.0, 2.0 * torch.pi]], dtype=torch.float32)
        self.assertLess(float(phase_smoothness_loss(phase, torch.zeros_like(phase))), 1e-10)

    def test_gray_monotonic_loss_penalizes_inversions(self):
        target = torch.linspace(0.0, 1.0, 16, dtype=torch.float32).reshape(1, 4, 4)
        good = target.clone()
        bad = 1.0 - target

        self.assertGreater(float(gray_monotonic_loss(bad, target)), float(gray_monotonic_loss(good, target)))

    def test_gray_monotonic_loss_averages_per_channel_valid_level_pairs(self):
        targets = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.5], [1.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        reconstruction = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.8], [0.4, 0.4]],
            ],
            dtype=torch.float32,
        )

        self.assertAlmostEqual(float(gray_monotonic_loss(reconstruction, targets, levels=3)), 0.2)

    def test_channel_energy_balance_uses_useful_region_efficiency_when_targets_provided(self):
        intensities = torch.tensor(
            [
                [[9.0, 1.0], [0.0, 0.0]],
                [[1.0, 9.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        self.assertGreater(float(channel_energy_balance_loss(intensities, targets)), 0.0)

    def test_background_loss_penalizes_dark_target_pixels(self):
        reconstruction = torch.tensor([[0.0, 0.5], [1.0, 0.25]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

        self.assertGreater(float(background_loss(reconstruction, targets)), 0.0)

    def test_background_loss_returns_zero_without_dark_pixels(self):
        reconstruction = torch.ones((2, 2), dtype=torch.float32)
        targets = torch.ones((2, 2), dtype=torch.float32)

        self.assertEqual(float(background_loss(reconstruction, targets)), 0.0)

    def test_compute_loss_terms_background_weight_affects_total(self):
        phdx = torch.zeros((2, 2), dtype=torch.float32)
        phdy = torch.zeros((2, 2), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        weights = torch.ones(1, dtype=torch.float32)
        intensities = torch.ones((1, 2, 2), dtype=torch.float32)

        with patch("holo_opt.field.compute_intensities", return_value=intensities):
            without_background = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0},
            )
            with_background = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 2.0},
            )

        self.assertGreater(float(with_background["background"]), 0.0)
        self.assertGreater(float(with_background["total"]), float(without_background["total"]))

    def test_training_loss_accepts_optional_loss_weights(self):
        phdx = torch.zeros((2, 2), dtype=torch.float32)
        phdy = torch.zeros((2, 2), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.ones((2, 2, 2), dtype=torch.float32)
        weights = torch.ones(2, dtype=torch.float32)

        loss = training_loss(phdx, phdy, pair_mat, targets, weights, {"image_weight": 1.0})

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
