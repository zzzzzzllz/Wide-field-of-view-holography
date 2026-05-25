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
    high_frequency_loss,
    local_uniformity_loss,
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
            "local_uniformity_weight": 0.02,
            "high_frequency_weight": 0.05,
        }

        terms = compute_loss_terms(phdx, phdy, pair_mat, targets, weights, loss_weights)

        self.assertEqual(
            set(terms),
            {"total", "image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background", "local_uniformity", "high_frequency"},
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

    def test_local_uniformity_loss_returns_zero_without_object_region(self):
        reconstruction = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        targets = torch.zeros((1, 2, 2), dtype=torch.float32)

        self.assertEqual(float(local_uniformity_loss(reconstruction, targets)), 0.0)

    def test_local_uniformity_loss_penalizes_noisy_flat_region_more_than_uniform_region(self):
        targets = torch.ones((1, 3, 3), dtype=torch.float32)
        uniform = torch.full((1, 3, 3), 0.5, dtype=torch.float32)
        noisy = uniform.clone()
        noisy[0, 1, 1] = 1.0

        self.assertGreater(float(local_uniformity_loss(noisy, targets)), float(local_uniformity_loss(uniform, targets)))

    def test_local_uniformity_loss_downweights_target_edges(self):
        reconstruction = torch.full((1, 3, 3), 0.5, dtype=torch.float32)
        reconstruction[0, 1, 1] = 1.0
        flat_target = torch.ones((1, 3, 3), dtype=torch.float32)
        edge_target = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]],
            dtype=torch.float32,
        )

        self.assertLess(float(local_uniformity_loss(reconstruction, edge_target)), float(local_uniformity_loss(reconstruction, flat_target)))

    def test_high_frequency_loss_penalizes_checkerboard_more_than_uniform_region(self):
        targets = torch.ones((1, 4, 4), dtype=torch.float32)
        uniform = torch.full((1, 4, 4), 0.5, dtype=torch.float32)
        checkerboard = torch.tensor(
            [[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )

        self.assertGreater(float(high_frequency_loss(checkerboard, targets)), float(high_frequency_loss(uniform, targets)))

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
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0, "local_uniformity_weight": 0.0, "high_frequency_weight": 0.0},
            )
            with_background = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 2.0, "local_uniformity_weight": 0.0, "high_frequency_weight": 0.0},
            )

        self.assertGreater(float(with_background["background"]), 0.0)
        self.assertGreater(float(with_background["total"]), float(without_background["total"]))

    def test_compute_loss_terms_local_uniformity_weight_affects_total(self):
        phdx = torch.zeros((3, 3), dtype=torch.float32)
        phdy = torch.zeros((3, 3), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.ones((1, 3, 3), dtype=torch.float32)
        weights = torch.ones(1, dtype=torch.float32)
        intensities = torch.full((1, 3, 3), 0.5, dtype=torch.float32)
        intensities[0, 1, 1] = 1.0

        with patch("holo_opt.field.compute_intensities", return_value=intensities):
            without_uniformity = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0, "local_uniformity_weight": 0.0, "high_frequency_weight": 0.0},
            )
            with_uniformity = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0, "local_uniformity_weight": 2.0, "high_frequency_weight": 0.0},
            )

        self.assertGreater(float(with_uniformity["local_uniformity"]), 0.0)
        self.assertGreater(float(with_uniformity["total"]), float(without_uniformity["total"]))

    def test_compute_loss_terms_high_frequency_weight_affects_total(self):
        phdx = torch.zeros((4, 4), dtype=torch.float32)
        phdy = torch.zeros((4, 4), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        targets = torch.ones((1, 4, 4), dtype=torch.float32)
        weights = torch.ones(1, dtype=torch.float32)
        intensities = torch.tensor(
            [[[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )

        with patch("holo_opt.field.compute_intensities", return_value=intensities):
            without_high_frequency = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0, "local_uniformity_weight": 0.0, "high_frequency_weight": 0.0},
            )
            with_high_frequency = compute_loss_terms(
                phdx,
                phdy,
                pair_mat,
                targets,
                weights,
                {"eta_balance_weight": 0.0, "gray_monotonic_weight": 0.0, "background_weight": 0.0, "local_uniformity_weight": 0.0, "high_frequency_weight": 0.5},
            )

        self.assertGreater(float(with_high_frequency["high_frequency"]), 0.0)
        self.assertGreater(float(with_high_frequency["total"]), float(without_high_frequency["total"]))

    def test_training_loss_accepts_optional_loss_weights(self):
        phdx = torch.zeros((2, 2), dtype=torch.float32)
        phdy = torch.zeros((2, 2), dtype=torch.float32)
        pair_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        targets = torch.ones((2, 2, 2), dtype=torch.float32)
        weights = torch.ones(2, dtype=torch.float32)

        loss = training_loss(phdx, phdy, pair_mat, targets, weights, {"image_weight": 1.0})

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
