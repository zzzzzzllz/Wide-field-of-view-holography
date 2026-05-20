import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image, ImageDraw

from holo_opt.line_targets import (
    build_center_weighted_line_image,
    build_dimmed_grayscale_image,
    generate_grayscale_target_artifacts,
    generate_grayscale_image_targets,
    generate_line_art_targets,
    load_rgb_image_as_square_grayscale,
    split_grayscale_image_into_channel_tiles,
    split_grayscale_image_into_channel_tiles_with_report,
)
from holo_opt.targets import (
    generate_direct_image_targets,
    generate_gray_step_targets,
    load_mat_targets,
    normalize_array,
    validate_targets,
)


class TargetsTest(unittest.TestCase):
    def _make_workspace_image_path(self, name: str) -> Path:
        temp_dir = Path.cwd() / "outputs" / "test_inputs" / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir / name

    def test_generate_standard_targets_shape_and_levels(self):
        targets = generate_gray_step_targets(n_channels=9, size=16, levels=16)
        self.assertEqual(targets.shape, (9, 16, 16))
        self.assertEqual(targets.dtype, np.float32)
        self.assertGreaterEqual(targets.min(), 0.0)
        self.assertLessEqual(targets.max(), 1.0)
        self.assertEqual(len(np.unique(targets[0])), 16)

    def test_load_rgb_image_as_square_grayscale_preserves_square_size(self):
        image_path = self._make_workspace_image_path("rect.png")
        image = Image.new("RGB", (20, 10), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((3, 2, 17, 8), outline=(255, 255, 255), width=1)
        image.save(image_path)

        grayscale = load_rgb_image_as_square_grayscale(image_path, size=16)

        self.assertEqual(grayscale.shape, (16, 16))
        self.assertGreater(float(grayscale.max()), 0.0)

    def test_build_center_weighted_line_image_brightens_line_centers(self):
        edge_mask = np.zeros((9, 9), dtype=bool)
        edge_mask[4, 4] = True

        weighted = build_center_weighted_line_image(edge_mask, line_radius=2)

        self.assertEqual(weighted.shape, (9, 9))
        self.assertGreater(float(weighted.max()), 0.0)
        self.assertGreater(float(weighted[4, 4]), float(weighted[4, 2]))

    def test_build_dimmed_grayscale_image_darkens_broad_bright_blocks(self):
        grayscale = np.zeros((16, 16), dtype=np.float32)
        grayscale[2:14, 2:14] = 1.0

        dimmed = build_dimmed_grayscale_image(grayscale)

        self.assertEqual(dimmed.shape, (16, 16))
        self.assertEqual(dimmed.dtype, np.float32)
        self.assertLessEqual(float(dimmed.max()), 0.65)
        self.assertLess(float(dimmed[8, 8]), 0.4)
        self.assertGreater(float(dimmed.max()), float(dimmed[8, 8]))

    def test_split_grayscale_image_into_channel_tiles_uses_row_major_grid(self):
        grayscale = np.zeros((9, 9), dtype=np.float32)
        for index in range(9):
            row = index // 3
            col = index % 3
            grayscale[row * 3: (row + 1) * 3, col * 3: (col + 1) * 3] = (index + 1) / 10.0

        targets = split_grayscale_image_into_channel_tiles(grayscale, expected_channels=9)

        self.assertEqual(targets.shape, (9, 9, 9))
        self.assertAlmostEqual(float(targets[0].mean()), 0.1, delta=0.01)
        self.assertAlmostEqual(float(targets[8].mean()), 0.9, delta=0.01)
        self.assertFalse(np.allclose(targets[0], targets[1]))

    def test_split_grayscale_image_into_channel_tiles_requires_square_channel_count(self):
        with self.assertRaisesRegex(ValueError, "square channel count"):
            split_grayscale_image_into_channel_tiles(np.ones((8, 8), dtype=np.float32), expected_channels=8)

    def test_split_grayscale_image_into_channel_tiles_with_report_balances_tiles(self):
        grayscale = np.zeros((12, 12), dtype=np.float32)
        grayscale[:4, :4] = 0.9
        grayscale[4:8, 4:8] = 0.2
        grayscale[8:, 8:] = 0.5

        targets, report_rows = split_grayscale_image_into_channel_tiles_with_report(
            grayscale,
            expected_channels=9,
            tile_balance_strength=0.5,
            tile_balance_clip=1.3,
        )

        self.assertEqual(targets.shape, (9, 12, 12))
        tile_rows = [row for row in report_rows if row["stage"] == "tile"]
        self.assertEqual(len(tile_rows), 9)
        self.assertLess(float(targets[0].mean()), 0.9)
        self.assertGreater(float(targets[4].mean()), 0.0)
        self.assertIn("budget_scale", tile_rows[0])

    def test_generate_line_art_targets_returns_repeated_channel_stack(self):
        image_path = self._make_workspace_image_path("cross.png")
        image = Image.new("RGB", (24, 24), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.line((4, 12, 20, 12), fill=(255, 0, 0), width=2)
        draw.line((12, 4, 12, 20), fill=(0, 255, 0), width=2)
        image.save(image_path)

        targets = generate_line_art_targets(image_path, expected_channels=9, size=16)

        self.assertEqual(targets.shape, (9, 16, 16))
        self.assertEqual(targets.dtype, np.float32)
        self.assertGreater(float(targets.max()), 0.0)
        self.assertAlmostEqual(float(targets.min()), 0.0)
        np.testing.assert_allclose(targets[0], targets[1])

    def test_generate_direct_image_targets_returns_repeated_channel_stack_without_extra_preprocessing(self):
        image_path = self._make_workspace_image_path("direct.png")
        image = Image.new("RGB", (24, 24), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 20, 20), fill=(128, 128, 128))
        image.save(image_path)

        targets = generate_direct_image_targets(image_path, expected_channels=9, size=16)

        self.assertEqual(targets.shape, (9, 16, 16))
        self.assertEqual(targets.dtype, np.float32)
        self.assertGreater(float(targets.max()), 0.0)
        self.assertLess(float(targets.max()), 1.0)
        np.testing.assert_allclose(targets[0], targets[1])

    def test_generate_grayscale_image_targets_returns_dimmed_split_channel_stack(self):
        image_path = self._make_workspace_image_path("blocks.png")
        image = Image.new("RGB", (30, 30), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        for index in range(9):
            row = index // 3
            col = index % 3
            value = 32 + index * 24
            draw.rectangle((col * 10, row * 10, col * 10 + 9, row * 10 + 9), fill=(value, value, value))
        image.save(image_path)

        targets = generate_grayscale_image_targets(image_path, expected_channels=9, size=18)

        self.assertEqual(targets.shape, (9, 18, 18))
        self.assertEqual(targets.dtype, np.float32)
        self.assertGreater(float(targets.max()), 0.0)
        self.assertLessEqual(float(targets.max()), 0.65)
        self.assertGreater(float(targets[8].mean()), float(targets[0].mean()))
        self.assertFalse(np.allclose(targets[0], targets[8]))

    def test_generate_grayscale_target_artifacts_returns_report_and_stitched_target(self):
        image_path = self._make_workspace_image_path("gradient.png")
        image = Image.new("RGB", (30, 30), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        for index in range(30):
            value = min(255, 40 + index * 6)
            draw.line((index, 0, index, 29), fill=(value, value, value), width=1)
        image.save(image_path)

        artifacts = generate_grayscale_target_artifacts(image_path, expected_channels=9, size=18)

        self.assertEqual(artifacts.targets.shape, (9, 18, 18))
        self.assertEqual(artifacts.stitched_target.shape, (18, 18))
        self.assertEqual(artifacts.source_grayscale.shape, (18, 18))
        self.assertEqual(artifacts.processed_grayscale.shape, (18, 18))
        self.assertEqual(artifacts.report_rows[0]["stage"], "source")
        self.assertEqual(artifacts.report_rows[1]["stage"], "processed")
        self.assertEqual(len([row for row in artifacts.report_rows if row["stage"] == "tile"]), 9)

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
