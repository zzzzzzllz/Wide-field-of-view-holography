import unittest

import numpy as np

from holo_opt.config import RegionMaskConfig
from holo_opt.region_masks import generate_region_masks


class RegionMasksTest(unittest.TestCase):
    def test_generate_region_masks_returns_expected_shapes_and_rows(self):
        targets = np.zeros((2, 8, 8), dtype=np.float32)
        targets[0, 2:6, 2:6] = 0.5
        targets[1, 1:7, 1:7] = np.linspace(0.0, 1.0, 36, dtype=np.float32).reshape(6, 6)

        masks = generate_region_masks(targets, RegionMaskConfig(enabled=True))

        for values in (masks.edge, masks.signal, masks.flat, masks.dark, masks.relaxed):
            self.assertEqual(values.shape, targets.shape)
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)
        self.assertEqual(len(masks.report_rows), 2)
        self.assertIn("edge_fraction", masks.report_rows[0])
        self.assertIn("edge_threshold", masks.report_rows[0])

    def test_dark_pixels_do_not_overlap_signal_edge_or_flat_masks(self):
        targets = np.zeros((1, 8, 8), dtype=np.float32)
        targets[0, 3:5, 3:5] = 1.0

        masks = generate_region_masks(
            targets,
            RegionMaskConfig(enabled=True, dark_threshold=0.1, signal_threshold=0.2, edge_dilation=1),
        )

        dark = masks.dark.astype(bool)
        self.assertFalse(np.any(masks.edge.astype(bool) & dark))
        self.assertFalse(np.any(masks.signal.astype(bool) & dark))
        self.assertFalse(np.any(masks.flat.astype(bool) & dark))

    def test_edge_dilation_increases_or_preserves_edge_fraction(self):
        targets = np.zeros((1, 12, 12), dtype=np.float32)
        targets[0, :, 6:] = 1.0

        no_dilation = generate_region_masks(targets, RegionMaskConfig(enabled=True, edge_dilation=0))
        dilation = generate_region_masks(targets, RegionMaskConfig(enabled=True, edge_dilation=2))

        self.assertGreaterEqual(float(dilation.edge.sum()), float(no_dilation.edge.sum()))

    def test_generate_region_masks_rejects_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "targets must have shape"):
            generate_region_masks(np.zeros((8, 8), dtype=np.float32), RegionMaskConfig(enabled=True))


if __name__ == "__main__":
    unittest.main()
