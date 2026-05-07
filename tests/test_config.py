import unittest

from holo_opt.config import (
    DEFAULT_PAIR_MAT,
    ExperimentConfig,
    GuidedModeConfig,
    PhysicalConfig,
    config_to_dict,
    validate_config,
)


class ConfigTest(unittest.TestCase):
    def test_default_pair_matrix_matches_spec(self):
        self.assertEqual(
            DEFAULT_PAIR_MAT,
            [
                [-2, 2], [-2, 1], [-2, 0],
                [-3, 2], [-3, 1], [-3, 0],
                [-4, 2], [-4, 1], [-4, 0],
            ],
        )

    def test_physical_defaults_match_metasurface_parameters(self):
        physical = PhysicalConfig()
        guided = GuidedModeConfig()
        self.assertEqual(physical.lambda_nm, 532.0)
        self.assertEqual(physical.px_nm, 830.0)
        self.assertEqual(physical.py_nm, 830.0)
        self.assertEqual(guided.neff, 2.05)
        self.assertEqual(guided.alpha_deg, -16.7)

    def test_validate_config_accepts_default_quick_run(self):
        config = ExperimentConfig(size=128, epochs_per_chunk=10, outer_loops=2)
        validate_config(config)
        data = config_to_dict(config)
        self.assertEqual(data["n_channels"], 9)
        self.assertEqual(data["pair_mat"][0], [-2, 2])
        self.assertEqual(data["physical"]["lambda_nm"], 532.0)

    def test_validate_config_rejects_pair_matrix_channel_mismatch(self):
        config = ExperimentConfig(n_channels=9, pair_mat=[[-2, 2]])
        with self.assertRaisesRegex(ValueError, "pair_mat length"):
            validate_config(config)

    def test_validate_config_rejects_noninteger_pair_matrix_values(self):
        cases = (
            [[-2.5, 2], [-2, 1], [-2, 0], [-3, 2], [-3, 1], [-3, 0], [-4, 2], [-4, 1], [-4, 0]],
            [["x", "y"], [-2, 1], [-2, 0], [-3, 2], [-3, 1], [-3, 0], [-4, 2], [-4, 1], [-4, 0]],
        )
        for pair_mat in cases:
            with self.subTest(pair_mat=pair_mat):
                config = ExperimentConfig(pair_mat=pair_mat)
                with self.assertRaisesRegex(ValueError, "pair_mat values"):
                    validate_config(config)

    def test_default_pair_matrix_instances_are_independent(self):
        first = ExperimentConfig()
        first.pair_mat[0][0] = 99
        second = ExperimentConfig()
        self.assertEqual(second.pair_mat[0][0], -2)

    def test_validate_config_rejects_nonpositive_training_values(self):
        config = ExperimentConfig(size=0)
        with self.assertRaisesRegex(ValueError, "size"):
            validate_config(config)
