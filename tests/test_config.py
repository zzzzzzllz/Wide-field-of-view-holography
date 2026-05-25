import math
import unittest

from holo_opt.config import (
    DEFAULT_PAIR_MAT,
    ExperimentConfig,
    GuidedModeConfig,
    LossConfig,
    PhysicalConfig,
    ScoreConfig,
    WeightUpdateConfig,
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

    def test_validate_config_accepts_lineart_mode_with_target_path(self):
        config = ExperimentConfig(target_mode="lineart", target_path="outline.png")
        validate_config(config)

    def test_validate_config_accepts_grayscale_mode_with_target_path(self):
        config = ExperimentConfig(target_mode="grayscale", target_path="blocks.png")
        validate_config(config)

    def test_validate_config_accepts_grayscale_direct_mode_with_target_path(self):
        config = ExperimentConfig(target_mode="grayscale_direct", target_path="blocks.png")
        validate_config(config)

    def test_validate_config_accepts_grayscale_direct_sink_mode_with_target_path(self):
        config = ExperimentConfig(target_mode="grayscale_direct_sink", target_path="blocks.png", sink_border_ratio=0.15)
        validate_config(config)

    def test_validate_config_requires_target_path_for_lineart_mode(self):
        config = ExperimentConfig(target_mode="lineart", target_path=None)
        with self.assertRaisesRegex(ValueError, "target_path"):
            validate_config(config)

    def test_validate_config_requires_target_path_for_grayscale_mode(self):
        config = ExperimentConfig(target_mode="grayscale", target_path=None)
        with self.assertRaisesRegex(ValueError, "target_path"):
            validate_config(config)

    def test_validate_config_requires_target_path_for_grayscale_direct_mode(self):
        config = ExperimentConfig(target_mode="grayscale_direct", target_path=None)
        with self.assertRaisesRegex(ValueError, "target_path"):
            validate_config(config)

    def test_validate_config_requires_target_path_for_grayscale_direct_sink_mode(self):
        config = ExperimentConfig(target_mode="grayscale_direct_sink", target_path=None)
        with self.assertRaisesRegex(ValueError, "target_path"):
            validate_config(config)

    def test_loss_config_defaults_are_quality_safe(self):
        config = ExperimentConfig()
        data = config_to_dict(config)
        self.assertEqual(config.loss.image_weight, 1.0)
        self.assertEqual(config.loss.eta_balance_weight, 0.05)
        self.assertEqual(config.loss.gray_monotonic_weight, 0.1)
        self.assertEqual(config.loss.phase_smoothness_weight, 1e-4)
        self.assertEqual(config.loss.background_weight, 0.0)
        self.assertEqual(config.loss.local_uniformity_weight, 0.02)
        self.assertEqual(config.loss.high_frequency_weight, 0.05)
        self.assertEqual(config.diagnostic_interval, 1)
        self.assertEqual(data["diagnostic_interval"], 1)
        self.assertEqual(data["loss"]["image_weight"], 1.0)

    def test_validate_config_rejects_invalid_loss_weights(self):
        cases = (
            LossConfig(image_weight=-1.0),
            LossConfig(eta_balance_weight=math.inf),
        )
        for loss in cases:
            with self.subTest(loss=loss):
                config = ExperimentConfig(loss=loss)
                with self.assertRaisesRegex(ValueError, "loss weights"):
                    validate_config(config)

    def test_validate_config_rejects_invalid_diagnostic_interval(self):
        config = ExperimentConfig(diagnostic_interval=0)
        with self.assertRaisesRegex(ValueError, "diagnostic_interval"):
            validate_config(config)

    def test_validate_config_rejects_nonfinite_top_level_values(self):
        cases = (
            ("n_channels", 0, "n_channels"),
            ("size", 0, "size"),
            ("levels", 1, "levels"),
            ("epochs_per_chunk", 0, "epochs_per_chunk"),
            ("outer_loops", 0, "outer_loops"),
            ("lr", math.nan, "lr"),
            ("diagnostic_interval", 0, "diagnostic_interval"),
        )
        for field_name, value, pattern in cases:
            with self.subTest(field_name=field_name):
                config = ExperimentConfig()
                setattr(config, field_name, value)
                with self.assertRaisesRegex(ValueError, pattern):
                    validate_config(config)

    def test_validate_config_rejects_invalid_physical_config(self):
        cases = (
            PhysicalConfig(lambda_nm=0.0),
            PhysicalConfig(px_nm=-1.0),
            PhysicalConfig(py_nm=math.inf),
        )
        for physical in cases:
            with self.subTest(physical=physical):
                config = ExperimentConfig(physical=physical)
                with self.assertRaisesRegex(ValueError, "physical"):
                    validate_config(config)

    def test_validate_config_rejects_invalid_guided_config(self):
        cases = (
            GuidedModeConfig(neff=0.0),
            GuidedModeConfig(alpha_deg=math.nan),
        )
        for guided in cases:
            with self.subTest(guided=guided):
                config = ExperimentConfig(guided_mode=guided)
                with self.assertRaisesRegex(ValueError, "guided"):
                    validate_config(config)

    def test_validate_config_rejects_invalid_weight_update_config(self):
        cases = (
            WeightUpdateConfig(alpha=-0.1),
            WeightUpdateConfig(beta=-0.1),
            WeightUpdateConfig(epsilon=0.0),
            WeightUpdateConfig(clip_min=0.0),
            WeightUpdateConfig(clip_min=2.0, clip_max=1.0),
        )
        for weight_update in cases:
            with self.subTest(weight_update=weight_update):
                config = ExperimentConfig(weight_update=weight_update)
                with self.assertRaisesRegex(ValueError, "weight_update"):
                    validate_config(config)

    def test_validate_config_rejects_invalid_sink_border_ratio(self):
        with self.assertRaisesRegex(ValueError, "sink_border_ratio"):
            validate_config(ExperimentConfig(target_mode="grayscale_direct_sink", target_path="blocks.png", sink_border_ratio=0.5))

    def test_validate_config_rejects_invalid_score_config(self):
        cases = (
            ScoreConfig(image_weight=-1.0),
            ScoreConfig(gray_level_weight=-1.0),
            ScoreConfig(balance_weight=-1.0),
            ScoreConfig(total_efficiency_weight=-1.0),
        )
        for score in cases:
            with self.subTest(score=score):
                config = ExperimentConfig(score=score)
                with self.assertRaisesRegex(ValueError, "score"):
                    validate_config(config)

    def test_validate_config_rejects_pair_matrix_channel_mismatch(self):
        config = ExperimentConfig(n_channels=9, pair_mat=[[-2, 2]])
        with self.assertRaisesRegex(ValueError, "pair_mat length"):
            validate_config(config)

    def test_validate_config_rejects_malformed_pair_matrix_rows(self):
        base_pair_mat = [row[:] for row in DEFAULT_PAIR_MAT]
        cases = (None, 1)
        for row in cases:
            with self.subTest(row=row):
                pair_mat = [entry[:] for entry in base_pair_mat]
                pair_mat[0] = row
                config = ExperimentConfig(pair_mat=pair_mat)
                with self.assertRaisesRegex(ValueError, "pair_mat row"):
                    validate_config(config)

    def test_validate_config_rejects_malformed_pair_matrix_container(self):
        config = ExperimentConfig(pair_mat=None)
        with self.assertRaisesRegex(ValueError, "pair_mat length"):
            validate_config(config)

    def test_validate_config_accepts_tuple_pair_matrix_rows(self):
        pair_mat = [tuple(row) for row in DEFAULT_PAIR_MAT]
        validate_config(ExperimentConfig(pair_mat=pair_mat))

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
