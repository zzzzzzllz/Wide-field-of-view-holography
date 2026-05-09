import subprocess
import sys
import unittest
from pathlib import Path

from holo_opt.cli import build_parser, config_from_args


class CliTest(unittest.TestCase):
    def test_parser_defaults_to_standard_mode(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.target_mode, "standard")
        self.assertEqual(args.size, 128)

    def test_module_help_exits_successfully(self):
        result = subprocess.run(
            [sys.executable, "-m", "holo_opt.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
            timeout=30,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--target-mode", result.stdout)

    def test_config_from_args_populates_nested_configs(self):
        args = build_parser().parse_args(
            [
                "--target-mode", "mat",
                "--target-path", "targets.mat",
                "--mat-variable", "images",
                "--size", "64",
                "--epochs-per-chunk", "2",
                "--outer-loops", "1",
                "--lr", "0.001",
                "--seed", "7",
                "--device", "cpu",
                "--output-root", "outputs/custom",
                "--label", "trial",
                "--lambda-nm", "633",
                "--px-nm", "900",
                "--py-nm", "901",
                "--guided-disabled",
                "--neff", "1.8",
                "--alpha-deg", "-10",
                "--weight-alpha", "0.25",
                "--weight-beta", "0.75",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(config.target_mode, "mat")
        self.assertEqual(config.target_path, "targets.mat")
        self.assertEqual(config.mat_variable, "images")
        self.assertEqual(config.size, 64)
        self.assertEqual(config.epochs_per_chunk, 2)
        self.assertEqual(config.outer_loops, 1)
        self.assertEqual(config.lr, 0.001)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.output_root, "outputs/custom")
        self.assertEqual(config.label, "trial")
        self.assertEqual(config.physical.lambda_nm, 633.0)
        self.assertEqual(config.physical.px_nm, 900.0)
        self.assertEqual(config.physical.py_nm, 901.0)
        self.assertFalse(config.guided_mode.enabled)
        self.assertEqual(config.guided_mode.neff, 1.8)
        self.assertEqual(config.guided_mode.alpha_deg, -10.0)
        self.assertEqual(config.weight_update.alpha, 0.25)
        self.assertEqual(config.weight_update.beta, 0.75)

    def test_parser_exposes_loss_weight_options(self):
        args = build_parser().parse_args(
            [
                "--eta-balance-weight",
                "0.2",
                "--gray-monotonic-weight",
                "0.3",
                "--phase-smoothness-weight",
                "0.004",
                "--background-weight",
                "0.1",
                "--diagnostic-interval",
                "2",
            ]
        )

        config = config_from_args(args)

        self.assertEqual(config.loss.eta_balance_weight, 0.2)
        self.assertEqual(config.loss.gray_monotonic_weight, 0.3)
        self.assertEqual(config.loss.phase_smoothness_weight, 0.004)
        self.assertEqual(config.loss.background_weight, 0.1)
        self.assertEqual(config.diagnostic_interval, 2)


if __name__ == "__main__":
    unittest.main()
