import csv
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from holo_opt.batch import build_parser, config_from_batch_args, run_seed_batch
from holo_opt.runner import ExperimentResult


class BatchTest(unittest.TestCase):
    def test_parser_accepts_multiple_seeds(self):
        args = build_parser().parse_args(["--seeds", "1", "2", "3"])

        self.assertEqual(args.seeds, [1, 2, 3])

    def test_config_from_batch_args_keeps_base_seed_when_seeds_omitted(self):
        args = build_parser().parse_args(["--seed", "17", "--label", "trial"])
        config, seeds = config_from_batch_args(args)

        self.assertEqual(config.seed, 17)
        self.assertEqual(config.label, "trial")
        self.assertEqual(seeds, [17])

    def test_run_seed_batch_writes_summary_and_marks_lowest_score_best(self):
        output_root = Path.cwd() / "outputs" / "test_batch" / uuid.uuid4().hex
        output_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))
        args = build_parser().parse_args(
            [
                "--size", "4",
                "--epochs-per-chunk", "1",
                "--outer-loops", "1",
                "--device", "cpu",
                "--output-root", str(output_root),
                "--label", "batch_unit",
                "--seeds", "11", "12",
            ]
        )
        config, seeds = config_from_batch_args(args)
        scores = {11: 0.3, 12: 0.1}

        def fake_run_experiment(run_config):
            run_dir = Path(run_config.output_root) / f"{run_config.label}_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            score = scores[run_config.seed]
            return ExperimentResult(
                run_dir=run_dir,
                final_intensities=None,
                final_metrics={
                    "summary": {
                        "score": score,
                        "image_error": score + 1.0,
                        "gray_level_error": score + 2.0,
                        "efficiency_balance_penalty": score + 3.0,
                        "mean_eta": score + 4.0,
                    }
                },
            )

        with patch("holo_opt.batch.run_experiment", side_effect=fake_run_experiment):
            batch_result = run_seed_batch(config, seeds)

        self.assertEqual(batch_result.best_seed, 12)
        self.assertTrue(batch_result.summary_path.exists())
        with batch_result.summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["seed"] for row in rows], ["11", "12"])
        self.assertEqual([row["best"] for row in rows], ["", "1"])
        self.assertTrue(rows[0]["run_dir"].endswith("batch_unit_seed11_run"))
        self.assertTrue(rows[1]["run_dir"].endswith("batch_unit_seed12_run"))


if __name__ == "__main__":
    unittest.main()
