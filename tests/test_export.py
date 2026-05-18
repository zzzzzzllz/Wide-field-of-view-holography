import csv
import importlib
import json
import shutil
import uuid
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from holo_opt.config import ExperimentConfig
from holo_opt.export import export_results
from holo_opt.line_targets import GrayscaleTargetArtifacts


class ExportResultsTest(unittest.TestCase):
    def test_export_module_does_not_force_matplotlib_backend_on_import(self):
        import holo_opt.export as export_module

        with mock.patch("matplotlib.use") as use_mock:
            importlib.reload(export_module)

        use_mock.assert_not_called()

    def test_export_results_writes_required_artifacts(self):
        tmp_path = Path.cwd() / "outputs" / "test_export" / uuid.uuid4().hex
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        tmp = str(tmp_path)
        try:
            config = ExperimentConfig(size=4, output_root=tmp, label="unit")
            targets = np.ones((9, 4, 4), dtype=np.float32)
            intensities = np.ones((9, 4, 4), dtype=np.float32) * 2.0
            phdx = np.zeros((4, 4), dtype=np.float32)
            phdy = np.zeros((4, 4), dtype=np.float32)
            losses = [1.0, 0.5]
            eta_history = [[0.5] * 9, [0.6] * 9]
            weights_history = [[1.0] * 9]
            metrics = {
                "rows": [
                    {"channel": index + 1, "mse": 0.1, "eta": 0.5, "gray_level_error": 0.2, "gray_means": [0.0] * 16}
                    for index in range(9)
                ],
                "summary": {
                    "score": 1.25,
                    "image_error": 0.1,
                    "gray_level_error": 0.2,
                    "efficiency_balance_penalty": 0.3,
                    "mean_eta": 0.5,
                },
            }

            run_dir = export_results(
                config,
                targets,
                intensities,
                phdx,
                phdy,
                losses,
                eta_history,
                weights_history,
                metrics,
            )

            self.assertIsInstance(run_dir, Path)
            self.assertEqual(run_dir.parent, Path(tmp))
            self.assertTrue(run_dir.name.startswith("unit_9ch_4_"))

            required_files = {
                "config.json",
                "optimized_results.npz",
                "metrics.csv",
                "metrics.json",
                "summary.png",
                "stitched_comparison.png",
                "eta_curve.png",
                "loss_curve.png",
                "gray_levels.png",
                "phdx.csv",
                "phdy.csv",
            }
            exported_files = {path.name for path in run_dir.iterdir()}
            self.assertTrue(required_files.issubset(exported_files))

            with (run_dir / "config.json").open(encoding="utf-8") as handle:
                exported_config = json.load(handle)
            self.assertEqual(exported_config["label"], "unit")
            self.assertEqual(exported_config["output_root"], tmp)
            with (run_dir / "metrics.json").open(encoding="utf-8") as handle:
                exported_metrics = json.load(handle)
            self.assertEqual(exported_metrics, metrics)

            with np.load(run_dir / "optimized_results.npz") as data:
                self.assertEqual(
                    set(data.files),
                    {
                        "phdx",
                        "phdy",
                        "targets",
                        "intensities",
                        "pairMat",
                        "loss",
                        "eta_history",
                        "weights_history",
                    },
                )
                np.testing.assert_allclose(data["phdx"], phdx)
                np.testing.assert_allclose(data["phdy"], phdy)
                np.testing.assert_allclose(data["targets"], targets)
                np.testing.assert_allclose(data["intensities"], intensities)
                np.testing.assert_array_equal(data["pairMat"], np.asarray(config.pair_mat, dtype=np.int32))
                np.testing.assert_allclose(data["loss"], np.asarray(losses, dtype=np.float32))

            with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["channel", "mse", "eta", "gray_level_error", "score"])
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[1], ["1", "0.1", "0.5", "0.2", "1.25"])

            np.testing.assert_allclose(np.loadtxt(run_dir / "phdx.csv", delimiter=","), phdx)
            np.testing.assert_allclose(np.loadtxt(run_dir / "phdy.csv", delimiter=","), phdy)
            for image_name in ("summary.png", "stitched_comparison.png", "eta_curve.png", "loss_curve.png", "gray_levels.png"):
                self.assertGreater((run_dir / image_name).stat().st_size, 0)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_export_results_sanitizes_label_path_component(self):
        tmp_path = Path.cwd() / "outputs" / "test_export" / uuid.uuid4().hex
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        tmp = str(tmp_path)
        try:
            config = ExperimentConfig(size=2, output_root=tmp, label="../bad:name")
            metrics = {
                "rows": [
                    {"channel": index + 1, "mse": 0.0, "eta": 1.0, "gray_level_error": 0.0, "gray_means": [0.0] * 16}
                    for index in range(9)
                ],
                "summary": {"score": 0.0},
            }
            run_dir = export_results(
                config,
                np.ones((9, 2, 2), dtype=np.float32),
                np.ones((9, 2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                [1.0],
                [[1.0] * 9],
                [[1.0] * 9],
                metrics,
            )

            self.assertEqual(run_dir.parent, Path(tmp))
            self.assertTrue(run_dir.name.startswith("bad_name_9ch_2_"))
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_export_results_writes_diagnostics_when_provided(self):
        tmp_path = Path.cwd() / "outputs" / "test_export" / uuid.uuid4().hex
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        tmp = str(tmp_path)
        try:
            config = ExperimentConfig(size=4, output_root=tmp, label="diag")
            targets = np.ones((9, 4, 4), dtype=np.float32)
            intensities = np.ones((9, 4, 4), dtype=np.float32)
            phdx = np.zeros((4, 4), dtype=np.float32)
            phdy = np.zeros((4, 4), dtype=np.float32)
            metrics = {
                "rows": [
                    {
                        "channel": index + 1,
                        "mse": 0.1,
                        "eta": 0.5,
                        "gray_level_error": 0.2,
                        "gray_means": [0.0] * 16,
                    }
                    for index in range(9)
                ],
                "summary": {
                    "score": 1.0,
                    "image_error": 0.1,
                    "gray_level_error": 0.2,
                    "efficiency_balance_penalty": 0.3,
                    "mean_eta": 0.5,
                },
            }
            diagnostics = [
                {
                    "outer": 1,
                    "loss": 2.0,
                    "score": 1.0,
                    "mean_eta": 0.5,
                    "eta_balance": 0.3,
                    "image_error": 0.1,
                    "gray_level_error": 0.2,
                    "weight_min": 1.0,
                    "weight_max": 1.0,
                },
                {
                    "outer": 2,
                    "loss": 1.5,
                    "score": 1.2,
                    "mean_eta": 0.6,
                    "eta_balance": 0.2,
                    "image_error": 0.08,
                    "gray_level_error": 0.15,
                    "weight_min": 0.9,
                    "weight_max": 1.1,
                    "extra_term": 0.25,
                }
            ]
            loss_terms_history = [
                {
                    "step": 1,
                    "total": 2.0,
                    "image_mse": 1.0,
                    "eta_balance": 0.2,
                    "gray_monotonic": 0.3,
                    "phase_smoothness": 0.4,
                    "background": 0.0,
                }
            ]

            run_dir = export_results(
                config,
                targets,
                intensities,
                phdx,
                phdy,
                [2.0],
                [[0.5] * 9],
                [[1.0] * 9],
                metrics,
                diagnostics=diagnostics,
                loss_terms_history=loss_terms_history,
                outer_summaries=[(1, intensities)],
            )

            with (run_dir / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
                diagnostics_rows = list(csv.reader(handle))
            self.assertEqual(
                diagnostics_rows[0],
                [
                    "outer",
                    "loss",
                    "score",
                    "mean_eta",
                    "eta_balance",
                    "image_error",
                    "gray_level_error",
                    "weight_min",
                    "weight_max",
                    "extra_term",
                ],
            )
            self.assertEqual(diagnostics_rows[1], ["1", "2.0", "1.0", "0.5", "0.3", "0.1", "0.2", "1.0", "1.0", ""])
            self.assertEqual(
                diagnostics_rows[2],
                ["2", "1.5", "1.2", "0.6", "0.2", "0.08", "0.15", "0.9", "1.1", "0.25"],
            )

            with (run_dir / "loss_terms.csv").open(newline="", encoding="utf-8") as handle:
                loss_terms_rows = list(csv.reader(handle))
            self.assertEqual(
                loss_terms_rows[0],
                ["step", "total", "image_mse", "eta_balance", "gray_monotonic", "phase_smoothness", "background"],
            )
            self.assertEqual(loss_terms_rows[1], ["1", "2.0", "1.0", "0.2", "0.3", "0.4", "0.0"])
            self.assertGreater((run_dir / "loss_terms.png").stat().st_size, 0)
            self.assertGreater((run_dir / "outer_001_summary.png").stat().st_size, 0)
            self.assertGreater((run_dir / "outer_001_stitched_comparison.png").stat().st_size, 0)
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_export_results_writes_grayscale_preprocess_artifacts_when_provided(self):
        tmp_path = Path.cwd() / "outputs" / "test_export" / uuid.uuid4().hex
        tmp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        tmp = str(tmp_path)
        try:
            config = ExperimentConfig(size=6, output_root=tmp, label="grayscale")
            metrics = {
                "rows": [
                    {"channel": index + 1, "mse": 0.1, "eta": 0.5, "gray_level_error": 0.2, "gray_means": [0.0] * 16}
                    for index in range(9)
                ],
                "summary": {
                    "score": 1.0,
                    "image_error": 0.1,
                    "gray_level_error": 0.2,
                    "efficiency_balance_penalty": 0.3,
                    "mean_eta": 0.5,
                },
            }
            grayscale_artifacts = GrayscaleTargetArtifacts(
                source_grayscale=np.full((6, 6), 0.5, dtype=np.float32),
                processed_grayscale=np.full((6, 6), 0.3, dtype=np.float32),
                targets=np.full((9, 6, 6), 0.2, dtype=np.float32),
                stitched_target=np.full((6, 6), 0.2, dtype=np.float32),
                report_rows=[
                    {"stage": "source", "tile": 0, "mean_intensity": 0.5},
                    {"stage": "processed", "tile": 0, "mean_intensity": 0.3},
                    {"stage": "tile", "tile": 1, "mean_intensity": 0.2, "budget_scale": 1.1},
                ],
            )

            run_dir = export_results(
                config,
                np.ones((9, 6, 6), dtype=np.float32),
                np.ones((9, 6, 6), dtype=np.float32),
                np.zeros((6, 6), dtype=np.float32),
                np.zeros((6, 6), dtype=np.float32),
                [1.0],
                [[0.5] * 9],
                [[1.0] * 9],
                metrics,
                grayscale_artifacts=grayscale_artifacts,
            )

            self.assertGreater((run_dir / "preprocess_comparison.png").stat().st_size, 0)
            with (run_dir / "target_energy_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["stage", "tile", "mean_intensity", "budget_scale"])
            self.assertEqual(rows[1], ["source", "0", "0.5", ""])
            self.assertEqual(rows[3], ["tile", "1", "0.2", "1.1"])
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
