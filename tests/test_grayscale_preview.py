import json
import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from holo_opt.grayscale_preview import PREVIEW_PRESETS, build_parser, generate_grayscale_preview


class GrayscalePreviewTest(unittest.TestCase):
    def _make_input_image(self) -> tuple[Path, Path]:
        base_dir = Path.cwd() / "outputs" / "test_grayscale_preview" / uuid.uuid4().hex
        input_dir = base_dir / "inputs"
        output_dir = base_dir / "preview"
        input_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(base_dir, ignore_errors=True))

        input_path = input_dir / "demo.png"
        image = Image.new("RGB", (48, 32), color=(18, 18, 18))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 44, 28), fill=(180, 180, 180))
        draw.line((6, 26, 42, 8), fill=(255, 255, 255), width=2)
        image.save(input_path)
        return input_path, output_dir

    def test_generate_grayscale_preview_writes_single_preset_outputs(self):
        input_path, output_dir = self._make_input_image()

        original_output, source_output, processed_outputs, comparison_output, report_output = generate_grayscale_preview(
            input_path,
            size=32,
            output_dir=output_dir,
            presets=("balanced",),
        )

        self.assertTrue(original_output.exists())
        self.assertTrue(source_output.exists())
        self.assertEqual(len(processed_outputs), 1)
        self.assertTrue(processed_outputs[0].exists())
        self.assertEqual(processed_outputs[0].name, "demo_balanced_grayscale.png")
        self.assertTrue(comparison_output.exists())
        self.assertTrue(report_output.exists())
        with Image.open(processed_outputs[0]) as processed:
            self.assertLessEqual(max(processed.getdata()), 165)
        with Image.open(comparison_output) as comparison:
            self.assertGreater(comparison.width, comparison.height)
        report = json.loads(report_output.read_text(encoding="utf-8"))
        self.assertEqual(report["size"], 32)
        self.assertIn("balanced", report["presets"])
        self.assertIn("recommended_preset", report)

    def test_generate_grayscale_preview_supports_multiple_presets(self):
        input_path, output_dir = self._make_input_image()

        _, source_output, processed_outputs, comparison_output, report_output = generate_grayscale_preview(
            input_path,
            size=32,
            output_dir=output_dir,
            presets=("balanced", "detail", "budget"),
        )

        self.assertTrue(source_output.exists())
        self.assertEqual(len(processed_outputs), 3)
        self.assertEqual(
            [path.name for path in processed_outputs],
            [
                "demo_balanced_grayscale.png",
                "demo_detail_grayscale.png",
                "demo_budget_grayscale.png",
            ],
        )
        report = json.loads(report_output.read_text(encoding="utf-8"))
        self.assertEqual(set(report["presets"].keys()), {"balanced", "detail", "budget"})
        self.assertIn(report["recommended_preset"], {"balanced", "detail", "budget"})
        with Image.open(comparison_output) as comparison:
            self.assertGreater(comparison.width, 4 * 32)
            self.assertGreater(comparison.height, 32 + 40)

    def test_generate_grayscale_preview_rejects_unknown_preset(self):
        input_path, output_dir = self._make_input_image()

        with self.assertRaisesRegex(ValueError, "unknown preview preset"):
            generate_grayscale_preview(
                input_path,
                size=32,
                output_dir=output_dir,
                presets=("missing",),
            )

    def test_parser_exposes_preview_presets(self):
        args = build_parser().parse_args(["--input", "demo.png", "--preset", "balanced", "detail"])
        self.assertEqual(args.preset, ["balanced", "detail"])
        self.assertEqual(set(PREVIEW_PRESETS.keys()), {"balanced", "detail", "budget"})


if __name__ == "__main__":
    unittest.main()
