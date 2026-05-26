import shutil
import uuid
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from holo_opt.mask_preview import main


class MaskPreviewTest(unittest.TestCase):
    def test_mask_preview_writes_summary_and_report(self):
        output_root = Path.cwd() / "outputs" / "test_mask_preview" / uuid.uuid4().hex
        output_root.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(output_root, ignore_errors=True))
        image_path = output_root / "input.png"
        image = Image.new("RGB", (24, 24), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 20, 20), fill=(180, 180, 180))
        draw.line((4, 12, 20, 12), fill=(255, 255, 255), width=2)
        image.save(image_path)

        exit_code = main([
            "--target-mode",
            "grayscale",
            "--target-path",
            str(image_path),
            "--size",
            "8",
            "--output-dir",
            str(output_root / "preview"),
        ])

        self.assertEqual(exit_code, 0)
        self.assertTrue((output_root / "preview" / "mask_summary.png").exists())
        self.assertTrue((output_root / "preview" / "region_mask_report.csv").exists())


if __name__ == "__main__":
    unittest.main()
