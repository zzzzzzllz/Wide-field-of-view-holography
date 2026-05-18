import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image, ImageDraw

from holo_opt.grayscale_preview import generate_grayscale_preview


class GrayscalePreviewTest(unittest.TestCase):
    def test_generate_grayscale_preview_writes_original_and_processed_images(self):
        base_dir = Path.cwd() / "outputs" / "test_grayscale_preview" / uuid.uuid4().hex
        input_dir = base_dir / "inputs"
        output_dir = base_dir / "preview"
        input_dir.mkdir(parents=True, exist_ok=False)
        self.addCleanup(lambda: shutil.rmtree(base_dir, ignore_errors=True))

        input_path = input_dir / "demo.png"
        image = Image.new("RGB", (32, 24), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, 28, 20), fill=(255, 255, 255))
        image.save(input_path)

        original_output, processed_output = generate_grayscale_preview(
            input_path,
            size=32,
            output_dir=output_dir,
        )

        self.assertTrue(original_output.exists())
        self.assertTrue(processed_output.exists())
        self.assertEqual(original_output.name, "demo_original.png")
        self.assertEqual(processed_output.name, "demo_grayscale.png")
        self.assertGreater(original_output.stat().st_size, 0)
        self.assertGreater(processed_output.stat().st_size, 0)
        with Image.open(processed_output) as processed:
            self.assertLessEqual(max(processed.getdata()), 165)


if __name__ == "__main__":
    unittest.main()
