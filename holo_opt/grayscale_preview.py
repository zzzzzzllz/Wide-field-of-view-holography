"""Preview utility for converting an RGB image into a dimmed grayscale target."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from holo_opt.line_targets import generate_grayscale_target_artifacts
from holo_opt.lineart_preview import DEFAULT_INPUT_DIR, resolve_input_path


DEFAULT_OUTPUT_DIR = Path("outputs") / "grayscale_preview"


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    return stem if stem else "grayscale"


def generate_grayscale_preview(
    input_path: str | Path,
    *,
    size: int = 256,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save a copy of the source image and the processed grayscale image."""
    resolved_input = resolve_input_path(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(resolved_input)
    original_output = output_root / f"{stem}_original.png"
    processed_output = output_root / f"{stem}_grayscale.png"

    shutil.copyfile(resolved_input, original_output)

    artifacts = generate_grayscale_target_artifacts(resolved_input, size=size)
    image = Image.fromarray(np.uint8(np.clip(artifacts.processed_grayscale, 0.0, 1.0) * 255.0), mode="L")
    image.save(processed_output)

    return original_output, processed_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save a dimmed grayscale preview from an RGB image."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=f"Image path, or a filename inside {DEFAULT_INPUT_DIR.as_posix()}.",
    )
    parser.add_argument("--size", type=int, default=256, help="Square processing size used for grayscale preview.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved original/grayscale preview images.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_output, processed_output = generate_grayscale_preview(
        args.input,
        size=args.size,
        output_dir=args.output_dir,
    )
    print(f"Saved original preview to {original_output}")
    print(f"Saved grayscale preview to {processed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
