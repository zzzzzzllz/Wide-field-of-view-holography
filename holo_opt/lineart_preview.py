"""Preview utility for converting an RGB image into a line-art grayscale target."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from holo_opt.line_targets import (
    build_center_weighted_line_image,
    extract_edge_mask,
    load_rgb_image_as_square_grayscale,
)


DEFAULT_INPUT_DIR = Path("inputs") / "lineart_sources"
DEFAULT_OUTPUT_DIR = Path("outputs") / "lineart_preview"


def resolve_input_path(raw_input: str | Path) -> Path:
    """Resolve an input image path, falling back to inputs/lineart_sources."""
    candidate = Path(raw_input)
    if candidate.exists():
        return candidate
    fallback = DEFAULT_INPUT_DIR / candidate
    if fallback.exists():
        return fallback
    raise FileNotFoundError(str(candidate))


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    return stem if stem else "lineart"


def generate_lineart_preview(
    input_path: str | Path,
    *,
    size: int = 256,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save a copy of the source image and the processed line-art image."""
    resolved_input = resolve_input_path(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(resolved_input)
    original_output = output_root / f"{stem}_original.png"
    processed_output = output_root / f"{stem}_lineart.png"

    shutil.copyfile(resolved_input, original_output)

    grayscale = load_rgb_image_as_square_grayscale(resolved_input, size=size)
    edge_mask = extract_edge_mask(grayscale)
    line_image = build_center_weighted_line_image(edge_mask, line_radius=max(2, size // 64))
    image = Image.fromarray(np.uint8(np.clip(line_image, 0.0, 1.0) * 255.0), mode="L")
    image.save(processed_output)

    return original_output, processed_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and save a line-art preview from an RGB image."
    )
    parser.add_argument("--input", required=True, help="Image path, or a filename inside inputs/lineart_sources.")
    parser.add_argument("--size", type=int, default=256, help="Square processing size used for line extraction.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved original/lineart preview images.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_output, processed_output = generate_lineart_preview(
        args.input,
        size=args.size,
        output_dir=args.output_dir,
    )
    print(f"Saved original preview to {original_output}")
    print(f"Saved lineart preview to {processed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
