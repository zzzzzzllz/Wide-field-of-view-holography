"""Preview utility for converting an RGB image into a dimmed grayscale target."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from holo_opt.config import GrayscalePreprocessConfig
from holo_opt.line_targets import generate_grayscale_target_artifacts
from holo_opt.lineart_preview import DEFAULT_INPUT_DIR, resolve_input_path


DEFAULT_OUTPUT_DIR = Path("outputs") / "grayscale_preview"
PREVIEW_PRESETS: dict[str, GrayscalePreprocessConfig] = {
    "balanced": GrayscalePreprocessConfig(),
    "detail": GrayscalePreprocessConfig(
        max_intensity=0.7,
        gamma=1.4,
        flat_region_darkening=0.68,
        detail_boost=0.35,
        tile_balance_strength=0.2,
        tile_balance_clip=1.25,
    ),
    "budget": GrayscalePreprocessConfig(
        max_intensity=0.58,
        gamma=1.8,
        flat_region_darkening=0.45,
        detail_boost=0.12,
        tile_balance_strength=0.45,
        tile_balance_clip=1.4,
    ),
}


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    return stem if stem else "grayscale"


def _array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L")


def _format_summary_lines(name: str, report: dict[str, object]) -> list[str]:
    processed = report["processed"]  # type: ignore[index]
    tile_budget = report["tile_budget"]  # type: ignore[index]
    if not isinstance(processed, dict) or not isinstance(tile_budget, dict):
        raise ValueError("preview report fields must be dictionaries")
    if name == "source grayscale":
        return [
            f"mean {float(processed['mean_intensity']):.3f}",
            f"peak {float(processed['peak_intensity']):.3f}",
            f"edge {float(processed['edge_density']):.3f}",
        ]
    return [
        f"mean {float(processed['mean_intensity']):.3f}",
        f"peak {float(processed['peak_intensity']):.3f}",
        f"budget {float(tile_budget['budget_scale_min']):.2f}-{float(tile_budget['budget_scale_max']):.2f}",
    ]


def _make_comparison_panel(
    original_rgb: Image.Image,
    source_grayscale: Image.Image,
    source_report: dict[str, object],
    processed_images: list[tuple[str, Image.Image, dict[str, object]]],
) -> Image.Image:
    labels = ["source RGB", "source grayscale", *[f"{name} processed" for name, _, _ in processed_images]]
    panel_images = [
        original_rgb.convert("RGB"),
        source_grayscale.convert("RGB"),
        *[image.convert("RGB") for _, image, _ in processed_images],
    ]
    summary_lines = [
        ["original image", "", ""],
        _format_summary_lines("source grayscale", source_report),
        *[_format_summary_lines(name, report) for name, _, report in processed_images],
    ]
    width, height = panel_images[0].size
    margin = 12
    col_gap = 8
    header_height = 28
    footer_height = 42
    canvas_width = margin * 2 + len(panel_images) * width + (len(panel_images) - 1) * col_gap
    canvas_height = margin * 2 + header_height + height + footer_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(248, 248, 246))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image, lines) in enumerate(zip(labels, panel_images, summary_lines, strict=False)):
        x = margin + index * (width + col_gap)
        canvas.paste(image, (x, margin + header_height))
        draw.text((x + 4, margin), label, fill=(24, 24, 24))
        draw.rectangle((x, margin + header_height, x + width - 1, margin + header_height + height - 1), outline=(90, 90, 90), width=1)
        text_y = margin + header_height + height + 6
        for line in lines:
            draw.text((x + 4, text_y), line, fill=(72, 72, 72))
            text_y += 12
    return canvas


def _report_for_artifacts(artifacts: object) -> dict[str, object]:
    report_rows = getattr(artifacts, "report_rows")
    source_row = next(row for row in report_rows if row["stage"] == "source")
    processed_row = next(row for row in report_rows if row["stage"] == "processed")
    tile_rows = [row for row in report_rows if row["stage"] == "tile"]
    tile_means = [float(row["mean_intensity"]) for row in tile_rows]
    tile_scales = [float(row["budget_scale"]) for row in tile_rows]
    return {
        "source": source_row,
        "processed": processed_row,
        "tile_budget": {
            "tile_count": len(tile_rows),
            "mean_intensity_min": min(tile_means) if tile_means else 0.0,
            "mean_intensity_max": max(tile_means) if tile_means else 0.0,
            "budget_scale_min": min(tile_scales) if tile_scales else 0.0,
            "budget_scale_max": max(tile_scales) if tile_scales else 0.0,
        },
    }


def _recommend_preset(report: dict[str, object]) -> str:
    preset_reports = report.get("presets")
    if not isinstance(preset_reports, dict) or not preset_reports:
        return "balanced"
    balanced_report = preset_reports.get("balanced")
    if not isinstance(balanced_report, dict):
        return "balanced"
    source = balanced_report.get("source")
    if not isinstance(source, dict):
        return "balanced"
    edge_density = float(source.get("edge_density", 0.0))
    flat_ratio = float(source.get("flat_region_ratio", 0.0))
    peak_intensity = float(source.get("peak_intensity", 0.0))
    if edge_density >= 0.2 and flat_ratio <= 0.5:
        return "detail"
    if flat_ratio >= 0.5 or peak_intensity >= 0.95:
        return "budget"
    return "balanced"


def generate_grayscale_preview(
    input_path: str | Path,
    *,
    size: int = 256,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    presets: Sequence[str] = ("balanced",),
) -> tuple[Path, Path, list[Path], Path, Path]:
    """Save source/processed grayscale previews plus a comparison panel and summary report."""
    resolved_input = resolve_input_path(input_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(resolved_input)
    original_output = output_root / f"{stem}_original.png"
    source_output = output_root / f"{stem}_source_grayscale.png"
    comparison_output = output_root / f"{stem}_comparison.png"
    report_output = output_root / f"{stem}_preview_report.json"

    shutil.copyfile(resolved_input, original_output)

    if not presets:
        raise ValueError("at least one preview preset is required")
    unknown_presets = [name for name in presets if name not in PREVIEW_PRESETS]
    if unknown_presets:
        raise ValueError(f"unknown preview preset(s): {', '.join(unknown_presets)}")

    with Image.open(resolved_input) as source:
        square_rgb = ImageOps.pad(source.convert("RGB"), (size, size), color=(0, 0, 0))
    processed_outputs: list[Path] = []
    processed_images: list[tuple[str, Image.Image, dict[str, object]]] = []
    report: dict[str, object] = {
        "input": str(resolved_input),
        "size": size,
        "presets": {},
    }

    source_preview: Image.Image | None = None
    source_report: dict[str, object] | None = None
    for preset_name in presets:
        artifacts = generate_grayscale_target_artifacts(
            resolved_input,
            size=size,
            preprocess=PREVIEW_PRESETS[preset_name],
        )
        processed_output = output_root / f"{stem}_{preset_name}_grayscale.png"
        source_image = _array_to_image(artifacts.source_grayscale)
        processed_image = _array_to_image(artifacts.processed_grayscale)
        source_preview = source_image
        preset_report = _report_for_artifacts(artifacts)
        source_report = preset_report
        processed_image.save(processed_output)
        processed_outputs.append(processed_output)
        processed_images.append((preset_name, processed_image, preset_report))
        report["presets"][preset_name] = preset_report  # type: ignore[index]

    if source_preview is None or source_report is None:
        raise RuntimeError("source preview was not generated")
    report["recommended_preset"] = _recommend_preset(report)
    source_preview.save(source_output)
    comparison = _make_comparison_panel(square_rgb, source_preview, source_report, processed_images)
    comparison.save(comparison_output)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return original_output, source_output, processed_outputs, comparison_output, report_output


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
        "--preset",
        nargs="+",
        choices=tuple(PREVIEW_PRESETS.keys()),
        default=["balanced"],
        help="One or more preview presets to render in the comparison output.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for saved original/grayscale preview images.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_output, source_output, processed_outputs, comparison_output, report_output = generate_grayscale_preview(
        args.input,
        size=args.size,
        output_dir=args.output_dir,
        presets=args.preset,
    )
    print(f"Saved original preview to {original_output}")
    print(f"Saved source grayscale preview to {source_output}")
    for processed_output in processed_outputs:
        print(f"Saved grayscale preview to {processed_output}")
    print(f"Saved comparison preview to {comparison_output}")
    print(f"Saved preview report to {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
