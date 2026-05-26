"""Preview target-derived region masks without running optimization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from holo_opt.cli import build_parser, config_from_args
from holo_opt.config import validate_config
from holo_opt.export import _plot_region_masks, _write_rows_csv
from holo_opt.region_masks import generate_region_masks
from holo_opt.runner import load_targets_for_config


def build_preview_parser():
    parser = build_parser()
    parser.description = "Preview target-derived region masks."
    parser.add_argument("--output-dir", default="outputs/mask_preview")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_preview_parser().parse_args(argv)
    config = config_from_args(args)
    config.region_mask.enabled = True
    validate_config(config)
    targets = load_targets_for_config(config)
    masks = generate_region_masks(targets, config.region_mask)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows_csv(output_dir / "region_mask_report.csv", masks.report_rows)
    _plot_region_masks(output_dir / "mask_summary.png", np.asarray(targets, dtype=np.float32), masks)
    print(f"Saved mask preview to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
