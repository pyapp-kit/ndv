"""command-line program."""

from __future__ import annotations

import argparse
from typing import Any

from ndv.util import imshow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ndv: ndarray viewer")
    parser.add_argument("path", type=str, help="File to view")
    parser.add_argument(
        "-s", "--series", type=int, default=0, help="Series index (default: 0)"
    )
    parser.add_argument(
        "-l", "--level", type=int, default=0, help="Resolution level (default: 0)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line program."""
    from ndv import io

    args = _parse_args()
    data = io.imread(args.path, series=args.series, level=args.level)

    display_kwargs: dict[str, Any] = {}
    if ndv_display := data.attrs.pop("ndv_display", None):
        if colors := ndv_display.get("channel_colors"):
            display_kwargs["luts"] = {i: {"cmap": c} for i, c in colors.items()}

    if "C" in data.dims and data.sizes["C"] > 1:
        display_kwargs.setdefault("channel_mode", "composite")

    imshow(data, **display_kwargs)
