from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cmap import Color


def pyd_color_to_vispy(color: Color | None) -> str:
    """Convert a color to a hex string."""
    return color.hex if color is not None else "black"
