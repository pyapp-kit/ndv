"""Bio-Formats fallback reader using bffile."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import ome_types
    import xarray as xr

EXTENSIONS: set[str] = set()  # fallback — handles anything


def can_read(path: Path) -> bool:
    return True


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read a file using Bio-Formats via bffile."""
    from bffile import BioFile

    bf = BioFile(path)
    bf.open()
    da = bf.to_xarray(series=series, resolution=-1)

    # Extract channel colors from OME metadata
    ndv_display: dict[str, object] = {}
    try:
        ome = bf.ome_metadata
        img = ome.images[series]
        channels = img.pixels.channels
        colors: dict[int, str] = {}
        for i, ch in enumerate(channels):
            if ch.color is not None:
                cmap_name = _ome_color_to_cmap(ch.color)
                if cmap_name:
                    colors[i] = cmap_name
        if colors:
            ndv_display["channel_colors"] = colors
    except Exception:
        pass

    if ndv_display:
        da.attrs["ndv_display"] = ndv_display

    # Keep file handle alive
    da.attrs["_biofile"] = bf
    return da


def _ome_color_to_cmap(color: ome_types.model.Color) -> str | None:
    """Convert an ome_types Color to a cmap name."""
    try:
        r, g, b = color.as_rgb_tuple(alpha=False)  # pyright: ignore[reportAssignmentType]
    except AttributeError:
        return None

    if r > 200 and g < 50 and b < 50:
        return "red"
    if r < 50 and g > 200 and b < 50:
        return "green"
    if r < 50 and g < 50 and b > 200:
        return "blue"
    if r > 200 and g > 200 and b < 50:
        return "yellow"
    if r > 200 and g < 50 and b > 200:
        return "magenta"
    if r < 50 and g > 200 and b > 200:
        return "cyan"
    if r > 200 and g > 200 and b > 200:
        return "gray"
    return None
