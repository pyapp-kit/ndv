"""ND2 reader using the nd2 library."""

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

EXTENSIONS: set[str] = {".nd2"}


def can_read(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read an ND2 file."""
    import nd2

    f = nd2.ND2File(path)
    atexit.register(f.close)
    da = f.to_xarray(position=series, squeeze=False)

    # Extract channel colors
    ndv_display: dict[str, object] = {}
    try:
        colors: dict[int, str] = {}
        for i, ch in enumerate(f.metadata.channels):
            rgb = ch.channel.colorRGB
            cmap_name = _rgb_to_cmap(rgb)
            if cmap_name:
                colors[i] = cmap_name
        if colors:
            ndv_display["channel_colors"] = colors
    except Exception:
        pass

    if ndv_display:
        da.attrs["ndv_display"] = ndv_display

    # Keep file handle alive — dask arrays in da reference it
    da.attrs["_nd2_file"] = f
    return da


def _rgb_to_cmap(rgb: int) -> str | None:
    """Convert an RGB integer to a cmap name."""
    r = (rgb >> 16) & 0xFF
    g = (rgb >> 8) & 0xFF
    b = rgb & 0xFF

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
