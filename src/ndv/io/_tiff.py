"""TIFF and OME-TIFF reader using tifffile."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

EXTENSIONS: set[str] = {".tif", ".tiff"}


def can_read(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read a TIFF or OME-TIFF file."""
    import tifffile
    import xarray as xr

    tf = tifffile.TiffFile(path)
    s = tf.series[series]

    if level > 0 and level < len(s.levels):
        s = s.levels[level]

    data = s.asarray()
    dims = tuple(s.axes.replace("S", "C"))

    coords: dict[str, list[str]] = {}
    attrs: dict[str, object] = {}
    ndv_display: dict[str, object] = {}

    # OME metadata for physical sizes and channel info
    if tf.is_ome:
        ome = tf.ome_metadata
        if ome is not None:
            _apply_ome_metadata(ome, series, dims, coords, ndv_display)

    if ndv_display:
        attrs["ndv_display"] = ndv_display

    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def _apply_ome_metadata(
    ome: str,
    series: int,
    dims: tuple[str, ...],
    coords: dict[str, list[str]],
    ndv_display: dict[str, object],
) -> None:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(ome)
    ns = {"ome": root.tag.split("}")[0].lstrip("{")} if "}" in root.tag else {}
    prefix = f"{{{ns['ome']}}}" if ns else ""

    images = root.findall(f"{prefix}Image")
    if series >= len(images):
        return
    image = images[series]
    pixels = image.find(f"{prefix}Pixels")
    if pixels is None:
        return

    # Channel names and colors
    channels = pixels.findall(f"{prefix}Channel")
    if channels and "C" in dims:
        names = []
        colors: dict[int, str] = {}
        for i, ch in enumerate(channels):
            name = ch.get("Name", f"Ch{i}")
            names.append(name)
            color = ch.get("Color")
            if color is not None:
                cmap_name = _ome_color_to_cmap(int(color))
                if cmap_name:
                    colors[i] = cmap_name
        coords["C"] = names
        if colors:
            ndv_display["channel_colors"] = colors


def _ome_color_to_cmap(color_int: int) -> str | None:
    """Convert OME integer color (RGBA packed) to a cmap name."""
    # OME colors are signed 32-bit RGBA packed integers
    color_int = color_int & 0xFFFFFFFF
    r = (color_int >> 24) & 0xFF
    g = (color_int >> 16) & 0xFF
    b = (color_int >> 8) & 0xFF

    # Map common colors to cmap names
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
