"""OME-Zarr reader using yaozarrs + tensorstore."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

EXTENSIONS: set[str] = {".zarr"}
logger = logging.getLogger(__name__)


def can_read(path: Path) -> bool:
    """Check if path is an OME-Zarr or plain Zarr dataset."""
    p = str(path)
    if p.endswith(".zarr"):
        return True
    # directory containing zarr structure
    if path.is_dir():
        return (path / ".zattrs").exists() or (path / "zarr.json").exists()
    return False


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read an OME-Zarr dataset."""
    import tensorstore as ts
    import xarray as xr
    import yaozarrs

    group = yaozarrs.open_group(str(path))
    subpath, ome_meta = _navigate_zarr(group, series=series, level=level)

    # Open the array via tensorstore for lazy access
    full_path = str(path) + subpath if subpath != "/" else str(path)
    store = ts.open({"driver": "zarr", "kvstore": full_path}).result()

    dims, coords, ndv_display = _extract_ome_metadata(ome_meta, series, level)

    # Fall back to generic dims if OME metadata didn't provide them
    if not dims:
        dims = tuple(f"dim_{i}" for i in range(store.ndim))

    attrs: dict[str, object] = {}
    if ndv_display:
        attrs["ndv_display"] = ndv_display

    return xr.DataArray(store, dims=dims, coords=coords, attrs=attrs)


def _navigate_zarr(
    group: Any, *, series: int = 0, level: int = 0
) -> tuple[str, dict[str, Any] | None]:
    """Navigate zarr group to find the target array and OME metadata."""
    try:
        ome = group.ome_metadata()
    except Exception:
        ome = None

    # Bioformats2Raw layout: nested series groups
    if ome is None:
        try:
            sub_group = group.open_group(str(series))
            ome = sub_group.ome_metadata()
            prefix = f"/{series}"
        except Exception:
            prefix = ""
    else:
        prefix = ""

    if ome is not None:
        multiscales = ome.get("multiscales", [])
        if multiscales:
            ms = multiscales[series] if series < len(multiscales) else multiscales[0]
            datasets = ms.get("datasets", [])
            if level < len(datasets):
                ds_path = datasets[level].get("path", "0")
            elif datasets:
                ds_path = datasets[0].get("path", "0")
            else:
                ds_path = "0"
            return f"{prefix}/{ds_path}", ome

    # No OME metadata — try to find any array
    return f"{prefix}/{level}", None


def _extract_ome_metadata(
    ome: dict[str, Any] | None, series: int, level: int
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, object]]:
    """Extract dims, coords, and display hints from OME metadata."""
    dims: tuple[str, ...] = ()
    coords: dict[str, Any] = {}
    ndv_display: dict[str, object] = {}

    if ome is None:
        return dims, coords, ndv_display

    multiscales = ome.get("multiscales", [])
    if not multiscales:
        return dims, coords, ndv_display

    ms = multiscales[series] if series < len(multiscales) else multiscales[0]

    # Extract axis names
    axes = ms.get("axes", [])
    if axes:
        dims = tuple(
            ax["name"].upper() if len(ax["name"]) == 1 else ax["name"] for ax in axes
        )

    # Extract channel colors from omero metadata
    omero = ome.get("omero", {})
    if omero:
        channels = omero.get("channels", [])
        colors: dict[int, str] = {}
        for i, ch in enumerate(channels):
            color = ch.get("color")
            if color:
                cmap_name = _hex_color_to_cmap(color)
                if cmap_name:
                    colors[i] = cmap_name
        if colors:
            ndv_display["channel_colors"] = colors

    return dims, coords, ndv_display


def _hex_color_to_cmap(color: str) -> str | None:
    """Convert hex color string to a cmap name."""
    color = color.lstrip("#").upper()
    if len(color) < 6:
        return None

    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)

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
