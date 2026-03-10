"""CZI reader using pylibCZIrw."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

EXTENSIONS: set[str] = {".czi"}


def can_read(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read a CZI file."""
    import numpy as np
    import pylibCZIrw.czi as pyczi
    import xarray as xr

    with pyczi.open_czi(str(path)) as czidoc:
        bb = czidoc.total_bounding_box
        scenes = czidoc.scenes_bounding_rectangle
        meta = ET.fromstring(czidoc.raw_metadata)

        # Determine ROI from scene or full bounding box
        if scenes and series in scenes:
            rect = scenes[series]
            roi = (rect.x, rect.y, rect.w, rect.h)
        else:
            roi = (bb["X"][0], bb["Y"][0], bb["X"][1], bb["Y"][1])

        n_t = bb["T"][1] - bb["T"][0]
        n_z = bb["Z"][1] - bb["Z"][0]
        n_c = bb["C"][1] - bb["C"][0]

        # Read a sample plane to detect shape/dtype/RGB
        # pylibCZIrw returns (Y, X, S) where S=1 for grayscale, 3/4 for RGB
        sample = czidoc.read(roi=roi, plane={"T": 0, "Z": 0, "C": 0})
        is_rgb = sample.ndim == 3 and sample.shape[-1] in (3, 4)
        if not is_rgb and sample.ndim == 3 and sample.shape[-1] == 1:
            sample = sample[..., 0]

        if n_t <= 1 and n_z <= 1 and n_c <= 1:
            data = sample
        else:
            # Read all planes and stack into (T, C, Z, Y, X)
            planes = np.empty((n_t, n_c, n_z, *sample.shape), dtype=sample.dtype)
            for t in range(n_t):
                for c in range(n_c):
                    for z in range(n_z):
                        plane = czidoc.read(roi=roi, plane={"T": t, "Z": z, "C": c})
                        if not is_rgb and plane.shape[-1] == 1:
                            plane = plane[..., 0]
                        planes[t, c, z] = plane
            data = planes

    # Build dims — squeeze leading singletons, always keep Y, X
    dims: list[str] = []
    squeeze_axes: list[int] = []
    if n_t <= 1 and n_z <= 1 and n_c <= 1:
        # Simple image — no leading dims
        dims = ["Y", "X"]
        if is_rgb:
            dims.append("S")
    else:
        for i, (label, n) in enumerate([("T", n_t), ("C", n_c), ("Z", n_z)]):
            if n <= 1:
                squeeze_axes.append(i)
            dims.append(label)
        dims.extend(["Y", "X"])
        if is_rgb:
            dims.append("S")
        if squeeze_axes:
            data = np.squeeze(data, axis=tuple(squeeze_axes))
            dims = [d for i, d in enumerate(dims) if i not in squeeze_axes]

    # Extract metadata
    coords: dict[str, Any] = {}
    ndv_display: dict[str, object] = {}
    scales = _parse_scales(meta)
    channels = _parse_channels(meta)

    for dim_label in ("Z", "Y", "X"):
        scale = scales.get(dim_label)
        if scale is not None and dim_label in dims:
            idx = dims.index(dim_label)
            n = data.shape[idx]
            coords[dim_label] = [i * scale for i in range(n)]

    if channels and "C" in dims:
        names = [ch["name"] for ch in channels]
        coords["C"] = names
        colors: dict[int, str] = {}
        for i, ch in enumerate(channels):
            if cmap := ch.get("cmap"):
                colors[i] = cmap
        if colors:
            ndv_display["channel_colors"] = colors

    attrs: dict[str, object] = {}
    if ndv_display:
        attrs["ndv_display"] = ndv_display

    return xr.DataArray(data, dims=dims, coords=coords, attrs=attrs)


def _parse_scales(meta: ET.Element) -> dict[str, float]:
    """Extract physical pixel sizes from CZI XML metadata."""
    scales: dict[str, float] = {}
    for item in meta.iter("Distance"):
        dim_id = item.get("Id")
        val_el = item.find("Value")
        if dim_id and val_el is not None and val_el.text:
            try:
                scales[dim_id] = float(val_el.text)
            except ValueError:
                pass
    return scales


def _parse_channels(meta: ET.Element) -> list[dict[str, str]]:
    """Extract acquisition channel names and colors."""
    channels: list[dict[str, str]] = []
    for dims_el in meta.iter("Dimensions"):
        ch_el = dims_el.find("Channels")
        if ch_el is None:
            continue
        for ch in ch_el.findall("Channel"):
            if ch.get("Id") is None:
                continue
            info: dict[str, str] = {}
            name = ch.get("Name") or ch.findtext("Name") or ""
            if name:
                info["name"] = name
            color = ch.findtext("Color")
            if color:
                cmap = _hex_color_to_cmap(color)
                if cmap:
                    info["cmap"] = cmap
            channels.append(info)
        break
    return channels


def _hex_color_to_cmap(color: str) -> str | None:
    """Convert #AARRGGBB hex color to a cmap name."""
    color = color.lstrip("#")
    if len(color) == 8:
        # ARGB format
        r = int(color[2:4], 16)
        g = int(color[4:6], 16)
        b = int(color[6:8], 16)
    elif len(color) == 6:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    else:
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
