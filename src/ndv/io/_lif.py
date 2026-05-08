"""LIF reader using readlif."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import xarray as xr

EXTENSIONS: set[str] = {".lif"}

# readlif internal dim index -> named dimension
_IDX_TO_DIM: dict[int, str] = {1: "X", 2: "Y", 3: "Z", 4: "T", 5: "M"}


def can_read(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def imread(path: Path, *, series: int = 0, level: int = 0) -> xr.DataArray:
    """Read a LIF file."""
    import numpy as np
    import xarray as xr
    from readlif.reader import LifFile

    lif = LifFile(str(path))
    images = list(lif.get_iter_image())
    if series >= len(images):
        raise IndexError(f"Series {series} not found (file has {len(images)} scenes)")
    img = images[series]

    display = img.display_dims  # e.g. (1, 3) for XZ
    dims_n = img.dims_n  # e.g. {1: 128, 3: 128, 4: 20}
    n_channels = img.channels

    # Separate non-display dims (ordered by index)
    nondisplay = {k: v for k, v in dims_n.items() if k not in display}
    nd_keys = sorted(nondisplay.keys())
    nd_sizes = [nondisplay[k] for k in nd_keys]

    # Read all planes: iterate (non-display-dims x channels)
    sample = np.asarray(img.get_plane(c=0))
    plane_shape = sample.shape

    total_planes = n_channels * (max(1, int(np.prod(nd_sizes))) if nd_sizes else 1)
    all_planes = np.empty((total_planes, *plane_shape), dtype=sample.dtype)
    idx = 0

    nd_ranges: list[Sequence] = [range(s) for s in nd_sizes] if nd_sizes else [()]
    for nd_vals in itertools.product(*nd_ranges):
        req = dict(zip(nd_keys, nd_vals, strict=False)) if nd_keys else {}
        for c in range(n_channels):
            plane = np.asarray(img.get_plane(c=c, requested_dims=req))
            all_planes[idx] = plane
            idx += 1

    # Build shape: (*non_display, C, display_h, display_w)
    shape = (*nd_sizes, n_channels, *plane_shape)
    data = all_planes.reshape(shape)

    # Build dim labels
    # numpy plane shape is (dims_n[display[1]], dims_n[display[0]])
    dim_labels: list[str] = []
    for k in nd_keys:
        dim_labels.append(_IDX_TO_DIM.get(k, f"dim_{k}"))
    dim_labels.append("C")
    dim_labels.append(_IDX_TO_DIM.get(display[1], f"dim_{display[1]}"))
    dim_labels.append(_IDX_TO_DIM.get(display[0], f"dim_{display[0]}"))

    # Squeeze singleton dims
    squeeze_axes = [i for i, s in enumerate(data.shape) if s == 1]
    if squeeze_axes:
        data = np.squeeze(data, axis=tuple(squeeze_axes))
        dim_labels = [d for i, d in enumerate(dim_labels) if i not in squeeze_axes]

    # Physical scales from readlif
    coords: dict[str, Any] = {}
    scale_n = img.scale_n  # {internal_idx: scale_in_microns_per_px}
    for internal_idx, scale in scale_n.items():
        dim_name = _IDX_TO_DIM.get(internal_idx)
        if dim_name and dim_name in dim_labels:
            ax = dim_labels.index(dim_name)
            n = data.shape[ax]
            coords[dim_name] = [i / scale for i in range(n)]

    return xr.DataArray(data, dims=dim_labels, coords=coords)
