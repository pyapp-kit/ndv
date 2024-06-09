"""All the io we can think of."""

from __future__ import annotations

from textwrap import indent, wrap
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr


def imread(path: str | Path) -> Any:
    """Just read the thing already."""
    path_str = str(path)
    if path_str.endswith(".npy"):
        return np.load(path_str)
    errors = {}
    try:
        return _read_aicsimageio(path)
    except Exception as e:
        errors["aicsimageio"] = e

    raise ValueError(_format_error_message(errors))


def _format_error_message(errors: dict[str, Exception]) -> str:
    lines = ["Could not read file. Here are all the things we tried", ""]
    for _key, value in errors.items():
        lines.append(f"{_key}:")
        wrapped = wrap(str(value), width=120)
        indented = indent("\n".join(wrapped), "    ")
        lines.append(indented)
    return "\n".join(lines)


def _read_aicsimageio(path: str | Path) -> xr.DataArray:
    from aicsimageio import AICSImage

    return AICSImage(str(path)).xarray_dask_data
