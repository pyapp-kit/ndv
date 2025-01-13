"""All the io we can think of."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from textwrap import indent, wrap
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr
    import zarr


class collect_errors:
    """Store exceptions in `errors` under `key`, rather than raising."""

    def __init__(self, errors: dict, key: str):
        self.errors = errors
        self.key = key

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: type[BaseException], exc_value: BaseException, traceback: Any
    ) -> bool:
        if exc_type is not None:
            self.errors[self.key] = exc_value
        return True


def imread(path: str | Path) -> Any:
    """Just read the thing already.

    Try to read `path` and return something that ndv can open.
    """
    path_str = str(path)
    if path_str.endswith(".npy"):
        return np.load(path_str)

    errors: dict[str, Exception] = {}

    with collect_errors(errors, "aicsimageio"):
        return _read_aicsimageio(path)

    if _is_zarr_folder(path):
        with collect_errors(errors, "tensorstore-zarr"):
            return _read_tensorstore(path)
        with collect_errors(errors, "zarr"):
            return _read_zarr_python(path)

    if _is_n5_folder(path):
        with collect_errors(errors, "tensorstore-n5"):
            return _read_tensorstore(path, driver="n5")

    raise ValueError(_format_error_message(errors))


def _is_n5_folder(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and any(path.glob("attributes.json"))


def _is_zarr_folder(path: str | Path) -> bool:
    if str(path).endswith(".zarr"):
        return True
    path = Path(path)
    return path.is_dir() and any(path.glob("*.zarr"))


def _read_tensorstore(path: str | Path, driver: str = "zarr", level: int = 0) -> Any:
    import tensorstore as ts

    sub = _array_path(path, level=level)
    store = ts.open({"driver": driver, "kvstore": str(path) + sub}).result()
    print("using tensorstore")
    return store


def _format_error_message(errors: dict[str, Exception]) -> str:
    lines = ["\nCould not read file. Here's what we tried and errors we got", ""]
    for _key, err in errors.items():
        lines.append(f"{_key}:")
        wrapped = wrap(str(err), width=120)
        indented = indent("\n".join(wrapped), "    ")
        lines.append(indented)
    return "\n".join(lines)


def _read_aicsimageio(path: str | Path) -> xr.DataArray:
    from aicsimageio import AICSImage

    data = AICSImage(str(path)).xarray_dask_data
    print("using aicsimageio")
    return data


def _read_zarr_python(path: str | Path, level: int = 0) -> zarr.Array:
    import zarr

    _subpath = _array_path(path, level=level)
    z = zarr.open(str(path) + _subpath, mode="r")
    print("using zarr python")
    return z


def _array_path(path: str | Path, level: int = 0) -> str:
    import zarr

    z = zarr.open(path, mode="r")
    if isinstance(z, zarr.Array):
        return "/"
    if isinstance(z, zarr.Group):
        with suppress(TypeError):
            zattrs = json.loads(z.store.get(".zattrs"))
            if "multiscales" in zattrs:
                levels: list[str] = []
                for dset in zattrs["multiscales"][0]["datasets"]:
                    if "path" in dset:
                        levels.append(dset["path"])
                if levels:
                    return "/" + levels[level]

        arrays = list(z.array_keys())
        if arrays:
            return f"/{arrays[0]}"

    if level != 0 and levels:
        raise ValueError(
            f"Could not find a dataset with level {level} in the group. Found: {levels}"
        )
    raise ValueError("Could not find an array or multiscales information in the group.")
