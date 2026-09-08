"""Reader registry and dispatch loop."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from textwrap import indent, wrap
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import xarray as xr

logger = logging.getLogger("ndv.io")


@runtime_checkable
class ReaderModule(Protocol):
    """Protocol for reader modules."""

    EXTENSIONS: set[str]

    def can_read(self, path: Path) -> bool: ...
    def imread(
        self, path: Path, *, series: int = 0, level: int = 0
    ) -> xr.DataArray: ...


_READER_MODULES: list[tuple[str, str]] = [
    ("ome-zarr", "ndv.io._ome_zarr"),
    ("tifffile", "ndv.io._tiff"),
    ("nd2", "ndv.io._nd2"),
    ("readlif", "ndv.io._lif"),
    ("pylibCZIrw", "ndv.io._czi"),
    ("bffile", "ndv.io._bioformats"),
]


def _load_reader(module_name: str) -> Any:
    return importlib.import_module(module_name)


def imread(
    path: str | Path,
    *,
    series: int = 0,
    level: int = 0,
) -> xr.DataArray:
    """Read a file into an xarray.DataArray with bioimage metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    errors: dict[str, Exception] = {}

    for name, mod_name in _READER_MODULES:
        reader = _load_reader(mod_name)
        if reader.can_read(path):
            try:
                result = reader.imread(path, series=series, level=level)
                logger.info("Read %s with %s", path, name)
                return _ensure_all_coords(result)
            except ImportError as e:
                errors[name] = e
            except Exception as e:
                errors[name] = e

    raise ValueError(_format_error_message(path, errors))


def _ensure_all_coords(da: xr.DataArray) -> xr.DataArray:
    """Ensure every dim has a coordinate (ndv needs them for sliders)."""
    missing = {
        dim: list(range(da.sizes[dim])) for dim in da.dims if dim not in da.coords
    }
    if missing:
        da = da.assign_coords(missing)
    return da


def _format_error_message(path: Path, errors: dict[str, Exception]) -> str:
    lines = [f"\nCould not read {path}. Tried the following readers:", ""]
    for key, err in errors.items():
        lines.append(f"{key}:")
        wrapped = wrap(str(err), width=120)
        indented = indent("\n".join(wrapped), "    ")
        lines.append(indented)
    msg = "\n".join(lines)

    if any(isinstance(e, ImportError) for e in errors.values()):
        msg += (
            "\n\nSome readers failed due to missing packages. "
            "Install IO support with: pip install 'ndv[io-all]'"
        )
    return msg
