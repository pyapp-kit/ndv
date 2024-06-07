"""In this module, we provide built-in support for many array types."""

from __future__ import annotations

import sys
import warnings
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Protocol, TypeGuard

    import dask.array as da
    import numpy.typing as npt
    import tensorstore as ts
    import xarray as xr
    from pymmcore_plus.mda.handlers import TensorStoreHandler
    from pymmcore_plus.mda.handlers._5d_writer_base import _5DWriterBase

    from ._dims_slider import Index, Indices

    class SupportsIndexing(Protocol):
        def __getitem__(self, key: Index | tuple[Index, ...]) -> npt.ArrayLike: ...
        @property
        def shape(self) -> tuple[int, ...]: ...


ArrayT = TypeVar("ArrayT")
MAX_CHANNELS = 16
# Create a global executor
_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class DataWrapper(Generic[ArrayT]):
    """Interface for wrapping different array-like data types.

    If DataWrapper.create(your_obj) raises an exception, you can implement a new
    DataWrapper subclass to handle your data type.

    It can be passed to NDViewer.
    """

    def __init__(self, data: ArrayT) -> None:
        self._data = data

    @classmethod
    def create(cls, data: ArrayT) -> DataWrapper[ArrayT]:
        if isinstance(data, DataWrapper):
            return data
        if MMTensorStoreWrapper.supports(data):
            return MMTensorStoreWrapper(data)
        if MM5DWriter.supports(data):
            return MM5DWriter(data)
        if XarrayWrapper.supports(data):
            return XarrayWrapper(data)
        if DaskWrapper.supports(data):
            return DaskWrapper(data)
        if TensorstoreWrapper.supports(data):
            return TensorstoreWrapper(data)
        if ArrayLikeWrapper.supports(data):
            return ArrayLikeWrapper(data)
        raise NotImplementedError(f"Don't know how to wrap type {type(data)}")

    @abstractmethod
    def isel(self, indexers: Indices) -> np.ndarray:
        """Select a slice from a data store using (possibly) named indices.

        For xarray.DataArray, use the built-in isel method.
        For any other duck-typed array, use numpy-style indexing, where indexers
        is a mapping of axis to slice objects or indices.
        """
        raise NotImplementedError

    def isel_async(
        self, indexers: list[Indices]
    ) -> Future[Iterable[tuple[Indices, np.ndarray]]]:
        """Asynchronous version of isel."""
        return _EXECUTOR.submit(lambda: [(idx, self.isel(idx)) for idx in indexers])

    @classmethod
    @abstractmethod
    def supports(cls, obj: Any) -> bool:
        """Return True if this wrapper can handle the given object."""
        raise NotImplementedError

    def guess_channel_axis(self) -> Hashable | None:
        """Return the (best guess) axis name for the channel dimension."""
        if isinstance(shp := getattr(self._data, "shape", None), Sequence):
            # for numpy arrays, use the smallest dimension as the channel axis
            if min(shp) <= MAX_CHANNELS:
                return shp.index(min(shp))
        return None

    def save_as_zarr(self, save_loc: str | Path) -> None:
        raise NotImplementedError("save_as_zarr not implemented for this data type.")

    def sizes(self) -> Mapping[Hashable, int]:
        if (shape := getattr(self._data, "shape", None)) and isinstance(shape, tuple):
            _sizes: dict[Hashable, int] = {}
            for i, val in enumerate(shape):
                if isinstance(val, int):
                    _sizes[i] = val
                elif isinstance(val, Sequence) and len(val) == 2:
                    _sizes[val[0]] = int(val[1])
                else:
                    raise ValueError(
                        f"Invalid size: {val}. Must be an int or a 2-tuple."
                    )
            return _sizes
        raise NotImplementedError(f"Cannot determine sizes for {type(self._data)}")

    def summary_info(self) -> str:
        """Return info label with information about the data."""
        package = getattr(self._data, "__module__", "").split(".")[0]
        info = f"{package}.{getattr(type(self._data), '__qualname__', '')}"

        if sizes := self.sizes():
            # if all of the dimension keys are just integers, omit them from size_str
            if all(isinstance(x, int) for x in sizes):
                size_str = repr(tuple(sizes.values()))
            # otherwise, include the keys in the size_str
            else:
                size_str = ", ".join(f"{k}:{v}" for k, v in sizes.items())
                size_str = f"({size_str})"
            info += f" {size_str}"
        if dtype := getattr(self._data, "dtype", ""):
            info += f", {dtype}"
        if nbytes := getattr(self._data, "nbytes", 0) / 1e6:
            info += f", {nbytes:.2f}MB"
        return info


class MMTensorStoreWrapper(DataWrapper["TensorStoreHandler"]):
    def sizes(self) -> Mapping[Hashable, int]:
        with suppress(Exception):
            return self._data.current_sequence.sizes  # type: ignore [no-any-return]
        return {}

    def guess_channel_axis(self) -> Hashable | None:
        return "c"

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[TensorStoreHandler]:
        with suppress(ImportError):
            from pymmcore_plus.mda.handlers import TensorStoreHandler

            return isinstance(obj, TensorStoreHandler)
        return False

    def isel(self, indexers: Indices) -> np.ndarray:
        return self._data.isel(indexers)  # type: ignore [no-any-return]

    def save_as_zarr(self, save_loc: str | Path) -> None:
        if (store := self._data.store) is None:
            return
        import tensorstore as ts

        new_spec = store.spec().to_json()
        new_spec["kvstore"] = {"driver": "file", "path": str(save_loc)}
        new_ts = ts.open(new_spec, create=True).result()
        new_ts[:] = store.read().result()


class MM5DWriter(DataWrapper["_5DWriterBase"]):
    def guess_channel_axis(self) -> Hashable | None:
        return "c"

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[_5DWriterBase]:
        with suppress(ImportError):
            try:
                from pymmcore_plus.mda.handlers._5d_writer_base import _5DWriterBase
            except ImportError:
                from pymmcore_plus.mda.handlers import OMETiffWriter, OMEZarrWriter

                _5DWriterBase = (OMETiffWriter, OMEZarrWriter)
            if isinstance(obj, _5DWriterBase):
                return True
        return False

    def save_as_zarr(self, save_loc: str | Path) -> None:
        import zarr
        from pymmcore_plus.mda.handlers import OMEZarrWriter

        if isinstance(self._data, OMEZarrWriter):
            zarr.copy_store(self._data.group.store, zarr.DirectoryStore(save_loc))
        raise NotImplementedError(f"Cannot save {type(self._data)} data to Zarr.")

    def isel(self, indexers: Indices) -> np.ndarray:
        p_index = indexers.get("p", 0)
        if isinstance(p_index, slice):
            warnings.warn("Cannot slice over position index", stacklevel=2)  # TODO
            p_index = p_index.start
        p_index = cast(int, p_index)

        try:
            sizes = [*list(self._data.position_sizes[p_index]), "y", "x"]
        except IndexError as e:
            raise IndexError(
                f"Position index {p_index} out of range for "
                f"{len(self._data.position_sizes)}"
            ) from e

        data = self._data.position_arrays[self._data.get_position_key(p_index)]
        full = slice(None, None)
        index = tuple(indexers.get(k, full) for k in sizes)
        return data[index]  # type: ignore [no-any-return]


class XarrayWrapper(DataWrapper["xr.DataArray"]):
    def isel(self, indexers: Indices) -> np.ndarray:
        return np.asarray(self._data.isel(indexers))

    def sizes(self) -> Mapping[Hashable, int]:
        return {k: int(v) for k, v in self._data.sizes.items()}

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[xr.DataArray]:
        if (xr := sys.modules.get("xarray")) and isinstance(obj, xr.DataArray):
            return True
        return False

    def guess_channel_axis(self) -> Hashable | None:
        for d in self._data.dims:
            if str(d).lower() in ("channel", "ch", "c"):
                return cast("Hashable", d)
        return None

    def save_as_zarr(self, save_loc: str | Path) -> None:
        self._data.to_zarr(save_loc)


class DaskWrapper(DataWrapper["da.Array"]):
    def isel(self, indexers: Indices) -> np.ndarray:
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self._data.shape)))
        return np.asarray(self._data[idx].compute())

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[da.Array]:
        if (da := sys.modules.get("dask.array")) and isinstance(obj, da.Array):
            return True
        return False

    def save_as_zarr(self, save_loc: str | Path) -> None:
        self._data.to_zarr(url=str(save_loc))


class TensorstoreWrapper(DataWrapper["ts.TensorStore"]):
    def __init__(self, data: Any) -> None:
        super().__init__(data)
        import tensorstore as ts

        self._ts = ts

    def sizes(self) -> Mapping[Hashable, int]:
        return {dim.label: dim.size for dim in self._data.domain}

    def isel(self, indexers: Indices) -> np.ndarray:
        result = (
            self._data[self._ts.d[tuple(indexers)][tuple(indexers.values())]]
            .read()
            .result()
        )
        return np.asarray(result)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[ts.TensorStore]:
        if (ts := sys.modules.get("tensorstore")) and isinstance(obj, ts.TensorStore):
            return True
        return False


class ArrayLikeWrapper(DataWrapper):
    def isel(self, indexers: Indices) -> np.ndarray:
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self._data.shape)))
        return np.asarray(self._data[idx])

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[SupportsIndexing]:
        if (
            isinstance(obj, np.ndarray)
            or hasattr(obj, "__array_function__")
            or hasattr(obj, "__array_namespace__")
            or (hasattr(obj, "__getitem__") and hasattr(obj, "__array__"))
        ):
            return True
        return False

    def save_as_zarr(self, save_loc: str | Path) -> None:
        import zarr

        if isinstance(self._data, zarr.Array):
            self._data.store = zarr.DirectoryStore(save_loc)
        else:
            zarr.save(str(save_loc), self._data)
