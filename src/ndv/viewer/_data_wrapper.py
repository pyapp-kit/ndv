"""In this module, we provide built-in support for many array types."""

from __future__ import annotations

import sys
from abc import abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Protocol, TypeAlias, TypeGuard

    import dask.array as da
    import numpy.typing as npt
    import pyopencl.array as cl_array
    import sparse
    import tensorstore as ts
    import xarray as xr

    from ._dims_slider import Index, Indices

    _T_contra = TypeVar("_T_contra", contravariant=True)

    class SupportsIndexing(Protocol):
        def __getitem__(self, key: Index | tuple[Index, ...]) -> npt.ArrayLike: ...
        @property
        def shape(self) -> tuple[int, ...]: ...

    class SupportsDunderLT(Protocol[_T_contra]):
        def __lt__(self, other: _T_contra, /) -> bool: ...

    class SupportsDunderGT(Protocol[_T_contra]):
        def __gt__(self, other: _T_contra, /) -> bool: ...

    SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]


ArrayT = TypeVar("ArrayT")
_T = TypeVar("_T")
MAX_CHANNELS = 16

# Global executor for slice requests
_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _recurse_subclasses(cls: _T) -> Iterator[_T]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _recurse_subclasses(subclass)


class DataWrapper(Generic[ArrayT]):
    """Interface for wrapping different array-like data types.

    `DataWrapper.create` is a factory method that returns a DataWrapper instance
    for the given data type. If your datastore type is not supported, you may implement
    a new DataWrapper subclass to handle your data type.  To do this, import and
    subclass DataWrapper, and (minimally) implement the supports and isel methods.
    Ensure that your class is imported before the DataWrapper.create method is called,
    and it will be automatically detected and used to wrap your data.
    """

    # Order in which subclasses are checked for support.
    # Lower numbers are checked first, and the first supporting subclass is used.
    # Default is 50, and fallback to numpy-like duckarray is 100.
    # Subclasses can override this to change the priority in which they are checked
    PRIORITY: ClassVar[SupportsRichComparison] = 50

    @classmethod
    def create(cls, data: ArrayT) -> DataWrapper[ArrayT]:
        if isinstance(data, DataWrapper):
            return data
        # check subclasses for support
        # This allows users to define their own DataWrapper subclasses which will
        # be automatically detected (assuming they have been imported by this point)
        for subclass in sorted(_recurse_subclasses(cls), key=lambda x: x.PRIORITY):
            with suppress(Exception):
                if subclass.supports(data):
                    print(f"Using {subclass.__name__} to wrap data.")
                    return subclass(data)
        raise NotImplementedError(f"Don't know how to wrap type {type(data)}")

    def __init__(self, data: ArrayT) -> None:
        self._data = data

    @property
    def data(self) -> ArrayT:
        return self._data

    @classmethod
    @abstractmethod
    def supports(cls, obj: Any) -> bool:
        """Return True if this wrapper can handle the given object.

        Any exceptions raised by this method will be suppressed, so it is safe to
        directly import necessary dependencies without a try/except block.
        """
        raise NotImplementedError

    @abstractmethod
    def isel(self, indexers: Indices) -> np.ndarray:
        """Select a slice from a data store using (possibly) named indices.

        This follows the xarray-style indexing, where indexers is a mapping of
        dimension names to indices or slices.  Subclasses should implement this
        method to return a numpy array.
        """
        raise NotImplementedError

    def isel_async(
        self, indexers: list[Indices]
    ) -> Future[Iterable[tuple[Indices, np.ndarray]]]:
        """Asynchronous version of isel."""
        return _EXECUTOR.submit(lambda: [(idx, self.isel(idx)) for idx in indexers])

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


class ArrayLikeWrapper(DataWrapper, Generic[ArrayT]):
    PRIORITY = 100

    def isel(self, indexers: Indices) -> np.ndarray:
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self._data.shape)))
        return self._asarray(self._data[idx])

    def _asarray(self, data: npt.ArrayLike) -> np.ndarray:
        return np.asarray(data)

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
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required to save this data type.") from None

        if isinstance(self._data, zarr.Array):
            self._data.store = zarr.DirectoryStore(save_loc)
        else:
            zarr.save(str(save_loc), self._data)


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


class CLArrayWrapper(ArrayLikeWrapper["cl_array.Array"]):
    PRIORITY = 50

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[cl_array.Array]:
        if (cl_array := sys.modules.get("pyopencl.array")) and isinstance(
            obj, cl_array.Array
        ):
            return True
        return False

    def _asarray(self, data: cl_array.Array) -> np.ndarray:
        return np.asarray(data.get())


class SparseArrayWrapper(ArrayLikeWrapper["sparse.Array"]):
    PRIORITY = 50

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[sparse.COO]:
        if (sparse := sys.modules.get("sparse")) and isinstance(obj, sparse.COO):
            return True
        return False

    def _asarray(self, data: sparse.COO) -> np.ndarray:
        return np.asarray(data.todense())
