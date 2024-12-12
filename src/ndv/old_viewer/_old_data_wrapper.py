"""In this module, we provide built-in support for many array types."""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from collections.abc import Container, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Generic,
    TypeVar,
)

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Protocol, TypeAlias, TypeGuard

    import dask.array as da
    import numpy.typing as npt
    import pyopencl.array as cl_array
    import sparse
    import tensorstore as ts
    import torch
    import xarray as xr
    import zarr
    from torch._tensor import Tensor

    Index = int | slice
    Indices = Mapping[Any, Index]
    Sizes = Mapping[Any, int]

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
_T = TypeVar("_T", bound=type)

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
    # These names will be checked when looking for a channel axis
    COMMON_CHANNEL_NAMES: ClassVar[Container[str]] = ("channel", "ch", "c")
    # Maximum dimension size consider when guessing the channel axis
    MAX_CHANNELS = 16

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
                    logging.debug(f"Using {subclass.__name__} to wrap {type(data)}")
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

    def guess_channel_axis(self) -> Any | None:
        """Return the (best guess) axis name for the channel dimension."""
        # for arrays with labeled dimensions,
        # see if any of the dimensions are named "channel"
        for dimkey, val in self.sizes().items():
            if str(dimkey).lower() in self.COMMON_CHANNEL_NAMES:
                if val <= self.MAX_CHANNELS:
                    return dimkey

        # for shaped arrays, use the smallest dimension as the channel axis
        shape = getattr(self._data, "shape", None)
        if isinstance(shape, Sequence):
            with suppress(ValueError):
                smallest_dim = min(shape)
                if smallest_dim <= self.MAX_CHANNELS:
                    return shape.index(smallest_dim)
        return None

    def save_as_zarr(self, save_loc: str | Path) -> None:
        raise NotImplementedError("save_as_zarr not implemented for this data type.")

    def sizes(self) -> Sizes:
        """Return a mapping of {dimkey: size} for the data.

        The default implementation uses the shape attribute of the data, and
        tries to find dimension names in the `dims`, `names`, or `labels` attributes.
        (`dims` is used by xarray, `names` is used by torch, etc...). If no labels
        are found, the dimensions are just named by their integer index.
        """
        shape = getattr(self._data, "shape", None)
        if not isinstance(shape, Sequence) or not all(
            isinstance(x, int) for x in shape
        ):
            raise NotImplementedError(f"Cannot determine sizes for {type(self._data)}")
        dims = range(len(shape))
        return {dim: int(size) for dim, size in zip(dims, shape)}

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
    """Wrapper for xarray DataArray objects."""

    def isel(self, indexers: Indices) -> np.ndarray:
        return np.asarray(self._data.isel(indexers))

    def sizes(self) -> Mapping[Hashable, int]:
        return {k: int(v) for k, v in self._data.sizes.items()}

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[xr.DataArray]:
        if (xr := sys.modules.get("xarray")) and isinstance(obj, xr.DataArray):
            return True
        return False

    def save_as_zarr(self, save_loc: str | Path) -> None:
        self._data.to_zarr(save_loc)


class TensorstoreWrapper(DataWrapper["ts.TensorStore"]):
    """Wrapper for tensorstore.TensorStore objects."""

    def __init__(self, data: Any) -> None:
        super().__init__(data)
        import json

        import tensorstore as ts

        self._ts = ts

        spec = self.data.spec().to_json()
        labels: Sequence[Hashable] | None = None
        self._ts = ts
        if (tform := spec.get("transform")) and ("input_labels" in tform):
            labels = [str(x) for x in tform["input_labels"]]
        elif (
            str(spec.get("driver")).startswith("zarr")
            and (zattrs := self.data.kvstore.read(".zattrs").result().value)
            and isinstance((zattr_dict := json.loads(zattrs)), dict)
            and "_ARRAY_DIMENSIONS" in zattr_dict
        ):
            labels = zattr_dict["_ARRAY_DIMENSIONS"]

        if isinstance(labels, Sequence) and len(labels) == len(self._data.domain):
            self._labels: list[Hashable] = [str(x) for x in labels]
            self._data = self.data[ts.d[:].label[self._labels]]
        else:
            self._labels = list(range(len(self._data.domain)))

    def sizes(self) -> Mapping[Hashable, int]:
        return dict(zip(self._labels, self._data.domain.shape))

    def isel(self, indexers: Indices) -> np.ndarray:
        if not indexers:
            slc = slice(None)
        else:
            labels, values = zip(*indexers.items())
            origins = (0,) * len(labels)
            slc = self._ts.d[labels].translate_to[origins][values]
        result = self._data[slc].read().result()
        return np.asarray(result)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[ts.TensorStore]:
        if (ts := sys.modules.get("tensorstore")) and isinstance(obj, ts.TensorStore):
            return True
        return False


class ArrayLikeWrapper(DataWrapper, Generic[ArrayT]):
    """Wrapper for numpy duck array-like objects."""

    PRIORITY = 100

    def isel(self, indexers: Indices) -> np.ndarray:
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self._data.shape)))
        return self._asarray(self._data[idx])

    def _asarray(self, data: npt.ArrayLike) -> np.ndarray:
        return np.asarray(data)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[SupportsIndexing]:
        if (
            (
                isinstance(obj, np.ndarray)
                or hasattr(obj, "__array_function__")
                or hasattr(obj, "__array_namespace__")
                or hasattr(obj, "__array__")
            )
            and hasattr(obj, "__getitem__")
            and hasattr(obj, "shape")
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
    """Wrapper for dask array objects."""

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
    """Wrapper for pyopencl array objects."""

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


class ZarrArrayWrapper(ArrayLikeWrapper["zarr.Array"]):
    """Wrapper for zarr array objects."""

    PRIORITY = 50

    def __init__(self, data: Any) -> None:
        super().__init__(data)
        self._name2index: dict[Hashable, int]
        if "_ARRAY_DIMENSIONS" in data.attrs:
            self._name2index = {
                name: i for i, name in enumerate(data.attrs["_ARRAY_DIMENSIONS"])
            }
        else:
            self._name2index = {i: i for i in range(data.ndim)}

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[zarr.Array]:
        if (zarr := sys.modules.get("zarr")) and isinstance(obj, zarr.Array):
            return True
        return False

    def sizes(self) -> Sizes:
        return dict(zip(self._name2index, self.data.shape))

    def isel(self, indexers: Indices) -> np.ndarray:
        # convert possibly named indices to integer indices
        real_indexers = {self._name2index.get(k, k): v for k, v in indexers.items()}
        return super().isel(real_indexers)


class TorchTensorWrapper(DataWrapper["torch.Tensor"]):
    """Wrapper for torch tensor objects."""

    def __init__(self, data: Tensor) -> None:
        super().__init__(data)
        self._name2index: dict[Hashable, int]
        if names := getattr(data, "names", None):
            # names may be something like (None, None, None)...
            self._name2index = {
                (i if name is None else name): i for i, name in enumerate(names)
            }
        else:
            self._name2index = {i: i for i in range(data.ndim)}

    def sizes(self) -> Sizes:
        return dict(zip(self._name2index, self.data.shape))

    def isel(self, indexers: Indices) -> np.ndarray:
        # convert possibly named indices to integer indices
        real_indexers = {self._name2index.get(k, k): v for k, v in indexers.items()}
        # convert to tuple of slices
        idx = tuple(real_indexers.get(i, slice(None)) for i in range(self.data.ndim))
        return self.data[idx].numpy(force=True)  # type: ignore [no-any-return]

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[torch.Tensor]:
        if (torch := sys.modules.get("torch")) and isinstance(obj, torch.Tensor):
            return True
        return False
