"""In this module, we provide built-in support for many array types."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Container, Iterator
    from typing import Any, TypeAlias, TypeGuard

    import dask.array.core as da
    import numpy.typing as npt
    import pydantic_core
    import pyopencl.array as cl_array
    import sparse
    import tensorstore as ts
    from pydantic import GetCoreSchemaHandler

    Index: TypeAlias = int | slice


class SupportsIndexing(Protocol):
    def __getitem__(self, key: Index | tuple[Index, ...]) -> npt.ArrayLike: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


ArrayT = TypeVar("ArrayT")
NPArrayLike = TypeVar("NPArrayLike", bound=SupportsIndexing)
_T = TypeVar("_T", bound=type)


def _recurse_subclasses(cls: _T) -> Iterator[_T]:
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _recurse_subclasses(subclass)


class DataWrapper(Generic[ArrayT], ABC):
    """Interface for wrapping different array-like data types.

    [`DataWrapper.create()`][ndv.DataWrapper.create] is a factory method that returns a
    `DataWrapper` instance for the given data type. If your datastore type is not
    supported, you may implement a new `DataWrapper` subclass to handle your data type.
    To do this, import and subclass `DataWrapper`, and (minimally) implement the
    supports and isel methods. Ensure that your class is imported before the
    `DataWrapper.create` method is called, and it will be automatically detected and
    used to wrap your data.
    """

    # Order in which subclasses are checked for support.
    # Lower numbers are checked first, and the first supporting subclass is used.
    # Default is 50, and fallback to numpy-like duckarray is 100.
    # Subclasses can override this to change the priority in which they are checked
    PRIORITY: ClassVar[int] = 50
    # These names will be checked when looking for a channel axis
    COMMON_CHANNEL_NAMES: ClassVar[Container[str]] = ("channel", "ch", "c")
    COMMON_Z_AXIS_NAMES: ClassVar[Container[str]] = ("z", "depth", "focus")

    # Maximum dimension size consider when guessing the channel axis
    MAX_CHANNELS = 16

    def __init__(self, data: ArrayT) -> None:
        self._data = data

    # ----------------------------- Mandatory methods -----------------------------

    @classmethod
    @abstractmethod
    def supports(cls, obj: Any) -> bool:
        """Return True if this wrapper can handle the given object.

        Any exceptions raised by this method will be suppressed, so it is safe to
        directly import necessary dependencies without a try/except block.
        """

    @property
    @abstractmethod
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""

    @property
    @abstractmethod
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""

    @abstractmethod
    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray:
        """Return a slice of the data as a numpy array.

        `index` will look like (e.g.) `{0: slice(0, 10), 1: 5}`.
        """

    def save_as_zarr(self, path: str) -> None:
        raise NotImplementedError("Saving as zarr is not supported for this data type")

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype for the data."""
        try:
            return np.dtype(self._data.dtype)  # type: ignore
        except AttributeError as e:
            raise NotImplementedError(
                "`dtype` property not properly implemented for DataWrapper of type: "
                f"{type(self)}"
            ) from e

    # -----------------------------

    @classmethod
    def create(cls, data: ArrayT) -> DataWrapper[ArrayT]:
        """Create a DataWrapper instance for the given data.

        This method will detect all subclasses of DataWrapper and check them in order of
        their `PRIORITY` class variable. The first subclass that
        [`supports`][ndv.DataWrapper.supports] the given data will be used to wrap it.

        !!! tip

            This means that you can subclass DataWrapper to handle new data types.
            Just make sure that your subclass is imported before calling `create`.

        If no subclasses support the data, a `NotImplementedError` is raised.

        If an instance of `DataWrapper` is passed in, it will be returned as-is.
        """
        if isinstance(data, DataWrapper):
            return data

        # check subclasses for support
        # This allows users to define their own DataWrapper subclasses which will
        # be automatically detected (assuming they have been imported by this point)
        for subclass in sorted(_recurse_subclasses(cls), key=lambda x: x.PRIORITY):
            try:
                if subclass.supports(data):
                    logging.debug(f"Using {subclass.__name__} to wrap {type(data)}")
                    return subclass(data)
            except Exception as e:
                warnings.warn(
                    f"Error checking DataWrapper subclass {subclass.__name__}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        raise NotImplementedError(f"Don't know how to wrap type {type(data)}")

    @property
    def data(self) -> ArrayT:
        """Return the data being wrapped."""
        return self._data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            function=cls.create,
            schema=core_schema.any_schema(),
        )

    def sizes(self) -> Mapping[Hashable, int]:
        """Return the sizes of the dimensions."""
        return {dim: len(self.coords[dim]) for dim in self.dims}

    # these guess_x methods may change in the future to become more agnostic to the
    # dimension name/semantics that they are guessing.

    def guess_channel_axis(self) -> Hashable | None:
        """Return the (best guess) axis name for the channel dimension."""
        # for arrays with labeled dimensions,
        # see if any of the dimensions are named "channel"
        sizes = self.sizes()
        for dimkey, val in sizes.items():
            if str(dimkey).lower() in self.COMMON_CHANNEL_NAMES:
                if val <= self.MAX_CHANNELS:
                    return self.normalized_axis_key(dimkey)

        # otherwise use the smallest dimension as the channel axis
        return min(sizes, key=sizes.get)  # type: ignore [arg-type]

    def guess_z_axis(self) -> Hashable | None:
        """Return the (best guess) axis name for the z (3rd spatial) dimension."""
        sizes = self.sizes()
        ch = self.guess_channel_axis()
        for dimkey in sizes:
            if str(dimkey).lower() in self.COMMON_Z_AXIS_NAMES:
                if (normed := self.normalized_axis_key(dimkey)) != ch:
                    return normed

        # otherwise return the LAST axis that is neither in the last two dimensions
        # or the channel axis guess
        return next(
            (self.normalized_axis_key(x) for x in reversed(self.dims[:-2]) if x != ch),
            None,
        )

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

    # TODO: this needs to be cleared when data.dims changes
    @cached_property
    def axis_map(self) -> Mapping[Hashable, int]:
        """Mapping of ALL valid axis keys to normalized, positive integer keys."""
        axis_index: dict[Hashable, int] = {}
        ndims = len(self.dims)
        for i, dim in enumerate(self.dims):
            axis_index[dim] = i  # map dimension label to positive index
            axis_index[i] = i  # map positive integer index to itself
            axis_index[-(ndims - i)] = i  # map negative integer index to positive index
        return axis_index

    def normalized_axis_key(self, axis: Hashable) -> int:
        """Return positive index for `axis` (which can be +/- int or str label)."""
        try:
            return self.axis_map[axis]
        except KeyError as e:
            ndims = len(self.dims)
            if isinstance(axis, int):
                raise IndexError(
                    f"Axis index {axis} out of bounds for data with {ndims} dimensions"
                ) from e
            raise IndexError(f"Axis label {axis} not found in data dimensions") from e

    def clear_cache(self) -> None:
        """Clear any cached properties."""
        if hasattr(self, "axis_map"):
            del self.axis_map


##########################


class TensorstoreWrapper(DataWrapper["ts.TensorStore"]):
    """Wrapper for tensorstore.TensorStore objects."""

    def __init__(self, data: Any) -> None:
        super().__init__(data)

        import tensorstore as ts

        self._ts = ts

        spec = self.data.spec().to_json()
        dims: Sequence[Hashable] | None = None
        self._ts = ts
        if (tform := spec.get("transform")) and ("input_labels" in tform):
            dims = [str(x) for x in tform["input_labels"]]
        elif (
            str(spec.get("driver")).startswith("zarr")
            and (zattrs := self.data.kvstore.read(".zattrs").result().value)
            and isinstance((zattr_dict := json.loads(zattrs)), dict)
            and "_ARRAY_DIMENSIONS" in zattr_dict
        ):
            dims = zattr_dict["_ARRAY_DIMENSIONS"]

        if isinstance(dims, Sequence) and len(dims) == len(self._data.domain):
            self._dims: tuple[Hashable, ...] = tuple(str(x) for x in dims)
            self._data = self.data[ts.d[:].label[self._dims]]
        else:
            self._dims = tuple(range(len(self._data.domain)))
        self._coords: Mapping[Hashable, Sequence] = {
            i: range(s) for i, s in zip(self._dims, self._data.domain.shape)
        }

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return self._dims

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return self._coords

    def sizes(self) -> Mapping[Hashable, int]:
        return dict(zip(self._dims, self._data.domain.shape))

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        if not indexers:
            slc: slice | tuple = slice(None)
        else:
            slc = tuple(
                indexers.get(i, slice(None)) for i in range(len(self._data.shape))
            )
        result = self._data[slc].read().result()
        return np.asarray(result)

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[ts.TensorStore]:
        if (ts := sys.modules.get("tensorstore")) and isinstance(obj, ts.TensorStore):
            return True
        return False


class ArrayLikeWrapper(DataWrapper[NPArrayLike]):
    """Wrapper for numpy duck array-like objects."""

    PRIORITY = 100

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimension labels for the data."""
        return tuple(range(len(self.data.shape)))

    @property
    def coords(self) -> Mapping[Hashable, Sequence]:
        """Return the coordinates for the data."""
        return {i: range(s) for i, s in enumerate(self.data.shape)}

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self.data.shape)))
        return self._asarray(self.data[idx])

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

    def _asarray(self, data: npt.ArrayLike) -> np.ndarray:
        """Convert data to a numpy array."""
        return np.asarray(data)


class SparseArrayWrapper(ArrayLikeWrapper["sparse.Array"]):
    PRIORITY = 50

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[sparse.COO]:
        if (sparse := sys.modules.get("sparse")) and isinstance(obj, sparse.COO):
            return True
        return False

    def _asarray(self, data: sparse.COO) -> np.ndarray:
        return np.asarray(data.todense())


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


class DaskWrapper(ArrayLikeWrapper["da.Array"]):
    """Wrapper for dask array objects."""

    @classmethod
    def supports(cls, obj: Any) -> TypeGuard[da.Array]:
        if (da := sys.modules.get("dask.array")) and isinstance(obj, da.Array):
            return True
        return False

    def _asarray(self, data: da.Array) -> np.ndarray:
        return np.asarray(data.compute())

    def save_as_zarr(self, path: str) -> None:
        self._data.to_zarr(url=path)
