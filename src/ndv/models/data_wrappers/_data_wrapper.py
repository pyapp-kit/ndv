"""In this module, we provide built-in support for many array types."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Container, Iterator
    from typing import Any, TypeAlias

    import pydantic_core
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

    `DataWrapper.create` is a factory method that returns a DataWrapper instance
    for the given data type. If your datastore type is not supported, you may implement
    a new DataWrapper subclass to handle your data type.  To do this, import and
    subclass DataWrapper, and (minimally) implement the supports and isel methods.
    Ensure that your class is imported before the DataWrapper.create method is called,
    and it will be automatically detected and used to wrap your data.
    """

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
        """Return a slice of the data as a numpy array."""

    # -----------------------------

    def save_as_zarr(self, path: str) -> None:
        raise NotImplementedError("Saving as zarr is not supported for this data type")

    # Order in which subclasses are checked for support.
    # Lower numbers are checked first, and the first supporting subclass is used.
    # Default is 50, and fallback to numpy-like duckarray is 100.
    # Subclasses can override this to change the priority in which they are checked
    PRIORITY: ClassVar[int] = 50
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

    def __init__(self, data: ArrayT) -> None:
        self._data = data

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(
            function=cls.create,
            schema=core_schema.any_schema(),
        )


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
        return np.asarray(self.data[idx])

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
