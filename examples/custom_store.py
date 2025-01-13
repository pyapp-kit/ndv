from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import ndv

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence


class MyArrayThing:
    """Some custom data type that we want to visualize."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self._data = np.random.randint(0, 256, shape).astype(np.uint16)

    def __getitem__(self, item: Any) -> np.ndarray:
        return self._data[item]  # type: ignore [no-any-return]


class MyWrapper(ndv.DataWrapper[MyArrayThing]):
    @classmethod
    def supports(cls, data: Any) -> bool:
        """Return True if the data is supported by this wrapper"""
        if isinstance(data, MyArrayThing):
            return True
        return False

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Return the dimensions of the data"""
        return tuple(f"dim_{k}" for k in range(len(self.data.shape)))

    @property
    def coords(self) -> dict[Hashable, Sequence]:
        """Return a mapping of {dim: coords} for the data"""
        return {f"dim_{k}": range(v) for k, v in enumerate(self.data.shape)}

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the data"""
        return self.data._data.dtype

    def isel(self, indexers: Mapping[int, int | slice]) -> np.ndarray:
        """Select a subset of the data.

        `indexers` is a mapping of {dim: index} where index is either an integer or a
        slice.
        """
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self.data.shape)))
        return self.data[idx]


data = MyArrayThing((10, 3, 512, 512))
ndv.imshow(data)
