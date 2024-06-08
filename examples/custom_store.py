from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import ndv

if TYPE_CHECKING:
    from ndv import Indices, Sizes


class MyArrayThing:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self._data = np.random.randint(0, 256, shape)

    def __getitem__(self, item: Any) -> np.ndarray:
        return self._data[item]  # type: ignore [no-any-return]


class MyWrapper(ndv.DataWrapper[MyArrayThing]):
    @classmethod
    def supports(cls, data: Any) -> bool:
        if isinstance(data, MyArrayThing):
            return True
        return False

    def sizes(self) -> Sizes:
        """Return a mapping of {dim: size} for the data"""
        return {f"dim_{k}": v for k, v in enumerate(self.data.shape)}

    def isel(self, indexers: Indices) -> Any:
        """Convert mapping of {dim: index} to conventional indexing"""
        idx = tuple(indexers.get(k, slice(None)) for k in range(len(self.data.shape)))
        return self.data[idx]


data = MyArrayThing((10, 3, 512, 512))
ndv.imshow(data)
