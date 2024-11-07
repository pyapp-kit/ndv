"""Model protocols for data display."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
from psygnal import SignalGroupDescriptor


@dataclass
class StatsModel:
    """A model of the statistics of a dataset.

    TODO: Consider refactoring into a protocol allowing subclassing for
    e.g. faster histogram computation, different data types?
    """

    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()

    standard_deviation: float | None = None
    average: float | None = None
    # TODO: Is the generality nice, or should we just say np.ndarray?
    histogram: tuple[Sequence[int], Sequence[float]] | None = None
    bins: int = 256

    _data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        """Returns the data backing this StatsModel."""
        if self._data is not None:
            return self._data
        raise Exception("Data has not yet been set!")

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Sets the data backing this StatsModel."""
        if data is None:
            return
        self._data = data
        self.histogram = cast(
            tuple[Sequence[int], Sequence[float]],
            np.histogram(self._data, bins=self.bins),
        )
        self.average = np.average(self._data)
        self.standard_deviation = np.std(self._data)
