# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from psygnal import SignalGroupDescriptor


@dataclass
class StatsModel:
    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()

    standard_deviation: float | None = None
    average: float | None = None
    histogram: tuple[np.ndarray, np.ndarray] | None = None
    bins: int = 256
    bin_edges: np.ndarray | None = None
    _data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        raise Exception("Data has not yet been set!")

    @data.setter
    def data(self, data: np.ndarray) -> None:
        if data is None:
            return
        self._data = data
        self.histogram = np.histogram(self._data, bins=self.bins)
        self.average = np.average(self._data)
        self.standard_deviation = np.std(self._data)
