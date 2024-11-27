from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import cast

import numpy as np


@dataclass(frozen=True)
class Stats:
    """The statistics of a dataset.

    TODO: Async. computation?

    Parameters
    ----------
    data : np.ndarray | None
        The dataset.
    bins : int
        Number of bins to use for histogram computation. Defaults to 256.
    average : float
        The average (mean) value of data.
    standard_deviation : float
        The standard deviation of data.
    histogram : tuple[Sequence[int], Sequence[float]]
        A 2-tuple of sequences.

        The first sequence contains (n) integers, where index i is the number of data
        points in the ith bin.

        The second sequence contains (n+1) floats. The ith bin spans the domain
        between the values at index i (inclusive) and index i+1 (exclusive).
    """

    data: np.ndarray
    bins: int = 256

    @cached_property
    def standard_deviation(self) -> float:
        """Computes the standard deviation of the dataset."""
        if self.data is None:
            return float("nan")
        return float(np.std(self.data))

    @cached_property
    def average(self) -> float:
        """Computes the average of the dataset."""
        if self.data is None:
            return float("nan")
        return float(np.mean(self.data))

    @cached_property
    def histogram(self) -> tuple[Sequence[int], Sequence[float]]:
        """Computes the histogram of the dataset."""
        if self.data is None:
            return ([], [])
        return cast(
            tuple[Sequence[int], Sequence[float]],
            np.histogram(self.data, bins=self.bins),
        )
