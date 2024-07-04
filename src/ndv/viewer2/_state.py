from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    import cmap

# either the name or the index of a dimension
DimKey = str | int
# position or slice along a specific dimension
Index = int | slice
# name or dimension index of a channel
# string is only supported for arrays with string-type coordinates along the channel dim
# None is a special value that means all channels
ChannelKey = int | str | None

SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**SLOTS)
class ChannelDisplay:
    visible: bool = True
    cmap: cmap._colormap.ColorStopsLike = "gray"
    clims: tuple[float, float] | None = None
    gamma: float = 1.0
    # whether to autoscale
    # if a tuple the first element is the lower quantile
    # and the second element is the upper quantile
    # if True or (0, 1) use (np.min(), np.max()) ... otherwise use np.quantile
    autoscale: bool | tuple[float, float] = (0, 1)


@dataclass(**SLOTS)
class ViewerState:
    # index of the currently displayed slice
    # for example (-2, -1) for the standard 2D viewer
    # if string, then name2index is used to convert to index
    visualized_indices: tuple[DimKey, DimKey] | tuple[DimKey, DimKey, DimKey] = (-2, -1)

    # the currently displayed position/slice along each dimension
    # missing indices are assumed to be slice(None) (or 0?)
    # if more than len(visualized_indices) have non-integer values, then
    # reducers are used to reduce the data along the remaining dimensions
    current_index: Mapping[DimKey, Index] = field(default_factory=dict)

    # functions to reduce data along axes remaining after slicing
    reducers: Reducer | Mapping[DimKey, Reducer] = np.max

    # note: it is an error for channel_index to be in visualized_indices
    channel_index: DimKey | None = None

    # settings for each channel along the channel dimension
    # None is a special value that means all channels
    # if channel_index is None, then luts[None] is used
    luts: Mapping[ChannelKey, ChannelDisplay] = field(default_factory=dict)
    # default colormap to use for channel [0, 1, 2, ...]
    colormap_options: Sequence[cmap._colormap.ColorStopsLike] = ("gray",)

    @property
    def ndim(self) -> Literal[2, 3]:
        return 2 if len(self.visualized_indices) == 2 else 3


class Reducer(Protocol):
    def __call__(
        self, data: np.ndarray, /, *, axis: int | tuple[int, ...] | None
    ) -> np.ndarray | float: ...
