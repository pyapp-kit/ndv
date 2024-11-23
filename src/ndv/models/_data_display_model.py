from collections.abc import Hashable, Mapping, Sequence
from functools import cached_property
from typing import Any, Protocol

import numpy as np
from pydantic import Field

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._base_model import NDVModel

from .data_wrappers import DataWrapper


class DataWrapperP(Protocol):
    """Unused protocol, just marking what we need from DataWrapper here."""

    @property
    def data(self) -> Any: ...
    @property
    def dims(self) -> tuple[Hashable, ...]: ...
    @property
    def coords(self) -> Mapping[Hashable, Sequence]: ...
    def isel(self, index: Mapping[int, int | slice]) -> np.ndarray: ...


class DataDisplayModel(NDVModel):
    """Combination of data and display models.

    Mostly this class exists to resolve AxisKeys in the display model
    (which can be axis labels, or positive/negative integers) to real/existing
    positive indices in the data.
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    data_wrapper: DataWrapper | None = None

    @property
    def data(self) -> Any | None:
        """Return the data being displayed."""
        if self.data_wrapper is None:
            return None
        return self.data_wrapper.data

    @data.setter
    def data(self, data: Any) -> None:
        """Set the data to be displayed."""
        if data is None:
            self.data_wrapper = None
        else:
            self.data_wrapper = DataWrapper.create(data)

    def _canonicalize_axis_key(self, axis: Hashable) -> int:
        """Return positive index for AxisKey (which can be +/- int or label)."""
        if self.data_wrapper is None:
            raise ValueError("Data not set")

        try:
            return self.canonicalized_axis_map[axis]
        except KeyError as e:
            ndims = len(self.data_wrapper.dims)
            if isinstance(axis, int):
                raise IndexError(
                    f"Axis index {axis} out of bounds for data with {ndims} dimensions"
                ) from e
            raise IndexError(f"Axis label {axis} not found in data dimensions") from e

    @property
    def canonical_data_coords(self) -> Mapping[int, Sequence]:
        """Return the coordinates of the data in canonical form."""
        if self.data_wrapper is None:
            return {}
        return {
            self._canonicalize_axis_key(d): self.data_wrapper.coords[d]
            for d in self.data_wrapper.dims
        }

    @property
    def canonical_visible_axes(self) -> tuple[int, ...]:
        """Return the visible axes in canonical form."""
        return tuple(
            self._canonicalize_axis_key(ax) for ax in self.display.visible_axes
        )

    @property
    def canonical_current_index(self) -> Mapping[int, int | slice]:
        """Return the current index in canonical form."""
        return {
            self._canonicalize_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }

    def current_slice_request(self) -> Mapping[int, int | slice]:
        """Return the current index request for the data.

        This reconciles the `current_index` and `visible_axes` attributes of the display
        with the available dimensions of the data to return a valid index request.
        In the returned mapping, the keys are the canonicalized (non-negative integer)
        axis indices and the values are either integers or slices (where axes present
        in `visible_axes` are guaranteed to be slices rather than integers).
        """
        if self.data_wrapper is None:
            return {}

        requested_slice = dict(self.canonical_current_index)
        for ax in self.canonical_visible_axes:
            if not isinstance(requested_slice.get(ax), slice):
                requested_slice[ax] = slice(None)

        if (c_ax := self.display.channel_axis) is not None:
            c_ax = self._canonicalize_axis_key(c_ax)
            if not isinstance(requested_slice.get(c_ax), slice):
                requested_slice[c_ax] = slice(None)

        # ensure that all axes are slices, so that we don't lose any dimensions.
        # data will be squeezed to remove singleton dimensions later after
        # transposing according to the order of visible axes
        # (this operation happens below in `current_data_slice`)
        for ax, val in requested_slice.items():
            if isinstance(val, int):
                requested_slice[ax] = slice(val, val + 1)
        return requested_slice

    # TODO: make async
    def current_data_slice(self) -> np.ndarray:
        """Return the slice of data requested by the current index (synchronous)."""
        if self.data_wrapper is None:
            raise ValueError("Data not set")

        # same shape, with singleton dims
        data = self.data_wrapper.isel(self.current_slice_request())
        # rearrange according to the order of visible axes
        t_dims = self.canonical_visible_axes
        t_dims += tuple(i for i in range(data.ndim) if i not in t_dims)
        return data.transpose(*t_dims).squeeze()

    # TODO: this needs to be cleared when data.dims changes
    @cached_property
    def canonicalized_axis_map(self) -> Mapping[Hashable, int]:
        """Create a mapping of ALL valid axis keys to canonicalized keys.

        This can be used later to quickly map any valid axis key
        (axis label, positive int, or negative int) to a positive integer index.
        """
        if not self.data_wrapper:
            raise ValueError("Data not set")

        axis_index: dict[Hashable, int] = {}
        dims = self.data_wrapper.dims
        ndims = len(dims)
        for i, dim in enumerate(dims):
            axis_index[dim] = i  # map dimension label to positive index
            axis_index[i] = i  # map positive integer index to itself
            axis_index[-(ndims - i)] = i  # map negative integer index to positive index
        return axis_index
