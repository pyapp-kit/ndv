from collections.abc import Hashable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Optional, Protocol, Union, cast

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
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
    def isel(self, index: Mapping[int, Union[int, slice]]) -> np.ndarray: ...


@dataclass
class DataResponse:
    """Response object for data requests."""

    data: np.ndarray
    channel_key: int | None


@dataclass
class DataRequest:
    """Request object for data slicing."""

    wrapper: DataWrapper
    index: Mapping[int, Union[int, slice]]
    visible_axes: tuple[int, ...]
    channel_axis: int | None


class DataDisplayModel(NDVModel):
    """Combination of data and display models.

    Mostly this class exists to resolve AxisKeys in the display model
    (which can be axis labels, or positive/negative integers) to real/existing
    positive indices in the data.
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    data_wrapper: Optional[DataWrapper] = None

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        self.display.events.channel_mode.connect(self._on_channel_mode_change)
        return self

    def _on_channel_mode_change(self, mode: ChannelMode) -> None:
        # if the mode is not grayscale, and the channel axis is not set,
        # we let the data wrapper guess the channel axis
        if (
            mode != ChannelMode.GRAYSCALE
            and self.display.channel_axis is None
            and self.data_wrapper is not None
        ):
            self.display.channel_axis = self.data_wrapper.guess_channel_axis()

    @property
    def data(self) -> Any:
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
    def canonical_channel_axis(self) -> int | None:
        """Return the channel axis in canonical form."""
        if self.display.channel_axis is None:
            return None
        return self._canonicalize_axis_key(self.display.channel_axis)

    @property
    def canonical_current_index(self) -> Mapping[int, Union[int, slice]]:
        """Return the current index in canonical form."""
        return {
            self._canonicalize_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }

    def current_slice_requests(self) -> list[DataRequest]:
        """Return the current index request for the data.

        This reconciles the `current_index` and `visible_axes` attributes of the display
        with the available dimensions of the data to return a valid index request.
        In the returned mapping, the keys are the canonicalized (non-negative integer)
        axis indices and the values are either integers or slices (where axes present
        in `visible_axes` are guaranteed to be slices rather than integers).
        """
        if self.data_wrapper is None:
            return []
        # first ensure that all visible axes (those that will be displayed in the view)
        # are slices and present in the request.
        requested_slice = dict(self.canonical_current_index)
        for ax in self.canonical_visible_axes:
            if not isinstance(requested_slice.get(ax), slice):
                requested_slice[ax] = slice(None)

        # if we need to request multiple channels (composite mode or RGB),
        # ensure that the channel axis is also sliced
        if c_ax := self.canonical_channel_axis:
            if self.display.channel_mode.is_multichannel():
                if not isinstance(requested_slice.get(c_ax), slice):
                    requested_slice[c_ax] = slice(None)
            else:
                # somewhat of a hack.
                # we heed DataRequest.channel_axis to be None if we want the view
                # to use the default_lut
                c_ax = None

        # ensure that all axes are slices, so that we don't lose any dimensions.
        # data will be squeezed to remove singleton dimensions later after
        # transposing according to the order of visible axes
        # (this operation happens below in `current_data_slice`)
        for ax, val in requested_slice.items():
            if isinstance(val, int):
                requested_slice[ax] = slice(val, val + 1)

        return [
            DataRequest(
                wrapper=self.data_wrapper,
                index=requested_slice,
                visible_axes=self.canonical_visible_axes,
                channel_axis=c_ax,
            )
        ]

    # TODO: make async
    def request_sliced_data(self) -> list[Future[DataResponse]]:
        """Return the slice of data requested by the current index (synchronous)."""
        if self.data_wrapper is None:
            raise ValueError("Data not set")

        if not (requests := self.current_slice_requests()):
            return []

        futures: list[Future[DataResponse]] = []
        for req in requests:
            data = req.wrapper.isel(req.index)

            # for transposing according to the order of visible axes
            vis_ax = req.visible_axes
            t_dims = vis_ax + tuple(i for i in range(data.ndim) if i not in vis_ax)

            if (ch_ax := req.channel_axis) is not None:
                ch_indices: Iterable[int | None] = range(data.shape[ch_ax])
            else:
                ch_indices = (None,)

            for i in ch_indices:
                if i is None:
                    ch_data = data
                else:
                    ch_keepdims = (slice(None),) * cast(int, ch_ax) + (i,) + (None,)
                    ch_data = data[ch_keepdims]
                future = Future[DataResponse]()
                future.set_result(
                    DataResponse(
                        data=ch_data.transpose(*t_dims).squeeze(),
                        channel_key=i,
                    )
                )
                futures.append(future)

        return futures

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
