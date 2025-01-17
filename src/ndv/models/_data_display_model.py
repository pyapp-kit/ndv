import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np
from pydantic import Field

from ndv.views import _app

from ._array_display_model import ArrayDisplayModel, ChannelMode
from ._base_model import NDVModel
from ._data_wrapper import DataWrapper

__all__ = ["DataRequest", "DataResponse", "_ArrayDataDisplayModel"]

SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(frozen=True, **SLOTS)
class DataRequest:
    """Request object for data slicing."""

    wrapper: DataWrapper = field(repr=False)
    index: Mapping[int, Union[int, slice]]
    visible_axes: tuple[int, ...]
    channel_axis: Optional[int]


@dataclass(frozen=True, **SLOTS)
class DataResponse:
    """Response object for data requests.

    In the response, the data is broken up according to channel keys.
    """

    # mapping of channel_key -> data
    n_visible_axes: int
    data: Mapping[Optional[int], np.ndarray] = field(repr=False)
    request: Optional[DataRequest] = None


# NOTE: nobody particularly likes this class.  It does important stuff, but we're
# not yet sure where this logic belongs.
class _ArrayDataDisplayModel(NDVModel):
    """Utility class combining ArrayDisplayModel model with a DataWrapper.

    The `ArrayDisplayModel` can be thought of as an "instruction" for how to display
    some data, while the `DataWrapper` is the actual data.  This class combines the two
    and provides a way to access the data in a normalized way (i.e. be converting
    AxisKeys in the display model to positive integers, based on the available
    dimensions of the DataWrapper).  This makes it easier to index into the data, even
    with named axes, which this class also helps manage with the `request_sliced_data`
    method.

    Having this class composed of the two other models (rather than inheriting from
    `ArrayDisplayModel`) allows for multiple models to share the same underlying
    display model (e.g. for linked views).

    Attributes
    ----------
    display : ArrayDisplayModel
        The display model. Provides instructions for how to display the data.
    data_wrapper : DataWrapper  | None
        The data wrapper. Provides the actual data to be displayed
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    data_wrapper: Optional[DataWrapper] = None

    def model_post_init(self, __context: Any) -> None:
        # connect the channel mode change signal to the channel axis guessing method
        self.display.events.channel_mode.connect(self._on_channel_mode_change)

    def _on_channel_mode_change(self) -> None:
        # if the mode is not grayscale, and the channel axis is not set,
        # we let the data wrapper guess the channel axis
        if (
            self.display.channel_mode != ChannelMode.GRAYSCALE
            and self.display.channel_axis is None
            and self.data_wrapper is not None
        ):
            # only use the guess if it's not already in the visible axes
            guess = self.data_wrapper.guess_channel_axis()
            if guess not in self.normed_visible_axes:
                self.display.channel_axis = guess

    # Properties for normalized data access -----------------------------------------
    # these all use positive integers as axis keys

    def _ensure_wrapper(self) -> DataWrapper:
        if self.data_wrapper is None:
            raise ValueError("Cannot normalize axes.  No data is set.")
        return self.data_wrapper

    @property
    def normed_data_coords(self) -> Mapping[int, Sequence]:
        """Return the coordinates of the data as positive integers."""
        if (wrapper := self.data_wrapper) is None:
            return {}
        return {wrapper.normalized_axis_key(d): wrapper.coords[d] for d in wrapper.dims}

    @property
    def normed_visible_axes(self) -> "tuple[int, int, int] | tuple[int, int]":
        """Return the visible axes as positive integers."""
        wrapper = self._ensure_wrapper()
        return tuple(  # type: ignore [return-value]
            wrapper.normalized_axis_key(ax) for ax in self.display.visible_axes
        )

    @property
    def normed_current_index(self) -> Mapping[int, Union[int, slice]]:
        """Return the current index with positive integer axis keys."""
        wrapper = self._ensure_wrapper()
        return {
            wrapper.normalized_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }

    @property
    def normed_channel_axis(self) -> "int | None":
        """Return the channel axis as positive integers."""
        if self.display.channel_axis is None:
            return None
        wrapper = self._ensure_wrapper()
        return wrapper.normalized_axis_key(self.display.channel_axis)

    # Indexing and Data Slicing -----------------------------------------------------

    def current_slice_requests(self) -> list[DataRequest]:
        """Return the current index request for the data.

        This reconciles the `current_index` and `visible_axes` attributes of the display
        with the available dimensions of the data to return a valid index request.
        In the returned mapping, the keys are the normalized (non-negative integer)
        axis indices and the values are either integers or slices (where axes present
        in `visible_axes` are guaranteed to be slices rather than integers).
        """
        if self.data_wrapper is None:
            return []

        requested_slice = dict(self.normed_current_index)
        for ax in self.normed_visible_axes:
            if not isinstance(requested_slice.get(ax), slice):
                requested_slice[ax] = slice(None)

        # if we need to request multiple channels (composite mode or RGB),
        # ensure that the channel axis is also sliced
        if c_ax := self.normed_channel_axis:
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

        request = DataRequest(
            wrapper=self.data_wrapper,
            index=requested_slice,
            visible_axes=self.normed_visible_axes,
            channel_axis=c_ax,
        )
        return [request]

    def request_sliced_data(
        self, asynchronous: bool = True
    ) -> Iterator[Future[DataResponse]]:
        """Return the slice of data requested by the current index (synchronous)."""
        if self.data_wrapper is None:
            raise ValueError("Data not set")

        if not (requests := self.current_slice_requests()):
            return

        if not asynchronous:
            for request in requests:
                future: Future[DataResponse] = Future()
                future.set_result(self.process_request(request))
                yield future
        else:
            for request in requests:
                yield _app.submit_task(self.process_request, request)

    @staticmethod
    def process_request(req: DataRequest) -> DataResponse:
        """Process a data request and return the sliced data as a DataResponse."""
        data = req.wrapper.isel(req.index)

        # for transposing according to the order of visible axes
        vis_ax = req.visible_axes
        t_dims = vis_ax + tuple(i for i in range(data.ndim) if i not in vis_ax)

        if (ch_ax := req.channel_axis) is not None:
            ch_indices: Iterable[Optional[int]] = range(data.shape[ch_ax])
        else:
            ch_indices = (None,)

        data_response: dict[int | None, np.ndarray] = {}
        for i in ch_indices:
            if i is None:
                ch_data = data
            else:
                ch_keepdims = (slice(None),) * cast(int, ch_ax) + (i,) + (None,)
                ch_data = data[ch_keepdims]
            data_response[i] = ch_data.transpose(*t_dims).squeeze()

        return DataResponse(n_visible_axes=len(vis_ax), data=data_response, request=req)
