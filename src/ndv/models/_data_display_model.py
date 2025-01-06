from collections.abc import Hashable, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, cast

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._base_model import NDVModel

from .data_wrappers import DataWrapper

if TYPE_CHECKING:
    from ndv._types import AxisKey
    from ndv.models._array_display_model import IndexMap, LutMap, TwoOrThreeAxisTuple
    from ndv.models._lut_model import LUTModel


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
    channel_key: Optional[int]


@dataclass
class DataRequest:
    """Request object for data slicing."""

    wrapper: DataWrapper
    index: Mapping[int, Union[int, slice]]
    visible_axes: tuple[int, ...]
    channel_axis: Optional[int]


class ArrayDataDisplayModel(NDVModel):
    """Combination of data and display models.

    Mostly this class exists to resolve AxisKeys in the display model
    (which can be axis labels, or positive/negative integers) to real/existing
    positive indices in the data.  But it also makes it easier for multiple data display
    models to share the same underlying display model (e.g. for linked views).
    """

    display: ArrayDisplayModel = Field(default_factory=ArrayDisplayModel)
    # TODO: add validator
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

    def _ensure_wrapper(self) -> DataWrapper:
        if self.data_wrapper is None:
            raise ValueError("Cannot normalize axes.  No data is set.")
        return self.data_wrapper

    @property
    def data_coords(self) -> Mapping[int, Sequence]:
        """Return the coordinates of the data as positive integers."""
        if (wrapper := self.data_wrapper) is None:
            return {}
        return {wrapper.normalized_axis_key(d): wrapper.coords[d] for d in wrapper.dims}

    # ArrayDisplayModel properties, passed through: --------------------------------

    @property
    def visible_axes(self) -> tuple[int, int, int] | tuple[int, int]:
        """Return the visible axes as positive integers."""
        wrapper = self._ensure_wrapper()
        return tuple(  # type: ignore [return-value]
            wrapper.normalized_axis_key(ax) for ax in self.display.visible_axes
        )

    @visible_axes.setter
    def visible_axes(self, axes: "TwoOrThreeAxisTuple") -> None:
        self.display.visible_axes = axes

    @property
    def current_index(self) -> Mapping[int, Union[int, slice]]:
        """Return the current index with positive integer axis keys."""
        wrapper = self._ensure_wrapper()
        return {
            wrapper.normalized_axis_key(ax): v
            for ax, v in self.display.current_index.items()
        }

    @current_index.setter
    def current_index(self, index: "IndexMap") -> None:
        self.display.current_index.assign(index)

    @property
    def channel_mode(self) -> ChannelMode:
        """Return the channel mode."""
        return self.display.channel_mode

    @channel_mode.setter
    def channel_mode(self, mode: ChannelMode) -> None:
        self.display.channel_mode = mode

    @property
    def channel_axis(self) -> Optional[int]:
        """Return the channel axis as positive integers."""
        if self.display.channel_axis is None:
            return None
        wrapper = self._ensure_wrapper()
        return wrapper.normalized_axis_key(self.display.channel_axis)

    @channel_axis.setter
    def channel_axis(self, axis: "AxisKey | None") -> None:
        self.display.channel_axis = axis

    @property
    def luts(self) -> "LutMap":
        """Return the lookup tables."""
        return self.display.luts

    @luts.setter
    def luts(self, luts: "LutMap") -> None:
        self.display.luts.assign(luts)

    @property
    def default_lut(self) -> "LUTModel":
        """Return the default lookup table."""
        return self.display.default_lut

    @property
    def current_slices(self) -> dict[int, Union[int, slice]]:
        """Return the current index, ensuring all visible axes are slices."""
        # first ensure that all visible axes (those that will be displayed in the view)
        # are slices and present in the request.
        requested = dict(self.current_index)
        for ax in self.visible_axes:
            if not isinstance(requested.get(ax), slice):
                requested[ax] = slice(None)
        return requested

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

        requested_slice = self.current_slices

        # if we need to request multiple channels (composite mode or RGB),
        # ensure that the channel axis is also sliced
        if c_ax := self.channel_axis:
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
                visible_axes=self.visible_axes,
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
                ch_indices: Iterable[Optional[int]] = range(data.shape[ch_ax])
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
