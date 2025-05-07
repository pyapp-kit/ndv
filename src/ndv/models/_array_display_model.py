"""General model for ndv."""

import logging
import warnings
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from concurrent.futures import Future
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)

from pydantic import Field, computed_field, model_validator
from typing_extensions import Self, TypeAlias

from ndv._types import AxisKey, ChannelKey, Slice
from ndv.models._data_wrapper import DataRequest, DataResponse, DataWrapper
from ndv.views import _app

from ._base_model import NDVModel
from ._lut_model import LUTModel
from ._mapping import ValidatedEventedDict
from ._reducer import ReducerType

if TYPE_CHECKING:
    from collections.abc import Hashable
    from typing import Callable  # noqa: F401

    import cmap
    import numpy as np
    import numpy.typing as npt  # noqa: F401  # used for mkdocstrings

    from ._lut_model import AutoscaleType

    class LutModelKwargs(TypedDict, total=False):
        """Keyword arguments for `LUTModel`."""

        visible: bool
        cmap: cmap.Colormap | cmap._colormap.ColorStopsLike
        clims: tuple[float, float] | None
        gamma: float
        autoscale: AutoscaleType

    class ArrayDisplayModelKwargs(TypedDict, total=False):
        """Keyword arguments for `ArrayDisplayModel`."""

        visible_axes: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey]
        current_index: Mapping[AxisKey, Union[int, slice]]
        channel_mode: "ChannelMode" | Literal["grayscale", "composite", "color", "rgba"]
        channel_axis: Optional[AxisKey]
        reducers: Mapping[AxisKey | None, ReducerType]
        luts: Mapping[int | None, LUTModel | LutModelKwargs]
        default_lut: LUTModel | LutModelKwargs


logger = logging.getLogger(__name__)

# map of axis to index/slice ... i.e. the current subset of data being displayed
IndexMap: TypeAlias = ValidatedEventedDict[AxisKey, Union[int, Slice]]
# map of index along channel axis to LUTModel object
LutMap: TypeAlias = ValidatedEventedDict[ChannelKey, LUTModel]
# map of axis to reducer
Reducers: TypeAlias = ValidatedEventedDict[Union[AxisKey, None], ReducerType]
# used for visible_axes
TwoOrThreeAxisTuple: TypeAlias = Union[
    tuple[AxisKey, AxisKey, AxisKey], tuple[AxisKey, AxisKey]
]


def _default_luts() -> LutMap:
    colors = ["green", "magenta", "cyan", "red", "blue", "yellow"]
    return ValidatedEventedDict(
        (i, LUTModel(cmap=color)) for i, color in enumerate(colors)
    )


class ChannelMode(str, Enum):
    """Channel display mode.

    Attributes
    ----------
    GRAYSCALE : str
        The array is displayed as a single channel, with a single lookup table applied.
        In this mode, there effective *is* no channel axis: all non-visible dimensions
        have sliders, and there is a single LUT control (the `default_lut`).
    COMPOSITE : str
        Display all (or a subset of) channels as a composite image, with a different
        lookup table applied to each channel.  In this mode, the slider for the channel
        axis is hidden by default, and LUT controls for each channel are shown.
    COLOR : str
        Display a single channel at a time as a color image, with a channel-specific
        lookup table applied.  In this mode, the slider for the channel axis is shown,
        and the user can select which channel to display.  LUT controls are shown for
        all channels.
    RGBA : str
        The array is displayed as an RGB image, with a single lookup table applied.
        In this mode, the slider for the channel axis is hidden, and a single LUT
        control is shown. Only valid when channel axis has length <= 4.
    RGB : str
        Alias for RGBA.
    """

    GRAYSCALE = "grayscale"
    COMPOSITE = "composite"
    COLOR = "color"
    RGBA = "rgba"

    def __str__(self) -> str:
        return self.value

    def is_multichannel(self) -> bool:
        """Return whether this mode displays multiple channels.

        If `is_multichannel` is True, then the `channel_axis` slider should be hidden.
        """
        return self in (ChannelMode.COMPOSITE, ChannelMode.RGBA)


ChannelMode._member_map_["RGB"] = ChannelMode.RGBA  #  ChannelMode["RGB"]
ChannelMode._value2member_map_["rgb"] = ChannelMode.RGBA  # ChannelMode("rgb")


class ArrayDisplayModel(NDVModel):
    """Model of how to display an array.

    An `ArrayDisplayModel` is used to specify how to display a multi-dimensional array.
    It specifies which axes are visible, how to reduce along axes that are not visible,
    how to display channels, and how to apply lookup tables to channels.  It is
    typically paired with a [`ndv.DataWrapper`][] in order to resolve axis keys and
    slice data.

    !!! info

        In the following types, `Hashable` is used to refer to a type that will
        typically be either an integer index or a string label for an axis.

    Attributes
    ----------
    visible_axes : tuple[Hashable, ...]
        Ordered list of axes to visualize, from slowest to fastest.
        e.g. `('Z', -2, -1)`
    current_index : Mapping[Hashable, int | slice]
        The currently displayed position/slice along each dimension.
        e.g. {0: 0, 'time': slice(10, 20)}
        Not all axes need be present, and axes not present are assumed to
        be slice(None), meaning it is up to the controller of this model to restrict
        indices to an efficient range for retrieval.
        If the number of non-singleton axes is greater than `n_visible_axes`,
        then `reducers` are used to reduce the data along the remaining axes.
    reducers : Mapping[Hashable | None, numpy.ufunc]
        Function used to reduce data along axes remaining after slicing.
        Ideally, the ufunc should accept an `axis` argument.
        *(TODO: what happens if not?)*
    default_reducer : numpy.ufunc
        Default reducer to use when no reducer is specified for an axis. By default,
        this is [`numpy.max`][].
    channel_mode : ChannelMode
        How to display channel information:

        - `GRAYSCALE`: ignore channel axis, use `default_lut`
        - `COMPOSITE`: display all channels as a composite image, using `luts`
        - `COLOR`: display a single channel at a time, using `luts`
        - `RGBA`: display as an RGB image, using `default_lut` (except for cmap)

        If `channel_mode` is set to anything other than `GRAYSCALE`, then `channel_axis`
        must be set to a valid axis; if no `channel_axis` is set, at the time of
        display, the [`DataWrapper`][ndv.DataWrapper] MAY guess the `channel_axis`,
        and set it on the model.
    channel_axis : Hashable | None
        The dimension index or name of the channel dimension.
        The implication of setting channel_axis is that *all* elements along the channel
        dimension are shown, with different LUTs applied to each channel.
        If None, then a single lookup table is used for all channels (`luts[None]`).
        NOTE: it is an error for channel_axis to be in `visible_axes` (or ignore it?)
    luts : Mapping[int | None, LUTModel]
        Instructions for how to display each channel of the array.
        Keys represent position along the dimension specified by `channel_axis`.
        Values are `LUT` objects that specify how to display the channel.
        The special key `None` is used to represent a fallback LUT for all channels,
        and is used when `channel_axis` is None.  It should always be present
    default_lut : LUTModel
        Default lookup table to use when no `LUTModel` is specified for a channel in
        `luts`.
    """

    # NOTE: In terms of requesting data, there is a slight "delocalization" of state
    # here in that we probably also want to avoid requesting data for channel
    # positions that are not visible.
    current_index: IndexMap = Field(default_factory=IndexMap, frozen=True)

    # map of axis to reducer (function that can reduce dimensionality along that axis)
    reducers: Reducers = Field(default_factory=Reducers, frozen=True)
    default_reducer: ReducerType = "numpy.max"  # type: ignore [assignment]  # FIXME

    channel_mode: ChannelMode = ChannelMode.GRAYSCALE
    channel_axis: Optional[AxisKey] = None
    # must come after channel_axis, since it is used to set default visible_axes
    visible_axes: TwoOrThreeAxisTuple = Field(
        default_factory=lambda k: (-3, -2) if k.get("channel_axis") == -1 else (-2, -1)
    )

    # map of index along channel axis to LUTModel object
    luts: LutMap = Field(default_factory=_default_luts)
    default_lut: LUTModel = Field(default_factory=LUTModel, frozen=True)

    @computed_field  # type: ignore [prop-decorator]
    @property
    def n_visible_axes(self) -> Literal[2, 3]:
        """Number of dims is derived from the length of `visible_axes`."""
        return cast("Literal[2, 3]", len(self.visible_axes))

    @model_validator(mode="after")
    def _validate_model(self) -> "Self":
        # prevent channel_axis from being in visible_axes
        if self.channel_axis in self.visible_axes:
            warnings.warn(
                "Channel_axis cannot be in visible_axes. Setting channel_axis to None.",
                UserWarning,
                stacklevel=2,
            )
            self.channel_axis = None
        return self

    # Indexing and Data Slicing -----------------------------------------------------

    def slice_wrapper(
        self, wrapper: DataWrapper, asynchronous: bool = True
    ) -> Iterator[Future[DataResponse]]:
        """Return the slice of data requested by the current index (synchronous)."""
        if not (requests := self._generate_requests(wrapper)):
            return

        if not asynchronous:
            for request in requests:
                future: Future[DataResponse] = Future()
                future.set_result(self.process_request(request))
                yield future
        else:
            for request in requests:
                yield _app.submit_task(self.process_request, request)

    def _generate_requests(self, wrapper: DataWrapper) -> list[DataRequest]:
        """Return the current index request for the data.

        This reconciles the `current_index` and `visible_axes` attributes of the display
        with the available dimensions of the data to return a valid index request.
        In the returned mapping, the keys are the normalized (non-negative integer)
        axis indices and the values are either integers or slices (where axes present
        in `visible_axes` are guaranteed to be slices rather than integers).
        """
        wrapper_visible_axes = self._normed_visible_axes(wrapper)
        requested_slice = dict(self._normed_current_index(wrapper))
        for ax in wrapper_visible_axes:
            if not isinstance(requested_slice.get(ax), slice):
                requested_slice[ax] = slice(None)

        # if we need to request multiple channels (composite mode or RGB),
        # ensure that the channel axis is also sliced
        if (c_ax := self._normed_channel_axis(wrapper)) is not None:
            if self.channel_mode.is_multichannel():
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
            wrapper=wrapper,
            index=requested_slice,
            visible_axes=wrapper_visible_axes,
            channel_axis=c_ax,
            channel_mode=self.channel_mode,
        )
        return [request]

    def _normed_visible_axes(
        self, wrapper: DataWrapper
    ) -> "tuple[int, int, int] | tuple[int, int]":
        """Return the visible axes as positive integers."""
        return tuple(  # type: ignore [return-value]
            wrapper.normalize_axis_key(ax) for ax in self.visible_axes
        )

    def _normed_channel_axis(self, wrapper: DataWrapper) -> "int | None":
        """Return the channel axis as positive integers."""
        if self.channel_axis is None:
            return None
        return wrapper.normalize_axis_key(self.channel_axis)

    def _normed_current_index(
        self, wrapper: DataWrapper
    ) -> Mapping[int, Union[int, slice]]:
        """Return the current index with positive integer axis keys.

        This method has to handle cases where the the model current index is expressed
        in a non-normalized way (e.g. with string labels for axes), AND it has to handle
        cases where a non-normalized key resolves a normalized key that may *also* be in
        the set: in that cases, priority is given to the non-normalized key and a
        warning is issued.

        For example, if current_index is {'Z': 0, 0: 10}, and the data has a 'Z'
        dimension label, but Z is also the 0th axis, then the output normed index will
        be {0: 0} (the string key is prioritized over the integer key), a warning
        is issued, and the 0 key is removed from the current_index.
        """
        output: MutableMapping[int, Union[int, slice]] = {}
        rev_map: dict[Hashable, Hashable] = {}
        to_remove: list[Hashable] = []

        for key, val in self.current_index.items():
            normed_key = wrapper.normalize_axis_key(key)
            if normed_key in output:
                to_remove.append(key)
                was_already_normed = normed_key == key
                if was_already_normed:
                    original_key = rev_map[normed_key]
                else:
                    original_key = key

                logger.warning(
                    "Axis key %r normalized to %r, which is also in current_index. "
                    "Using %r value.",
                    original_key,
                    normed_key,
                    original_key,
                )
                if was_already_normed:
                    # in the case of duplication, we give priority to NON-normed keys
                    # (e.g. "Z" over 0)
                    continue

            output[normed_key] = val
            rev_map[normed_key] = key

        # cleanup the model so it doesn't have to be done again
        for key in to_remove:
            del self.current_index[key]

        return output

    # TODO: Move to DataWrapper?
    def _normed_data_coords(self, wrapper: DataWrapper) -> Mapping[int, Sequence]:
        """Return the coordinates of the data as positive integers."""
        return {wrapper.normalize_axis_key(d): wrapper.coords[d] for d in wrapper.dims}

    @staticmethod
    def process_request(req: DataRequest) -> DataResponse:
        """Process a data request and return the sliced data as a DataResponse."""
        data = req.wrapper.isel(req.index)

        # for transposing according to the order of visible axes
        vis_ax = req.visible_axes
        t_dims = vis_ax + tuple(i for i in range(data.ndim) if i not in vis_ax)

        data_response: dict[ChannelKey, np.ndarray] = {}
        ch_ax = req.channel_axis
        # For RGB and Grayscale - keep the whole array together
        if req.channel_mode == ChannelMode.RGBA:
            data_response["RGB"] = data.transpose(*t_dims).squeeze()
        elif req.channel_axis is None:
            data_response[None] = data.transpose(*t_dims).squeeze()
        # For Composite and Color - slice along channel axis
        else:
            for i in range(data.shape[req.channel_axis]):
                ch_keepdims = (slice(None),) * cast("int", ch_ax) + (i,) + (None,)
                ch_data = data[ch_keepdims]
                data_response[i] = ch_data.transpose(*t_dims).squeeze()

        return DataResponse(n_visible_axes=len(vis_ax), data=data_response, request=req)
