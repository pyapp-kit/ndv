"""General model for ndv."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Mapping, NamedTuple, Protocol, cast
from venv import logger

import numpy as np

if TYPE_CHECKING:
    from concurrent.futures import Future

    import cmap
    from cmap._colormap import ColorStopsLike

# The Term "Axis" is used when referring to a specific dimension of an array
# We can change this to dimension if desired... but it should be uniform.

# Axis keys can be either a direct integer index or name (for labeled arrays)
# we leave it to the DataWrapper to convert `AxisKey -> AxisIndex`
AxisIndex = int
AxisLabel = str
AxisKey = AxisIndex | AxisLabel

# a specific position along a dimension
CoordIndex = int | str


class DataWrapper(Protocol):
    """What is required of a datasource."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the data."""

    def axis_index(self, axis_key: AxisKey) -> AxisIndex:
        """Convert any axis key (int or str) to an axis index (int).

        Raises a KeyError if the axis_key is not found.
        Raises a IndexError if the axis_key is out of bounds.
        """


@dataclass
class ArrayDisplayModel:
    """Model of how to display an array."""

    # VISIBLE AXIS SELECTION

    visible_axes: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey] = (-2, -1)
    """Ordered list of axes to visualize, from slowest to fastest.

    e.g. ('z', -2, -1)
    """

    @property
    def n_visible_axes(self) -> Literal[2, 3]:
        """Number of dims is derived from the length of `visible_axes`."""
        return cast(Literal[2, 3], len(self.visible_axes))

    # INDEXING AND REDUCTION

    current_index: Mapping[AxisKey, int | slice] = field(default_factory=dict)
    """The currently displayed position/slice along each dimension.

    e.g. {0: 0, 'time': slice(10, 20)}

    Not all axes need be present, and axes not present are assumed to
    be slice(None), meaning it is up to the controller of this model to restrict
    indices to an efficient range for retrieval.

    If the number of non-singleton axes is greater than `n_visible_axes`,
    then reducers are used to reduce the data along the remaining axes.

    NOTE: In terms of requesting data, there is a slight "delocalization" of state here
    in that we probably also want to avoid requesting data for channel positions that
    are not visible.
    """

    reducers: np.ufunc | Mapping[AxisKey, np.ufunc] = np.max
    """Callable to reduce data along axes remaining after slicing.

    Ideally, the ufunc should accept an `axis` argument.  (TODO: what happens if not?)
    """

    # CHANNELS AND DISPLAY

    channel_axis: AxisKey | None = None
    """The dimension index or name of the channel dimension.

    The implication of setting channel_axis is that *all* elements along the channel
    dimension are shown, with different LUTs applied to each channel.

    NOTE: it is an error for channel_axis to be in `visible_axes` (or ignore it?)

    If None, then a single lookup table is used for all channels (`luts[None]`)
    """

    luts: Mapping[CoordIndex | None, LUT] = field(default_factory=lambda: {None: LUT()})
    """Instructions for how to display each channel of the array.

    Keys represent position along the dimension specified by `channel_axis`.
    Values are `LUT` objects that specify how to display the channel.

    The special key `None` is used to represent a fallback LUT for all channels,
    and is used when `channel_axis` is None.  It should always be present
    """


@dataclass
class LUT:
    """Representation of how to display a channel of an array."""

    visible: bool = True
    """Whether to display this channel.

    NOTE: This has implications for data retrieval, as we may not want to request
    channels that are not visible.  See current_index above.
    """

    cmap: ColorStopsLike = "gray"
    """Colormap to use for this channel."""

    clims: tuple[float, float] | None = None
    """Contrast limits for this channel.

    TODO: What does `None` imply?  Autoscale?
    """

    gamma: float = 1.0
    """Gamma correction for this channel."""

    autoscale: bool | tuple[float, float] = (0, 1)
    """Whether to autoscale the colormap.

    If a tuple, then the first element is the lower quantile and the second element is
    the upper quantile.  If `True` or `(0, 1)` (np.min(), np.max()) should be used,
    otherwise, use np.quantile.  Nan values should be ignored (n.b. nanmax is slower
    and should only be used if necessary).
    """


class Texture:
    """A thing representing a backend Texture that can be updated with new data."""

    def set_data(self, data: np.ndarray, offset: tuple[int, int, int] | None):
        """Update the texture with new data.

        If offset is None, then the entire texture is replaced.
        """

    def set_clims(self, clims: tuple[float, float]):
        """Set the contrast limits for the texture."""

    def set_gamma(self, gamma: float):
        """Set the gamma correction for the texture."""

    def set_cmap(self, cmap: cmap.Colormap):
        """Set the colormap for the texture."""

    def set_visible(self, visible: bool):
        """Set the visibility of the texture."""


class ArrayViewer:
    """View on an ArrayDisplayModel."""

    data: DataWrapper

    def __init__(self, model: ArrayDisplayModel | None = None):
        self.model = model or ArrayDisplayModel()

        # async executor for requesting data
        self.chunker = Chunker()

        # handles to all textures draw on the canvas
        self._textures: dict[int | None, Texture] = {}

        # connect model events
        model.current_index.changed.connect(self._view_index)
        model.visible_axes.changed.connect(self._on_visible_axes_changed)
        model.channel_axis.changed.connect(self._on_channel_axis_changed)
        model.luts.changed.connect(self._on_luts_changed)

    def set_data(self, data: DataWrapper) -> None:
        """Prepare model for new data and update viewer."""
        self.data = data

        # when new data is assigned, we ensure that the model is consistent
        for key, value in reconcile_model_with_data(self.model, self.data).items():
            # block events
            setattr(self.model, key, value)

        # what else? <<<<<<<<<<<<<<

        self.redraw()

    def _on_visible_axes_changed(
        self, axes: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey]
    ):
        # we're switching from 2D to 3D or vice versa
        # clear all of the visuals and redraw
        self.clear()
        self.redraw()

    def _on_channel_axis_changed(self, channel_axis: AxisKey | None):
        if channel_axis is None:
            # we're turning off "composite" mode.
            # All data now uses the lut defined at self.model.luts[None]
            ...
        else:
            # we're turning on "composite" mode.
            ...

    def _on_luts_changed(self, event):
        # it's unclear what this event structure will be, and it will likely
        # be multiple methods, for each field on the LUT class.
        # but we need to update all textures in the corresponding channel with the
        # new LUTs

        channel = ...
        self._textures[channel].set_clims(...)  # etc.
        ...

    def _view_index(self, index: Mapping[AxisKey, int | slice]):
        """Request data at the given index and queue it for display.

        This is the main method that gets called when the index changes (either via
        sliders or programmatically).  It is responsible for sending requests for
        data to the chunker.
        """
        bounds = self._bounds_for_index(index)
        pixel_ratio = self._current_pixel_ratio()

        # Existing data within the area that is going to be updated should be cleared
        # However it should only be cleared if the new data represents a different
        # spatial region of the data (i.e. if the bounds are the same, then the data
        # should not be cleared, but may be updated if the pixel ratio has changed)
        # here is where an octree-like structure would be useful to determine which
        # chunks need to be invalidated.

        # request chunks from the data source and queue the callback
        for future in self.chunker.request_chunks(self.data, bounds, pixel_ratio):
            future.add_done_callback(self._on_chunk_ready)

    def _bounds_for_index(self, index: Mapping[AxisKey, int | slice]) -> Bounds:
        """Return the bounds of the data to be displayed at the given index.

        This method is responsible for converting the index into a set of bounds
        that can be used to request data from the data source.

        In addition to the `index` in will need to take into account:
        - the `visible_axes` of the model
        - the `channel_axis` of the model
        - any channels that are not visible

        TODO:
        open question is exactly what form the Bounds should take. Should it be
        mapping of `{AxisKey: Bound}`? (which allows the data to worry about indexing)
        or a `tuple[Bound, ...]` (where we've already chosen the axes)?
        """

    def _current_pixel_ratio(self) -> float:
        """Return the ratio of data/world pixels to canvas pixels.

        This will depend on the current zoom level, the size of the canvas, and the
        shape of the data.

        A pixel ratio greater than 1 means that there are more data pixels than
        canvas pixels, and that the data may be safely downsampled if desired.
        """
        pass

    def _on_chunk_ready(self, future: Future[ChunkResponse]):
        if future.cancelled():
            return
        try:
            chunk = future.result()
        except Exception as e:
            logger.debug(f"Error retrieving chunk: {e}")
            return

        self._update_data_at_offset(chunk.data, chunk.offset, chunk.channel)

    def _update_data_at_offset(
        self, data: np.ndarray, offset: tuple[int, int], channel: int | None = None
    ):
        """Update the texture at the given offset and channel."""
        self._textures[channel].set_data(data, offset)

    def clear(self):
        """Erase all visuals."""
        pass

    def redraw(self):
        """Redraw the current view."""
        self._view_index(self.model.current_index)


Bound = tuple[int, int]
Bounds = tuple[Bound, ...]


class Chunker:
    """Something backed by an async executor that can request chunks of data."""

    def request_chunks(
        self,
        data: DataWrapper,
        bounds: tuple[tuple[int, int], ...],
        pixel_ratio: float,
        *,
        cancel_pending_futures: bool = False,
    ) -> Iterable[Future]:
        """Request chunks of data from the data source.

        Note, the DataWrapper is responsible for converting the bounds into
        chunks, and may use the pixel ratio to determine the scale/resolution
        from which to request data.

        TODO: we need a mechanism for multi-resolution datasets to send "quick" chunks
        at a lower resolution level while they load the full resolution data. (It's fine
        for `request_chunks` to return multiple futures for the same chunk).  But we
        also need a way to avoid requesting lower resolution data if higher resolution
        has already been loaded.
        """


class ChunkResponse(NamedTuple):
    """Response from a Chunker including data at a specific offset."""

    # data for a single chunk
    data: np.ndarray
    # position of the chunk in texture
    offset: tuple[int, int] | tuple[int, int, int]
    # channel to which this chunk belongs
    # (not sure about this one...)
    channel: int | None = None


def reconcile_model_with_data(model: ArrayDisplayModel, data: DataWrapper) -> dict:
    """Ensure that the model is consistent with the data, returning data for new model.

    This method should be called whenever the data is changed, and should
    ensure that the model is consistent with the data.  This includes
    ensuring that the `visible_axes` are valid, that the `current_index` is
    within the bounds of the data, and that the `channel_axis` is valid.
    """
    shape = data.shape
    if (ndim := len(shape)) < 2:
        raise ValueError("Data must have at least 2 dimensions")

    visible_axes = model.visible_axes
    if len(visible_axes) > ndim:
        visible_axes = visible_axes[-ndim:]
    for axis in visible_axes:
        try:
            data.axis_index(axis)
        except (KeyError, IndexError) as e:
            # TODO: fallback to graceful default of (-3, -2, -1)[-ndim:]
            raise ValueError(f"Cannot visualize axis {axis}: {e}") from e

    current_index = {}
    for key, index in model.current_index.items():
        try:
            axis = data.axis_index(key)
            current_index[axis] = index
        except (KeyError, IndexError):
            # this dataset does not have this axis or the index is out of bounds
            # warn?
            pass

    try:
        channel_axis = data.axis_index(model.channel_axis)
    except (KeyError, IndexError):
        channel_axis = None

    # it's not so terrible to include extra luts, or to omit some
    # but we could prune them here if desired
    luts = copy.copy(model.luts)
    for key in luts:
        if key is not None:
            try:
                data.axis_index(key)
            except (KeyError, IndexError):
                del luts[key]

    return {
        "visible_axes": visible_axes,
        "current_index": current_index,
        "reducers": model.reducers,
        "channel_axis": channel_axis,
        "luts": luts,
    }
