"""General model for ndv."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, cast

import numpy as np

if TYPE_CHECKING:
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
    def n_visible_dims(self) -> Literal[2, 3]:
        """Number of dims is derived from the length of `visible_dims`."""
        return cast(Literal[2, 3], len(self.visible_axes))

    # INDEXING AND REDUCTION

    current_index: Mapping[AxisKey, int | slice] = field(default_factory=dict)
    """The currently displayed position/slice along each dimension.

    e.g. {0: 0, 'time': slice(10, 20)}

    Not all axes need be present, and axes not present are assumed to
    be slice(None), meaning it is up to the controller of this model to restrict
    indices to an efficient range for retrieval.

    If the number of non-singleton axes is greater than `n_visible_dims`,
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

    NOTE: it is an error for channel_axis to be in visible_dims (or ignore it?)

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


class ArrayViewer:
    """View on an ArrayDisplayModel."""

    data: DataWrapper

    def __init__(self, model: ArrayDisplayModel | None = None):
        self.model = model or ArrayDisplayModel()

        model.visible_axes.changed.connect(self._on_visible_dims_changed)
        model.current_index.changed.connect(self._on_current_index_changed)
        model.channel_axis.changed.connect(self._on_channel_axis_changed)

        model.luts.changed.connect(self._on_luts_changed)

    def set_data(self, data: DataWrapper) -> None:
        """Prepare model for new data and update viewer."""
        self.data = data
        # when new data is assigned, we ensure that the model is consistent
        self.model = reconcile_model_with_data(self.model, data)
        self.redraw()

    def _on_visible_dims_changed(
        self, visible_dims: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey]
    ):
        pass

    def _on_current_index_changed(self, index: Mapping[AxisKey, int | slice]):
        pass

    def _on_channel_axis_changed(self, channel_axis: AxisKey | None):
        pass

    def _on_luts_changed(self, event):
        pass

    def redraw(self):
        """Redraw the current view."""
        self._view_index(self.model.current_index)

    def _view_index(self, index: Mapping[AxisKey, int | slice]):
        """Request data and display it at the given index."""
        pass


def reconcile_model_with_data(
    model: ArrayDisplayModel, data: DataWrapper
) -> ArrayDisplayModel:
    """Ensure that the model is consistent with the data, returning a new model.

    This method should be called whenever the data is changed, and should
    ensure that the model is consistent with the data.  This includes
    ensuring that the `visible_dims` are valid, that the `current_index` is
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

    return type(model)(
        visible_axes=visible_axes,
        current_index=current_index,
        reducers=model.reducers,
        channel_axis=channel_axis,
        luts=luts,
    )
