"""General model for ndv."""

import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

from pydantic import Field, PlainValidator, computed_field, model_validator

from ._base_model import NDVModel
from ._lut_model import LUTModel
from ._mapping import ValidatedEventedDict
from ._reducer import ReducerType

# The Term "Axis" is used when referring to a specific dimension of an array
# We can change this to dimension if desired... but it should be uniform.

# Axis keys can be either a direct integer index or name (for labeled arrays)
# we leave it to the DataWrapper to convert `AxisKey -> AxisIndex`
AxisIndex: TypeAlias = int
AxisLabel: TypeAlias = str
# a specific position along a dimension
# this could eventually be any hashable (like in xarray), but for now we start with int
CoordIndex: TypeAlias = int


def _maybe_int(val: Any) -> Any:
    # try to convert to int if possible
    with suppress(ValueError, TypeError):
        val = int(float(val))
    return val


def _to_slice(val: Any) -> slice:
    # slices are returned as is
    if isinstance(val, slice):
        return val
    # single integers are converted to slices starting at that index
    if isinstance(val, int):
        return slice(val, val + 1)
    # sequences are interpreted as arguments to the slice constructor
    if isinstance(val, Sequence) and not isinstance(val, str):
        return slice(*(int(x) if x is not None else None for x in val))
    raise TypeError(f"Expected int or slice, got {type(val)}")


Slice = Annotated[slice, PlainValidator(_to_slice)]
AxisKey: TypeAlias = Annotated[AxisIndex | AxisLabel, PlainValidator(_maybe_int)]
# map of axis to index/slice ... i.e. the current subset of data being displayed
IndexMap: TypeAlias = ValidatedEventedDict[AxisKey, int | Slice]
# map of index along channel axis to LUTModel object
LutMap: TypeAlias = ValidatedEventedDict[CoordIndex | None, LUTModel]
# map of axis to reducer
Reducers: TypeAlias = ValidatedEventedDict[AxisKey | None, ReducerType]


class ArrayDisplayModel(NDVModel):
    """Model of how to display an array.

    In the following types, `AxisKey` can be either an integer index or a string label.

    Parameters
    ----------
    visible_axes : tuple[AxisKey, ...]
        Ordered list of axes to visualize, from slowest to fastest.
        e.g. ('z', -2, -1)
    current_index : Mapping[AxisKey, int | Slice]
        The currently displayed position/slice along each dimension.
        e.g. {0: 0, 'time': slice(10, 20)}
        Not all axes need be present, and axes not present are assumed to
        be slice(None), meaning it is up to the controller of this model to restrict
        indices to an efficient range for retrieval.
        If the number of non-singleton axes is greater than `n_visible_axes`,
        then reducers are used to reduce the data along the remaining axes.
        NOTE: In terms of requesting data, there is a slight "delocalization" of state
        here in that we probably also want to avoid requesting data for channel
        positions that are not visible.
    reducers : Mapping[AxisKey | None, ReducerType]
        Callable to reduce data along axes remaining after slicing.
        Ideally, the ufunc should accept an `axis` argument.
        (TODO: what happens if not?)
    channel_axis : AxisKey | None
        The dimension index or name of the channel dimension.
        The implication of setting channel_axis is that *all* elements along the channel
        dimension are shown, with different LUTs applied to each channel.
        If None, then a single lookup table is used for all channels (`luts[None]`).
        NOTE: it is an error for channel_axis to be in `visible_axes` (or ignore it?)
    luts : Mapping[CoordIndex | None, LUTModel]
        Instructions for how to display each channel of the array.
        Keys represent position along the dimension specified by `channel_axis`.
        Values are `LUT` objects that specify how to display the channel.
        The special key `None` is used to represent a fallback LUT for all channels,
        and is used when `channel_axis` is None.  It should always be present
    """

    visible_axes: tuple[AxisKey, AxisKey, AxisKey] | tuple[AxisKey, AxisKey] = (-2, -1)
    current_index: IndexMap = Field(default_factory=IndexMap, frozen=True)
    channel_axis: AxisKey | None = None

    # map of axis to reducer (function that can reduce dimensionality along that axis)
    reducers: Reducers = Field(default_factory=Reducers, frozen=True)
    default_reducer: ReducerType = "numpy.max"  # type: ignore [assignment]  # FIXME

    # map of index along channel axis to LUTModel object
    luts: LutMap = Field(default_factory=LutMap)
    default_lut: LUTModel = Field(default_factory=LUTModel, frozen=True)

    @computed_field  # type: ignore [prop-decorator]
    @property
    def n_visible_axes(self) -> Literal[2, 3]:
        """Number of dims is derived from the length of `visible_axes`."""
        return cast(Literal[2, 3], len(self.visible_axes))

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # prevent channel_axis from being in visible_axes
        if self.channel_axis in self.visible_axes:
            warnings.warn(
                "Channel_axis cannot be in visible_axes. Setting channel_axis to None.",
                UserWarning,
                stacklevel=2,
            )
            self.channel_axis = None
        return self
