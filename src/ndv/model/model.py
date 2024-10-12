"""General model for ndv."""

import warnings
from collections.abc import Sequence
from contextlib import suppress
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    Self,
    SupportsIndex,
    TypeAlias,
    cast,
)

import numpy as np
import numpy.typing as npt
from cmap import Colormap
from psygnal import SignalGroupDescriptor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    computed_field,
    model_validator,
)
from pydantic_core import core_schema

from ._mapping import ValidatedEventedDict

_ShapeLike = SupportsIndex | Sequence[SupportsIndex]

# The Term "Axis" is used when referring to a specific dimension of an array
# We can change this to dimension if desired... but it should be uniform.

# Axis keys can be either a direct integer index or name (for labeled arrays)
# we leave it to the DataWrapper to convert `AxisKey -> AxisIndex`
AxisIndex: TypeAlias = int
AxisLabel: TypeAlias = str


def _maybe_int(val: Any) -> Any:
    # prefer integers, but allow strings
    with suppress(ValueError):
        val = int(float(val))
    return val


AxisKey: TypeAlias = Annotated[AxisIndex | AxisLabel, PlainValidator(_maybe_int)]

# a specific position along a dimension
CoordIndex: TypeAlias = int | str


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


class Reducer(Protocol):
    """Function to reduce an array along an axis.

    A reducer is any function that takes an array-like, and an optional axis argument,
    and returns a reduced array.  Examples include `np.max`, `np.mean`, etc.
    """

    def __call__(self, a: npt.ArrayLike, axis: _ShapeLike = ...) -> npt.ArrayLike:
        """Reduce an array along an axis."""


def _str_to_callable(obj: Any) -> Callable:
    """Deserialize a callable from a string."""
    if isinstance(obj, str):
        # e.g. "numpy.max" -> numpy.max
        try:
            mod_name, qual_name = obj.rsplit(".", 1)
            mod = __import__(mod_name, fromlist=[qual_name])
            obj = getattr(mod, qual_name)
        except Exception:
            try:
                # fallback to numpy
                # e.g. "max" -> numpy.max
                obj = getattr(np, obj)
            except Exception:
                raise

    if not callable(obj):
        raise TypeError(f"Expected a callable or string, got {type(obj)}")
    return cast("Callable", obj)


def _callable_to_str(obj: str | Callable) -> str:
    """Serialize a callable to a string."""
    if isinstance(obj, str):
        return obj
    # e.g. numpy.max -> "numpy.max"
    return f"{obj.__module__}.{obj.__qualname__}"


class ReducerType(Reducer):
    """Reducer type for pydantic.

    This just provides a pydantic core schema for a generic callable that accepts an
    array and an axis argument and returns an array (of reduced dimensionality).
    This serializes/deserializes the callable as a string (module.qualname).
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        """Get the Pydantic schema for this object."""
        ser_schema = core_schema.plain_serializer_function_ser_schema(_callable_to_str)
        return core_schema.no_info_before_validator_function(
            _str_to_callable,
            # using callable_schema() would be more correct, but prevents dumping schema
            core_schema.any_schema(),
            serialization=ser_schema,
        )


class _NDVModel(BaseModel):
    """Base eventd model for NDV models."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)
    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()


class LUTModel(_NDVModel):
    """Representation of how to display a channel of an array.

    Parameters
    ----------
    visible : bool
        Whether to display this channel.
        NOTE: This has implications for data retrieval, as we may not want to request
        channels that are not visible.  See current_index above.
    cmap : Colormap
        Colormap to use for this channel.
    clims : tuple[float, float] | None
        Contrast limits for this channel.
        TODO: What does `None` imply?  Autoscale?
    gamma : float
        Gamma correction for this channel. By default, 1.0.
    autoscale : bool | tuple[float, float]
        Whether/how to autoscale the colormap.
        If `False`, then autoscaling is disabled.
        If `True` or `(0, 1)` then autoscale using the min/max of the data.
        If a tuple, then the first element is the lower quantile and the second element
        is the upper quantile.
        If a callable, then it should be a function that takes an array and returns a
        tuple of (min, max) values to use for scaling.

        NaN values should be ignored (n.b. nanmax is slower and should only be used if
        necessary).
    """

    visible: bool = True
    cmap: Colormap = Field(default_factory=lambda: Colormap("gray"))
    clims: tuple[float, float] | None = None
    gamma: float = 1.0
    autoscale: (
        bool | tuple[float, float] | Callable[[npt.ArrayLike], tuple[float, float]]
    ) = (0, 1)

    @model_validator(mode="before")
    def _validate_model(cls, v: Any) -> Any:
        # cast bare string/colormap inputs to cmap declaration
        if isinstance(v, (str, Colormap)):
            return {"cmap": v}
        return v


# map of axis to index/slice ... i.e. the current subset of data being displayed
IndexMap: TypeAlias = ValidatedEventedDict[AxisKey, int | Slice]
# map of index along channel axis to LUTModel object
LutMap: TypeAlias = ValidatedEventedDict[CoordIndex | None, LUTModel]
# map of axis to reducer
Reducers: TypeAlias = ValidatedEventedDict[AxisKey | None, ReducerType]


class ArrayDisplayModel(_NDVModel):
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
    current_index: IndexMap = Field(default_factory=IndexMap)
    channel_axis: AxisKey | None = None

    # map of axis to reducer
    reducers: Reducers = Field(default_factory=Reducers)
    default_reducer: ReducerType = "numpy.max"  # type: ignore [assignment]  # FIXME
    # map of index along channel axis to LUTModel object
    luts: LutMap = Field(default_factory=LutMap)
    default_lut: LUTModel = Field(default_factory=LUTModel, frozen=True)

    @computed_field
    def n_visible_axes(self) -> Literal[2, 3]:
        """Number of dims is derived from the length of `visible_axes`."""
        return cast(Literal[2, 3], len(self.visible_axes))

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        # prevent channel_axis from being in visible_axes
        if self.channel_axis in self.visible_axes:
            warnings.warn(
                f"Channel_axis cannot be in visible_axes. "
                f"Removing {self.channel_axis!r} from visible_axes",
                UserWarning,
                stacklevel=2,
            )
            self.channel_axis = None
        return self
