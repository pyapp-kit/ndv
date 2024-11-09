"""General model for ndv."""

from collections.abc import Hashable, Sequence
from contextlib import suppress
from typing import Annotated, Any, TypeAlias

from pydantic import PlainValidator


def _maybe_int(val: Any) -> Any:
    # try to convert to int if possible
    with suppress(ValueError, TypeError):
        val = int(float(val))
    return val


def _to_slice(val: Any) -> slice:
    # slices are returned as is
    if isinstance(val, slice):
        if not all(
            isinstance(i, (int, type(None))) for i in (val.start, val.stop, val.step)
        ):
            raise TypeError(f"Slice start/stop/step must all be integers: {val!r}")
        return val
    # single integers are converted to slices starting at that index
    if isinstance(val, int):
        return slice(val, val + 1)
    # sequences are interpreted as arguments to the slice constructor
    if isinstance(val, Sequence):
        return slice(*(int(x) if x is not None else None for x in val))
    raise TypeError(f"Expected int or slice, got {type(val)}")


Slice = Annotated[slice, PlainValidator(_to_slice)]

# An axis key is any hashable object that can be used to index an axis
# In many cases it will be an integer, but for some labeled arrays it may be a string
# or other hashable object.  It is up to the DataWrapper to convert these keys to
# actual integer indices.
AxisKey: TypeAlias = Annotated[Hashable, PlainValidator(_maybe_int)]
