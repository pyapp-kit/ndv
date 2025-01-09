from collections.abc import Sequence
from typing import Any, Callable, Protocol, SupportsIndex, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic_core import core_schema
from typing_extensions import TypeAlias

_ShapeLike: TypeAlias = Union[SupportsIndex, Sequence[SupportsIndex]]


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


def _callable_to_str(obj: Union[str, Callable]) -> str:
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
