"""Model protocols for data display."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    computed_field,
    model_validator,
)
from pydantic_core import core_schema

from ndv.models._base_model import NDVModel

if TYPE_CHECKING:
    from typing import Any

    from pydantic.json_schema import JsonSchemaValue


# copied from https://github.com/tlambert03/microsim
class _NumpyNdarrayPydanticAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        def validate_from_any(value: Any) -> np.ndarray:
            try:
                return np.asarray(value)
            except Exception as e:
                raise ValueError(f"Cannot cast {value} to numpy.ndarray: {e}") from e

        from_any_schema = core_schema.chain_schema(
            [
                core_schema.any_schema(),
                core_schema.no_info_plain_validator_function(validate_from_any),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_any_schema,
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(np.ndarray),
                    from_any_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.tolist()
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # Use the same schema that would be used for arrays
        return handler(core_schema.list_schema(core_schema.any_schema()))


NumpyNdarray = Annotated[np.ndarray, _NumpyNdarrayPydanticAnnotation]


class StatsModel(NDVModel):
    """Representation of the statistics of a dataset.

    A model that computes and caches statistical properties of a dataset,
    including standard deviation, average, and histogram.

    Those interested in statistics should listen to the events.data and events.bins
    signals emitted by this object.

    TODO can we only have the data signal?

    Parameters
    ----------
    data : np.ndarray | None
        The dataset.
    bins : int
        Number of bins to use for histogram computation. Defaults to 256.
    average : float
        The average (mean) value of data.
    standard_deviation : float
        The standard deviation of data.
    histogram : tuple[Sequence[int], Sequence[float]]
        A 2-tuple of sequences.

        The first sequence contains (n) integers, where index i is the number of data
        points in the ith bin.

        The second sequence contains (n+1) floats. The ith bin spans the domain
        between the values at index i (inclusive) and index i+1 (exclusive).
    """

    data: NumpyNdarray | None = None
    bins: int = 256

    @model_validator(mode="before")
    def validate_data(cls, input: dict[str, Any], *args: Any) -> dict[str, Any]:
        """Delete computed fields when data changes."""
        # Recompute computed stats when bins/data changes
        if "data" in input:
            for field in ["average", "standard_deviation", "histogram"]:
                if field in input:
                    del input[field]
        return input

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def standard_deviation(self) -> float:
        """Computes the standard deviation of the dataset."""
        if self.data is None:
            return float("nan")
        return float(np.std(self.data))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def average(self) -> float:
        """Computes the average of the dataset."""
        if self.data is None:
            return float("nan")
        return float(np.mean(self.data))

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def histogram(self) -> tuple[Sequence[int], Sequence[float]]:
        """Computes the histogram of the dataset."""
        if self.data is None:
            return ([], [])
        return cast(
            tuple[Sequence[int], Sequence[float]],
            np.histogram(self.data, bins=self.bins),
        )
