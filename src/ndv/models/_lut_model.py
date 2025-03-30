from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from annotated_types import Gt, Interval
from cmap import Colormap
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAlias

from ._base_model import NDVModel

AutoscaleType: TypeAlias = Union[
    Callable[[npt.ArrayLike], tuple[float, float]], tuple[float, float], bool
]


class ClimPolicy(BaseModel, ABC):
    """ABC for contrast limit policies."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    _cached_clims: Optional[tuple[float, float]] = PrivateAttr(None)

    @abstractmethod
    def get_limits(self, image: npt.NDArray) -> tuple[float, float]:
        """Return the contrast limits for the given image."""

    def calc_clims(self, image: npt.NDArray) -> tuple[float, float]:
        self._cached_clims = value = self.get_limits(image)
        return value

    @property
    def cached_clims(self) -> Optional[tuple[float, float]]:
        """Return the last calculated clims."""
        return self._cached_clims

    @property
    def is_manual(self) -> bool:
        return self.__class__ == ClimsManual


class ClimsManual(ClimPolicy):
    """Manually specified contrast limits.

    Attributes
    ----------
    min: float
        The minimum contrast limit.
    max: float
        The maximum contrast limit.
    """

    clim_type: Literal["manual"] = "manual"
    min: float
    max: float

    def get_limits(self, data: npt.NDArray) -> tuple[float, float]:
        return self.min, self.max

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ClimsManual)
            and self.min == other.min
            and self.max == other.max
        )


class ClimsMinMax(ClimPolicy):
    """Autoscale contrast limits based on the minimum and maximum values in the data."""

    clim_type: Literal["minmax"] = "minmax"

    def get_limits(self, data: npt.NDArray) -> tuple[float, float]:
        return (np.nanmin(data), np.nanmax(data))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClimsMinMax)


class ClimsPercentile(ClimPolicy):
    """Autoscale contrast limits based on percentiles of the data.

    Attributes
    ----------
    min_percentile: float
        The lower percentile for the contrast limits.
    max_percentile: float
        The upper percentile for the contrast limits.
    """

    clim_type: Literal["percentile"] = "percentile"
    min_percentile: Annotated[float, Interval(ge=0, le=100)] = 0
    max_percentile: Annotated[float, Interval(ge=0, le=100)] = 100

    def get_limits(self, data: npt.NDArray) -> tuple[float, float]:
        return tuple(np.nanpercentile(data, [self.min_percentile, self.max_percentile]))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ClimsPercentile)
            and self.min_percentile == other.min_percentile
            and self.max_percentile == other.max_percentile
        )


class ClimsStdDev(ClimPolicy):
    """Automatically set contrast limits based on standard deviations from the mean.

    Attributes
    ----------
    n_stdev: float
        Number of standard deviations to use.
    center: Optional[float]
        Center value for the standard deviation calculation. If None, the mean is
        used.
    """

    clim_type: Literal["stddev"] = "stddev"
    n_stdev: Annotated[float, Gt(0)] = 2  # number of standard deviations
    center: Optional[float] = None  # None means center around the mean

    def get_limits(self, data: npt.NDArray) -> tuple[float, float]:
        center = np.nanmean(data) if self.center is None else self.center
        diff = self.n_stdev * np.nanstd(data)
        return center - diff, center + diff

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ClimsStdDev)
            and self.n_stdev == other.n_stdev
            and self.center == other.center
        )


# we can add this, but it needs to have a proper pydantic serialization method
# similar to ReducerType
# class CustomClims(ClimPolicy):
#     type_: Literal["custom"] = "custom"
#     func: Callable[[npt.ArrayLike], tuple[float, float]]

#     def get_limits(self, data: npt.NDArray) -> tuple[float, float]:
#         return self.func(data)


ClimsType = Union[ClimsManual, ClimsPercentile, ClimsStdDev, ClimsMinMax]


class LUTModel(NDVModel):
    """Representation of how to display a channel of an array.

    Attributes
    ----------
    visible : bool
        Whether to display this channel.
        NOTE: This has implications for data retrieval, as we may not want to request
        channels that are not visible.
        See [`ArrayDisplayModel.current_index`][ndv.models.ArrayDisplayModel].
    cmap : cmap.Colormap
        [`cmap.Colormap`](https://cmap-docs.readthedocs.io/colormaps/) to use for this
        channel.  This can be expressed as any channel.  This can be expressed as any
        ["colormap like" object](https://cmap-docs.readthedocs.io/en/latest/colormaps/#colormaplike-objects)
    clims : Union[ClimsManual, ClimsPercentile, ClimsStdDev, ClimsMinMax]
        Method for determining the contrast limits for this channel.  If a 2-element
        `tuple` or `list` is provided, it is interpreted as a manual contrast limit.
    gamma : float
        Gamma applied to the data before applying the colormap. By default, `1.0`.
    """

    visible: bool = True
    cmap: Colormap = Field(default_factory=lambda: Colormap("gray"))
    clims: ClimsType = Field(discriminator="clim_type", default_factory=ClimsMinMax)
    gamma: float = 1.0

    @model_validator(mode="before")
    def _validate_model(cls, v: Any) -> Any:
        # cast bare string/colormap inputs to cmap declaration
        if isinstance(v, (str, Colormap)):
            return {"cmap": v}
        return v

    @field_validator("clims", mode="before")
    @classmethod
    def _validate_clims(cls, v: ClimsType) -> ClimsType:
        if v is None or (
            isinstance(v, dict)
            and v.get("min_percentile") == 0
            and v.get("max_percentile") == 100
        ):
            return ClimsMinMax()
        if isinstance(v, (tuple, list, np.ndarray)):
            if len(v) == 2:
                return ClimsManual(min=v[0], max=v[1])
            raise ValueError("Clims sequence must have exactly 2 elements.")
        return v
