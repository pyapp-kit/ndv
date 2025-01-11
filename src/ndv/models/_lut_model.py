from typing import Any, Callable, Optional, Union

import numpy.typing as npt
from cmap import Colormap
from pydantic import Field, model_validator
from typing_extensions import TypeAlias

from ._base_model import NDVModel

AutoscaleType: TypeAlias = Union[
    Callable[[npt.ArrayLike], tuple[float, float]], tuple[float, float], bool
]


class LUTModel(NDVModel):
    """Representation of how to display a channel of an array.

    Attributes
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
    clims: Optional[tuple[float, float]] = None
    gamma: float = 1.0
    autoscale: AutoscaleType = Field(default=True, union_mode="left_to_right")

    @model_validator(mode="before")
    def _validate_model(cls, v: Any) -> Any:
        # cast bare string/colormap inputs to cmap declaration
        if isinstance(v, (str, Colormap)):
            return {"cmap": v}
        return v
