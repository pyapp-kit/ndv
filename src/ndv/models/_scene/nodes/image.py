from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol

from cmap import Colormap
from pydantic import Field

from ndv._types import ImageInterpolation
from ndv.models._lut_model import ClimsMinMax, ClimsType

from .node import Node, NodeAdaptorProtocol

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ImageBackend(NodeAdaptorProtocol["Image"], Protocol):
    """Protocol for a backend Image adaptor object."""

    @abstractmethod
    def _vis_set_data(self, arg: NDArray) -> None: ...
    @abstractmethod
    def _vis_set_cmap(self, arg: Colormap) -> None: ...
    @abstractmethod
    def _vis_set_clim(self, arg: tuple[float, float] | None) -> None: ...
    @abstractmethod
    def _vis_set_gamma(self, arg: float) -> None: ...
    @abstractmethod
    def _vis_set_interpolation(self, arg: ImageInterpolation) -> None: ...


class Image(Node[ImageBackend]):
    """A Image that can be placed in scene."""

    node_type: Literal["image"] = "image"

    data: Any = None
    cmap: Colormap = Field(
        default_factory=lambda: Colormap("gray"),
        description="The colormap to use for the image.",
    )
    clims: ClimsType = Field(discriminator="clim_type", default_factory=ClimsMinMax)
    gamma: float = Field(default=1.0, description="The gamma correction to use.")
    interpolation: ImageInterpolation = Field(
        default=ImageInterpolation.NEAREST,
        description="The interpolation to use.",
    )
