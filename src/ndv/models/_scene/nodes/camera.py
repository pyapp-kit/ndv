from __future__ import annotations

from abc import abstractmethod
from typing import Literal, Protocol

from pydantic import Field

from ndv._types import CameraType

from .node import Node, NodeAdaptorProtocol


class CameraAdaptorProtocol(NodeAdaptorProtocol["Camera"], Protocol):
    """Protocol for a backend camera adaptor object."""

    @abstractmethod
    def _vis_set_type(self, arg: CameraType) -> None: ...
    @abstractmethod
    def _vis_set_zoom(self, arg: float) -> None: ...
    @abstractmethod
    def _vis_set_center(self, arg: tuple[float, ...]) -> None: ...
    @abstractmethod
    def _vis_set_range(self, margin: float) -> None: ...


class Camera(Node["CameraAdaptorProtocol"]):
    """A camera that defines the view of a scene."""

    node_type: Literal["camera"] = "camera"

    type: CameraType = Field(default=CameraType.PANZOOM, description="Camera type.")
    interactive: bool = Field(
        default=True,
        description="Whether the camera responds to user interaction, "
        "such as mouse and keyboard events.",
    )
    zoom: float = Field(default=1.0, description="Zoom factor of the camera.")
    center: tuple[float, float, float] | tuple[float, float] = Field(
        default=(0, 0, 0), description="Center position of the view."
    )

    def _set_range(self, margin: float = 0) -> None:
        adaptor = self.backend_adaptor()
        # TODO: this method should probably be pulled off of the backend,
        # calculated directly in the core, and then applied as a change to the
        # camera transform
        adaptor._vis_set_range(margin=margin)
