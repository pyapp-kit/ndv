from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from cmap import Color
from pydantic import ConfigDict, Field

from ._vis_model import ModelBase, SupportsVisibility, VisModel
from .nodes import Camera, Scene
from .nodes.node import Node

if TYPE_CHECKING:
    from psygnal import EmissionInfo

    from .canvas import Canvas

NodeType = TypeVar("NodeType", bound=Node)
logger = logging.getLogger(__name__)


class ViewAdaptorProtocol(SupportsVisibility["View"], Protocol):
    """Protocol defining the interface for a View adaptor."""

    @abstractmethod
    def _vis_set_camera(self, arg: Camera) -> None: ...
    @abstractmethod
    def _vis_set_scene(self, arg: Scene) -> None: ...
    @abstractmethod
    def _vis_set_position(self, arg: tuple[float, float]) -> None: ...
    @abstractmethod
    def _vis_set_size(self, arg: tuple[float, float] | None) -> None: ...
    @abstractmethod
    def _vis_set_background_color(self, arg: Color | None) -> None: ...
    @abstractmethod
    def _vis_set_border_width(self, arg: float) -> None: ...
    @abstractmethod
    def _vis_set_border_color(self, arg: Color | None) -> None: ...
    @abstractmethod
    def _vis_set_padding(self, arg: int) -> None: ...
    @abstractmethod
    def _vis_set_margin(self, arg: int) -> None: ...

    def _vis_set_layout(self, arg: Layout) -> None:
        pass


class Layout(ModelBase):
    """Rectangular layout model.

        y
        |
        v
    x-> +--------------------------------+  ^
        |            margin              |  |
        |  +--------------------------+  |  |
        |  |         border           |  |  |
        |  |  +--------------------+  |  |  |
        |  |  |      padding       |  |  |  |
        |  |  |  +--------------+  |  |  |   height
        |  |  |  |   content    |  |  |  |  |
        |  |  |  |              |  |  |  |  |
        |  |  |  +--------------+  |  |  |  |
        |  |  +--------------------+  |  |  |
        |  +--------------------------+  |  |
        +--------------------------------+  v

        <------------ width ------------->
    """

    x: float = Field(
        default=0, description="The x-coordinate of the object (wrt parent)."
    )
    y: float = Field(
        default=0, description="The y-coordinate of the object (wrt parent)."
    )
    width: float = Field(default=0, description="The width of the object.")
    height: float = Field(default=0, description="The height of the object.")
    background_color: Color | None = Field(
        default=Color("black"),
        description="The background color (inside of the border). "
        "None implies transparent.",
    )
    border_width: float = Field(
        default=0, description="The width of the border in pixels."
    )
    border_color: Color | None = Field(
        default=Color("black"), description="The color of the border."
    )
    padding: int = Field(
        default=0,
        description="The amount of padding in the widget "
        "(i.e. the space reserved between the contents and the border).",
    )
    margin: int = Field(
        default=0, description="he margin to keep outside the widget's border"
    )

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    @property
    def size(self) -> tuple[float, float]:
        return self.width, self.height


class View(VisModel[ViewAdaptorProtocol]):
    """A rectangular area on a canvas that displays a scene, with a camera.

    A canvas can have one or more views. Each view has a single scene (i.e. a
    scene graph of nodes) and a single camera. The camera defines the view
    transformation.  This class just exists to associate a single scene and
    camera.
    """

    scene: Scene = Field(default_factory=Scene)
    camera: Camera = Field(default_factory=Camera)
    layout: Layout = Field(default_factory=Layout, frozen=True)

    model_config = ConfigDict(repr_exclude_defaults=False)  # type: ignore

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.layout.events.connect(self._on_layout_event)

    def _on_layout_event(self, info: EmissionInfo) -> None:
        _signal_name = info.signal.name
        ...

    def show(self) -> Canvas:
        """Show the view.

        Convenience method for showing the canvas that the view is on.
        If no canvas exists, a new one is created.
        """
        if not hasattr(self, "_canvas"):
            from .canvas import Canvas

            # TODO: we need to know/check somehow if the view is already on a canvas
            # This just creates a new canvas every time
            self._canvas = Canvas()
            self._canvas.add_view(self)
        self._canvas.show()
        return self._canvas

    def add_node(self, node: NodeType) -> NodeType:
        """Add any node to the scene."""
        self.scene.add(node)
        self.camera._set_range(margin=0)
        return node

    def _create_adaptor(self, cls: type[ViewAdaptorProtocol]) -> ViewAdaptorProtocol:
        adaptor = super()._create_adaptor(cls)
        logger.debug("VIEW Setting scene %r and camera %r", self.scene, self.camera)
        adaptor._vis_set_scene(self.scene)
        adaptor._vis_set_camera(self.camera)
        return adaptor
