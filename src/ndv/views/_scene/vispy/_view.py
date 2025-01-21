from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vispy import scene
from vispy.scene import subscene

from ndv.models import _scene as core

from ._node import Node
from ._util import pyd_color_to_vispy

if TYPE_CHECKING:
    from cmap import Color

logger = logging.getLogger(__name__)


class View(Node, core.view.ViewAdaptorProtocol):
    """View interface for Vispy Backend."""

    _vispy_node: scene.ViewBox

    def __init__(self, view: core.View, **backend_kwargs: Any) -> None:
        backend_kwargs.update(
            {
                "pos": view.layout.position,
                "border_color": pyd_color_to_vispy(view.layout.border_color),
                "border_width": view.layout.border_width,
                "bgcolor": pyd_color_to_vispy(view.layout.background_color),
                "padding": view.layout.padding,
                "margin": view.layout.margin,
            }
        )
        if (size := view.layout.size) is not None:
            backend_kwargs["size"] = size
        self._vispy_node = scene.ViewBox(**backend_kwargs)

    def _vis_set_camera(self, cam: core.Camera) -> None:
        vispy_cam = cam.backend_adaptor("vispy")._vis_get_native()
        if not isinstance(vispy_cam, scene.cameras.BaseCamera):
            raise TypeError("Camera must be a Vispy Camera")
        try:
            # hitting singular matrix here... probably a bad order of operations
            self._vispy_node.camera = vispy_cam
            # vispy_cam.set_range(margin=0)  # TODO: put this elsewhere
        except Exception as e:
            logger.error("Error setting camera: %s", e)

    def _vis_set_scene(self, scene: core.Scene) -> None:
        vispy_scene = scene.backend_adaptor("vispy")._vis_get_native()
        if not isinstance(vispy_scene, subscene.SubScene):
            raise TypeError("Scene must be a Vispy SubScene")

        self._vispy_node._scene = vispy_scene
        vispy_scene.parent = self._vispy_node

    def _vis_set_position(self, arg: tuple[float, float]) -> None:
        self._vispy_node.pos = arg

    def _vis_set_size(self, arg: tuple[float, float] | None) -> None:
        self._vispy_node.size = arg

    def _vis_set_background_color(self, arg: Color | None) -> None:
        self._vispy_node.bgcolor = pyd_color_to_vispy(arg)

    def _vis_set_border_width(self, arg: float) -> None:
        self._vispy_node._border_width = arg
        self._vispy_node._update_line()
        self._vispy_node.update()

    def _vis_set_border_color(self, arg: Color | None) -> None:
        self._vispy_node.border_color = pyd_color_to_vispy(arg)

    def _vis_set_padding(self, arg: int) -> None:
        self._vispy_node.padding = arg

    def _vis_set_margin(self, arg: int) -> None:
        self._vispy_node.margin = arg
