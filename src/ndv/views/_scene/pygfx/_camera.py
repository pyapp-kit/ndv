from __future__ import annotations

import warnings
from typing import Any

import pygfx

from ndv._types import CameraType
from ndv.models._scene.nodes import camera

from ._node import Node


class Camera(Node, camera.CameraAdaptorProtocol):
    """Adaptor for pygfx camera."""

    _pygfx_node: pygfx.Camera

    def __init__(self, camera: camera.Camera, **backend_kwargs: Any) -> None:
        if camera.type == CameraType.PANZOOM:
            self._pygfx_node = pygfx.OrthographicCamera(**backend_kwargs)
            self.controller = pygfx.PanZoomController(self._pygfx_node)
        elif camera.type == CameraType.ARCBALL:
            self._pygfx_node = pygfx.PerspectiveCamera(**backend_kwargs)
            self.controller = pygfx.OrbitOrthoController(self._pygfx_node)

        # FIXME: hardcoded
        # self._pygfx_cam.scale.y = -1

    def _vis_set_zoom(self, zoom: float) -> None:
        raise NotImplementedError

    def _vis_set_center(self, arg: tuple[float, ...]) -> None:
        raise NotImplementedError

    def _vis_set_type(self, arg: CameraType) -> None:
        raise NotImplementedError

    def _view_size(self) -> tuple[float, float] | None:
        """Return the size of first parent viewbox in pixels."""
        raise NotImplementedError

    def update_controller(self) -> None:
        # This is called by the View Adaptor in the `_visit` method
        # ... which is in turn called by the Canvas backend adaptor's `_animate` method
        # i.e. the main render loop.
        self.controller.update_camera(self._pygfx_node)

    def set_viewport(self, viewport: pygfx.Viewport) -> None:
        # This is used by the Canvas backend adaptor...
        # and should perhaps be moved to the View Adaptor
        self.controller.add_default_event_handlers(viewport, self._pygfx_node)

    def _vis_set_range(self, margin: float) -> None:
        warnings.warn(
            "set_range not implemented for pygfx", RuntimeWarning, stacklevel=2
        )
