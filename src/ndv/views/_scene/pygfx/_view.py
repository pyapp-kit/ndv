from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import pygfx

from ndv.models import _scene as core

if TYPE_CHECKING:
    from cmap import Color

    from . import _camera, _canvas, _scene


class View(core.view.ViewAdaptorProtocol):
    """View interface for pygfx Backend.

    A view combines a scene and a camera to render a scene (onto a canvas).
    """

    _pygfx_scene: pygfx.Scene
    _pygfx_cam: pygfx.Camera

    def __init__(self, view: core.View, **backend_kwargs: Any) -> None:
        canvas_adaptor = cast("_canvas.Canvas", view.canvas.backend_adaptor("pygfx"))
        wgpu_canvas = canvas_adaptor._vis_get_native()
        self._renderer = pygfx.renderers.WgpuRenderer(wgpu_canvas)

        self._vis_set_scene(view.scene)
        self._vis_set_camera(view.camera)

    def _vis_get_native(self) -> pygfx.Viewport:
        return pygfx.Viewport(self._renderer)

    def _vis_set_visible(self, arg: bool) -> None:
        pass

    def _vis_set_scene(self, scene: core.Scene) -> None:
        self._scene_adaptor = cast("_scene.Scene", scene.backend_adaptor("pygfx"))
        self._pygfx_scene = self._scene_adaptor._pygfx_node

    def _vis_set_camera(self, cam: core.Camera) -> None:
        self._cam_adaptor = cast("_camera.Camera", cam.backend_adaptor("pygfx"))
        self._pygfx_cam = self._cam_adaptor._pygfx_node
        self._cam_adaptor.pygfx_controller.register_events(self._renderer)

    def _draw(self) -> None:
        renderer = self._renderer
        renderer.render(self._pygfx_scene, self._pygfx_cam)
        renderer.request_draw()

    def _vis_set_position(self, arg: tuple[float, float]) -> None:
        warnings.warn(
            "set_position not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_size(self, arg: tuple[float, float] | None) -> None:
        warnings.warn(
            "set_size not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_background_color(self, color: Color | None) -> None:
        colors = (color.rgba,) if color is not None else ()
        background = pygfx.Background(None, material=pygfx.BackgroundMaterial(*colors))
        self._pygfx_scene.add(background)

    def _vis_set_border_width(self, arg: float) -> None:
        warnings.warn(
            "set_border_width not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_border_color(self, arg: Color | None) -> None:
        warnings.warn(
            "set_border_color not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_padding(self, arg: int) -> None:
        warnings.warn(
            "set_padding not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_margin(self, arg: int) -> None:
        warnings.warn(
            "set_margin not implemented for pygfx", RuntimeWarning, stacklevel=2
        )
