from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pygfx

from ndv.models import _scene as core

from ._node import Node

if TYPE_CHECKING:
    from cmap import Color

# FIXME: I broke this in the last commit.  attribute errors all over


class View(Node, core.view.ViewAdaptorProtocol):
    """View interface for pygfx Backend."""

    # _native: scene.ViewBox
    # TODO: i think pygfx doesn't see a view as part of the scene like vispy does
    _pygfx_cam: pygfx.Camera

    def __init__(self, view: core.View, **backend_kwargs: Any) -> None:
        # FIXME:  hardcoded camera and scene
        self.scene = pygfx.Scene()

    # XXX: both of these methods deserve scrutiny, and fixing :)
    def _vis_set_camera(self, cam: core.Camera) -> None:
        pygfx_cam = cam.backend_adaptor("pygfx")._vis_get_native()

        if not isinstance(pygfx_cam, pygfx.Camera):
            raise TypeError(f"cam must be a pygfx.Camera, got {type(pygfx_cam)}")
        self._pygfx_cam = pygfx_cam

    def _vis_set_scene(self, scene: core.Scene) -> None:
        # XXX: Tricky!  this call to scene.native actually has the side effect of
        # creating the backend adaptor for the scene!  That needs to be more explicit.
        pygfx_scene = scene.backend_adaptor("pygfx")._vis_get_native()

        if not isinstance(pygfx_scene, pygfx.Scene):
            raise TypeError("Scene must be a pygfx.Scene")
        self.scene = pygfx_scene

    def _vis_set_position(self, arg: tuple[float, float]) -> None:
        warnings.warn(
            "set_position not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_size(self, arg: tuple[float, float] | None) -> None:
        warnings.warn(
            "set_size not implemented for pygfx", RuntimeWarning, stacklevel=2
        )

    def _vis_set_background_color(self, arg: Color | None) -> None:
        warnings.warn(
            "set_background_color not implemented for pygfx",
            RuntimeWarning,
            stacklevel=2,
        )

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

    def _visit(self, viewport: pygfx.Viewport) -> None:
        viewport.render(self.scene, self._pygfx_cam)
        self._pygfx_cam.update_controller()
