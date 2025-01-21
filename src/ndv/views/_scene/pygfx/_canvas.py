from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Union, cast

import pygfx

from ndv.models import _scene as core

if TYPE_CHECKING:
    import numpy as np
    from cmap import Color
    from typing_extensions import TypeAlias
    from wgpu.gui import glfw, jupyter, offscreen, qt

    from ._view import View

    # from wgpu.gui.auto import WgpuCanvas
    # ... will result in one of the following canvas classes
    TypeWgpuCanvasType: TypeAlias = Union[
        type[offscreen.WgpuCanvas],  # if WGPU_FORCE_OFFSCREEN=1
        type[jupyter.WgpuCanvas],  # if is_jupyter()
        type[glfw.WgpuCanvas],  # if glfw is installed
        type[qt.WgpuCanvas],  # if any pyqt backend is installed
    ]
    # TODO: lol... there's probably a better way to do this :)
    WgpuCanvasType: TypeAlias = Union[
        offscreen.WgpuCanvas,  # if WGPU_FORCE_OFFSCREEN=1
        jupyter.WgpuCanvas,  # if is_jupyter()
        glfw.WgpuCanvas,  # if glfw is installed
        qt.WgpuCanvas,  # if any pyqt backend is installed
    ]


class Canvas(core.canvas.CanvasAdaptorProtocol):
    """Canvas interface for pygfx Backend."""

    def __init__(self, canvas: core.Canvas, **backend_kwargs: Any) -> None:
        # wgpu.gui.auto.WgpuCanvas is a "magic" import that itself is context sensitive
        # see TYPE_CHECKING section above for details
        from wgpu.gui.auto import WgpuCanvas

        WgpuCanvas = cast("TypeWgpuCanvasType", WgpuCanvas)

        canvas = WgpuCanvas(size=canvas.size, title=canvas.title, **backend_kwargs)
        self._wgpu_canvas = cast("WgpuCanvasType", canvas)
        # TODO: background_color
        # the qt backend, this shows by default...
        # if we need to prevent it, we could potentially monkeypatch during init.
        if hasattr(self._wgpu_canvas, "hide"):
            self._wgpu_canvas.hide()

        self._renderer = pygfx.renderers.WgpuRenderer(self._wgpu_canvas)
        self._viewport: pygfx.Viewport = pygfx.Viewport(self._renderer)
        self._views: list[View] = []
        # self._grid: dict[tuple[int, int], View] = {}

    def _vis_get_native(self) -> WgpuCanvasType:
        return self._wgpu_canvas

    def _vis_set_visible(self, arg: bool) -> None:
        if hasattr(self._wgpu_canvas, "show"):
            self._wgpu_canvas.show()
        self._wgpu_canvas.request_draw(self._animate)

    def _animate(self, viewport: pygfx.Viewport | None = None) -> None:
        vp = viewport or self._viewport
        for view in self._views:
            view._visit(vp)
        if hasattr(vp.renderer, "flush"):
            # an attribute error can occur if flush() is called before render()
            # https://github.com/pygfx/pygfx/issues/946
            with suppress(AttributeError):
                vp.renderer.flush()
        if viewport is None:
            self._wgpu_canvas.request_draw()

    def _vis_add_view(self, view: core.View) -> None:
        adaptor = cast("View", view.backend_adaptor())
        # adaptor._pygfx_cam.set_viewport(self._viewport)
        self._views.append(adaptor)

    def _vis_set_width(self, arg: int) -> None:
        _, height = self._wgpu_canvas.get_logical_size()
        self._wgpu_canvas.set_logical_size(arg, height)

    def _vis_set_height(self, arg: int) -> None:
        width, _ = self._wgpu_canvas.get_logical_size()
        self._wgpu_canvas.set_logical_size(width, arg)

    def _vis_set_background_color(self, arg: Color | None) -> None:
        raise NotImplementedError()

    def _vis_set_title(self, arg: str) -> None:
        raise NotImplementedError()

    def _vis_close(self) -> None:
        """Close canvas."""
        self._wgpu_canvas.close()

    def _vis_render(
        self,
        region: tuple[int, int, int, int] | None = None,
        size: tuple[int, int] | None = None,
        bgcolor: Color | None = None,
        crop: np.ndarray | tuple[int, int, int, int] | None = None,
        alpha: bool = True,
    ) -> np.ndarray:
        """Render to screenshot."""
        from wgpu.gui.offscreen import WgpuCanvas

        w, h = self._wgpu_canvas.get_logical_size()
        canvas = WgpuCanvas(width=w, height=h, pixel_ratio=1)
        renderer = pygfx.renderers.WgpuRenderer(canvas)
        viewport = pygfx.Viewport(renderer)
        canvas.request_draw(lambda: self._animate(viewport))
        return cast("np.ndarray", canvas.draw())
