from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ndv.models import _scene as core

if TYPE_CHECKING:
    import numpy as np
    from cmap import Color
    from rendercanvas.auto import RenderCanvas

    from ._view import View


class Canvas(core.canvas.CanvasAdaptorProtocol):
    """Canvas interface for pygfx Backend."""

    def __init__(self, canvas: core.Canvas, **backend_kwargs: Any) -> None:
        from rendercanvas.auto import RenderCanvas

        self._wgpu_canvas = RenderCanvas()
        # Qt RenderCanvas calls show() in its __init__ method, so we need to hide it
        if hasattr(self._wgpu_canvas, "hide"):
            self._wgpu_canvas.hide()

        self._wgpu_canvas.set_logical_size(canvas.width, canvas.height)
        self._wgpu_canvas.set_title(canvas.title)
        self._views = canvas.views

    def _vis_get_native(self) -> RenderCanvas:
        return self._wgpu_canvas

    def _vis_set_visible(self, arg: bool) -> None:
        # show the qt canvas we patched earlier in __init__
        if hasattr(self._wgpu_canvas, "show"):
            self._wgpu_canvas.show()
        self._wgpu_canvas.request_draw(self._draw)

    def _draw(self) -> None:
        for view in self._views:
            adaptor = cast("View", view.backend_adaptor("pygfx"))
            adaptor._draw()

    def _vis_add_view(self, view: core.View) -> None:
        pass
        # adaptor = cast("View", view.backend_adaptor())
        # adaptor._pygfx_cam.set_viewport(self._viewport)
        # self._views.append(adaptor)

    def _vis_set_width(self, arg: int) -> None:
        _, height = self._wgpu_canvas.get_logical_size()
        self._wgpu_canvas.set_logical_size(arg, height)

    def _vis_set_height(self, arg: int) -> None:
        width, _ = self._wgpu_canvas.get_logical_size()
        self._wgpu_canvas.set_logical_size(width, arg)

    def _vis_set_background_color(self, arg: Color) -> None:
        # not sure if pygfx has both a canavs and view background color...
        pass

    def _vis_set_title(self, arg: str) -> None:
        self._wgpu_canvas.set_title(arg)

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
        from rendercanvas.offscreen import OffscreenRenderCanvas

        # not sure about this...
        w, h = self._wgpu_canvas.get_logical_size()
        canvas = OffscreenRenderCanvas(width=w, height=h, pixel_ratio=1)
        canvas.request_draw(self._draw)
        return cast("np.ndarray", canvas.draw())
