from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vispy import scene

from ndv.models import _scene as core

from ._util import pyd_color_to_vispy

if TYPE_CHECKING:
    import numpy as np
    from cmap import Color


class Canvas(core.canvas.CanvasAdaptorProtocol):
    """Canvas interface for Vispy Backend."""

    def __init__(self, canvas: core.Canvas, **backend_kwargs: Any) -> None:
        backend_kwargs.setdefault("keys", "interactive")
        self._vispy_canvas = scene.SceneCanvas(
            size=(canvas.width, canvas.height),
            title=canvas.title,
            show=canvas.visible,
            bgcolor=pyd_color_to_vispy(canvas.background_color),
            **backend_kwargs,
        )

    def _vis_get_native(self) -> scene.SceneCanvas:
        return self._vispy_canvas

    def _vis_set_visible(self, arg: bool) -> None:
        self._vispy_canvas.show(visible=arg)

    def _vis_add_view(self, view: core.View) -> None:
        vispy_view = view.backend_adaptor("vispy")._vis_get_native()
        if not isinstance(vispy_view, scene.ViewBox):
            raise TypeError("View must be a Vispy ViewBox")
        self._vispy_canvas.central_widget.add_widget(vispy_view)

    def _vis_set_width(self, arg: int) -> None:
        _height = self._vispy_canvas.size[1]
        self._vispy_canvas.size = (int(arg), int(_height))

    def _vis_set_height(self, arg: int) -> None:
        _width = self._vispy_canvas.size[0]
        self._vispy_canvas.size = (int(_width), int(arg))

    def _vis_set_background_color(self, arg: Color | None) -> None:
        self._vispy_canvas.bgcolor = pyd_color_to_vispy(arg)

    def _vis_set_title(self, arg: str) -> None:
        self._vispy_canvas.title = arg

    def _vis_close(self) -> None:
        """Close canvas."""
        self._vispy_canvas.close()

    def _vis_render(
        self,
        region: tuple[int, int, int, int] | None = None,
        size: tuple[int, int] | None = None,
        bgcolor: Color | None = None,
        crop: np.ndarray | tuple[int, int, int, int] | None = None,
        alpha: bool = True,
    ) -> np.ndarray:
        """Render to screenshot."""
        data = self._vispy_canvas.render(
            region=region, size=size, bgcolor=bgcolor, crop=crop, alpha=alpha
        )
        return cast("np.ndarray", data)

    def _vis_get_ipython_mimebundle(
        self, *args: Any, **kwargs: Any
    ) -> dict | tuple[dict, dict]:
        return self._vis_get_native()._repr_mimebundle_(*args, **kwargs)  # type: ignore
