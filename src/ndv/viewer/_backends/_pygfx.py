from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pygfx
from qtpy.QtCore import QSize
from wgpu.gui.qt import QWgpuCanvas

if TYPE_CHECKING:
    import cmap
    from pygfx.materials import ImageBasicMaterial
    from pygfx.resources import Texture
    from qtpy.QtWidgets import QWidget


class PyGFXImageHandle:
    def __init__(self, image: pygfx.Image, render: Callable) -> None:
        self._image = image
        self._render = render
        self._grid = cast("Texture", image.geometry.grid)
        self._material = cast("ImageBasicMaterial", image.material)

    @property
    def data(self) -> np.ndarray:
        return self._grid.data  # type: ignore [no-any-return]

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._grid.data[:] = data
        self._grid.update_range((0, 0, 0), self._grid.size)

    @property
    def visible(self) -> bool:
        return bool(self._image.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._image.visible = visible
        self._render()

    @property
    def clim(self) -> Any:
        return self._material.clim

    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None:
        self._material.clim = clims
        self._render()

    @property
    def cmap(self) -> cmap.Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None:
        self._cmap = cmap
        self._material.map = cmap.to_pygfx()
        self._render()

    def remove(self) -> None:
        if (par := self._image.parent) is not None:
            par.remove(self._image)


class _QWgpuCanvas(QWgpuCanvas):
    def sizeHint(self) -> QSize:
        return QSize(512, 512)


class PyGFXViewerCanvas:
    """pygfx-based canvas wrapper."""

    def __init__(self, set_info: Callable[[str], None]) -> None:
        self._set_info = set_info

        self._canvas = _QWgpuCanvas(size=(512, 512))
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
        # requires https://github.com/pygfx/pygfx/pull/752
        self._renderer.blend_mode = "additive"
        self._scene = pygfx.Scene()
        self._camera = cam = pygfx.OrthographicCamera(512, 512)
        cam.local.scale_y = -1

        cam.local.position = (256, 256, 0)
        self._controller = pygfx.PanZoomController(cam, register_events=self._renderer)
        # increase zoom wheel gain
        self._controller.controls.update({"wheel": ("zoom_to_point", "push", -0.005)})

    def qwidget(self) -> QWidget:
        return cast("QWidget", self._canvas)

    def refresh(self) -> None:
        self._canvas.update()
        self._canvas.request_draw(self._animate)

    def _animate(self) -> None:
        self._renderer.render(self._scene, self._camera)

    def set_ndim(self, ndim: int) -> None:
        """Set the number of dimensions of the displayed data."""
        if ndim != 2:
            raise NotImplementedError(
                "Volume rendering with pygfx is not yet supported."
            )

    def add_volume(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> PyGFXImageHandle:
        raise NotImplementedError("Volume rendering with pygfx is not yet supported.")

    def add_image(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> PyGFXImageHandle:
        """Add a new Image node to the scene."""
        image = pygfx.Image(
            pygfx.Geometry(grid=pygfx.Texture(data, dim=2)),
            # depth_test=False for additive-like blending
            pygfx.ImageBasicMaterial(depth_test=False),
        )
        self._scene.add(image)
        # FIXME: I suspect there are more performant ways to refresh the canvas
        # look into it.
        handle = PyGFXImageHandle(image, self.refresh)
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0.05,
    ) -> None:
        """Update the range of the PanZoomCamera.

        When called with no arguments, the range is set to the full extent of the data.
        """
        if not self._scene.children:
            return

        cam = self._camera
        cam.show_object(self._scene)

        width, height, depth = np.ptp(self._scene.get_world_bounding_box(), axis=0)
        if width < 0.01:
            width = 1
        if height < 0.01:
            height = 1
        cam.width = width
        cam.height = height
        cam.zoom = 1 - margin
        self.refresh()

    # def _on_mouse_move(self, event: SceneMouseEvent) -> None:
    #     """Mouse moved on the canvas, display the pixel value and position."""
    #     images = []
    #     # Get the images the mouse is over
    #     seen = set()
    #     while visual := self._canvas.visual_at(event.pos):
    #         if isinstance(visual, scene.visuals.Image):
    #             images.append(visual)
    #         visual.interactive = False
    #         seen.add(visual)
    #     for visual in seen:
    #         visual.interactive = True
    #     if not images:
    #         return

    #     tform = images[0].get_transform("canvas", "visual")
    #     px, py, *_ = (int(x) for x in tform.map(event.pos))
    #     text = f"[{py}, {px}]"
    #     for c, img in enumerate(images):
    #         with suppress(IndexError):
    #             text += f" c{c}: {img._data[py, px]}"
    #     self._set_info(text)
