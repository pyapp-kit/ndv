from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Literal, cast
from weakref import WeakKeyDictionary

import cmap as _cmap
import numpy as np
import pygfx
import pylinalg as la

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.views.bases import ArrayCanvas, CanvasElement, ImageHandle, filter_mouse_events
from ndv.views.bases.graphics._canvas_elements import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from pygfx.materials import ImageBasicMaterial
    from pygfx.resources import Texture
    from wgpu.gui.jupyter import JupyterWgpuCanvas
    from wgpu.gui.qt import QWgpuCanvas

    WgpuCanvas: TypeAlias = QWgpuCanvas | JupyterWgpuCanvas


def _is_inside(bounding_box: np.ndarray, pos: Sequence[float]) -> bool:
    return bool(
        bounding_box[0, 0] + 0.5 <= pos[0]
        and pos[0] <= bounding_box[1, 0] + 0.5
        and bounding_box[0, 1] + 0.5 <= pos[1]
        and pos[1] <= bounding_box[1, 1] + 0.5
    )


class PyGFXImageHandle(ImageHandle):
    def __init__(self, image: pygfx.Image | pygfx.Volume, render: Callable) -> None:
        self._image = image
        self._render = render
        self._grid = cast("Texture", image.geometry.grid)
        self._material = cast("ImageBasicMaterial", image.material)

    def data(self) -> np.ndarray:
        return self._grid.data  # type: ignore [no-any-return]

    def set_data(self, data: np.ndarray) -> None:
        self._grid.data[:] = data
        self._grid.update_range((0, 0, 0), self._grid.size)

    def visible(self) -> bool:
        return bool(self._image.visible)

    def set_visible(self, visible: bool) -> None:
        self._image.visible = visible
        self._render()

    def can_select(self) -> bool:
        return False

    def selected(self) -> bool:
        return False

    def set_selected(self, selected: bool) -> None:
        raise NotImplementedError("Images cannot be selected")

    def clim(self) -> Any:
        return self._material.clim

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._material.clim = clims
        self._render()

    def gamma(self) -> float:
        return 1

    def set_gamma(self, gamma: float) -> None:
        # self._material.gamma = gamma
        # self._render()
        warnings.warn("Gamma correction is not supported in pygfx", stacklevel=2)

    def cmap(self) -> _cmap.Colormap:
        return self._cmap

    def set_cmap(self, cmap: _cmap.Colormap) -> None:
        self._cmap = cmap
        self._material.map = cmap.to_pygfx()
        self._render()

    def start_move(self, pos: Sequence[float]) -> None:
        pass

    def move(self, pos: Sequence[float]) -> None:
        pass

    def remove(self) -> None:
        if (par := self._image.parent) is not None:
            par.remove(self._image)

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        return None


class PyGFXBoundingBox(BoundingBox):
    owner_of: WeakKeyDictionary[pygfx.WorldObject, PyGFXBoundingBox] = (
        WeakKeyDictionary()
    )

    def __init__(
        self,
        render: Callable,
        canvas_to_world: Callable,
        parent: pygfx.WorldObject | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._selected = False
        # self._hover_marker: scene.Markers | None = None
        # self._on_move: Callable[[tuple[float, float]], None] | None = None
        # self._move_offset: tuple[float, float] = (0, 0)

        # self._rect = scene.Rectangle(
        #     center=[0, 0], width=1, height=1, border_color="yellow", parent=parent
        # )
        self._container = pygfx.WorldObject(*args, **kwargs)
        if parent:
            parent.add(self._container)

        self._point_rad = 5  # PIXELS
        self._drag_pos: tuple[float, float] | None = None
        self._move_offset: tuple[float, float] | None = None
        self._offset = np.zeros((5, 2))
        self._on_move = None
        # NB we need 5 points, where self._positions[4] == self._positions[0]
        self._positions: np.ndarray = np.zeros((5, 3), dtype=np.float32)
        self._fill = self._create_fill()
        if self._fill:
            self._container.add(self._fill)
        self._outline = self._create_outline()
        if self._outline:
            self._container.add(self._outline)
        self._handles = self._create_handles()
        if self._handles:
            self._container.add(self._handles)
        PyGFXBoundingBox.owner_of[self._container] = self

        self._render: Callable = render
        self._canvas_to_world: Callable = canvas_to_world

        self.set_selected(False)
        self.set_visible(True)

    def _create_fill(self) -> pygfx.Mesh | None:
        fill = pygfx.Mesh(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.MeshBasicMaterial(color=(0, 0, 0, 0)),
        )
        return fill

    def _create_outline(self) -> pygfx.Line | None:
        outline = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.LineMaterial(thickness=1, color=(1, 1, 0, 0)),
        )
        return outline

    def _create_handles(self) -> pygfx.Points | None:
        geometry = pygfx.Geometry(positions=self._positions[:-1])
        handles = pygfx.Points(
            geometry=geometry,
            # FIXME Size in pixels is not ideal for selection.
            # TODO investigate what size_mode = vertex does...
            material=pygfx.PointsMaterial(color=(1, 1, 1), size=1.5 * self._point_rad),
        )

        # NB: Default bounding box for points does not consider the radius of
        # those points. We need to HACK it for handle selection
        def get_handle_bb(old: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
            def new_get_bb() -> np.ndarray:
                bb = old().copy()
                bb[0, :2] -= self._point_rad
                bb[1, :2] += self._point_rad
                return bb

            return new_get_bb

        geometry.get_bounding_box = get_handle_bb(geometry.get_bounding_box)
        return handles

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if self._handles:
            self._handles.visible = selected

    # def _set_hover(self, vis: scene.Node) -> None:
    #     self._hover_marker = vis if isinstance(vis, scene.Markers) else None

    def set_fill(self, color: Any) -> None:
        if self._fill:
            if color is None:
                color = _cmap.Color("transparent")
            if not isinstance(color, _cmap.Color):
                color = _cmap.Color(color)
            self._fill.material.color = color.rgba
            self._render()

    def set_border(self, color: Any) -> None:
        if self._outline:
            if color is None:
                color = _cmap.Color("yellow")
            if not isinstance(color, _cmap.Color):
                color = _cmap.Color(color)
            self._outline.material.color = color.rgba
            self._render()

    # TODO: Misleading name?
    def set_handles(self, color: Any) -> None:
        if self._handles:
            if color is None:
                color = _cmap.Color("white")
            if not isinstance(color, _cmap.Color):
                color = _cmap.Color(color)
            self._handles.material.color = color.rgba
            self._render()

    def set_bounding_box(
        self, mi: tuple[float, float], ma: tuple[float, float]
    ) -> None:
        # NB: Support two diagonal points, not necessarily true min/max
        x1 = float(min(mi[0], ma[0]))
        y1 = float(min(mi[1], ma[1]))
        x2 = float(max(mi[0], ma[0]))
        y2 = float(max(mi[1], ma[1]))

        # Update each handle
        self._positions[0, :2] = [x1, y1]
        self._positions[1, :2] = [x2, y1]
        self._positions[2, :2] = [x2, y2]
        self._positions[3, :2] = [x1, y2]
        self._positions[4, :2] = [x1, y1]
        self._refresh()

    def _refresh(self) -> None:
        if self._fill:
            self._fill.geometry.positions.data[:, :] = self._positions
            self._fill.geometry.positions.update_range()
        if self._outline:
            self._outline.geometry.positions.data[:, :] = self._positions
            self._outline.geometry.positions.update_range()
        if self._handles:
            self._handles.geometry.positions.data[:, :] = self._positions[:-1]
            self._handles.geometry.positions.update_range()
        self._render()

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        if self._drag_pos is not None:
            # And, on move, put the bounding box between these two points
            self.boundingBoxChanged.emit(((event.x, event.y), self._drag_pos))
        else:
            dx = event.x - self._move_offset[0]
            dy = event.y - self._move_offset[1]
            new_min = (self._positions[0, 0] + dx, self._positions[0, 1] + dy)
            new_max = (self._positions[2, 0] + dx, self._positions[2, 1] + dy)
            self.boundingBoxChanged.emit((new_min, new_max))
            self._move_offset = (event.x, event.y)

        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        self.set_selected(True)
        # Convert canvas -> world
        world_pos = self._canvas_to_world((event.x, event.y))
        drag_idx = self._handle_hover_idx(world_pos)
        # If a marker is pressed
        if drag_idx is not None:
            opposite_idx = (drag_idx + 2) % 4
            self._drag_pos = tuple(self._positions[opposite_idx, :2].copy())
        # If the rectangle is pressed
        else:
            self._drag_pos = None
            self._move_offset = world_pos
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        self._on_move = None
        return False

    def visible(self) -> bool:
        if self._outline:
            return bool(self._outline.visible)
        if self._fill:
            return bool(self._fill.visible)
        # Nothing to see
        return False

    def set_visible(self, visible: bool) -> None:
        if fill := getattr(self, "_fill", None):
            fill.visible = visible
        if outline := getattr(self, "_outline", None):
            outline.visible = visible
        if handles := getattr(self, "_handles", None):
            handles.visible = visible and self.selected()
        self._render()

    def _handle_hover_idx(self, pos: Sequence[float]) -> int | None:
        # FIXME: Ideally, Renderer.get_pick_info would do this for us. But it
        # seems broken.
        for i, p in enumerate(self._positions[:-1]):
            if (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 <= self._point_rad**2:
                return i
        return None

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        # Convert canvas -> world
        world_pos = self._canvas_to_world(pos)
        # Step 1: Check if over handle
        if self._selected:
            if (idx := self._handle_hover_idx(world_pos)) is not None:
                if np.array_equal(
                    self._positions[idx], self._positions.min(axis=0)
                ) or np.array_equal(self._positions[idx], self._positions.max(axis=0)):
                    return CursorType.FDIAG_ARROW
                return CursorType.BDIAG_ARROW

        # Step 2: Check if over ROI
        if self._outline:
            roi_bb = self._outline.geometry.get_bounding_box()
            if _is_inside(roi_bb, world_pos):
                return CursorType.ALL_ARROW
        return None


class PyGFXRoiHandle(pygfx.WorldObject):
    _render: Callable = lambda _: None

    def __init__(self, render: Callable, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, *kwargs)
        self._fill = self._create_fill()
        if self._fill:
            self.add(self._fill)
        self._outline = self._create_outline()
        if self._outline:
            self.add(self._outline)
        self._handles = self._create_handles()
        if self._handles:
            self.add(self._handles)

        self._render = render

    def _create_fill(self) -> pygfx.Mesh | None:
        # To be implemented by subclasses needing a fill
        return None

    def _create_outline(self) -> pygfx.Line | None:
        # To be implemented by subclasses needing an outline
        return None

    def _create_handles(self) -> pygfx.Points | None:
        # To be implemented by subclasses needing handles
        return None

    @property
    def vertices(self) -> Sequence[Sequence[float]]:
        # To be implemented by subclasses
        raise NotImplementedError("Must be implemented in subclasses")

    @vertices.setter
    def vertices(self, data: Sequence[Sequence[float]]) -> None:
        # To be implemented by subclasses
        raise NotImplementedError("Must be implemented in subclasses")

    def visible(self) -> bool:
        if self._outline:
            return bool(self._outline.visible)
        if self._fill:
            return bool(self._fill.visible)
        # Nothing to see
        return False

    def set_visible(self, visible: bool) -> None:
        if fill := getattr(self, "_fill", None):
            fill.visible = visible
        if outline := getattr(self, "_outline", None):
            outline.visible = visible
        if handles := getattr(self, "_handles", None):
            handles.visible = visible and self.selected()
        self._render()

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        if self._handles:
            return bool(self._handles.visible)
        # Can't be selected without handles
        return False

    def set_selected(self, selected: bool) -> None:
        if self._handles:
            self._handles.visible = selected

    def color(self) -> Any:
        if self._fill:
            return _cmap.Color(self._fill.material.color)
        return _cmap.Color("transparent")

    def set_color(self, color: _cmap.Color | None = None) -> None:
        if self._fill:
            if color is None:
                color = _cmap.Color("transparent")
            if not isinstance(color, _cmap.Color):
                color = _cmap.Color(color)
            self._fill.material.color = color.rgba
            self._render()

    def border_color(self) -> Any:
        if self._outline:
            return _cmap.Color(self._outline.material.color)
        return _cmap.Color("transparent")

    def set_border_color(self, color: _cmap.Color | None = None) -> None:
        if self._outline:
            if color is None:
                color = _cmap.Color("yellow")
            if not isinstance(color, _cmap.Color):
                color = _cmap.Color(color)
            self._outline.material.color = color.rgba
            self._render()

    def start_move(self, pos: Sequence[float]) -> None:
        # To be implemented by subclasses
        raise NotImplementedError("Must be implemented in subclasses")

    def move(self, pos: Sequence[float]) -> None:
        # To be implemented by subclasses
        raise NotImplementedError("Must be implemented in subclasses")

    def remove(self) -> None:
        if (par := self.parent) is not None:
            par.remove(self)

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        # To be implemented by subclasses
        raise NotImplementedError("Must be implemented in subclasses")


class RectangularROIHandle(PyGFXRoiHandle):
    def __init__(
        self, render: Callable, canvas_to_world: Callable, *args: Any, **kwargs: Any
    ) -> None:
        self._point_rad = 5  # PIXELS
        self._positions: np.ndarray = np.zeros((5, 3), dtype=np.float32)

        super().__init__(render, *args, *kwargs)
        self._canvas_to_world = canvas_to_world

        # drag_reference defines the offset between where the user clicks and the center
        # of the rectangle
        self._drag_idx: int | None = None
        self._offset = np.zeros((5, 2))
        self._on_drag = [
            self._move_handle_0,
            self._move_handle_1,
            self._move_handle_2,
            self._move_handle_3,
        ]

    @property
    def vertices(self) -> Sequence[Sequence[float]]:
        # Buffer object
        return [p[:2] for p in self._positions]

    @vertices.setter
    def vertices(self, vertices: Sequence[Sequence[float]]) -> None:
        if len(vertices) != 4 or any(len(v) != 2 for v in vertices):
            raise Exception("Only 2D rectangles are currently supported")
        is_aligned = (
            vertices[0][1] == vertices[1][1]
            and vertices[1][0] == vertices[2][0]
            and vertices[2][1] == vertices[3][1]
            and vertices[3][0] == vertices[0][0]
        )
        if not is_aligned:
            raise Exception(
                "Only rectangles aligned with the axes are currently supported"
            )

        # Update each handle
        self._positions[:-1, :2] = vertices
        self._positions[-1, :2] = vertices[0]
        self._refresh()

    def start_move(self, pos: Sequence[float]) -> None:
        self._drag_idx = self._handle_hover_idx(pos)

        if self._drag_idx is None:
            self._offset[:, :] = self._positions[:, :2] - pos[:2]

    def move(self, pos: Sequence[float]) -> None:
        if self._drag_idx is not None:
            self._on_drag[self._drag_idx](pos)
        else:
            # TODO: We could potentially do this smarter via transforms
            self._positions[:, :2] = self._offset[:, :2] + pos[:2]
        self._refresh()

    def _move_handle_0(self, pos: Sequence[float]) -> None:
        # NB pygfx requires (idx 0) = (idx 4)
        self._positions[0, :2] = pos[:2]
        self._positions[4, :2] = pos[:2]

        self._positions[3, 0] = pos[0]
        self._positions[1, 1] = pos[1]

    def _move_handle_1(self, pos: Sequence[float]) -> None:
        self._positions[1, :2] = pos[:2]

        self._positions[2, 0] = pos[0]
        # NB pygfx requires (idx 0) = (idx 4)
        self._positions[0, 1] = pos[1]
        self._positions[4, 1] = pos[1]

    def _move_handle_2(self, pos: Sequence[float]) -> None:
        self._positions[2, :2] = pos[:2]

        self._positions[1, 0] = pos[0]
        self._positions[3, 1] = pos[1]

    def _move_handle_3(self, pos: Sequence[float]) -> None:
        self._positions[3, :2] = pos[:2]

        # NB pygfx requires (idx 0) = (idx 4)
        self._positions[0, 0] = pos[0]
        self._positions[4, 0] = pos[0]
        self._positions[2, 1] = pos[1]

    def _create_fill(self) -> pygfx.Mesh | None:
        fill = pygfx.Mesh(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.MeshBasicMaterial(color=(0, 0, 0, 0)),
        )
        return fill

    def _create_outline(self) -> pygfx.Line | None:
        outline = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.LineMaterial(thickness=1, color=(0, 0, 0, 0)),
        )
        return outline

    def _create_handles(self) -> pygfx.Points | None:
        geometry = pygfx.Geometry(positions=self._positions[:-1])
        handles = pygfx.Points(
            geometry=geometry,
            # FIXME Size in pixels is not ideal for selection.
            # TODO investigate what size_mode = vertex does...
            material=pygfx.PointsMaterial(color=(1, 1, 1), size=1.5 * self._point_rad),
        )

        # NB: Default bounding box for points does not consider the radius of
        # those points. We need to HACK it for handle selection
        def get_handle_bb(old: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
            def new_get_bb() -> np.ndarray:
                bb = old().copy()
                bb[0, :2] -= self._point_rad
                bb[1, :2] += self._point_rad
                return bb

            return new_get_bb

        geometry.get_bounding_box = get_handle_bb(geometry.get_bounding_box)
        return handles

    def _refresh(self) -> None:
        if self._fill:
            self._fill.geometry.positions.data[:, :] = self._positions
            self._fill.geometry.positions.update_range()
        if self._outline:
            self._outline.geometry.positions.data[:, :] = self._positions
            self._outline.geometry.positions.update_range()
        if self._handles:
            self._handles.geometry.positions.data[:, :] = self._positions[:-1]
            self._handles.geometry.positions.update_range()
        self._render()

    def _handle_hover_idx(self, pos: Sequence[float]) -> int | None:
        # FIXME: Ideally, Renderer.get_pick_info would do this for us. But it
        # seems broken.
        for i, p in enumerate(self._positions[:-1]):
            if (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 <= self._point_rad**2:
                return i
        return None

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        # Convert canvas -> world
        world_pos = self._canvas_to_world(pos)
        # Step 1: Check if over handle
        if (idx := self._handle_hover_idx(world_pos)) is not None:
            if np.array_equal(
                self._positions[idx], self._positions.min(axis=0)
            ) or np.array_equal(self._positions[idx], self._positions.max(axis=0)):
                return CursorType.FDIAG_ARROW
            return CursorType.BDIAG_ARROW

        # Step 2: Check if over ROI
        if self._outline:
            roi_bb = self._outline.geometry.get_bounding_box()
            if _is_inside(roi_bb, world_pos):
                return CursorType.ALL_ARROW
        return None


def get_canvas_class() -> WgpuCanvas:
    from ndv.views._app import GuiFrontend, gui_frontend

    frontend = gui_frontend()
    if frontend == GuiFrontend.QT:
        from qtpy.QtCore import QSize
        from wgpu.gui import qt

        class QWgpuCanvas(qt.QWgpuCanvas):
            def installEventFilter(self, filter: Any) -> None:
                self._subwidget.installEventFilter(filter)

            def sizeHint(self) -> QSize:
                return QSize(self.width(), self.height())

        return QWgpuCanvas
    if frontend == GuiFrontend.JUPYTER:
        from wgpu.gui.jupyter import JupyterWgpuCanvas

        return JupyterWgpuCanvas


class GfxArrayCanvas(ArrayCanvas):
    """pygfx-based canvas wrapper."""

    def __init__(self) -> None:
        self._current_shape: tuple[int, ...] = ()
        self._last_state: dict[Literal[2, 3], Any] = {}

        cls = get_canvas_class()
        self._canvas = cls(size=(600, 600))
        # this filter needs to remain in scope for the lifetime of the canvas
        # or mouse events will not be intercepted
        # the returned function can be called to remove the filter, (and it also
        # closes on the event filter and keeps it in scope).
        self._disconnect_mouse_events = filter_mouse_events(self._canvas, self)

        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas, show_fps=True)
        try:
            # requires https://github.com/pygfx/pygfx/pull/752
            self._renderer.blend_mode = "additive"
        except ValueError:
            warnings.warn(
                "This version of pygfx does not yet support additive blending.",
                stacklevel=3,
            )
            self._renderer.blend_mode = "weighted_depth"

        self._scene = pygfx.Scene()
        self._camera: pygfx.Camera | None = None
        self._ndim: Literal[2, 3] | None = None

        self._elements = WeakKeyDictionary[pygfx.WorldObject, CanvasElement]()
        self._selection: CanvasElement | None = None
        # FIXME: Remove
        self._initializing_roi: PyGFXRoiHandle | None = None

    def frontend_widget(self) -> Any:
        return self._canvas

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        """Set the number of dimensions of the displayed data."""
        if ndim == self._ndim:
            return
        elif self._ndim is not None and self._camera is not None:
            # remember the current state before switching to the new camera
            self._last_state[self._ndim] = self._camera.get_state()

        self._ndim = ndim
        if ndim == 3:
            self._camera = cam = pygfx.PerspectiveCamera(0, 1)
            cam.show_object(self._scene, up=(0, -1, 0), view_dir=(0, 0, 1))
            controller = pygfx.OrbitController(cam, register_events=self._renderer)
            zoom = "zoom"
            # FIXME: there is still an issue with rotational centration.
            # the controller is not rotating around the middle of the volume...
            # but I think it might actually be a pygfx issue... the critical state
            # seems to be somewhere outside of the camera's get_state dict.
        else:
            self._camera = cam = pygfx.OrthographicCamera(512, 512)
            cam.local.scale_y = -1
            cam.local.position = (256, 256, 0)
            controller = pygfx.PanZoomController(cam, register_events=self._renderer)
            zoom = "zoom_to_point"

        self._controller = controller
        # increase zoom wheel gain
        self._controller.controls.update({"wheel": (zoom, "push", -0.005)})

        # restore the previous state if it exists
        if state := self._last_state.get(ndim):
            cam.set_state(state)

    def add_image(self, data: np.ndarray | None = None) -> PyGFXImageHandle:
        """Add a new Image node to the scene."""
        tex = pygfx.Texture(data, dim=2)
        image = pygfx.Image(
            pygfx.Geometry(grid=tex),
            # depth_test=False for additive-like blending
            pygfx.ImageBasicMaterial(depth_test=False),
        )
        self._scene.add(image)

        if data is not None:
            self._current_shape, prev_shape = data.shape, self._current_shape
            if not prev_shape:
                self.set_range()

        # FIXME: I suspect there are more performant ways to refresh the canvas
        # look into it.
        handle = PyGFXImageHandle(image, self.refresh)
        self._elements[image] = handle
        return handle

    def add_volume(self, data: np.ndarray | None = None) -> PyGFXImageHandle:
        tex = pygfx.Texture(data, dim=3)
        vol = pygfx.Volume(
            pygfx.Geometry(grid=tex),
            # depth_test=False for additive-like blending
            pygfx.VolumeRayMaterial(interpolation="nearest", depth_test=False),
        )
        self._scene.add(vol)

        if data is not None:
            vol.local_position = [-0.5 * i for i in data.shape[::-1]]
            self._current_shape, prev_shape = data.shape, self._current_shape
            if len(prev_shape) != 3:
                self.set_range()

        # FIXME: I suspect there are more performant ways to refresh the canvas
        # look into it.
        handle = PyGFXImageHandle(vol, self.refresh)
        self._elements[vol] = handle
        return handle

    def add_bounding_box(self) -> PyGFXBoundingBox:
        """Add a new Rectangular ROI node to the scene."""
        roi = PyGFXBoundingBox(
            render=self.refresh, canvas_to_world=self.canvas_to_world
        )
        # FIXME: Parameter to roi
        self._scene.add(roi._container)
        self._selection = roi
        return roi

    # def add_roi(
    #     self,
    #     vertices: Sequence[tuple[float, float]] | None = None,
    #     color: _cmap.Color | None = None,
    #     border_color: _cmap.Color | None = None,
    #     visible: bool = False,
    # ) -> PyGFXRoiHandle:
    #     """Add a new Rectangular ROI node to the scene."""
    #     roi = PyGFXBoundingBox(self.refresh, self.canvas_to_world)
    #     self._scene.add(roi._container)
    #     handle = RectangularROIHandle(self.refresh, self.canvas_to_world)
    #     self._scene.add(handle)
    #     if vertices:
    #         handle.vertices = vertices
    #     else:
    #         # FIXME: Ugly
    #         self._initializing_roi = handle
    #     handle.set_color(color)
    #     handle.set_border_color(border_color)
    #     handle.set_visible(visible)

    #     self._elements[handle] = handle
    #     return handle

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
        if not self._scene.children or self._camera is None:
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

    def refresh(self) -> None:
        with suppress(AttributeError):
            self._canvas.update()
        self._canvas.request_draw(self._animate)

    def _animate(self) -> None:
        self._renderer.render(self._scene, self._camera)

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        viewport = pygfx.Viewport.from_viewport_or_renderer(self._renderer)
        if not viewport.is_inside(*pos_xy):
            return (-1, -1, -1)

        # Get position relative to viewport
        pos_rel = (
            pos_xy[0] - viewport.rect[0],
            pos_xy[1] - viewport.rect[1],
        )

        vs = viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        if self._camera:
            pos_ndc += la.vec_transform(
                self._camera.world.position, self._camera.camera_matrix
            )
            pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

            # NB In vispy, (0.5,0.5) is a center of an image pixel, while in pygfx
            # (0,0) is the center. We conform to vispy's standard.
            return (pos_world[0] + 0.5, pos_world[1] + 0.5, pos_world[2] + 0.5)
        else:
            return (-1, -1, -1)

    def elements_at(self, pos_xy: tuple[float, float]) -> list[pygfx.WorldObject]:
        """Obtains all elements located at pos."""
        # FIXME: Ideally, Renderer.get_pick_info would do this and
        # canvas_to_world for us. But it seems broken.
        elements: list[pygfx.WorldObject] = []
        pos = self.canvas_to_world((pos_xy[0], pos_xy[1]))
        for c in self._scene.children:
            bb = c.get_bounding_box()
            if _is_inside(bb, pos):
                elements.append(c)
        return elements

    def set_visible(self, visible: bool) -> None:
        """Set the visibility of the canvas."""
        self._canvas.visible = visible

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        # if roi := self._initializing_roi:
        #     self._initializing_roi = None
        #     pos = self.canvas_to_world((event.x, event.y))
        #     roi.move(pos)
        #     roi.set_visible(True)

        # ev_pos = (event.x, event.y)
        # pos = self.canvas_to_world(ev_pos)
        # # TODO why does the canvas need this point untransformed??
        # elements = self.elements_at(ev_pos)
        # # Deselect prior selection before editing new selection
        # if self._selection:
        #     self._selection.set_selected(False)
        # for e in elements:
        #     if e.can_select():
        #         e.start_move(pos)
        #         # Select new selection
        #         self._selection = e
        #         self._selection.set_selected(True)
        #         return False
        # return False

        # TODO: Make work
        # if roi := self._initializing_roi:
        #     self._initializing_roi = None
        #     pos = self.canvas_to_world((event.x, event.y))
        #     roi.move(pos)
        #     roi.set_visible(True)
        if self._selection:
            self._selection.set_selected(False)
            self._selection = None

        # Find all visuals at the point
        ev_pos = (event.x, event.y)
        for vis in self.elements_at(ev_pos):
            # If any belong to a bounding box, direct output there
            if bbox := PyGFXBoundingBox.owner_of.get(vis, None):
                self._selection = bbox
                self.canvas_to_world(ev_pos)
                # FIXME: Use the same event?
                self._selection.set_selected(True)
                self._selection.on_mouse_press(
                    MousePressEvent(ev_pos[0], ev_pos[1], event.btn)
                )
                # self._camera.interactive = False
                return False

        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        # ev_pos = (event.x, event.y)
        # if event.btn == MouseButton.LEFT:
        #     if self._selection and self._selection.selected():
        #         ev_pos = (event.x, event.y)
        #         pos = self.canvas_to_world(ev_pos)
        #         self._selection.move(pos)
        #         # If we are moving the object, we don't want to move the camera
        #         return True
        # return False
        ev_pos = (event.x, event.y)
        # for vis in self._canvas.visuals_at(ev_pos):
        #     if bbox := VispyBoundingBox.owner_of.get(vis, None):
        #         bbox._set_hover(vis)
        #         break
        if event.btn == MouseButton.LEFT:
            if self._selection and self._selection.selected():
                ev_pos = (event.x, event.y)
                pos = self.canvas_to_world(ev_pos)
                # FIXME: Use the same event?
                self._selection.on_mouse_move(MouseMoveEvent(pos[0], pos[1], event.btn))
                # If we are moving the object, we don't want to move the camera
                return True
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False

    def get_cursor(self, pos: tuple[float, float]) -> CursorType:
        if self._initializing_roi:
            return CursorType.CROSS
        for vis in self.elements_at(pos):
            if bbox := PyGFXBoundingBox.owner_of.get(vis, None):
                self.canvas_to_world(pos)[:2]
                if cursor := bbox.get_cursor(pos):
                    return cursor
        return CursorType.DEFAULT
