from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Literal, cast
from weakref import ReferenceType, WeakKeyDictionary, ref

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
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views._app import filter_mouse_events
from ndv.views.bases import ArrayCanvas, CanvasElement, ImageHandle
from ndv.views.bases._graphics._canvas_elements import RectangularROIHandle, ROIMoveMode

from ._util import rendercanvas_class

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from pygfx.materials import ImageBasicMaterial
    from pygfx.resources import Texture
    from wgpu.gui.jupyter import JupyterWgpuCanvas
    from wgpu.gui.qt import QWgpuCanvas
    from wgpu.gui.wx import WxWgpuCanvas

    WgpuCanvas: TypeAlias = QWgpuCanvas | JupyterWgpuCanvas | WxWgpuCanvas


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
        # If dimensions are unchanged, reuse the buffer
        if data.shape == self._grid.data.shape:
            self._grid.data[:] = data
            self._grid.update_range((0, 0, 0), self._grid.size)
        # Otherwise, the size (and maybe number of dimensions) changed
        # - we need a new buffer
        else:
            self._grid = pygfx.Texture(data, dim=2)
            self._image.geometry = pygfx.Geometry(grid=self._grid)
            # RGB images (i.e. 3D datasets) cannot have a colormap
            self._material.map = None if self._is_rgb() else self._cmap.to_pygfx()

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

    def clims(self) -> Any:
        return self._material.clim

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._material.clim = clims
        self._render()

    def gamma(self) -> float:
        return 1

    def set_gamma(self, gamma: float) -> None:
        if gamma != 1:
            warnings.warn("Gamma correction is not supported in pygfx", stacklevel=2)

    def colormap(self) -> _cmap.Colormap:
        return self._cmap

    def set_colormap(self, cmap: _cmap.Colormap) -> None:
        self._cmap = cmap
        # RGB (i.e. 3D) images should not have a colormap
        if not self._is_rgb():
            self._material.map = cmap.to_pygfx()
        self._render()

    def start_move(self, pos: Sequence[float]) -> None:
        pass

    def move(self, pos: Sequence[float]) -> None:
        pass

    def remove(self) -> None:
        if (par := self._image.parent) is not None:
            par.remove(self._image)

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType | None:
        return None

    def _is_rgb(self) -> bool:
        return self.data().ndim == 3 and isinstance(self._image, pygfx.Image)


class PyGFXRectangle(RectangularROIHandle):
    def __init__(
        self,
        render: Callable,
        canvas_to_world: Callable,
        parent: pygfx.WorldObject | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Positional array backing visual objects
        # NB we need five points for the outline
        # The first and last rows should be identical
        self._positions: np.ndarray = np.zeros((5, 3), dtype=np.float32)

        # Visual objects
        self._fill = self._create_fill()
        self._outline = self._create_outline()

        # Handles used for ROI manipulation
        self._handle_rad = 5  # PIXELS
        self._handles = self._create_handles()

        # containing all ROI objects makes selection easier.
        self._container = pygfx.WorldObject(*args, **kwargs)
        self._container.add(self._fill, self._outline, self._handles)
        if parent:
            parent.add(self._container)

        # Utilities for moving ROI
        self._selected = False
        self._move_mode: ROIMoveMode | None = None
        # NB _move_anchor has different meanings depending on _move_mode
        self._move_anchor: tuple[float, float] = (0, 0)
        self._render: Callable = render
        self._canvas_to_world: Callable = canvas_to_world

        # Initialize
        self.set_fill(_cmap.Color("transparent"))
        self.set_border(_cmap.Color("yellow"))
        self.set_handles(_cmap.Color("white"))
        self.set_visible(False)

    # -- BoundingBox methods -- #

    def set_bounding_box(
        self, minimum: tuple[float, float], maximum: tuple[float, float]
    ) -> None:
        # NB: Support two diagonal points, not necessarily true min/max
        x1 = float(min(minimum[0], maximum[0]))
        y1 = float(min(minimum[1], maximum[1]))
        x2 = float(max(minimum[0], maximum[0]))
        y2 = float(max(minimum[1], maximum[1]))

        # Update each handle
        self._positions[0, :2] = [x1, y1]
        self._positions[1, :2] = [x2, y1]
        self._positions[2, :2] = [x2, y2]
        self._positions[3, :2] = [x1, y2]
        self._positions[4, :2] = [x1, y1]
        self._refresh()

    def set_fill(self, color: _cmap.Color) -> None:
        if self._fill:
            self._fill.material.color = color.rgba
            self._render()

    def set_border(self, color: _cmap.Color) -> None:
        if self._outline:
            self._outline.material.color = color.rgba
            self._render()

    # TODO: Misleading name?
    def set_handles(self, color: _cmap.Color) -> None:
        if self._handles:
            self._handles.material.color = color.rgba
            self._render()

    def _create_fill(self) -> pygfx.Mesh | None:
        fill = pygfx.Mesh(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.MeshBasicMaterial(),
        )
        return fill

    def _create_outline(self) -> pygfx.Line | None:
        outline = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=self._positions,
                indices=np.array([[0, 1, 2, 3]], dtype=np.int32),
            ),
            material=pygfx.LineMaterial(thickness=1),
        )
        return outline

    def _create_handles(self) -> pygfx.Points | None:
        geometry = pygfx.Geometry(positions=self._positions[:-1])
        handles = pygfx.Points(
            geometry=geometry,
            # FIXME Size in pixels is not ideal for selection.
            # TODO investigate what size_mode = vertex does...
            material=pygfx.PointsMaterial(size=1.5 * self._handle_rad),
        )

        # NB: Default bounding box for points does not consider the radius of
        # those points. We need to HACK it for handle selection
        def get_handle_bb(old: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
            def new_get_bb() -> np.ndarray:
                bb = old().copy()
                bb[0, :2] -= self._handle_rad
                bb[1, :2] += self._handle_rad
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
        # Convert canvas -> world
        world_pos = tuple(self._canvas_to_world((event.x, event.y))[:2])
        # moving a handle
        if self._move_mode == ROIMoveMode.HANDLE:
            # The anchor is set to the opposite handle, which never moves.
            self.boundingBoxChanged.emit((world_pos, self._move_anchor))
        # translating the whole roi
        elif self._move_mode == ROIMoveMode.TRANSLATE:
            # The anchor is the mouse position reported in the previous mouse event.
            dx = world_pos[0] - self._move_anchor[0]
            dy = world_pos[1] - self._move_anchor[1]
            # If the mouse moved (dx, dy) between events, the whole ROI needs to be
            # translated that amount.
            new_min = (self._positions[0, 0] + dx, self._positions[0, 1] + dy)
            new_max = (self._positions[2, 0] + dx, self._positions[2, 1] + dy)
            self.boundingBoxChanged.emit((new_min, new_max))
            self._move_anchor = world_pos

        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        self.set_selected(True)
        # Convert canvas -> world
        world_pos = self._canvas_to_world((event.x, event.y))
        drag_idx = self._handle_under(world_pos)
        # If a marker is pressed
        if drag_idx is not None:
            opposite_idx = (drag_idx + 2) % 4
            self._move_mode = ROIMoveMode.HANDLE
            self._move_anchor = tuple(self._positions[opposite_idx, :2].copy())
        # If the rectangle is pressed
        else:
            self._move_mode = ROIMoveMode.TRANSLATE
            self._move_anchor = world_pos
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
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

    def _handle_under(self, pos: Sequence[float]) -> int | None:
        """Returns an int in [0, 3], or None.

        If an int i, means that the handle at self._positions[i] is at pos.
        If None, there is no handle at pos.
        """
        # FIXME: Ideally, Renderer.get_pick_info would do this for us. But it
        # seems broken.
        for i, p in enumerate(self._positions[:-1]):
            if (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 <= self._handle_rad**2:
                return i
        return None

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType | None:
        # Convert event pos (on canvas) to world pos
        world_pos = self._canvas_to_world((mme.x, mme.y))
        # Step 1: Handles
        # Preferred over the rectangle
        # Can only be moved if ROI is selected
        if (idx := self._handle_under(world_pos)) is not None and self.selected():
            # Idx 0 is top left, 2 is bottom right
            if idx % 2 == 0:
                return CursorType.FDIAG_ARROW
            # Idx 1 is bottom left, 3 is top right
            return CursorType.BDIAG_ARROW
        # Step 2: Entire ROI
        if self._outline:
            roi_bb = self._outline.geometry.get_bounding_box()
            if _is_inside(roi_bb, world_pos):
                return CursorType.ALL_ARROW
        return None

    def remove(self) -> None:
        if (par := self._container.parent) is not None:
            par.remove(self._container)


class GfxArrayCanvas(ArrayCanvas):
    """pygfx-based canvas wrapper."""

    def __init__(self, viewer_model: ArrayViewerModel) -> None:
        self._viewer = viewer_model

        self._current_shape: tuple[int, ...] = ()
        self._last_state: dict[Literal[2, 3], Any] = {}

        cls = rendercanvas_class()
        self._canvas = cls(size=(600, 600))

        # this filter needs to remain in scope for the lifetime of the canvas
        # or mouse events will not be intercepted
        # the returned function can be called to remove the filter, (and it also
        # closes on the event filter and keeps it in scope).
        self._disconnect_mouse_events = filter_mouse_events(self._canvas, self)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)
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
        # Maintain a weak reference to the last ROI created.
        self._last_roi_created: ReferenceType[PyGFXRectangle] | None = None

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
            with suppress(ValueError):
                # if the scene has no children yet, this will raise a ValueErrors
                # FIXME: there's a bit of order-of-call problem here:
                # this method needs to be called *after* the scene is constructed...
                # that's what controller._on_model_visible_axes_changed does, but
                # it seems fragile and should be fixed.
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
        if data is not None:
            # pygfx uses a view of the data without copy, so if we don't
            # copy it here, the original data will be modified when the
            # texture changes.
            data = data.copy()
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
        if data is not None:
            # pygfx uses a view of the data without copy, so if we don't
            # copy it here, the original data will be modified when the
            # texture changes.
            data = data.copy()
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

    def add_bounding_box(self) -> PyGFXRectangle:
        """Add a new Rectangular ROI node to the scene."""
        roi = PyGFXRectangle(
            render=self.refresh,
            canvas_to_world=self.canvas_to_world,
            parent=self._scene,
        )
        roi.set_visible(False)
        self._elements[roi._container] = roi
        self._last_roi_created = ref(roi)
        return roi

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

    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]:
        """Obtains all elements located at pos."""
        # FIXME: Ideally, Renderer.get_pick_info would do this and
        # canvas_to_world for us. But it seems broken.
        elements: list[CanvasElement] = []
        pos = self.canvas_to_world((pos_xy[0], pos_xy[1]))
        for c in self._scene.children:
            bb = c.get_bounding_box()
            if _is_inside(bb, pos):
                elements.append(self._elements[c])
        return elements

    def set_visible(self, visible: bool) -> None:
        """Set the visibility of the canvas."""
        self._canvas.visible = visible

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        if self._selection:
            self._selection.set_selected(False)
            self._selection = None
        canvas_pos = (event.x, event.y)
        world_pos = self.canvas_to_world(canvas_pos)[:2]

        # If in CREATE_ROI mode, the new ROI should "start" here.
        if self._viewer.interaction_mode == InteractionMode.CREATE_ROI:
            if self._last_roi_created is None:
                raise ValueError("No ROI to create!")
            if new_roi := self._last_roi_created():
                self._last_roi_created = None
                # HACK: Provide a non-zero starting size so that if the user clicks
                # and immediately releases, it's visible and can be selected again
                _min = world_pos
                _max = (world_pos[0] + 1, world_pos[1] + 1)
                # Put the ROI where the user clicked
                new_roi.boundingBoxChanged.emit((_min, _max))
                # Make it visible
                new_roi.set_visible(True)
                # Select it so the mouse press event below triggers ROIMoveMode.HANDLE
                # TODO: Make behavior more direct
                new_roi.set_selected(True)

            # All done - exit the mode
            self._viewer.interaction_mode = InteractionMode.PAN_ZOOM

        # Select first selectable object at clicked point
        for vis in self.elements_at(canvas_pos):
            if vis.can_select():
                self._selection = vis
                self._selection.on_mouse_press(event)
                return False

        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        if event.btn == MouseButton.LEFT:
            if self._selection and self._selection.selected():
                self._selection.on_mouse_move(event)
                # If we are moving the object, we don't want to move the camera
                return True
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        if self._selection:
            self._selection.on_mouse_release(event)
        return False

    def get_cursor(self, event: MouseMoveEvent) -> CursorType:
        if self._viewer.interaction_mode == InteractionMode.CREATE_ROI:
            return CursorType.CROSS
        for vis in self.elements_at((event.x, event.y)):
            if cursor := vis.get_cursor(event):
                return cursor
        return CursorType.DEFAULT
