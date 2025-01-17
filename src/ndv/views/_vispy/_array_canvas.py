from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import WeakKeyDictionary

import cmap as _cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from vispy import scene
from vispy.color import Color
from vispy.util.quaternion import Quaternion

from ndv._types import CursorType
from ndv.views._app import filter_mouse_events
from ndv.views.bases import ArrayCanvas
from ndv.views.bases._graphics._canvas_elements import (
    CanvasElement,
    ImageHandle,
    RoiHandle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable

    import vispy.app


turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


class Handle(scene.visuals.Markers):
    """A Marker that allows specific ROI alterations."""

    def __init__(
        self,
        parent: RectangularROI,
        on_move: Callable[[Sequence[float]], None] | None = None,
        cursor: CursorType
        | Callable[[Sequence[float]], CursorType] = CursorType.ALL_ARROW,
    ) -> None:
        super().__init__(parent=parent)
        self.unfreeze()
        self.parent = parent
        # on_move function(s)
        self.on_move: list[Callable[[Sequence[float]], None]] = []
        if on_move:
            self.on_move.append(on_move)
        # cusror preference function
        if not callable(cursor):

            def cursor(_: Any) -> CursorType:
                return cursor

        self._cursor_at = cursor
        self._selected = False
        # NB VisPy asks that the data is a 2D array
        self._pos = np.array([[0, 0]], dtype=np.float32)
        self.interactive = True
        self.freeze()

    def start_move(self, pos: Sequence[float]) -> None:
        pass

    def move(self, pos: Sequence[float]) -> None:
        for func in self.on_move:
            func(pos)

    @property
    def pos(self) -> Sequence[float]:
        return cast("Sequence[float]", self._pos[0, :])

    @pos.setter
    def pos(self, pos: Sequence[float]) -> None:
        self._pos[:] = pos[:2]
        self.set_data(self._pos)

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, selected: bool) -> None:
        self._selected = selected
        self.parent.selected = selected

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
        return self._cursor_at(self.pos)


class RectangularROI(scene.visuals.Rectangle):
    """A VisPy Rectangle visual whose attributes can be edited."""

    def __init__(
        self,
        parent: scene.visuals.Visual,
        center: list[float] | None = None,
        width: float = 1e-6,
        height: float = 1e-6,
    ) -> None:
        if center is None:
            center = [0, 0]
        scene.visuals.Rectangle.__init__(
            self, center=center, width=width, height=height, radius=0, parent=parent
        )
        self.unfreeze()
        self.parent = parent
        self.interactive = True

        self._handles = [
            Handle(
                self,
                on_move=self.move_top_left,
                cursor=self._handle_cursor_pref,
            ),
            Handle(
                self,
                on_move=self.move_top_right,
                cursor=self._handle_cursor_pref,
            ),
            Handle(
                self,
                on_move=self.move_bottom_right,
                cursor=self._handle_cursor_pref,
            ),
            Handle(
                self,
                on_move=self.move_bottom_left,
                cursor=self._handle_cursor_pref,
            ),
        ]

        # drag_reference defines the offset between where the user clicks and the center
        # of the rectangle
        self.drag_reference = [0.0, 0.0]
        self.interactive = True
        self._selected = False
        self.freeze()

    def _handle_cursor_pref(self, handle_pos: Sequence[float]) -> CursorType:
        # Bottom left handle
        if handle_pos[0] < self.center[0] and handle_pos[1] < self.center[1]:
            return CursorType.FDIAG_ARROW
        # Top right handle
        if handle_pos[0] > self.center[0] and handle_pos[1] > self.center[1]:
            return CursorType.FDIAG_ARROW
        # Top left, bottom right
        return CursorType.BDIAG_ARROW

    def move_top_left(self, pos: Sequence[float]) -> None:
        self._handles[3].pos = [pos[0], self._handles[3].pos[1]]
        self._handles[0].pos = pos
        self._handles[1].pos = [self._handles[1].pos[0], pos[1]]
        self.redraw()

    def move_top_right(self, pos: Sequence[float]) -> None:
        self._handles[0].pos = [self._handles[0].pos[0], pos[1]]
        self._handles[1].pos = pos
        self._handles[2].pos = [pos[0], self._handles[2].pos[1]]
        self.redraw()

    def move_bottom_right(self, pos: Sequence[float]) -> None:
        self._handles[1].pos = [pos[0], self._handles[1].pos[1]]
        self._handles[2].pos = pos
        self._handles[3].pos = [self._handles[3].pos[0], pos[1]]
        self.redraw()

    def move_bottom_left(self, pos: Sequence[float]) -> None:
        self._handles[2].pos = [self._handles[2].pos[0], pos[1]]
        self._handles[3].pos = pos
        self._handles[0].pos = [pos[0], self._handles[0].pos[1]]
        self.redraw()

    def redraw(self) -> None:
        left, top, *_ = self._handles[0].pos
        right, bottom, *_ = self._handles[2].pos

        self.center = [(left + right) / 2, (top + bottom) / 2]
        self.width = max(abs(left - right), 1e-6)
        self.height = max(abs(top - bottom), 1e-6)

    # --------------------- EditableROI interface --------------------------
    # In the future, if any other objects implement these same methods, this
    # could be extracted into an ABC.

    @property
    def vertices(self) -> Sequence[Sequence[float]]:
        return [h.pos for h in self._handles]

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
        for i, handle in enumerate(self._handles):
            handle.pos = vertices[i]
        # Redraw
        self.redraw()

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, selected: bool) -> None:
        self._selected = selected
        for h in self._handles:
            h.visible = selected

    def start_move(self, pos: Sequence[float]) -> None:
        self.drag_reference = [
            pos[0] - self.center[0],
            pos[1] - self.center[1],
        ]

    def move(self, pos: Sequence[float]) -> None:
        new_center = [
            pos[0] - self.drag_reference[0],
            pos[1] - self.drag_reference[1],
        ]
        old_center = self.center
        # TODO: Simplify
        for h in self._handles:
            existing_pos = h.pos
            h.pos = [
                existing_pos[0] + new_center[0] - old_center[0],
                existing_pos[1] + new_center[1] - old_center[1],
            ]
        self.center = new_center

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
        return CursorType.ALL_ARROW

    # ------------------- End EditableROI interface -------------------------


class VispyImageHandle(ImageHandle):
    def __init__(self, visual: scene.Image | scene.Volume) -> None:
        self._visual = visual
        self._ndim = 2 if isinstance(visual, scene.visuals.Image) else 3

    def data(self) -> np.ndarray:
        try:
            return self._visual._data  # type: ignore [no-any-return]
        except AttributeError:
            return self._visual._last_data  # type: ignore [no-any-return]

    def set_data(self, data: np.ndarray) -> None:
        if not data.ndim == self._ndim:
            warnings.warn(
                f"Got wrong number of dimensions ({data.ndim}) for vispy "
                f"visual of type {type(self._visual)}.",
                stacklevel=2,
            )
            return
        self._visual.set_data(data)

    def visible(self) -> bool:
        return bool(self._visual.visible)

    def set_visible(self, visible: bool) -> None:
        self._visual.visible = visible

    # TODO: shouldn't be needed
    def can_select(self) -> bool:
        return False

    def selected(self) -> bool:
        return False

    def set_selected(self, selected: bool) -> None:
        raise NotImplementedError("Images cannot be selected")

    def clims(self) -> Any:
        return self._visual.clim

    def set_clims(self, clims: tuple[float, float]) -> None:
        with suppress(ZeroDivisionError):
            self._visual.clim = clims

    def gamma(self) -> float:
        return self._visual.gamma  # type: ignore [no-any-return]

    def set_gamma(self, gamma: float) -> None:
        self._visual.gamma = gamma

    def cmap(self) -> _cmap.Colormap:
        return self._cmap  # FIXME

    def set_cmap(self, cmap: _cmap.Colormap) -> None:
        self._cmap = cmap
        self._visual.cmap = cmap.to_vispy()

    def transform(self) -> np.ndarray:
        raise NotImplementedError

    def set_transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def start_move(self, pos: Sequence[float]) -> None:
        pass

    def move(self, pos: Sequence[float]) -> None:
        pass

    def remove(self) -> None:
        self._visual.parent = None

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
        return None


# FIXME: Unfortunate naming :)
class VispyHandleHandle(CanvasElement):
    def __init__(self, handle: Handle, parent: CanvasElement) -> None:
        self._handle = handle
        self._parent = parent

    def visible(self) -> bool:
        return cast("bool", self._handle.visible)

    def set_visible(self, visible: bool) -> None:
        self._handle.visible = visible

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._handle.selected

    def set_selected(self, selected: bool) -> None:
        self._handle.selected = selected

    def start_move(self, pos: Sequence[float]) -> None:
        self._handle.start_move(pos)

    def move(self, pos: Sequence[float]) -> None:
        self._handle.move(pos)

    def remove(self) -> None:
        self._parent.remove()

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
        return self._handle.cursor_at(pos)


class VispyRoiHandle(RoiHandle):
    def __init__(self, roi: RectangularROI) -> None:
        self._roi = roi

    def vertices(self) -> Sequence[Sequence[float]]:
        return self._roi.vertices

    def set_vertices(self, vertices: Sequence[Sequence[float]]) -> None:
        self._roi.vertices = vertices

    def visible(self) -> bool:
        return bool(self._roi.visible)

    def set_visible(self, visible: bool) -> None:
        self._roi.visible = visible

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._roi.selected

    def set_selected(self, selected: bool) -> None:
        self._roi.selected = selected

    def start_move(self, pos: Sequence[float]) -> None:
        self._roi.start_move(pos)

    def move(self, pos: Sequence[float]) -> None:
        self._roi.move(pos)

    def color(self) -> Any:
        return self._roi.color

    def set_color(self, color: _cmap.Color | None) -> None:
        if color is None:
            color = _cmap.Color("transparent")
        # NB: To enable dragging the shape within the border,
        # we require a positive alpha.
        alpha = max(color.alpha, 1e-6)
        self._roi.color = Color(color.hex, alpha=alpha)

    def border_color(self) -> _cmap.Color:
        return _cmap.Color(self._roi.border_color.rgba)

    def set_border_color(self, color: _cmap.Color | None) -> None:
        if color is None:
            color = _cmap.Color("yellow")
        self._roi.border_color = Color(color.hex, alpha=color.alpha)

    def remove(self) -> None:
        self._roi.parent = None

    def cursor_at(self, pos: Sequence[float]) -> CursorType | None:
        return self._roi.cursor_at(pos)


class VispyArrayCanvas(ArrayCanvas):
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self) -> None:
        self._canvas = scene.SceneCanvas(size=(600, 600))

        # this filter needs to remain in scope for the lifetime of the canvas
        # or mouse events will not be intercepted
        # the returned function can be called to remove the filter, (and it also
        # closes on the event filter and keeps it in scope).
        self._disconnect_mouse_events = filter_mouse_events(self._canvas.native, self)

        self._last_state: dict[Literal[2, 3], Any] = {}

        central_wdg: scene.Widget = self._canvas.central_widget
        self._view: scene.ViewBox = central_wdg.add_view()
        self._ndim: Literal[2, 3] | None = None

        self._elements: WeakKeyDictionary = WeakKeyDictionary()

    @property
    def _camera(self) -> vispy.scene.cameras.BaseCamera:
        return self._view.camera

    def set_ndim(self, ndim: Literal[2, 3]) -> None:
        """Set the number of dimensions of the displayed data."""
        if ndim == self._ndim:
            return
        elif self._ndim is not None:
            # remember the current state before switching to the new camera
            self._last_state[self._ndim] = self._camera.get_state()

        self._ndim = ndim
        if ndim == 3:
            cam = scene.ArcballCamera(fov=0)
            # this sets the initial view similar to what the panzoom view would have.
            cam._quaternion = DEFAULT_QUATERNION
        else:
            cam = scene.PanZoomCamera(aspect=1, flip=(0, 1))

        # restore the previous state if it exists
        if state := self._last_state.get(ndim):
            cam.set_state(state)
        self._view.camera = cam

    def frontend_widget(self) -> Any:
        return self._canvas.native

    def set_visible(self, visible: bool) -> None: ...

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    def refresh(self) -> None:
        self._canvas.update()

    def add_image(self, data: np.ndarray | None = None) -> VispyImageHandle:
        """Add a new Image node to the scene."""
        data = _downcast(data)
        try:
            img = scene.visuals.Image(
                data, parent=self._view.scene, texture_format="auto"
            )
        except ValueError as e:
            warnings.warn(f"{e}. Falling back to CPUScaledTexture", stacklevel=2)
            img = scene.visuals.Image(data, parent=self._view.scene)

        img.set_gl_state("additive", depth_test=False)
        img.interactive = True
        handle = VispyImageHandle(img)
        self._elements[img] = handle
        if data is not None:
            self.set_range()
        return handle

    def add_volume(self, data: np.ndarray | None = None) -> VispyImageHandle:
        data = _downcast(data)
        try:
            vol = scene.visuals.Volume(
                data,
                parent=self._view.scene,
                interpolation="nearest",
                texture_format="auto",
            )
        except ValueError as e:
            warnings.warn(f"{e}. Falling back to CPUScaledTexture", stacklevel=2)
            vol = scene.visuals.Volume(
                data, parent=self._view.scene, interpolation="nearest"
            )

        vol.set_gl_state("additive", depth_test=False)
        vol.interactive = True
        handle = VispyImageHandle(vol)
        self._elements[vol] = handle
        if data is not None:
            self.set_range()
        return handle

    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: _cmap.Color | None = None,
        border_color: _cmap.Color | None = None,
    ) -> VispyRoiHandle:
        """Add a new Rectangular ROI node to the scene."""
        roi = RectangularROI(parent=self._view.scene)
        handle = VispyRoiHandle(roi)
        self._elements[roi] = handle
        for h in roi._handles:
            self._elements[h] = VispyHandleHandle(h, handle)
        if vertices:
            handle.set_vertices(vertices)
            self.set_range()
        handle.set_color(color)
        handle.set_border_color(border_color)
        return handle

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0.01,
    ) -> None:
        """Update the range of the PanZoomCamera.

        When called with no arguments, the range is set to the full extent of the data.
        """
        # temporary
        self._camera.set_range()
        return
        _x = [0.0, 0.0]
        _y = [0.0, 0.0]
        _z = [0.0, 0.0]

        for handle in self._elements.values():
            if isinstance(handle, VispyImageHandle):
                shape = handle.data.shape
                _x[1] = max(_x[1], shape[0])
                _y[1] = max(_y[1], shape[1])
                if len(shape) > 2:
                    _z[1] = max(_z[1], shape[2])
            elif isinstance(handle, VispyRoiHandle):
                for v in handle.vertices:
                    _x[0] = min(_x[0], v[0])
                    _x[1] = max(_x[1], v[0])
                    _y[0] = min(_y[0], v[1])
                    _y[1] = max(_y[1], v[1])
                    if len(v) > 2:
                        _z[0] = min(_z[0], v[2])
                        _z[1] = max(_z[1], v[2])

        x = cast(tuple[float, float], _x) if x is None else x
        y = cast(tuple[float, float], _y) if y is None else y
        z = cast(tuple[float, float], _z) if z is None else z

        is_3d = isinstance(self._camera, scene.ArcballCamera)
        if is_3d:
            self._camera._quaternion = DEFAULT_QUATERNION
        self._view.camera.set_range(x=x, y=y, z=z, margin=margin)
        if is_3d:
            max_size = max(x[1], y[1], z[1])
            self._camera.scale_factor = max_size + 6

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        return self._view.scene.transform.imap(pos_xy)[:3]  # type: ignore [no-any-return]

    def elements_at(self, pos_xy: tuple[float, float]) -> list[CanvasElement]:
        elements = []
        visuals = self._canvas.visuals_at(pos_xy)
        for vis in visuals:
            if (handle := self._elements.get(vis)) is not None:
                elements.append(handle)
        return elements


def _downcast(data: np.ndarray | None) -> np.ndarray | None:
    """Downcast >32bit data to 32bit."""
    # downcast to 32bit, preserving int/float
    if data is not None:
        if np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize > 2:
            warnings.warn("Downcasting integer data to uint16.", stacklevel=2)
            data = data.astype(np.uint16)
        elif np.issubdtype(data.dtype, np.floating) and data.dtype.itemsize > 4:
            data = data.astype(np.float32)
    return data
