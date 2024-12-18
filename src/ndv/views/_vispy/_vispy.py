from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import WeakKeyDictionary

import numpy as np
import vispy
import vispy.app
import vispy.color
import vispy.scene
import vispy.visuals
from vispy import scene
from vispy.util.quaternion import Quaternion

from ndv._types import (
    CursorType,
    MouseButton,
    MouseMoveEvent,
    MousePressEvent,
    MouseReleaseEvent,
)
from ndv.models._viewer_model import CanvasMode, ViewerModel
from ndv.views.bases import ArrayCanvas, filter_mouse_events
from ndv.views.bases.graphics._canvas_elements import (
    BoundingBox,
    CanvasElement,
    ImageHandle,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap as _cmap


turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


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

    def clim(self) -> Any:
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

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        return None


class VispyBoundingBox(BoundingBox):
    owner_of: WeakKeyDictionary[scene.Node, VispyBoundingBox] = WeakKeyDictionary()

    def __init__(self, parent: Any) -> None:
        self._selected = False
        self._hover_marker: scene.Markers | None = None
        self._move_mode: BoundingBox.MoveMode | None = None
        # NB _move_anchor has different meanings depending on _move_mode
        self._move_anchor: tuple[float, float] = (0, 0)

        self._rect = scene.Rectangle(
            center=[0, 0], width=1, height=1, border_color="yellow", parent=parent
        )
        # NB: Should be greater than image orders BUT NOT handle order
        self._rect.order = 10
        VispyBoundingBox.owner_of[self._rect] = self
        self._rect.interactive = True

        self._handle_data = np.zeros((4, 1, 2))
        self._handles: list[scene.Markers] = []
        for i in range(4):
            h = scene.Markers(pos=self._handle_data[i], parent=parent)
            # NB: Should be greater than image orders and rect order
            h.order = 100
            h.interactive = True
            VispyBoundingBox.owner_of[h] = self
            self._handles.append(h)

        self.set_fill("transparent")
        self.set_border("yellow")
        self.set_handles("white")
        self.set_visible(False)

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        visible = self.visible()
        for h in self._handles:
            h.visible = selected and visible

    def _set_hover(self, vis: scene.Node) -> None:
        self._hover_marker = vis if isinstance(vis, scene.Markers) else None

    def set_fill(self, color: Any) -> None:
        color = vispy.color.Color(color)
        # NB We need alpha>0 for selection
        color.alpha = max(color.alpha, 1e-6)
        self._rect.color = color

    def set_border(self, color: Any) -> None:
        self._rect.border_color = color

    # TODO: Misleading name?
    def set_handles(self, color: Any) -> None:
        for h in self._handles:
            h.set_data(face_color=color)

    def set_bounding_box(
        self, mi: tuple[float, float], ma: tuple[float, float]
    ) -> None:
        # NB: Support two diagonal points, not necessarily true min/max
        x1 = float(min(mi[0], ma[0]))
        y1 = float(min(mi[1], ma[1]))
        x2 = float(max(mi[0], ma[0]))
        y2 = float(max(mi[1], ma[1]))

        # Update rectangle
        self._rect.center = [(x1 + x2) / 2, (y1 + y2) / 2]
        self._rect.width = max(float(x2 - x1), 1e-30)
        self._rect.height = max(float(y2 - y1), 1e-30)

        # Update handles
        self._handle_data[0] = x1, y1
        self._handle_data[1] = x2, y1
        self._handle_data[2] = x2, y2
        self._handle_data[3] = x1, y2
        for i, h in enumerate(self._handles):
            h.set_data(pos=self._handle_data[i])

        # FIXME: These should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._rect._bounds_changed()
        for v in self._rect._subvisuals:
            v._bounds_changed()
        for h in self._handles:
            h._bounds_changed()

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        # Convert canvas -> world
        world_pos = (event.x, event.y)
        # moving a handle
        if self._move_mode == BoundingBox.MoveMode.HANDLE:
            # The anchor is set to the opposite handle, which never moves.
            self.boundingBoxChanged.emit((world_pos, self._move_anchor))
        # translating the whole roi
        elif self._move_mode == BoundingBox.MoveMode.TRANSLATE:
            # The anchor is the mouse position reported in the previous mouse event.
            dx = world_pos[0] - self._move_anchor[0]
            dy = world_pos[1] - self._move_anchor[1]
            # If the mouse moved (dx, dy) between events, the whole ROI needs to be
            # translated that amount.
            new_min = (self._handle_data[0, 0, 0] + dx, self._handle_data[0, 0, 1] + dy)
            new_max = (self._handle_data[2, 0, 0] + dx, self._handle_data[2, 0, 1] + dy)
            self.boundingBoxChanged.emit((new_min, new_max))
            self._move_anchor = world_pos

        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        self.set_selected(True)
        # Convert canvas -> world
        world_pos = (event.x, event.y)
        # If a marker is pressed
        if self._hover_marker is not None:
            idx = self._handles.index(self._hover_marker)
            opposite_idx = (idx + 2) % 4
            self._move_mode = BoundingBox.MoveMode.HANDLE
            self._move_anchor = tuple(self._handle_data[opposite_idx, 0].copy())
        # If the rectangle is pressed
        else:
            self._move_mode = BoundingBox.MoveMode.TRANSLATE
            self._move_anchor = world_pos
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False

    def get_cursor(self, pos: tuple[float, float]) -> CursorType | None:
        if self._hover_marker:
            center = self._rect.center
            if pos[0] < center[0] and pos[1] < center[1]:
                return CursorType.FDIAG_ARROW
            if pos[0] > center[0] and pos[1] > center[1]:
                return CursorType.FDIAG_ARROW
            return CursorType.BDIAG_ARROW
        return CursorType.ALL_ARROW

    def visible(self) -> bool:
        return bool(self._rect.visible)

    def set_visible(self, visible: bool) -> None:
        self._rect.visible = visible
        for h in self._handles:
            h.visible = visible and self._selected

    def remove(self) -> None:
        self._rect.parent = None
        for h in self._handles:
            h.parent = None


class VispyViewerCanvas(ArrayCanvas):
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self, viewer_model: ViewerModel) -> None:
        self._viewer = viewer_model

        self._canvas = scene.SceneCanvas(size=(600, 600))
        self._canvas.measure_fps()

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
        self._selection: VispyBoundingBox | None = None
        # TODO: Weak Reference?
        self._last_roi_created: VispyBoundingBox | None = None

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

    def refresh(self) -> None:
        self._canvas.update()

    def add_image(self, data: np.ndarray | None = None) -> VispyImageHandle:
        """Add a new Image node to the scene."""
        img = scene.visuals.Image(data, parent=self._view.scene)
        img.set_gl_state("additive", depth_test=False)
        img.interactive = True
        handle = VispyImageHandle(img)
        self._elements[img] = handle
        if data is not None:
            self.set_range()
        return handle

    def add_volume(self, data: np.ndarray | None = None) -> VispyImageHandle:
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

    def add_bounding_box(self) -> VispyBoundingBox:
        """Add a new Rectangular ROI node to the scene."""
        self._last_roi_created = VispyBoundingBox(parent=self._view.scene)
        return self._last_roi_created

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

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        if self._selection:
            self._selection.set_selected(False)
            self._selection = None
        canvas_pos = (event.x, event.y)
        world_pos = self.canvas_to_world(canvas_pos)[:2]

        if self._viewer.mode == CanvasMode.CREATE_ROI:
            if self._last_roi_created is None:
                raise ValueError("No ROI to create!")
            self._selection = self._last_roi_created
            # HACK: Provide a non-zero starting size so that if the user clicks
            # and immediately releases, it's visible and can be selected again
            _min = world_pos
            _max = (world_pos[0] + 1, world_pos[1] + 1)
            self._selection.set_bounding_box(_min, _max)
            self._selection.set_visible(True)
            self._selection.set_selected(True)
            self._selection._set_hover(self._selection._handles[2])
            self._viewer.mode = CanvasMode.PAN_ZOOM

        # Find all visuals at the point
        for vis in self._canvas.visuals_at(canvas_pos):
            # If any belong to a bounding box, direct output there
            if bbox := VispyBoundingBox.owner_of.get(vis, None):
                self._selection = bbox
                # FIXME: Use the same event?
                self._selection.on_mouse_press(
                    MousePressEvent(world_pos[0], world_pos[1], event.btn)
                )
                self._camera.interactive = False
                return False

        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        ev_pos = (event.x, event.y)
        for vis in self._canvas.visuals_at(ev_pos):
            if bbox := VispyBoundingBox.owner_of.get(vis, None):
                bbox._set_hover(vis)
                break
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
        if self._selection:
            self._selection.on_mouse_release(event)
        if self._viewer.mode == CanvasMode.CREATE_ROI:
            self._viewer.mode = CanvasMode.PAN_ZOOM
        self._camera.interactive = True
        return False

    def get_cursor(self, pos: tuple[float, float]) -> CursorType:
        if self._viewer.mode == CanvasMode.CREATE_ROI:
            return CursorType.CROSS
        for vis in self._canvas.visuals_at(pos):
            if bbox := VispyBoundingBox.owner_of.get(vis, None):
                world_pos = self.canvas_to_world(pos)[:2]
                if cursor := bbox.get_cursor(world_pos):
                    return cursor
        return CursorType.DEFAULT
