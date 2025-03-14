from __future__ import annotations

import warnings
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import ReferenceType, WeakKeyDictionary

import cmap as _cmap
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
from ndv.models._viewer_model import ArrayViewerModel, InteractionMode
from ndv.views._app import filter_mouse_events
from ndv.views.bases import ArrayCanvas
from ndv.views.bases._graphics._canvas_elements import (
    CanvasElement,
    ImageHandle,
    RectangularROIHandle,
    ROIMoveMode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


class VispyImageHandle(ImageHandle):
    def __init__(self, visual: scene.Image | scene.Volume) -> None:
        self._visual = visual
        self._allowed_dims = {2, 3} if isinstance(visual, scene.visuals.Image) else {3}

    def data(self) -> np.ndarray:
        try:
            return self._visual._data  # type: ignore [no-any-return]
        except AttributeError:
            return self._visual._last_data  # type: ignore [no-any-return]

    def set_data(self, data: np.ndarray) -> None:
        if data.ndim not in self._allowed_dims:
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

    def colormap(self) -> _cmap.Colormap:
        return self._cmap  # FIXME

    def set_colormap(self, cmap: _cmap.Colormap) -> None:
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

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType | None:
        return None


class VispyRectangle(RectangularROIHandle):
    def __init__(self, parent: Any) -> None:
        self._selected = False
        self._move_mode: ROIMoveMode | None = None
        # NB _move_anchor has different meanings depending on _move_mode
        self._move_anchor: tuple[float, float] = (0, 0)

        # Rectangle handles both fill and border
        self._rect = scene.Rectangle(center=[0, 0], width=1, height=1, parent=parent)
        # NB: Should be greater than image orders BUT NOT handle order
        self._rect.order = 10
        self._rect.interactive = True

        self._handle_data = np.zeros((4, 2))
        self._handle_size = 10  # px
        self._handles = scene.Markers(
            pos=self._handle_data,
            size=self._handle_size,
            scaling="fixed",
            parent=parent,
        )
        # NB: Should be greater than image orders and rect order
        self._handles.order = 100
        self._handles.interactive = True

        self.set_fill(_cmap.Color("transparent"))
        self.set_border(_cmap.Color("yellow"))
        self.set_handles(_cmap.Color("white"))
        self.set_visible(False)

    def _tform(self) -> scene.transforms.BaseTransform:
        return self._rect.transforms.get_transform("canvas", "scene")

    def can_select(self) -> bool:
        return True

    def selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._handles.visible = selected and self.visible()

    def set_fill(self, color: _cmap.Color) -> None:
        _vis_color = vispy.color.Color(color.hex)
        # NB We need alpha>0 for selection
        _vis_color.alpha = max(color.alpha, 1e-6)
        self._rect.color = _vis_color

    def set_border(self, color: _cmap.Color) -> None:
        _vis_color = vispy.color.Color(color.hex)
        _vis_color.alpha = color.alpha
        self._rect.border_color = _vis_color

    # TODO: Misleading name?
    def set_handles(self, color: _cmap.Color) -> None:
        _vis_color = vispy.color.Color(color.hex)
        _vis_color.alpha = color.alpha
        self._handles.set_data(face_color=_vis_color)

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
        self._handles.set_data(pos=self._handle_data)

        # FIXME: These should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._rect._bounds_changed()
        for v in self._rect._subvisuals:
            v._bounds_changed()
        self._handles._bounds_changed()

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        # Convert canvas -> world
        canvas_pos = (event.x, event.y)
        world_pos = self._tform().map(canvas_pos)[:2]
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
            new_min = (self._handle_data[0, 0] + dx, self._handle_data[0, 1] + dy)
            new_max = (self._handle_data[2, 0] + dx, self._handle_data[2, 1] + dy)
            self.boundingBoxChanged.emit((new_min, new_max))
            self._move_anchor = world_pos

        return False

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        self.set_selected(True)
        # Convert canvas -> world
        canvas_pos = (event.x, event.y)
        world_pos = self._tform().map(canvas_pos)[:2]
        drag_idx = self._handle_under(world_pos)
        # If a marker is pressed
        if drag_idx is not None:
            opposite_idx = (drag_idx + 2) % 4
            self._move_mode = ROIMoveMode.HANDLE
            self._move_anchor = tuple(self._handle_data[opposite_idx].copy())
        # If the rectangle is pressed
        else:
            self._move_mode = ROIMoveMode.TRANSLATE
            self._move_anchor = world_pos
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        return False

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType | None:
        canvas_pos = (mme.x, mme.y)
        pos = self._tform().map(canvas_pos)[:2]
        if self._handle_under(pos) is not None:
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
        self._handles.visible = visible and self.selected()

    def remove(self) -> None:
        self._rect.parent = None
        self._handles.parent = None

    def _handle_under(self, pos: Sequence[float]) -> int | None:
        """Returns an int in [0, 3], or None.

        If an int i, means that the handle at self._positions[i] is at pos.
        If None, there is no handle at pos.
        """
        rad2 = (self._handle_size / 2) ** 2
        for i, p in enumerate(self._handle_data):
            if (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 <= rad2:
                return i
        return None


class VispyArrayCanvas(ArrayCanvas):
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self, viewer_model: ArrayViewerModel) -> None:
        self._viewer = viewer_model

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
        self._selection: CanvasElement | None = None
        # Maintain weak reference to last ROI created
        self._last_roi_created: ReferenceType[VispyRectangle] | None = None

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

    def add_bounding_box(self) -> VispyRectangle:
        """Add a new Rectangular ROI node to the scene."""
        roi = VispyRectangle(parent=self._view.scene)
        roi.set_visible(False)
        self._elements[roi._handles] = roi
        self._elements[roi._rect] = roi
        self._last_roi_created = ReferenceType(roi)
        return roi

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
            elif isinstance(handle, VispyRectangle):
                for v in handle.vertices:
                    _x[0] = min(_x[0], v[0])
                    _x[1] = max(_x[1], v[0])
                    _y[0] = min(_y[0], v[1])
                    _y[1] = max(_y[1], v[1])
                    if len(v) > 2:
                        _z[0] = min(_z[0], v[2])
                        _z[1] = max(_z[1], v[2])

        x = cast("tuple[float, float]", _x) if x is None else x
        y = cast("tuple[float, float]", _y) if y is None else y
        z = cast("tuple[float, float]", _z) if z is None else z

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
                # new_roi.set_bounding_box(_min, _max)
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

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType:
        if self._viewer.interaction_mode == InteractionMode.CREATE_ROI:
            return CursorType.CROSS
        for vis in self.elements_at((mme.x, mme.y)):
            if cursor := vis.get_cursor(mme):
                return cursor
        return CursorType.DEFAULT


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
