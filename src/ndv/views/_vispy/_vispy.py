from __future__ import annotations

import warnings
from contextlib import suppress
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast
from weakref import WeakKeyDictionary

import cmap
import numpy as np
import vispy
import vispy.scene
import vispy.visuals
from psygnal import Signal
from qtpy.QtCore import Qt
from vispy import scene
from vispy.color import Color
from vispy.util.quaternion import Quaternion

from ndv.views import get_cursor_class
from ndv.views.protocols import CursorType, PCanvas, PHistogramView

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable

    import numpy.typing as npt
    from qtpy.QtWidgets import QWidget

    from ndv.views.protocols import CanvasElement

turn = np.sin(np.pi / 4)
DEFAULT_QUATERNION = Quaternion(turn, turn, 0, 0)


class Handle(scene.visuals.Markers):
    """A Marker that allows specific ROI alterations."""

    def __init__(
        self,
        parent: RectangularROI,
        on_move: Callable[[Sequence[float]], None] | None = None,
        cursor: Qt.CursorShape
        | Callable[[Sequence[float]], Qt.CursorShape] = Qt.CursorShape.SizeAllCursor,
    ) -> None:
        super().__init__(parent=parent)
        self.unfreeze()
        self.parent = parent
        # on_move function(s)
        self.on_move: list[Callable[[Sequence[float]], None]] = []
        if on_move:
            self.on_move.append(on_move)
        # cusror preference function
        if isinstance(cursor, Qt.CursorShape):
            self._cursor_at = cast(
                "Callable[[Sequence[float]], Qt.CursorShape]", lambda _: cursor
            )
        else:
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

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
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

    def _handle_cursor_pref(self, handle_pos: Sequence[float]) -> Qt.CursorShape:
        # Bottom left handle
        if handle_pos[0] < self.center[0] and handle_pos[1] < self.center[1]:
            return Qt.CursorShape.SizeFDiagCursor
        # Top right handle
        if handle_pos[0] > self.center[0] and handle_pos[1] > self.center[1]:
            return Qt.CursorShape.SizeFDiagCursor
        # Top left, bottom right
        return Qt.CursorShape.SizeBDiagCursor

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

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        return Qt.CursorShape.SizeAllCursor

    # ------------------- End EditableROI interface -------------------------


class VispyImageHandle:
    def __init__(self, visual: scene.visuals.Image | scene.visuals.Volume) -> None:
        self._visual = visual
        self._ndim = 2 if isinstance(visual, scene.visuals.Image) else 3

    @property
    def data(self) -> np.ndarray:
        try:
            return self._visual._data  # type: ignore [no-any-return]
        except AttributeError:
            return self._visual._last_data  # type: ignore [no-any-return]

    @data.setter
    def data(self, data: np.ndarray) -> None:
        if not data.ndim == self._ndim:
            warnings.warn(
                f"Got wrong number of dimensions ({data.ndim}) for vispy "
                f"visual of type {type(self._visual)}.",
                stacklevel=2,
            )
            return
        self._visual.set_data(data)

    @property
    def visible(self) -> bool:
        return bool(self._visual.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._visual.visible = visible

    @property
    def can_select(self) -> bool:
        return False

    @property
    def selected(self) -> bool:
        return False

    @selected.setter
    def selected(self, selected: bool) -> None:
        raise NotImplementedError("Images cannot be selected")

    @property
    def clim(self) -> Any:
        return self._visual.clim

    @clim.setter
    def clim(self, clims: tuple[float, float]) -> None:
        with suppress(ZeroDivisionError):
            self._visual.clim = clims

    @property
    def cmap(self) -> cmap.Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: cmap.Colormap) -> None:
        self._cmap = cmap
        self._visual.cmap = cmap.to_vispy()

    @property
    def transform(self) -> np.ndarray:
        raise NotImplementedError

    @transform.setter
    def transform(self, transform: np.ndarray) -> None:
        raise NotImplementedError

    def start_move(self, pos: Sequence[float]) -> None:
        pass

    def move(self, pos: Sequence[float]) -> None:
        pass

    def remove(self) -> None:
        self._visual.parent = None

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        return None


# FIXME: Unfortunate naming :)
class VispyHandleHandle:
    def __init__(self, handle: Handle, parent: CanvasElement) -> None:
        self._handle = handle
        self._parent = parent

    @property
    def visible(self) -> bool:
        return cast("bool", self._handle.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._handle.visible = visible

    @property
    def can_select(self) -> bool:
        return True

    @property
    def selected(self) -> bool:
        return self._handle.selected

    @selected.setter
    def selected(self, selected: bool) -> None:
        self._handle.selected = selected

    def start_move(self, pos: Sequence[float]) -> None:
        self._handle.start_move(pos)

    def move(self, pos: Sequence[float]) -> None:
        self._handle.move(pos)

    def remove(self) -> None:
        self._parent.remove()

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        return self._handle.cursor_at(pos)


class VispyRoiHandle:
    def __init__(self, roi: RectangularROI) -> None:
        self._roi = roi

    @property
    def vertices(self) -> Sequence[Sequence[float]]:
        return self._roi.vertices

    @vertices.setter
    def vertices(self, vertices: Sequence[Sequence[float]]) -> None:
        self._roi.vertices = vertices

    @property
    def visible(self) -> bool:
        return bool(self._roi.visible)

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._roi.visible = visible

    @property
    def can_select(self) -> bool:
        return True

    @property
    def selected(self) -> bool:
        return self._roi.selected

    @selected.setter
    def selected(self, selected: bool) -> None:
        self._roi.selected = selected

    def start_move(self, pos: Sequence[float]) -> None:
        self._roi.start_move(pos)

    def move(self, pos: Sequence[float]) -> None:
        self._roi.move(pos)

    @property
    def color(self) -> Any:
        return self._roi.color

    @color.setter
    def color(self, color: Any | None = None) -> None:
        if color is None:
            color = cmap.Color("transparent")
        if not isinstance(color, cmap.Color):
            color = cmap.Color(color)
        # NB: To enable dragging the shape within the border,
        # we require a positive alpha.
        alpha = max(color.alpha, 1e-6)
        self._roi.color = Color(color.hex, alpha=alpha)

    @property
    def border_color(self) -> Any:
        return self._roi.border_color

    @border_color.setter
    def border_color(self, color: Any | None = None) -> None:
        if color is None:
            color = cmap.Color("yellow")
        if not isinstance(color, cmap.Color):
            color = cmap.Color(color)
        self._roi.border_color = Color(color.hex, alpha=color.alpha)

    def remove(self) -> None:
        self._roi.parent = None

    def cursor_at(self, pos: Sequence[float]) -> Qt.CursorShape | None:
        return self._roi.cursor_at(pos)


class VispyViewerCanvas(PCanvas):
    """Vispy-based viewer for data.

    All vispy-specific code is encapsulated in this class (and non-vispy canvases
    could be swapped in if needed as long as they implement the same interface).
    """

    def __init__(self) -> None:
        self._canvas = scene.SceneCanvas(size=(600, 600))
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

    def qwidget(self) -> QWidget:
        return cast("QWidget", self._canvas.native)

    def refresh(self) -> None:
        self._canvas.update()

    def add_image(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> VispyImageHandle:
        """Add a new Image node to the scene."""
        img = scene.visuals.Image(data, parent=self._view.scene)
        img.set_gl_state("additive", depth_test=False)
        img.interactive = True
        handle = VispyImageHandle(img)
        self._elements[img] = handle
        if data is not None:
            self.set_range()
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def add_volume(
        self, data: np.ndarray | None = None, cmap: cmap.Colormap | None = None
    ) -> VispyImageHandle:
        vol = scene.visuals.Volume(
            data, parent=self._view.scene, interpolation="nearest"
        )
        vol.set_gl_state("additive", depth_test=False)
        vol.interactive = True
        handle = VispyImageHandle(vol)
        self._elements[vol] = handle
        if data is not None:
            self.set_range()
        if cmap is not None:
            handle.cmap = cmap
        return handle

    def add_roi(
        self,
        vertices: Sequence[tuple[float, float]] | None = None,
        color: cmap.Color | None = None,
        border_color: cmap.Color | None = None,
    ) -> VispyRoiHandle:
        """Add a new Rectangular ROI node to the scene."""
        roi = RectangularROI(parent=self._view.scene)
        handle = VispyRoiHandle(roi)
        self._elements[roi] = handle
        for h in roi._handles:
            self._elements[h] = VispyHandleHandle(h, handle)
        if vertices:
            handle.vertices = vertices
            self.set_range()
        handle.color = color
        handle.border_color = border_color
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


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


if TYPE_CHECKING:
    # just here cause vispy has poor type hints
    from collections.abc import Sequence

    from vispy.app.canvas import MouseEvent

    class Grid(scene.Grid):
        def add_view(
            self,
            row: int | None = None,
            col: int | None = None,
            row_span: int = 1,
            col_span: int = 1,
            **kwargs: Any,
        ) -> scene.ViewBox:
            super().add_view(...)

        def add_widget(
            self,
            widget: None | scene.Widget = None,
            row: int | None = None,
            col: int | None = None,
            row_span: int = 1,
            col_span: int = 1,
            **kwargs: Any,
        ) -> scene.Widget:
            super().add_widget(...)

    class WidgetKwargs(TypedDict, total=False):
        pos: tuple[float, float]
        size: tuple[float, float]
        border_color: str
        border_width: float
        bgcolor: str
        padding: float
        margin: float

    class TextVisualKwargs(TypedDict, total=False):
        text: str
        color: str
        bold: bool
        italic: bool
        face: str
        font_size: float
        pos: tuple[float, float] | tuple[float, float, float]
        rotation: float
        method: Literal["cpu", "gpu"]
        depth_test: bool

    class AxisWidgetKwargs(TypedDict, total=False):
        orientation: Literal["left", "bottom"]
        tick_direction: tuple[int, int]
        axis_color: str
        tick_color: str
        text_color: str
        minor_tick_length: float
        major_tick_length: float
        tick_width: float
        tick_label_margin: float
        tick_font_size: float
        axis_width: float
        axis_label: str
        axis_label_margin: float
        axis_font_size: float
        font_size: float  # overrides tick_font_size and axis_font_size


__all__ = ["PlotWidget"]


DEFAULT_AXIS_KWARGS: AxisWidgetKwargs = {
    "text_color": "w",
    "axis_color": "w",
    "tick_color": "w",
    "tick_width": 1,
    "tick_font_size": 8,
    "tick_label_margin": 12,
    "axis_label_margin": 50,
    "minor_tick_length": 2,
    "major_tick_length": 5,
    "axis_width": 1,
    "axis_font_size": 10,
}


class Component(str, Enum):
    PAD_LEFT = "pad_left"
    PAD_RIGHT = "pad_right"
    PAD_BOTTOM = "pad_bottom"
    TITLE = "title"
    CBAR_TOP = "cbar_top"
    CBAR_LEFT = "cbar_left"
    CBAR_RIGHT = "cbar_right"
    CBAR_BOTTOM = "cbar_bottom"
    YAXIS = "yaxis"
    XAXIS = "xaxis"
    XLABEL = "xlabel"
    YLABEL = "ylabel"

    def __str__(self) -> str:
        return self.value


class PlotWidget(scene.Widget):
    """Widget to facilitate plotting.

    Parameters
    ----------
    fg_color : str
        The default color for the plot.
    xlabel : str
        The x-axis label.
    ylabel : str
        The y-axis label.
    title : str
        The title of the plot.
    lock_axis : {'x', 'y', None}
        Prevent panning and zooming along a particular axis.
    **widget_kwargs : dict
        Keyword arguments to pass to the parent class.
    """

    def __init__(
        self,
        fg_color: str = "k",
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        lock_axis: Literal["x", "y", None] = None,
        **widget_kwargs: Unpack[WidgetKwargs],
    ) -> None:
        self._fg_color = fg_color
        self._visuals: list[scene.VisualNode] = []
        super().__init__(**widget_kwargs)
        self.unfreeze()
        self.grid = cast("Grid", self.add_grid(spacing=0, margin=10))

        title_kwargs: TextVisualKwargs = {"font_size": 14, "color": "w"}
        label_kwargs: TextVisualKwargs = {"font_size": 10, "color": "w"}
        self._title = scene.Label(str(title), **title_kwargs)
        self._xlabel = scene.Label(str(xlabel), **label_kwargs)
        self._ylabel = scene.Label(str(ylabel), rotation=-90, **label_kwargs)

        axis_kwargs: AxisWidgetKwargs = DEFAULT_AXIS_KWARGS
        self.yaxis = scene.AxisWidget(orientation="left", **axis_kwargs)
        self.xaxis = scene.AxisWidget(orientation="bottom", **axis_kwargs)

        # 2D Plot layout:
        #
        #         c0        c1      c2      c3      c4      c5         c6
        #     +----------+-------+-------+-------+---------+---------+-----------+
        #  r0 |          |                       |  title  |         |           |
        #     |          +-----------------------+---------+---------+           |
        #  r1 |          |                       |  cbar   |         |           |
        #     |----------+-------+-------+-------+---------+---------+ ----------|
        #  r2 | pad_left | cbar  | ylabel| yaxis |  view   | cbar    | pad_right |
        #     |----------+-------+-------+-------+---------+---------+ ----------|
        #  r3 |          |                       |  xaxis  |         |           |
        #     |          +-----------------------+---------+---------+           |
        #  r4 |          |                       |  xlabel |         |           |
        #     |          +-----------------------+---------+---------+           |
        #  r5 |          |                       |  cbar   |         |           |
        #     |---------+------------------------+---------+---------+-----------|
        #  r6 |                                 | pad_bottom |                   |
        #     +---------+------------------------+---------+---------+-----------+

        self._grid_wdgs: dict[Component, scene.Widget] = {}
        for name, row, col, widget in [
            (Component.PAD_LEFT, 2, 0, None),
            (Component.PAD_RIGHT, 2, 6, None),
            (Component.PAD_BOTTOM, 6, 4, None),
            (Component.TITLE, 0, 4, self._title),
            (Component.CBAR_TOP, 1, 4, None),
            (Component.CBAR_LEFT, 2, 1, None),
            (Component.CBAR_RIGHT, 2, 5, None),
            (Component.CBAR_BOTTOM, 5, 4, None),
            (Component.YAXIS, 2, 3, self.yaxis),
            (Component.XAXIS, 3, 4, self.xaxis),
            (Component.XLABEL, 4, 4, self._xlabel),
            (Component.YLABEL, 2, 2, self._ylabel),
        ]:
            self._grid_wdgs[name] = wdg = self.grid.add_widget(widget, row=row, col=col)
            # If we don't set max size, they will expand to fill the entire grid
            # occluding pretty much everything else.
            if str(name).startswith(("cbar", "pad")):
                if name in {
                    Component.PAD_LEFT,
                    Component.PAD_RIGHT,
                    Component.CBAR_LEFT,
                    Component.CBAR_RIGHT,
                }:
                    wdg.width_max = 2
                else:
                    wdg.height_max = 2

        # The main view into which plots are added
        self._view = self.grid.add_view(row=2, col=4)

        # NOTE: this is a mess of hardcoded values... not sure whether they will work
        # cross-platform.  Note that `width_max` and `height_max` of 2 is actually
        # *less* visible than 0 for some reason.  They should also be extracted into
        # some sort of `hide/show` logic for each component
        self._grid_wdgs[Component.YAXIS].width_max = 30  # otherwise it takes too much
        self._grid_wdgs[Component.PAD_LEFT].width_max = 20  # otherwise you get clipping
        self._grid_wdgs[Component.XAXIS].height_max = 20  # otherwise it takes too much
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.title = title

        # VIEWBOX (this has to go last, see vispy #1748)
        self.camera = self._view.camera = PanZoom1DCamera(lock_axis)
        # this has to come after camera is set
        self.xaxis.link_view(self._view)
        self.yaxis.link_view(self._view)
        self.freeze()

    @property
    def title(self) -> str:
        """The title label."""
        return self._title.text  # type: ignore [no-any-return]

    @title.setter
    def title(self, text: str) -> None:
        """Set the title of the plot."""
        self._title.text = text
        wdg = self._grid_wdgs[Component.TITLE]
        wdg.height_min = wdg.height_max = 30 if text else 2

    @property
    def xlabel(self) -> str:
        """The x-axis label."""
        return self._xlabel.text  # type: ignore [no-any-return]

    @xlabel.setter
    def xlabel(self, text: str) -> None:
        """Set the x-axis label."""
        self._xlabel.text = text
        wdg = self._grid_wdgs[Component.XLABEL]
        wdg.height_min = wdg.height_max = 40 if text else 2

    @property
    def ylabel(self) -> str:
        """The y-axis label."""
        return self._ylabel.text  # type: ignore [no-any-return]

    @ylabel.setter
    def ylabel(self, text: str) -> None:
        """Set the x-axis label."""
        self._ylabel.text = text
        wdg = self._grid_wdgs[Component.YLABEL]
        wdg.width_min = wdg.width_max = 20 if text else 2

    def lock_axis(self, axis: Literal["x", "y", None]) -> None:
        """Prevent panning and zooming along a particular axis."""
        self.camera._axis = axis
        # self.camera.set_range()


class PanZoom1DCamera(scene.cameras.PanZoomCamera):
    """Camera that allows panning and zooming along one axis only.

    Parameters
    ----------
    axis : {'x', 'y', None}
        The axis along which to allow panning and zooming.
    *args : tuple
        Positional arguments to pass to the parent class.
    **kwargs : dict
        Keyword arguments to pass to the parent class.
    """

    def __init__(
        self, axis: Literal["x", "y", None] = None, *args: Any, **kwargs: Any
    ) -> None:
        self._axis: Literal["x", "y", None] = axis
        super().__init__(*args, **kwargs)

    @property
    def axis_index(self) -> Literal[0, 1, None]:
        """Return the index of the axis along which to pan and zoom."""
        if self._axis in ("x", 0):
            return 0
        elif self._axis in ("y", 1):
            return 1
        return None

    def zoom(
        self,
        factor: float | tuple[float, float],
        center: tuple[float, ...] | None = None,
    ) -> None:
        """Zoom the camera by `factor` around `center`."""
        if self.axis_index is None:
            super().zoom(factor, center=center)
            return

        if isinstance(factor, (float, int)):
            factor = (factor, factor)
        _factor = list(factor)
        _factor[self.axis_index] = 1
        super().zoom(_factor, center=center)

    def pan(self, pan: Sequence[float]) -> None:
        """Pan the camera by `pan`."""
        if self.axis_index is None:
            super().pan(pan)
            return
        _pan = list(pan)
        _pan[self.axis_index] = 0
        super().pan(*_pan)

    def set_range(
        self,
        x: tuple | None = None,
        y: tuple | None = None,
        z: tuple | None = None,
        margin: float = 0,  # overriding to create a different default from super()
    ) -> None:
        """Reset the camera view to the specified range."""
        super().set_range(x, y, z, margin)


# TODO: Move much of this logic to _qt
class VispyHistogramView(PHistogramView):
    """A HistogramView on a VisPy SceneCanvas."""

    visibleChanged = Signal(bool)
    autoscaleChanged = Signal(bool)
    cmapChanged = Signal(cmap.Colormap)
    climsChanged = Signal(tuple)
    gammaChanged = Signal(float)

    def __init__(self) -> None:
        # ------------ data and state ------------ #

        self._values: Sequence[float] | np.ndarray | None = None
        self._bin_edges: Sequence[float] | np.ndarray | None = None
        self._clims: tuple[float, float] | None = None
        self._gamma: float = 1

        # the currently grabbed object
        self._grabbed: Grabbable = Grabbable.NONE
        # whether the y-axis is logarithmic
        self._log_y: bool = False
        # whether the histogram is vertical
        self._vertical: bool = False
        # The values of the left and right edges on the canvas (respectively)
        self._domain: tuple[float, float] | None = None
        # The values of the bottom and top edges on the canvas (respectively)
        self._range: tuple[float, float] | None = None

        # ------------ VisPy Canvas ------------ #

        self._canvas = scene.SceneCanvas()
        self._canvas.unfreeze()
        self._canvas.on_mouse_press = self.on_mouse_press
        self._canvas.on_mouse_move = self.on_mouse_move
        self._canvas.on_mouse_release = self.on_mouse_release
        self._canvas.freeze()

        self._cursor = get_cursor_class()(self._canvas.native)

        ## -- Visuals -- ##

        # NB We directly use scene.Mesh, instead of scene.Histogram,
        # so that we can control the calculation of the histogram ourselves
        self._hist_mesh = scene.Mesh(color="red")

        # The Lut Line visualizes both the clims (vertical line segments connecting the
        # first two and last two points, respectively) and the gamma curve
        # (the polyline between all remaining points)
        self._lut_line = scene.LinePlot(
            data=(0),  # Dummy value to prevent resizing errors
            color="k",
            connect="strip",
            symbol=None,
            line_kind="-",
            width=1.5,
            marker_size=10.0,
            edge_color="k",
            face_color="b",
            edge_width=1.0,
        )
        self._lut_line.visible = False
        self._lut_line.order = -1

        # The gamma handle appears halfway between the clims
        self._gamma_handle_pos: np.ndarray = np.ndarray((1, 2))
        self._gamma_handle = scene.Markers(
            pos=self._gamma_handle_pos,
            size=6,
            edge_width=0,
        )
        self._gamma_handle.visible = False
        self._gamma_handle.order = -2

        # One transform to rule them all!
        self._handle_transform = scene.transforms.STTransform()
        self._lut_line.transform = self._handle_transform
        self._gamma_handle.transform = self._handle_transform

        ## -- Plot -- ##
        self.plot = PlotWidget()
        self.plot.lock_axis("y")
        self._canvas.central_widget.add_widget(self.plot)
        self.node_tform = self.plot.node_transform(self.plot._view.scene)

        self.plot._view.add(self._hist_mesh)
        self.plot._view.add(self._lut_line)
        self.plot._view.add(self._gamma_handle)

    def refresh(self) -> None:
        self._canvas.update()

    # ------------- StatsView Protocol methods ------------- #

    def set_histogram(self, values: Sequence[float], bin_edges: Sequence[float]) -> None:
        """Set the histogram values and bin edges.

        These inputs follow the same format as the return value of numpy.histogram.
        """
        self._values = values
        self._bin_edges = bin_edges
        self._update_histogram()
        if self._clims is None:
            self.set_clims((self._bin_edges[0], self._bin_edges[-1]))
            self._resize()

    def set_std_dev(self, std_dev: float) -> None:
        # Nothing to do.
        # TODO: maybe show text somewhere
        pass

    def set_average(self, average: float) -> None:
        # Nothing to do
        # TODO: maybe show text somewhere
        pass

    def view(self) -> Any:
        return self._canvas.native

    # ------------- LutView Protocol methods ------------- #

    def set_name(self, name: str) -> None:
        # Nothing to do
        # TODO: maybe show text somewhere
        pass

    def set_lut_visible(self, visible: bool) -> None:
        if self._hist_mesh is None:
            return  # pragma: no cover
        self._hist_mesh.visible = visible
        self._lut_line.visible = visible
        self._gamma_handle.visible = visible

    def set_colormap(self, lut: cmap.Colormap) -> None:
        if self._hist_mesh is not None:
            self._hist_mesh.color = lut.color_stops[-1].color.hex

    def set_gamma(self, gamma: float) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative!")
        self._gamma = gamma
        self._update_lut_lines()

    def set_clims(self, clims: tuple[float, float]) -> None:
        if clims[1] < clims[0]:
            clims = (clims[1], clims[0])
        self._clims = clims
        self._update_lut_lines()

    def set_auto_scale(self, autoscale: bool) -> None:
        # Nothing to do (yet)
        pass

    # ------------- HistogramView Protocol methods ------------- #

    def set_domain(self, bounds: tuple[float, float] | None) -> None:
        if bounds is not None:
            if bounds[0] is None or bounds[1] is None:
                # TODO: Sensible defaults?
                raise ValueError("Domain min/max cannot be None!")
            if bounds[0] > bounds[1]:
                bounds = (bounds[1], bounds[0])
        self._domain = bounds
        self._resize()

    def set_range(self, bounds: tuple[float, float] | None) -> None:
        if bounds is not None:
            if bounds[0] is None or bounds[1] is None:
                # TODO: Sensible defaults?
                raise ValueError("Range min/max cannot be None!")
            if bounds[0] > bounds[1]:
                bounds = (bounds[1], bounds[0])
        self._range = bounds
        self._resize()

    def set_vertical(self, vertical: bool) -> None:
        self._vertical = vertical
        self._update_histogram()
        self.plot.lock_axis("x" if vertical else "y")
        # When vertical, smaller values should appear at the top of the canvas
        self.plot.camera.flip = [False, vertical, False]
        self._update_lut_lines()
        self._resize()

    def set_range_log(self, enabled: bool) -> None:
        if enabled != self._log_y:
            self._log_y = enabled
            self._update_histogram()
            self._update_lut_lines()
            self._resize()

    # ------------- Private methods ------------- #

    def _update_histogram(self) -> None:
        """
        Updates the displayed histogram with current View parameters.

        NB: Much of this code is graciously borrowed from:

        https://github.com/vispy/vispy/blob/af847424425d4ce51f144a4d1c75ab4033fe39be/vispy/visuals/histogram.py#L28
        """
        if self._values is None or self._bin_edges is None:
            return  # pragma: no cover
        values = self._values
        if self._log_y:
            # Replace zero values with 1 (which will be log10(1) = 0)
            values = np.where(values == 0, 1, values)
            values = np.log10(values)

        verts, faces = _hist_counts_to_mesh(values, self._bin_edges, self._vertical)
        self._hist_mesh.set_data(vertices=verts, faces=faces)

        # FIXME: This should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._hist_mesh._bounds_changed()

    def _update_lut_lines(self, npoints: int = 256) -> None:
        if self._clims is None or self._gamma is None:
            return  # pragma: no cover

        # 2 additional points for each of the two vertical clims lines
        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)
        if self._vertical:
            # clims lines
            X[0:2], Y[0:2] = (1, 0.5), self._clims[0]
            X[-2:], Y[-2:] = (0.5, 0), self._clims[1]
            # gamma line
            X[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
            Y[2:-2] = np.linspace(self._clims[0], self._clims[1], npoints)
            midpoint = np.array([(2**-self._gamma, np.mean(self._clims))])
        else:
            # clims lines
            X[0:2], Y[0:2] = self._clims[0], (1, 0.5)
            X[-2:], Y[-2:] = self._clims[1], (0.5, 0)
            # gamma line
            X[2:-2] = np.linspace(self._clims[0], self._clims[1], npoints)
            Y[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
            midpoint = np.array([(np.mean(self._clims), 2**-self._gamma)])

        # TODO: Move to self.edit_cmap
        color = np.linspace(0.2, 0.8, npoints + 4).repeat(4).reshape(-1, 4)
        c1, c2 = [0.4] * 4, [0.7] * 4
        color[0:3] = [c1, c2, c1]
        color[-3:] = [c1, c2, c1]

        self._lut_line.set_data((X, Y), marker_size=0, color=color)
        self._lut_line.visible = True

        self._gamma_handle_pos[:] = midpoint[0]
        self._gamma_handle.set_data(pos=self._gamma_handle_pos)
        self._gamma_handle.visible = True

        # FIXME: These should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._lut_line._bounds_changed()
        for v in self._lut_line._subvisuals:
            v._bounds_changed()
        self._gamma_handle._bounds_changed()

    def on_mouse_press(self, event: MouseEvent) -> None:
        if event.pos is None:
            return  # pragma: no cover
        # check whether the user grabbed a node
        self._grabbed = self._find_nearby_node(event)
        if self._grabbed != Grabbable.NONE:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.plot.camera.interactive = False

    def on_mouse_release(self, event: MouseEvent) -> None:
        self._grabbed = Grabbable.NONE
        self.plot.camera.interactive = True

    def on_mouse_move(self, event: MouseEvent) -> None:
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return  # pragma: no cover
        if self._clims is None:
            return  # pragma: no cover

        if self._grabbed in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            if self._vertical:
                c = self._to_plot_coords(event.pos)[1]
            else:
                c = self._to_plot_coords(event.pos)[0]
            if self._grabbed is Grabbable.LEFT_CLIM:
                newlims = (min(self._clims[1], c), self._clims[1])
            elif self._grabbed is Grabbable.RIGHT_CLIM:
                newlims = (self._clims[0], max(self._clims[0], c))
            self.climsChanged.emit(newlims)
            return
        elif self._grabbed is Grabbable.GAMMA:
            y0, y1 = (
                self.plot.xaxis.axis.domain
                if self._vertical
                else self.plot.yaxis.axis.domain
            )
            y = self._to_plot_coords(event.pos)[0 if self._vertical else 1]
            if y < np.maximum(y0, 0) or y > y1:
                return
            self.gammaChanged.emit(-np.log2(y / y1))
            return

        nearby = self._find_nearby_node(event)

        if nearby in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            if self._vertical:
                cursor_type = CursorType.V_ARROW
            else:
                cursor_type = CursorType.H_ARROW
        elif nearby is Grabbable.GAMMA:
            if self._vertical:
                cursor_type = CursorType.H_ARROW
            else:
                cursor_type = CursorType.V_ARROW
        else:
            x, y = self._to_plot_coords(event.pos)
            x1, x2 = self.plot.xaxis.axis.domain
            y1, y2 = self.plot.yaxis.axis.domain
            if (x1 < x <= x2) and (y1 <= y <= y2):
                cursor_type = CursorType.ALL_ARROW
            else:
                cursor_type = CursorType.DEFAULT
        self._cursor.set(cursor_type)

    def _find_nearby_node(self, event: MouseEvent, tolerance: int = 5) -> Grabbable:
        """Describes whether the event is near a clim."""
        click_x, click_y = event.pos

        # NB Computations are performed in canvas-space
        # for easier tolerance computation.
        plot_to_canvas = self.node_tform.imap
        gamma_to_plot = self._handle_transform.map

        if self._clims is not None:
            if self._vertical:
                click = click_y
                right = plot_to_canvas([0, self._clims[1]])[1]
                left = plot_to_canvas([0, self._clims[0]])[1]
            else:
                click = click_x
                right = plot_to_canvas([self._clims[1], 0])[0]
                left = plot_to_canvas([self._clims[0], 0])[0]

            # Right bound always selected on overlap
            if bool(abs(right - click) < tolerance):
                return Grabbable.RIGHT_CLIM
            if bool(abs(left - click) < tolerance):
                return Grabbable.LEFT_CLIM

        if self._gamma_handle_pos is not None:
            gx, gy = plot_to_canvas(gamma_to_plot(self._gamma_handle_pos[0]))[:2]
            if bool(abs(gx - click_x) < tolerance and abs(gy - click_y) < tolerance):
                return Grabbable.GAMMA

        return Grabbable.NONE

    def _to_plot_coords(self, pos: Sequence[float]) -> tuple[float, float]:
        """Return the plot coordinates of the given position."""
        x, y = self.node_tform.map(pos)[:2]
        return x, y

    def _resize(self) -> None:
        self.plot.camera.set_range(
            x=self._range if self._vertical else self._domain,
            y=self._domain if self._vertical else self._range,
            # FIXME: Bitten by https://github.com/vispy/vispy/issues/1483
            # It's pretty visible in logarithmic mode
            margin=1e-30,
        )
        if self._vertical:
            scale = 0.98 * self.plot.xaxis.axis.domain[1]
            self._handle_transform.scale = (scale, 1)
        else:
            scale = 0.98 * self.plot.yaxis.axis.domain[1]
            self._handle_transform.scale = (1, scale)


def _hist_counts_to_mesh(
    values: Sequence[float] | npt.NDArray,
    bin_edges: Sequence[float] | npt.NDArray,
    vertical: bool = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert histogram counts to mesh vertices and faces for plotting."""
    n_edges = len(bin_edges)
    X, Y = (1, 0) if vertical else (0, 1)

    #   4-5
    #   | |
    # 1-2/7-8
    # |/| | |
    # 0-3-6-9
    # construct vertices
    vertices = np.zeros((3 * n_edges - 2, 3), np.float32)
    vertices[:, X] = np.repeat(bin_edges, 3)[1:-1]
    vertices[1::3, Y] = values
    vertices[2::3, Y] = values
    vertices[vertices == float("-inf")] = 0

    # construct triangles
    faces = np.zeros((2 * n_edges - 2, 3), np.uint32)
    offsets = 3 * np.arange(n_edges - 1, dtype=np.uint32)[:, np.newaxis]
    faces[::2] = np.array([0, 2, 1]) + offsets
    faces[1::2] = np.array([2, 0, 3]) + offsets

    return vertices, faces
