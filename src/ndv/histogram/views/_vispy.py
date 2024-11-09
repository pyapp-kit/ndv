# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
from qtpy.QtCore import Qt
from vispy import scene

from ndv.histogram.view import HistogramView


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


if TYPE_CHECKING:
    # just here cause vispy has poor type hints
    from collections.abc import Sequence

    import cmap

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

    from vispy.scene.events import SceneMouseEvent

__all__ = ["PlotWidget"]


class WidgetKwargs(TypedDict, total=False):
    pos: tuple[float, float]
    size: tuple[float, float]
    border_color: str
    border_width: float
    bgcolor: str
    padding: float
    margin: float


class LabelKwargs(TypedDict, total=False):
    text: str
    color: str
    bold: bool
    italic: bool
    face: str
    font_size: float
    rotation: float
    method: str
    depth_test: bool
    pos: tuple[float, float]


class PlotWidget(scene.Widget):
    """Widget to facilitate plotting.

    Parameters
    ----------
    *args : arguments
        Arguments passed to the `ViewBox` super class.
    **kwargs : keywoard arguments
        Keyword arguments passed to the `ViewBox` super class.

    Notes
    -----
    This class is typically instantiated implicitly by a `Figure`
    instance, e.g., by doing ``fig[0, 0]``.
    """

    def __init__(
        self,
        fg_color: str = "k",
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        lock_axis: Literal["x", "y", None] = None,
        **widget_kwargs: Any,
    ) -> None:
        self._fg_color = fg_color
        self._visuals: list[scene.VisualNode] = []
        super().__init__(**widget_kwargs)
        self.unfreeze()
        self.grid = cast("Grid", self.add_grid(spacing=0, margin=10))

        title_kwargs = {"font_size": 14, "color": "w"}
        label_kwargs = {"font_size": 10, "color": "w"}
        self._title = scene.Label(str(title), **title_kwargs)
        self._xlabel = scene.Label(str(xlabel), **label_kwargs)
        self._ylabel = scene.Label(str(ylabel), rotation=-90, **label_kwargs)

        axis_kwargs = {
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

        self._grid_wdg: dict[str, scene.Widget] = {}
        for name, row, col, widget in [
            ("pad_left", 2, 0, None),
            ("pad_right", 2, 6, None),
            ("pad_bottom", 6, 4, None),
            ("title", 0, 4, self._title),
            ("cbar_top", 1, 4, None),
            ("cbar_left", 2, 1, None),
            ("cbar_right", 2, 5, None),
            ("cbar_bottom", 5, 4, None),
            ("yaxis", 2, 3, self.yaxis),
            ("xaxis", 3, 4, self.xaxis),
            ("xlabel", 4, 4, self._xlabel),
            ("ylabel", 2, 2, self._ylabel),
        ]:
            self._grid_wdg[name] = wdg = self.grid.add_widget(widget, row=row, col=col)
            if name.startswith(("cbar", "pad")):
                if "left" in name or "right" in name:
                    wdg.width_max = 2
                else:
                    wdg.height_max = 2

        # The main view into which plots are added
        self._view = self.grid.add_view(row=2, col=4)

        # FIXME: this is a mess of hardcoded values... not sure whether they will work
        # cross-platform.  Note that `width_max` and `height_max` of 2 is actually
        # *less* visible than 0 for some reason.  They should also be extracted into
        # some sort of `hide/show` logic for each component
        self._grid_wdg["yaxis"].width_max = 30
        self._grid_wdg["pad_left"].width_max = 20
        self._grid_wdg["xaxis"].height_max = 20
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
        wdg = self._grid_wdg["title"]
        wdg.height_min = wdg.height_max = 30 if text else 2

    @property
    def xlabel(self) -> str:
        """The x-axis label."""
        return self._xlabel.text  # type: ignore [no-any-return]

    @xlabel.setter
    def xlabel(self, text: str) -> None:
        """Set the x-axis label."""
        self._xlabel.text = text
        wdg = self._grid_wdg["xlabel"]
        wdg.height_min = wdg.height_max = 40 if text else 2

    @property
    def ylabel(self) -> str:
        """The y-axis label."""
        return self._ylabel.text  # type: ignore [no-any-return]

    @ylabel.setter
    def ylabel(self, text: str) -> None:
        """Set the x-axis label."""
        self._ylabel.text = text
        wdg = self._grid_wdg["ylabel"]
        wdg.width_min = wdg.width_max = 20 if text else 2

    def lock_axis(self, axis: Literal["x", "y", None]) -> None:
        """Prevent panning and zooming along a particular axis."""
        self.camera._axis = axis
        # self.camera.set_range()


class PanZoom1DCamera(scene.cameras.PanZoomCamera):
    def __init__(
        self, axis: Literal["x", "y", None] = None, *args: Any, **kwargs: Any
    ) -> None:
        self._axis: Literal["x", "y", None] = axis
        super().__init__(*args, **kwargs)

    @property
    def axis_index(self) -> Literal[0, 1, None]:
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
        if self.axis_index is None:
            super().zoom(factor, center=center)
            return

        if isinstance(factor, float):
            factor = (factor, factor)
        _factor = list(factor)
        _factor[self.axis_index] = 1
        super().zoom(_factor, center=center)

    def pan(self, pan: Sequence[float]) -> None:
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
        margin: float = 0,  # different default from super()
    ) -> None:
        super().set_range(x, y, z, margin)


class VispyHistogramView(HistogramView):
    """A HistogramView on a VisPy SceneCanvas."""

    def __init__(self) -> None:
        ## -- Canvas -- ##

        self._canvas = scene.SceneCanvas()
        self._canvas.unfreeze()
        self._canvas.on_mouse_press = self.on_mouse_press
        self._canvas.on_mouse_move = self.on_mouse_move
        self._canvas.on_mouse_release = self.on_mouse_release
        self._canvas.freeze()

        ## -- Visuals -- ##

        # NB We use a Mesh here, instead of a histogram,
        # because it gives us flexibility in computing the histogram
        self._hist = scene.Mesh(color="red")
        self._values: Sequence[float] | None = None
        self._bin_edges: Sequence[float] | None = None

        # The Lut Line visualizes both the clims (line segments connecting the
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

        self._clims: tuple[float, float] | None = None

        # The gamma handle appears halfway between the clims
        self._gamma: float = 1
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
        self._grabbed: Grabbable = Grabbable.NONE
        self._log_y: bool = False
        self._vertical: bool = False
        # The values of the left and right edges on the canvas (respectively)
        self._domain: tuple[float, float] | None = None
        # The values of the bottom and top edges on the canvas (respectively)
        self._range: tuple[float, float] | None = None

        self.plot._view.add(self._hist)
        self.plot._view.add(self._lut_line)
        self.plot._view.add(self._gamma_handle)

    # -- Protocol methods -- #

    def set_histogram(
        self, values: Sequence[float], bin_edges: Sequence[float]
    ) -> None:
        """Calculate and show a histogram of data."""
        self._values = values
        self._bin_edges = bin_edges
        self._update_histogram()
        if self._clims is None:
            self.set_clims((self._bin_edges[0], self._bin_edges[-1]))
            self._resize()

    def set_std_dev(self, std_dev: float) -> None:
        # Nothing to do
        pass

    def set_average(self, average: float) -> None:
        # Nothing to do
        pass

    def view(self) -> Any:
        return self._canvas.native

    def set_visibility(self, visible: bool) -> None:
        if self._hist is None:
            return
        self._hist.visible = visible
        self._lut_line.visible = visible
        self._gamma_handle.visible = visible

    def set_cmap(self, lut: cmap.Colormap) -> None:
        if self._hist is not None:
            self._hist.color = lut.color_stops[-1].color.hex

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

    def set_autoscale(self, autoscale: bool | tuple[float, float]) -> None:
        # Nothing to do (yet)
        pass

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
        self._update_lut_lines()
        self._resize()

    def set_range_log(self, enabled: bool) -> None:
        if enabled != self._log_y:
            self._log_y = enabled
            self._update_histogram()
            self._update_lut_lines()
            self._resize()

    # -- Helper Methods -- #

    def _update_histogram(self) -> scene.Mesh:
        """
        Updates the displayed histogram with current View parameters.

        NB: Much of this code is graciously borrowed from:

        https://github.com/vispy/vispy/blob/af847424425d4ce51f144a4d1c75ab4033fe39be/vispy/visuals/histogram.py#L28
        """
        if self._values is None or self._bin_edges is None:
            return
        v = self._values
        # FIXME: Warnings about divide by zero from this.
        values = np.log10(v) if self._log_y else v
        bin_edges = self._bin_edges
        #   4-5
        #   | |
        # 1-2/7-8
        # |/| | |
        # 0-3-6-9
        X, Y = (1, 0) if self._vertical else (0, 1)
        # construct our vertices
        rr = np.zeros((3 * len(bin_edges) - 2, 3), np.float32)
        rr[:, X] = np.repeat(bin_edges, 3)[1:-1]
        rr[1::3, Y] = values
        rr[2::3, Y] = values
        rr[rr == float("-inf")] = 0
        # and now our tris
        tris = np.zeros((2 * len(bin_edges) - 2, 3), np.uint32)
        offsets = 3 * np.arange(len(bin_edges) - 1, dtype=np.uint32)[:, np.newaxis]
        tri_1 = np.array([0, 2, 1])
        tri_2 = np.array([2, 0, 3])
        tris[::2] = tri_1 + offsets
        tris[1::2] = tri_2 + offsets

        self._hist.set_data(vertices=rr, faces=tris)
        # FIXME: This should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._hist._bounds_changed()

    def _update_lut_lines(self, npoints: int = 256) -> None:
        if self._clims is None or self._gamma is None:
            return

        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)
        if self._vertical:
            # clims lines
            X[0:2], Y[0:2] = (1, 1 / 2), self._clims[0]
            X[-2:], Y[-2:] = (1 / 2, 0), self._clims[1]
            # gamma line
            X[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
            Y[2:-2] = np.linspace(self._clims[0], self._clims[1], npoints)
            midpoint = np.array([(2**-self._gamma, np.mean(self._clims))])
        else:
            # clims lines
            X[0:2], Y[0:2] = self._clims[0], (1, 1 / 2)
            X[-2:], Y[-2:] = self._clims[1], (1 / 2, 0)
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

    def on_mouse_press(self, event: SceneMouseEvent) -> None:
        if event.pos is None:
            return
        # check whether the user grabbed a node
        self._grabbed = self._find_nearby_node(event)
        if self._grabbed != Grabbable.NONE:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.plot.camera.interactive = False

    def on_mouse_release(self, event: SceneMouseEvent) -> None:
        self._grabbed = Grabbable.NONE
        self.plot.camera.interactive = True

    def _find_nearby_node(
        self, event: SceneMouseEvent, tolerance: int = 3
    ) -> Grabbable:
        """Describes whether the event is near a clim."""
        x, y = self._to_plot_coords(event.pos)

        if self._clims is not None:
            left, right = self._clims
            # Right bound always selected on overlap
            if bool(abs(right - x) < tolerance):
                return Grabbable.RIGHT_CLIM
            if bool(abs(left - x) < tolerance):
                return Grabbable.LEFT_CLIM

        if self._gamma_handle_pos is not None:
            gx, gy, _, _ = self._handle_transform.map(self._gamma_handle_pos[0])
            if bool(abs(gx - x) < tolerance and abs(gy - y) < tolerance):
                return Grabbable.GAMMA

        return Grabbable.NONE

    def on_mouse_move(self, event: SceneMouseEvent) -> None:
        """Called whenever mouse moves over canvas."""
        if event.pos is None:
            return
        if self._clims is None:
            return

        if self._grabbed in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            newlims = list(self._clims)
            if self._vertical:
                c = self._to_plot_coords(event.pos)[1]
            else:
                c = self._to_plot_coords(event.pos)[0]
            if self._grabbed is Grabbable.LEFT_CLIM:
                newlims[0] = min(newlims[1], c)
            elif self._grabbed is Grabbable.RIGHT_CLIM:
                newlims[1] = max(newlims[0], c)
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

        self._canvas.native.unsetCursor()

        nearby = self._find_nearby_node(event)

        if nearby in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            if self._vertical:
                cursor = Qt.CursorShape.SplitVCursor
            else:
                cursor = Qt.CursorShape.SplitHCursor
            self._canvas.native.setCursor(cursor)
        elif nearby is Grabbable.GAMMA:
            if self._vertical:
                cursor = Qt.CursorShape.SplitHCursor
            else:
                cursor = Qt.CursorShape.SplitVCursor
            self._canvas.native.setCursor(cursor)
        else:
            x, y = self._to_plot_coords(event.pos)
            x1, x2 = self.plot.xaxis.axis.domain
            y1, y2 = self.plot.yaxis.axis.domain
            if (x1 < x <= x2) and (y1 <= y <= y2):
                self._canvas.native.setCursor(Qt.CursorShape.SizeAllCursor)

    def _to_plot_coords(self, pos: Sequence[float]) -> tuple[float, float]:
        x, y, _, _ = self.node_tform.map(pos)
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
            self._lut_line.transform.scale = (scale, 1)
        else:
            scale = 0.98 * self.plot.yaxis.axis.domain[1]
            self._lut_line.transform.scale = (1, scale)
