# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypedDict,
    Unpack,
    cast,
)

import numpy as np
from vispy import scene

if TYPE_CHECKING:

    class Grid(scene.Grid):
        def add_view(
            self,
            row: int | None = None,
            col: int | None = None,
            row_span: int = 1,
            col_span: int = 1,
            **kwargs: Any,
        ) -> scene.ViewBox: ...

        def add_widget(
            self,
            widget: None | scene.Widget = None,
            row: int | None = None,
            col: int | None = None,
            row_span: int = 1,
            col_span: int = 1,
            **kwargs: Any,
        ) -> scene.Widget: ...


__all__ = ["PlotWidget"]


class WidgetKwargs(TypedDict, total=False):
    pos: tuple[float, float]
    size: tuple[float, float]
    border_color: str
    border_width: float
    bgcolor: str
    padding: float
    margin: float


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
        **widget_kwargs: Unpack[WidgetKwargs],
    ) -> None:
        self._fg_color = fg_color
        self.visuals: list[scene.VisualNode] = []
        super().__init__(**widget_kwargs)
        self.unfreeze()
        self.grid = cast("Grid", self.add_grid(spacing=0, margin=10))

        self.show_xaxis = True
        self.show_yaxis = True
        self.axis_kwargs = {
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
            "axis_font_size": 9,
        }
        self.title_kwargs = {"font_size": 16, "color": "#ff0000"}
        self.title = scene.Label(str(title), **self.title_kwargs)
        self.xlabel = scene.Label(str(xlabel))
        self.ylabel = scene.Label(str(ylabel), rotation=-90)

        # CONFIGURE 2D
        #         c0        c1      c2      c3       c4
        #     +---------+-------+-------+-------+---------+
        #  r0 |         |               | title |         |
        #     | ------- +-------+-------+-------+ ------- |
        #  r1 | padding | ylabel| yaxis | view  | padding |
        #     | ------- +-------+-------+-------+ ------- |
        #  r2 |         |               | xaxis |         |
        #     |         +---------------+-------+         |
        #  r3 |         |               | xlabel|         |
        #     |---------+---------------+-------+---------|
        #  r4 |                         |padding|         |
        #     +---------+---------------+-------+---------+
        #         c0        c1      c2      c3      c4      c5         c6
        #     +---------+-------+-------+-------+-------+---------+---------+
        #  r0 |         |                       | title |         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r1 |         |                       | cbar  |         |         |
        #     | ------- +-------+-------+-------+-------+---------+ ------- |
        #  r2 | padding | cbar  | ylabel| yaxis |  view | cbar    | padding |
        #     | ------- +-------+-------+-------+-------+---------+ ------- |
        #  r3 |         |                       | xaxis |         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r4 |         |                       | xlabel|         |         |
        #     |         +-----------------------+-------+---------+         |
        #  r5 |         |                       | cbar  |         |         |
        #     |---------+-----------------------+-------+---------+---------|
        #  r6 |                                 |padding|                   |
        #     +---------+-----------------------+-------+---------+---------+

        # PADDING
        self.padding_right = self.grid.add_widget(None, row=2, col=6)
        self.padding_right.width_min = 1
        self.padding_right.width_max = 5
        self.padding_bottom = self.grid.add_widget(None, row=6, col=4)
        self.padding_bottom.height_min = 1
        self.padding_bottom.height_max = 3

        # TITLE
        self.title_widget = self.grid.add_widget(self.title, row=0, col=4)
        self.title_widget.height_min = self.title_widget.height_max = (
            30 if self.title.text else 5
        )

        # COLORBARS
        self.cbar_top = self.grid.add_widget(None, row=1, col=4)
        self.cbar_top.height_max = 0
        self.cbar_left = self.grid.add_widget(None, row=2, col=1)
        self.cbar_left.width_max = 0
        self.cbar_right = self.grid.add_widget(None, row=2, col=5)
        self.cbar_right.width_max = 0
        self.cbar_bottom = self.grid.add_widget(None, row=5, col=4)
        self.cbar_bottom.height_max = 0

        # Y AXIS
        self.yaxis = scene.AxisWidget(orientation="left", **self.axis_kwargs)
        self.yaxis_widget = self.grid.add_widget(self.yaxis, row=2, col=3)
        if self.show_yaxis:
            self.yaxis_widget.width_max = 30
            self.ylabel_widget = self.grid.add_widget(self.ylabel, row=2, col=2)
            self.ylabel_widget.width_max = 10 if self.ylabel.text else 1
            self.padding_left = self.grid.add_widget(None, row=2, col=0)
            self.padding_left.width_min = 1
            self.padding_left.width_max = 10
        else:
            self.yaxis.visible = False
            self.yaxis.width_max = 1
            self.padding_left = self.grid.add_widget(None, row=2, col=0, col_span=3)
            self.padding_left.width_min = 1
            self.padding_left.width_max = 5

        # X AXIS
        self.xaxis = scene.AxisWidget(orientation="bottom", **self.axis_kwargs)
        self.xaxis_widget = self.grid.add_widget(self.xaxis, row=3, col=4)
        self.xaxis_widget.height_max = 20 if self.show_xaxis else 0
        self.xlabel_widget = self.grid.add_widget(self.xlabel, row=4, col=4)
        self.xlabel_widget.height_max = 10 if self.xlabel.text else 0

        # VIEWBOX (this has to go last, see vispy #1748)
        self._view = self.grid.add_view(row=2, col=4, border_color=None, bgcolor=None)
        self.camera = self._view.camera = PanZoom1DCamera(lock_axis)

        self.xaxis.link_view(self._view)
        self.yaxis.link_view(self._view)

        self.freeze()

    def lock_axis(self, axis: Literal["x", "y", None]) -> None:
        self.camera._axis = axis
        self.camera.set_range()

    def histogram(
        self,
        data: np.ndarray,
        bins: int | np.ndarray = 10,
        color: str = "w",
        orientation: Literal["h", "w"] = "h",
    ) -> scene.Histogram:
        """Calculate and show a histogram of data.

        Parameters
        ----------
        data : array-like
            Data to histogram. Currently only 1D data is supported.
        bins : int | array-like
            Number of bins, or bin edges.
        color : instance of Color
            Color of the histogram.
        orientation : {'h', 'v'}
            Orientation of the histogram.

        Returns
        -------
        hist : instance of Polygon
            The histogram polygon.
        """
        hist = scene.Histogram(data, bins, color, orientation)
        self._view.add(hist)
        self.camera.set_range()
        return hist

    # def plot(
    #     self,
    #     data,
    #     color="k",
    #     symbol=None,
    #     line_kind="-",
    #     width=1.0,
    #     marker_size=10.0,
    #     edge_color="k",
    #     face_color="b",
    #     edge_width=1.0,
    #     title=None,
    #     xlabel=None,
    #     ylabel=None,
    #     connect="strip",
    # ):
    #     """Plot a series of data using lines and markers.

    #     Parameters
    #     ----------
    #     data : array | two arrays
    #         Arguments can be passed as ``(Y,)``, ``(X, Y)`` or
    #         ``np.array((X, Y))``.
    #     color : instance of Color
    #         Color of the line.
    #     symbol : str
    #         Marker symbol to use.
    #     line_kind : str
    #         Kind of line to draw. For now, only solid lines (``'-'``)
    #         are supported.
    #     width : float
    #         Line width.
    #     marker_size : float
    #         Marker size. If `size == 0` markers will not be shown.
    #     edge_color : instance of Color
    #         Color of the marker edge.
    #     face_color : instance of Color
    #         Color of the marker face.
    #     edge_width : float
    #         Edge width of the marker.
    #     title : str | None
    #         The title string to be displayed above the plot
    #     xlabel : str | None
    #         The label to display along the bottom axis
    #     ylabel : str | None
    #         The label to display along the left axis.
    #     connect : str | array
    #         Determines which vertices are connected by lines.

    #     Returns
    #     -------
    #     line : instance of LinePlot
    #         The line plot.

    #     See Also
    #     --------
    #     LinePlot
    #     """
    #     line = scene.LinePlot(
    #         data,
    #         connect=connect,
    #         color=color,
    #         symbol=symbol,
    #         line_kind=line_kind,
    #         width=width,
    #         marker_size=marker_size,
    #         edge_color=edge_color,
    #         face_color=face_color,
    #         edge_width=edge_width,
    #     )
    #     self._view.add(line)
    #     self._view.camera.set_range()
    #     self.visuals.append(line)

    #     if title is not None:
    #         self.title.text = title
    #     if xlabel is not None:
    #         self.xlabel.text = xlabel
    #     if ylabel is not None:
    #         self.ylabel.text = ylabel

    #     return line


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
        center: Optional[tuple[float, ...]] = None,
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
