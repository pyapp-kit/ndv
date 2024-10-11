# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict, Unpack, cast

import numpy as np
from vispy import scene

if TYPE_CHECKING:
    # just here cause vispy has poor type hints
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
        **widget_kwargs: Unpack[WidgetKwargs],
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
        self.camera.set_range()

    def histogram(
        self,
        data: np.ndarray,
        bins: int | np.ndarray = 10,
        color: str | tuple[float, ...] = "w",
        orientation: Literal["h", "w"] = "h",
    ) -> scene.Histogram:
        """Calculate and show a histogram of data."""
        # TODO: extract histogram calculation to a separate function
        # and make it possible to directly accept counts and bin_edges
        hist = scene.Histogram(data, bins, color, orientation)
        self._view.add(hist)
        self.camera.set_range()
        return hist

    def plot(
        self,
        data: np.ndarray,
        *,
        color: str = "k",
        symbol: str | None = None,
        line_kind: str = "-",
        width: float = 1.0,
        marker_size: float = 10.0,
        edge_color: str = "k",
        face_color: str = "b",
        edge_width: float = 1.0,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        connect: str | np.ndarray = "strip",
    ) -> scene.LinePlot:
        """Plot a series of data using lines and markers."""
        line = scene.LinePlot(
            data,
            connect=connect,
            color=color,
            symbol=symbol,
            line_kind=line_kind,
            width=width,
            marker_size=marker_size,
            edge_color=edge_color,
            face_color=face_color,
            edge_width=edge_width,
        )
        self._view.add(line)
        self.camera.set_range()
        self._visuals.append(line)

        if title is not None:
            self._title.text = title
        if xlabel is not None:
            self._xlabel.text = xlabel
        if ylabel is not None:
            self._ylabel.text = ylabel
        return line


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
