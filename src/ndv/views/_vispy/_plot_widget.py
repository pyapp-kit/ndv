from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, cast

from vispy import scene

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeVar

    # just here cause vispy has poor type hints
    T = TypeVar("T")

    class Grid(scene.Grid, Generic[T]):
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

        def __getitem__(self, idxs: int | tuple[int, int]) -> T:
            return super().__getitem__(idxs)  # type: ignore [no-any-return]

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
        **widget_kwargs: Any,
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
        # TODO: dynamic max based on max tick value?
        self._grid_wdgs[Component.YAXIS].width_max = 40  # otherwise it takes too much
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
