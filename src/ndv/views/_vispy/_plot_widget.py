from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, cast

import numpy as np
from vispy import geometry, scene
from vispy.visuals.axis import Ticker

if TYPE_CHECKING:
    from typing import TypeVar

    from vispy.scene.events import SceneMouseEvent

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
            return super().add_view(...)  # pyright: ignore[reportReturnType]

        def add_widget(
            self,
            widget: None | scene.Widget = None,
            row: int | None = None,
            col: int | None = None,
            row_span: int = 1,
            col_span: int = 1,
            **kwargs: Any,
        ) -> scene.Widget:
            return super().add_widget(...)

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


__all__ = ["LogTicker", "PlotWidget"]

# Quantized width steps for axis sizing (only grows, snaps to these values)
_AXIS_WIDTH_STEPS = (24, 34, 44, 54)


class LogTicker(Ticker):
    """Ticker that displays log-scale labels on a log-transformed axis.

    The axis domain is assumed to be in log-transformed space (i.e., values are
    already log(count+1)/log(base)). This ticker places major ticks at positions
    corresponding to powers of 10 in the original count space, and labels them
    with the original counts (e.g., 1, 10, 100, 1000).
    """

    def __init__(self, axis: Any, base: float = 2, anchors: Any = None) -> None:
        super().__init__(axis, anchors=anchors)
        self._log_base = base

    def _get_tick_frac_labels(self) -> tuple[Any, Any, list[str]]:
        domain = self.axis.domain
        if domain[1] < domain[0]:
            flip = True
            domain = domain[::-1]
        else:
            flip = False

        d_min, d_max = domain
        scale = d_max - d_min
        if scale == 0:
            return np.array([]), np.array([]), []

        # Convert domain back to original counts
        # domain is in log_base space: val = log(count+1)/log(base)
        # so count = base^val - 1
        log_b = np.log(self._log_base)
        count_min = self._log_base**d_min - 1
        count_max = self._log_base**d_max - 1

        # Generate major ticks at powers of 10
        if count_max <= 0:
            return np.array([]), np.array([]), []

        min_exp = int(np.floor(np.log10(max(count_min, 1))))
        max_exp = int(np.ceil(np.log10(max(count_max, 1))))
        # Always include 0
        major_counts = [0.0]
        for exp in range(min_exp, max_exp + 1):
            val = 10.0**exp
            if val > count_min and val <= count_max * 1.01:
                major_counts.append(val)

        major_counts_arr = np.array(major_counts)
        # Convert counts to log-transformed positions
        major_pos = np.log(major_counts_arr + 1) / log_b
        # Normalize to fractions
        major_frac = (major_pos - d_min) / scale

        labels = [f"{c:g}" for c in major_counts_arr]

        # Minor ticks: place at 2, 3, ..., 9 within each decade
        minor_list: list[float] = []
        for exp in range(min_exp, max_exp + 1):
            for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                val = mult * 10.0**exp
                if count_min < val <= count_max:
                    pos = np.log(val + 1) / log_b
                    frac = (pos - d_min) / scale
                    minor_list.append(frac)
        minor_frac = np.array(minor_list) if minor_list else np.array([])

        # Filter to visible range
        mask = (major_frac > -0.0001) & (major_frac < 1.0001)
        major_frac = major_frac[mask]
        labels = [lb for i, lb in enumerate(labels) if mask[i]]
        if len(minor_frac) > 0:
            minor_frac = minor_frac[(minor_frac > -0.0001) & (minor_frac < 1.0001)]

        if flip:
            major_frac = 1 - major_frac
            minor_frac = 1 - minor_frac

        return major_frac, minor_frac, labels


DEFAULT_AXIS_KWARGS: AxisWidgetKwargs = {
    "text_color": "w",
    "axis_color": "w",
    "tick_color": "w",
    "tick_width": 1,
    "tick_font_size": 6,
    "tick_label_margin": 6,
    "axis_label_margin": 50,
    "minor_tick_length": 2,
    "major_tick_length": 4,
    "axis_width": 1,
    "axis_font_size": 8,
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
        self.grid = cast("Grid", self.add_grid(spacing=0, margin=0))

        title_kwargs: TextVisualKwargs = {"font_size": 14, "color": "w"}
        label_kwargs: TextVisualKwargs = {"font_size": 10, "color": "w"}
        self._title = scene.Label(str(title), **title_kwargs)
        self._xlabel = scene.Label(str(xlabel), **label_kwargs)
        self._ylabel = scene.Label(str(ylabel), rotation=-90, **label_kwargs)

        axis_kwargs: AxisWidgetKwargs = DEFAULT_AXIS_KWARGS
        self.yaxis = scene.AxisWidget(orientation="left", **axis_kwargs)
        self.xaxis = scene.AxisWidget(
            orientation="bottom", **{**axis_kwargs, "tick_label_margin": 12}
        )

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

        # NOTE: `width_max` and `height_max` of 2 is actually *less* visible
        # than 0 for some reason.  They should also be extracted into some sort
        # of `hide/show` logic for each component
        self._yaxis_width = _AXIS_WIDTH_STEPS[0]
        self._grid_wdgs[Component.YAXIS].width_max = self._yaxis_width
        self._grid_wdgs[Component.PAD_LEFT].width_max = 2
        self._grid_wdgs[Component.XAXIS].height_max = 14
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

    def update_yaxis_width(self, domain: tuple[float, float] | None = None) -> None:
        """Update y-axis width to fit tick labels."""
        if domain is None:
            domain = cast("tuple[float, float]", self.yaxis.axis.domain)
        # Estimate the widest tick label (ticks are integers on a histogram)
        max_val = round(max(abs(domain[0]), abs(domain[1])))
        label = str(max_val)
        # ~5px per character + padding for tick marks
        needed = len(label) * 5 + 10
        # Snap to the nearest quantized step
        for step in _AXIS_WIDTH_STEPS:
            if step >= needed:
                needed = step
                break
        else:
            needed = _AXIS_WIDTH_STEPS[-1]
        if needed != self._yaxis_width:
            self._yaxis_width = needed
            self._grid_wdgs[Component.YAXIS].width_max = needed

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
        # Domain bounds - user can specify min/max for both axes
        self.xbounds: tuple[float | None, float | None] = (None, None)
        self.ybounds: tuple[float | None, float | None] = (None, None)
        super().__init__(*args, **kwargs)

    @property
    def axis_index(self) -> Literal[0, 1, None]:
        """Return the index of the axis along which to pan and zoom."""
        if self._axis in ("x", 0):
            return 0
        elif self._axis in ("y", 1):
            return 1
        return None

    @scene.cameras.PanZoomCamera.rect.setter  # type: ignore[untyped-decorator]
    def rect(self, args: Any) -> None:
        """Setter for the camera rect."""
        # Convert 4-tuple (x, y, w, h) to Rect
        if isinstance(args, tuple):
            args = geometry.Rect(*args)
        if isinstance(args, geometry.Rect):
            # Note that this code preserves camera width so long as the
            # desired width is possible given the bounds. This is why
            # width clamping must come before the checks against each bound.

            # Constrain width and height within bounds
            if None not in self.xbounds:
                max_width = self.xbounds[1] - self.xbounds[0]  # type: ignore[operator]
                args.width = min(args.width, max_width)
            if None not in self.ybounds:
                max_height = self.ybounds[1] - self.ybounds[0]  # type: ignore[operator]
                args.height = min(args.height, max_height)

            # Constrain position+/-radius within bounds
            x, y = args.pos
            if self.xbounds[0] is not None:
                x = max(x, self.xbounds[0])
            if self.xbounds[1] is not None:
                x = min(x, self.xbounds[1] - args.width)
            if self.ybounds[0] is not None:
                y = max(y, self.ybounds[0])
            if self.ybounds[1] is not None:
                y = min(y, self.ybounds[1] - args.height)

            args.pos = (x, y)
        super(PanZoom1DCamera, type(self)).rect.fset(self, args)  # pyright: ignore[reportAttributeAccessIssue]

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

    def pan(self, *pan: float) -> None:
        """Pan the camera by `pan`."""
        if self.axis_index is None:
            super().pan(*pan)
            return
        _pan = list(np.ravel(pan))
        if self.axis_index < len(_pan):
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

    def viewbox_mouse_event(self, event: SceneMouseEvent) -> None:
        if event.type == "mouse_wheel":
            dx, dy = event.delta
            if abs(dx) > abs(dy):
                # Horizontal scroll -> pan
                pan_dist = 0.1 * self.rect.width
                self.pan(*[pan_dist if dx < 0 else -pan_dist, 0])
                event.handled = True
                return
            # Vertical scroll -> zoom anchored at the current minimum
            # (only scale the max end of the free axis)
            s = 1.1 ** (-dy)
            rect = self.rect
            if self._axis in ("y", 1):
                # Free axis is x: keep left, scale width
                new_w = rect.width * s
                self.rect = geometry.Rect(rect.left, rect.bottom, new_w, rect.height)
            else:
                # Free axis is y (or None): keep bottom, scale height
                new_h = rect.height * s
                self.rect = geometry.Rect(rect.left, rect.bottom, rect.width, new_h)
            event.handled = True
            return
        super().viewbox_mouse_event(event)
