from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx
from fastplotlib import Figure
from fastplotlib.graphics import LineGraphic
from fastplotlib.graphics._base import Graphic
from fastplotlib.graphics.selectors import LinearRegionSelector
from pygfx.controllers import PanZoomController

from ndv.views.bases import HistogramCanvas

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap
    import numpy.typing as npt


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


class PanZoom1DController(PanZoomController):
    """A PanZoomController that locks one axis."""

    _zeros = np.zeros(3)

    def _update_pan(self, delta: tuple, *, vecx: Any, vecy: Any) -> None:
        super()._update_pan(delta, vecx=vecx, vecy=self._zeros)

    def _update_zoom(self, delta: tuple, *, vecx: Any, vecy: Any) -> None:
        super()._update_zoom(delta, vecx=vecx, vecy=self._zeros)


class HistogramGraphic(Graphic):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._group = pygfx.Group()
        self._set_world_object(self._group)
        self._vmin = -10
        self._vmax = 10
        # FIXME: Avoid hardcoded data
        # TODO: Can you change the size of a line?
        self._line = LineGraphic(np.zeros((131074, 3)))
        self._group.add(self._line.world_object)
        self._clims = LinearRegionSelector(
            selection=[-5, 5],
            limits=[0, 65536],
            center=0,
            size=0,
            axis="x",
            edge_thickness=8,
            resizable=True,
            parent=self,
        )
        # there will be a small difference with the histogram edges so this makes
        # them both line up exactly
        self._clims.selection = (
            self._vmin,
            self._vmax,
        )
        self._group.add(self._clims.world_object)

    def _create_line(self, n: int) -> None:
        if self._line:
            self._group.remove(self._line.world_object)

    def _fpl_add_plot_area_hook(self, plot_area) -> None:
        self._plot_area = plot_area
        self._clims._fpl_add_plot_area_hook(plot_area)
        self._line._fpl_add_plot_area_hook(plot_area)
        self._plot_area.auto_scale()
        self._plot_area.controller.enabled = True

    @property
    def vmin(self) -> float:
        return self._vmin

    @vmin.setter
    def vmin(self, value: float) -> None:
        # with pause_events(self.image_graphic, self._clims):
        # must use world coordinate values directly from selection()
        # otherwise the linear region bounds jump to the closest bin edges
        self._clims.selection = (
            value * self._scale_factor,
            self._clims.selection[1],
        )
        self.image_graphic.vmin = value
        self._vmin = value
        if self._colorbar is not None:
            self._colorbar.vmin = value
        vmin_str, vmax_str = self._get_vmin_vmax_str()
        self._text_vmin.offset = (-120, self._clims.selection[0], 0)
        self._text_vmin.text = vmin_str

    @property
    def vmax(self) -> float:
        return self._vmax

    @vmax.setter
    def vmax(self, value: float) -> None:
        # with pause_events(self.image_graphic, self._clims):
        # must use world coordinate values directly from selection()
        # otherwise the linear region bounds jump to the closest bin edges
        self._clims.selection = (
            self._clims.selection[0],
            value * self._scale_factor,
        )
        self.image_graphic.vmax = value
        self._vmax = value
        if self._colorbar is not None:
            self._colorbar.vmax = value
        vmin_str, vmax_str = self._get_vmin_vmax_str()
        self._text_vmax.offset = (-120, self._clims.selection[1], 0)
        self._text_vmax.text = vmax_str

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        self._line.data[:, 0] = np.repeat(bin_edges, 2)  # xs
        self._line.data[0, 1] = self._line.data[21, 1] = 0
        self._line.data[1:-1, 1] = np.repeat(values, 2)  # xs
        self._clims.limits = [bin_edges[0], bin_edges[-1]]
        height = values.max() * 0.98
        self._clims.fill.geometry = pygfx.box_geometry(1, height, 1)
        self._clims.edges[0].geometry.positions.data[:, 1] = [height / 2, -height / 2]
        self._clims.edges[1].geometry.positions.data[:, 1] = [height / 2, -height / 2]
        self._clims.offset = [0, height / 2, 0]


class PyGFXHistogramCanvas(HistogramCanvas):
    """A HistogramCanvas utilizing VisPy."""

    def __init__(self, *, vertical: bool = False) -> None:
        # ------------ data and state ------------ #

        self._values: Sequence[float] | np.ndarray | None = None
        self._bin_edges: Sequence[float] | np.ndarray | None = None
        self._clims: tuple[float, float] | None = None
        self._gamma: float = 1

        # the currently grabbed object
        self._grabbed: Grabbable = Grabbable.NONE
        # whether the y-axis is logarithmic
        self._log_base: float | None = None
        # whether the histogram is vertical
        self._vertical: bool = vertical
        # The values of the left and right edges on the canvas (respectively)
        self._domain: tuple[float, float] | None = None
        # The values of the bottom and top edges on the canvas (respectively)
        self._range: tuple[float, float] | None = None

        # ------------ VisPy Canvas ------------ #

        self._figure = Figure()
        self._figure[0, 0].camera.maintain_aspect = False
        self._figure[0, 0].controller = PanZoom1DController()
        self._figure[0, 0].toolbar = False

        # self._disconnect_mouse_events = filter_mouse_events(self._canvas.native, self)

        ## -- Visuals -- ##

        self._graphic = HistogramGraphic()
        self._figure[0, 0].add_graphic(self._graphic)

        self._graphic._clims.add_event_handler(self._on_view_clims_changed, "selection")

    def refresh(self) -> None:
        # with suppress(AttributeError):
        #     self._canvas.update()
        # self._canvas.request_draw(self._animate)
        pass

    def set_visible(self, visible: bool) -> None: ...

    # ------------- LutView Protocol methods ------------- #

    def set_channel_name(self, name: str) -> None:
        # Nothing to do
        # TODO: maybe show text somewhere
        pass

    def set_channel_visible(self, visible: bool) -> None:
        # TODO
        pass
        # self._lut_line.visible = visible
        # self._gamma_handle.visible = visible

    def set_colormap(self, lut: cmap.Colormap) -> None:
        # TODO
        return
        if self._hist_mesh is not None:
            self._hist_mesh.color = lut.color_stops[-1].color.hex

    def set_gamma(self, gamma: float) -> None:
        # TODO
        return
        if gamma < 0:
            raise ValueError("gamma must be non-negative!")
        self._gamma = gamma
        self._update_lut_lines()

    def set_clims(self, clims: tuple[float, float]) -> None:
        # TODO
        self._graphic._clims.selection = clims
        return

    def set_auto_scale(self, autoscale: bool) -> None:
        # Nothing to do (yet)
        pass

    # ------------- HistogramView Protocol methods ------------- #

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Set the histogram values and bin edges.

        These inputs follow the same format as the return value of numpy.histogram.
        """
        self._values, self._bin_edges = values, bin_edges
        self._graphic.set_data(values, bin_edges)
        self._figure[0, 0].auto_scale()

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0,
    ) -> None:
        # TODO:
        return
        if x:
            if x[0] > x[1]:
                x = (x[1], x[0])
        elif self._bin_edges is not None:
            x = self._bin_edges[0], self._bin_edges[-1]
        if y:
            if y[0] > y[1]:
                y = (y[1], y[0])
        elif self._values is not None:
            y = (0, np.max(self._values))
        self._range = y
        self._domain = x
        self._resize()

    def set_vertical(self, vertical: bool) -> None:
        # TODO:
        return
        self._vertical = vertical
        self._update_histogram()
        self.plot.lock_axis("x" if vertical else "y")
        # When vertical, smaller values should appear at the top of the canvas
        self.plot.camera.flip = [False, vertical, False]
        self._update_lut_lines()
        self._resize()

    def set_log_base(self, base: float | None) -> None:
        # TODO:
        return
        if base != self._log_base:
            self._log_base = base
            self._update_histogram()
            self._update_lut_lines()
            self._resize()

    def frontend_widget(self) -> Any:
        return self._figure.show()

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        raise NotImplementedError

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        raise NotImplementedError

    # ------------- Private methods ------------- #

    def _on_view_clims_changed(self, ev) -> None:
        selection = self._graphic._clims.selection
        self.climsChanged.emit(selection)

    # def _update_lut_lines(self, npoints: int = 256) -> None:
    #     if self._clims is None or self._gamma is None:
    #         return  # pragma: no cover

    #     # 2 additional points for each of the two vertical clims lines
    #     X = np.empty(npoints + 4)
    #     Y = np.empty(npoints + 4)
    #     if self._vertical:
    #         # clims lines
    #         X[0:2], Y[0:2] = (1, 0.5), self._clims[0]
    #         X[-2:], Y[-2:] = (0.5, 0), self._clims[1]
    #         # gamma line
    #         X[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
    #         Y[2:-2] = np.linspace(self._clims[0], self._clims[1], npoints)
    #         midpoint = np.array([(2**-self._gamma, np.mean(self._clims))])
    #     else:
    #         # clims lines
    #         X[0:2], Y[0:2] = self._clims[0], (1, 0.5)
    #         X[-2:], Y[-2:] = self._clims[1], (0.5, 0)
    #         # gamma line
    #         X[2:-2] = np.linspace(self._clims[0], self._clims[1], npoints)
    #         Y[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
    #         midpoint = np.array([(np.mean(self._clims), 2**-self._gamma)])

    #     # TODO: Move to self.edit_cmap
    #     color = np.linspace(0.2, 0.8, npoints + 4).repeat(4).reshape(-1, 4)
    #     c1, c2 = [0.4] * 4, [0.7] * 4
    #     color[0:3] = [c1, c2, c1]
    #     color[-3:] = [c1, c2, c1]

    #     self._lut_line.set_data((X, Y), marker_size=0, color=color)

    #     self._gamma_handle_pos[:] = midpoint[0]
    #     self._gamma_handle.set_data(pos=self._gamma_handle_pos)

    #     # FIXME: These should be called internally upon set_data, right?
    #     # Looks like https://github.com/vispy/vispy/issues/1899
    #     self._lut_line._bounds_changed()
    #     for v in self._lut_line._subvisuals:
    #         v._bounds_changed()
    #     self._gamma_handle._bounds_changed()

    # def get_cursor(self, pos: tuple[float, float]) -> CursorType:
    #     nearby = self._find_nearby_node(pos)

    #     if nearby in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
    #         return CursorType.V_ARROW if self._vertical else CursorType.H_ARROW
    #     elif nearby is Grabbable.GAMMA:
    #         return CursorType.H_ARROW if self._vertical else CursorType.V_ARROW
    #     else:
    #         x, y = self._to_plot_coords(pos)
    #         x1, x2 = self.plot.xaxis.axis.domain
    #         y1, y2 = self.plot.yaxis.axis.domain
    #         if (x1 < x <= x2) and (y1 <= y <= y2):
    #             return CursorType.ALL_ARROW
    #         else:
    #             return CursorType.DEFAULT

    # def on_mouse_press(self, event: MousePressEvent) -> bool:
    #     pos = event.x, event.y
    #     # check whether the user grabbed a node
    #     self._grabbed = self._find_nearby_node(pos)
    #     if self._grabbed != Grabbable.NONE:
    #         # disconnect the pan/zoom mouse events until handle is dropped
    #         self.plot.camera.interactive = False
    #     return False

    # def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
    #     self._grabbed = Grabbable.NONE
    #     self.plot.camera.interactive = True
    #     return False

    # def on_mouse_move(self, event: MouseMoveEvent) -> bool:
    #     """Called whenever mouse moves over canvas."""
    #     pos = event.x, event.y
    #     if self._clims is None:
    #         return False  # pragma: no cover

    #     if self._grabbed in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
    #         if self._vertical:
    #             c = self._to_plot_coords(pos)[1]
    #         else:
    #             c = self._to_plot_coords(pos)[0]
    #         if self._grabbed is Grabbable.LEFT_CLIM:
    #             newlims = (min(self._clims[1], c), self._clims[1])
    #         elif self._grabbed is Grabbable.RIGHT_CLIM:
    #             newlims = (self._clims[0], max(self._clims[0], c))
    #         self.climsChanged.emit(newlims)
    #         return False

    #     if self._grabbed is Grabbable.GAMMA:
    #         y0, y1 = (
    #             self.plot.xaxis.axis.domain
    #             if self._vertical
    #             else self.plot.yaxis.axis.domain
    #         )
    #         y = self._to_plot_coords(pos)[0 if self._vertical else 1]
    #         if y < np.maximum(y0, 0) or y > y1:
    #             return False
    #         self.gammaChanged.emit(-np.log2(y / y1))
    #         return False

    #     self.get_cursor(pos).apply_to(self)
    #     return False

    # def _find_nearby_node(
    #     self, pos: tuple[float, float], tolerance: int = 5
    # ) -> Grabbable:
    #     """Describes whether the event is near a clim."""
    #     click_x, click_y = pos

    #     # NB Computations are performed in canvas-space
    #     # for easier tolerance computation.
    #     plot_to_canvas = self.node_tform.imap
    #     gamma_to_plot = self._handle_transform.map

    #     if self._clims is not None:
    #         if self._vertical:
    #             click = click_y
    #             right = plot_to_canvas([0, self._clims[1]])[1]
    #             left = plot_to_canvas([0, self._clims[0]])[1]
    #         else:
    #             click = click_x
    #             right = plot_to_canvas([self._clims[1], 0])[0]
    #             left = plot_to_canvas([self._clims[0], 0])[0]

    #         # Right bound always selected on overlap
    #         if bool(abs(right - click) < tolerance):
    #             return Grabbable.RIGHT_CLIM
    #         if bool(abs(left - click) < tolerance):
    #             return Grabbable.LEFT_CLIM

    #     if self._gamma_handle_pos is not None:
    #         gx, gy = plot_to_canvas(gamma_to_plot(self._gamma_handle_pos[0]))[:2]
    #         if bool(abs(gx - click_x) < tolerance and abs(gy - click_y) < tolerance):
    #             return Grabbable.GAMMA

    #     return Grabbable.NONE

    # def _to_plot_coords(self, pos: Sequence[float]) -> tuple[float, float]:
    #     """Return the plot coordinates of the given position."""
    #     x, y = self.node_tform.map(pos)[:2]
    #     return x, y

    # def _resize(self) -> None:
    #     self.plot.camera.set_range(
    #         x=self._range if self._vertical else self._domain,
    #         y=self._domain if self._vertical else self._range,
    #         # FIXME: Bitten by https://github.com/vispy/vispy/issues/1483
    #         # It's pretty visible in logarithmic mode
    #         margin=1e-30,
    #     )
    #     if self._vertical:
    #         scale = 0.98 * self.plot.xaxis.axis.domain[1]
    #         self._handle_transform.scale = (scale, 1)
    #     else:
    #         scale = 0.98 * self.plot.yaxis.axis.domain[1]
    #         self._handle_transform.scale = (1, scale)

    # def setVisible(self, visible: bool) -> None: ...


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
