from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
from vispy import scene

from ndv._types import CursorType
from ndv.models._lut_model import ClimPolicy, ClimsManual
from ndv.views._app import filter_mouse_events
from ndv.views.bases import HistogramCanvas

from ._plot_widget import PlotWidget

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cmap
    import numpy.typing as npt

    from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

MIN_GAMMA: np.float64 = np.float64(1e-6)


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


class VispyHistogramCanvas(HistogramCanvas):
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

        self._canvas = scene.SceneCanvas()
        self._disconnect_mouse_events = filter_mouse_events(self._canvas.native, self)

        ## -- Visuals -- ##

        # NB We directly use scene.Mesh, instead of scene.Histogram,
        # so that we can control the calculation of the histogram ourselves
        self._hist_mesh = scene.Mesh(color="#888888")

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

        self.set_vertical(vertical)

    def refresh(self) -> None:
        self._canvas.update()

    def set_visible(self, visible: bool) -> None: ...

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    # ------------- LutView Protocol methods ------------- #

    def set_channel_name(self, name: str) -> None:
        # Nothing to do
        # TODO: maybe show text somewhere
        pass

    def set_channel_visible(self, visible: bool) -> None:
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

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        # Nothing to do (yet)
        pass

    # ------------- HistogramView Protocol methods ------------- #

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Set the histogram values and bin edges.

        These inputs follow the same format as the return value of numpy.histogram.
        """
        self._values, self._bin_edges = values, bin_edges
        self._update_histogram()

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0,
    ) -> None:
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
        self._vertical = vertical
        self._update_histogram()
        self.plot.lock_axis("x" if vertical else "y")
        # When vertical, smaller values should appear at the top of the canvas
        self.plot.camera.flip = [False, vertical, False]
        self._update_lut_lines()
        self._resize()

    def set_log_base(self, base: float | None) -> None:
        if base != self._log_base:
            self._log_base = base
            self._update_histogram()
            self._update_lut_lines()
            self._resize()

    def frontend_widget(self) -> Any:
        return self._canvas.native

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        raise NotImplementedError

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        raise NotImplementedError

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
        if self._log_base:
            #  Replace zero values with 1
            values = np.where(values == 0, 1, values)
            values = np.log(values) / np.log(self._log_base)

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

        self._gamma_handle_pos[:] = midpoint[0]
        self._gamma_handle.set_data(pos=self._gamma_handle_pos)

        # FIXME: These should be called internally upon set_data, right?
        # Looks like https://github.com/vispy/vispy/issues/1899
        self._lut_line._bounds_changed()
        for v in self._lut_line._subvisuals:
            v._bounds_changed()
        self._gamma_handle._bounds_changed()

    def get_cursor(self, event: MouseMoveEvent) -> CursorType:
        pos = (event.x, event.y)
        nearby = self._find_nearby_node(pos)

        if nearby in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            return CursorType.V_ARROW if self._vertical else CursorType.H_ARROW
        elif nearby is Grabbable.GAMMA:
            return CursorType.H_ARROW if self._vertical else CursorType.V_ARROW
        else:
            x, y = self._to_plot_coords(pos)
            x1, x2 = self.plot.xaxis.axis.domain
            y1, y2 = self.plot.yaxis.axis.domain
            if (x1 < x <= x2) and (y1 <= y <= y2):
                return CursorType.ALL_ARROW
            else:
                return CursorType.DEFAULT

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        pos = event.x, event.y
        # check whether the user grabbed a node
        self._grabbed = self._find_nearby_node(pos)
        if self._grabbed != Grabbable.NONE:
            # disconnect the pan/zoom mouse events until handle is dropped
            self.plot.camera.interactive = False
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        self._grabbed = Grabbable.NONE
        self.plot.camera.interactive = True
        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        """Called whenever mouse moves over canvas."""
        pos = event.x, event.y
        if self._clims is None:
            return False  # pragma: no cover

        if self._grabbed in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            if self._vertical:
                c = self._to_plot_coords(pos)[1]
            else:
                c = self._to_plot_coords(pos)[0]
            if self._grabbed is Grabbable.LEFT_CLIM:
                newlims = (min(self._clims[1], c), self._clims[1])
            elif self._grabbed is Grabbable.RIGHT_CLIM:
                newlims = (self._clims[0], max(self._clims[0], c))
            if self.model:
                self.model.clims = ClimsManual(min=newlims[0], max=newlims[1])
            return False

        if self._grabbed is Grabbable.GAMMA:
            y0, y1 = (
                self.plot.xaxis.axis.domain
                if self._vertical
                else self.plot.yaxis.axis.domain
            )
            y = self._to_plot_coords(pos)[0 if self._vertical else 1]
            if y < np.maximum(y0, 0) or y > y1:
                return False
            if self.model:
                self.model.gamma = max(MIN_GAMMA, -np.log2(y / y1))
            return False

        self.get_cursor(event).apply_to(self)
        return False

    def _find_nearby_node(
        self, pos: tuple[float, float], tolerance: int = 5
    ) -> Grabbable:
        """Describes whether the event is near a clim."""
        click_x, click_y = pos

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

    def setVisible(self, visible: bool) -> None: ...


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
