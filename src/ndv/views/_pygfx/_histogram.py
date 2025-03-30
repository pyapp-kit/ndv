from __future__ import annotations

import warnings
from contextlib import suppress
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx
import pylinalg as la

from ndv._types import CursorType, MouseMoveEvent, MousePressEvent, MouseReleaseEvent
from ndv.models._lut_model import ClimPolicy, ClimsManual
from ndv.views._app import filter_mouse_events
from ndv.views.bases import HistogramCanvas

from ._util import rendercanvas_class

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    import cmap
    import numpy.typing as npt
    from wgpu.gui.jupyter import JupyterWgpuCanvas
    from wgpu.gui.qt import QWgpuCanvas

    WgpuCanvas: TypeAlias = QWgpuCanvas | JupyterWgpuCanvas


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


class PyGFXHistogramCanvas(HistogramCanvas):
    """A HistogramCanvas utilizing VisPy."""

    def __init__(self, *, vertical: bool = False) -> None:
        # ------------ data and state ------------ #

        self._values: np.ndarray | None = None
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
        # Canvas Margins, in pixels (around the data)
        # TODO: Computation might better support different displays
        self.margin_left = 50  # Provide room for y-axis ticks
        self.margin_bottom = 20  # Provide room for x-axis ticks
        self.margin_right = 10
        self.margin_top = 15

        # ------------ PyGFX Canvas ------------ #
        cls = rendercanvas_class()
        self._size = (600, 600)
        self._canvas = cls(size=self._size)

        # this filter needs to remain in scope for the lifetime of the canvas
        # or mouse events will not be intercepted
        # the returned function can be called to remove the filter, (and it also
        # closes on the event filter and keeps it in scope).
        self._disconnect_mouse_events = filter_mouse_events(self._canvas, self)

        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)

        try:
            # requires https://github.com/pygfx/pygfx/pull/752
            self._renderer.blend_mode = "additive"
        except ValueError:
            warnings.warn(
                "This version of pygfx does not yet support additive blending.",
                stacklevel=3,
            )
            self._renderer.blend_mode = "weighted_depth"

        self._scene = pygfx.Scene()
        self._viewport = pygfx.Viewport(self._renderer)
        self._camera: pygfx.OrthographicCamera = pygfx.OrthographicCamera()
        self._camera.maintain_aspect = False

        self._histogram = pygfx.Mesh(
            geometry=pygfx.Geometry(
                # NB placeholder arrays
                positions=np.zeros((1, 3), dtype=np.float32),
                indices=np.zeros((1, 3), dtype=np.uint16),
            ),
            material=pygfx.MeshBasicMaterial(color=(1, 1, 1, 1)),
        )
        self._scene.add(self._histogram)

        clim_npoints = 256
        self._clim_handles = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=self._generate_clim_positions(clim_npoints),
                colors=self._generate_clim_colors(clim_npoints),
            ),
            material=pygfx.LineMaterial(
                color=(1.0, 0.0, 0.0),
                color_mode="vertex",
            ),
        )
        self._scene.add(self._clim_handles)

        self._x = pygfx.Ruler(
            start_pos=(0, 0, 0),
            end_pos=(1, 0, 0),
            start_value=0,
            tick_format="",  # Avoid scientific notation
            tick_side="right",
        )
        self._y = pygfx.Ruler(
            start_pos=(0, 0, 0),
            end_pos=(0, 1, 0),
            start_value=0,
            tick_side="left",
        )
        self._scene.add(self._x, self._y)

        # TODO: Re-implement pan/zoom?
        controller = pygfx.PanZoomController(register_events=self._viewport)
        controller.add_camera(
            self._camera,
            include_state={"x", "width"},
        )
        self._controller = controller
        # increase zoom wheel gain
        self._controller.controls.update({"wheel": ("zoom_to_point", "push", -0.005)})

        self.refresh()

    def refresh(self) -> None:
        with suppress(AttributeError):
            self._canvas.update()
        self._canvas.request_draw(self._animate)

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    def _resize(self) -> None:
        # Construct the bounding box
        bb = np.zeros([2, 2])
        # Priority is given to user range specifications
        # If the user does not specify the data display range,
        # display the extent of the data if it exists
        if self._domain:
            # User-specified
            bb[:, 0] = self._domain
        elif self._bin_edges is not None:
            # Data-specified
            bb[:, 0] = (self._bin_edges[0], self._bin_edges[-1])
        else:
            # Default
            bb[:, 0] = (0, 1)
        if self._range:
            # User-specified
            bb[:, 1] = self._range
        else:
            # Data-specified/default
            bb[:, 1] = (0, self._clim_handles.local.scale_y)

        # Update camera
        # NB this is tricky because the axis tick sizes are specified in pixels.
        px_width, px_height = self._viewport.logical_size
        world_width, world_height = np.ptp(bb, axis=0)
        # 2D Plot layout:
        #
        #         c0                c1               c2
        #     +-------------+-----------------+---------------+
        #  r0 |             | margin_top      |               |
        #     |-------------+-----------------+---------------+
        #  r1 | margin_left | data            | margin_right  |
        #     |-------------+-----------------+---------------+
        #  r2 |             | margin_bottom   |               |
        #     |-------------+-----------------+---------------+
        #
        width_frac = px_width / (px_width - self.margin_left - self.margin_right)
        self._camera.width = world_width * width_frac

        height_frac = px_height / (px_height - self.margin_top - self.margin_bottom)
        self._camera.height = world_height * height_frac

        self._camera.local.position = [
            world_width * (1 - (px_width / 2 / (px_width - self.margin_left))),
            world_height * (1 - (px_height / 2 / (px_height - self.margin_bottom))),
            0,
        ]

        # Update rulers
        self._x.start_pos = [bb[0, 0], bb[0, 1], 0]
        self._x.end_pos = [bb[1, 0], bb[0, 1], 0]

        self._y.start_pos = [bb[0, 0], bb[0, 1], 0]
        self._y.end_pos = [bb[0, 0], bb[1, 1], 0]
        # TODO For short canvases, pygfx has a tough time assigning ticks.
        # For lack of a more thorough dive/fix, just mark the maximum of the histogram
        if self._values is not None:
            self._y.ticks = [int(self._values.max())]
        else:
            self._y.ticks = bb[1, 1]
        self._y.min_tick_distance = bb[1, 1]  # Prevents any other ticks

    def _animate(self) -> None:
        # Dynamically rescale the graph when canvas size changes
        rect = self._viewport.logical_size
        if rect != self._size:
            self._size = rect
            self._resize()

        self._x.update(self._camera, self._viewport.logical_size)
        self._y.update(self._camera, self._viewport.logical_size)

        # Update viewport
        self._viewport.render(self._scene, self._camera, flush=True)

    def set_visible(self, visible: bool) -> None: ...

    # ------------- LutView Protocol methods ------------- #

    def set_channel_name(self, name: str) -> None:
        # Nothing to do
        # TODO: maybe show text somewhere
        pass

    def set_channel_visible(self, visible: bool) -> None:
        self._clim_handles.visible = visible
        self.refresh()

    def set_colormap(self, lut: cmap.Colormap) -> None:
        self._histogram.material.color = lut.color_stops[-1].color.hex
        self.refresh()

    def set_gamma(self, gamma: float) -> None:
        # TODO: Support gamma
        if gamma != 1:
            raise NotImplementedError("Setting gamma in PyGFX not yet supported")

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._clims = clims
        # Move clims line via translate/scale
        # NB relies on position data lying within [0, 1]
        # Translate by minimum
        _, off_y, off_z = self._clim_handles.local.position
        self._clim_handles.local.position = clims[0], off_y, off_z
        # Scale by (maximum - minimum)
        diff = clims[1] - clims[0]
        diff = diff if abs(diff) > 1e-6 else 1e-6
        self._clim_handles.local.scale_x = diff

        # Redraw
        self.refresh()

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        # Nothing to do (yet)
        pass

    # ------------- HistogramView Protocol methods ------------- #

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Set the histogram values and bin edges.

        These inputs follow the same format as the return value of numpy.histogram.
        """
        # Convert values/bins to vertices/faces
        self._values, self._bin_edges = values, bin_edges
        verts, faces = _hist_counts_to_mesh(
            self._values, self._bin_edges, self._vertical
        )

        # Number of bins unchanged - reuse existing geometry for performance
        if (
            verts.shape == self._histogram.geometry.positions.data.shape
            and faces.shape == self._histogram.geometry.indices.data.shape
        ):
            self._histogram.geometry.positions.data[:, :] = verts
            self._histogram.geometry.positions.update_range()

            self._histogram.geometry.indices.data[:, :] = faces
            self._histogram.geometry.indices.update_range()
        # Number of bins changed - must create new geometry
        else:
            self._histogram.geometry = pygfx.Geometry(positions=verts, indices=faces)

        self._clim_handles.local.scale_y = values.max() / 0.98

        self._resize()
        self.refresh()

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0,
    ) -> None:
        """Update the range of the PanZoomCamera.

        When called with no arguments, the range is set to the full extent of the data.
        """
        if not self._scene.children or self._camera is None:
            return
        self._domain = x
        self._range = y
        if margin != 0:
            raise NotImplementedError("Nonzero margins not currently implemented")

        self._resize()
        self.refresh()
        return

    def set_vertical(self, vertical: bool) -> None:
        # TODO:
        raise NotImplementedError()

    def set_log_base(self, base: float | None) -> None:
        # TODO:
        raise NotImplementedError()

    def frontend_widget(self) -> Any:
        return self._canvas

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        raise NotImplementedError()

    # ------------- Private methods ------------- #

    def _generate_clim_positions(self, npoints: int = 256) -> np.ndarray:
        clims = [0, 1]

        # 2 additional points for each of the two vertical clims lines
        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)
        Z = np.zeros(npoints + 4)
        if self._vertical:
            # clims lines
            X[0:2], Y[0:2] = (1, 0.5), clims[0]
            X[-2:], Y[-2:] = (0.5, 0), clims[1]
            # gamma line
            X[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
            Y[2:-2] = np.linspace(clims[0], clims[1], npoints)
            np.array([(2**-self._gamma, np.mean(clims))])
        else:
            # clims lines
            X[0:2], Y[0:2] = clims[0], (1, 0.5)
            X[-2:], Y[-2:] = clims[1], (0.5, 0)
            # gamma line
            X[2:-2] = np.linspace(clims[0], clims[1], npoints)
            Y[2:-2] = np.linspace(0, 1, npoints) ** self._gamma
            np.array([(np.mean(clims), 2**-self._gamma)])

        return np.vstack((X, Y, Z)).astype(np.float32).transpose()

    def _generate_clim_colors(self, npoints: int) -> np.ndarray:
        # Gamma curve intensity between 0.2 and 0.8
        color = (
            np.linspace(0.2, 0.8, npoints + 4, dtype=np.float32)
            .repeat(4)
            .reshape(-1, 4)
        )
        # The entire line should be opaque
        color[:, 3] = 1
        # Clims intensity between 0.4 and 0.7
        c1, c2 = [0.4] * 3, [0.7] * 3
        color[0:3, :3] = [c1, c2, c1]
        color[-3:, :3] = [c1, c2, c1]

        return color

    def get_cursor(self, mme: MouseMoveEvent) -> CursorType:
        pos = (mme.x, mme.y)
        nearby = self._find_nearby_node(pos)

        if nearby in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            return CursorType.V_ARROW if self._vertical else CursorType.H_ARROW
        elif nearby is Grabbable.GAMMA:
            return CursorType.H_ARROW if self._vertical else CursorType.V_ARROW
        else:
            x, y = pos
            x_max, y_max = self._viewport.logical_size
            if (0 < x <= x_max) and (0 <= y <= y_max):
                return CursorType.ALL_ARROW
            else:
                return CursorType.DEFAULT

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        pos = event.x, event.y
        # check whether the user grabbed a node
        self._grabbed = self._find_nearby_node(pos)
        if self._grabbed != Grabbable.NONE:
            # disconnect pan/zoom events until handle is dropped
            self._controller.enabled = False
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        self._grabbed = Grabbable.NONE
        self._controller.enabled = True
        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        """Called whenever mouse moves over canvas."""
        pos = event.x, event.y
        if self._clims is None:
            return False  # pragma: no cover

        if self._grabbed in [Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM]:
            c = self.canvas_to_world(pos)[1 if self._vertical else 0]
            if self._grabbed is Grabbable.LEFT_CLIM:
                # The left clim must stay to the left of the right clim
                new_left = min(c, self._clims[1])
                # ...and no less than the minimum value
                if self._bin_edges is not None:
                    new_left = max(new_left, self._bin_edges[0])
                newlims = (new_left, self._clims[1])
            elif self._grabbed is Grabbable.RIGHT_CLIM:
                # The right clim must stay to the right of the left clim
                new_right = max(self._clims[0], c)
                # ...and no more than the minimum value
                if self._bin_edges is not None:
                    new_right = min(new_right, self._bin_edges[-1])
                newlims = (self._clims[0], new_right)
            if self.model:
                self.model.clims = ClimsManual(min=newlims[0], max=newlims[1])
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
        plot_to_canvas = self.world_to_canvas
        # gamma_to_plot = self._handle_transform.map

        if self._clims is not None:
            if self._vertical:
                click = click_y
                right = plot_to_canvas((0, self._clims[1], 0))[1]
                left = plot_to_canvas((0, self._clims[0], 0))[1]
            else:
                click = click_x
                right = plot_to_canvas((self._clims[1], 0, 0))[0]
                left = plot_to_canvas((self._clims[0], 0, 0))[0]

            # Right bound always selected on overlap
            if bool(abs(right - click) < tolerance):
                return Grabbable.RIGHT_CLIM
            if bool(abs(left - click) < tolerance):
                return Grabbable.LEFT_CLIM

        return Grabbable.NONE

    def world_to_canvas(
        self, pos_xyz: tuple[float, float, float]
    ) -> tuple[float, float]:
        """Map XYZ coordinate in world space to XY canvas position (pixels)."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR420
        screen_space = pygfx.utils.transform.AffineTransform()
        screen_space.position = (-1, 1, 0)
        x_d, y_d = self._viewport.logical_size
        screen_space.scale = (2 / x_d, -2 / y_d, 1)
        ndc_to_screen = screen_space.inverse_matrix
        canvas_pos = la.vec_transform(
            pos_xyz, ndc_to_screen @ self._camera.camera_matrix
        )
        return (canvas_pos[0], canvas_pos[1])

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        """Map XY canvas position (pixels) to XYZ coordinate in world space."""
        # Code adapted from:
        # https://github.com/pygfx/pygfx/pull/753/files#diff-173d643434d575e67f8c0a5bf2d7ea9791e6e03a4e7a64aa5fa2cf4172af05cdR395
        # Get position relative to viewport
        pos_rel = (
            pos_xy[0] - self._viewport.rect[0],
            pos_xy[1] - self._viewport.rect[1],
        )

        vs = self._viewport.logical_size

        # Convert position to NDC
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)

        if self._camera:
            pos_ndc += la.vec_transform(
                self._camera.world.position, self._camera.camera_matrix
            )
            pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)

            # NB In vispy, (0.5,0.5) is a center of an image pixel, while in pygfx
            # (0,0) is the center. We conform to vispy's standard.
            return (pos_world[0] + 0.5, pos_world[1] + 0.5, pos_world[2] + 0.5)
        else:
            return (-1, -1, -1)


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
