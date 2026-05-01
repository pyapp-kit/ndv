from __future__ import annotations

from math import ceil, floor, log10
from typing import TYPE_CHECKING, Any

import cmap
import numpy as np
import numpy.typing as npt
import scenex as snx
from scenex.adaptors import get_adaptor_registry
from scenex.app import CursorType, events
from scenex.utils import projections

from ndv.models._lut_model import ClimsManual
from ndv.views.bases import LUTView

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndv.models._lut_model import ClimPolicy

Y_AXIS = 40  # width (pixels) reserved for y axis view
X_AXIS = 25  # height (pixels) reserved for x axis view


def _calc_hist_bins(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maxval = np.iinfo(data.dtype).max
    counts = np.bincount(data.flatten(), minlength=maxval + 1)
    bin_edges = np.arange(maxval + 2) - 0.5
    return counts, bin_edges


class Histogram(LUTView):
    def __init__(self) -> None:
        self._clims: tuple[float, float] = (0, 65535)
        self._gamma = 1.0
        self._grabbed: snx.Node | None = None
        self._initialized = False

        # State variables - will be used when creating objects
        self._containers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._values: np.ndarray | None = None
        self._bins: np.ndarray | None = None
        self._log_base: float | None = None
        self._max_bin: float | None = None

        # Create canvas early so it's available before set_data
        self.canvas = snx.Canvas()
        self.canvas.visible = True

        # Create views with empty scenes in constructor
        self.x_view = snx.View(
            scene=snx.Scene(),
            camera=snx.Camera(),
        )
        self.view = snx.View(
            scene=snx.Scene(name="main scene"),
            camera=snx.Camera(interactive=True),
        )
        self.y_view = snx.View(
            scene=snx.Scene(),
            camera=snx.Camera(),
        )

        # Layout (pixel-based)
        self.canvas.views.append(self.x_view)
        self.canvas.views.append(self.y_view)
        self.canvas.views.append(self.view)
        # Define barrier between x axis and main/y view
        self.x_view.layout.y_start = f"-{X_AXIS}px"
        self.y_view.layout.y_end = f"-{X_AXIS}px"
        self.view.layout.y_end = f"-{X_AXIS}px"
        # Define barrier between y axis and main view
        self.view.layout.x_start = f"{Y_AXIS}px"
        self.y_view.layout.x_end = f"{Y_AXIS}px"

        # Scene contents will be created on first set_data call
        # FIXME: We do this because there's a VisPy bug that causes a blank canvas when
        # there is a non-empty scene at first render.
        # (RuntimeError: OpenGL got errors (periodic check): GL_INVALID_OPERATION)
        # The same thing actually happens

        self.x_axis: snx.Line | None = None
        self._tick_objects: list[snx.Text] = []
        self.y_axis: snx.Line | None = None
        self.y_max: snx.Text | None = None
        self.mesh: snx.Mesh | None = None
        self.highlight_line: snx.Line | None = None
        self.left_clim: snx.Line | None = None
        self.gamma_curve: snx.Line | None = None
        self.right_clim: snx.Line | None = None
        self.gamma_handle: snx.Points | None = None
        self.controls: snx.Scene | None = None

    def _initialize_views(self) -> None:
        """Lazy initialization of scene contents on first set_data call."""
        if self._initialized:
            return

        # 1. Populate x axis view scene
        self.x_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [1, 0, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("white")),
        )
        self.x_view.scene.add_child(self.x_axis)

        # Pre-create 10 tick objects with line children (enough for min, max, and ticks)
        for _ in range(10):
            tick_line = snx.Line(
                vertices=np.array([[0, 0, 0], [0, -0.1, 0]]),
                width=1,
                color=snx.UniformColor(color=cmap.Color("white")),
                transform=snx.Transform().translated((0, 0.4, 0)),
            )
            tick_text = snx.Text(text="0", children=[tick_line], antialias=True)
            self._tick_objects.append(tick_text)

        # 2. Populate y axis view scene
        self.y_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("white")),
        )
        self.y_max = snx.Text(
            text="1",
            transform=snx.Transform().translated((-0.5, 0.95)),
            antialias=True,
        )
        self.y_view.scene.add_child(self.y_axis)
        self.y_view.scene.add_child(self.y_max)

        # 3. Populate main histogram view scene
        self.mesh = snx.Mesh(
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.uint16),
            color=snx.UniformColor(color=cmap.Color("steelblue")),
        )

        self.highlight_line = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("yellow")),
            visible=False,  # Start hidden
        )

        # Split LUT line into three interactive components
        self.left_clim = snx.Line(
            name="left clim",
            interactive=True,
        )
        self.gamma_curve = snx.Line(
            name="gamma curve",
            interactive=False,
        )
        self.right_clim = snx.Line(
            name="right clim",
            interactive=True,
        )
        self.gamma_handle = snx.Points(
            name="gamma handle",
            vertices=np.array([[0.5, 0.5, 0]]),
            size=8,
            scaling="fixed",
            face_color=snx.UniformColor(color=cmap.Color("white")),
            edge_color=snx.UniformColor(color=cmap.Color("black")),
            interactive=True,
        )

        self._create_static_clim_lines()
        self._update_lut_line()

        self.controls = snx.Scene(
            name="controls scene",
            children=[
                self.left_clim,
                self.gamma_curve,
                self.right_clim,
                self.gamma_handle,
            ],
            interactive=True,
        )

        # Draw order (from bottom to top):
        # 0: histogram mesh
        self.mesh.order = 0
        self.view.scene.add_child(self.mesh)
        # 1: controls (clim lines, gamma curve, handle)
        self.controls.order = 1
        self.view.scene.add_child(self.controls)
        # 2: highlight line
        self.highlight_line.order = 2
        self.view.scene.add_child(self.highlight_line)

        # Set up event handlers and controllers
        self.view.camera.controller = snx.PanZoom(lock_y=True)

        self.view.camera.events.transform.connect(self._update_x_axis)
        self.view.camera.events.projection.connect(self._update_x_axis)
        self.canvas.events.width.connect(self._update_x_axis)
        self.view.set_event_filter(self._on_main_view)

        self._initialized = True

        self.synchronize()
        self.set_clims(self._clims)

    def _on_main_view(self, event: events.Event) -> bool:
        if not self._initialized:
            return False

        if isinstance(event, events.MousePressEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            intersections = [
                node
                for node, _dist in ray.intersections(self.controls)
                if node.interactive
            ]
            if len(intersections):
                self._grabbed = intersections[0]
                self.view.camera.interactive = False
        elif isinstance(event, events.MouseDoublePressEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            intersections = [
                node
                for node, _dist in ray.intersections(self.controls)
                if node.interactive
            ]
            if self.gamma_handle in intersections and (model := self.model):
                model.gamma = 1
        if isinstance(event, events.MouseMoveEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            if self._grabbed is self.left_clim:
                # The left clim must stay to the left of the right clim
                new_left = min(ray.origin[0], self._clims[1])
                # ...and no less than the minimum value
                if self._bins is not None:
                    new_left = max(new_left, self._bins[0])
                # Set it
                if model := self.model:
                    model.clims = ClimsManual(min=new_left, max=self._clims[1])
            elif self._grabbed is self.right_clim:
                # The right clim must stay to the right of the left clim
                new_right = max(self._clims[0], ray.origin[0])
                # ...and no more than the minimum value
                if self._bins is not None:
                    new_right = min(new_right, self._bins[-1])
                # Set it
                if model := self.model:
                    model.clims = ClimsManual(min=self._clims[0], max=new_right)
            elif self._grabbed is self.gamma_handle:
                # Set it
                if model := self.model:
                    model.gamma = -np.log2(ray.origin[1])
            elif self._grabbed is None:
                intersections = [
                    node
                    for node, _dist in ray.intersections(self.controls)
                    if node.interactive
                ]
                if self.right_clim in intersections or self.left_clim in intersections:
                    snx.set_cursor(self.canvas, CursorType.H_ARROW)
                elif self.gamma_handle in intersections:
                    snx.set_cursor(self.canvas, CursorType.V_ARROW)
                else:
                    snx.set_cursor(self.canvas, CursorType.DEFAULT)

        if isinstance(event, events.MouseReleaseEvent | events.MouseLeaveEvent):
            self._grabbed = None
            self.view.camera.interactive = True
        return False

    def _create_static_clim_lines(self) -> None:
        """Create the static left and right clim lines that don't change with gamma."""
        # Left clim line (vertical line)
        left_x = np.array([0, 0, 0])
        left_y = np.array([1, 0.5, 0])
        left_z = np.zeros(3)
        if line := self.left_clim:
            line.vertices = np.column_stack((left_x, left_y, left_z))

        # Right clim line (vertical line)
        right_x = np.array([1, 1, 1])
        right_y = np.array([1, 0.5, 0])
        right_z = np.zeros(3)
        if line := self.right_clim:
            line.vertices = np.column_stack((right_x, right_y, right_z))

        # Color the clim lines
        dark_clim_color = cmap.Color((0.4, 0.4, 0.4))
        light_clim_color = cmap.Color((0.7, 0.7, 0.7))
        if line := self.left_clim:
            line.color = snx.VertexColors(
                color=[dark_clim_color, light_clim_color, dark_clim_color],
            )
        if line := self.right_clim:
            line.color = snx.VertexColors(
                color=[dark_clim_color, light_clim_color, dark_clim_color],
            )

    def _update_lut_line(self) -> None:
        """Updates the gamma curve vertices and colors."""
        if self.gamma_curve is None or self.gamma_handle is None:
            return

        npoints = 256
        # Gamma curve (non-interactive) - updates when gamma changes
        gamma_x = np.linspace(0, 1, npoints)
        gamma_y = np.linspace(0, 1, npoints) ** (
            self.model.gamma if self.model is not None else 1
        )
        gamma_z = np.zeros(npoints)
        self.gamma_curve.vertices = np.column_stack((gamma_x, gamma_y, gamma_z))

        # Gamma curve gets gradient colors
        gamma_colors = [
            cmap.Color(c)
            for c in np.linspace(0.2, 0.8, npoints).repeat(3).reshape(-1, 3)
        ]
        self.gamma_curve.color = snx.VertexColors(color=gamma_colors)
        gamma = self.model.gamma if self.model is not None else 1
        self.gamma_handle.transform = snx.Transform().translated((0, 0.5**gamma - 0.5))

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        # Initialize views on first call
        self._initialize_views()

        uninitialized = self._values is None
        # Update the histogram mesh
        self._values = values
        self._bins = bin_edges

        self._max_bin = np.max(self._values)
        if mesh := self.mesh:
            mesh.vertices, mesh.faces = self._hist_counts_to_mesh(
                self._values, self._bins, False
            )
        # Reapply log scaling if necessary
        if log := self._log_base:
            self._log_base = None
            self.set_log_base(log)

        # Rescale the y axis
        self._update_y_axis()

        if uninitialized:
            self.set_range()

    def _has_data(self) -> bool:
        return self.mesh is not None and self.mesh.vertices.shape[0] > 1

    # ---- LutView interface implementations ----

    def set_channel_name(self, name: str) -> None:
        pass

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        pass

    def set_colormap(self, lut: cmap.Colormap) -> None:
        if self.mesh is not None:
            self.mesh.color = snx.UniformColor(color=lut.color_stops[-1].color)

    def set_clims(self, clims: tuple[float, float]) -> None:
        self._clims = clims
        if self.controls is not None:
            self.controls.transform = (
                snx.Transform()
                .scaled((self._clims[1] - self._clims[0], 1, 1))
                .translated((self._clims[0], 0, 0))
            )

    def set_clim_bounds(
        self, bounds: tuple[float | None, float | None] = (None, None)
    ) -> None:
        # TODO Implement
        pass

    def set_channel_visible(self, visible: bool) -> None:
        # TODO Implement
        pass

    def set_gamma(self, gamma: float) -> None:
        self._update_lut_line()

    def set_log_base(self, base: float | None) -> None:
        if self.mesh is None:
            return

        old_log, new_log = self._log_base, base
        verts = np.zeros_like(self.mesh.vertices)
        verts[:, :] = self.mesh.vertices[:, :]
        if old_log is not None:
            verts[:, 1] = np.power(old_log, verts[:, 1]) - 1
        # use a count+1 histogram to gracefully handle 0, 1
        self._log_base = base
        if new_log is not None:
            verts[:, 1] = np.log(verts[:, 1] + 1) / np.log(new_log)
        # FIXME: Just telling scenex to refresh would be great
        verts[:, 0] = self.mesh.vertices[:, 0]
        self.mesh.vertices = verts

        self._update_y_axis()

    # ---- Viewable interface implementations ----

    def set_range(self) -> None:
        if not self._initialized:
            return

        projections.zoom_to_fit(self.view, "orthographic", zoom_factor=1)
        self.x_view.camera.projection = projections.orthographic(1, 1, 1)
        self.y_view.camera.projection = projections.orthographic(1, 1, 1)
        self.x_view.camera.transform = snx.Transform().translated((0.5, -0.5, 0))
        self.y_view.camera.transform = snx.Transform().translated((-0.5, 0.5, 0))

    def set_visible(self, visible: bool) -> None:
        self.canvas.visible = visible

    def frontend_widget(self) -> Any:
        return get_adaptor_registry().get_adaptor(self.canvas)._snx_get_native()

    def close(self) -> None:
        # TODO Implement
        pass

    def highlight(self, value: float | None) -> None:
        """Highlight a specific value on the histogram."""
        if self.highlight_line is None:
            return
        self.highlight_line.visible = value is not None
        self.highlight_line.transform = (
            self.highlight_line.transform
            if value is None
            else snx.Transform().translated((value, 0, 0))
        )

    def _calculate_tick_step(
        self, min_val: float, max_val: float, target_ticks: int = 5
    ) -> float:
        """Calculate a nice tick step for the given range."""
        if max_val <= min_val:
            return 1.0

        range_val = max_val - min_val
        approx_step = range_val / target_ticks

        # Find a "nice" step size
        power10 = 10.0 ** floor(log10(approx_step))
        for multiplier in [1.0, 2.0, 2.5, 5.0, 10.0]:
            step = multiplier * power10
            if step >= approx_step:
                return step

        return power10

    def _get_tick_positions(
        self, min_val: float, max_val: float, step: float
    ) -> list[float]:
        """Get tick positions within range, including min/max and culling overlaps."""
        if step <= 0:
            return [min_val, max_val]

        # Calculate intermediate tick positions
        first_tick = ceil(min_val / step) * step
        last_tick = floor(max_val / step) * step

        intermediate_ticks: list[float] = []
        current = first_tick
        while current <= last_tick and len(intermediate_ticks) < 20:  # Safety limit
            intermediate_ticks.append(current)
            current += step

        # Filter out ticks too close to min/max to avoid overlap
        min_distance = step * 0.15
        filtered_ticks = [
            t
            for t in intermediate_ticks
            if abs(t - min_val) >= min_distance and abs(t - max_val) >= min_distance
        ]

        # Always include min and max, deduplicate while preserving order
        seen: set[float] = set()
        unique_ticks: list[float] = []
        for tick in [min_val, *filtered_ticks, max_val]:
            if tick not in seen:
                seen.add(tick)
                unique_ticks.append(tick)

        return unique_ticks

    def _clear_ticks(self) -> None:
        """Remove all existing tick marks and labels from the scene."""
        for tick_obj in self._tick_objects:
            if tick_obj in self.x_view.scene.children:
                self.x_view.scene.remove_child(tick_obj)

    def _update_x_axis(self) -> None:
        # Update the x-axis labels based on the current camera projection
        if not self._initialized:
            return

        cam = self.view.camera
        left, *_others = cam.transform.map(cam.projection.imap((-1, 0)))
        right, *_others = cam.transform.map(cam.projection.imap((1, 0)))

        # Clear existing ticks and labels
        self._clear_ticks()

        # Calculate tick positions (includes min/max and culling logic)
        tick_step = self._calculate_tick_step(left, right)
        unique_positions = self._get_tick_positions(left, right, tick_step)

        _x, _y, w, _h = self.canvas.rect_for(self.x_view)
        start = Y_AXIS / w

        # Use cached tick objects for all positions
        for tick_idx, tick_val in enumerate(unique_positions):
            if tick_idx >= len(self._tick_objects):
                break

            norm_pos = (
                start + (tick_val - left) / (right - left) * (1 - start)
                if right != left
                else 0.5
            )

            tick_obj = self._tick_objects[tick_idx]
            tick_obj.text = f"{tick_val:.0f}"
            tick_obj.transform = snx.Transform().translated((norm_pos, -0.5, 0))

            self.x_view.scene.add_child(tick_obj)

    def _update_y_axis(self) -> None:
        if self.mesh is None or self.y_max is None:
            return

        max_val = self.mesh.bounding_box[1][1]
        # Scale the y-axis to [0, 1] with a small top margin
        self.mesh.transform = snx.Transform().scaled((1, 0.95 / max(max_val, 1), 1))
        # Resize the y-axis against the new data
        self.y_max.text = f"{max_val:.2f}"

    def _hist_counts_to_mesh(
        self,
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
        # TODO: Reusing the arrays would be nice.
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
