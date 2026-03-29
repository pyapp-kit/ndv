from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pygfx
import pylinalg as la

from ndv._types import CursorType
from ndv.views._app import filter_mouse_events
from ndv.views.bases import SharedHistogramCanvas
from ndv.views.bases._graphics._histogram_utils import (
    LUT_LINE_ALPHA,
    MIN_GAMMA,
    Grabbable,
    area_to_mesh,
    downsample_histogram,
)

from ._histogram import _Controller, _OrthographicCamera
from ._util import rendercanvas_class

if TYPE_CHECKING:
    from ndv._types import (
        ChannelKey,
        MouseMoveEvent,
        MousePressEvent,
        MouseReleaseEvent,
    )


FILL_ALPHA = 0.08
_NO_KEY = object()  # sentinel for "no channel grabbed"


@dataclass
class _ChannelVisuals:
    """All visuals for a single channel on the shared histogram."""

    area_mesh: pygfx.Mesh
    outline: pygfx.Line
    left_clim: pygfx.Line
    right_clim: pygfx.Line
    gamma_line: pygfx.Line
    gamma_handle: pygfx.Points
    # per-channel state
    color: tuple = (1, 1, 1, 1)
    clims: tuple[float, float] | None = None
    gamma: float = 1.0
    counts: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    visible: bool = True
    name: str = ""
    _display_centers: np.ndarray | None = field(default=None, repr=False)
    _display_counts: np.ndarray | None = field(default=None, repr=False)


class PyGFXSharedHistogramCanvas(SharedHistogramCanvas):
    """Shared multi-channel histogram using PyGFX."""

    def __init__(self) -> None:
        self._channels: dict[object, _ChannelVisuals] = {}
        self._log_base: float | None = None
        self._grabbed: Grabbable = Grabbable.NONE
        self._grabbed_key: object = _NO_KEY
        self._clim_bounds: tuple[float | None, float | None] = (None, None)
        self._has_initial_range = False

        # Margins (pixels)
        self.margin_left = 10
        self.margin_bottom = 20
        self.margin_right = 4
        self.margin_top = 14  # room for legend text

        # ------------ PyGFX Canvas ------------ #
        cls = rendercanvas_class()
        self._size = (600, 600)
        self._canvas = cls(size=self._size)
        self._disconnect_mouse_events = filter_mouse_events(self._canvas, self)
        self._renderer = pygfx.renderers.WgpuRenderer(self._canvas)

        # Scene 0: full-canvas black background (rendered first)
        self._bg_scene = pygfx.Scene()
        self._bg_scene.add(pygfx.Background(None, pygfx.BackgroundMaterial("black")))
        self._bg_cam = pygfx.OrthographicCamera()

        # Scene 1: main plot (data, clim handles, highlight) — no background
        self._scene = pygfx.Scene()
        self._plot_view = pygfx.Viewport(self._renderer)
        self._controller = _Controller(register_events=self._plot_view)
        self._controller.controls.update({"wheel": ("zoom_to_point", "push", -0.005)})
        self._camera = _OrthographicCamera(maintain_aspect=False)
        self._controller.add_camera(self._camera, include_state={"x", "width"})

        # Scene 2: x-axis (synced camera for pan/zoom with main)
        # No background — layers on top of the main scene's background
        self._x_scene = pygfx.Scene()
        self._x_cam = _OrthographicCamera(maintain_aspect=False, width=1, height=1)
        self._controller.add_camera(self._x_cam, include_state={"x", "width"})

        # Scene 3: static overlays (y-label, legend) rendered full-canvas
        # No background — this renders on top of the plot and x-axis
        self._y_scene = pygfx.Scene()
        self._y_cam = pygfx.OrthographicCamera(maintain_aspect=False, width=1, height=1)
        self._y_cam.local.position = [0.5, 0.5, 0]

        # ------------ Nodes ------------ #

        # Highlight line
        self._highlight = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            ),
            material=pygfx.LineMaterial(
                color=(1.0, 1.0, 0.2, 0.75),
                dash_pattern=[4, 4],
                thickness=1.5,
            ),
            visible=False,
        )
        self._scene.add(self._highlight)

        # X-axis ruler
        self._x = pygfx.Ruler(
            start_pos=(0, 0, 0),
            end_pos=(1, 0, 0),
            start_value=0,
            tick_format=lambda v, *_: f"{v:g}",
            tick_side="right",
            tick_size=4,
            line_width=1,
        )
        self._x.text.font_size = 10
        self._x.text.material.weight_offset = -300
        self._x_scene.add(self._x)

        # Y-axis: just a max label (no ruler to save space)
        self._y_max_label = pygfx.MultiText(
            text="",
            material=pygfx.TextMaterial(color="white", aa=True, weight_offset=-300),
            screen_space=True,
            font_size=10,
            anchor="bottom-left",
        )
        self._y_scene.add(self._y_max_label)

        # Legend labels (also in y_scene for screen-space rendering)
        self._legend_labels: list[pygfx.MultiText] = []

        self.refresh()

    # ------------ GraphicsCanvas methods ------------ #

    def refresh(self) -> None:
        with suppress(AttributeError):
            self._canvas.update()
        self._canvas.request_draw(self._animate)

    def set_range(
        self,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
        z: tuple[float, float] | None = None,
        margin: float = 0,
    ) -> None:
        if x is None:
            x = self._compute_x_range()
        if y is None:
            y = self._compute_y_range()
        if x and y:
            self._resize(x, y)

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        pos_rel = (
            pos_xy[0] - self._plot_view.rect[0],
            pos_xy[1] - self._plot_view.rect[1],
        )
        vs = self._plot_view.logical_size
        x = pos_rel[0] / vs[0] * 2 - 1
        y = -(pos_rel[1] / vs[1] * 2 - 1)
        pos_ndc = (x, y, 0)
        if self._camera:
            pos_ndc += la.vec_transform(
                self._camera.world.position, self._camera.camera_matrix
            )
            pos_world = la.vec_unproject(pos_ndc[:2], self._camera.camera_matrix)
            return (pos_world[0], pos_world[1], pos_world[2])
        return (-1, -1, -1)

    def world_to_canvas(
        self, pos_xyz: tuple[float, float, float]
    ) -> tuple[float, float]:
        screen_space = pygfx.utils.transform.AffineTransform()
        screen_space.position = (-1, 1, 0)
        x_d, y_d = self._plot_view.logical_size
        screen_space.scale = (2 / x_d, -2 / y_d, 1)
        ndc_to_screen = screen_space.inverse_matrix
        canvas_pos = la.vec_transform(
            pos_xyz, ndc_to_screen @ self._camera.camera_matrix
        )
        return (
            canvas_pos[0] + self._plot_view.rect[0],
            canvas_pos[1] + self._plot_view.rect[1],
        )

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        raise NotImplementedError

    def set_visible(self, visible: bool) -> None: ...

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    def frontend_widget(self) -> Any:
        return self._canvas

    # ------------ SharedHistogramCanvas methods ------------ #

    def set_channel_data(
        self, key: ChannelKey, counts: np.ndarray, bin_edges: np.ndarray
    ) -> None:
        ch = self._ensure_channel(key)
        ch.counts = counts
        ch.bin_edges = bin_edges
        ch._display_centers, ch._display_counts = downsample_histogram(
            counts, bin_edges
        )
        self._update_channel_area(key)
        if not self._has_initial_range:
            self._has_initial_range = True
            self._auto_range()
        else:
            self._auto_range_y_only()

    def set_channel_color(self, key: ChannelKey, color: tuple) -> None:
        ch = self._ensure_channel(key)
        ch.color = color
        self._apply_channel_colors(key)

    def set_channel_visible(self, key: ChannelKey, visible: bool) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.visible = visible
        for obj in (
            ch.area_mesh,
            ch.outline,
            ch.left_clim,
            ch.right_clim,
            ch.gamma_line,
            ch.gamma_handle,
        ):
            obj.visible = visible
        self._update_legend()
        self._auto_range_y_only()

    def set_channel_clims(self, key: ChannelKey, clims: tuple[float, float]) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.clims = clims
        self._update_lut_visuals(key)

    def set_channel_gamma(self, key: ChannelKey, gamma: float) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.gamma = gamma
        self._update_lut_visuals(key)

    def remove_channel(self, key: ChannelKey) -> None:
        ch = self._channels.pop(key, None)
        if ch is None:
            return
        for obj in (
            ch.area_mesh,
            ch.outline,
            ch.left_clim,
            ch.right_clim,
            ch.gamma_line,
            ch.gamma_handle,
        ):
            self._scene.remove(obj)
        self._update_legend()
        self._auto_range()

    def set_channel_name(self, key: ChannelKey, name: str) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.name = name
        self._update_legend()

    def set_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        self._clim_bounds = bounds
        self._camera.xbounds = bounds
        self._x_cam.xbounds = bounds
        # Update x-axis ruler domain
        self._x.start_pos = [0 if bounds[0] is None else int(bounds[0]), 0, 0]
        self._x.end_pos = [65535 if bounds[1] is None else int(bounds[1]), 0, 0]

    def set_log_base(self, base: float | None) -> None:
        if base == self._log_base:
            return
        self._log_base = base
        for key in self._channels:
            self._update_channel_area(key)
        self._auto_range()

    def highlight(self, value: float | None) -> None:
        self._highlight.visible = value is not None
        if value is not None:
            self._highlight.local.x = value
            y_range = self._compute_y_range()
            if y_range:
                self._highlight.local.scale_y = y_range[1]
        self.refresh()

    # ------------ Mouse interaction ------------ #

    def get_cursor(self, event: MouseMoveEvent) -> CursorType:
        pos = (event.x, event.y)
        _key, nearby = self._find_nearest_grabbable(pos)
        if nearby in (Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM):
            return CursorType.H_ARROW
        elif nearby is Grabbable.GAMMA:
            return CursorType.V_ARROW
        else:
            x, y = pos
            x_max, y_max = self._plot_view.logical_size
            if (0 < x <= x_max) and (0 <= y <= y_max):
                return CursorType.ALL_ARROW
            return CursorType.DEFAULT

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        pos = event.x, event.y
        self._grabbed_key, self._grabbed = self._find_nearest_grabbable(pos)
        if self._grabbed != Grabbable.NONE:
            self._controller.enabled = False
        return False

    def on_mouse_double_press(self, event: MousePressEvent) -> bool:
        pos = event.x, event.y
        key, nearby = self._find_nearest_grabbable(pos)
        if nearby == Grabbable.GAMMA and key is not _NO_KEY:
            self.gammaChanged.emit(key, 1.0)
        return False

    def on_mouse_release(self, event: MouseReleaseEvent) -> bool:
        self._grabbed = Grabbable.NONE
        self._grabbed_key = _NO_KEY
        self._controller.enabled = True
        return False

    def on_mouse_move(self, event: MouseMoveEvent) -> bool:
        pos = event.x, event.y
        key = self._grabbed_key
        if key is _NO_KEY or self._grabbed == Grabbable.NONE:
            self.get_cursor(event).apply_to(self)
            return False

        ch = self._channels.get(key)
        if ch is None or ch.clims is None:
            return False

        if self._grabbed in (Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM):
            c = self.canvas_to_world(pos)[0]
            lo, hi = self._clim_bounds
            if lo is not None:
                c = max(c, lo)
            if hi is not None:
                c = min(c, hi)
            if self._grabbed is Grabbable.LEFT_CLIM:
                new_clims = (min(ch.clims[1], c), ch.clims[1])
            else:
                new_clims = (ch.clims[0], max(ch.clims[0], c))
            self.climsChanged.emit(key, new_clims)
            return False

        if self._grabbed is Grabbable.GAMMA:
            y_range = self._compute_y_range()
            y_top = (y_range[1] if y_range else 1.0) * 0.98
            y = self.canvas_to_world(pos)[1]
            if y <= 0 or y > y_top:
                return False
            gamma = max(MIN_GAMMA, -np.log2(y / y_top)) if y_top != 0 else 1.0
            self.gammaChanged.emit(key, float(gamma))
            return False

        self.get_cursor(event).apply_to(self)
        return False

    # ------------ Private helpers ------------ #

    def _ensure_channel(self, key: object) -> _ChannelVisuals:
        if key in self._channels:
            return self._channels[key]

        area_mesh = pygfx.Mesh(
            geometry=pygfx.Geometry(
                positions=np.zeros((1, 3), dtype=np.float32),
                indices=np.zeros((1, 3), dtype=np.uint32),
            ),
            material=pygfx.MeshBasicMaterial(
                color=(0.5, 0.5, 0.5, FILL_ALPHA),
                color_mode="uniform",
                depth_test=False,
                alpha_mode="blend",
            ),
        )

        outline = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0]], dtype=np.float32),
            ),
            material=pygfx.LineMaterial(color="white", thickness=1),
        )

        left_clim = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            ),
            material=pygfx.LineMaterial(color=(1, 1, 1, LUT_LINE_ALPHA), thickness=1),
            visible=False,
            render_order=-9,
        )

        right_clim = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            ),
            material=pygfx.LineMaterial(color=(1, 1, 1, LUT_LINE_ALPHA), thickness=1),
            visible=False,
            render_order=-9,
        )

        gamma_line = pygfx.Line(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0]], dtype=np.float32),
            ),
            material=pygfx.LineMaterial(color=(1, 1, 1, LUT_LINE_ALPHA), thickness=1),
            visible=False,
            render_order=-9,
        )

        gamma_handle = pygfx.Points(
            geometry=pygfx.Geometry(
                positions=np.array([[0, 0, 0]], dtype=np.float32),
            ),
            material=pygfx.PointsMaterial(
                size=6, color=(1, 1, 1), color_mode="uniform"
            ),
            visible=False,
            render_order=-10,
        )

        self._scene.add(area_mesh)
        self._scene.add(outline)
        self._scene.add(left_clim)
        self._scene.add(right_clim)
        self._scene.add(gamma_line)
        self._scene.add(gamma_handle)

        ch = _ChannelVisuals(
            area_mesh=area_mesh,
            outline=outline,
            left_clim=left_clim,
            right_clim=right_clim,
            gamma_line=gamma_line,
            gamma_handle=gamma_handle,
        )
        self._channels[key] = ch
        self._update_legend()
        return ch

    def _apply_channel_colors(self, key: object) -> None:
        ch = self._channels[key]
        r, g, b = ch.color[:3]
        a = ch.color[3] if len(ch.color) > 3 else 1.0

        ch.area_mesh.material.color = (r, g, b, FILL_ALPHA)
        ch.outline.material.color = (r, g, b, a)
        ch.left_clim.material.color = (r, g, b, LUT_LINE_ALPHA)
        ch.right_clim.material.color = (r, g, b, LUT_LINE_ALPHA)
        ch.gamma_line.material.color = (r, g, b, LUT_LINE_ALPHA)
        ch.gamma_handle.material.color = (r, g, b, a)

        self._update_channel_area(key)
        self._update_lut_visuals(key)

    def _update_channel_area(self, key: object) -> None:
        ch = self._channels.get(key)
        if ch is None or ch._display_centers is None or ch._display_counts is None:
            return

        counts = ch._display_counts
        centers = ch._display_centers
        if self._log_base:
            counts = np.log(counts + 1) / np.log(self._log_base)

        verts, faces = area_to_mesh(centers, counts)
        if (
            verts.shape == ch.area_mesh.geometry.positions.data.shape
            and faces.shape == ch.area_mesh.geometry.indices.data.shape
        ):
            ch.area_mesh.geometry.positions.data[:] = verts
            ch.area_mesh.geometry.positions.update_range()
            ch.area_mesh.geometry.indices.data[:] = faces
            ch.area_mesh.geometry.indices.update_range()
        else:
            ch.area_mesh.geometry = pygfx.Geometry(positions=verts, indices=faces)

        outline_pos = np.zeros((len(centers), 3), dtype=np.float32)
        outline_pos[:, 0] = centers
        outline_pos[:, 1] = counts
        self._update_line_positions(ch.outline, outline_pos)

        self.refresh()

    def _update_lut_visuals(self, key: object, npoints: int = 64) -> None:
        ch = self._channels.get(key)
        if ch is None or ch.clims is None:
            return

        clims = ch.clims
        gamma = ch.gamma

        # Use global y range for consistent clim handle height
        y_range = self._compute_y_range()
        y_max = y_range[1] if y_range else 1.0
        if y_max == 0:
            y_max = 1.0
        y_top = y_max * 0.98

        # Left clim line (full height)
        left_pos = np.array([[clims[0], 0, 0], [clims[0], y_top, 0]], dtype=np.float32)
        self._update_line_positions(ch.left_clim, left_pos)
        ch.left_clim.visible = ch.visible

        # Right clim line (full height)
        right_pos = np.array([[clims[1], 0, 0], [clims[1], y_top, 0]], dtype=np.float32)
        self._update_line_positions(ch.right_clim, right_pos)
        ch.right_clim.visible = ch.visible

        # Gamma curve
        t = np.linspace(0, 1, npoints)
        gx = np.linspace(clims[0], clims[1], npoints)
        gy = t**gamma * y_top
        gamma_pos = np.zeros((npoints, 3), dtype=np.float32)
        gamma_pos[:, 0] = gx
        gamma_pos[:, 1] = gy
        self._update_line_positions(ch.gamma_line, gamma_pos)
        ch.gamma_line.visible = ch.visible

        # Gamma handle
        mid_x = np.mean(clims)
        mid_y = (2 ** (-gamma)) * y_top
        handle_pos = np.array([[float(mid_x), mid_y, 0]], dtype=np.float32)
        ch.gamma_handle.geometry.positions.data[:] = handle_pos
        ch.gamma_handle.geometry.positions.update_range()
        ch.gamma_handle.visible = ch.visible

        self.refresh()

    def _update_line_positions(self, line: pygfx.Line, pos: np.ndarray) -> None:
        if pos.shape == line.geometry.positions.data.shape:
            line.geometry.positions.data[:] = pos
            line.geometry.positions.update_range()
        else:
            line.geometry = pygfx.Geometry(positions=pos)

    def _channel_y_max(self, ch: _ChannelVisuals) -> float:
        if ch._display_counts is None:
            return 1.0
        counts = ch._display_counts
        if self._log_base:
            counts = np.log(counts + 1) / np.log(self._log_base)
        return float(np.max(counts)) if len(counts) > 0 else 1.0

    def _compute_x_range(self) -> tuple[float, float] | None:
        x_min, x_max = float("inf"), float("-inf")
        for ch in self._channels.values():
            if not ch.visible or ch.bin_edges is None:
                continue
            x_min = min(x_min, ch.bin_edges[0])
            x_max = max(x_max, ch.bin_edges[-1])
        if x_min == float("inf"):
            return None
        return (float(x_min), float(x_max))

    def _compute_y_range(self) -> tuple[float, float] | None:
        """Compute y range across all visible channels, with headroom."""
        y_max = 0.0
        for ch in self._channels.values():
            if not ch.visible:
                continue
            y_max = max(y_max, self._channel_y_max(ch))
        if y_max == 0:
            return None
        return (0, y_max * 1.05)

    def _auto_range(self) -> None:
        x = self._compute_x_range()
        y = self._compute_y_range()
        if x and y:
            self._resize(x, y)
        self._refresh_all_lut_visuals()
        self._update_legend()

    def _auto_range_y_only(self) -> None:
        """Update y range only, preserving current x pan/zoom."""
        y = self._compute_y_range()
        if y:
            cx = self._camera.local.x
            cw = self._camera.width
            self._camera.height = y[1] - y[0]
            self._camera.local.position = [cx, (y[0] + y[1]) / 2, 0]
            self._camera.width = cw
            c_w = max(self._canvas.get_logical_size()[0], 1)
            c_h = max(self._canvas.get_logical_size()[1], 1)
            self._update_y_ruler(c_w, c_h, y[1])
        self._refresh_all_lut_visuals()
        self._update_legend()
        self.refresh()

    def _refresh_all_lut_visuals(self) -> None:
        for key in self._channels:
            self._update_lut_visuals(key)

    def _resize(self, x: tuple[float, float], y: tuple[float, float]) -> None:
        self._x_cam.width = self._camera.width = x[1] - x[0]
        self._camera.height = y[1] - y[0]

        self._x_cam.local.position = [(x[0] + x[1]) / 2, 0, 0]
        self._camera.local.position = [
            (x[0] + x[1]) / 2,
            (y[0] + y[1]) / 2,
            0,
        ]

        c_w = max(self._canvas.get_logical_size()[0], 1)
        c_h = max(self._canvas.get_logical_size()[1], 1)
        self._update_y_ruler(c_w, c_h, y[1])
        self.refresh()

    def _update_y_ruler(self, canvas_w: float, canvas_h: float, max_val: float) -> None:
        """Update the y-axis max label position and text."""
        x0 = self.margin_left / canvas_w
        y1 = (canvas_h - self.margin_top) / canvas_h
        if max_val > 0:
            count = self._log_base**max_val - 1 if self._log_base else max_val
            self._y_max_label.set_text(f"{count:.0f}")
            self._y_max_label.local.position = (x0, y1 + 0.005, 0)
        else:
            self._y_max_label.set_text("")

    def _update_legend(self) -> None:
        """Position legend entries horizontally at the top-right."""
        c_w = max(self._canvas.get_logical_size()[0], 1)
        c_h = max(self._canvas.get_logical_size()[1], 1)

        # Collect visible channel entries
        entries: list[tuple[str, tuple]] = []
        for ch in self._channels.values():
            if ch.visible and ch.name:
                r, g, b = ch.color[:3]
                a = ch.color[3] if len(ch.color) > 3 else 1.0
                entries.append((f"{ch.name}", (r, g, b, a)))

        # Ensure we have enough legend labels
        while len(self._legend_labels) < len(entries):
            label = pygfx.MultiText(
                text="",
                material=pygfx.TextMaterial(color="white", aa=True, weight_offset=-300),
                screen_space=True,
                font_size=10,
                anchor="bottom-right",
            )
            self._y_scene.add(label)
            self._legend_labels.append(label)

        # Position entries right-to-left, inline with y-max label
        x_frac = (c_w - 8) / c_w
        # Same y as y_max_label (both use "bottom-*" anchor now)
        y_frac = (c_h - self.margin_top) / c_h + 0.005
        for i, label in enumerate(self._legend_labels):
            if i < len(entries):
                text, color = entries[len(entries) - 1 - i]
                label.set_text(text)
                label.material.color = color
                label.local.position = (x_frac, y_frac, 0)
                label.visible = True
                x_frac -= (len(text) * 6 + 6) / c_w
            else:
                label.visible = False

    def _animate(self) -> None:
        rect = self._canvas.get_logical_size()
        if rect != self._size:
            self._plot_view.rect = (
                self.margin_left,
                self.margin_top,
                max(0, rect[0] - self.margin_left - self.margin_right),
                max(0, rect[1] - self.margin_top - self.margin_bottom),
            )
            self._size = rect

            y_range = self._compute_y_range()
            max_val = y_range[1] if y_range else 0
            self._update_y_ruler(max(rect[0], 1), max(rect[1], 1), max_val)
            self._update_legend()

        self._x.update(self._x_cam, self._canvas.get_logical_size())

        # Render background full-canvas first
        self._renderer.render(self._bg_scene, self._bg_cam, flush=False)
        # Render the plot on top
        self._plot_view.render(self._scene, self._camera, flush=False)
        # Render the x-axis
        self._renderer.render(
            self._x_scene,
            self._x_cam,
            rect=(
                self.margin_left,
                self.margin_top + self._plot_view.rect[3] - self.margin_bottom,
                self._plot_view.rect[2],
                2 * self.margin_bottom,
            ),
            flush=False,
        )
        # Render the y-axis / legend overlay
        self._renderer.render(self._y_scene, self._y_cam, flush=False)
        self._renderer.flush()

    def _find_nearest_grabbable(
        self, pos: tuple[float, float], tolerance: int = 5
    ) -> tuple[object, Grabbable]:
        click_x, click_y = pos
        best_dist = float("inf")
        best_key: object = _NO_KEY
        best_grab = Grabbable.NONE

        y_range = self._compute_y_range()
        global_y_max = y_range[1] if y_range else 1.0

        for key, ch in self._channels.items():
            if not ch.visible or ch.clims is None:
                continue

            left_cx = self.world_to_canvas((ch.clims[0], 0, 0))[0]
            right_cx = self.world_to_canvas((ch.clims[1], 0, 0))[0]

            d_right = abs(right_cx - click_x)
            if d_right < tolerance and d_right < best_dist:
                best_dist = d_right
                best_key = key
                best_grab = Grabbable.RIGHT_CLIM

            d_left = abs(left_cx - click_x)
            if d_left < tolerance and d_left < best_dist:
                best_dist = d_left
                best_key = key
                best_grab = Grabbable.LEFT_CLIM

            mid_x = np.mean(ch.clims)
            mid_y = (2 ** (-ch.gamma)) * global_y_max * 0.98
            gx, gy = self.world_to_canvas((float(mid_x), mid_y, 0))
            d_gamma = ((gx - click_x) ** 2 + (gy - click_y) ** 2) ** 0.5
            if d_gamma < tolerance and d_gamma < best_dist:
                best_dist = d_gamma
                best_key = key
                best_grab = Grabbable.GAMMA

        return best_key, best_grab
