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
    _NO_KEY,
    LUT_LINE_ALPHA,
    Grabbable,
    apply_log_counts,
    area_to_mesh,
    clamp_clim_drag,
    compute_x_range,
    compute_y_range,
    downsample_histogram,
    find_nearest_grabbable,
    gamma_from_mouse_y,
    gamma_handle_pos,
    y_top_from_range,
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


# PyGFX's blending produces brighter fills than Vispy at the same alpha;
# use a lower value here so both backends look visually similar.
FILL_ALPHA = 0.08


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
        self._last_cam_state: tuple[float, float] = (0.0, 0.0)  # (x, width)

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

        # Per-channel highlight lines (created on demand)
        self._highlight_lines: dict[object, pygfx.Line] = {}
        self._highlight_unit_pos = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32)

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
        if (hl := self._highlight_lines.pop(key, None)) is not None:
            self._scene.remove(hl)
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
        self._auto_range_y_only()

    def highlight(self, channel_values: dict[object, float]) -> None:
        y_range = self._compute_y_range()
        y_scale = y_range[1] * 0.5 if y_range else 1.0
        active_keys = set()
        for key, value in channel_values.items():
            active_keys.add(key)
            line = self._highlight_lines.get(key)
            if line is None:
                ch = self._channels.get(key)
                color = (*ch.color[:3], 0.5) if ch else (1.0, 1.0, 0.2, 0.5)
                line = pygfx.Line(
                    geometry=pygfx.Geometry(positions=self._highlight_unit_pos),
                    material=pygfx.LineMaterial(
                        color=color, dash_pattern=[4, 4], thickness=1
                    ),
                )
                self._scene.add(line)
                self._highlight_lines[key] = line
            line.visible = True
            line.local.x = value
            line.local.scale_y = y_scale
        for key, line in self._highlight_lines.items():
            if key not in active_keys:
                line.visible = False
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
            new_clims = clamp_clim_drag(self._grabbed, c, ch.clims, self._clim_bounds)
            self.climsChanged.emit(key, new_clims)
            return False

        if self._grabbed is Grabbable.GAMMA:
            y = self.canvas_to_world(pos)[1]
            gamma = gamma_from_mouse_y(y, self._compute_y_range())
            if gamma is None:
                return False
            self.gammaChanged.emit(key, gamma)
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
        if ch is None or ch.counts is None or ch.bin_edges is None:
            return

        canvas_w = max(int(self._canvas.get_logical_size()[0]), 64)
        visible = self._visible_x_range()
        centers, display_counts = downsample_histogram(
            ch.counts,
            ch.bin_edges,
            max_display_bins=canvas_w,
            visible_range=visible,
        )
        ch._display_centers = centers
        ch._display_counts = display_counts
        counts = apply_log_counts(display_counts, self._log_base)

        verts, faces = area_to_mesh(centers, counts)
        if len(centers) < 2:
            return
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
        y_top = y_top_from_range(self._compute_y_range())

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
        mid_x, mid_y = gamma_handle_pos(clims, gamma, y_top)
        handle_pos = np.array([[mid_x, mid_y, 0]], dtype=np.float32)
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

    def _compute_x_range(self) -> tuple[float, float] | None:
        return compute_x_range(self._channels)

    def _compute_y_range(self) -> tuple[float, float] | None:
        return compute_y_range(self._channels, self._log_base)

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

    def _visible_x_range(self) -> tuple[float, float] | None:
        """Get the currently visible x-range from the camera."""
        if not self._has_initial_range:
            return None
        cx = self._camera.local.x
        hw = self._camera.width / 2
        return (cx - hw, cx + hw)

    def _redownsample_all(self) -> None:
        """Re-downsample all channels for the current visible range."""
        for key in self._channels:
            self._update_channel_area(key)
        # Refit y-axis to the visible data
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

        # Re-downsample when camera pans/zooms
        cam_state = (self._camera.local.x, self._camera.width)
        if cam_state != self._last_cam_state:
            self._last_cam_state = cam_state
            self._redownsample_all()

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
        w2c = self.world_to_canvas
        return find_nearest_grabbable(
            self._channels,
            pos,
            lambda x, y: w2c((x, y, 0))[:2],
            self._compute_y_range(),
            tolerance,
        )
