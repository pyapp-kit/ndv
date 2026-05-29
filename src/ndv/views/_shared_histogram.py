from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor, log10
from typing import TYPE_CHECKING, Any

import cmap
import numpy as np
import scenex as snx
from psygnal import Signal
from scenex.adaptors import get_adaptor_registry
from scenex.app import CursorType, events
from scenex.util import projections

from ndv.views._util import (
    apply_log_counts,
    area_to_mesh,
    downsample_histogram,
)

if TYPE_CHECKING:
    from ndv._types import ChannelKey

# PyGFX's blending produces brighter fills than Vispy at the same alpha;
# use a lower value here so both backends look visually similar.
FILL_ALPHA = 0.08


@dataclass
class _ChannelVisuals:
    """All visuals for a single channel on the shared histogram."""

    controls: snx.Scene
    area_mesh: snx.Mesh
    outline: snx.Line
    left_clim: snx.Line
    right_clim: snx.Line
    gamma_line: snx.Line
    highlight: snx.Line
    gamma_handle: snx.Points
    legend_text: snx.Text
    # per-channel state
    color: cmap.Color
    clims: tuple[float, float] | None = None
    gamma: float = 1.0
    counts: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    visible: bool = True
    name: str = ""
    _display_centers: np.ndarray | None = field(default=None, repr=False)
    _display_counts: np.ndarray | None = field(default=None, repr=False)


Y_AXIS = 40  # pixels reserved for y-axis strip
X_AXIS = 25  # pixels reserved for x-axis strip
LEGEND_W = 50  # width of legend area
LEGEND_H = 20  # height of legend area


class SharedHistogram:
    """Shared multi-channel histogram using Scenex."""

    climsChanged = Signal(object, tuple)
    gammaChanged = Signal(object, float)

    def __init__(self) -> None:
        self._channels: dict[object, _ChannelVisuals] = {}
        self._log_base: float | None = None
        self._grabbed: snx.Node | None = None
        self._grabbed_key: object = None
        self._clim_bounds: tuple[float | None, float | None] = (None, None)
        self._has_initial_range = False
        self._last_cam_state: tuple[float, float] = (0.0, 0.0)  # (x, width)

        # Margins (pixels)
        self.margin_left = 10
        self.margin_bottom = 20
        self.margin_right = 4
        self.margin_top = 14  # room for legend text

        # ------------ Scenex setup ------------ #

        # NOTE: Keep the canvas hidden until we're ready to show it.
        # More than anything else, this prevents the GL context from being created
        # on the vispy backends before we're ready, leading to some nasty segfaults
        self.canvas = snx.Canvas(visible=False)

        self.x_view = snx.View(scene=snx.Scene(), camera=snx.Camera())
        self.view = snx.View(
            scene=snx.Scene(name="main scene"),
            camera=snx.Camera(interactive=True),
        )
        self.y_view = snx.View(scene=snx.Scene(), camera=snx.Camera())

        self.legend_view = snx.View(
            scene=snx.Scene(name="legend"),
            camera=snx.Camera(),
        )
        self.legend_view.layout.background_color = cmap.Color((0, 0, 0, 0))

        self.canvas.views.append(self.x_view)
        self.canvas.views.append(self.y_view)
        self.canvas.views.append(self.view)
        self.canvas.views.append(self.legend_view)

        self.x_view.layout.y_start = f"-{X_AXIS}px"
        self.y_view.layout.y_end = f"-{X_AXIS}px"
        self.view.layout.y_end = f"-{X_AXIS}px"
        self.view.layout.x_start = f"{Y_AXIS}px"
        self.y_view.layout.x_end = f"{Y_AXIS}px"
        self.legend_view.layout.x_start = f"-{LEGEND_W}px"
        self.legend_view.layout.y_end = f"{LEGEND_H}px"

        # ------------ Nodes ------------ #

        self.x_axis = snx.Line(
            vertices=np.array([[0, 0, 0], [1, 0, 0]]),
            width=2,
            color=snx.UniformColor(color=cmap.Color("white")),
        )
        self.x_view.scene.add_child(self.x_axis)
        self._tick_objects: list[snx.Text] = []
        for _ in range(10):
            tick_line = snx.Line(
                vertices=np.array([[0, 0, 0], [0, -0.1, 0]]),
                width=1,
                color=snx.UniformColor(color=cmap.Color("white")),
                transform=snx.Transform().translated((0, 0.4, 0)),
            )
            tick_text = snx.Text(text="0", children=[tick_line], antialias=True)
            self._tick_objects.append(tick_text)

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

        # Legend labels (also in y_scene for screen-space rendering)
        self._legend_labels: list[snx.Text] = []

        self.view.camera.controller = snx.PanZoom(lock_y=True)
        self.view.camera.events.transform.connect(self._update_x_axis)
        self.view.camera.events.projection.connect(self._update_x_axis)
        self.canvas.events.width.connect(self._update_x_axis)
        self.view.set_event_filter(self._on_main_view)

    # ------------ SharedHistogramCanvas methods ------------ #

    def widget(self) -> Any:
        return get_adaptor_registry().get_adaptor(self.canvas)._snx_get_native()

    def set_channel_data(
        self, key: ChannelKey, counts: np.ndarray, bin_edges: np.ndarray
    ) -> None:
        ch = self._ensure_channel(key)
        ch.counts = counts
        ch.bin_edges = bin_edges
        self._update_channel_area(key)

        if not self._has_initial_range:
            self._has_initial_range = True
            self.set_range()
        else:
            self.set_range(skip_x=True)
        self._update_y_ruler()

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
            ch.legend_text,
        ):
            obj.visible = visible
        self._update_legend()
        self.set_range(skip_x=True)

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
            ch.controls,
        ):
            self.view.scene.remove_child(obj)
        self._update_legend()
        self.set_range()

    def set_channel_name(self, key: ChannelKey, name: str) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.name = name
        ch.legend_text.text = f"● {name}"
        self._update_legend()

    def set_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        self._clim_bounds = bounds
        # self._camera.xbounds = bounds
        # self._x_cam.xbounds = bounds
        # # Update x-axis ruler domain
        # self._x.start_pos = [0 if bounds[0] is None else int(bounds[0]), 0, 0]
        # self._x.end_pos = [65535 if bounds[1] is None else int(bounds[1]), 0, 0]

    def set_log_base(self, base: float | None) -> None:
        if base == self._log_base:
            return
        self._log_base = base
        for key in self._channels:
            self._update_channel_area(key)
        self._update_y_ruler()

    def highlight(self, channel_values: dict[object, float]) -> None:
        # NOTE: This behavior differs from the previous version.
        # Previously, if a channel in the dict has not yet been added to the histogram,
        # a line would be added in. This is not the case anymore.
        for key, vis in self._channels.items():
            vis.highlight.visible = key in channel_values
            vis.highlight.transform = snx.Transform().translated(
                (channel_values.get(key, 0), 0, 0)
            )

    # ------------ Private helpers ------------ #

    def _ensure_channel(self, key: object) -> _ChannelVisuals:
        if key in self._channels:
            return self._channels[key]

        area_mesh = snx.Mesh(
            parent=self.view.scene,
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.uint32),
            opacity=0.3,
        )

        outline = snx.Line(
            parent=self.view.scene,
            vertices=np.array([[0, 0, 0]], dtype=np.float32),
        )

        left_clim = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            interactive=True,
        )

        right_clim = snx.Line(
            vertices=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            interactive=True,
        )

        gamma_line = snx.Line(
            parent=self.view.scene,
            vertices=np.array([[0, 0, 0]], dtype=np.float32),
        )

        gamma_handle = snx.Points(
            vertices=np.array([[0, 0, 0]], dtype=np.float32),
            scaling="fixed",
            size=12,
            interactive=True,
        )

        highlight = snx.Line(
            parent=self.view.scene,
            vertices=np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32),
            visible=False,  # initially hidden until a channel is highlighted
        )

        legend_text = snx.Text(
            parent=self.legend_view.scene,
            text="",
            size=10,
        )

        controls = snx.Scene(
            parent=self.view.scene, children=[left_clim, right_clim, gamma_handle]
        )

        ch = _ChannelVisuals(
            controls=controls,
            area_mesh=area_mesh,
            outline=outline,
            left_clim=left_clim,
            right_clim=right_clim,
            gamma_line=gamma_line,
            gamma_handle=gamma_handle,
            highlight=highlight,
            legend_text=legend_text,
            color=cmap.Color("white"),
        )
        self._channels[key] = ch
        self._update_legend()
        return ch

    def _apply_channel_colors(self, key: object) -> None:
        ch = self._channels[key]
        color = snx.UniformColor(color=ch.color)
        ch.area_mesh.color = color
        ch.outline.color = color
        ch.left_clim.color = color
        ch.right_clim.color = color
        ch.gamma_line.color = color
        ch.gamma_handle.edge_color = color
        ch.highlight.color = color
        ch.legend_text.color = ch.color

        self._update_channel_area(key)
        self._update_lut_visuals(key)

    def _update_channel_area(self, key: object) -> None:
        ch = self._channels.get(key)
        if ch is None or ch.counts is None or ch.bin_edges is None:
            return

        canvas_w = max(int(self.canvas.content_rect_for(self.view)[0]), 64)
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
        ch.area_mesh.vertices = verts
        ch.area_mesh.faces = faces

        outline_pos = np.zeros((len(centers), 3), dtype=np.float32)
        outline_pos[:, 0] = centers
        outline_pos[:, 1] = counts
        ch.outline.vertices = outline_pos

    def _update_lut_visuals(self, key: object, npoints: int = 64) -> None:
        ch = self._channels.get(key)
        if ch is None or ch.clims is None:
            return

        clims = ch.clims
        gamma = ch.gamma

        # Left clim line (full height)
        left_pos = np.array([[clims[0], 0, 0], [clims[0], 1, 0]], dtype=np.float32)
        ch.left_clim.vertices = left_pos
        ch.left_clim.visible = ch.visible

        # Right clim line (full height)
        right_pos = np.array([[clims[1], 0, 0], [clims[1], 1, 0]], dtype=np.float32)
        ch.right_clim.vertices = right_pos
        ch.right_clim.visible = ch.visible

        # Gamma curve
        t = np.linspace(0, 1, npoints)
        gx = np.linspace(clims[0], clims[1], npoints)
        gy = t**gamma * 1
        gamma_pos = np.zeros((npoints, 3), dtype=np.float32)
        gamma_pos[:, 0] = gx
        gamma_pos[:, 1] = gy
        ch.gamma_line.vertices = gamma_pos
        ch.gamma_line.visible = ch.visible

        # Gamma handle
        mid_x, mid_y = float(np.mean(clims)), (2 ** (-gamma))
        handle_pos = np.array([[mid_x, mid_y, 0]], dtype=np.float32)
        ch.gamma_handle.vertices = handle_pos
        ch.gamma_handle.visible = ch.visible

    def set_range(
        self,
        skip_x: bool = False,
        skip_y: bool = False,
        x: tuple[float, float] | None = None,
        y: tuple[float, float] | None = None,
    ) -> None:
        bb = self.view.scene.bounding_box
        center = np.mean(bb, axis=0) if bb else (0, 0, 0)
        w, h, d = np.maximum(np.ptp(bb, axis=0) if bb else (1, 1, 1), 1e-6)
        if skip_x:
            center = (self.view.camera.transform.root[3, 0], center[1], center[2])
            w = 2 / self.view.camera.projection.root[0, 0]
        if skip_y:
            center = (center[0], self.view.camera.transform.root[3, 1], center[2])
            h = 2 / self.view.camera.projection.root[1, 1]
        if x is not None:
            center = (np.mean(x), center[1], center[2])
            w = x[1] - x[0]
        if y is not None:
            center = (center[0], np.mean(y), center[2])
            h = y[1] - y[0]
        self.view.camera.transform = snx.Transform().translated(center)
        self.view.camera.projection = projections.orthographic(w, h, d)

        self.x_view.camera.projection = projections.orthographic(1, 1, 1)
        self.y_view.camera.projection = projections.orthographic(1, 1, 1)
        self.x_view.camera.transform = snx.Transform().translated((0.5, -0.5, 0))
        self.y_view.camera.transform = snx.Transform().translated((-0.5, 0.5, 0))

    def _visible_x_range(self) -> tuple[float, float] | None:
        """Get the currently visible x-range from the camera."""
        if not self._has_initial_range:
            return None
        cx = self.view.camera.transform.root[3, 0]
        w = 1 / self.view.camera.projection.root[0, 0]
        return (cx - w, cx + w)

    def _redownsample_all(self) -> None:
        """Re-downsample all channels for the current visible range."""
        for key in self._channels:
            self._update_channel_area(key)

    def _refresh_all_lut_visuals(self) -> None:
        for key in self._channels:
            self._update_lut_visuals(key)

    def _update_y_ruler(self) -> None:
        """Update the y-axis max label position and text."""
        max_val = max(
            ch.counts.max() for ch in self._channels.values() if ch.counts is not None
        )
        if self._log_base:
            max_val = np.log(max_val + 1) / np.log(self._log_base)
        self.y_max.text = f"{max_val:.0f}" if max_val > 0 else ""
        tform = snx.Transform().scaled((1, 1 / max_val, 1))
        for ch in self._channels.values():
            ch.area_mesh.transform = ch.outline.transform = tform

    def _update_legend(self) -> None:
        """Position legend entries horizontally at the top-right."""
        # Collect visible channel entries
        visible_channels: list[snx.Text] = []
        for ch in self._channels.values():
            should_display = ch.visible and bool(ch.name)
            ch.legend_text.visible = should_display
            if should_display:
                visible_channels.append(ch.legend_text)
        self.legend_view.layout.y_end = f"{len(visible_channels) * 20}px"

        # Position entries top-to-bottom
        for i, text in enumerate(visible_channels):
            text.transform = snx.Transform().translated(
                (0, i / len(visible_channels), 0)
            )

    def _on_main_view(self, event: events.Event) -> bool:
        if isinstance(event, events.MousePressEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            for key, ch in self._channels.items():
                hits = [n for n, _ in ray.intersections(ch.controls) if n.interactive]
                if hits:
                    self._grabbed_key = key
                    self._grabbed = hits[0]
                    self.view.camera.interactive = False
                    break

        elif isinstance(event, events.MouseDoublePressEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            for key, ch in self._channels.items():
                hits = [n for n, _ in ray.intersections(ch.controls) if n.interactive]
                if ch.gamma_handle in hits:
                    self.gammaChanged.emit(key, 1.0)
                    break

        if isinstance(event, events.MouseMoveEvent):
            if not (ray := self.view.to_ray(event.pos)):
                return False
            grabbed_ch = self._channels.get(self._grabbed_key, None)
            if grabbed_ch is not None and self._grabbed is not None:
                if (
                    clims := grabbed_ch.clims
                ) and self._grabbed is grabbed_ch.left_clim:
                    new_left = min(ray.origin[0], clims[1])
                    if grabbed_ch.bin_edges is not None:
                        new_left = max(new_left, float(grabbed_ch.bin_edges[0]))
                    self.climsChanged.emit(self._grabbed_key, (new_left, clims[1]))
                    return True
                elif (
                    clims := grabbed_ch.clims
                ) and self._grabbed is grabbed_ch.right_clim:
                    new_right = max(clims[0], ray.origin[0])
                    if grabbed_ch.bin_edges is not None:
                        new_right = min(new_right, float(grabbed_ch.bin_edges[-1]))
                    self.climsChanged.emit(self._grabbed_key, (clims[0], new_right))
                    return True
                elif self._grabbed is grabbed_ch.gamma_handle:
                    y = ray.origin[1]
                    self.gammaChanged.emit(self._grabbed_key, max(-np.log2(y), 0.1))
                    return True
            else:
                cursor = CursorType.DEFAULT
                for ch in self._channels.values():
                    if ray.intersections(ch.right_clim) or ray.intersections(
                        ch.left_clim
                    ):
                        cursor = CursorType.H_ARROW
                        break
                    elif ray.intersections(ch.gamma_handle):
                        cursor = CursorType.V_ARROW
                        break
                snx.set_cursor(self.canvas, cursor)

        elif isinstance(event, events.MouseReleaseEvent | events.MouseLeaveEvent):
            self._grabbed_key = None
            self._grabbed = None
            self.view.camera.interactive = True

        return False

    # ---- x-axis ticks ----

    def _calculate_tick_step(
        self, min_val: float, max_val: float, target_ticks: int = 5
    ) -> float:
        if max_val <= min_val:
            return 1.0
        approx_step = (max_val - min_val) / target_ticks
        power10 = 10.0 ** floor(log10(approx_step))
        for m in [1.0, 2.0, 2.5, 5.0, 10.0]:
            if m * power10 >= approx_step:
                return m * power10
        return power10

    def _get_tick_positions(
        self, min_val: float, max_val: float, step: float
    ) -> list[float]:
        if step <= 0:
            return [min_val, max_val]
        first = ceil(min_val / step) * step
        last = floor(max_val / step) * step
        ticks: list[float] = []
        cur = first
        while cur <= last and len(ticks) < 20:
            ticks.append(cur)
            cur += step
        min_dist = step * 0.15
        filtered = [
            t
            for t in ticks
            if abs(t - min_val) >= min_dist and abs(t - max_val) >= min_dist
        ]
        seen: set[float] = set()
        result: list[float] = []
        for t in [min_val, *filtered, max_val]:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _clear_ticks(self) -> None:
        for tick in self._tick_objects:
            if tick in self.x_view.scene.children:
                self.x_view.scene.remove_child(tick)

    def _update_x_axis(self) -> None:
        cam = self.view.camera
        left, *_ = cam.transform.map(cam.projection.imap((-1, 0)))
        right, *_ = cam.transform.map(cam.projection.imap((1, 0)))
        self._clear_ticks()
        step = self._calculate_tick_step(left, right)
        positions = self._get_tick_positions(left, right, step)
        _x, _y, w, _h = self.canvas.rect_for(self.x_view)
        start = Y_AXIS / w
        for idx, val in enumerate(positions):
            if idx >= len(self._tick_objects):
                break
            norm = (
                start + (val - left) / (right - left) * (1 - start)
                if right != left
                else 0.5
            )
            tick = self._tick_objects[idx]
            tick.text = f"{val:.0f}"
            tick.transform = snx.Transform().translated((norm, -0.5, 0))
            self.x_view.scene.add_child(tick)

        self._redownsample_all()
