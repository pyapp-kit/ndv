from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from vispy import scene

from ndv._types import CursorType
from ndv.views._app import filter_mouse_events
from ndv.views.bases import SharedHistogramCanvas

from ._plot_widget import LogTicker, PlotWidget

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ndv._types import (
        ChannelKey,
        MouseMoveEvent,
        MousePressEvent,
        MouseReleaseEvent,
    )

MIN_GAMMA: np.float64 = np.float64(1e-6)
_NO_KEY = object()  # sentinel for "no channel grabbed"
# Max bins to display (downsample larger histograms for performance)
MAX_DISPLAY_BINS = 1024
# Alpha for area fill
FILL_ALPHA = 0.3
# Alpha for clim/gamma lines
LUT_LINE_ALPHA = 0.6


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


@dataclass
class _ChannelVisuals:
    """All visuals for a single channel on the shared histogram."""

    area_mesh: scene.Mesh
    outline: scene.LinePlot
    lut_line: scene.LinePlot
    gamma_handle: scene.Markers
    legend_text: scene.Text
    # per-channel state
    color: tuple = (1, 1, 1, 1)
    clims: tuple[float, float] | None = None
    gamma: float = 1.0
    counts: np.ndarray | None = None
    bin_edges: np.ndarray | None = None
    visible: bool = True
    name: str = ""
    # downsampled data for display
    _display_centers: np.ndarray | None = field(default=None, repr=False)
    _display_counts: np.ndarray | None = field(default=None, repr=False)


class VispySharedHistogramCanvas(SharedHistogramCanvas):
    """Shared multi-channel histogram using VisPy."""

    def __init__(self) -> None:
        self._channels: dict[object, _ChannelVisuals] = {}
        self._log_base: float | None = None
        self._grabbed: Grabbable = Grabbable.NONE
        self._grabbed_key: object = _NO_KEY
        self._clim_bounds: tuple[float | None, float | None] = (None, None)

        # VisPy canvas and plot
        self._canvas = scene.SceneCanvas()
        self._disconnect_mouse_events = filter_mouse_events(self._canvas.native, self)

        # Highlight line for domain value
        self._highlight = scene.Line(
            pos=np.array([[0, 0], [0, 1]]),
            color=(1, 1, 0.2, 0.75),
            connect="strip",
            width=1,
        )
        self._highlight_tform = scene.transforms.STTransform()
        self._highlight.visible = False
        self._highlight.order = -3

        self.plot = PlotWidget()
        self.plot.lock_axis("y")
        # Minimize left-side spacing
        from ._plot_widget import Component

        self.plot._grid_wdgs[Component.YLABEL].width_max = 2
        self.plot._grid_wdgs[Component.PAD_LEFT].width_max = 0
        # Start with a narrow y-axis (will grow as needed via update_yaxis_width)
        self.plot._yaxis_width = 14
        self.plot._grid_wdgs[Component.YAXIS].width_max = 14
        self._canvas.central_widget.add_widget(self.plot)
        self.node_tform = cast("scene.Node", self.plot).node_transform(
            self.plot._view.scene
        )

        self.plot._view.add(self._highlight)
        self._highlight.transform = scene.transforms.ChainTransform(
            scene.transforms.STTransform(),
            self._highlight_tform,
        )

        self._has_initial_range = False
        self._canvas.events.resize.connect(self._on_canvas_resize)

    # ------------ GraphicsCanvas methods ------------ #

    def refresh(self) -> None:
        self._canvas.update()

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
            self.plot.camera.set_range(x=x, y=y, margin=1e-30)
            self.plot.update_yaxis_width(y)

    def canvas_to_world(
        self, pos_xy: tuple[float, float]
    ) -> tuple[float, float, float]:
        return self.plot._view.scene.transform.imap(pos_xy)[:3]  # type: ignore[no-any-return]

    def elements_at(self, pos_xy: tuple[float, float]) -> list:
        raise NotImplementedError

    def set_visible(self, visible: bool) -> None: ...

    def close(self) -> None:
        self._disconnect_mouse_events()
        self._canvas.close()

    def frontend_widget(self) -> Any:
        return self._canvas.native

    # ------------ SharedHistogramCanvas methods ------------ #

    def set_channel_data(
        self, key: ChannelKey, counts: np.ndarray, bin_edges: np.ndarray
    ) -> None:
        ch = self._ensure_channel(key)
        ch.counts = counts
        ch.bin_edges = bin_edges
        # Downsample for display
        ch._display_centers, ch._display_counts = _downsample(counts, bin_edges)
        self._update_channel_area(key)
        if not self._has_initial_range:
            # First data: set both x and y range
            self._has_initial_range = True
            self._auto_range()
        else:
            # Subsequent data: only update y range, preserve x pan/zoom
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
        ch.area_mesh.visible = visible
        ch.outline.visible = visible
        ch.lut_line.visible = visible
        ch.gamma_handle.visible = visible
        ch.legend_text.visible = visible
        self._update_legend_positions()
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
        for visual in (
            ch.area_mesh,
            ch.outline,
            ch.lut_line,
            ch.gamma_handle,
            ch.legend_text,
        ):
            visual.parent = None
        self._update_legend_positions()
        self._auto_range()

    def set_channel_name(self, key: ChannelKey, name: str) -> None:
        ch = self._channels.get(key)
        if ch is None:
            return
        ch.name = name
        ch.legend_text.text = name
        self._update_legend_positions()

    def set_log_base(self, base: float | None) -> None:
        if base == self._log_base:
            return
        self._log_base = base
        # Re-render all channels
        for key in self._channels:
            self._update_channel_area(key)
        # Swap ticker
        count_axis = self.plot.yaxis
        if base is not None:
            count_axis.axis.ticker = LogTicker(count_axis.axis, base=base)
        else:
            from vispy.visuals.axis import Ticker

            count_axis.axis.ticker = Ticker(count_axis.axis)
        self._auto_range()

    def set_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        self._clim_bounds = bounds
        self.plot.camera.xbounds = bounds

    def highlight(self, value: float | None) -> None:
        self._highlight.visible = value is not None
        self._highlight_tform.translate = (value,)

    # ------------ Mouse interaction ------------ #

    def get_cursor(self, event: MouseMoveEvent) -> CursorType:
        pos = (event.x, event.y)
        _key, nearby = self._find_nearest_grabbable(pos)
        if nearby in (Grabbable.LEFT_CLIM, Grabbable.RIGHT_CLIM):
            return CursorType.H_ARROW
        elif nearby is Grabbable.GAMMA:
            return CursorType.V_ARROW
        else:
            x, y = self._to_plot_coords(pos)
            x1, x2 = self.plot.xaxis.axis.domain
            y1, y2 = self.plot.yaxis.axis.domain
            if (x1 < x <= x2) and (y1 <= y <= y2):
                return CursorType.ALL_ARROW
            return CursorType.DEFAULT

    def on_mouse_press(self, event: MousePressEvent) -> bool:
        pos = event.x, event.y
        self._grabbed_key, self._grabbed = self._find_nearest_grabbable(pos)
        if self._grabbed != Grabbable.NONE:
            self.plot.camera.interactive = False
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
        self.plot.camera.interactive = True
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
            c = self._to_plot_coords(pos)[0]
            # Clamp to bounds
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
            y = self._to_plot_coords(pos)[1]
            if y <= 0 or y > y_top:
                return False
            gamma = max(MIN_GAMMA, -np.log2(y / y_top)) if y_top != 0 else 1.0
            self.gammaChanged.emit(key, float(gamma))
            return False

        self.get_cursor(event).apply_to(self)
        return False

    # ------------ Private helpers ------------ #

    def _ensure_channel(self, key: object) -> _ChannelVisuals:
        """Get or create channel visuals."""
        if key in self._channels:
            return self._channels[key]

        area_mesh = scene.Mesh(color=(0.5, 0.5, 0.5, FILL_ALPHA))
        area_mesh.set_gl_state("translucent", depth_test=False)
        outline = scene.LinePlot(
            data=(0,),
            color="w",
            connect="strip",
            symbol=None,
            line_kind="-",
            width=1.2,
            marker_size=0,
        )
        lut_line = scene.LinePlot(
            data=(0,),
            color="w",
            connect="strip",
            symbol=None,
            line_kind="-",
            width=1.0,
            marker_size=0,
        )
        lut_line.visible = False
        lut_line.order = -1

        gamma_handle = scene.Markers(
            pos=np.array([[0, 0]]),
            size=6,
            edge_width=0,
        )
        gamma_handle.visible = False
        gamma_handle.order = -2

        legend_text = scene.Text(
            text="",
            color="w",
            font_size=8,
            anchor_x="right",
            anchor_y="top",
            parent=self._canvas.scene,
        )
        legend_text.order = -4

        self.plot._view.add(area_mesh)
        self.plot._view.add(outline)
        self.plot._view.add(lut_line)
        self.plot._view.add(gamma_handle)

        ch = _ChannelVisuals(
            area_mesh=area_mesh,
            outline=outline,
            lut_line=lut_line,
            gamma_handle=gamma_handle,
            legend_text=legend_text,
        )
        self._channels[key] = ch
        self._update_legend_positions()
        return ch

    def _apply_channel_colors(self, key: object) -> None:
        """Apply color to all visuals for a channel."""
        ch = self._channels[key]
        r, g, b = ch.color[:3]
        a = ch.color[3] if len(ch.color) > 3 else 1.0

        # Area fill: semi-transparent
        ch.area_mesh.color = (r, g, b, FILL_ALPHA)
        # Outline: full color
        ch.outline.set_data(
            ch.outline._line.pos,
            color=(r, g, b, a),
            marker_size=0,
        )
        # LUT line color
        ch.lut_line.set_data(
            ch.lut_line._line.pos,
            color=(r, g, b, LUT_LINE_ALPHA),
            marker_size=0,
        )
        # Gamma handle
        ch.gamma_handle.set_data(
            pos=ch.gamma_handle._data["a_position"][:1],
            face_color=(r, g, b, a),
            edge_width=0,
        )
        # Legend
        ch.legend_text.color = (r, g, b, a)

        # Re-render the area mesh with new color
        self._update_channel_area(key)
        self._update_lut_visuals(key)

    def _update_channel_area(self, key: object) -> None:
        """Re-render area fill + outline for a channel."""
        ch = self._channels.get(key)
        if ch is None or ch._display_centers is None or ch._display_counts is None:
            return

        counts = ch._display_counts
        centers = ch._display_centers

        if self._log_base:
            counts = np.log(counts + 1) / np.log(self._log_base)

        r, g, b = ch.color[:3]

        # Area fill mesh: triangle strip from centers to baseline
        verts, faces = _area_to_mesh(centers, counts)
        ch.area_mesh.set_data(vertices=verts, faces=faces, color=(r, g, b, FILL_ALPHA))
        ch.area_mesh._bounds_changed()

        # Outline
        outline_data = np.column_stack([centers, counts])
        ch.outline.set_data(outline_data, color=(r, g, b, 1.0), marker_size=0)
        ch.outline._bounds_changed()
        for v in ch.outline._subvisuals:
            v._bounds_changed()

    def _update_lut_visuals(self, key: object, npoints: int = 64) -> None:
        """Update clim lines and gamma curve for a channel."""
        ch = self._channels.get(key)
        if ch is None or ch.clims is None:
            return

        r, g, b = ch.color[:3]
        clims = ch.clims
        gamma = ch.gamma

        # Use global y range for consistent clim handle height
        y_range = self._compute_y_range()
        y_max = y_range[1] if y_range else 1.0
        if y_max == 0:
            y_max = 1.0

        # Build the LUT line: left clim line + gamma curve + right clim line
        # 2 points for each vertical clim line + npoints for gamma curve
        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)

        # Use 98% of y_max so handles aren't clipped at the camera edge
        y_top = y_max * 0.98

        # Left clim line (vertical, full height)
        X[0:2] = clims[0]
        Y[0:2] = (y_top, y_top * 0.5)
        # Gamma curve
        X[2:-2] = np.linspace(clims[0], clims[1], npoints)
        Y[2:-2] = np.linspace(0, 1, npoints) ** gamma * y_top
        # Right clim line (vertical, full height)
        X[-2:] = clims[1]
        Y[-2:] = (y_top * 0.5, 0)

        color = np.full((npoints + 4, 4), (r, g, b, LUT_LINE_ALPHA))
        ch.lut_line.set_data((X, Y), color=color, marker_size=0)
        ch.lut_line.visible = ch.visible
        ch.lut_line._bounds_changed()
        for v in ch.lut_line._subvisuals:
            v._bounds_changed()

        # Gamma handle at midpoint
        mid_x = np.mean(clims)
        mid_y = (2 ** (-gamma)) * y_top
        ch.gamma_handle.set_data(
            pos=np.array([[mid_x, mid_y]]),
            face_color=(r, g, b, 1.0),
            size=6,
            edge_width=0,
        )
        ch.gamma_handle.visible = ch.visible
        ch.gamma_handle._bounds_changed()

    def _channel_y_max(self, ch: _ChannelVisuals) -> float:
        """Get the max displayed count for a channel."""
        if ch._display_counts is None:
            return 1.0
        counts = ch._display_counts
        if self._log_base:
            counts = np.log(counts + 1) / np.log(self._log_base)
        return float(np.max(counts)) if len(counts) > 0 else 1.0

    def _compute_x_range(self) -> tuple[float, float] | None:
        """Compute x range across all visible channels."""
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
        """Auto-fit camera to encompass all visible data."""
        x = self._compute_x_range()
        y = self._compute_y_range()
        if x and y:
            self.plot.camera.set_range(x=x, y=y, margin=1e-30)
            self.plot.update_yaxis_width(y)
        self._refresh_all_lut_visuals()
        self._update_legend_positions()

    def _auto_range_y_only(self) -> None:
        """Update y range only, preserving current x pan/zoom."""
        y = self._compute_y_range()
        if y:
            camera_rect = self.plot.camera.rect
            self.plot.camera.set_range(
                x=(camera_rect.left, camera_rect.right), y=y, margin=1e-30
            )
            self.plot.update_yaxis_width(y)
        self._refresh_all_lut_visuals()
        self._update_legend_positions()

    def _refresh_all_lut_visuals(self) -> None:
        """Re-render clim/gamma visuals for all channels."""
        for key in self._channels:
            self._update_lut_visuals(key)

    def _on_canvas_resize(self, event: Any = None) -> None:
        self._update_legend_positions()

    def _update_legend_positions(self) -> None:
        """Position legend entries horizontally at the top-right."""
        # Build entries right-to-left so last channel is rightmost
        canvas_w = self._canvas.size[0]
        x_offset = canvas_w - 8
        for ch in reversed(list(self._channels.values())):
            if not ch.visible or not ch.name:
                ch.legend_text.visible = False
                continue
            ch.legend_text.visible = True
            ch.legend_text.text = f"● {ch.name}"
            ch.legend_text.pos = (x_offset, 14)
            x_offset -= len(ch.name) * 7 + 18  # approximate width

    def _find_nearest_grabbable(
        self, pos: tuple[float, float], tolerance: int = 5
    ) -> tuple[object, Grabbable]:
        """Find the nearest grabbable handle across all channels."""
        click_x, click_y = pos
        plot_to_canvas = self.node_tform.imap

        best_dist = float("inf")
        best_key: object = _NO_KEY
        best_grab = Grabbable.NONE

        y_range = self._compute_y_range()
        global_y_max = y_range[1] if y_range else 1.0

        for key, ch in self._channels.items():
            if not ch.visible or ch.clims is None:
                continue

            # Check clim lines
            left_cx = plot_to_canvas([ch.clims[0], 0])[0]
            right_cx = plot_to_canvas([ch.clims[1], 0])[0]

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

            # Check gamma handle
            mid_x = np.mean(ch.clims)
            mid_y = (2 ** (-ch.gamma)) * global_y_max * 0.98
            gx, gy = plot_to_canvas([mid_x, mid_y])[:2]
            d_gamma = ((gx - click_x) ** 2 + (gy - click_y) ** 2) ** 0.5
            if d_gamma < tolerance and d_gamma < best_dist:
                best_dist = d_gamma
                best_key = key
                best_grab = Grabbable.GAMMA

        return best_key, best_grab

    def _to_plot_coords(self, pos: Sequence[float]) -> tuple[float, float]:
        x, y = self.node_tform.map(pos)[:2]
        return x, y


def _downsample(
    counts: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample histogram to ~MAX_DISPLAY_BINS bins, return (centers, counts)."""
    n = len(counts)
    if n > MAX_DISPLAY_BINS:
        factor = n // MAX_DISPLAY_BINS
        trim = n - (n % factor)
        counts = counts[:trim].reshape(-1, factor).sum(axis=1)
        bin_edges = np.concatenate(
            [bin_edges[:trim:factor], bin_edges[trim : trim + 1]]
        )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, counts


def _area_to_mesh(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert area plot data to mesh vertices and faces (triangle strip)."""
    n = len(centers)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)

    # 2 vertices per point: one on the curve, one on the baseline
    vertices = np.zeros((2 * n, 3), np.float32)
    vertices[0::2, 0] = centers  # x: curve points
    vertices[0::2, 1] = counts  # y: curve points
    vertices[1::2, 0] = centers  # x: baseline points
    # vertices[1::2, 1] = 0  # y: baseline (already 0)

    # Triangle strip: for each pair of consecutive points, two triangles
    faces = np.zeros((2 * (n - 1), 3), np.uint32)
    for i in range(n - 1):
        top_left = 2 * i
        bot_left = 2 * i + 1
        top_right = 2 * (i + 1)
        bot_right = 2 * (i + 1) + 1
        faces[2 * i] = [top_left, bot_left, top_right]
        faces[2 * i + 1] = [bot_left, bot_right, top_right]

    return vertices, faces
