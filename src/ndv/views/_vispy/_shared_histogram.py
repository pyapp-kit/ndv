from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from vispy import scene, visuals

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

from ._plot_widget import LogTicker, PlotWidget

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ndv._types import (
        ChannelKey,
        MouseMoveEvent,
        MousePressEvent,
        MouseReleaseEvent,
    )

FILL_ALPHA = 0.3


@dataclass
class _ChannelVisuals:
    """All visuals for a single channel on the shared histogram."""

    area_mesh: scene.Mesh  # pyright: ignore[reportInvalidTypeForm]
    outline: scene.LinePlot  # pyright: ignore[reportInvalidTypeForm]
    lut_line: scene.LinePlot  # pyright: ignore[reportInvalidTypeForm]
    gamma_handle: scene.Markers  # pyright: ignore[reportInvalidTypeForm]
    legend_text: scene.Text  # pyright: ignore[reportInvalidTypeForm]
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

        # Per-channel highlight lines (created on demand)
        self._highlight_lines: dict[object, visuals.LineVisual] = {}
        self._highlight_unit_pos = np.array([[0, 0], [0, 1]], dtype=np.float32)

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

        self._has_initial_range = False
        self._redownsampling = False
        self._canvas.events.resize.connect(self._on_canvas_resize)
        self._canvas.events.draw.connect(self._on_draw)
        self._last_cam_rect: tuple[float, float] = (0.0, 0.0)  # (left, right)

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
        if (hl := self._highlight_lines.pop(key, None)) is not None:
            hl.parent = None
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
        self._auto_range_y_only()

    def set_clim_bounds(self, bounds: tuple[float | None, float | None]) -> None:
        self._clim_bounds = bounds
        self.plot.camera.xbounds = bounds

    def highlight(self, channel_values: dict[object, float]) -> None:
        y_range = self._compute_y_range()
        y_scale = y_range[1] * 0.5 if y_range else 1.0
        for key, line in self._highlight_lines.items():
            if key not in channel_values:
                line.visible = False
        for key, value in channel_values.items():
            if (line := self._highlight_lines.get(key)) is None:
                ch = self._channels.get(key)
                color = (*ch.color[:3], 0.5) if ch else (1, 1, 0.2, 0.5)
                line = scene.Line(pos=self._highlight_unit_pos, color=color, width=1)
                self.plot._view.add(line)
                line.transform = scene.transforms.STTransform()
                self._highlight_lines[key] = line
            line.visible = True
            line.transform.translate = (value, 0, 0, 0)
            line.transform.scale = (1, y_scale, 1, 1)

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
            x1, x2 = cast("tuple[float, float]", self.plot.xaxis.axis.domain)
            y1, y2 = cast("tuple[float, float]", self.plot.yaxis.axis.domain)
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
            new_clims = clamp_clim_drag(self._grabbed, c, ch.clims, self._clim_bounds)
            self.climsChanged.emit(key, new_clims)
            return False

        if self._grabbed is Grabbable.GAMMA:
            y = self._to_plot_coords(pos)[1]
            gamma = gamma_from_mouse_y(y, self._compute_y_range())
            if gamma is None:
                return False
            self.gammaChanged.emit(key, gamma)
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
        ch.legend_text.color = (r, g, b, a)
        # Re-render area and LUT visuals with new color
        self._update_channel_area(key)
        self._update_lut_visuals(key)

    def _update_channel_area(self, key: object) -> None:
        """Re-render area fill + outline for a channel."""
        ch = self._channels.get(key)
        if ch is None or ch.counts is None or ch.bin_edges is None:
            return

        canvas_w = max(self._canvas.size[0], 64)
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

        r, g, b = ch.color[:3]

        # Area fill mesh: triangle strip from centers to baseline
        verts, faces = area_to_mesh(centers, counts)
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
        y_top = y_top_from_range(self._compute_y_range())

        # Build the LUT line: left clim line + gamma curve + right clim line
        # 2 points for each vertical clim line + npoints for gamma curve
        X = np.empty(npoints + 4)
        Y = np.empty(npoints + 4)

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
        mid_x, mid_y = gamma_handle_pos(clims, gamma, y_top)
        ch.gamma_handle.set_data(
            pos=np.array([[mid_x, mid_y]]),
            face_color=(r, g, b, 1.0),
            size=6,
            edge_width=0,
        )
        ch.gamma_handle.visible = ch.visible
        ch.gamma_handle._bounds_changed()

    def _compute_x_range(self) -> tuple[float, float] | None:
        return compute_x_range(self._channels)

    def _compute_y_range(self) -> tuple[float, float] | None:
        return compute_y_range(self._channels, self._log_base)

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

    def _visible_x_range(self) -> tuple[float, float] | None:
        """Get the currently visible x-range from the camera."""
        if not self._has_initial_range:
            return None
        r = self.plot.camera.rect
        return (r.left, r.right)

    def _redownsample_all(self) -> None:
        """Re-downsample all channels for the current visible range."""
        for key in self._channels:
            self._update_channel_area(key)
        # Refit y-axis to the visible data
        y = self._compute_y_range()
        if y:
            camera_rect = self.plot.camera.rect
            self.plot.camera.set_range(
                x=(camera_rect.left, camera_rect.right), y=y, margin=1e-30
            )
            self.plot.update_yaxis_width(y)
        self._refresh_all_lut_visuals()

    def _refresh_all_lut_visuals(self) -> None:
        """Re-render clim/gamma visuals for all channels."""
        for key in self._channels:
            self._update_lut_visuals(key)

    def _on_draw(self, event: Any = None) -> None:
        """Re-downsample when camera pans/zooms.

        Guard against an infinite draw loop caused by vispy's set_range
        adding a tiny margin on every call (https://github.com/vispy/vispy/issues/1483).
        We use margin=1e-30 to avoid the 0.1 fallback, but each set_range
        still shifts the rect by ~1e-28. Without the guard, this creates:
        _on_draw -> _redownsample_all -> set_range (shifts rect) -> draw -> ...
        The _redownsampling flag blocks synchronous re-entrant draws (wx),
        and re-reading the rect after redownsampling absorbs the drift so
        deferred draws (Qt) don't see it as a change.
        """
        if self._redownsampling:
            return
        r = self.plot.camera.rect
        cam_rect = (r.left, r.right)
        if cam_rect != self._last_cam_rect:
            self._last_cam_rect = cam_rect
            self._redownsampling = True
            try:
                self._redownsample_all()
            finally:
                r = self.plot.camera.rect
                self._last_cam_rect = (r.left, r.right)
                self._redownsampling = False

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
        imap = self.node_tform.imap
        return find_nearest_grabbable(
            self._channels,
            pos,
            lambda x, y: tuple(imap([x, y])[:2]),
            self._compute_y_range(),
            tolerance,
        )

    def _to_plot_coords(self, pos: Sequence[float]) -> tuple[float, float]:
        x, y = self.node_tform.map(pos)[:2]
        return x, y
