# /// script
# dependencies = [
#     "ndv[pyqt,vispy]",
#     "imageio[tifffile]",
#     "pyqtgraph",
# ]
#
# [tool.uv.sources]
# ndv = { path = "../", editable = true }
# ///
"""Example: custom histogram widget using ndv's stats_updated signal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import QHBoxLayout, QWidget

import ndv

if TYPE_CHECKING:
    from ndv.models import LUTModel


class CustomHistogramWidget(QWidget):
    """A pyqtgraph histogram that updates from ndv's stats signal."""

    def __init__(self, viewer: ndv.ArrayViewer) -> None:
        super().__init__()
        self._viewer = viewer
        self._curves: dict[object, pg.PlotDataItem] = {}
        self._plot = pg.PlotWidget(title="Custom Histogram")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

        # Track colormap changes on existing and future LUT models
        for key, lut in viewer.display_model.luts.items():
            self._watch_lut(key, lut)
        viewer.display_model.luts.item_added.connect(self._watch_lut)

    def _channel_color(self, key: object) -> tuple:
        """Get the RGBA color for a channel from the viewer's LUT model."""
        luts = self._viewer.display_model.luts
        lut = luts.get(key, self._viewer.display_model.default_lut)
        return lut.cmap.to_pyqtgraph().getColors()[-1]

    def _update_curve_color(self, key: object) -> None:
        if curve := self._curves.get(key):
            color = self._channel_color(key)
            curve.setPen(pg.mkPen(color, width=2))
            curve.setFillBrush(pg.mkBrush(*color[:3], 64))

    def _watch_lut(self, key: Any, lut: LUTModel) -> None:
        lut.events.cmap.connect(lambda *_: self._update_curve_color(key))

    def on_stats(self, key: object, stats: ndv.ImageStats) -> None:
        if stats.counts is None or stats.bin_edges is None:
            return

        counts = stats.counts
        edges = stats.bin_edges
        # Downsample to ~256 bins for fast rendering
        n = len(counts)
        if n > 256:
            factor = n // 256
            trim = n - (n % factor)
            counts = counts[:trim].reshape(-1, factor).sum(axis=1)
            edges = np.concatenate([edges[:trim:factor], edges[trim : trim + 1]])

        centers = (edges[:-1] + edges[1:]) / 2
        color = self._channel_color(key)

        if key in self._curves:
            self._curves[key].setData(centers, counts)
        else:
            self._curves[key] = self._plot.plot(
                centers,
                counts,
                pen=pg.mkPen(color, width=2),
                fillLevel=0.5,
                fillBrush=pg.mkBrush(*color[:3], 64),
                name=f"ch {key}",
            )


# --- Setup ---

try:
    img = ndv.data.cells3d()
except Exception:
    img = ndv.data.nd_sine_wave((10, 3, 8, 512, 512))

viewer = ndv.ArrayViewer(
    img,
    current_index={0: 30},
    channel_mode="composite",
    luts={0: {"name": "FITC"}, 1: {"name": "DAPI", "cmap": "magenta"}},
    scales={0: 0.4, -1: 0.2, -2: 0.2},
)

hist_widget = CustomHistogramWidget(viewer)
viewer.stats_updated.connect(hist_widget.on_stats)
viewer.refresh_stats()

container = QWidget()
layout = QHBoxLayout(container)
layout.addWidget(viewer.widget())
layout.addWidget(hist_widget)
container.resize(1200, 600)
container.show()

ndv.run_app()
