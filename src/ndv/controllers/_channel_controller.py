from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from psygnal import Signal

from ndv.controllers._image_stats import ImageStats, compute_image_stats

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np

    from ndv._types import ChannelKey
    from ndv.models._lut_model import LUTModel
    from ndv.views.bases import LUTView
    from ndv.views.bases._graphics._canvas_elements import ImageHandle


class ChannelController:
    """Controller for a single channel in the viewer.

    This manages the connection between the LUT model (settings like colormap,
    contrast limits and visibility) and the LUT view (the front-end widget that
    allows the user to interact with these settings), as well as the image handle
    that displays the data, all for a single "channel" extracted from the data.
    """

    stats_updated = Signal(ImageStats)

    @property
    def needs_histogram(self) -> bool:
        """Whether any listener needs histogram data."""
        return len(self.stats_updated) > 0

    def __init__(
        self, key: ChannelKey, lut_model: LUTModel, views: Sequence[LUTView]
    ) -> None:
        self.key = key
        self.lut_views: list[LUTView] = []
        self.lut_model = lut_model
        self.lut_model.events.clims.connect(self._auto_scale)
        self.handles: list[ImageHandle] = []
        self._last_clims: tuple[float, float] | None = None

        for v in views:
            self.add_lut_view(v)

    def add_lut_view(self, view: LUTView) -> None:
        """Add a LUT view to the controller."""
        view.model = self.lut_model
        self.lut_views.append(view)
        if self._last_clims is not None:
            view.set_clims(self._last_clims)

    def synchronize(self, *views: LUTView) -> None:
        """Aligns all views against the backing model."""
        _views: Iterable[LUTView] = views or self.lut_views
        for view in _views:
            view.synchronize()

    def update_texture_data(
        self,
        data: np.ndarray,
        *,
        need_histogram: bool = False,
        significant_bits: int | None = None,
    ) -> ImageStats | None:
        """Update the data in the image handle and compute stats."""
        # WIP:
        # until we have a more sophisticated way to handle updating data
        # for multiple handles, we'll just update the first one
        if not (handles := self.handles):
            return None
        handles[0].set_data(data)
        need_histogram = need_histogram or self.needs_histogram
        stats = compute_image_stats(
            data,
            self.lut_model.clims,
            need_histogram=need_histogram,
            significant_bits=significant_bits,
        )
        self._set_clims(stats.clims)
        if self.needs_histogram:
            self.stats_updated.emit(stats)
        return stats

    def add_handle(self, handle: ImageHandle) -> None:
        """Add an image texture handle to the controller."""
        self.handles.append(handle)
        self.add_lut_view(handle)

    def get_value_at_index(self, idx: tuple[int, ...]) -> np.ndarray | float | None:
        """Get the value of the data at the given index."""
        if not (handles := self.handles):
            return None
        # only getting one handle per channel for now
        handle = handles[0]
        if not handle.visible():
            return None
        with suppress(IndexError):  # skip out of bounds
            # here, we're retrieving the value from the in-memory data
            # stored by the backend visual, rather than querying the data itself
            # this is a quick workaround to get the value without having to
            # worry about other dimensions in the data source (since the
            # texture has already been reduced to RGB/RGBA/2D). But a more complete
            # implementation would gather the full current nD index and query
            # the data source directly.
            return handle.data()[idx]  # type: ignore [no-any-return]
        return None

    def _set_clims(self, clims: tuple[float, float]) -> None:
        self._last_clims = clims
        for view in self.lut_views:
            view.set_clims(clims)

    def _auto_scale(self) -> None:
        if self.lut_model and self.handles:
            policy = self.lut_model.clims
            all_clims = [
                compute_image_stats(h.data(), policy, need_histogram=False).clims
                for h in self.handles
            ]
            mi = min(c[0] for c in all_clims)
            ma = max(c[1] for c in all_clims)
            self._set_clims((mi, ma))
