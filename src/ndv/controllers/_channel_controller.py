from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

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

    def __init__(
        self, key: ChannelKey, lut_model: LUTModel, views: Sequence[LUTView]
    ) -> None:
        self.key = key
        self.lut_views: list[LUTView] = []
        self.lut_model = lut_model
        self.lut_model.events.clims.connect(self._auto_scale)
        self.handles: list[ImageHandle] = []
        self.significant_bits: int | None = None

        for v in views:
            self.add_lut_view(v)

    def add_lut_view(self, view: LUTView) -> None:
        """Add a LUT view to the controller."""
        view.model = self.lut_model
        self.lut_views.append(view)
        # TODO: Could probably reuse cached clims
        self._auto_scale()

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
        if not (handles := self.handles):
            return None
        handles[0].set_data(data)
        stats = compute_image_stats(
            data, self.lut_model.clims, need_histogram, significant_bits
        )
        self.lut_model.clims._cached_clims = stats.clims
        for view in self.lut_views:
            view.set_clims(stats.clims)
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

    def _auto_scale(self) -> None:
        if self.lut_model and len(self.handles):
            policy = self.lut_model.clims
            handle_clims = [policy.calc_clims(handle.data()) for handle in self.handles]
            mi, ma = handle_clims[0]
            for clims in handle_clims[1:]:
                mi = min(mi, clims[0])
                ma = max(ma, clims[1])

            for view in self.lut_views:
                view.set_clims((mi, ma))
