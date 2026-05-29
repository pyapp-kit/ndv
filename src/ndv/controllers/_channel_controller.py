from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psygnal import Signal

from ndv.controllers._image_stats import ImageStats, compute_image_stats
from ndv.views.bases import LUTView

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import cmap
    import numpy as np
    import scenex as snx

    from ndv._types import ChannelKey
    from ndv.models._lut_model import ClimPolicy, LUTModel


class ChannelController:
    """Controller for a single channel in the viewer.

    This manages the connection between the LUT model (settings like colormap,
    contrast limits and visibility) and the LUT view (the front-end widget that
    allows the user to interact with these settings), as well as the image handle
    that displays the data, all for a single "channel" extracted from the data.
    """

    stats_updated = Signal(ImageStats)
    clims_resolved = Signal(tuple)

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
        self.handles: list[SnxLUTView] = []
        self._last_clims: tuple[float, float] | None = None

        for v in views:
            self.add_lut_view(v)

    def clear_channel_data(self) -> None:
        """Clear the image/volume handles associated with this channel."""
        new_lut_views: list[LUTView] = []
        for view in self.lut_views:
            if isinstance(view, SnxLUTView):
                view.close()
            else:
                new_lut_views.append(view)
        self.lut_views = new_lut_views
        self.handles = []

    def add_lut_view(self, view: LUTView) -> None:
        """Add a LUT view to the controller."""
        view.model = self.lut_model
        self.lut_views.append(view)
        if self._last_clims is not None:
            view.set_clims(self._last_clims)

    def synchronize(self, *views: LUTView) -> None:
        """Aligns all views against the backing model."""
        _views: Iterable[LUTView] = views or self.lut_views
        name = str(self.key) if self.key is not None else ""
        for view in _views:
            view.synchronize()
            view.set_channel_name(name)

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
        handles[0].img.data = data
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

    def add_image(
        self,
        image: snx.Image,
        *,
        need_histogram: bool = False,
        significant_bits: int | None = None,
    ) -> ImageStats | None:
        """Add an image (or volume) texture handle to the controller."""
        handle = SnxLUTView(image)
        self.handles.append(handle)
        self.add_lut_view(handle)

        stats = compute_image_stats(
            image.data,
            self.lut_model.clims,
            need_histogram=need_histogram,
            significant_bits=significant_bits,
        )
        self._set_clims(stats.clims)
        return stats

    def _auto_scale(self) -> None:
        if self.lut_model and len(self.handles):
            policy = self.lut_model.clims
            all_clims = [
                compute_image_stats(h.img.data, policy, need_histogram=False).clims
                for h in self.handles
            ]
            mi = min(c[0] for c in all_clims)
            ma = max(c[1] for c in all_clims)
            self._set_clims((mi, ma))

    def _set_clims(self, clims: tuple[float, float]) -> None:
        self._last_clims = clims
        for view in self.lut_views:
            view.set_clims(clims)
        self.clims_resolved.emit(clims)


class SnxLUTView(LUTView):
    def __init__(self, img: snx.Image) -> None:
        self.img = img

    def close(self) -> None:
        self.img.parent = None

    def frontend_widget(self) -> Any:
        return None

    def set_channel_name(self, name: str) -> None:
        self.img.name = name

    def set_clim_policy(self, policy: ClimPolicy) -> None:
        pass

    def set_channel_visible(self, visible: bool) -> None:
        self.set_visible(visible)

    def set_clims(self, clims: tuple[float, float]) -> None:
        self.img.clims = clims

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        self.img.cmap = cmap

    def set_visible(self, visible: bool) -> None:
        self.img.visible = visible

    def set_gamma(self, gamma: float) -> None:
        # These bounds coerce the gamma into the range allowed by scenex
        self.img.gamma = max(1e-6, min(gamma, 2))
