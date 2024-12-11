from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from ndv.models._array_display_model import ChannelMode

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from ndv._types import AxisKey
    from ndv.views.bases._lut_view import LutView


class ArrayView(ABC):
    """ABC for ND Array viewers widget.

    Currently, this is the "main" widget that contains the array display and
    all the controls for interacting with the array, includings sliders, LUTs,
    and histograms.
    """

    currentIndexChanged = Signal()
    resetZoomClicked = Signal()
    histogramRequested = Signal()
    channelModeChanged = Signal(ChannelMode)

    @abstractmethod
    def __init__(self, canvas_widget: Any, **kwargs: Any) -> None: ...
    @abstractmethod
    def create_sliders(self, coords: Mapping[int, Sequence]) -> None: ...
    @abstractmethod
    def current_index(self) -> Mapping[AxisKey, int | slice]: ...
    @abstractmethod
    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None: ...
    @abstractmethod
    def set_channel_mode(self, mode: ChannelMode) -> None: ...
    @abstractmethod
    def set_data_info(self, data_info: str) -> None: ...
    @abstractmethod
    def set_hover_info(self, hover_info: str) -> None: ...
    @abstractmethod
    def hide_sliders(
        self, axes_to_hide: Container[Hashable], *, show_remainder: bool = ...
    ) -> None: ...
    @abstractmethod
    def add_lut_view(self) -> LutView: ...
    @abstractmethod
    def remove_lut_view(self, view: LutView) -> None: ...
    @abstractmethod
    def set_visible(self, visible: bool) -> None: ...
    @abstractmethod
    def add_histogram(self, widget: Any) -> None: ...
    @abstractmethod
    def remove_histogram(self, widget: Any) -> None: ...
