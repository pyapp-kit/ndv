from typing import Any, Protocol

import cmap
import numpy as np
from psygnal import Signal


class StatsView(Protocol):
    """A view for data statistics."""

    def set_histogram(self, values: np.ndarray, bin_edges: np.ndarray) -> None: ...
    def set_std_dev(self, std_dev: float) -> None: ...
    def set_average(self, avg: float) -> None: ...
    def view(self) -> Any:
        """Returns some object a backend knows how to display."""
        # The thought is that this would return the QWidget
        # TODO: Maybe have generics?
        ...


class LutView(Protocol):
    """A view for LUT parameters."""

    cmapChanged: Signal = Signal(cmap.Colormap)
    gammaChanged: Signal = Signal(float)
    climsChanged: Signal = Signal(tuple[float, float])
    autoscaleChanged: Signal = Signal(object)

    def set_visibility(self, visible: bool) -> None: ...
    def set_cmap(self, lut: cmap.Colormap) -> None: ...
    def set_gamma(self, gamma: float) -> None: ...
    def set_clims(self, clims: tuple[float, float]) -> None: ...
    def set_autoscale(self, autoscale: bool | tuple[float, float]) -> None: ...
    def view(self) -> Any:
        """Returns some object a backend knows how to display."""
        # The thought is that this would return the QWidget
        # TODO: Maybe have generics?
        ...


class HistogramView(StatsView, LutView):
    """A histogram-based view ."""

    def set_domain(self, domain: tuple[float, float] | None) -> None:
        """Sets the domain of the view.

        If a tuple, sets the displayed extremes of the x axis to the passed values.
        If None, sets them to the extent of the data instead.
        """
        ...

    def set_range(self, range: tuple[float, float] | None) -> None:
        """Sets the range of the view.

        If a tuple, sets the displayed extremes of the y axis to the passed values.
        If None, sets them to the extent of the data instead.
        """
        ...

    def set_vertical(self, vertical: bool) -> None:
        """Sets the axis of the domain.

        If true, views the domain along the y axis and the range along the x axis.
        Otherwise views the domain along the x axis and the range along the y axis.
        """
        ...

    def enable_range_log(self, enabled: bool) -> None:
        """Sets the axis scale of the range.

        If true, the range will be displayed with a logarithmic (base 10) scale.
        If false, the range will be displayed with a linear scale.
        """
        ...
