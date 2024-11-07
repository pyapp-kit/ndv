"""View interfaces for data display."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cmap
from psygnal import Signal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


class StatsView(Protocol):
    """A view of the statistics of a dataset."""

    def set_histogram(self, values: Sequence[int], bin_edges: Sequence[float]) -> None:
        """Defines the distribution of the dataset.

        Properties
        ----------
        values : Sequence[int]
            A length (n) sequence of values representing clustered counts of data
            points. values[i] defines the number of data points falling between
            bin_edges[i] and bin_edges[i+1].
        bin_edges : Sequence[float]
            A length (n+1) sequence of values defining the intervals partitioning
            all data points. Must be non-decreasing.
        """
        ...

    def set_std_dev(self, std_dev: float) -> None:
        """Defines the standard deviation of the dataset.

        Properties
        ----------
        std_dev : float
            The standard deviation.
        """
        ...

    def set_average(self, avg: float) -> None:
        """Defines the average value of the dataset.

        Properties
        ----------
        std_dev : float
            The average value of the dataset.
        """
        ...

    def view(self) -> Any:
        """The native object that can be displayed."""
        ...


class LutView(Protocol):
    """An (interactive) view of a LookUp Table (LUT)."""

    cmapChanged: Signal = Signal(cmap.Colormap)
    gammaChanged: Signal = Signal(float)
    climsChanged: Signal = Signal(tuple[float, float])
    autoscaleChanged: Signal = Signal(object)

    def set_visibility(self, visible: bool) -> None:
        """Defines whether this view is visible.

        Properties
        ----------
        visible : bool
            True iff the view should be visible.
        """
        ...

    def set_cmap(self, lut: cmap.Colormap) -> None:
        """Defines the colormap backing the view.

        Properties
        ----------
        lut : cmap.Colormap
            The object mapping scalar values to RGB(A) colors.
        """
        ...

    def set_gamma(self, gamma: float) -> None:
        """Defines the exponent used for gamma correction.

        Properties
        ----------
        gamma : float
            The exponent used for gamma correction
        """
        ...

    def set_clims(self, clims: tuple[float, float]) -> None:
        """Defines the input clims.

        The contrast limits (clims) are the input values mapped to the minimum and
        maximum (respectively) of the LUT.

        Properties
        ----------
        clims : tuple[float, float]
            The clims
        """
        ...

    def set_autoscale(self, autoscale: bool | tuple[float, float]) -> None:
        """Defines whether autoscale has been enabled.

        Autoscale defines whether the contrast limits (clims) are adjusted when the
        data changes.

        Properties
        ----------
        autoscale : bool | tuple[float, float]
            If a boolean, true iff clims automatically changed on dataset alteration.
            If a tuple, indicated that clims automatically changed. Values denote
            the fraction of the dataset located below and above the lower and
            upper clims, respectively.
        """
        ...

    def view(self) -> Any:
        """The native object that can be displayed."""
        ...


class HistogramView(StatsView, LutView):
    """A histogram-based view for LookUp Table (LUT) adjustment."""

    def set_domain(self, bounds: tuple[float, float] | None) -> None:
        """Sets the domain of the view.

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the x axis to the passed
            values. If None, sets them to the extent of the data instead.
        """
        ...

    def set_range(self, bounds: tuple[float, float] | None) -> None:
        """Sets the range of the view.

        Properties
        ----------
        bounds : tuple[float, float] | None
            If a tuple, sets the displayed extremes of the y axis to the passed
            values. If None, sets them to the extent of the data instead.
        """
        ...

    def set_vertical(self, vertical: bool) -> None:
        """Sets the axis of the domain.

        Properties
        ----------
        vertical : bool
            If true, views the domain along the y axis and the range along the x
            axis. If false, views the domain along the x axis and the range along
            the y axis.
        """
        ...

    def set_range_log(
        self,
        enabled: bool,
    ) -> None:
        """Sets the axis scale of the range.

        Properties
        ----------
        enabled : bool
            If true, the range will be displayed with a logarithmic (base 10)
            scale. If false, the range will be displayed with a linear scale.
        """
        ...
