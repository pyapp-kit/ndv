from typing import Any

import numpy as np

from ndv.views.bases._lut_view import LutView

from ._canvas import GraphicsCanvas


class HistogramCanvas(GraphicsCanvas, LutView):
    """A histogram-based view for LookUp Table (LUT) adjustment."""

    # TODO: Remove?
    def refresh(self) -> None: ...

    def frontend_widget(self) -> Any:
        """Returns an object understood by the widget frontend."""

    def set_vertical(self, vertical: bool) -> None:
        """Sets the axis of the domain.

        Properties
        ----------
        vertical : bool
            If true, views the domain along the y axis and the range along the x
            axis. If false, views the domain along the x axis and the range along
            the y axis.
        """

    def set_range_log(self, enabled: bool) -> None:
        """Sets the axis scale of the range.

        Properties
        ----------
        enabled : bool
            If true, the range will be displayed with a logarithmic (base 10)
            scale. If false, the range will be displayed with a linear scale.
        """

    def set_data(self, values: np.ndarray, bin_edges: np.ndarray) -> None:
        """Sets the histogram data.

        Properties
        ----------
        values : np.ndarray
            The histogram values.
        bin_edges : np.ndarray
            The bin edges of the histogram.
        """
