from __future__ import annotations

from unittest.mock import Mock

import ipywidgets
from pytest import fixture

from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._jupyter._array_view import JupyterArrayView


@fixture
def viewer() -> JupyterArrayView:
    viewer = JupyterArrayView(
        ipywidgets.DOMWidget(), _ArrayDataDisplayModel(), ArrayViewerModel()
    )
    viewer.add_lut_view(None)
    return viewer


def test_array_options(viewer: JupyterArrayView) -> None:
    lut = viewer._luts[None]

    assert viewer._ndims_btn.layout.display is None
    viewer._viewer_model.show_3d_button = False
    assert viewer._ndims_btn.layout.display == "none"

    assert lut._histogram.layout.display is None
    viewer._viewer_model.show_histogram_button = False
    assert lut._histogram.layout.display == "none"

    assert viewer._reset_zoom_btn.layout.display is None
    viewer._viewer_model.show_reset_zoom_button = False
    assert viewer._reset_zoom_btn.layout.display == "none"

    assert viewer._channel_mode_combo.layout.display is None
    viewer._viewer_model.show_channel_mode_selector = False
    assert viewer._channel_mode_combo.layout.display == "none"

    assert viewer._add_roi_btn.layout.display is None
    viewer._viewer_model.show_roi_button = False
    assert viewer._add_roi_btn.layout.display == "none"


def test_histogram(viewer: JupyterArrayView) -> None:
    channel = None
    lut = viewer._luts[channel]

    # Ensure lut signal gets passed through the viewer with the channel as the arg
    histogram_mock = Mock()
    viewer.histogramRequested.connect(histogram_mock)
    lut._histogram.value = True
    histogram_mock.assert_called_once_with(channel)

    # FIXME: Throws event loop errors
    # histogram = get_histogram_canvas_class()()  # will raise if not supported
    # histogram_wdg = histogram.frontend_widget()
    # viewer.add_histogram(channel, histogram_wdg)
