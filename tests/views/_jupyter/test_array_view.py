from __future__ import annotations

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
