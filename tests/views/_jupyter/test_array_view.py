from __future__ import annotations

import ipywidgets
from pytest import fixture

from ndv.models._data_display_model import _ArrayDataDisplayModel
from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._jupyter._array_view import JupyterArrayView
from ndv.views.bases._array_view import ArrayViewOptions


@fixture
def viewer() -> JupyterArrayView:
    return JupyterArrayView(
        ipywidgets.DOMWidget(), _ArrayDataDisplayModel(), ArrayViewerModel()
    )


def test_array_options(viewer: JupyterArrayView) -> None:
    assert viewer._ndims_btn.layout.display is None
    assert viewer._reset_zoom_btn.layout.display is None
    assert viewer._channel_mode_combo.layout.display is None
    assert viewer._add_roi_btn.layout.display is None

    options = ArrayViewOptions(
        show_3d_button=False,
        show_channel_mode_selector=False,
        show_histogram_button=False,
        show_reset_zoom_button=False,
        show_roi_button=False,
    )
    viewer.set_options(options)

    assert viewer._ndims_btn.layout.display == "none"
    assert viewer._reset_zoom_btn.layout.display == "none"
    assert viewer._channel_mode_combo.layout.display == "none"
    assert viewer._add_roi_btn.layout.display == "none"
