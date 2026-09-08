from __future__ import annotations

from unittest.mock import Mock

from pytest import fixture

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._jupyter._array_view import JupyterArrayView


@fixture
def viewer() -> JupyterArrayView:
    canvas = Mock()
    canvas.model_id = "test-canvas-id"
    viewer = JupyterArrayView(canvas, ArrayViewerModel())
    viewer.add_lut_view(None)
    return viewer


def test_viewer_model_flags(viewer: JupyterArrayView) -> None:
    w = viewer._widget

    assert w.show_3d_button is True
    viewer._viewer_model.show_3d_button = False
    assert w.show_3d_button is False

    assert w.show_histogram_button is True
    viewer._viewer_model.show_histogram_button = False
    assert w.show_histogram_button is False

    assert w.show_reset_zoom_button is True
    viewer._viewer_model.show_reset_zoom_button = False
    assert w.show_reset_zoom_button is False

    assert w.show_channel_mode_selector is True
    viewer._viewer_model.show_channel_mode_selector = False
    assert w.show_channel_mode_selector is False

    assert w.show_roi_button is False
    viewer._viewer_model.show_roi_button = True
    assert w.show_roi_button is True


def test_sliders(viewer: JupyterArrayView) -> None:
    viewer.create_sliders({"z": range(10), "t": range(20)})
    assert len(viewer._widget.sliders) == 2
    assert viewer.current_index() == {"z": 0, "t": 0}

    viewer.set_current_index({"z": 5})
    assert viewer.current_index()["z"] == 5
    # Check widget state is updated
    z_slider = next(s for s in viewer._widget.sliders if s["axis"] == "z")
    assert z_slider["value"] == 5


def test_hide_sliders(viewer: JupyterArrayView) -> None:
    viewer.create_sliders({"z": range(10), "t": range(20)})
    viewer.hide_sliders({"z"})
    z_slider = next(s for s in viewer._widget.sliders if s["axis"] == "z")
    t_slider = next(s for s in viewer._widget.sliders if s["axis"] == "t")
    assert z_slider["visible"] is False
    assert t_slider["visible"] is True


def test_lut_view(viewer: JupyterArrayView) -> None:
    lut = viewer._luts[None]
    assert len(viewer._widget.luts) == 1
    assert viewer._widget.luts[0]["key"] == "None"

    lut.set_channel_name("DAPI")
    assert viewer._widget.luts[0]["name"] == "DAPI"

    lut.set_clims((100.0, 500.0))
    assert viewer._widget.luts[0]["clim_min"] == 100.0
    assert viewer._widget.luts[0]["clim_max"] == 500.0

    lut.set_channel_visible(False)
    assert viewer._widget.luts[0]["visible"] is False

    viewer.remove_lut_view(lut)
    assert len(viewer._widget.luts) == 0


def test_channel_mode(viewer: JupyterArrayView) -> None:
    from ndv.models._array_display_model import ChannelMode

    mock = Mock()
    viewer.channelModeChanged.connect(mock)

    # Controller sets mode — should NOT emit signal
    viewer.set_channel_mode(ChannelMode.COMPOSITE)
    assert viewer._widget.channel_mode == ChannelMode.COMPOSITE.value
    mock.assert_not_called()


def test_command_handler(viewer: JupyterArrayView) -> None:
    viewer.create_sliders({"z": range(10)})

    mock = Mock()
    viewer.currentIndexChanged.connect(mock)

    # Simulate JS slider event via _js_event field
    viewer._widget._js_event = {"type": "slider_changed", "axis": "z", "value": 7}
    assert viewer.current_index()["z"] == 7
    mock.assert_called_once()


def test_find_rfb() -> None:
    from jupyter_rfb import RemoteFrameBuffer

    from ndv.views._jupyter._app import _find_rfb

    class FakeRFB(RemoteFrameBuffer):
        pass

    rfb = FakeRFB()

    # Direct match
    assert _find_rfb(rfb) is rfb

    # Via _canvas_ref attribute
    widget = Mock()
    widget._canvas_ref = rfb
    widget.children = ()
    assert _find_rfb(widget) is rfb

    # Not found
    widget2 = Mock(spec=[])
    assert _find_rfb(widget2) is None
