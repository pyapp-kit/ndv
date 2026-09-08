from __future__ import annotations

import cmap
from pytest import fixture

from ndv.models._lut_model import ClimsManual, ClimsPercentile, LUTModel
from ndv.views._jupyter._array_view import JupyterLUTView, NdvWidgetState


@fixture
def parent() -> NdvWidgetState:
    return NdvWidgetState()


@fixture
def model() -> LUTModel:
    return LUTModel()


@fixture
def view(parent: NdvWidgetState, model: LUTModel) -> JupyterLUTView:
    view = JupyterLUTView(parent, channel=0)
    # Add initial LUT entry to parent state
    parent.luts = [
        {
            "key": "0",
            "name": "0",
            "visible": True,
            "cmap_name": "gray",
            "cmap_colors": [],
            "cmap_options": ["gray", "green", "magenta"],
            "clim_min": 0,
            "clim_max": 65535,
            "clim_bound_min": 0,
            "clim_bound_max": 65535,
            "auto_clim": True,
            "auto_lower_tail": 0,
            "auto_upper_tail": 0,
            "gamma": 1.0,
            "show_histogram_btn": True,
            "show_cmap": True,
            "row_visible": True,
        }
    ]
    view.model = model
    return view


def _get_lut(parent: NdvWidgetState) -> dict:
    """Get the first LUT dict from parent state."""
    return parent.luts[0]


def test_model_to_view(
    parent: NdvWidgetState, model: LUTModel, view: JupyterLUTView
) -> None:
    """Model changes propagate to the widget state dict."""
    # clim policy
    model.clims = ClimsManual(min=100, max=500)
    assert not _get_lut(parent)["auto_clim"]

    model.clims = ClimsPercentile(min_percentile=1, max_percentile=99)
    lut = _get_lut(parent)
    assert lut["auto_clim"]
    assert lut["auto_lower_tail"] == 1
    assert lut["auto_upper_tail"] == 1

    # visibility
    model.visible = False
    assert not _get_lut(parent)["visible"]
    model.visible = True
    assert _get_lut(parent)["visible"]

    # colormap
    new_cmap = cmap.Colormap("red")
    model.cmap = new_cmap
    assert _get_lut(parent)["cmap_name"] == new_cmap.name.split(":")[-1]

    # clim bounds
    model.clim_bounds = (10, 1000)
    lut = _get_lut(parent)
    assert lut["clim_bound_min"] == 10.0
    assert lut["clim_bound_max"] == 1000.0

    # gamma
    model.gamma = 0.5
    assert _get_lut(parent)["gamma"] == 0.5


def test_view_set_methods(parent: NdvWidgetState, view: JupyterLUTView) -> None:
    """Direct view set_* calls update the widget state dict."""
    view.set_channel_name("DAPI")
    assert _get_lut(parent)["name"] == "DAPI"

    view.set_clims((200.0, 800.0))
    lut = _get_lut(parent)
    assert lut["clim_min"] == 200.0
    assert lut["clim_max"] == 800.0

    view.set_channel_visible(False)
    assert not _get_lut(parent)["visible"]

    view.set_visible(False)
    assert not _get_lut(parent)["row_visible"]


def test_close_removes_lut(parent: NdvWidgetState, view: JupyterLUTView) -> None:
    assert len(parent.luts) == 1
    view.close()
    assert len(parent.luts) == 0
