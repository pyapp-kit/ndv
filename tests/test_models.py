from unittest.mock import Mock

import numpy as np

from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._data_wrapper import DataWrapper
from ndv.models._resolve import (
    _resolve_channel_names,
    _resolve_visible_scales,
    resolve,
)
from ndv.models._roi_model import RectangularROIModel


def test_array_display_model() -> None:
    m = ArrayDisplayModel()

    mock = Mock()
    m.events.channel_axis.connect(mock)
    m.current_index.item_added.connect(mock)
    m.current_index.item_changed.connect(mock)

    m.channel_axis = 4
    mock.assert_called_once_with(4, None)  # new, old
    mock.reset_mock()
    m.current_index["5"] = 1
    mock.assert_called_once_with(5, 1)  # key, value
    mock.reset_mock()
    m.current_index[5] = 4
    mock.assert_called_once_with(5, 4, 1)  # key, new, old
    mock.reset_mock()

    assert ArrayDisplayModel.model_json_schema(mode="validation")
    assert ArrayDisplayModel.model_json_schema(mode="serialization")


def test_rectangular_roi_model() -> None:
    m = RectangularROIModel()

    mock = Mock()
    m.events.bounding_box.connect(mock)
    m.events.visible.connect(mock)

    m.bounding_box = ((10, 10), (20, 20))
    mock.assert_called_once_with(
        ((10, 10), (20, 20)),  # New bounding box value
        ((0, 0), (0, 0)),  # Initial bounding box on construction
    )
    mock.reset_mock()

    m.visible = False
    mock.assert_called_once_with(
        False,  # New visibility
        True,  # Initial visibility on construction
    )
    mock.reset_mock()

    assert RectangularROIModel.model_json_schema(mode="validation")
    assert RectangularROIModel.model_json_schema(mode="serialization")


def test_resolve_visible_scales_user_overrides_data() -> None:
    """User-set scales take priority over data-derived scales."""
    wrapper = DataWrapper.create(np.empty((5, 100, 200)))
    model = ArrayDisplayModel(scales={-2: 0.5, -1: 0.1})
    vis = (1, 2)
    result = _resolve_visible_scales(model, wrapper, vis)
    assert result == (0.5, 0.1)


def test_resolve_visible_scales_default() -> None:
    """Unset scales default to 1.0."""
    wrapper = DataWrapper.create(np.empty((5, 100, 200)))
    model = ArrayDisplayModel()
    vis = (1, 2)
    result = _resolve_visible_scales(model, wrapper, vis)
    assert result == (1.0, 1.0)


def test_resolve_visible_scales_from_xarray() -> None:
    """Scales inferred from xarray coord spacing."""
    xr = __import__("pytest").importorskip("xarray")
    da = xr.DataArray(
        np.empty((3, 10, 20)),
        dims=["c", "y", "x"],
        coords={"y": np.arange(10) * 0.5, "x": np.arange(20) * 0.25},
    )
    wrapper = DataWrapper.create(da)
    model = ArrayDisplayModel()
    vis = (1, 2)
    result = _resolve_visible_scales(model, wrapper, vis)
    assert result[0] == 0.5
    assert result[1] == 0.25


def test_resolve_channel_names_user_overrides_data() -> None:
    """User-set channel names take priority over data-derived names."""
    xr = __import__("pytest").importorskip("xarray")
    da = xr.DataArray(
        np.empty((3, 10, 20)),
        dims=["c", "y", "x"],
        coords={"c": ["red", "green", "blue"]},
    )
    wrapper = DataWrapper.create(da)
    model = ArrayDisplayModel(channel_names={0: "DAPI", 1: "GFP"})
    result = _resolve_channel_names(model, wrapper, channel_axis=0)
    assert result[0] == "DAPI"
    assert result[1] == "GFP"
    assert result[2] == "blue"  # data-derived, not overridden


def test_resolve_channel_names_no_channel_axis() -> None:
    """No channel axis returns empty dict."""
    wrapper = DataWrapper.create(np.empty((10, 20)))
    model = ArrayDisplayModel()
    result = _resolve_channel_names(model, wrapper, channel_axis=None)
    assert result == {}


def test_axis_scales_nan_inf_skipped() -> None:
    """NaN/inf coords should be silently skipped."""
    xr = __import__("pytest").importorskip("xarray")
    da = xr.DataArray(
        np.empty((5, 10)),
        dims=["y", "x"],
        coords={"y": [0, 1, float("nan"), 3, 4], "x": np.arange(10) * 2.0},
    )
    wrapper = DataWrapper.create(da)
    scales = wrapper.axis_scales()
    assert "y" not in scales  # nan in coords, should be skipped
    assert scales["x"] == 2.0


def test_axis_scales_descending() -> None:
    """Descending coords should produce negative scale."""
    xr = __import__("pytest").importorskip("xarray")
    da = xr.DataArray(
        np.empty((5, 10)),
        dims=["y", "x"],
        coords={"y": [4.0, 3.0, 2.0, 1.0, 0.0], "x": np.arange(10)},
    )
    wrapper = DataWrapper.create(da)
    scales = wrapper.axis_scales()
    assert scales["y"] == -1.0
    assert scales["x"] == 1.0


def test_axis_scales_non_uniform_skipped() -> None:
    """Non-uniform coord spacing should not produce a scale."""
    xr = __import__("pytest").importorskip("xarray")
    da = xr.DataArray(
        np.empty((5,)),
        dims=["x"],
        coords={"x": [0, 1, 3, 6, 10]},
    )
    wrapper = DataWrapper.create(da)
    scales = wrapper.axis_scales()
    assert "x" not in scales


def test_full_resolve_includes_scales_and_names() -> None:
    """resolve() populates visible_scales and channel_names."""
    wrapper = DataWrapper.create(np.empty((3, 100, 200)))
    model = ArrayDisplayModel(
        channel_axis=0,
        channel_mode="composite",
        scales={1: 0.5, 2: 0.1},
        channel_names={0: "DAPI", 1: "GFP", 2: "mCherry"},
    )
    resolved = resolve(model, wrapper)
    assert resolved.visible_scales == (0.5, 0.1)
    assert resolved.channel_names == {0: "DAPI", 1: "GFP", 2: "mCherry"}


def test_scale_change_triggers_equality_diff() -> None:
    """Changing visible_scales should make two states unequal."""
    wrapper = DataWrapper.create(np.empty((100, 200)))
    model1 = ArrayDisplayModel()
    model2 = ArrayDisplayModel(scales={-2: 2.0})
    r1 = resolve(model1, wrapper)
    r2 = resolve(model2, wrapper)
    assert r1 != r2  # different scales


def test_channel_names_change_no_equality_diff() -> None:
    """Changing channel_names should NOT make two states unequal (cosmetic)."""
    wrapper = DataWrapper.create(np.empty((3, 100, 200)))
    model1 = ArrayDisplayModel(channel_axis=0, channel_mode="composite")
    model2 = ArrayDisplayModel(
        channel_axis=0,
        channel_mode="composite",
        channel_names={0: "DAPI"},
    )
    r1 = resolve(model1, wrapper)
    r2 = resolve(model2, wrapper)
    # channel_names is excluded from __eq__, so states should be equal
    assert r1 == r2
