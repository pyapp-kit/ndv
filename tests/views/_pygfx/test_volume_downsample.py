from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pygfx
import pytest

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._pygfx._array_canvas import GfxArrayCanvas

PATCH_TARGET = "ndv.views._pygfx._array_canvas._get_max_texture_sizes"


def _force_canvas_size(canvas: GfxArrayCanvas, w: int = 600, h: int = 600) -> None:
    rc = canvas._canvas
    rc._size_info.set_physical_size(w, h, 1.0)


@pytest.mark.usefixtures("any_app")
def test_volume_downsampled_when_exceeding_texture_limit() -> None:
    """Volume data should be stride-downsampled to fit GPU texture limits."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(3)

    data = np.zeros((10, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data)

    # shape (10, 100, 100) with max 64 -> strides (1, 2, 2)
    assert handle._downsample_factors == (1, 2, 2)
    assert handle.data().shape == (10, 50, 50)

    # set_data with the same original shape should also downsample
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle.set_data(data)
    assert handle.data().shape == (10, 50, 50)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_volume_no_downsample_when_within_limit() -> None:
    """Volume data within texture limits should not be downsampled."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(3)

    data = np.zeros((10, 50, 50), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data)

    assert handle._downsample_factors == (1, 1, 1)
    assert handle.data().shape == (10, 50, 50)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_image_downsampled_when_exceeding_2d_texture_limit() -> None:
    """2D image data should be stride-downsampled to fit GPU texture limits."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(2)

    data = np.zeros((100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(64, None)):
        handle = canvas.add_image(data)

    assert handle._downsample_factors == (2, 2)
    assert handle.data().shape == (50, 50)

    # set_data should also downsample
    with patch(PATCH_TARGET, return_value=(64, None)):
        handle.set_data(data)
    assert handle.data().shape == (50, 50)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_set_scales_compensates_for_volume_downsample() -> None:
    """set_scales should multiply by downsample factors so world coords stay correct."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(3)

    data = np.zeros((10, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data)

    assert handle._downsample_factors == (1, 2, 2)

    # scales in data order (Z, Y, X) = (0.4, 0.2, 0.2)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        canvas.set_scales((0.4, 0.2, 0.2))

    vol = handle._image
    assert isinstance(vol, pygfx.Volume)
    sx, sy, sz = vol.local.scale
    # pygfx order is (x=W, y=H, z=D), reversed from data order
    # x scale: 0.2 (X) * 2 (fw) = 0.4
    # y scale: 0.2 (Y) * 2 (fh) = 0.4
    # z scale: 0.4 (Z) * 1 (fd) = 0.4
    assert sx == pytest.approx(0.4)
    assert sy == pytest.approx(0.4)
    assert sz == pytest.approx(0.4)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_no_downsample_when_limits_none() -> None:
    """When GPU limits are unavailable, data should pass through unchanged."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(3)

    data = np.zeros((10, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, None)):
        handle = canvas.add_volume(data)

    assert handle._downsample_factors == ()
    assert handle.data().shape == (10, 100, 100)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_set_data_with_different_shape() -> None:
    """set_data with a new shape should re-downsample correctly."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(3)

    data1 = np.zeros((10, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data1)
    assert handle.data().shape == (10, 50, 50)

    # now set_data with a larger volume
    data2 = np.zeros((200, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle.set_data(data2)
    assert handle._downsample_factors == (4, 2, 2)
    assert handle.data().shape == (50, 50, 50)

    canvas.close()
