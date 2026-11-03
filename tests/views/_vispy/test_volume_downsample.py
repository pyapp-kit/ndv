from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import vispy.visuals.transforms

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._vispy._array_canvas import VispyArrayCanvas

PATCH_TARGET = "ndv.views._vispy._array_canvas.get_max_texture_sizes"


@pytest.mark.usefixtures("any_app")
def test_volume_downsampled_when_exceeding_texture_limit() -> None:
    """Volume data should be stride-downsampled to fit GPU texture limits."""
    canvas = VispyArrayCanvas(ArrayViewerModel())
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
    canvas = VispyArrayCanvas(ArrayViewerModel())
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
    canvas = VispyArrayCanvas(ArrayViewerModel())
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
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(3)

    # original shape (400, 2200, 2200), factors (1, 2, 2)
    data = np.zeros((10, 100, 100), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data)

    assert handle._downsample_factors == (1, 2, 2)

    # scales in data order (Z, Y, X) = (0.4, 0.2, 0.2)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        canvas.set_scales((0.4, 0.2, 0.2))

    tform = handle._visual.transform
    assert isinstance(tform, vispy.visuals.transforms.STTransform)
    sx, sy, sz = tform.scale[:3]
    # scene order is (x=W, y=H, z=D), reversed from data order
    # x scale: 0.2 (X) * 2 (fw) = 0.4
    # y scale: 0.2 (Y) * 2 (fh) = 0.4
    # z scale: 0.4 (Z) * 1 (fd) = 0.4
    assert sx == pytest.approx(0.4)
    assert sy == pytest.approx(0.4)
    assert sz == pytest.approx(0.4)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_world_origin_and_camera_state_are_public() -> None:
    canvas = VispyArrayCanvas(ArrayViewerModel())
    changed = []
    canvas.cameraChanged.connect(lambda: changed.append(True))
    canvas.set_ndim(3)
    handle = canvas.add_volume(np.zeros((10, 20, 30), dtype=np.float32))
    canvas.set_scales((2.0, 3.0, 4.0))
    canvas.set_origins((100.0, 200.0, 300.0))
    canvas.set_range()

    transform = handle._visual.transform
    assert isinstance(transform, vispy.visuals.transforms.STTransform)
    assert transform.scale[:3] == pytest.approx((4.0, 3.0, 2.0))
    assert transform.translate[:3] == pytest.approx((300.0, 200.0, 100.0))
    viewport, world_to_clip = canvas.camera_state()
    assert viewport == tuple(int(value) for value in canvas._canvas.size)
    assert all(value > 0 for value in viewport)
    assert world_to_clip.shape == (4, 4)
    assert np.isfinite(world_to_clip).all()

    before = world_to_clip.copy()
    canvas._camera.scale_factor /= 2
    canvas._camera.view_changed()
    assert changed
    assert not np.allclose(before, canvas.camera_state()[1])
    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_camera_state_preserves_projective_point_mapping() -> None:
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(3)
    canvas.add_volume(np.zeros((10, 20, 30), dtype=np.float32))
    canvas.set_range()

    viewport, world_to_clip = canvas.camera_state()
    width, height = viewport
    data_points = np.asarray(((0.0, 0.0, 0.0), (3.0, 7.0, 11.0), (8.0, 17.0, 27.0)))
    # VisPy scene order is XYZ while the public camera matrix consumes ZYX.
    scene_points = np.column_stack((data_points[:, ::-1], np.ones(len(data_points))))
    framebuffer = np.asarray(
        canvas._view.scene.transform.map(scene_points), dtype=np.float64
    )
    expected = framebuffer[:, :3] / framebuffer[:, 3, np.newaxis]
    expected[:, 0] = 2.0 * expected[:, 0] / width - 1.0
    expected[:, 1] = 1.0 - 2.0 * expected[:, 1] / height

    homogeneous = np.column_stack((data_points, np.ones(len(data_points))))
    actual = (world_to_clip @ homogeneous.T).T
    actual = actual[:, :3] / actual[:, 3, np.newaxis]

    assert actual == pytest.approx(expected)
    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_image_handles_can_have_independent_world_transforms() -> None:
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(3)
    coarse = canvas.add_volume(np.zeros((4, 4, 4), dtype=np.float32))
    fine = canvas.add_volume(np.zeros((4, 4, 4), dtype=np.float32))

    coarse.set_world_transform((4.0, 4.0, 4.0), (0.0, 0.0, 0.0))
    fine.set_world_transform((1.0, 1.0, 1.0), (8.0, 12.0, 16.0))

    assert coarse._visual.transform.scale[:3] == pytest.approx((4.0, 4.0, 4.0))
    assert fine._visual.transform.scale[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert fine._visual.transform.translate[:3] == pytest.approx((16.0, 12.0, 8.0))
    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_set_range_correct_bounds_after_downsample() -> None:
    """set_range should compute world bounds as if data were full-resolution."""
    canvas = VispyArrayCanvas(ArrayViewerModel())
    canvas.set_ndim(3)

    # shape (10, 100, 80) with max 64 -> factors (1, 2, 2)
    # downsampled shape: (10, 50, 40)
    data = np.zeros((10, 100, 80), dtype=np.float32)
    with patch(PATCH_TARGET, return_value=(None, 64)):
        handle = canvas.add_volume(data)
        canvas.set_scales((1.0, 1.0, 1.0))

    # After set_scales with (1,1,1), the transform should be (2, 2, 1)
    # set_range reads downsampled shape and multiplies by transform scale:
    #   x = shape[2] * sx = 40 * 2 = 80  (matches original W)
    #   y = shape[1] * sy = 50 * 2 = 100 (matches original H)
    #   z = shape[0] * sz = 10 * 1 = 10  (matches original D)
    tform = handle._visual.transform
    assert isinstance(tform, vispy.visuals.transforms.STTransform)
    ds_shape = handle.data().shape
    sx, sy, sz = tform.scale[:3]
    assert ds_shape[2] * sx == pytest.approx(80.0)
    assert ds_shape[1] * sy == pytest.approx(100.0)
    assert ds_shape[0] * sz == pytest.approx(10.0)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_no_downsample_when_limits_none() -> None:
    """When GPU limits are unavailable, data should pass through unchanged."""
    canvas = VispyArrayCanvas(ArrayViewerModel())
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
    canvas = VispyArrayCanvas(ArrayViewerModel())
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
