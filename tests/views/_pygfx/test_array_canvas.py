from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import wgpu

from ndv.models._viewer_model import ArrayViewerModel
from ndv.views._pygfx._array_canvas import GfxArrayCanvas

if TYPE_CHECKING:
    from wgpu._diagnostics import ObjectCountDiagnostics


def _force_canvas_size(canvas: GfxArrayCanvas, w: int = 600, h: int = 600) -> None:
    """Force the rendercanvas to report a valid size (needed before show)."""
    rc = canvas._canvas
    rc._size_info.set_physical_size(w, h, 1.0)


@pytest.mark.usefixtures("any_app")
def test_zoom_center() -> None:
    """Zoom should keep the center point fixed in world space."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(2)
    canvas.add_image(np.random.rand(100, 100).astype(np.float32))
    canvas.set_range()

    cam = canvas._camera
    assert cam is not None

    # Record initial state (copy position since it's mutable)
    initial_pos = tuple(cam.local.position)
    initial_zoom = cam.zoom

    # Zoom in at a specific world point
    center = (30.0, 70.0)
    canvas.zoom(factor=0.5, center=center)

    # Camera zoom should have increased (zoomed in)
    assert cam.zoom > initial_zoom

    # Camera should have moved toward the center point
    new_pos = tuple(cam.local.position)
    assert abs(center[0] - new_pos[0]) < abs(center[0] - initial_pos[0])
    assert abs(center[1] - new_pos[1]) < abs(center[1] - initial_pos[1])

    # Zoom back out by the inverse factor
    canvas.zoom(factor=2.0, center=center)

    # Should return approximately to initial state
    restored_pos = tuple(cam.local.position)
    assert restored_pos[:2] == pytest.approx(initial_pos[:2])
    assert cam.zoom == pytest.approx(initial_zoom)

    canvas.close()


@pytest.mark.usefixtures("any_app")
def test_canvas_to_world_scale_aware_offset() -> None:
    """canvas_to_world pixel-center offset must scale with pixel size.

    Regression: a constant +0.5 world-space offset (instead of +0.5*scale)
    caused data-index errors when scales != 1.
    """
    for sx, sy in [(1.0, 1.0), (0.2, 0.2), (5.0, 3.0)]:
        canvas = GfxArrayCanvas(ArrayViewerModel())
        _force_canvas_size(canvas)
        canvas.set_ndim(2)
        # must keep a reference: _elements is a WeakValueDictionary
        handle = canvas.add_image(np.zeros((100, 100), dtype=np.float32))
        canvas.set_scales((sy, sx))
        canvas.set_range()

        wx, wy, _ = canvas.canvas_to_world((300, 300))
        # canvas center should map to approximately the image center
        data_x, data_y = int(wx / sx), int(wy / sy)
        assert 49 <= data_x <= 51, f"scale=({sx},{sy}): data_x={data_x}"
        assert 49 <= data_y <= 51, f"scale=({sx},{sy}): data_y={data_y}"

        del handle
        canvas.close()


@pytest.mark.usefixtures("any_app")
def test_no_gpu_memory_leak_on_remove() -> None:
    """GPU resource memory should not grow when image handles are removed."""
    canvas = GfxArrayCanvas(ArrayViewerModel())
    _force_canvas_size(canvas)
    canvas.set_ndim(2)

    tracker = cast("ObjectCountDiagnostics", wgpu.diagnostics.object_counts).tracker

    # create and remove an image to warm up
    handle = canvas.add_image(np.zeros((64, 64), dtype=np.float32))
    handle.remove()
    gc.collect()

    baseline = sum(tracker.amounts.values())

    for _ in range(5):
        handle = canvas.add_image(np.zeros((64, 64), dtype=np.float32))
        handle.remove()
        gc.collect()

    # assert that no new GPU resources are still alive after the add/remove cycles
    assert sum(tracker.amounts.values()) <= baseline

    canvas.close()
