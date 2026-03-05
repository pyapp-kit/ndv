"""Profile ArrayViewer hot code paths with mocked views.

Usage:
    uv run python scripts/profile_viewer.py [--viz]

Simulates the most common user interactions (slider moves, mode changes,
ndim toggles) without a real GUI, then prints a cProfile summary sorted by
cumulative time.  Pass --viz to open a snakeviz browser tab instead.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from typing import Any, no_type_check
from unittest.mock import MagicMock, patch

import numpy as np

from ndv.controllers import ArrayViewer
from ndv.models._array_display_model import ChannelMode
from ndv.views import _app
from ndv.views.bases import ArrayView, LutView
from ndv.views.bases._graphics._canvas import ArrayCanvas
from ndv.views.bases._graphics._canvas_elements import ImageHandle

# --------------- mock setup (same pattern as test_controller) ---------------

SHAPE_2D = (30, 5, 256, 256)  # TZYX-ish
SHAPE_RGB = (10, 128, 128, 3)  # TYXC


def _get_mock_canvas(*_: Any) -> ArrayCanvas:
    mock = MagicMock(spec=ArrayCanvas)
    img = MagicMock(spec=ImageHandle)
    img.data.return_value = np.zeros((256, 256), dtype=np.uint8)
    img.visible.return_value = True
    mock.add_image.return_value = img

    vol = MagicMock(spec=ImageHandle)
    vol.data.return_value = np.zeros((30, 256, 256), dtype=np.uint8)
    vol.visible.return_value = True
    mock.add_volume.return_value = vol
    mock.canvas_to_world.return_value = (50, 50, 0)
    return mock


def _get_mock_view(*_: Any) -> ArrayView:
    mock = MagicMock(spec=ArrayView)
    lut = MagicMock(spec=LutView)
    mock.add_lut_view.return_value = lut
    return mock


_patches = [
    patch.object(_app, "get_array_canvas_class", lambda: _get_mock_canvas),
    patch.object(_app, "get_array_view_class", lambda: _get_mock_view),
]


@no_type_check
def _make_ctrl(data: np.ndarray) -> ArrayViewer:
    ctrl = ArrayViewer()
    ctrl._async = False
    ctrl.data = data
    return ctrl


# --------------- simulation scenarios ----------------------------------------


@no_type_check
def simulate_slider_scrub(ctrl: ArrayViewer, n: int = 200) -> None:
    """Simulate a user scrubbing through the first non-visible axis."""
    max_t = ctrl._data_wrapper.data.shape[0]
    for i in range(n):
        idx = {0: i % max_t}
        ctrl._view.current_index.return_value = idx
        ctrl._on_view_current_index_changed()


@no_type_check
def simulate_channel_mode_cycling(ctrl: ArrayViewer, n: int = 30) -> None:
    """Cycle through channel modes repeatedly."""
    modes = [ChannelMode.GRAYSCALE, ChannelMode.COMPOSITE, ChannelMode.COLOR]
    for i in range(n):
        ctrl._on_view_channel_mode_changed(modes[i % len(modes)])


@no_type_check
def simulate_ndim_toggle(ctrl: ArrayViewer, n: int = 30) -> None:
    """Toggle between 2D and 3D display."""
    for i in range(n):
        ctrl._on_view_ndim_toggle_requested(i % 2 == 0)


@no_type_check
def simulate_mouse_hover(ctrl: ArrayViewer, n: int = 200) -> None:
    """Simulate mouse hover over the canvas."""
    from ndv._types import MouseMoveEvent

    for i in range(n):
        x, y = i % 200, (i * 3) % 200
        ctrl._canvas.canvas_to_world.return_value = (x, y, 0)
        ctrl._on_canvas_mouse_moved(MouseMoveEvent(x, y))


@no_type_check
def simulate_visible_axes_change(ctrl: ArrayViewer, n: int = 20) -> None:
    """Change visible axes repeatedly."""
    combos = [(2, 3), (0, 3), (1, 3), (0, 2)]
    for i in range(n):
        ctrl.display_model.visible_axes = combos[i % len(combos)]


@no_type_check
def simulate_data_replacement(n: int = 10) -> None:
    """Replace data entirely (simulates loading new files)."""
    ctrl = _make_ctrl(np.zeros(SHAPE_2D, dtype=np.uint8))
    for _ in range(n):
        ctrl.data = np.zeros(SHAPE_2D, dtype=np.uint8)


@no_type_check
def simulate_resolve_only(n: int = 500) -> None:
    """Benchmark resolve() in isolation."""
    from ndv.models._resolve import resolve

    from ndv.models import ArrayDisplayModel, DataWrapper

    wrapper = DataWrapper.create(np.zeros(SHAPE_2D, dtype=np.uint8))
    model = ArrayDisplayModel()
    for i in range(n):
        model.current_index.assign({0: i % SHAPE_2D[0]})
        resolve(model, wrapper)


def run_all() -> None:
    """Run all simulation scenarios."""
    # --- 2D grayscale (most common) ---
    data = np.random.randint(0, 255, SHAPE_2D, dtype=np.uint8)
    ctrl = _make_ctrl(data)

    simulate_slider_scrub(ctrl, n=500)
    simulate_channel_mode_cycling(ctrl, n=100)
    simulate_ndim_toggle(ctrl, n=100)
    simulate_mouse_hover(ctrl, n=500)
    simulate_visible_axes_change(ctrl, n=50)

    # --- RGB data ---
    rgb_data = np.random.randint(0, 255, SHAPE_RGB, dtype=np.uint8)
    ctrl_rgb = _make_ctrl(rgb_data)
    simulate_slider_scrub(ctrl_rgb, n=300)
    simulate_mouse_hover(ctrl_rgb, n=300)

    # --- data replacement ---
    simulate_data_replacement(n=20)

    # --- resolve in isolation ---
    simulate_resolve_only(n=1000)


def _ndv_filter(stats: pstats.Stats) -> pstats.Stats:
    """Filter stats to only show ndv-related entries (exclude mock/import noise)."""
    noise = {"mock.py", "enum.py", "importlib", "_imp.", "enums_compat", "_app.py:34"}
    filtered = {}
    for key, value in stats.stats.items():
        filename = key[0]
        if not any(n in filename for n in noise):
            filtered[key] = value
    stats.stats = filtered
    return stats


def main() -> None:
    for p in _patches:
        p.start()
    try:
        # warm up (imports, app creation, etc.)
        _make_ctrl(np.zeros((2, 2, 2), dtype=np.uint8))

        profiler = cProfile.Profile()
        profiler.enable()
        run_all()
        profiler.disable()
    finally:
        for p in _patches:
            p.stop()

    if "--viz" in sys.argv:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            profiler.dump_stats(f.name)
            print(f"Profile saved to {f.name}")
        import subprocess

        subprocess.Popen(["uvx", "snakeviz", f.name])
    else:
        stats = pstats.Stats(profiler)
        stats.strip_dirs()

        print("=" * 80)
        print("TOP 50 BY CUMULATIVE TIME (ndv code only)")
        print("=" * 80)
        _ndv_filter(stats).sort_stats("cumulative").print_stats(50)

        # reload unfiltered for tottime
        stats2 = pstats.Stats(profiler)
        stats2.strip_dirs()
        print("=" * 80)
        print("TOP 50 BY TOTAL (SELF) TIME (ndv code only)")
        print("=" * 80)
        _ndv_filter(stats2).sort_stats("tottime").print_stats(50)

        # unfiltered callers
        stats3 = pstats.Stats(profiler)
        stats3.strip_dirs()
        print("=" * 80)
        print("CALLERS OF resolve-related functions")
        print("=" * 80)
        stats3.sort_stats("tottime").print_callers(
            "resolve",
            "_re_resolve",
            "_request_data",
            "_apply_changes",
            "_norm_data_coords",
            "summary_info",
            "guess_channel_axis",
        )


if __name__ == "__main__":
    main()
