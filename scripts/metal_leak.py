# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "ndv[pyqt,pygfx]",
#     "pyobjc-framework-metal>=12.1",
#     "pytest",
# ]
#
# [tool.uv.sources]
# ndv = { path = "../", editable = true }
# ///
"""Test for GPU/CPU memory leak when reassigning ArrayViewer.data.

Demonstrates https://github.com/pyapp-kit/ndv/issues/209

Run directly:
    uv run test_handle_leak.py vispy
    uv run test_handle_leak.py pygfx
    uv run test_handle_leak.py vispy --plot   # also save a plot

Run with pytest:
    NDV_CANVAS_BACKEND=vispy uv run pytest test_handle_leak.py -v
    NDV_CANVAS_BACKEND=pygfx uv run pytest test_handle_leak.py -v
"""

from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# ===================================================================
# Configuration
# ===================================================================

N_WARMUP = 3  # iterations before we start measuring (let caches settle)
N_MEASURE = 20  # iterations to measure growth over

# Allowed growth per iteration (leak thresholds).
# Set these to what "fixed" should look like.
MAX_VRAM_GROWTH_MB_PER_ITER = 0.5  # should be ~0 when fixed
MAX_OBJECT_GROWTH_PER_ITER = 0.1  # should be 0 when fixed (fractional to allow jitter)

DATA_SHAPE = (2048, 2048)
DATA_DTYPE = np.uint16

# Backend -> type names to track
BACKEND_TYPES: dict[str, tuple[str, ...]] = {
    "vispy": ("Texture2D", "Texture3D"),
    "pygfx": ("Texture", "Buffer"),
}


# ===================================================================
# Probes
# ===================================================================

_metal_device: Any = None


def _init_metal() -> None:
    global _metal_device
    if _metal_device is not None:
        return
    try:
        import Metal

        _metal_device = Metal.MTLCreateSystemDefaultDevice()
    except Exception:
        _metal_device = False


def get_metal_vram_bytes() -> int | None:
    _init_metal()
    if _metal_device is False:
        return None
    return _metal_device.currentAllocatedSize()


def get_rss_bytes() -> int:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss if sys.platform == "darwin" else rusage.ru_maxrss * 1024


def count_objects_by_type(type_names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(type_names, 0)
    for obj in gc.get_objects():
        tname = type(obj).__name__
        if tname in counts:
            counts[tname] += 1
    return counts


# ===================================================================
# Snapshot / Profiler
# ===================================================================


@dataclass
class Snapshot:
    iteration: int
    time_s: float
    metal_vram_bytes: int | None
    rss_bytes: int
    object_counts: dict[str, int]


class LeakProfiler:
    def __init__(self, track_types: Sequence[str] = ()):
        self.track_types = list(track_types)
        self.snapshots: list[Snapshot] = []
        self._t0 = time.perf_counter()

    def snapshot(self, iteration: int) -> Snapshot:
        gc.collect()
        gc.collect()
        snap = Snapshot(
            iteration=iteration,
            time_s=time.perf_counter() - self._t0,
            metal_vram_bytes=get_metal_vram_bytes(),
            rss_bytes=get_rss_bytes(),
            object_counts=count_objects_by_type(self.track_types),
        )
        self.snapshots.append(snap)
        return snap

    def print_last(self) -> None:
        s = self.snapshots[-1]
        parts = [f"[{s.iteration:>4d}]"]
        if s.metal_vram_bytes is not None:
            parts.append(f"VRAM={s.metal_vram_bytes / 1e6:.1f}MB")
        parts.append(f"RSS={s.rss_bytes / 1e6:.1f}MB")
        for name, count in s.object_counts.items():
            parts.append(f"{name}={count}")
        print("  ".join(parts))

    # --- analysis helpers ---

    def vram_growth_per_iter(self) -> float | None:
        """MB of VRAM growth per iteration (linear fit slope)."""
        vram = [s.metal_vram_bytes for s in self.snapshots]
        if vram[0] is None:
            return None
        iters = np.array([s.iteration for s in self.snapshots], dtype=float)
        vram_mb = np.array(vram, dtype=float) / 1e6
        slope, _ = np.polyfit(iters, vram_mb, 1)
        return float(slope)

    def rss_growth_per_iter(self) -> float:
        """MB of RSS growth per iteration (linear fit slope)."""
        iters = np.array([s.iteration for s in self.snapshots], dtype=float)
        rss_mb = np.array([s.rss_bytes for s in self.snapshots], dtype=float) / 1e6
        slope, _ = np.polyfit(iters, rss_mb, 1)
        return float(slope)

    def object_growth_per_iter(self, type_name: str) -> float:
        """Object count growth per iteration (linear fit slope)."""
        iters = np.array([s.iteration for s in self.snapshots], dtype=float)
        counts = np.array(
            [s.object_counts.get(type_name, 0) for s in self.snapshots], dtype=float
        )
        slope, _ = np.polyfit(iters, counts, 1)
        return float(slope)

    def plot(self, save_path: str | None = None) -> None:
        import matplotlib.pyplot as plt

        iters = [s.iteration for s in self.snapshots]
        panels: list[tuple[str, list[float], str]] = []

        vram = [s.metal_vram_bytes for s in self.snapshots]
        if vram[0] is not None:
            panels.append(("Metal VRAM", [v / 1e6 for v in vram], "MB"))  # type: ignore[arg-type]
        panels.append(
            ("Process RSS", [s.rss_bytes / 1e6 for s in self.snapshots], "MB")
        )
        for tname in self.track_types:
            panels.append(
                (
                    f"Live {tname} objects",
                    [s.object_counts.get(tname, 0) for s in self.snapshots],
                    "count",
                )
            )

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for ax, (title, values, ylabel) in zip(axes, panels, strict=False):
            ax.plot(iters, values, "o-", markersize=3)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Iteration")
        fig.suptitle("Leak Profiling", fontsize=14, y=1.01)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        else:
            plt.show()


# ===================================================================
# Core test logic
# ===================================================================


def _process_events() -> None:
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        for _ in range(4):
            app.processEvents()


def _run_leak_test(backend: str) -> LeakProfiler:
    """Run the leak test for a given backend, return the profiler with results."""
    os.environ["NDV_CANVAS_BACKEND"] = backend

    import ndv

    track_types = (
        *BACKEND_TYPES.get(backend, ()),
        "Texture2D",
        "Texture3D",
        "Texture",
        "Buffer",
    )
    # dedupe while preserving order
    track_types = tuple(dict.fromkeys(track_types))

    profiler = LeakProfiler(track_types=track_types)

    viewer = ndv.ArrayViewer(np.random.randint(0, 10000, DATA_SHAPE, dtype=DATA_DTYPE))
    viewer.show()
    _process_events()

    # warmup: let caches, JIT, lazy init settle
    for _i in range(N_WARMUP):
        viewer.data = np.random.randint(0, 10000, DATA_SHAPE, dtype=DATA_DTYPE)
        _process_events()

    # baseline snapshot (after warmup)
    profiler.snapshot(0)
    profiler.print_last()

    # measurement iterations
    for i in range(1, N_MEASURE + 1):
        viewer.data = np.random.randint(0, 10000, DATA_SHAPE, dtype=DATA_DTYPE)
        _process_events()
        profiler.snapshot(i)
        profiler.print_last()

    viewer.close()
    return profiler


# ===================================================================
# pytest tests
# ===================================================================


def _get_backend() -> str:
    backend = os.environ.get("NDV_CANVAS_BACKEND", "pygfx")
    if backend not in BACKEND_TYPES:
        pytest.skip(f"Unknown backend: {backend!r}")
    return backend


@pytest.fixture(scope="module")
def leak_profiler() -> LeakProfiler:
    """Run the leak test once per process using NDV_CANVAS_BACKEND env var.

    Usage:
        NDV_CANVAS_BACKEND=vispy pytest test_handle_leak.py -v
        NDV_CANVAS_BACKEND=pygfx pytest test_handle_leak.py -v
    """
    return _run_leak_test(_get_backend())


def test_no_vram_leak(leak_profiler: LeakProfiler) -> None:
    """VRAM should not grow linearly when reassigning viewer.data."""
    slope = leak_profiler.vram_growth_per_iter()
    if slope is None:
        pytest.skip("Metal VRAM probe not available")
    print(f"VRAM growth: {slope:.2f} MB/iter")
    assert slope < MAX_VRAM_GROWTH_MB_PER_ITER, (
        f"VRAM leak detected: {slope:.2f} MB/iter "
        f"(threshold: {MAX_VRAM_GROWTH_MB_PER_ITER})"
    )


def test_no_texture_object_leak(leak_profiler: LeakProfiler) -> None:
    """GPU texture/buffer object count should not grow when reassigning viewer.data."""
    backend = _get_backend()
    for tname in BACKEND_TYPES[backend]:
        slope = leak_profiler.object_growth_per_iter(tname)
        print(f"{tname} growth: {slope:.2f} objects/iter")
        assert slope < MAX_OBJECT_GROWTH_PER_ITER, (
            f"{tname} object leak detected: {slope:.2f} objects/iter "
            f"(threshold: {MAX_OBJECT_GROWTH_PER_ITER})"
        )


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "pygfx"
    do_plot = "--plot" in sys.argv

    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)

    profiler = _run_leak_test(backend)

    # --- report ---
    print("\n" + "=" * 60)
    print(f"RESULTS ({backend})")
    print("=" * 60)

    passed = True

    vram_slope = profiler.vram_growth_per_iter()
    if vram_slope is not None:
        ok = vram_slope < MAX_VRAM_GROWTH_MB_PER_ITER
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] VRAM growth: {vram_slope:.2f} MB/iter "
            f"(max: {MAX_VRAM_GROWTH_MB_PER_ITER})"
        )
        passed &= ok

    rss_slope = profiler.rss_growth_per_iter()
    print(f"  [INFO] RSS growth:  {rss_slope:.2f} MB/iter")

    for tname in BACKEND_TYPES.get(backend, ()):
        slope = profiler.object_growth_per_iter(tname)
        ok = slope < MAX_OBJECT_GROWTH_PER_ITER
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] {tname} growth: {slope:.2f} objects/iter "
            f"(max: {MAX_OBJECT_GROWTH_PER_ITER})"
        )
        passed &= ok

    print("=" * 60)
    print(f"{'ALL CHECKS PASSED' if passed else 'LEAK DETECTED'}")
    print("=" * 60)

    if do_plot:
        profiler.plot(save_path=f"leak_profile_{backend}.png")

    sys.exit(0 if passed else 1)
