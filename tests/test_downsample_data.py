from __future__ import annotations

import numpy as np

from ndv.views._util import downsample_data


def test_no_downsample_when_within_limit() -> None:
    data = np.zeros((50, 60), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert factors == (1, 1)
    assert result is data  # no copy, same object


def test_exact_boundary_no_downsample() -> None:
    data = np.zeros((64, 64, 64), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert factors == (1, 1, 1)
    assert result is data


def test_single_axis_overflow() -> None:
    data = np.zeros((10, 100), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert factors == (1, 2)
    assert result.shape == (10, 50)


def test_all_axes_overflow() -> None:
    data = np.zeros((200, 150, 130), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert factors == (4, 3, 3)
    assert result.shape == (50, 50, 44)


def test_2d_input() -> None:
    data = np.zeros((100, 80), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert factors == (2, 2)
    assert result.shape == (50, 40)


def test_4d_input() -> None:
    data = np.zeros((10, 100, 80, 3), dtype=np.float32)
    result, factors = downsample_data(data, 64)
    assert len(factors) == 4
    assert factors == (1, 2, 2, 1)
    assert result.shape == (10, 50, 40, 3)


def test_returns_view_not_copy() -> None:
    """Downsampled result should be a view (no memory copy)."""
    data = np.ones((100, 100), dtype=np.float32)
    result, _ = downsample_data(data, 64)
    assert result.base is data


def test_max_size_1() -> None:
    data = np.zeros((5, 3), dtype=np.float32)
    result, factors = downsample_data(data, 1)
    assert factors == (5, 3)
    assert result.shape == (1, 1)
