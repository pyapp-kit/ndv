from __future__ import annotations

import numpy as np
import pytest

from ndv.controllers._image_stats import _compute_histogram
from ndv.models import DataWrapper
from ndv.models._array_display_model import ArrayDisplayModel
from ndv.models._resolve import build_slice_requests, process_request, resolve


@pytest.mark.parametrize(
    "shape, model_kwargs, expected_ndim_offset",
    [
        pytest.param((1, 10), {"visible_axes": (0, 1)}, 0, id="singleton_first_vis"),
        pytest.param((1, 1, 5), {"visible_axes": (0, 1)}, 0, id="all_singleton_vis"),
        pytest.param(
            (1, 3, 5),
            {"visible_axes": (0, 2), "channel_axis": 1, "channel_mode": "composite"},
            0,
            id="singleton_vis_with_channel",
        ),
        pytest.param(
            (1, 10, 3),
            {"visible_axes": (0, 1), "channel_axis": 2, "channel_mode": "rgba"},
            1,  # RGBA adds a channel dim
            id="singleton_vis_rgba",
        ),
    ],
)
def test_singleton_visible_axes_preserved(
    shape: tuple[int, ...],
    model_kwargs: dict,
    expected_ndim_offset: int,
) -> None:
    """Visible axes of size 1 must not be squeezed out of the response."""
    wrapper = DataWrapper.create(np.ones(shape, dtype=np.uint8))
    model = ArrayDisplayModel(**model_kwargs)
    resolved = resolve(model, wrapper)

    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    expected_ndim = response.n_visible_axes + expected_ndim_offset
    for key, arr in response.data.items():
        assert arr.ndim == expected_ndim, (
            f"Channel {key}: expected {expected_ndim}D, "
            f"got {arr.ndim}D (shape {arr.shape})."
        )


def test_negative_index_produces_nonempty_data() -> None:
    """current_index={0: -1} should not produce empty data."""
    data = np.arange(150).reshape(3, 5, 10).astype(np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(-2, -1))
    model.current_index[0] = -1

    resolved = resolve(model, wrapper)
    reqs = build_slice_requests(resolved, wrapper)

    response = process_request(reqs[0])
    for arr in response.data.values():
        assert arr.size > 0, (
            "Negative current_index produced empty data. "
            "slice(-1, 0) gives an empty array."
        )


def test_resolved_index_clamps_negative_to_valid_range() -> None:
    """Negative index values should be clamped to [0, max_val]."""
    data = np.ones((5, 8, 10), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(-2, -1))
    model.current_index[0] = -1

    resolved = resolve(model, wrapper)
    assert isinstance(resolved.current_index[0], int)
    assert resolved.current_index[0] >= 0


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)
def test_calc_hist_bins_float32(dtype) -> None:
    """_compute_histogram should return valid counts and edges for various dtypes."""
    if np.issubdtype(dtype, np.floating):
        data = np.random.rand(1000).astype(dtype) * 100
    else:
        data = np.random.randint(0, 100, size=(1000,), dtype=dtype)
    counts, edges = _compute_histogram(data, significant_bits=None)
    assert len(counts) > 0
    assert len(edges) == len(counts) + 1
