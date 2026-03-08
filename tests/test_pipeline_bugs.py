"""Tests exposing bugs in the model->resolve->view pipeline.

Each test demonstrates a specific bug and is expected to FAIL until fixed.
"""

from __future__ import annotations

import numpy as np

from ndv.controllers._array_viewer import _calc_hist_bins
from ndv.models import DataWrapper
from ndv.models._array_display_model import ArrayDisplayModel, ChannelMode
from ndv.models._resolve import build_slice_requests, process_request, resolve

# =====================================================================
# Bug 1: squeeze() in process_request removes singleton visible axes
#
# process_request uses .squeeze() after transpose, which strips ALL
# singleton dimensions — including visible axes that happen to have
# size 1. The viewer then receives data with fewer dimensions than
# n_visible_axes.
# =====================================================================


def test_squeeze_singleton_first_visible_axis() -> None:
    """Data (1, 10): visible_axes=(0, 1) -> should be 2D, not 1D."""
    data = np.ones((1, 10), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(0, 1))
    resolved = resolve(model, wrapper)

    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    for arr in response.data.values():
        assert arr.ndim == response.n_visible_axes, (
            f"Expected {response.n_visible_axes}D, got {arr.ndim}D "
            f"(shape {arr.shape}). squeeze() removed a singleton visible axis."
        )


def test_squeeze_singleton_visible_axis_with_channel() -> None:
    """Data (1, 3, 5): visible=(0, 2), ch=1, COMPOSITE.

    Each channel slice should be 2D (1, 5), not 1D (5).
    """
    data = np.ones((1, 3, 5), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(
        visible_axes=(0, 2),
        channel_axis=1,
        channel_mode=ChannelMode.COMPOSITE,
    )
    resolved = resolve(model, wrapper)

    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    for key, arr in response.data.items():
        assert arr.ndim == response.n_visible_axes, (
            f"Channel {key}: expected {response.n_visible_axes}D, "
            f"got {arr.ndim}D (shape {arr.shape})."
        )


def test_squeeze_singleton_visible_axis_rgba() -> None:
    """Data (1, 10, 3): visible=(0, 1), ch=2, RGBA.

    Result should be 3D (1, 10, 3), not 2D (10, 3).
    """
    data = np.ones((1, 10, 3), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(
        visible_axes=(0, 1),
        channel_axis=2,
        channel_mode=ChannelMode.RGBA,
    )
    resolved = resolve(model, wrapper)

    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    for arr in response.data.values():
        # RGBA: n_visible_axes spatial dims + 1 channel dim
        expected_ndim = response.n_visible_axes + 1
        assert arr.ndim == expected_ndim, (
            f"RGBA: expected {expected_ndim}D, got {arr.ndim}D (shape {arr.shape})."
        )


def test_squeeze_all_singleton_visible_axes() -> None:
    """Data (1, 1, 5): visible=(0, 1) -> should be 2D (1, 1), not 0D."""
    data = np.ones((1, 1, 5), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(0, 1))
    resolved = resolve(model, wrapper)

    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    for arr in response.data.values():
        assert arr.ndim == response.n_visible_axes, (
            f"Expected {response.n_visible_axes}D, got {arr.ndim}D (shape {arr.shape})."
        )


# =====================================================================
# Bug 2: negative current_index values are not clamped
#
# _norm_current_index only clamps values exceeding max_val but does
# not handle negative values. -1 passes through and becomes
# slice(-1, 0) in build_slice_requests, producing an empty array
# (or a transpose error).
# =====================================================================


def test_negative_index_produces_nonempty_data() -> None:
    """current_index={0: -1} should not produce empty data."""
    data = np.arange(30).reshape(3, 10).astype(np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(-1,) * 2)
    model.current_index[0] = -1

    resolved = resolve(model, wrapper)
    reqs = build_slice_requests(resolved, wrapper)

    # This currently raises ValueError or produces empty data
    response = process_request(reqs[0])

    for arr in response.data.values():
        assert arr.size > 0, (
            "Negative current_index produced empty data. "
            "slice(-1, 0) gives an empty array."
        )


def test_resolved_index_clamps_negative_to_valid_range() -> None:
    """Negative index values should be clamped to [0, max_val]."""
    data = np.ones((5, 10), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(visible_axes=(1,) * 2)
    model.current_index[0] = -1

    resolved = resolve(model, wrapper)

    assert resolved.current_index[0] >= 0, (
        f"Resolved current_index has negative value: {resolved.current_index[0]}"
    )


# =====================================================================
# Bug 3: _calc_hist_bins crashes on float data
#
# _calc_hist_bins calls np.iinfo(data.dtype).max, which raises
# ValueError for float dtypes.
# =====================================================================


def test_calc_hist_bins_float32() -> None:
    """_calc_hist_bins should not crash on float32 data."""
    data = np.random.rand(100, 100).astype(np.float32)
    counts, edges = _calc_hist_bins(data)
    assert len(counts) > 0
    assert len(edges) == len(counts) + 1


def test_calc_hist_bins_float64() -> None:
    """_calc_hist_bins should not crash on float64 data."""
    data = np.random.rand(50, 50)
    counts, _edges = _calc_hist_bins(data)
    assert len(counts) > 0


# =====================================================================
# Bug 4: COLOR mode data keyed as None, but None LUT view is hidden
#
# In COLOR mode, build_slice_requests sets c_ax=None (COLOR is not
# multichannel), so data arrives with key=None. But
# _update_lut_visibility hides the key=None LUT view in COLOR mode.
# The integer-keyed LUT views are shown but have no data.
#
# Expected: data for the current channel should use the per-channel
# LUT (keyed by the channel index from the slider).
# =====================================================================


def test_color_mode_data_keyed_by_channel_index() -> None:
    """In COLOR mode, data response should be keyed by channel index, not None."""
    data = np.ones((3, 10, 10), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(
        visible_axes=(-2, -1),
        channel_axis=0,
        channel_mode=ChannelMode.COLOR,
    )
    resolved = resolve(model, wrapper)
    reqs = build_slice_requests(resolved, wrapper)
    response = process_request(reqs[0])

    assert None not in response.data, (
        "COLOR mode data should not use key=None. "
        "It should use the channel index so the correct per-channel "
        "LUT is applied."
    )


# =====================================================================
# Bug 5 (discussion): channel_axis guessed in GRAYSCALE mode
#
# _norm_channel_axis always calls wrapper.guess_channel_axis() when
# model.channel_axis is None, even in GRAYSCALE mode. The guessed
# axis is mostly inert, but causes unnecessary data fetches when
# it changes and calls _push_fallback_channel_names unnecessarily.
#
# This might be intentional (pre-caching the guess for mode switches).
# Flagged for discussion.
# =====================================================================


def test_grayscale_resolved_channel_axis_is_none() -> None:
    """In GRAYSCALE mode, resolved channel_axis should be None."""
    data = np.ones((3, 10, 10), dtype=np.uint8)
    wrapper = DataWrapper.create(data)
    model = ArrayDisplayModel(channel_mode=ChannelMode.GRAYSCALE)
    resolved = resolve(model, wrapper)

    assert resolved.channel_axis is None, (
        f"In GRAYSCALE mode with no explicit channel_axis, "
        f"resolved channel_axis should be None but got "
        f"{resolved.channel_axis}."
    )
