from __future__ import annotations

import math

import cmap
import numpy as np
import pytest
from vispy.color import Color

from ndv.histogram.views._vispy import VispyHistogramView

# Accounts for differences between 32-bit and 64-bit floats
EPSILON = 1e-6
# FIXME: Why do plot checks need a larger epsilon?
PLOT_EPSILON = 1e-4


@pytest.fixture
def data() -> np.ndarray:
    gen = np.random.default_rng(seed=0xDEADBEEF)
    return gen.normal(10, 10, 10000).astype(np.float64)


@pytest.fixture
def view(data: np.ndarray) -> VispyHistogramView:
    view = VispyHistogramView()
    values, bin_edges = np.histogram(data)
    view.set_histogram(values, bin_edges)
    return view


def test_clims(data: np.ndarray, view: VispyHistogramView) -> None:
    # on startup, clims should be at the extent of the data
    clims = np.min(data), np.max(data)
    assert clims[0] == view._clims[0]
    assert clims[1] == view._clims[1]
    assert abs(clims[0] - view._lut_line._line.pos[0, 0]) <= EPSILON
    assert abs(clims[1] - view._lut_line._line.pos[-1, 0]) <= EPSILON
    # set clims, assert a change
    clims = 9, 11
    view.set_clims(clims)
    assert clims[0] == view._clims[0]
    assert clims[1] == view._clims[1]
    assert abs(clims[0] - view._lut_line._line.pos[0, 0]) <= EPSILON
    assert abs(clims[1] - view._lut_line._line.pos[-1, 0]) <= EPSILON
    # set clims backwards - ensure the view flips them
    clims = 5, 3
    view.set_clims(clims)
    assert clims[1] == view._clims[0]
    assert clims[0] == view._clims[1]
    assert abs(clims[1] - view._lut_line._line.pos[0, 0]) <= EPSILON
    assert abs(clims[0] - view._lut_line._line.pos[-1, 0]) <= EPSILON


def test_gamma(data: np.ndarray, view: VispyHistogramView) -> None:
    # on startup, gamma should be 1
    assert 1 == view._gamma
    gx, gy = (np.max(data) + np.min(data)) / 2, 0.5**view._gamma
    assert abs(gx - view._gamma_handle_position[0, 0]) <= EPSILON
    assert abs(gy - view._gamma_handle_position[0, 1]) <= EPSILON
    # set gamma, assert a change
    g = 2
    view.set_gamma(g)
    assert g == view._gamma
    gx, gy = (np.max(data) + np.min(data)) / 2, 0.5**view._gamma
    assert abs(gx - view._gamma_handle_position[0, 0]) <= EPSILON
    assert abs(gy - view._gamma_handle_position[0, 1]) <= EPSILON
    # set invalid gammas, assert no change
    with pytest.raises(ValueError):
        view.set_gamma(-1)


def test_cmap(view: VispyHistogramView) -> None:
    # By default, histogram is red
    assert view._hist.color == Color("red")
    # Set cmap, assert a change
    view.set_cmap(cmap.Colormap("blue"))
    assert view._hist.color == Color("blue")


def test_visibility(view: VispyHistogramView) -> None:
    # By default, everything is visible
    assert view._hist.visible
    assert view._lut_line.visible
    assert view._gamma_handle.visible
    assert view._lut_handles.visible
    # Visible = False
    view.set_visibility(False)
    assert not view._hist.visible
    assert not view._lut_line.visible
    assert not view._gamma_handle.visible
    assert not view._lut_handles.visible
    # Visible = True
    view.set_visibility(True)
    assert view._hist.visible
    assert view._lut_line.visible
    assert view._gamma_handle.visible
    assert view._lut_handles.visible


def test_domain(data: np.ndarray, view: VispyHistogramView) -> None:
    def assert_extent(min_x: float, max_x: float) -> None:
        domain = view.plot.xaxis.axis.domain
        assert abs(min_x - domain[0]) <= PLOT_EPSILON
        assert abs(max_x - domain[1]) <= PLOT_EPSILON
        min_y, max_y = 0, np.max(np.histogram(data)[0])
        range = view.plot.yaxis.axis.domain  # noqa: A001
        assert abs(min_y - range[0]) <= PLOT_EPSILON
        assert abs(max_y - range[1]) <= PLOT_EPSILON

    # By default, the view should be around the histogram
    assert_extent(np.min(data), np.max(data))
    # Set the domain, request a change
    new_domain = (10, 12)
    view.set_domain(new_domain)
    assert_extent(*new_domain)
    # Set the domain to None, assert going back
    new_domain = None
    view.set_domain(new_domain)
    assert_extent(np.min(data), np.max(data))
    # Assert None value in tuple raises ValueError
    with pytest.raises(ValueError):
        view.set_domain((None, 12))
    # Set the domain with min>max, ensure values flipped
    new_domain = (12, 10)
    view.set_domain(new_domain)
    assert_extent(10, 12)


def test_range(data: np.ndarray, view: VispyHistogramView) -> None:
    # FIXME: Why do we need a larger epsilon?
    _EPSILON = 1e-4

    def assert_extent(min_y: float, max_y: float) -> None:
        min_x, max_x = np.min(data), np.max(data)
        domain = view.plot.xaxis.axis.domain
        assert abs(min_x - domain[0]) <= _EPSILON
        assert abs(max_x - domain[1]) <= _EPSILON
        range = view.plot.yaxis.axis.domain  # noqa: A001
        assert abs(min_y - range[0]) <= _EPSILON
        assert abs(max_y - range[1]) <= _EPSILON

    # By default, the view should be around the histogram
    assert_extent(0, np.max(np.histogram(data)[0]))
    # Set the range, request a change
    new_range = (10, 12)
    view.set_range(new_range)
    assert_extent(*new_range)
    # Set the range to None, assert going back
    new_range = None
    view.set_range(new_range)
    assert_extent(0, np.max(np.histogram(data)[0]))
    # Assert None value in tuple raises ValueError
    with pytest.raises(ValueError):
        view.set_range((None, 12))
    # Set the range with min>max, ensure values flipped
    new_range = (12, 10)
    view.set_range(new_range)
    assert_extent(10, 12)


def test_vertical(view: VispyHistogramView) -> None:
    # Start out Horizontal
    assert not view._vertical
    domain_before = view.plot.xaxis.axis.domain
    range_before = view.plot.yaxis.axis.domain
    # Toggle vertical, assert domain <-> range
    view.set_vertical(True)
    assert view._vertical
    domain_after = view.plot.xaxis.axis.domain
    range_after = view.plot.yaxis.axis.domain
    assert abs(domain_before[0] - range_after[0]) <= PLOT_EPSILON
    assert abs(domain_before[1] - range_after[1]) <= PLOT_EPSILON
    assert abs(range_before[0] - domain_after[0]) <= PLOT_EPSILON
    assert abs(range_before[1] - domain_after[1]) <= PLOT_EPSILON
    # Toggle vertical again, assert domain <-> range again
    view.set_vertical(False)
    assert not view._vertical
    domain_after = view.plot.xaxis.axis.domain
    range_after = view.plot.yaxis.axis.domain
    assert abs(domain_before[0] - domain_after[0]) <= PLOT_EPSILON
    assert abs(domain_before[1] - domain_after[1]) <= PLOT_EPSILON
    assert abs(range_before[0] - range_after[0]) <= PLOT_EPSILON
    assert abs(range_before[1] - range_after[1]) <= PLOT_EPSILON


def test_log(view: VispyHistogramView) -> None:
    # Start out linear
    assert not view._log_y
    linear_range = view.plot.yaxis.axis.domain[1]
    linear_hist = view._hist.bounds(1)[1]
    # lut line, gamma markers controlled by scale
    linear_line_scale = view._handle_transform.scale[1]

    # Toggle log, assert range shrinks
    view.set_range_log(True)
    assert view._log_y
    log_range = view.plot.yaxis.axis.domain[1]
    log_hist = view._hist.bounds(1)[1]
    log_line_scale = view._handle_transform.scale[1]
    assert abs(math.log10(linear_range) - log_range) <= EPSILON
    assert abs(math.log10(linear_hist) - log_hist) <= EPSILON
    # NB This final check isn't so simple because of margins, scale checks,
    # etc - so need a larger epsilon.
    assert abs(math.log10(linear_line_scale) - log_line_scale) <= 0.1

    # Toggle log, assert range reverts
    view.set_range_log(False)
    assert not view._log_y
    revert_range = view.plot.yaxis.axis.domain[1]
    revert_hist = view._hist.bounds(1)[1]
    revert_line_scale = view._handle_transform.scale[1]
    assert abs(linear_range - revert_range) <= EPSILON
    assert abs(linear_hist - revert_hist) <= EPSILON
    assert abs(linear_line_scale - revert_line_scale) <= EPSILON
