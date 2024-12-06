from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cmap
import numpy as np
import pytest
from vispy.color import Color

from ndv.models._stats import Stats
from ndv.views._vispy._vispy import Grabbable, VispyHistogramCanvas

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

# Accounts for differences between 32-bit and 64-bit floats
EPSILON = 1e-6
# FIXME: Why do plot checks need a larger epsilon?
PLOT_EPSILON = 1e-4


@pytest.fixture
def stats() -> Stats:
    gen = np.random.default_rng(seed=0xDEADBEEF)
    data = gen.normal(10, 10, 10000).astype(np.float64)
    return Stats(data)


@pytest.fixture
def view(stats: Stats) -> VispyHistogramCanvas:
    # Create view
    view = VispyHistogramCanvas()
    view._canvas.size = (100, 100)
    # Set statistics
    view.set_stats(stats)

    return view


def test_plot(view: VispyHistogramCanvas) -> None:
    plot = view.plot

    assert plot.title == ""
    plot.title = "foo"
    assert plot._title.text == "foo"

    assert plot.xlabel == ""
    plot.xlabel = "bar"
    assert plot._xlabel.text == "bar"

    assert plot.ylabel == ""
    plot.ylabel = "baz"
    assert plot._ylabel.text == "baz"

    # Test axis lock - pan
    _domain = plot.xaxis.axis.domain
    _range = plot.yaxis.axis.domain
    plot.camera.pan([20, 20])
    assert np.all(np.isclose(_domain, [x - 20 for x in plot.xaxis.axis.domain]))
    assert np.all(np.isclose(_range, plot.yaxis.axis.domain))

    # Test axis lock - zoom
    _domain = plot.xaxis.axis.domain
    _range = plot.yaxis.axis.domain
    plot.camera.zoom(0.5)
    dx = (_domain[1] - _domain[0]) / 4
    assert np.all(
        np.isclose([_domain[0] + dx, _domain[1] - dx], plot.xaxis.axis.domain)
    )
    assert np.all(np.isclose(_range, plot.yaxis.axis.domain))


def test_clims(stats: Stats, view: VispyHistogramCanvas) -> None:
    # on startup, clims should be at the extent of the data
    clims = stats.minimum, stats.maximum
    assert view._clims is not None
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


def test_gamma(stats: Stats, view: VispyHistogramCanvas) -> None:
    # on startup, gamma should be 1
    assert 1 == view._gamma
    gx, gy = (stats.minimum + stats.maximum) / 2, 0.5**view._gamma
    assert abs(gx - view._gamma_handle_pos[0, 0]) <= EPSILON
    assert abs(gy - view._gamma_handle_pos[0, 1]) <= EPSILON
    # set gamma, assert a change
    g = 2
    view.set_gamma(g)
    assert g == view._gamma
    gx, gy = (stats.minimum + stats.maximum) / 2, 0.5**view._gamma
    assert abs(gx - view._gamma_handle_pos[0, 0]) <= EPSILON
    assert abs(gy - view._gamma_handle_pos[0, 1]) <= EPSILON
    # set invalid gammas, assert no change
    with pytest.raises(ValueError):
        view.set_gamma(-1)


def test_cmap(view: VispyHistogramCanvas) -> None:
    # By default, histogram is red
    assert view._hist_mesh.color == Color("red")
    # Set cmap, assert a change
    view.set_colormap(cmap.Colormap("blue"))
    assert view._hist_mesh.color == Color("blue")


def test_visibility(view: VispyHistogramCanvas) -> None:
    # By default, the lut components are invisible
    assert view._hist_mesh.visible
    assert not view._lut_line.visible
    assert not view._gamma_handle.visible
    # Make them visible
    view.set_lut_visible(True)
    assert view._hist_mesh.visible
    assert view._lut_line.visible
    assert view._gamma_handle.visible
    # Make them invisible again
    view.set_lut_visible(False)
    assert view._hist_mesh.visible
    assert not view._lut_line.visible
    assert not view._gamma_handle.visible


def test_domain(stats: Stats, view: VispyHistogramCanvas) -> None:
    def assert_extent(min_x: float, max_x: float) -> None:
        domain = view.plot.xaxis.axis.domain
        assert abs(min_x - domain[0]) <= PLOT_EPSILON
        assert abs(max_x - domain[1]) <= PLOT_EPSILON
        min_y, max_y = 0, np.max(stats.histogram[0])
        range = view.plot.yaxis.axis.domain  # noqa: A001
        assert abs(min_y - range[0]) <= PLOT_EPSILON
        assert abs(max_y - range[1]) <= PLOT_EPSILON

    # By default, the view should be around the histogram
    assert_extent(stats.minimum, stats.maximum)
    # Set the domain, request a change
    new_domain = (10, 12)
    view.set_domain(new_domain)
    assert_extent(*new_domain)
    # Set the domain to None, assert going back
    new_domain = None
    view.set_domain(new_domain)
    assert_extent(stats.minimum, stats.maximum)
    # Assert None value in tuple raises ValueError
    with pytest.raises(ValueError):
        view.set_domain((None, 12))
    # Set the domain with min>max, ensure values flipped
    new_domain = (12, 10)
    view.set_domain(new_domain)
    assert_extent(10, 12)


def test_range(stats: Stats, view: VispyHistogramCanvas) -> None:
    # FIXME: Why do we need a larger epsilon?
    _EPSILON = 1e-4

    def assert_extent(min_y: float, max_y: float) -> None:
        min_x, max_x = stats.minimum, stats.maximum
        domain = view.plot.xaxis.axis.domain
        assert abs(min_x - domain[0]) <= _EPSILON
        assert abs(max_x - domain[1]) <= _EPSILON
        range = view.plot.yaxis.axis.domain  # noqa: A001
        assert abs(min_y - range[0]) <= _EPSILON
        assert abs(max_y - range[1]) <= _EPSILON

    # By default, the view should be around the histogram
    assert_extent(0, np.max(stats.histogram[0]))
    # Set the range, request a change
    new_range = (10, 12)
    view.set_range(new_range)
    assert_extent(*new_range)
    # Set the range to None, assert going back
    new_range = None
    view.set_range(new_range)
    assert_extent(0, np.max(stats.histogram[0]))
    # Assert None value in tuple raises ValueError
    with pytest.raises(ValueError):
        view.set_range((None, 12))
    # Set the range with min>max, ensure values flipped
    new_range = (12, 10)
    view.set_range(new_range)
    assert_extent(10, 12)


def test_vertical(view: VispyHistogramCanvas) -> None:
    # Start out Horizontal
    assert not view._vertical
    domain_before = view.plot.xaxis.axis.domain
    range_before = view.plot.yaxis.axis.domain
    # Toggle vertical, assert domain <-> range
    view.set_vertical(True)
    assert view._vertical
    domain_after = view.plot.xaxis.axis.domain
    # NB vertical mode inverts y axis
    range_after = view.plot.yaxis.axis.domain[::-1]
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


def test_log(view: VispyHistogramCanvas) -> None:
    # Start out linear
    assert not view._log_y
    linear_range = view.plot.yaxis.axis.domain[1]
    linear_hist = view._hist_mesh.bounds(1)[1]
    # lut line, gamma markers controlled by scale
    linear_line_scale = view._handle_transform.scale[1]

    # Toggle log, assert range shrinks
    view.set_range_log(True)
    assert view._log_y
    log_range = view.plot.yaxis.axis.domain[1]
    log_hist = view._hist_mesh.bounds(1)[1]
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
    revert_hist = view._hist_mesh.bounds(1)[1]
    revert_line_scale = view._handle_transform.scale[1]
    assert abs(linear_range - revert_range) <= EPSILON
    assert abs(linear_hist - revert_hist) <= EPSILON
    assert abs(linear_line_scale - revert_line_scale) <= EPSILON


# @pytest.mark.skipif(sys.platform != "darwin", reason="the mouse event is tricky")
def test_move_clim(qtbot: QtBot, view: VispyHistogramCanvas) -> None:
    # Set clims within the viewbox
    view.set_domain((0, 100))
    view.set_clims((10, 90))
    # Click on the left clim
    press_pos = view.node_tform.imap([10])[:2]
    view.on_mouse_press(press_pos)
    assert view._grabbed == Grabbable.LEFT_CLIM
    assert not view.plot.camera.interactive
    # Move it to 50
    move_pos = view.node_tform.imap([50])[:2]
    with qtbot.waitSignal(view.climsChanged):
        view.on_mouse_move(move_pos)
    assert view._grabbed == Grabbable.LEFT_CLIM
    assert not view.plot.camera.interactive
    # Release mouse
    release_pos = move_pos
    view.on_mouse_release(release_pos)
    assert view._grabbed == Grabbable.NONE
    assert view.plot.camera.interactive

    # Move both clims to 50
    view.set_clims((50, 50))
    # Ensure clicking and moving at 50 moves the right clim
    press_pos = view.node_tform.imap([50])[:2]
    view.on_mouse_press(press_pos)
    assert view._grabbed == Grabbable.RIGHT_CLIM
    assert not view.plot.camera.interactive
    # Move it to 70
    move_pos = view.node_tform.imap([70])[:2]
    with qtbot.waitSignal(view.climsChanged):
        view.on_mouse_move(move_pos)
    assert view._grabbed == Grabbable.RIGHT_CLIM
    assert not view.plot.camera.interactive
    # Release mouse
    release_pos = move_pos
    view.on_mouse_release(release_pos)
    assert view._grabbed == Grabbable.NONE
    assert view.plot.camera.interactive


def test_move_gamma(qtbot: QtBot, view: VispyHistogramCanvas) -> None:
    # Set clims outside the viewbox
    # NB the canvas is small in this test, so we have to put the clims
    # far away or they'll be grabbed over the gamma
    view.set_domain((0, 100))
    view.set_clims((-9950, 10050))
    # Click on the gamma handle
    press_pos = view.node_tform.imap(view._handle_transform.map([50, 0.5]))[:2]
    view.on_mouse_press(press_pos)
    assert view._grabbed == Grabbable.GAMMA
    assert not view.plot.camera.interactive
    # Move it to 50
    move_pos = view.node_tform.imap(view._handle_transform.map([50, 0.75]))[:2]
    with qtbot.waitSignal(view.gammaChanged):
        view.on_mouse_move(move_pos)
    assert view._grabbed == Grabbable.GAMMA
    assert not view.plot.camera.interactive
    # Release mouse
    release_pos = move_pos
    view.on_mouse_release(release_pos)
    assert view._grabbed == Grabbable.NONE
    assert view.plot.camera.interactive
