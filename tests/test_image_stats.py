from __future__ import annotations

import numpy as np
import pytest

from ndv.controllers._image_stats import (
    _compute_int_histogram,
    _minmax_from_counts,
    _percentile_from_counts,
    _stddev_from_counts,
    compute_image_stats,
    resolve_significant_bits,
)
from ndv.models._lut_model import (
    ClimsManual,
    ClimsMinMax,
    ClimsPercentile,
    ClimsStdDev,
)

# --- resolve_significant_bits ---


def test_resolve_significant_bits_uint8():
    data = np.array([0, 100, 200], dtype=np.uint8)
    assert resolve_significant_bits(data) == 8


def test_resolve_significant_bits_uint16_12bit():
    data = np.array([0, 2000, 4000], dtype=np.uint16)
    assert resolve_significant_bits(data) == 12


def test_resolve_significant_bits_uint16_full():
    data = np.array([0, 65535], dtype=np.uint16)
    assert resolve_significant_bits(data) == 16


def test_resolve_significant_bits_float():
    data = np.array([0.0, 1.0], dtype=np.float32)
    assert resolve_significant_bits(data) is None


def test_resolve_significant_bits_empty():
    data = np.array([], dtype=np.uint16)
    assert resolve_significant_bits(data) == 16


def test_resolve_significant_bits_all_zero():
    data = np.zeros(100, dtype=np.uint16)
    assert resolve_significant_bits(data) == 16


# --- _compute_int_histogram ---


def test_compute_int_histogram_uint8():
    data = np.array([[0, 1, 2, 255]], dtype=np.uint8)
    counts, edges = _compute_int_histogram(data, bits=8)
    assert counts.shape == (256,)
    assert edges.shape == (257,)
    assert counts[0] == 1
    assert counts[255] == 1


def test_compute_int_histogram_12bit():
    data = np.array([[0, 100, 4095]], dtype=np.uint16)
    counts, _edges = _compute_int_histogram(data, bits=12)
    assert counts.shape == (4096,)
    assert counts[0] == 1
    assert counts[100] == 1
    assert counts[4095] == 1


# --- helpers: derive clims from counts ---


def test_minmax_from_counts():
    counts = np.array([0, 0, 5, 3, 0, 2, 0])
    edges = np.arange(8, dtype=np.float64) - 0.5
    mi, ma = _minmax_from_counts(counts, edges)
    assert mi == pytest.approx(1.5)
    assert ma == pytest.approx(5.5)


def test_minmax_from_counts_empty():
    counts = np.zeros(10, dtype=np.int64)
    edges = np.arange(11, dtype=np.float64)
    assert _minmax_from_counts(counts, edges) == (0.0, 0.0)


def test_percentile_from_counts():
    counts = np.zeros(256, dtype=np.int64)
    counts[10] = 50
    counts[200] = 50
    edges = np.arange(257, dtype=np.float64) - 0.5
    lo, hi = _percentile_from_counts(counts, edges, 25, 75)
    # 25th percentile falls in bin 10, 75th in bin 200
    assert lo == pytest.approx(10.0)
    assert hi == pytest.approx(200.0)


def test_stddev_from_counts():
    # Uniform data in a single bin
    counts = np.zeros(256, dtype=np.int64)
    counts[128] = 1000
    edges = np.arange(257, dtype=np.float64) - 0.5
    lo, hi = _stddev_from_counts(counts, edges, n_stdev=2, center=None)
    # With all data in one bin, std ≈ 0, so lo ≈ hi ≈ center
    assert lo == pytest.approx(128.0)
    assert hi == pytest.approx(128.0)


# --- compute_image_stats: manual ---


def test_manual_clims_no_histogram():
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    policy = ClimsManual(min=10, max=200)
    stats = compute_image_stats(data, policy, need_histogram=False)
    assert stats.clims == (10, 200)
    assert stats.counts is None
    assert stats.bin_edges is None


def test_manual_clims_with_histogram():
    data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    policy = ClimsManual(min=10, max=200)
    stats = compute_image_stats(data, policy, need_histogram=True)
    assert stats.clims == (10, 200)
    assert stats.counts is not None
    assert stats.bin_edges is not None


# --- compute_image_stats: minmax ---


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_minmax_no_histogram(dtype: np.dtype):
    data = np.array([10, 20, 30, 40, 50], dtype=dtype).reshape(1, -1)
    policy = ClimsMinMax()
    stats = compute_image_stats(data, policy, need_histogram=False)
    assert stats.clims[0] == pytest.approx(10)
    assert stats.clims[1] == pytest.approx(50)
    assert stats.counts is None


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_minmax_with_histogram(dtype: np.dtype):
    data = np.array([10, 20, 30, 40, 50], dtype=dtype).reshape(1, -1)
    policy = ClimsMinMax()
    stats = compute_image_stats(data, policy, need_histogram=True)
    assert stats.clims[0] == pytest.approx(10, abs=1)
    assert stats.clims[1] == pytest.approx(50, abs=1)
    assert stats.counts is not None


# --- compute_image_stats: percentile ---


def test_percentile_uint16():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 4095, (512, 512), dtype=np.uint16)
    policy = ClimsPercentile(min_percentile=1, max_percentile=99)
    stats = compute_image_stats(data, policy, need_histogram=False, significant_bits=12)
    ref = tuple(np.nanpercentile(data, [1, 99]))
    # Histogram-derived percentiles are approximate
    assert stats.clims[0] == pytest.approx(ref[0], abs=50)
    assert stats.clims[1] == pytest.approx(ref[1], abs=50)


def test_percentile_float32():
    rng = np.random.default_rng(42)
    data = rng.random((100, 100), dtype=np.float32)
    policy = ClimsPercentile(min_percentile=5, max_percentile=95)
    stats = compute_image_stats(data, policy, need_histogram=False)
    ref = tuple(np.nanpercentile(data, [5, 95]))
    assert stats.clims[0] == pytest.approx(ref[0], abs=0.05)
    assert stats.clims[1] == pytest.approx(ref[1], abs=0.05)


# --- compute_image_stats: stddev ---


def test_stddev_uint8():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    policy = ClimsStdDev(n_stdev=2)
    stats = compute_image_stats(data, policy, need_histogram=False)
    ref_mean = float(np.nanmean(data))
    ref_std = float(np.nanstd(data))
    assert stats.clims[0] == pytest.approx(ref_mean - 2 * ref_std, abs=5)
    assert stats.clims[1] == pytest.approx(ref_mean + 2 * ref_std, abs=5)


def test_stddev_float_no_histogram():
    rng = np.random.default_rng(42)
    data = rng.random((100, 100), dtype=np.float32)
    policy = ClimsStdDev(n_stdev=2)
    stats = compute_image_stats(data, policy, need_histogram=False)
    ref_mean = float(np.nanmean(data))
    ref_std = float(np.nanstd(data))
    assert stats.clims[0] == pytest.approx(ref_mean - 2 * ref_std, abs=0.01)
    assert stats.clims[1] == pytest.approx(ref_mean + 2 * ref_std, abs=0.01)


# --- edge cases ---


def test_single_value_data():
    data = np.full((10, 10), 42, dtype=np.uint8)
    policy = ClimsMinMax()
    stats = compute_image_stats(data, policy, need_histogram=True)
    assert stats.clims[0] == pytest.approx(42, abs=1)
    assert stats.clims[1] == pytest.approx(42, abs=1)
    assert stats.counts is not None


def test_all_nan_float():
    data = np.full((10, 10), np.nan, dtype=np.float32)
    policy = ClimsMinMax()
    # nanmin/nanmax will raise a warning for all-NaN slices
    with pytest.warns(RuntimeWarning):
        stats = compute_image_stats(data, policy, need_histogram=False)
    # nan clims are acceptable for all-nan data
    assert np.isnan(stats.clims[0])


def test_float_single_value():
    data = np.full((10, 10), 3.14, dtype=np.float32)
    policy = ClimsMinMax()
    stats = compute_image_stats(data, policy, need_histogram=True)
    assert stats.clims == pytest.approx((3.14, 3.14))


# --- ihist fallback ---


def test_significant_bits_passed_through():
    """12-bit significant_bits should produce 4096-bin histogram."""
    data = np.array([[0, 2000, 4000]], dtype=np.uint16)
    policy = ClimsMinMax()
    stats = compute_image_stats(data, policy, need_histogram=True, significant_bits=12)
    assert stats.counts is not None
    assert stats.counts.shape == (4096,)
