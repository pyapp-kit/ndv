from __future__ import annotations

import numpy as np
import pytest

from ndv.controllers._image_stats import (
    _compute_int_histogram,
    _minmax_from_counts,
    _percentile_from_counts,
    _stddev_from_counts,
    compute_image_stats,
)
from ndv.models._lut_model import ClimsManual, ClimsMinMax, ClimsPercentile, ClimsStdDev

RNG = np.random.default_rng(42)


@pytest.mark.parametrize(
    "data, bits, expected_nbins, check_indices",
    [
        (np.array([[0, 1, 2, 255]], dtype=np.uint8), 8, 256, [0, 255]),
        (np.array([[0, 100, 4095]], dtype=np.uint16), 12, 4096, [0, 100, 4095]),
    ],
)
def test_compute_int_histogram(
    data: np.ndarray, bits: int, expected_nbins: int, check_indices: list[int]
) -> None:
    counts, edges = _compute_int_histogram(data, bits=bits)
    assert counts.shape == (expected_nbins,)
    assert edges.shape == (expected_nbins + 1,)
    for idx in check_indices:
        assert counts[idx] == 1


@pytest.mark.parametrize(
    "counts, edges, expected",
    [
        (
            np.array([0, 0, 5, 3, 0, 2, 0]),
            np.arange(8, dtype=np.float64) - 0.5,
            (2.0, 5.0),
        ),
        (np.zeros(10, dtype=np.int64), np.arange(11, dtype=np.float64), (0.0, 0.0)),
    ],
)
def test_minmax_from_counts(
    counts: np.ndarray, edges: np.ndarray, expected: tuple[float, float]
) -> None:
    assert _minmax_from_counts(counts, edges) == pytest.approx(expected)


def test_percentile_from_counts() -> None:
    counts = np.zeros(256, dtype=np.int64)
    counts[10] = 50
    counts[200] = 50
    edges = np.arange(257, dtype=np.float64) - 0.5
    assert _percentile_from_counts(counts, edges, 25, 75) == pytest.approx(
        (10.0, 200.0)
    )


def test_stddev_from_counts() -> None:
    counts = np.zeros(256, dtype=np.int64)
    counts[128] = 1000
    edges = np.arange(257, dtype=np.float64) - 0.5
    # All data in one bin => std ≈ 0, so lo ≈ hi ≈ center
    assert _stddev_from_counts(counts, edges, n_stdev=2, center=None) == pytest.approx(
        (128.0, 128.0)
    )


@pytest.mark.parametrize("need_histogram", [False, True])
def test_manual_clims(need_histogram: bool) -> None:
    data = RNG.integers(0, 255, (100, 100), dtype=np.uint8)
    stats = compute_image_stats(
        data, ClimsManual(min=10, max=200), need_histogram=need_histogram
    )
    assert stats.clims == (10, 200)
    assert (stats.counts is not None) == need_histogram
    assert (stats.bin_edges is not None) == need_histogram


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("need_histogram", [False, True])
def test_minmax(dtype: np.dtype, need_histogram: bool) -> None:
    data = np.array([10, 20, 30, 40, 50], dtype=dtype).reshape(1, -1)
    stats = compute_image_stats(data, ClimsMinMax(), need_histogram=need_histogram)
    assert stats.clims[0] == pytest.approx(10, abs=1)
    assert stats.clims[1] == pytest.approx(50, abs=1)
    assert (stats.counts is not None) == need_histogram


@pytest.mark.parametrize(
    "data, policy, abs_tol, sig_bits",
    [
        (
            RNG.integers(0, 4095, (512, 512), dtype=np.uint16),
            ClimsPercentile(min_percentile=1, max_percentile=99),
            50,
            12,
        ),
        (
            RNG.random((100, 100), dtype=np.float32),
            ClimsPercentile(min_percentile=5, max_percentile=95),
            0.05,
            None,
        ),
        (
            np.array([-5, -1, 0, 3], dtype=np.int16),
            ClimsPercentile(min_percentile=1, max_percentile=99),
            1.0,
            None,
        ),
    ],
    ids=["uint16-12bit", "float32", "int16-negative"],
)
def test_percentile(
    data: np.ndarray,
    policy: ClimsPercentile,
    abs_tol: float,
    sig_bits: int | None,
) -> None:
    kwargs = {"significant_bits": sig_bits} if sig_bits else {}
    stats = compute_image_stats(data, policy, need_histogram=False, **kwargs)
    ref = tuple(np.nanpercentile(data, [policy.min_percentile, policy.max_percentile]))
    assert stats.clims[0] == pytest.approx(ref[0], abs=abs_tol)
    assert stats.clims[1] == pytest.approx(ref[1], abs=abs_tol)


@pytest.mark.parametrize(
    "data, abs_tol",
    [
        (RNG.integers(0, 255, (100, 100), dtype=np.uint8), 5),
        (RNG.random((100, 100), dtype=np.float32), 0.01),
    ],
    ids=["uint8", "float32"],
)
def test_stddev(data: np.ndarray, abs_tol: float) -> None:
    stats = compute_image_stats(data, ClimsStdDev(n_stdev=2), need_histogram=False)
    ref_mean = float(np.nanmean(data))
    ref_std = float(np.nanstd(data))
    assert stats.clims[0] == pytest.approx(ref_mean - 2 * ref_std, abs=abs_tol)
    assert stats.clims[1] == pytest.approx(ref_mean + 2 * ref_std, abs=abs_tol)


@pytest.mark.parametrize(
    "data, expected_clims",
    [
        (np.full((10, 10), 42, dtype=np.uint8), (42, 42)),
        (np.full((10, 10), 3.14, dtype=np.float32), (3.14, 3.14)),
    ],
    ids=["uint8", "float32"],
)
def test_single_value_data(
    data: np.ndarray, expected_clims: tuple[float, float]
) -> None:
    stats = compute_image_stats(data, ClimsMinMax(), need_histogram=True)
    assert stats.clims == pytest.approx(expected_clims, abs=1)
    assert stats.counts is not None


def test_all_nan_float() -> None:
    data = np.full((10, 10), np.nan, dtype=np.float32)
    with pytest.warns(RuntimeWarning):
        stats = compute_image_stats(data, ClimsMinMax(), need_histogram=False)
    assert np.isnan(stats.clims[0])


def test_significant_bits_passed_through() -> None:
    """12-bit significant_bits should produce 4096-bin histogram."""
    data = np.array([[0, 2000, 4000]], dtype=np.uint16)
    stats = compute_image_stats(
        data, ClimsMinMax(), need_histogram=True, significant_bits=12
    )
    assert stats.counts is not None
    assert stats.counts.shape == (4096,)
