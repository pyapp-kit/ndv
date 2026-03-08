"""Unified image statistics computation for histogram and contrast limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ihist
import numpy as np

from ndv.models._lut_model import (
    ClimsManual,
    ClimsMinMax,
    ClimsPercentile,
    ClimsStdDev,
)

if TYPE_CHECKING:
    from ndv.models._lut_model import ClimPolicy

_EMPTY = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class ImageStats:
    """Result of computing image statistics."""

    counts: np.ndarray | None  # None when histogram not needed
    bin_edges: np.ndarray | None
    clims: tuple[float, float]


def compute_image_stats(
    data: np.ndarray,
    clim_policy: ClimPolicy,
    need_histogram: bool,
    significant_bits: int | None = None,
) -> ImageStats:
    """Compute histogram and/or contrast limits in a single optimized pass."""
    # Manual clims: skip expensive computation, only compute histogram if needed
    if isinstance(clim_policy, ClimsManual):
        counts, edges = (
            _compute_histogram(data, significant_bits)
            if need_histogram
            else (None, None)
        )
        return ImageStats(counts, edges, (clim_policy.min, clim_policy.max))

    is_small_int = data.dtype.kind in "iu" and data.dtype.itemsize * 8 <= 16

    # Fast path: minmax without histogram — direct nanmin/nanmax is fastest
    if isinstance(clim_policy, ClimsMinMax) and not need_histogram:
        return ImageStats(None, None, _nanminmax(data))

    # Fast path: float stddev without histogram — direct mean/std
    if isinstance(clim_policy, ClimsStdDev) and not need_histogram and not is_small_int:
        mean = (
            float(np.nanmean(data))
            if clim_policy.center is None
            else clim_policy.center
        )
        std = float(np.nanstd(data))
        clims = (mean - clim_policy.n_stdev * std, mean + clim_policy.n_stdev * std)
        return ImageStats(None, None, clims)

    # Build histogram, derive clims from counts
    if is_small_int:
        bits = significant_bits or data.dtype.itemsize * 8
        counts, edges = _compute_int_histogram(data, bits)
        clims = _clims_from_counts(counts, edges, clim_policy, data)
    else:
        mi, ma = _nanminmax(data)
        if mi == ma:
            clims = (mi, ma)
            counts = np.array([data.size], dtype=np.intp)
            edges = np.array([mi, ma + 1], dtype=np.float64)
        else:
            counts, edges = np.histogram(data.ravel(), bins=256, range=(mi, ma))
            clims = _clims_from_counts(counts, edges, clim_policy, data)
    return ImageStats(
        counts if need_histogram else None, edges if need_histogram else None, clims
    )


def _clims_from_counts(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    policy: ClimPolicy,
    data: np.ndarray,
) -> tuple[float, float]:
    """Derive contrast limits from pre-computed histogram counts."""
    if isinstance(policy, ClimsMinMax):
        return _minmax_from_counts(counts, bin_edges)
    if isinstance(policy, ClimsPercentile):
        return _percentile_from_counts(
            counts, bin_edges, policy.min_percentile, policy.max_percentile
        )
    if isinstance(policy, ClimsStdDev):
        return _stddev_from_counts(counts, bin_edges, policy.n_stdev, policy.center)
    return policy.get_limits(data)


def _nanminmax(data: np.ndarray) -> tuple[float, float]:
    return (float(np.nanmin(data)), float(np.nanmax(data)))


def _compute_histogram(
    data: np.ndarray, significant_bits: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram, choosing best strategy based on dtype."""
    if data.dtype.kind in "iu" and data.dtype.itemsize * 8 <= 16:
        bits = significant_bits or data.dtype.itemsize * 8
        return _compute_int_histogram(data, bits)
    mi, ma = _nanminmax(data)
    if mi == ma:
        return (
            np.array([data.size], dtype=np.intp),
            np.array([mi, ma + 1], dtype=np.float64),
        )
    return np.histogram(data.ravel(), bins=256, range=(mi, ma))


def _compute_int_histogram(
    data: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram for integer data using ihist (uint8/uint16) or bincount."""
    nbins = 1 << bits  # 2^bits
    if data.dtype.kind == "u" and bits <= 16:
        counts = ihist.histogram(data, bits=bits)
    else:
        counts = np.bincount(data.ravel(), minlength=nbins)
        if len(counts) > nbins:
            counts = counts[:nbins]
    bin_edges = np.arange(nbins + 1, dtype=np.float64) - 0.5
    return counts, bin_edges


def resolve_significant_bits(data: np.ndarray) -> int | None:
    """Infer significant bits from integer data."""
    if data.dtype.kind not in "iu":
        return None
    full_bits = data.dtype.itemsize * 8
    if full_bits > 16:
        return None
    if data.size == 0:
        return full_bits
    dmax = int(data.max())
    if dmax <= 0 or dmax >= np.iinfo(data.dtype).max:
        return full_bits
    inferred = int(np.log2(dmax)) + 1
    # Round up to common bit depths
    for common in (8, 10, 12, 14, 16):
        if inferred <= common <= full_bits:
            return common
    return full_bits


# --- helpers to derive clims from histogram counts ---


def _minmax_from_counts(
    counts: np.ndarray, bin_edges: np.ndarray
) -> tuple[float, float]:
    nonzero = np.nonzero(counts)[0]
    if len(nonzero) == 0:
        return _EMPTY
    return (float(bin_edges[nonzero[0]]), float(bin_edges[nonzero[-1] + 1]))


def _percentile_from_counts(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[float, float]:
    cdf = np.cumsum(counts)
    total = cdf[-1]
    if total == 0:
        return _EMPTY
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    lo_idx = min(np.searchsorted(cdf, total * lo / 100.0), len(centers) - 1)
    hi_idx = min(np.searchsorted(cdf, total * hi / 100.0), len(centers) - 1)
    return (float(centers[lo_idx]), float(centers[hi_idx]))


def _stddev_from_counts(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    n_stdev: float,
    center: float | None,
) -> tuple[float, float]:
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    total = counts.sum()
    if total == 0:
        return _EMPTY
    weights = counts.astype(np.float64)
    mean = float(np.dot(centers, weights) / total) if center is None else center
    std = np.sqrt(float(np.dot((centers - mean) ** 2, weights) / total))
    return (mean - n_stdev * std, mean + n_stdev * std)
