"""Unified image statistics computation for histogram and contrast limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ndv.models._lut_model import ClimPolicy

try:
    import ihist as _ihist
except ImportError:
    _ihist = None


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
    from ndv.models._lut_model import (
        ClimsManual,
        ClimsMinMax,
        ClimsPercentile,
        ClimsStdDev,
    )

    # Manual clims: skip expensive computation, only compute histogram if needed
    if isinstance(clim_policy, ClimsManual):
        clims = (clim_policy.min, clim_policy.max)
        if need_histogram:
            counts, bin_edges = _compute_histogram(data, significant_bits)
            return ImageStats(counts=counts, bin_edges=bin_edges, clims=clims)
        return ImageStats(counts=None, bin_edges=None, clims=clims)

    is_int = data.dtype.kind in "iu"
    nbits = data.dtype.itemsize * 8

    if is_int and nbits <= 16:
        return _stats_for_small_int(data, clim_policy, need_histogram, significant_bits)

    # Float / large-int path
    if isinstance(clim_policy, ClimsStdDev) and not need_histogram:
        # Direct mean/std is fastest for stddev-only
        clims = _stddev_clims_direct(data, clim_policy.n_stdev, clim_policy.center)
        return ImageStats(counts=None, bin_edges=None, clims=clims)

    if isinstance(clim_policy, ClimsMinMax) and not need_histogram:
        clims = (float(np.nanmin(data)), float(np.nanmax(data)))
        return ImageStats(counts=None, bin_edges=None, clims=clims)

    # Need histogram (or need percentile from float data)
    mi, ma = float(np.nanmin(data)), float(np.nanmax(data))
    if mi == ma:
        clims = (mi, ma)
        counts = np.array([data.size], dtype=np.intp)
        bin_edges = np.array([mi, ma + 1], dtype=np.float64)
        return ImageStats(counts=counts, bin_edges=bin_edges, clims=clims)

    nbins = 256
    counts, bin_edges = np.histogram(data.ravel(), bins=nbins, range=(mi, ma))

    if isinstance(clim_policy, ClimsMinMax):
        clims = (mi, ma)
    elif isinstance(clim_policy, ClimsPercentile):
        clims = _percentile_from_counts(
            counts, bin_edges, clim_policy.min_percentile, clim_policy.max_percentile
        )
    elif isinstance(clim_policy, ClimsStdDev):
        clims = _stddev_from_counts(
            counts, bin_edges, clim_policy.n_stdev, clim_policy.center
        )
    else:
        # Fallback: use policy's own method
        clims = clim_policy.get_limits(data)

    if not need_histogram:
        return ImageStats(counts=None, bin_edges=None, clims=clims)
    return ImageStats(counts=counts, bin_edges=bin_edges, clims=clims)


def _stats_for_small_int(
    data: np.ndarray,
    clim_policy: ClimPolicy,
    need_histogram: bool,
    significant_bits: int | None,
) -> ImageStats:
    """Optimized path for uint8/uint16 data using integer histograms."""
    from ndv.models._lut_model import (
        ClimsMinMax,
        ClimsPercentile,
        ClimsStdDev,
    )

    # For minmax without histogram, direct nanmin/nanmax is fastest
    if isinstance(clim_policy, ClimsMinMax) and not need_histogram:
        clims = (float(np.nanmin(data)), float(np.nanmax(data)))
        return ImageStats(counts=None, bin_edges=None, clims=clims)

    # All other cases benefit from building a histogram
    bits = significant_bits or data.dtype.itemsize * 8
    counts, bin_edges = _compute_int_histogram(data, bits)

    if isinstance(clim_policy, ClimsMinMax):
        clims = _minmax_from_counts(counts, bin_edges)
    elif isinstance(clim_policy, ClimsPercentile):
        clims = _percentile_from_counts(
            counts, bin_edges, clim_policy.min_percentile, clim_policy.max_percentile
        )
    elif isinstance(clim_policy, ClimsStdDev):
        clims = _stddev_from_counts(
            counts, bin_edges, clim_policy.n_stdev, clim_policy.center
        )
    else:
        clims = clim_policy.get_limits(data)

    if not need_histogram:
        return ImageStats(counts=None, bin_edges=None, clims=clims)
    return ImageStats(counts=counts, bin_edges=bin_edges, clims=clims)


def _compute_histogram(
    data: np.ndarray, significant_bits: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram, choosing best strategy based on dtype."""
    if data.dtype.kind in "iu" and data.dtype.itemsize * 8 <= 16:
        bits = significant_bits or data.dtype.itemsize * 8
        return _compute_int_histogram(data, bits)
    # Float path
    mi, ma = float(np.nanmin(data)), float(np.nanmax(data))
    if mi == ma:
        return (
            np.array([data.size], dtype=np.intp),
            np.array([mi, ma + 1], dtype=np.float64),
        )
    return np.histogram(data.ravel(), bins=256, range=(mi, ma))


def _compute_int_histogram(
    data: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram for integer data using ihist or np.bincount."""
    nbins = 1 << bits  # 2^bits

    if _ihist is not None and data.dtype.kind == "u" and bits <= 16:
        counts = _ihist.histogram(data, bits=bits)
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
    if dmax <= 0:
        return full_bits
    dtype_max = np.iinfo(data.dtype).max
    if dmax >= dtype_max:
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
        return (0.0, 0.0)
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
        return (0.0, 0.0)
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    lo_idx = np.searchsorted(cdf, total * lo / 100.0)
    hi_idx = np.searchsorted(cdf, total * hi / 100.0)
    lo_idx = min(lo_idx, len(centers) - 1)
    hi_idx = min(hi_idx, len(centers) - 1)
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
        return (0.0, 0.0)
    weights = counts.astype(np.float64)
    if center is None:
        mean = float(np.dot(centers, weights) / total)
    else:
        mean = center
    var = float(np.dot((centers - mean) ** 2, weights) / total)
    std = np.sqrt(var)
    return (mean - n_stdev * std, mean + n_stdev * std)


def _stddev_clims_direct(
    data: np.ndarray, n_stdev: float, center: float | None
) -> tuple[float, float]:
    mean = float(np.nanmean(data)) if center is None else center
    std = float(np.nanstd(data))
    return (mean - n_stdev * std, mean + n_stdev * std)
