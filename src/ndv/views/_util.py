"""Shared utilities for canvas backends."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger("ndv")

LUT_LINE_ALPHA = 0.6


def downsample_data(
    data: np.ndarray, max_size: int, *, warn: bool = True
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Downsample data so no axis exceeds max_size.

    Returns the (possibly downsampled view) data and the per-axis stride factors.
    """
    factors = tuple(
        int(np.ceil(s / max_size)) if s > max_size else 1 for s in data.shape
    )
    if any(f > 1 for f in factors):
        if warn:
            logger.warning(
                "Data shape %s exceeds max texture dimension (%d) and will be "
                "downsampled for rendering (strides: %s).",
                data.shape,
                max_size,
                factors,
            )
        slices = tuple(slice(None, None, f) for f in factors)
        data = data[slices]
    return data, factors


# ------------ Histogram data helpers ------------ #


def downsample_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    max_display_bins: int = 800,
    visible_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample histogram for display, optionally cropping to visible range.

    Parameters
    ----------
    counts : np.ndarray
        Raw histogram counts.
    bin_edges : np.ndarray
        Raw bin edges (len = len(counts) + 1).
    max_display_bins : int
        Target number of bins after downsampling.
    visible_range : tuple[float, float] | None
        If provided, crop to this (x_lo, x_hi) range before downsampling.
        A small margin of extra bins is kept so panning feels seamless.
    """
    if visible_range is not None:
        counts, bin_edges = _crop_histogram(counts, bin_edges, *visible_range)

    n = len(counts)
    if n > max_display_bins:
        factor = n // max_display_bins
        trim = n - (n % factor)
        counts = counts[:trim].reshape(-1, factor).mean(axis=1)
        bin_edges = np.concatenate(
            [bin_edges[:trim:factor], bin_edges[trim : trim + 1]]
        )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, counts


def _crop_histogram(
    counts: np.ndarray,
    bin_edges: np.ndarray,
    x_lo: float,
    x_hi: float,
    margin_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop histogram to the visible range plus a margin for panning."""
    span = x_hi - x_lo
    margin = span * margin_frac
    lo = x_lo - margin
    hi = x_hi + margin

    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mask = (centers >= lo) & (centers <= hi)
    if not np.any(mask):
        return counts, bin_edges

    idx = np.nonzero(mask)[0]
    i0, i1 = idx[0], idx[-1] + 1
    return counts[i0:i1], bin_edges[i0 : i1 + 1]


def area_to_mesh(
    centers: np.ndarray,
    counts: np.ndarray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint32]]:
    """Convert area plot data to mesh vertices and faces (triangle strip)."""
    n = len(centers)
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32)

    # 2 vertices per point: one on the curve, one on the baseline
    vertices = np.zeros((2 * n, 3), np.float32)
    vertices[0::2, 0] = centers
    vertices[0::2, 1] = counts
    vertices[1::2, 0] = centers
    # vertices[1::2, 1] = 0  (baseline, already 0)

    # Vectorized face generation
    idx = np.arange(n - 1, dtype=np.uint32)
    faces = np.zeros((2 * (n - 1), 3), np.uint32)
    faces[0::2, 0] = 2 * idx  # top_left
    faces[0::2, 1] = 2 * idx + 1  # bot_left
    faces[0::2, 2] = 2 * idx + 2  # top_right
    faces[1::2, 0] = 2 * idx + 1  # bot_left
    faces[1::2, 1] = 2 * idx + 3  # bot_right
    faces[1::2, 2] = 2 * idx + 2  # top_right

    return vertices, faces


# ------------ Log scale ------------ #


def apply_log_counts(counts: np.ndarray, log_base: float | None) -> np.ndarray:
    """Apply log transform to counts if log_base is set."""
    if log_base:
        return np.log(counts + 1) / np.log(log_base)  # type: ignore[no-any-return]
    return counts
