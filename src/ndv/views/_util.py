"""Shared utilities for canvas backends."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("ndv")


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
