"""Shared utilities for histogram canvas implementations."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

MIN_GAMMA: np.float64 = np.float64(1e-6)
MAX_DISPLAY_BINS = 1024
FILL_ALPHA = 0.3
LUT_LINE_ALPHA = 0.6


class Grabbable(Enum):
    NONE = auto()
    LEFT_CLIM = auto()
    RIGHT_CLIM = auto()
    GAMMA = auto()


def downsample_histogram(
    counts: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample histogram to ~MAX_DISPLAY_BINS bins, return (centers, counts)."""
    n = len(counts)
    if n > MAX_DISPLAY_BINS:
        factor = n // MAX_DISPLAY_BINS
        trim = n - (n % factor)
        counts = counts[:trim].reshape(-1, factor).sum(axis=1)
        bin_edges = np.concatenate(
            [bin_edges[:trim:factor], bin_edges[trim : trim + 1]]
        )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, counts


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
