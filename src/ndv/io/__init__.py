"""Opinionated bioimage IO — read common formats into xarray.DataArray."""

from __future__ import annotations

from ndv.io._dispatch import imread

__all__ = ["imread"]
