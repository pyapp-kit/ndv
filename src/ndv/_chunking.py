from __future__ import annotations

import math
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import product
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Hashable,
    Mapping,
    NamedTuple,
    Sequence,
    cast,
)

import numpy as np
from rich import print

if TYPE_CHECKING:
    from collections import deque
    from typing import Callable, Iterable, Iterator, TypeAlias

    from .viewer._data_wrapper import DataWrapper

# any hashable represent a single dimension in an ND array
DimKey: TypeAlias = Hashable
# any object that can be used to index a single dimension in an ND array
Index: TypeAlias = int | slice
# a mapping from dimension keys to indices (eg. {"x": 0, "y": slice(5, 10)})
# this object is used frequently to query or set the currently displayed slice
Indices: TypeAlias = Mapping[DimKey, Index]
# mapping of dimension keys to the maximum value for that dimension
Sizes: TypeAlias = Mapping[DimKey, int]


class ChunkResponse(NamedTuple):
    idx: tuple[int | slice, ...]  # index that was requested
    data: np.ndarray  # the data that was returned
    offset: tuple[int, ...]  # offset of the data in the full array (derived from idx)
    channel_index: int = -1


RequestFinished = object()


class Chunker:
    def __init__(
        self,
        data_wrapper: DataWrapper | None = None,
        chunks: int | tuple[int, ...] | None = None,
        on_ready: Callable[[ChunkResponse], Any] | None = None,
    ) -> None:
        self.chunks = chunks
        self.data_wrapper: DataWrapper | None = data_wrapper
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_futures: deque[Future[ChunkResponse]] = Deque()
        self.on_ready = on_ready
        self.channel_axis: int | None = None

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self.executor.shutdown(cancel_futures=True, wait=True)

    def _request_chunk_sync(
        self, idx: tuple[int | slice, ...], channel_axis: int | None
    ) -> ChunkResponse:
        # idx is guaranteed to have length equal to the number of dimensions
        if channel_axis is not None:
            channel_index = idx[channel_axis]
            if isinstance(channel_index, slice):
                channel_index = channel_index.start
        else:
            channel_index = -1

        data = self.data_wrapper[idx]  # type: ignore [index]
        data = _reduce_data_for_display(data, 2)
        # FIXME: temporary
        # this needs to be aware of nvisible dimensions
        try:
            offset = tuple(int(getattr(sl, "start", sl)) for sl in idx)[-2:]
        except TypeError:
            offset = (0, 0)

        import time

        time.sleep(0.02)
        return ChunkResponse(
            idx=idx, data=data, offset=offset, channel_index=channel_index
        )

    def request_index(self, index: Indices, cancel_existing: bool = True) -> None:
        if cancel_existing:
            for future in list(self.pending_futures):
                future.cancel()

        if self.data_wrapper is None:
            return
        idx = self.data_wrapper.to_conventional(index)

        if (chunks := self.chunks) is None:
            subchunks: Iterable[tuple[int | slice, ...]] = [idx]
        else:
            shape = self.data_wrapper.data.shape

            # we never chunk the channel axis
            if isinstance(chunks, int):
                _chunks = [chunks] * len(shape)
            else:
                _chunks = list(chunks)
            if self.channel_axis is not None:
                _chunks[self.channel_axis] = 1

            # TODO: allow the viewer to pass a center coord, to load chunks
            # preferentially around that point
            subchunks = sorted(
                iter_chunk_aligned_slices(shape, _chunks, idx),
                key=lambda x: distance_from_coord(x, shape),
            )
        # print("Requesting index:", idx)
        # print("subchunks", subchunks)
        # print()
        for chunk_idx in subchunks:
            future = self.executor.submit(
                self._request_chunk_sync, chunk_idx, self.channel_axis
            )
            self.pending_futures.append(future)
            future.add_done_callback(self._on_chunk_ready)

    def _on_chunk_ready(self, future: Future[ChunkResponse]) -> None:
        self.pending_futures.remove(future)
        if future.cancelled():
            return
        if err := future.exception():
            print(f"{type(err).__name__}: in chunk request: {err}")
            return
        if self.on_ready is not None:
            self.on_ready(future.result())
            if not self.pending_futures:
                # FIXME: this emits multiple times sometimes
                # Fix typing
                self.on_ready(RequestFinished)


def _reduce_data_for_display(
    data: np.ndarray, ndims: int, reductor: Callable[..., np.ndarray] = np.max
) -> np.ndarray:
    """Reduce the number of dimensions in the data for display.

    This function takes a data array and reduces the number of dimensions to
    the max allowed for display. The default behavior is to reduce the smallest
    dimensions, using np.max.  This can be improved in the future.

    This also coerces 64-bit data to 32-bit data.
    """
    # TODO
    # - allow dimensions to control how they are reduced (as opposed to just max)
    # - for better way to determine which dims need to be reduced (currently just
    #   the smallest dims)
    data = data.squeeze()
    if extra_dims := data.ndim - ndims:
        shapes = sorted(enumerate(data.shape), key=lambda x: x[1])
        smallest_dims = tuple(i for i, _ in shapes[:extra_dims])
        data = reductor(data, axis=smallest_dims)

    if data.dtype.itemsize > 4:  # More than 32 bits
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int32)
        else:
            data = data.astype(np.float32)
    return data


# def _axis_chunks(total_length: int, chunk_size: int) -> tuple[int, ...]:
#     """Break `total_length` into chunks of `chunk_size` plus remainder.

#     Examples
#     --------
#     >>> _axis_chunks(10, 3)
#     (3, 3, 3, 1)
#     """
#     sequence = (chunk_size,) * (total_length // chunk_size)
#     if remainder := total_length % chunk_size:
#         sequence += (remainder,)
#     return sequence


# def _shape_chunks(
#     shape: tuple[int, ...], chunks: int | tuple[int, ...]
# ) -> tuple[tuple[int, ...], ...]:
#     """Break `shape` into chunks of `chunks` along each axis.

#     Examples
#     --------
#     >>> _shape_chunks((10, 10, 10), 3)
#     ((3, 3, 3, 1), (3, 3, 3, 1), (3, 3, 3, 1))
#     """
#     if isinstance(chunks, int):
#         chunks = (chunks,) * len(shape)
#     elif isinstance(chunks, Sequence):
#         if len(chunks) != len(shape):
#             raise ValueError("Length of `chunks` must match length of `shape`")
#     else:
#         raise TypeError("`chunks` must be an int or sequence of ints")
#     return tuple(_axis_chunks(length, chunk) for length, chunk in zip(shape, chunks))


def _slice2range(sl: slice | int, dim_size: int) -> range:
    """Convert slice to range, handling single int as well.

    Examples
    --------
    >>> _slice2range(3, 10)
    range(3, 4)
    """
    if isinstance(sl, int):
        return range(sl, sl + 1)
    start = 0 if sl.start is None else max(sl.start, 0)
    stop = dim_size if sl.stop is None else min(sl.stop, dim_size)
    return range(start, stop)


def iter_chunk_aligned_slices(
    shape: Sequence[int],
    chunks: Sequence[int],
    slices: tuple[int | slice | EllipsisType, ...],
) -> Iterator[tuple[slice, ...]]:
    """Yield chunk-aligned slices for a given shape and slices.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array to slice.
    chunks : int or tuple[int, ...]
        The size of each chunk. If a single int, the same size is used for all
        dimensions.
    slices : tuple[int | slice | Ellipsis, ...]
        The full slices to apply to the array. Ellipsis is supported to
        represent multiple slices.

    Examples
    --------
    >>> list(iter_chunk_aligned_slices((6, 6), 4, (slice(1, 4), ...)))
    [
        (slice(1, 4, None), slice(0, 4, None)),
        (slice(1, 4, None), slice(4, 6, None)),
    ]

    >>> list(iter_chunk_aligned_slices((10, 9), (4, 3), (slice(3, 9), slice(1, None))))
    [
        (slice(3, 4, None), slice(1, 3, None)),
        (slice(3, 4, None), slice(3, 6, None)),
        (slice(3, 4, None), slice(6, 9, None)),
        (slice(4, 8, None), slice(1, 3, None)),
        (slice(4, 8, None), slice(3, 6, None)),
        (slice(4, 8, None), slice(6, 9, None)),
        (slice(8, 9, None), slice(1, 3, None)),
        (slice(8, 9, None), slice(3, 6, None)),
        (slice(8, 9, None), slice(6, 9, None)),
    ]
    """
    # Make chunks same length as shape if single int
    ndim = len(shape)
    if any(x == 0 for x in chunks):
        raise ValueError("Chunk size must be greater than zero")

    if any(isinstance(sl, EllipsisType) for sl in slices):
        # Replace Ellipsis with multiple slices
        if slices.count(Ellipsis) > 1:
            raise ValueError("Only one Ellipsis is allowed")
        el_idx = slices.index(Ellipsis)
        n_remaining = ndim - len(slices) + 1
        slices = slices[:el_idx] + (slice(None),) * n_remaining + slices[el_idx + 1 :]

    if not (len(chunks) == ndim == len(slices)):
        raise ValueError("Length of `chunks`, `shape`, and `slices` must match")

    # Create ranges for each dimension based on the slices provided
    slices = cast(tuple[int | slice, ...], slices)
    ranges = [_slice2range(sl, dim) for sl, dim in zip(slices, shape)]

    # Generate indices for each dimension that align with chunks
    aligned_ranges = (
        range(r.start - (r.start % ch), r.stop, ch) for r, ch in zip(ranges, chunks)
    )

    # Create all combinations of these aligned ranges
    for indices in product(*aligned_ranges):
        chunk_slices = []
        for idx, rng, ch in zip(indices, ranges, chunks):
            # Calculate the actual slice for each dimension
            start = max(rng.start, idx)
            stop = min(rng.stop, idx + ch)
            if start >= stop:  # Skip empty slices
                break
            chunk_slices.append(slice(start, stop))
        else:
            # Only add this combination of slices if all dimensions are valid
            yield tuple(chunk_slices)


def slice_center(s: slice | int, dim_size: int) -> float:
    """Calculate the center of a slice based on its start and stop attributes."""
    if isinstance(s, int):
        return s
    start = float(s.start) if s.start is not None else 0
    stop = float(s.stop) if s.stop is not None else dim_size
    return (start + stop) / 2


def distance_from_coord(
    slice_tuple: tuple[slice | int, ...],
    shape: tuple[int, ...],
    coord: Iterable[float] = (),  # defaults to center of shape
) -> float:
    """Euclidean distance from the center of an nd slice to the center of shape."""
    if not coord:
        coord = (dim / 2 for dim in shape)
    slice_centers = (slice_center(s, dim) for s, dim in zip(slice_tuple, shape))
    return math.hypot(*(sc - cc for sc, cc in zip(slice_centers, coord)))
