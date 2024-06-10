from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from itertools import product
from types import EllipsisType
from typing import (
    Callable,
    Deque,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TypeAlias,
    cast,
)

import cmap
import numpy as np
from attr import dataclass

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


@dataclass
class ChannelSetting:
    visible: bool = True
    colormap: cmap.Colormap | str = "gray"
    clims: tuple[float, float] | None = None
    gamma: float = 1
    auto_clim: bool = False


class Response(NamedTuple):
    idx: tuple[int | slice, ...]
    data: np.ndarray


class Slicer:
    def __init__(
        self,
        data_wrapper: DataWrapper | None = None,
        chunks: int | tuple[int, ...] | None = None,
    ) -> None:
        self.chunks = chunks
        self.data_wrapper: DataWrapper | None = data_wrapper
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_futures: Deque[Future[Response]] = Deque()

    def __del__(self) -> None:
        self.executor.shutdown(cancel_futures=True, wait=True)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    def _request_chunk_sync(self, idx: tuple[int | slice, ...]) -> Response:
        if self.data_wrapper is None:
            raise ValueError("No data wrapper set")
        data = self.data_wrapper[idx]
        return Response(idx=idx, data=data)

    def request_index(self, index: Indices, func: Callable) -> None:
        if self.data_wrapper is None:
            return
        idx = self.data_wrapper.to_conventional(index)
        if self.chunks is None:
            subchunks: Iterable[tuple[int | slice, ...]] = [idx]
        else:
            shape = self.data_wrapper.data.shape
            subchunks = sorted(
                iter_chunk_aligned_slices(shape, self.chunks, idx),
                key=lambda x: center_distance_key(x, shape),
            )
        for chunk_idx in subchunks:
            future = self.executor.submit(self._request_chunk_sync, chunk_idx)
            self.pending_futures.append(future)
            future.add_done_callback(partial(self._on_chunk_ready, func))

    def _on_chunk_ready(self, func: Callable, future: Future[Response]) -> None:
        chunk = future.result()
        # process the chunk data

        # print(start, chunk.data.squeeze().shape)
        func(chunk)
        self.pending_futures.remove(future)


def _axis_chunks(total_length: int, chunk_size: int) -> tuple[int, ...]:
    """Break `total_length` into chunks of `chunk_size` plus remainder.

    Examples
    --------
    >>> _axis_chunks(10, 3)
    (3, 3, 3, 1)
    """
    sequence = (chunk_size,) * (total_length // chunk_size)
    if remainder := total_length % chunk_size:
        sequence += (remainder,)
    return sequence


def _shape_chunks(
    shape: tuple[int, ...], chunks: int | tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Break `shape` into chunks of `chunks` along each axis.

    Examples
    --------
    >>> _shape_chunks((10, 10, 10), 3)
    ((3, 3, 3, 1), (3, 3, 3, 1), (3, 3, 3, 1))
    """
    if isinstance(chunks, int):
        chunks = (chunks,) * len(shape)
    elif isinstance(chunks, Sequence):
        if len(chunks) != len(shape):
            raise ValueError("Length of `chunks` must match length of `shape`")
    else:
        raise TypeError("`chunks` must be an int or sequence of ints")
    return tuple(_axis_chunks(length, chunk) for length, chunk in zip(shape, chunks))


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
    shape: tuple[int, ...],
    chunks: int | tuple[int, ...],
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
    if isinstance(chunks, int):
        chunks = (chunks,) * ndim
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


def slice_center(s, dim_size):
    """Calculate the center of a slice based on its start and stop attributes."""
    # For integer slices, center is the integer itself.
    if isinstance(s, int):
        return s
    # For slice objects, calculate the middle point.
    start = s.start if s.start is not None else 0
    stop = s.stop if s.stop is not None else dim_size
    return (start + stop) / 2


def center_distance_key(slice_tuple, shape):
    """Calculate the Euclidean distance from the center of the slices to the center of the shape."""
    shape_center = [dim / 2 for dim in shape]
    slice_centers = [slice_center(s, dim) for s, dim in zip(slice_tuple, shape)]

    # Calculate Euclidean distance from the slice centers to the shape center
    distance = np.sqrt(
        sum((sc - cc) ** 2 for sc, cc in zip(slice_centers, shape_center))
    )
    return distance
