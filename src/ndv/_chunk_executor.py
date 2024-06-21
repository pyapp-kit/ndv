from __future__ import annotations

import math
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from itertools import product
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    SupportsIndex,
    cast,
)

import numpy as np

if TYPE_CHECKING:
    from types import EllipsisType
    from typing import TypeAlias

    import numpy.typing as npt

    class SupportsDunderLT(Protocol):
        def __lt__(self, other: Any) -> bool: ...

    class SupportsDunderGT(Protocol):
        def __gt__(self, other: Any) -> bool: ...

    SupportsComparison: TypeAlias = SupportsDunderLT | SupportsDunderGT

NULL = object()


class SupportsChunking(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...
    def __getitem__(self, idx: tuple[int | slice, ...]) -> npt.ArrayLike: ...


class ChunkResponse(NamedTuple):
    # location in the original array
    location: tuple[int | slice, ...]
    # the data that was returned
    data: np.ndarray

    @property
    def offset(self) -> tuple[int, ...]:
        return tuple(i.start if isinstance(i, slice) else i for i in self.location)


ChunkFuture = Future[ChunkResponse]


class Chunker:
    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor or self._default_executor()
        self._pending_futures: deque[ChunkFuture] = deque()
        self._request_chunk = _get_chunk

    @classmethod
    def _default_executor(cls) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(thread_name_prefix=cls.__name__)

    def is_idle(self) -> bool:
        return all(f.done() for f in self._pending_futures)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> Chunker:
        return self

    def __exit__(self, *_: Any) -> Literal[False]:
        self.shutdown(wait=True)
        return False

    def request_chunks(
        self,
        data: SupportsChunking,
        index: Mapping[int, int | slice] | tuple[int | slice, ...] | None = None,
        # None implies no chunking
        chunk_shape: int | None | Sequence[int | None] = None,
        *,
        sort_key: Callable[[tuple[int | slice, ...]], SupportsComparison] | None = NULL,  # type: ignore
        cancel_existing: bool = False,
    ) -> list[ChunkFuture]:
        """Request chunks from `data` based on the given `index` and `chunk_shape`.

        Parameters
        ----------
        data : SupportsChunking
            The data to request chunks from. Must have a `shape` attribute and support
            indexing (`__getitem__`) with a tuple of int or slice.
        index : Mapping[int, int | slice] | tuple[int | slice, ...] | None
            A subarray to request.
            If a Mapping, it should look like {dim: index} where `index` is a single
            index or a slice and dim is the dimension to index.
            If an tuple, it should be a regular tuple of integer or slice:
            e.g. (6, 0, slice(None, None, None), slice(1, 10, None))
            If `None` (default), the full array is requested.
        chunk_shape : int | tuple[int, ...] | None
            The shape of each chunk. If a single int, the same size is used for all
            dimensions. If `None`, no chunking is done. Note that chunk shape applies
            to the data *prior* to indexing.  Chunks will be aligned with the original
            data, not the indexed data... so given an axis `0` with length 100, if you
            request a slice from that index `index={0: slice(40,60)}` and provide a
            `chunk_shape=50`, you will get two chunks: (40, 50) and (50, 60).
            The intention is that chunk_shape should align with the chunk layout of the
            original data, to optimize reading from disk or other sources, even when
            reading a subset of the data that is not aligned with the chunks.
        sort_key: Callable[[tuple[slice, ...]], SupportsComparison] | None
            A function to sort the chunks before submitting them.  This can be used to
            prioritize chunks that are more likely to be needed first (such as those
            within a certain distance of the current view).  The function should take
            a tuple of slices and return a value that can be compared with `<` and `>`.
            If None, no sorting is done.
        cancel_existing : bool
            If True, cancel any existing pending futures before submitting new ones.

        Returns
        -------
        list[Future[ChunkResponse]]
            A list of futures that will contain the requested chunks when they are
            available.  Use `Future.add_done_callback` to register a callback to handle
            the results.
        """
        if cancel_existing:
            for future in list(self._pending_futures):
                future.cancel()

        if index is None:
            index = tuple(slice(None) for _ in range(len(data.shape)))

        if isinstance(index, Mapping):
            index = indexers_to_conventional_slice(index)

        # at this point, index is a tuple of int or slice
        # e.g. (6, 0, slice(None, None, None), slice(1, 10, None))
        # now, determine the subchunk indices to request
        if chunk_shape is None:
            # TODO: check whether we need to cast this to something without integers.
            indices: Iterable[tuple[int | slice, ...]] = [index]
        else:
            indices = iter_chunk_aligned_slices(data.shape, chunk_shape, index)
            if sort_key is not None:
                if sort_key is NULL:

                    def sort_key(x: tuple[int | slice, ...]) -> SupportsComparison:
                        return distance_from_coord(x, data.shape)

                indices = sorted(indices, key=sort_key)

        # submit the a request for each subchunk
        futures = []
        for chunk_index in indices:
            future = self._executor.submit(self._request_chunk, data, chunk_index)
            self._pending_futures.append(future)
            future.add_done_callback(self._pending_futures.remove)
            futures.append(future)
        return futures


def _get_chunk(data: SupportsChunking, index: tuple[int | slice, ...]) -> ChunkResponse:
    chunk_data = _reduce_data_for_display(data[index], len(data.shape))
    # import time

    # time.sleep(0.05)
    return ChunkResponse(location=index, data=chunk_data)


def indexers_to_conventional_slice(
    indexers: Mapping[int, int | slice], ndim: int | None = None
) -> tuple[int | slice, ...]:
    """Convert Mapping of {dim: index} to a conventional tuple of int or slice.

    `indexers` need not be ordered.  If `ndim` is not provided, it is inferred
    from the maximum key in `indexers`.

    Parameters
    ----------
    indexers : Mapping[int, int | slice]
        Mapping of {dim: index} where `index` is a single index or a slice.
    ndim : int | None
        Number of dimensions. If None, inferred from the maximum key in `indexers`.

    Examples
    --------
    >>> indexers_to_conventional_slice({1: 0, 0: 6, 3: slice(1, 10, None)})
    (6, 0, slice(None, None, None), slice(1, 10, None))

    """
    if not indexers:
        return (slice(None),)

    if ndim is None:
        ndim = max(indexers) + 1
    return tuple(indexers.get(k, slice(None)) for k in range(ndim))


def _slice_indices(sl: SupportsIndex | slice, dim_size: int) -> tuple[int, int, int]:
    """Convert slice to range arguments, handling single int as well.

    Examples
    --------
    >>> _slice2range(3, 10)
    (3, 4, 1)
    >>> _slice2range(slice(1, 4), 10)
    (1, 4, 1)
    >>> _slice2range(slice(1, None), 10)
    (1, 10, 1)
    """
    if isinstance(sl, slice):
        return sl.indices(dim_size)
    return (sl.__index__(), sl.__index__() + 1, 1)


def iter_chunk_aligned_slices(
    shape: Sequence[int],
    chunks: int | Sequence[int | None],
    slices: Sequence[int | slice | EllipsisType],
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

    Returns
    -------
    Iterator[tuple[slice, ...]]
        An iterator of chunk-aligned slices.

    Raises
    ------
    ValueError
        If the length of `chunks`, `shape`, and `slices` do not match, or any chunks
        are zero.
    IndexError
        If more than one Ellipsis is present in `slices`.

    Examples
    --------
    >>> list(
    ...     iter_chunk_aligned_slices(shape=(6, 6), chunks=4, slices=(slice(1, 4), ...))
    ... )
    [
        (slice(1, 4, None), slice(0, 4, None)),
        (slice(1, 4, None), slice(4, 6, None)),
    ]

    >>> x = iter_chunk_aligned_slices(
    ...     shape=(10, 9), chunks=(4, 3), slices=(slice(3, 9), slice(1, None))
    ... )
    >>> list(x)
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
    elif not len(chunks) == ndim:
        raise ValueError("Length of `chunks` must match length of `shape`")

    if any(x == 0 for x in chunks):
        raise ValueError("Chunk size must be greater than zero")

    # convert any `None` chunks to full size of the dimension
    chunks = tuple(x if x is not None else shape[i] for i, x in enumerate(chunks))

    if num_ellipsis := slices.count(Ellipsis):
        if num_ellipsis > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        # Replace Ellipsis with multiple slices
        el_idx = slices.index(Ellipsis)
        n_remaining = ndim - len(slices) + 1
        slices = (
            tuple(slices[:el_idx])
            + (slice(None),) * n_remaining
            + tuple(slices[el_idx + 1 :])
        )
    slices = cast(tuple[int | slice, ...], slices)  # now we have no Ellipsis
    if not ndim == len(slices):
        # Fill in remaining dimensions with full slices
        slices = slices + (slice(None),) * (ndim - len(slices))

    # Create ranges for each dimension based on the slices provided
    ranges = [_slice_indices(sl, dim) for sl, dim in zip(slices, shape)]

    # Generate indices for each dimension that align with chunks
    aligned_ranges = (
        range(start - (start % chunk_size), stop, chunk_size)
        for (start, stop, _), chunk_size in zip(ranges, chunks)
    )

    # Create all combinations of these aligned ranges
    for indices in product(*aligned_ranges):
        chunk_slices = []
        for idx, (start, stop, step), ch in zip(indices, ranges, chunks):
            # Calculate the actual slice for each dimension
            start = max(start, idx)
            stop = min(stop, idx + ch)
            if start >= stop:  # Skip empty slices
                break
            chunk_slices.append(slice(start, stop, step))
        else:
            # Only add this combination of slices if all dimensions are valid
            yield tuple(chunk_slices)


def _slice_center(s: slice | int, dim_size: int) -> float:
    """Calculate the center of a slice based on its start and stop attributes."""
    if isinstance(s, int):
        return s
    start = float(s.start) if s.start is not None else 0
    stop = float(s.stop) if s.stop is not None else dim_size
    return (start + stop) / 2


def distance_from_coord(
    slice_tuple: Sequence[slice | int],
    shape: Sequence[int],
    coord: Iterable[float] = (),  # defaults to center of shape
) -> float:
    """Euclidean distance from the center of an nd slice to the center of shape."""
    if not coord:
        coord = (dim / 2 for dim in shape)
    slice_centers = (_slice_center(s, dim) for s, dim in zip(slice_tuple, shape))
    return math.hypot(*(sc - cc for sc, cc in zip(slice_centers, coord)))


def _reduce_data_for_display(
    data: npt.ArrayLike, ndims: int, reductor: Callable[..., np.ndarray] = np.max
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
    #   the first extra dims)
    data = np.asarray(data).squeeze()
    if extra_dims := data.ndim - ndims:
        axis = tuple(range(extra_dims))
        data = reductor(data, axis=axis)

    if data.dtype.itemsize > 4:  # More than 32 bits
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int32)
        else:
            data = data.astype(np.float32)
    return data


# class DaskChunker:
#     def __init__(self) -> None:
#         try:
#             import dask
#             import dask.array as da
#             from dask.distributed import Client
#         except ImportError as e:
#             raise ImportError("Dask is required for DaskChunker") from e
#         self._dask = dask
#         self._da = da
#         self._client = Client()

#     def request_chunks(
#         self,
#         data: SupportsChunking,
#         index: Mapping[int, int | slice] | IndexTuple | None = None,
#         chunk_shape: int | tuple[int, ...] | None = None,  # None implies no chunking
#         *,
#         cancel_existing: bool = False,
#     ) -> list[Future[ChunkResponse]]:
#         if isinstance(index, Mapping):
#             index = indexers_to_conventional_slice(index)

#         if isinstance(data, self._da.Array):  # type: ignore
#             dask_data = data
#         else:
#             dask_data = self._da.from_array(data, chunks=chunk_shape)  # type: ignore

#         subarray = dask_data[index]
#         block_ranges = (range(x) for x in subarray.numblocks)
#         for blk in product(*(block_ranges)):
#             offset = tuple(sum(sizes[:x]) for sizes, x in zip(subarray.chunks, blk))
#             chunk = subarray.blocks[blk]
#             future = self._client.compute(chunk)

#             @no_type_check
#             def _set_result(_chunk=chunk, _future=future):
#                 _future.set_result(
#                     ChunkResponse(idx=index, data=_chunk.compute(), offset=offset)
#                 )

#             futures.append()

#         return [data]
