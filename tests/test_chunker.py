import numpy as np
import numpy.testing as npt

from ndv._chunk_executor import Chunker, iter_chunk_aligned_slices


def test_iter_chunk_aligned_slices() -> None:
    x = iter_chunk_aligned_slices(
        shape=(10, 9), chunks=(4, 3), slices=np.index_exp[3:9, 1:None]
    )
    assert list(x) == [
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

    # this one tests that slices doesn't need to be the same length as shape
    # ... is added at the end
    y = iter_chunk_aligned_slices(shape=(6, 6), chunks=4, slices=np.index_exp[1:4])
    assert list(y) == [
        (slice(1, 4, None), slice(0, 4, None)),
        (slice(1, 4, None), slice(4, 6, None)),
    ]

    # this tests ellipsis in the middle
    z = iter_chunk_aligned_slices(
        shape=(3, 3, 3), chunks=2, slices=np.index_exp[1, ..., :2]
    )
    assert list(z) == [
        (slice(1, 2, None), slice(0, 2, None), slice(0, 2, None)),
        (slice(1, 2, None), slice(2, 3, None), slice(0, 2, None)),
    ]


def test_chunker() -> None:
    data = np.random.rand(100, 100).astype(np.float32)

    with Chunker() as chunker:
        futures = chunker.request_chunks(data)

    assert len(futures) == 1
    npt.assert_array_equal(data, futures[0].result().data)

    data2 = np.random.rand(30, 30, 30).astype(np.float32)
    # test that the data is correctly chunked with weird chunk shapes
    with Chunker() as chunker:
        futures = chunker.request_chunks(
            data2, index={0: 0}, chunk_shape=(None, 17, 12)
        )

    new = np.empty_like(data2[0])
    for future in futures:
        result = future.result()
        new[result.array_location[1:]] = result.data
    npt.assert_array_equal(new, data2[0])


# # this test is provided as an example of using dask to accomplish a similar thing
# # this library should try to retain support for using dask instead of the internal
# # chunker ... but it's nice not to have to depend on dask otherwise.
# def test_dask_chunker() -> None:
#     try:
#         import dask.array as da
#         from dask.distributed import Client
#     except ImportError:
#         pytest.skip("Dask not installed")

#     from itertools import product

#     data = np.random.rand(100, 100).astype(np.float32)
#     dask_data_chunked = da.from_array(data, chunks=(25, 20))  # type: ignore
#     chunk_sizes = dask_data_chunked.chunks

#     with Client() as client:  # type: ignore [no-untyped-call]
#         for idx in product(*(range(x) for x in dask_data_chunked.numblocks)):
#             # Calculate the start indices (offsets) for the chunk
#             # THIS is the main thing we'd need for visualization purposes.
#             # wish there was an easier way to get this from the chunk_result alone
#             offset = tuple(sum(sizes[:x]) for sizes, x in zip(chunk_sizes, idx))

#             chunk = dask_data_chunked.blocks[idx]

#             future = client.compute(chunk)
#             chunk_result = future.result()

#             # Test that the data is correctly chunked and equal to the original data
#             sub_idx = tuple(slice(x, x + y) for x, y in zip(offset, chunk_result.shape))
#             expected = data[sub_idx]
#             npt.assert_array_equal(expected, chunk_result)


# def test_dask_map_blocks() -> None:
#     if TYPE_CHECKING:
#         import dask.array
#     else:
#         dask = pytest.importorskip("dask")

#     dask_array = dask.array.random.randint(100, size=(100, 100), chunks=(25, 20))
#     block_ranges = (range(x) for x in dask_array.numblocks)
#     for block_id in product(*(block_ranges)):
#         _offset = tuple(sum(sizes[:x]) for sizes, x in zip(dask_array.chunks, block_id))
#         _chunk = dask_array.blocks[block_id]
#         print(_offset, _chunk)
