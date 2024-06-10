import random

import dask.array as da
from dask.distributed import Client, as_completed


# Function to load a chunk
def load_chunk(chunk):
    # Simulate loading time
    import time

    t = random.random() * 5
    print(t)
    time.sleep(t)
    return chunk


if __name__ == "__main__":
    # Set up Dask Client
    client = Client()
    # Create a Dask array (simulate chunked storage)
    x = da.random.random((10, 10), chunks=(5, 5))

    # Submit tasks directly to the scheduler and get futures
    futures = []
    for i in range(x.numblocks[0]):
        for j in range(x.numblocks[1]):
            chunk = x.blocks[i, j]
            future = client.submit(load_chunk, chunk)
            futures.append(future)

    # Monitor progress using as_completed
    for future in as_completed(futures):
        result = future.result()
        print("Chunk ready:", result.shape)

    # Close the client
    client.close()
