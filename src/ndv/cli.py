import argparse
from contextlib import suppress
from typing import Any

import numpy as np

from ndv.util import imshow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NDV: Numpy Data Viewer")
    parser.add_argument("path", type=str, help="The filename of the numpy file to view")
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="For multiscales files, the level to read (default: 0)",
    )
    return parser.parse_args()


def _read_zarr(path: str, level: int = 0) -> Any:
    import tensorstore as ts

    driver = path.split(".")[-1]
    for level_path in (path, f"{path}/s{level}", f"{path}/{level}"):
        with suppress(ValueError):
            return ts.open({"driver": driver, "kvstore": level_path}).result()
    raise ValueError(f"Could not find level {level} in {path}")


def imread(path: str, level: int = 0) -> Any:
    from bioio import BioImage

    img = BioImage(path)
    return img.xarray_dask_data

    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".zarr", ".n5")):
        return _read_zarr(path, level)


def main() -> None:
    args = parse_args()

    data = imread(args.path, args.level)
    imshow(data)
