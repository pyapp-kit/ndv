"""command-line program."""

import argparse

from ndv.util import imshow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ndv: ndarray viewer")
    parser.add_argument("path", type=str, help="The filename of the numpy file to view")
    return parser.parse_args()


def main() -> None:
    """Run the command-line program."""
    from ndv import io

    args = _parse_args()

    imshow(io.imread(args.path))
