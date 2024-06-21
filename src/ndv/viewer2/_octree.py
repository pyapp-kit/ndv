from typing import Generic, Iterator, NamedTuple, TypeVar

MAX_DEPTH = 8


class Coord(NamedTuple):
    x: float
    y: float
    z: float = 0


class Bounds(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float = 0
    z_max: float = 0

    @property
    def midpoint(self) -> Coord:
        return Coord(
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )

    def isdisjoint(self, other: "Bounds") -> bool:
        return (
            self.x_max < other.x_min
            or self.x_min > other.x_max
            or self.y_max < other.y_min
            or self.y_min > other.y_max
            or self.z_max < other.z_min
            or self.z_min > other.z_max
        )

    def intersects(self, other: "Bounds") -> bool:
        return not self.isdisjoint(other)

    def split(self) -> tuple["Bounds", ...]:
        x_min, x_max, y_min, y_max, z_min, z_max = self
        x_mid, y_mid, z_mid = self.midpoint
        return (
            Bounds(x_min, x_mid, y_min, y_mid, z_min, z_mid),
            Bounds(x_mid, x_max, y_min, y_mid, z_min, z_mid),
            Bounds(x_min, x_mid, y_mid, y_max, z_min, z_mid),
            Bounds(x_mid, x_max, y_mid, y_max, z_min, z_mid),
            Bounds(x_min, x_mid, y_min, y_mid, z_mid, z_max),
            Bounds(x_mid, x_max, y_min, y_mid, z_mid, z_max),
            Bounds(x_min, x_mid, y_mid, y_max, z_mid, z_max),
            Bounds(x_mid, x_max, y_mid, y_max, z_mid, z_max),
        )


T = TypeVar("T")


class OctreeNode(Generic[T]):
    def __init__(self, bounds: Bounds, depth: int = 0) -> None:
        # spatial bounds of this node (xmin, xmax, ymin, ymax, zmin, zmax)
        self.bounds = bounds
        self.children: list[OctreeNode] = []  # children of this node
        self.depth = depth  # depth of the node in the tree
        self.data: T | None = None  # placeholder for storing references to data chunks

    def is_leaf(self) -> bool:
        return not self.children

    def split(self, max_depth: int = MAX_DEPTH) -> None:
        if self.depth < max_depth:
            self.children = [
                OctreeNode(bounds, self.depth + 1) for bounds in self.bounds.split()
            ]

    def insert_data(self, chunk: T, chunk_bounds: Bounds) -> None:
        """Insert data into the tree.

        If `self` is a leaf and the depth is less than the maximum depth, `self` is
        split into 8 children. The data is then inserted into the first child that
        intersects with the data bounds.
        """
        if self.is_leaf():
            if self.depth < MAX_DEPTH:
                self.split(MAX_DEPTH)
            else:
                self.data = chunk
                return
        for child in self.children:
            if child.bounds.intersects(chunk_bounds):
                child.insert_data(chunk, chunk_bounds)
                break

    def query(self, view_bounds: Bounds) -> Iterator[T]:
        """Query the tree for data chunks that intersect with the given bounds.

        Yields data chunks that intersect with the given bounds.
        """
        if self.bounds.intersects(view_bounds):
            if self.is_leaf():
                if self.data is not None:
                    yield self.data
            else:
                for child in self.children:
                    yield from child.query(view_bounds)
