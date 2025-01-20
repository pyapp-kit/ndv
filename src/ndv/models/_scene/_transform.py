from __future__ import annotations

import functools
import math
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pydantic import ConfigDict, RootModel
from pydantic_core import core_schema

from ._vis_model import Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Sized

    from pydantic import GetCoreSchemaHandler


def _arg_to_vec4(
    func: Callable[[Transform, ArrayLike], NDArray],
) -> Callable[[Transform, ArrayLike], NDArray]:
    """Return method decorator that converts arg to vec4, suitable for 4x4 matrix mul.

    [x, y]      =>  [[x, y, 0, 1]]

    [x, y, z]   =>  [[x, y, z, 1]]

    [[x1, y1],      [[x1, y1, 0, 1],
     [x2, y2],  =>   [x2, y2, 0, 1],
     [x3, y3]]       [x3, y3, 0, 1]]

    If 1D input is provided, then the return value will be flattened.
    Accepts input of any dimension, as long as shape[-1] <= 4
    """

    @functools.wraps(func)
    def wrapper(self_: Transform, arg: ArrayLike) -> NDArray:
        if not isinstance(arg, (tuple, list, np.ndarray)):
            raise TypeError(f"Cannot convert argument to 4D vector: {arg!r}")
        arg = np.array(arg)
        flatten = arg.ndim == 1
        arg = as_vec4(arg)

        ret = func(self_, arg)
        return np.copy(np.ravel(ret)) if flatten and ret is not None else ret

    return wrapper


def _validate_matrix(val: Any) -> np.ndarray:
    if val is None:
        return np.eye(4)
    if not isinstance(val, np.ndarray):
        val = np.asarray(val, dtype=float)
    if val.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4, not {val.shape}")
    return val  # type: ignore


class Matrix(np.ndarray):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        # Serialize the matrix as a list of lists, unless it is the identity matrix
        # (for brevity in the serialized output).
        def _serialize(val: np.ndarray) -> list | None:
            if np.allclose(val, np.eye(4)):
                return None
            return val.tolist()  # type: ignore

        list_of_float = core_schema.list_schema(core_schema.float_schema())
        ser_schema = core_schema.plain_serializer_function_ser_schema(
            _serialize,
            return_schema=core_schema.union_schema(
                [core_schema.list_schema(list_of_float), core_schema.none_schema()]
            ),
        )

        return core_schema.no_info_before_validator_function(
            _validate_matrix,
            core_schema.any_schema(),
            serialization=ser_schema,
        )


class Transform(RootModel):
    """Transformation."""

    root: Matrix = Field(
        default_factory=lambda: np.eye(4),  # type: ignore
        description="Transformation matrix.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, validate_default=True)

    def __array__(self, dtype: DTypeLike | None = None) -> np.ndarray:
        return self.root.astype(dtype)

    def __repr__(self):
        return repr(self.root).replace("array", "Transform")

    def is_null(self) -> bool:
        return np.allclose(self.root, np.eye(4))

    def __matmul__(self, other: Transform | ArrayLike) -> Transform:
        """Return the dot product of this transform with another."""
        if isinstance(other, Transform):
            other = other.root
        return Transform(self.root @ other)  # type: ignore

    def dot(self, other: Transform | ArrayLike) -> Transform:
        """Return the dot product of this transform with another."""
        if isinstance(other, Transform):
            other = other.root
        return Transform(np.dot(self.root, other))

    @property
    def T(self) -> Transform:
        """Return the transpose of the transform."""
        return Transform(self.root.T)

    def inv(self) -> Transform:
        """Return the inverse of the transform."""
        return Transform(np.linalg.inv(self.root))  # type: ignore

    def translated(self, pos: ArrayLike) -> Transform:
        """Return new transform, translated by pos.

        The translation is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        pos : ArrayLike
            Position (x, y, z) to translate by.
        """
        pos = as_vec4(np.array(pos))
        return self.dot(translate(pos[0, :3]))

    def rotated(
        self, angle: float, axis: ArrayLike, about: ArrayLike | None = None
    ) -> Transform:
        """Return new transform, rotated some angle about a given axis.

        The rotation is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        angle : float
            The angle of rotation, in degrees.
        axis : array-like
            The x, y and z coordinates of the axis vector to rotate around.
        about : array-like or None
            The x, y and z coordinates to rotate around. If None, will rotate around
            the origin (0, 0, 0).
        """
        if about is not None:
            about = as_vec4(about)[0, :3]
            return self.translated(-about).dot(rotate(angle, axis)).translated(about)
        return self.dot(rotate(angle, axis))

    def scaled(
        self, scale_factor: ArrayLike, center: ArrayLike | None = None
    ) -> Transform:
        """Return new transform, scaled about a given origin.

        The scaling is applied *after* the transformations already present
        in the matrix.

        Parameters
        ----------
        scale_factor : array-like
            Scale factors along x, y and z axes.
        center : array-like or None
            The x, y and z coordinates to scale around. If None,
            (0, 0, 0) will be used.
        """
        _scale = scale(as_vec4(scale_factor, default=(1, 1, 1, 1))[0, :3])
        if center is not None:
            center = as_vec4(center)[0, :3]
            _scale = np.dot(np.dot(translate(-center), _scale), translate(center))
        return self.dot(_scale)

    @_arg_to_vec4
    def map(self, coords: ArrayLike) -> NDArray:
        """Map coordinates.

        Parameters
        ----------
        coords : array-like
            Coordinates to map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        # looks backwards, but both matrices are transposed.
        return cast(NDArray, np.dot(coords, self.root))

    @_arg_to_vec4
    def imap(self, coords: ArrayLike) -> NDArray:
        """Inverse map coordinates.

        Parameters
        ----------
        coords : array-like
            Coordinates to inverse map.

        Returns
        -------
        coords : ndarray
            Coordinates.
        """
        return cast(NDArray, np.dot(coords, np.linalg.inv(self.root)))

    @classmethod
    def chain(cls, *transforms: Transform) -> Transform:
        """Chain multiple transforms together.

        Parameters
        ----------
        transforms : Transform
            Transforms to chain.

        Returns
        -------
        transform : Transform
            Chained transform.
        """
        return reduce(lambda a, b: a @ b, transforms, cls())

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transform):
            return NotImplemented
        return np.allclose(self.root, value.root)

    def __hash__(self) -> int:
        return hash(self.root.tobytes())


# from vispy ...


def rotate(angle: float, axis: ArrayLike) -> np.ndarray:
    """Return 4x4 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.

    Returns
    -------
    M : ndarray
        Transformation matrix describing the rotation.
    """
    angle = np.radians(angle)
    axis = np.array(axis, copy=False)
    if len(axis) != 3:
        raise ValueError("axis must be a 3-element vector")
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = [
        [cx * x + c, cy * x - z * s, cz * x + y * s, 0.0],
        [cx * y + z * s, cy * y + c, cz * y - x * s, 0.0],
        [cx * z - y * s, cy * z + x * s, cz * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return np.array(M).T


def translate(offset: Iterable[float]) -> np.ndarray:
    """Translate by an offset (x, y, z) .

    Parameters
    ----------
    offset : Iterable[float]
        Must be length 3. Translation in x, y, z.

    Returns
    -------
    M : ndarray
        Transformation matrix describing the translation.
    """
    _offset = tuple(offset)
    if len(_offset) != 3:
        raise ValueError("offset must be a length 3 sequence")
    x, y, z = _offset
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0],
        ]
    )


def scale(s: Sized) -> np.ndarray:
    """Non-uniform scaling along the x, y, and z axes.

    Parameters
    ----------
    s : array-like, shape (3,)
        Scaling in x, y, z.

    Returns
    -------
    M : ndarray
        Transformation matrix describing the scaling.
    """
    if len(s) != 3:
        raise ValueError("scale must be a length 3 sequence")
    return np.array(np.diag(np.concatenate([s, (1.0,)])))


def as_vec4(obj: ArrayLike, default: ArrayLike = (0, 0, 0, 1)) -> np.ndarray:
    """Convert `obj` to 4-element vector (numpy array with shape[-1] == 4).

    Parameters
    ----------
    obj : array-like
        Original object.
    default : array-like
        The defaults to use if the object does not have 4 entries.

    Returns
    -------
    obj : array-like
        The object promoted to have 4 elements.

    Notes
    -----
    `obj` will have at least two dimensions.

    If `obj` has < 4 elements, then new elements are added from `default`.
    For inputs intended as a position or translation, use default=(0,0,0,1).
    For inputs intended as scale factors, use default=(1,1,1,1).

    """
    obj = np.atleast_2d(obj)
    # For multiple vectors, reshape to (..., 4)
    if obj.shape[-1] < 4:
        new = np.empty(obj.shape[:-1] + (4,), dtype=obj.dtype)
        new[:] = default
        new[..., : obj.shape[-1]] = obj
        obj = new
    elif obj.shape[-1] > 4:
        raise TypeError(f"Array shape {obj.shape} cannot be converted to vec4")
    return obj
