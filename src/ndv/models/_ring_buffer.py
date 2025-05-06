"""Numpy Ring Buffer.

Vendored from https://github.com/eric-wieser/numpy_ringbuffer
April 27, 2025
(and subsequently updated).

MIT License

Copyright (c) 2016 Eric Wieser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, cast, overload

import numpy as np
from psygnal import Signal

if TYPE_CHECKING:
    from typing import Any, Callable, SupportsIndex

    import numpy.typing as npt


class RingBuffer(Sequence):
    """Ring buffer structure with a given capacity and element type.

    Parameters
    ----------
    max_capacity: int
        The maximum capacity of the ring buffer.
    dtype: npt.DTypeLike
        Desired type (and shape) of individual buffer elements.
        This is passed to `np.empty`, so it can be any
        [dtype-like object](https://numpy.org/doc/stable/reference/arrays.dtypes.html).
        Common scenarios will be:
            - a fixed dtype (e.g. `int`, `np.uint8`, `'u2'`, `np.dtype('f4')`)
            - a `(fixed_dtype, shape)` tuple (e.g. `('uint16', (512, 512))`)
    allow_overwrite: bool
        If false, throw an IndexError when trying to append to an already full
        buffer.
    create_buffer: Callable[[int, npt.DTypeLike], npt.NDArray]
        A callable that creates the underlying array.
        May be used to customize the initialization of the array.  Defaults to
        `np.empty`.

    Notes
    -----
    Vendored from [numpy-ringbuffer](https://github.com/eric-wieser/numpy_ringbuffer),
    by Eric Wieser (MIT License).  And updated with typing and signals.
    """

    resized = Signal(int)

    def __init__(
        self,
        max_capacity: int,
        dtype: npt.DTypeLike = float,
        *,
        allow_overwrite: bool = True,
        create_buffer: Callable[[int, npt.DTypeLike], npt.NDArray] = np.empty,
    ) -> None:
        self._arr = create_buffer(max_capacity, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = max_capacity
        self._allow_overwrite = allow_overwrite

    # -------------------- Properties --------------------

    @property
    def is_full(self) -> bool:
        """True if there is no more space in the buffer."""
        return len(self) == self._capacity

    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the buffer."""
        return self._arr.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the valid buffer (excluding unused space)."""
        return (len(self),) + self._arr.shape[1:]

    # these mirror methods from deque
    @property
    def maxlen(self) -> int:
        """Return the maximum capacity of the buffer."""
        return self._capacity

    # -------------------- Methods --------------------

    def append(self, value: npt.ArrayLike) -> None:
        """Append a value to the right end of the buffer."""
        if was_full := self.is_full:
            if not self._allow_overwrite:
                raise IndexError("append to a full RingBuffer with overwrite disabled")
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()
        if not was_full:
            self.resized.emit(len(self))

    def appendleft(self, value: npt.ArrayLike) -> None:
        """Append a value to the left end of the buffer."""
        if was_full := self.is_full:
            if not self._allow_overwrite:
                raise IndexError("append to a full RingBuffer with overwrite disabled")
            elif not len(self):
                return
            else:
                self._right_index -= 1

        self._left_index -= 1
        self._fix_indices()
        self._arr[self._left_index] = value
        if not was_full:
            self.resized.emit(len(self))

    def pop(self) -> np.ndarray:
        """Pop a value from the right end of the buffer."""
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        self._right_index -= 1
        self._fix_indices()
        res = cast("np.ndarray", self._arr[self._right_index % self._capacity])
        self.resized.emit(len(self))
        return res

    def popleft(self) -> np.ndarray:
        """Pop a value from the left end of the buffer."""
        if len(self) == 0:
            raise IndexError("pop from an empty RingBuffer")
        res = cast("np.ndarray", self._arr[self._left_index])
        self._left_index += 1
        self._fix_indices()
        self.resized.emit(len(self))
        return res

    def extend(self, values: npt.ArrayLike) -> None:
        """Extend the buffer with the given values."""
        values = np.asarray(values)
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError(
                    "Extending a RingBuffer such that it would overflow, "
                    "with overwrite disabled."
                )
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[-self._capacity :]
            self._right_index = self._capacity
            self._left_index = 0
            return

        was_full = self.is_full
        ri = self._right_index % self._capacity
        sl1 = np.s_[ri : min(ri + lv, self._capacity)]
        sl2 = np.s_[: max(ri + lv - self._capacity, 0)]
        self._arr[sl1] = values[: sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start :]
        self._right_index += lv

        self._left_index = max(self._left_index, self._right_index - self._capacity)
        self._fix_indices()
        if not was_full:
            self.resized.emit(len(self))

    def extendleft(self, values: npt.ArrayLike) -> None:
        """Prepend the buffer with the given values."""
        values = np.asarray(values)
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError(
                    "Extending a RingBuffer such that it would overflow, "
                    "with overwrite disabled"
                )
            elif not len(self):
                return
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[: self._capacity]
            self._right_index = self._capacity
            self._left_index = 0
            return

        was_full = self.is_full
        self._left_index -= lv
        self._fix_indices()
        li = self._left_index
        sl1 = np.s_[li : min(li + lv, self._capacity)]
        sl2 = np.s_[: max(li + lv - self._capacity, 0)]
        self._arr[sl1] = values[: sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start :]

        self._right_index = min(self._right_index, self._left_index + self._capacity)
        if not was_full:
            self.resized.emit(len(self))

    # numpy compatibility
    def __array__(
        self, dtype: npt.DTypeLike = None, copy: bool | None = None
    ) -> np.ndarray:
        if copy is False:
            warnings.warn(
                "`copy=False` isn't supported. A copy is always created.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.asarray(self._unwrap(), dtype=dtype)

    # implement Sequence methods
    def __len__(self) -> int:
        """Return the number of valid elements in the buffer."""
        return self._right_index - self._left_index

    @overload  # type: ignore [override]
    def __getitem__(self, key: SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: Any, /) -> np.ndarray: ...
    def __getitem__(self, key: Any) -> np.ndarray | Any:
        """Index into the buffer.

        This supports both simple and fancy indexing.
        """
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing quickly
        if not isinstance(key, tuple):
            item_arr = np.asarray(key)
            if issubclass(item_arr.dtype.type, np.integer):
                # Map negative indices to positive ones
                item_arr = np.where(item_arr < 0, item_arr + len(self), item_arr)
                # Map indices to the range of the buffer
                item_arr = (item_arr + self._left_index) % self._capacity
                return self._arr[item_arr]

        # for everything else, get it right at the expense of efficiency
        return self._unwrap()[key]

    def __iter__(self) -> Iterator[Any]:
        # this is comparable in speed to using itertools.chain
        return iter(self._unwrap())

    def __repr__(self) -> str:
        """Return a string representation of the buffer."""
        return f"<{self.__class__.__name__} of {np.asarray(self)!r}>"

    def _unwrap(self) -> np.ndarray:
        """Copy the data from this buffer into unwrapped form."""
        return np.concatenate(
            (
                self._arr[self._left_index : min(self._right_index, self._capacity)],
                self._arr[: max(self._right_index - self._capacity, 0)],
            )
        )

    def _fix_indices(self) -> None:
        """Enforce our invariant that 0 <= self._left_index < self._capacity."""
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity
