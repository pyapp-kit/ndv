import numpy as np
import pytest

from ndv.models._ring_buffer import RingBuffer


def test_dtype() -> None:
    r = RingBuffer(5)
    assert r.dtype == np.dtype(np.float64)

    r = RingBuffer(5, dtype=bool)
    assert r.dtype == np.dtype(bool)


def test_sizes() -> None:
    rb = RingBuffer(5, dtype=(int, 2))
    assert rb.maxlen == 5
    assert len(rb) == 0
    assert rb.shape == (0, 2)

    rb.append([0, 0])
    assert rb.maxlen == 5
    assert len(rb) == 1
    assert rb.shape == (1, 2)


def test_append() -> None:
    rb = RingBuffer(5)

    rb.append(1)
    np.testing.assert_equal(rb, np.array([1]))
    assert len(rb) == 1

    rb.append(2)
    np.testing.assert_equal(rb, np.array([1, 2]))
    assert len(rb) == 2

    rb.append(3)
    rb.append(4)
    rb.append(5)
    np.testing.assert_equal(rb, np.array([1, 2, 3, 4, 5]))
    assert len(rb) == 5

    rb.append(6)
    np.testing.assert_equal(rb, np.array([2, 3, 4, 5, 6]))
    assert len(rb) == 5

    assert rb[4] == 6
    assert rb[-1] == 6


def test_getitem() -> None:
    rb = RingBuffer(5)
    rb.extend([1, 2, 3])
    rb.extendleft([4, 5])
    expected = np.array([4, 5, 1, 2, 3])
    np.testing.assert_equal(rb, expected)

    for i in range(rb.maxlen):
        assert expected[i] == rb[i]

    ii = [0, 4, 3, 1, 2]
    np.testing.assert_equal(rb[ii], expected[ii])


def test_getitem_negative_index() -> None:
    rb = RingBuffer(5)
    rb.extend([1, 2, 3])
    assert rb[-1] == 3


def test_appendleft() -> None:
    rb = RingBuffer(5)

    rb.appendleft(1)
    np.testing.assert_equal(rb, np.array([1]))
    assert len(rb) == 1

    rb.appendleft(2)
    np.testing.assert_equal(rb, np.array([2, 1]))
    assert len(rb) == 2

    rb.appendleft(3)
    rb.appendleft(4)
    rb.appendleft(5)
    np.testing.assert_equal(rb, np.array([5, 4, 3, 2, 1]))
    assert len(rb) == 5

    rb.appendleft(6)
    np.testing.assert_equal(rb, np.array([6, 5, 4, 3, 2]))
    assert len(rb) == 5


def test_extend() -> None:
    rb = RingBuffer(5)
    rb.extend([1, 2, 3])
    np.testing.assert_equal(rb, np.array([1, 2, 3]))
    rb.popleft()
    rb.extend([4, 5, 6])
    np.testing.assert_equal(rb, np.array([2, 3, 4, 5, 6]))
    rb.extendleft([0, 1])
    np.testing.assert_equal(rb, np.array([0, 1, 2, 3, 4]))

    rb.extendleft([1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_equal(rb, np.array([1, 2, 3, 4, 5]))

    rb.extend([1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_equal(rb, np.array([3, 4, 5, 6, 7]))


def test_pops() -> None:
    rb = RingBuffer(3)
    rb.append(1)
    rb.appendleft(2)
    rb.append(3)
    np.testing.assert_equal(rb, np.array([2, 1, 3]))

    assert rb.pop() == 3
    np.testing.assert_equal(rb, np.array([2, 1]))

    assert rb.popleft() == 2
    np.testing.assert_equal(rb, np.array([1]))

    # test empty pops
    empty = RingBuffer(1)
    with pytest.raises(IndexError, match="pop from an empty RingBuffer"):
        empty.pop()
    with pytest.raises(IndexError, match="pop from an empty RingBuffer"):
        empty.popleft()


def test_2d() -> None:
    rb = RingBuffer(5, dtype=(float, 2))

    rb.append([1, 2])
    np.testing.assert_equal(rb, np.array([[1, 2]]))
    assert len(rb) == 1
    assert np.shape(rb) == (1, 2)

    rb.append([3, 4])
    np.testing.assert_equal(rb, np.array([[1, 2], [3, 4]]))
    assert len(rb) == 2
    assert np.shape(rb) == (2, 2)

    rb.appendleft([5, 6])
    np.testing.assert_equal(rb, np.array([[5, 6], [1, 2], [3, 4]]))
    assert len(rb) == 3
    assert np.shape(rb) == (3, 2)

    np.testing.assert_equal(rb[0], [5, 6])
    np.testing.assert_equal(rb[0, :], [5, 6])
    np.testing.assert_equal(rb[:, 0], [5, 1, 3])


def test_3d() -> None:
    np.random.seed(0)
    frame_shape = (32, 32)

    rb = RingBuffer(5, dtype=("u2", frame_shape))
    frame = np.random.randint(0, 65535, frame_shape, dtype="u2")
    rb.append(frame)
    np.testing.assert_equal(rb, frame[None])
    frame2 = np.random.randint(0, 65535, frame_shape, dtype="u2")
    rb.append(frame2)
    np.testing.assert_equal(rb[-1], frame2)
    np.testing.assert_equal(rb, np.array([frame, frame2]))

    # fill buffer
    for _ in range(5):
        rb.append(np.random.randint(0, 65535, frame_shape, dtype="u2"))

    # add known frame
    frame3 = np.random.randint(0, 65535, frame_shape, dtype="u2")
    rb.append(frame3)
    np.testing.assert_equal(rb[-1], frame3)


def test_iter() -> None:
    rb = RingBuffer(5)
    for i in range(5):
        rb.append(i)
    for i, j in zip(rb, range(5)):
        assert i == j


def test_repr() -> None:
    rb = RingBuffer(5, dtype=int)
    for i in range(5):
        rb.append(i)

    assert repr(rb) == "<RingBuffer of array([0, 1, 2, 3, 4])>"


def test_no_overwrite() -> None:
    rb = RingBuffer(3, allow_overwrite=False)
    rb.append(1)
    rb.append(2)
    rb.appendleft(3)
    with pytest.raises(IndexError, match="overwrite"):
        rb.appendleft(4)
    with pytest.raises(IndexError, match="overwrite"):
        rb.extendleft([4])
    rb.extendleft([])

    np.testing.assert_equal(rb, np.array([3, 1, 2]))
    with pytest.raises(IndexError, match="overwrite"):
        rb.append(4)
    with pytest.raises(IndexError, match="overwrite"):
        rb.extend([4])
    rb.extend([])

    # works fine if we pop the surplus
    rb.pop()
    rb.append(4)
    np.testing.assert_equal(rb, np.array([3, 1, 4]))


def test_degenerate() -> None:
    r = RingBuffer(0)
    np.testing.assert_equal(r, np.array([]))

    # this does not error with deque(maxlen=0), so should not error here
    try:
        r.append(0)
        r.appendleft(0)
    except IndexError:
        pytest.fail("IndexError raised when appending to a degenerate RingBuffer")
