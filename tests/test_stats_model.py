import numpy as np
import pytest

from ndv.histogram.model import StatsModel

EPSILON = 1e-6


@pytest.fixture
def data() -> np.ndarray:
    gen = np.random.default_rng(0xDEADBEEF)
    # Average - 1.000104
    # Std. Dev. - 10.003385
    data = gen.normal(1, 10, (1000, 1000))
    return data


def test_empty_stats_model() -> None:
    model = StatsModel()
    with pytest.raises(RuntimeError):
        _ = model.data
    assert model.average is None
    assert model.standard_deviation is None
    assert model.histogram is None
    assert model.bins == 256


def test_stats_model(data: np.ndarray) -> None:
    model = StatsModel()
    model.data = data
    assert np.all(model.data == data)
    # Basic regression tests
    assert abs(model.average - 1.000104) < 1e-6
    assert abs(model.standard_deviation - 10.003385) < 1e-6
    assert 256 == model.bins
    values, edges = model.histogram
    assert len(values) == 256
    assert np.all(values >= 0)
    assert np.all(values <= data.size)
    assert len(edges) == 257
    assert edges[0] == np.min(data)
    assert edges[256] == np.max(data)
