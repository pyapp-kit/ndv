import numpy as np
from qtpy.QtWidgets import QApplication

from ndv.histogram._model import StatsModel
from ndv.histogram._vispy import VispyHistogramView

app = QApplication([])

stats = StatsModel()
view = VispyHistogramView()


def connection(data: tuple[np.ndarray, np.ndarray]) -> None:
    values, bins = data
    view.set_histogram(values, bins)
    view.set_clims((bins[0], bins[-1]))
    view.set_gamma(1)


stats.events.histogram.connect(connection)


# TODO: Once we have a LutModel to play with, direct these view signals to the model
# The controller will then change its state to the signal value, which should then
# be hooked up to call edit_X.
def foo(gamma: float) -> None:
    view.set_gamma(gamma)


def bar(clims: tuple[float, float]) -> None:
    view.set_clims(clims)


view.gammaChanged.connect(foo)
view.climsChanged.connect(bar)

view.view().show()

stats.data = np.random.normal(10, 10, 10000)
app.exec()
