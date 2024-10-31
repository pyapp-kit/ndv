import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QPushButton

from ndv.histogram._model import StatsModel
from ndv.histogram._qt import QtHistogramView
from ndv.histogram._vispy import VispyHistogramView

app = QApplication([])

stats = StatsModel()
view = VispyHistogramView()


def _connection(data: tuple[np.ndarray, np.ndarray]) -> None:
    values, bins = data
    view.set_histogram(values, bins)
    # view.set_clims((bins[0], bins[-1]))
    # view.set_gamma(1)


stats.events.histogram.connect(_connection)


# TODO: Once we have a LutModel to play with, direct these view signals to the model
# The controller will then change its state to the signal value, which should then
# be hooked up to call edit_X.
def _foo(gamma: float) -> None:
    view.set_gamma(gamma)


def _bar(clims: tuple[float, float]) -> None:
    view.set_clims(clims)


view.set_domain((-40, 60))
# view.set_range((0, 3500))
view.gammaChanged.connect(_foo)
view.climsChanged.connect(_bar)

biggerview = QtHistogramView(view)
biggerview.view().show()

data_btn = QPushButton("Change Data")
data_btn.setCheckable(True)
timer = QTimer()
timer.setInterval(10)
timer.blockSignals(True)


def _update_data() -> None:
    """Replaces the displayed data."""
    stats.data = np.random.normal(10, 10, 10000)


timer.timeout.connect(_update_data)


biggerview._layout.addWidget(data_btn)
data_btn.toggled.connect(lambda toggle: timer.blockSignals(not toggle))
timer.start()


app.exec()
