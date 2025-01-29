# /// script
# dependencies = [
#   "ndv[vispy,pyqt]",
#   "imageio[tifffile]",
# ]
# ///
"""An example on how to embed the `ArrayViewer` controller in a custom Qt widget."""

from qtpy import QtWidgets

from ndv import ArrayViewer, run_app
from ndv.data import astronaut, cat


class EmbeddingWidget(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._viewer = ArrayViewer()

        self._cat_button = QtWidgets.QPushButton("Load cat image")
        self._cat_button.clicked.connect(self._load_cat)

        self._astronaut_button = QtWidgets.QPushButton("Load astronaut image")
        self._astronaut_button.clicked.connect(self._load_astronaut)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._cat_button)
        btns.addWidget(self._astronaut_button)

        layout = QtWidgets.QVBoxLayout(self)
        # `ArrayViewer.widget()` returns the native Qt widget
        layout.addWidget(self._viewer.widget())
        layout.addLayout(btns)

        self._load_cat()

    def _load_cat(self) -> None:
        self._viewer.data = cat().mean(axis=-1)

    def _load_astronaut(self) -> None:
        self._viewer.data = astronaut().mean(axis=-1)


app = QtWidgets.QApplication([])
widget = EmbeddingWidget()
widget.show()
run_app()
