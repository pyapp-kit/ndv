"""An example on how to embed the `ArrayViewer` controller in a custom Qt widget.

To run this example install `ndv` with the following:

```bash
pip install ndv[vispy,pyqt] imageio
```

"""

from typing import TYPE_CHECKING

from qtpy import QtWidgets

from ndv import ArrayViewer, run_app
from ndv.data import astronaut, cat

if TYPE_CHECKING:
    import numpy.typing as npt


class EmbeddingWidget(QtWidgets.QWidget):

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self._data: npt.NDArray = cat()[:, :, 0]
        self._viewer = ArrayViewer(self._data)

        self._cat_button = QtWidgets.QPushButton("Load cat image")
        self._astronaut_button = QtWidgets.QPushButton("Load astronaut image")

        self._cat_button.clicked.connect(self._load_cat)
        self._astronaut_button.clicked.connect(self._load_astronaut)

        # get `ArrayViewer` widget and add it to the layout;
        # you can specify the row and column span of the widget
        layout.addWidget(self._viewer.widget(), 0, 0, 1, 5)

        # add buttons to the layout
        layout.addWidget(self._cat_button, 1, 0)
        layout.addWidget(self._astronaut_button, 1, 1)

        self.setLayout(layout)

    def _load_cat(self) -> None:
        self._viewer.data = None
        self._viewer.data = cat()[:, :, 0]

    def _load_astronaut(self) -> None:
        self._viewer.data = None
        self._viewer.data = astronaut()[:, :, 0]

app = QtWidgets.QApplication([])
widget = EmbeddingWidget()
widget.show()
run_app()

