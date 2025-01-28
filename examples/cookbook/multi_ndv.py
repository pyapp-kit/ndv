# /// script
# dependencies = [
#   "imageio",
#   "ndv[vispy,pyqt]",
# ]
# ///
"""An example on how to embed multiple `ArrayViewer` controllers in a custom Qt widget.

It shows the `astronaut` and `cells3d` images side by side on two different viewers.
"""

from qtpy import QtWidgets

from ndv import ArrayViewer, run_app
from ndv.data import astronaut, cells3d


class MultiNDVWrapper(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        layout = QtWidgets.QGridLayout()
        self._astronaut_viewer = ArrayViewer(astronaut()[:, :, 0])
        self._cells_virewer = ArrayViewer(cells3d())

        # get `ArrayViewer` widget and add it to the layout
        layout.addWidget(self._astronaut_viewer.widget(), 0, 0, 4, 4)
        layout.addWidget(self._cells_virewer.widget(), 0, 5, 4, 4)

        self.setLayout(layout)


app = QtWidgets.QApplication([])
widget = MultiNDVWrapper()
widget.show()
run_app()
