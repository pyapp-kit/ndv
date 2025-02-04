# /// script
# dependencies = [
#   "ndv[vispy,pyqt]",
#   "imageio[tifffile]",
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

        self._astronaut_viewer = ArrayViewer(astronaut().mean(axis=-1))
        self._cells_virewer = ArrayViewer(cells3d(), current_index={0: 30, 1: 1})

        # get `ArrayViewer` widget and add it to the layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self._astronaut_viewer.widget())
        layout.addWidget(self._cells_virewer.widget())


app = QtWidgets.QApplication([])
widget = MultiNDVWrapper()
widget.show()
run_app()
