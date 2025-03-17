# /// script
# dependencies = [
#   "ndv[vispy,pyqt]",
#   "openwfs",
# ]
# ///
"""A minimal microscope dashboard.

The `SyntheticManager` acts as an microscope simulator.
It generates images of a specimen and sends them to the `DashboardWidget`.

The `DashboardWidget` is a simple GUI that displays the images and allows the user to:
- starts/stops the simulation;
- move the stage in the x and y directions.
"""

from typing import Any, cast

import astropy.units as u
import numpy as np
from openwfs.simulation import Camera, Microscope, StaticSource
from qtpy import QtWidgets as QtW
from qtpy.QtCore import QObject, Qt, Signal

from ndv import ArrayViewer


class SyntheticManager(QObject):
    newFrame = Signal(np.ndarray)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        specimen_resolution = (1024, 1024)  # (height, width) in pixels of the image
        specimen_pixel_size = 60 * u.nm  # resolution (pixel size) of the specimen image
        magnification = 40  # magnification from object plane to camera.
        numerical_aperture = 0.85  # numerical aperture of the microscope objective
        wavelength = 532.8 * u.nm  # wavelength of the light, for computing diffraction.
        camera_resolution = (256, 256)  # number of pixels on the camera

        img = np.random.randint(-10000, 10, specimen_resolution, dtype=np.int16)
        img = np.maximum(img, 0)
        src = StaticSource(img, pixel_size=specimen_pixel_size)

        self._microscope = Microscope(
            src,
            magnification=magnification,
            numerical_aperture=numerical_aperture,
            wavelength=wavelength,
        )
        self._camera = Camera(
            self._microscope,
            analog_max=None,
            shot_noise=True,
            digital_max=255,
            shape=camera_resolution,
        )

    def move_stage(self, axis: str, step: float) -> None:
        if axis == "x":
            self._microscope.xy_stage.x += step * u.um
        if axis == "y":
            self._microscope.xy_stage.y += step * u.um

    def toggle_simulation(self, start: bool) -> None:
        if start:
            self._timer_id = self.startTimer(100)
        elif hasattr(self, "_timer_id"):
            self.killTimer(self._timer_id)

    def timerEvent(self, e: Any) -> None:
        self.emit_frame()

    def emit_frame(self) -> None:
        self.newFrame.emit(self._camera.read())


class StageWidget(QtW.QGroupBox):
    stageMoved = Signal(str, int)

    def __init__(self, name: str, axes: list[str], parent: QtW.QWidget) -> None:
        super().__init__(name, parent)
        self._data_key = "data"

        def _make_button(txt: str, *data: Any) -> QtW.QPushButton:
            btn = QtW.QPushButton(txt)
            btn.setAutoRepeat(True)
            btn.setProperty(self._data_key, data)
            btn.clicked.connect(self._move_stage)
            return btn

        layout = QtW.QVBoxLayout(self)
        for ax in axes:
            # spinbox showing stage position
            spin = QtW.QDoubleSpinBox()
            spin.setMinimumWidth(80)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            spin.setSuffix(" µm")
            spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            spin.setButtonSymbols(QtW.QAbstractSpinBox.ButtonSymbols.NoButtons)

            # buttons to move the stage
            down = _make_button("-", ax, spin)
            up = _make_button("+", ax, spin)

            row = QtW.QHBoxLayout()
            row.addWidget(QtW.QLabel(f"<strong>{ax}:</strong>"), 0)
            row.addWidget(spin, 0, Qt.AlignmentFlag.AlignRight)
            row.addWidget(down, 1)
            row.addWidget(up, 1)
            layout.addLayout(row)

    def _move_stage(self) -> None:
        button = cast("QtW.QPushButton", self.sender())
        ax, spin = button.property(self._data_key)
        step = 1 if button.text() == "+" else -1
        cast("QtW.QDoubleSpinBox", spin).stepBy(step)
        self.stageMoved.emit(ax, step)


class DashboardWidget(QtW.QWidget):
    simulationStarted = Signal(bool)
    stageMoved = Signal(str, int)

    def __init__(self) -> None:
        super().__init__()

        self._stage_widget = StageWidget("Stage", ["x", "y"], self)
        self._stage_widget.setEnabled(False)
        self._stage_widget.stageMoved.connect(self.stageMoved)

        self._viewer = ArrayViewer()
        self._viewer._async = False  # Disable async rendering for simplicity

        self._start_button = QtW.QPushButton("Start")
        self._start_button.setMinimumWidth(120)
        self._start_button.toggled.connect(self.start_simulation)
        self._start_button.setCheckable(True)

        bottom = QtW.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self._start_button)
        bottom.addSpacing(120)
        bottom.addWidget(self._stage_widget)

        layout = QtW.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addWidget(self._viewer.widget(), 1)
        layout.addLayout(bottom)

    def start_simulation(self, checked: bool) -> None:
        self._stage_widget.setEnabled(checked)
        self._start_button.setText("Stop" if checked else "Start")
        self.simulationStarted.emit(checked)

    def set_data(self, frame: np.ndarray) -> None:
        self._viewer.data = frame


app = QtW.QApplication([])
manager = SyntheticManager()
wrapper = DashboardWidget()

manager.newFrame.connect(wrapper.set_data)
wrapper.simulationStarted.connect(manager.toggle_simulation)
wrapper.stageMoved.connect(manager.move_stage)
manager.emit_frame()  # just to populate the viewer with an image
wrapper.show()
app.exec()
