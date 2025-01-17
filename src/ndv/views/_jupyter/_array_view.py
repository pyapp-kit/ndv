from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cmap
import ipywidgets as widgets

from ndv.models._array_display_model import ChannelMode
from ndv.views.bases import ArrayView, LutView

if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Mapping, Sequence

    from vispy.app.backends import _jupyter_rfb

    from ndv._types import AxisKey
    from ndv.models._data_display_model import _ArrayDataDisplayModel

# not entirely sure why it's necessary to specifically annotat signals as : PSignal
# i think it has to do with type variance?


class JupyterLutView(LutView):
    def __init__(self) -> None:
        # WIDGETS
        self._visible = widgets.Checkbox(value=True, indent=False)
        self._visible.layout.width = "60px"
        self._cmap = widgets.Dropdown(
            options=[
                "gray",
                "red",
                "green",
                "blue",
                "cyan",
                "magenta",
                "yellow",
                "viridis",
                "magma",
            ],
            value="gray",
        )
        self._cmap.layout.width = "200px"
        self._clims = widgets.FloatRangeSlider(
            value=[0, 2**16],
            min=0,
            max=2**16,
            step=1,
            orientation="horizontal",
            readout=True,
            readout_format=".0f",
        )
        self._clims.layout.width = "100%"
        self._auto_clim = widgets.ToggleButton(
            value=True,
            description="Auto",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Auto scale",
            layout=widgets.Layout(width="65px"),
        )

        # LAYOUT

        self.layout = widgets.HBox(
            [self._visible, self._cmap, self._clims, self._auto_clim]
        )

        # CONNECTIONS
        self._visible.observe(self._on_visible_changed, names="value")
        self._cmap.observe(self._on_cmap_changed, names="value")
        self._clims.observe(self._on_clims_changed, names="value")
        self._auto_clim.observe(self._on_autoscale_changed, names="value")

    # ------------------ emit changes to the controller ------------------

    def _on_clims_changed(self, change: dict[str, Any]) -> None:
        self.climsChanged.emit(self._clims.value)

    def _on_visible_changed(self, change: dict[str, Any]) -> None:
        self.visibilityChanged.emit(self._visible.value)

    def _on_cmap_changed(self, change: dict[str, Any]) -> None:
        self.cmapChanged.emit(cmap.Colormap(self._cmap.value))

    def _on_autoscale_changed(self, change: dict[str, Any]) -> None:
        self.autoscaleChanged.emit(self._auto_clim.value)

    # ------------------ receive changes from the controller ---------------

    def set_channel_name(self, name: str) -> None:
        self._visible.description = name

    # NOTE: it's important to block signals when setting values from the controller
    # to avoid loops, unnecessary updates, and unexpected behavior

    def set_auto_scale(self, auto: bool) -> None:
        with self.autoscaleChanged.blocked():
            self._auto_clim.value = auto

    def set_colormap(self, cmap: cmap.Colormap) -> None:
        with self.cmapChanged.blocked():
            self._cmap.value = cmap.name.split(":")[-1]  # FIXME: this is a hack

    def set_clims(self, clims: tuple[float, float]) -> None:
        with self.climsChanged.blocked():
            self._clims.value = clims

    def set_channel_visible(self, visible: bool) -> None:
        with self.visibilityChanged.blocked():
            self._visible.value = visible

    def set_gamma(self, gamma: float) -> None:
        pass

    def set_visible(self, visible: bool) -> None:
        # show or hide the actual widget itself
        self.layout.layout.display = "flex" if visible else "none"

    def close(self) -> None:
        self.layout.close()

    def frontend_widget(self) -> Any:
        return self.layout


SPIN_GIF = str(Path(__file__).parent.parent / "_resources" / "spin.gif")


class JupyterArrayView(ArrayView):
    def __init__(
        self,
        canvas_widget: _jupyter_rfb.CanvasBackend,
        data_model: _ArrayDataDisplayModel,
    ) -> None:
        # WIDGETS
        self._data_model = data_model
        self._canvas_widget = canvas_widget
        self._visible_axes: Sequence[AxisKey] = []

        self._sliders: dict[Hashable, widgets.IntSlider] = {}
        self._slider_box = widgets.VBox([], layout=widgets.Layout(width="100%"))
        self._luts_box = widgets.VBox([], layout=widgets.Layout(width="100%"))

        # labels for data and hover info
        self._data_info_label = widgets.Label()
        self._hover_info_label = widgets.Label()

        # spinner to indicate progress
        self._progress_spinner = widgets.Image.from_file(
            SPIN_GIF, width=18, height=18, layout=widgets.Layout(display="none")
        )

        # the button that controls the display mode of the channels
        self._channel_mode_combo = widgets.Dropdown(
            options=[ChannelMode.GRAYSCALE, ChannelMode.COMPOSITE],
            value=str(ChannelMode.GRAYSCALE),
        )
        self._channel_mode_combo.layout.width = "120px"
        self._channel_mode_combo.layout.align_self = "flex-end"
        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

        # Reset zoom button
        self._reset_zoom_btn = widgets.Button(
            tooltip="Reset Zoom",
            icon="expand",
            layout=widgets.Layout(width="40px"),
        )
        self._reset_zoom_btn.on_click(self._on_reset_zoom_clicked)

        # 3d view button
        self._ndims_btn = widgets.ToggleButton(
            value=False,
            description="3D",
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="View in 3D",
            layout=widgets.Layout(width="40px"),
        )
        self._ndims_btn.observe(self._on_ndims_toggled, names="value")

        # LAYOUT

        top_row = widgets.HBox(
            [self._data_info_label, self._progress_spinner],
            layout=widgets.Layout(
                display="flex",
                justify_content="space-between",
                align_items="center",
            ),
        )

        try:
            width = getattr(canvas_widget, "css_width", "600px").replace("px", "")
            width = f"{int(width) + 4}px"
        except Exception:
            width = "604px"

        btns = widgets.HBox(
            [self._channel_mode_combo, self._ndims_btn, self._reset_zoom_btn],
            layout=widgets.Layout(justify_content="flex-end"),
        )
        self.layout = widgets.VBox(
            [
                top_row,
                self._canvas_widget,
                self._hover_info_label,
                self._slider_box,
                self._luts_box,
                btns,
            ],
            layout=widgets.Layout(width=width),
        )

        # CONNECTIONS

        self._channel_mode_combo.observe(self._on_channel_mode_changed, names="value")

    def create_sliders(self, coords: Mapping[int, Sequence]) -> None:
        """Update sliders with the given coordinate ranges."""
        sliders = []
        self._sliders.clear()
        for axis, _coords in coords.items():
            if not isinstance(_coords, range):
                raise NotImplementedError("Only range is supported for now")

            sld = widgets.IntSlider(
                value=_coords.start,
                min=_coords.start,
                max=_coords.stop - 1,
                step=_coords.step,
                description=str(axis),
                continuous_update=True,
                orientation="horizontal",
            )
            sld.layout.width = "99%"
            sld.observe(self._on_slider_change, "value")
            sliders.append(sld)
            self._sliders[axis] = sld
        self._slider_box.children = sliders

        self.currentIndexChanged.emit()

    def hide_sliders(
        self, axes_to_hide: Container[Hashable], show_remainder: bool = True
    ) -> None:
        """Hide sliders based on visible axes."""
        for ax, slider in self._sliders.items():
            if ax in axes_to_hide:
                slider.layout.display = "none"
            elif show_remainder:
                slider.layout.display = "flex"

    def current_index(self) -> Mapping[AxisKey, int | slice]:
        """Return the current value of the sliders."""
        return {axis: slider.value for axis, slider in self._sliders.items()}

    def set_current_index(self, value: Mapping[AxisKey, int | slice]) -> None:
        """Set the current value of the sliders."""
        changed = False
        # this type ignore is only necessary because we had to override the signal
        # to be a PSignal in the class def above :(
        with self.currentIndexChanged.blocked():
            for axis, val in value.items():
                if isinstance(val, slice):
                    raise NotImplementedError("Slices are not supported yet")

                if sld := self._sliders.get(axis):
                    if sld.value != val:
                        sld.value = val
                        changed = True
                else:  # pragma: no cover
                    warnings.warn(f"Axis {axis} not found in sliders", stacklevel=2)
        if changed:
            self.currentIndexChanged.emit()

    def add_lut_view(self) -> JupyterLutView:
        """Add a LUT view to the viewer."""
        wdg = JupyterLutView()
        layout = self._luts_box
        layout.children = (*layout.children, wdg.layout)
        return wdg

    def remove_lut_view(self, view: LutView) -> None:
        """Remove a LUT view from the viewer."""
        view = cast("JupyterLutView", view)
        layout = self._luts_box
        layout.children = tuple(
            wdg for wdg in layout.children if wdg != view.frontend_widget()
        )

    def set_data_info(self, data_info: str) -> None:
        self._data_info_label.value = data_info

    def set_hover_info(self, hover_info: str) -> None:
        self._hover_info_label.value = hover_info

    def set_channel_mode(self, mode: ChannelMode) -> None:
        with self.channelModeChanged.blocked():
            self._channel_mode_combo.value = mode.value

    def _on_slider_change(self, change: dict[str, Any]) -> None:
        """Emit signal when a slider value changes."""
        self.currentIndexChanged.emit()

    def _on_channel_mode_changed(self, change: dict[str, Any]) -> None:
        """Emit signal when the channel mode changes."""
        self.channelModeChanged.emit(ChannelMode(change["new"]))

    def add_histogram(self, widget: Any) -> None:
        """Add a histogram widget to the viewer."""
        warnings.warn("Histograms are not supported in Jupyter frontend", stacklevel=2)

    def remove_histogram(self, widget: Any) -> None:
        """Remove a histogram widget from the viewer."""

    def frontend_widget(self) -> Any:
        return self.layout

    def set_visible(self, visible: bool) -> None:
        # show or hide the actual widget itself
        from IPython import display

        if visible:
            display.display(self.layout)  # type: ignore [no-untyped-call]
        else:
            display.clear_output()  # type: ignore [no-untyped-call]

    def visible_axes(self) -> Sequence[AxisKey]:
        return self._visible_axes

    def set_visible_axes(self, axes: Sequence[AxisKey]) -> None:
        self._visible_axes = tuple(axes)
        self._ndims_btn.value = len(axes) == 3

    def _on_ndims_toggled(self, change: dict[str, Any]) -> None:
        if len(self._visible_axes) > 2:
            if not change["new"]:  # is now 2D
                self._visible_axes = self._visible_axes[-2:]
        else:
            z_ax = None
            if wrapper := self._data_model.data_wrapper:
                z_ax = wrapper.guess_z_axis()
            if z_ax is None:
                # get the last slider that is not in visible axes
                z_ax = next(
                    ax for ax in reversed(self._sliders) if ax not in self._visible_axes
                )
            self._visible_axes = (z_ax, *self._visible_axes)
        # TODO: a future PR may decide to set this on the model directly...
        # since we now have access to it.
        self.visibleAxesChanged.emit()

    def _on_reset_zoom_clicked(self, change: dict[str, Any]) -> None:
        self.resetZoomClicked.emit()

    def close(self) -> None:
        self.layout.close()

    def set_progress_spinner_visible(self, visible: bool) -> None:
        self._progress_spinner.layout.display = "flex" if visible else "none"
