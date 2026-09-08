from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rendercanvas import BaseRenderCanvas


def rendercanvas_class() -> "type[BaseRenderCanvas]":
    from ndv.views._app import GuiFrontend, gui_frontend

    frontend = gui_frontend()
    if frontend == GuiFrontend.QT:
        import rendercanvas.qt
        from qtpy.QtCore import QSize

        class QRenderWidget(rendercanvas.qt.QRenderWidget):
            def sizeHint(self) -> QSize:
                return QSize(self.width(), self.height())

            def keyPressEvent(self, event: Any) -> None:
                super().keyPressEvent(event)
                event.ignore()  # pass event to parent for global shortcuts

            def keyReleaseEvent(self, event: Any) -> None:
                super().keyReleaseEvent(event)
                event.ignore()  # pass event to parent for global shortcuts

        return QRenderWidget

    if frontend in (GuiFrontend.JUPYTER, GuiFrontend.MARIMO):
        import rendercanvas.jupyter

        class JupyterRenderCanvas(rendercanvas.jupyter.JupyterRenderCanvas):
            def get_frame(self) -> Any:
                # Workaround for async GPU readback in rendercanvas:
                # _time_to_draw() calls _draw_and_present(force_sync=False),
                # which may complete the present asynchronously, meaning
                # _last_image still holds the previous frame when get_frame()
                # returns. Force a synchronous present so the returned image
                # is always up-to-date.
                self._process_events()
                self._draw_and_present(force_sync=True)
                return self._last_image

        return JupyterRenderCanvas
    if frontend == GuiFrontend.WX:
        import rendercanvas.wx
        import wx

        class WxRenderWidget(rendercanvas.wx.WxRenderWidget):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                # wx.Window requires a parent on macOS to avoid segfaults.
                # Create a temporary hidden frame if no parent is provided,
                # which will be destroyed when the widget is reparented.
                if "parent" not in kwargs and (not args or args[0] is None):
                    kwargs["parent"] = parent = wx.Frame(None)
                    parent.Hide()
                super().__init__(*args, **kwargs)

            def _rc_close(self) -> None:
                # Guard against accessing self.Parent on a deleted C++ object
                try:
                    super()._rc_close()
                except RuntimeError:
                    self._is_closed = True

        return WxRenderWidget

    raise ValueError(f"Unsupported frontend: {frontend}")  # pragma: no cover
