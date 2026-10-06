from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rendercanvas import BaseRenderCanvas


def close_rendercanvas(canvas: "BaseRenderCanvas") -> None:
    """Close a render canvas after resolving any asynchronous bitmap download."""
    context = getattr(canvas, "_canvas_context", None)
    downloader = getattr(context, "_downloader", None)
    if downloader is not None:
        # rendercanvas 2.3 leaves an in-flight bitmap presentation alive during
        # close.  Its completion can race destruction of an embedded Qt/wx
        # widget.  Cancel it and synchronously unmap its staging buffer first.
        downloader._clear_pending_download()
    canvas.close()


def rendercanvas_class() -> "type[BaseRenderCanvas]":
    from ndv.views._app import GuiFrontend, gui_frontend

    frontend = gui_frontend()
    if frontend == GuiFrontend.QT:
        import rendercanvas.qt
        from qtpy.QtCore import QSize

        class QRenderWidget(rendercanvas.qt.QRenderWidget):
            def _rc_request_paint(self) -> None:
                # An asynchronous bitmap presentation can complete after close.
                # rendercanvas 2.3 otherwise calls QWidget.update() on the
                # already-deleted PySide object from its completion callback.
                if not self.get_closed():
                    super()._rc_request_paint()

            def _rc_close(self) -> None:
                # This widget is embedded in and owned by the frontend view.
                # Base rendercanvas cleanup has already released its context and
                # event queue when this hook runs.  Let Qt close the native child
                # with its parent; closing it here leaves queued PySide paint
                # events referring to a deleted QRenderWidget.
                self._is_closed = True

            def sizeHint(self) -> QSize:
                return QSize(self.width(), self.height())

            def keyPressEvent(self, event: Any) -> None:
                super().keyPressEvent(event)
                event.ignore()  # pass event to parent for global shortcuts

            def keyReleaseEvent(self, event: Any) -> None:
                super().keyReleaseEvent(event)
                event.ignore()  # pass event to parent for global shortcuts

        return QRenderWidget

    if frontend == GuiFrontend.JUPYTER:
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
