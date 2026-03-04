from typing import TYPE_CHECKING

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

        return QRenderWidget

    if frontend == GuiFrontend.JUPYTER:
        import rendercanvas.jupyter

        return rendercanvas.jupyter.JupyterRenderCanvas  # type: ignore[no-any-return]
    if frontend == GuiFrontend.WX:
        import rendercanvas.wx
        import wx

        class WxRenderWidget(rendercanvas.wx.WxRenderWidget):
            """Ensure the widget always has a parent to avoid segfaults."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                if "parent" not in kwargs and (not args or args[0] is None):
                    # wx.Window segfaults on Reparent if created without
                    # a parent, so use a temporary hidden frame.
                    kwargs["parent"] = wx.Frame(None)
                super().__init__(*args, **kwargs)

        return WxRenderWidget

    raise ValueError(f"Unsupported frontend: {frontend}")  # pragma: no cover
