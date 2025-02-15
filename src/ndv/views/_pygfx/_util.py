from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rendercanvas import BaseRenderCanvas


def rendercanvas_class() -> "BaseRenderCanvas":
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

        return rendercanvas.jupyter.JupyterRenderCanvas
    if frontend == GuiFrontend.WX:
        # ...still not working
        # import rendercanvas.wx
        # return rendercanvas.wx.WxRenderWidget
        from wgpu.gui.wx import WxWgpuCanvas

        return WxWgpuCanvas
