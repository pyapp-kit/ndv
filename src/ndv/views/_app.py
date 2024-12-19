from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from contextlib import suppress
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

if TYPE_CHECKING:
    from types import TracebackType

    from IPython.core.interactiveshell import InteractiveShell

    from ndv.views.bases import ArrayCanvas, ArrayView, HistogramCanvas


GUI_ENV_VAR = "NDV_GUI_FRONTEND"
"""Preferred GUI frontend. If not set, the first available GUI frontend is used."""

CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
"""Preferred canvas backend. If not set, the first available canvas backend is used."""

DEBUG_EXCEPTIONS = "NDV_DEBUG_EXCEPTIONS"
"""Whether to drop into a debugger when an exception is raised. Default False."""

EXIT_ON_EXCEPTION = "NDV_EXIT_ON_EXCEPTION"
"""Whether to exit the application when an exception is raised. Default False."""

IPYTHON_GUI_QT = "NDV_IPYTHON_GUI_QT"
"""Whether to use gui_qt magic when running in IPython. Default True."""


class GuiFrontend(str, Enum):
    QT = "qt"
    JUPYTER = "jupyter"
    WX = "wx"


class CanvasBackend(str, Enum):
    VISPY = "vispy"
    PYGFX = "pygfx"


class GuiProvider(Protocol):
    @staticmethod
    def is_running() -> bool: ...
    @staticmethod
    def create_app() -> bool: ...
    @staticmethod
    def array_view_class() -> type[ArrayView]: ...
    @staticmethod
    def exec() -> None: ...


class CanvasProvider(Protocol):
    @staticmethod
    def is_imported() -> bool: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def array_canvas_class() -> type[ArrayCanvas]: ...
    @staticmethod
    def histogram_canvas_class() -> type[HistogramCanvas]: ...


class QtProvider(GuiProvider):
    """Provider for PyQt5/PySide2/PyQt6/PySide6."""

    _APP_INSTANCE: ClassVar[Any] = None

    @staticmethod
    def is_running() -> bool:
        for mod_name in ("PyQt5", "PySide2", "PySide6", "PyQt6"):
            if mod := sys.modules.get(f"{mod_name}.QtWidgets"):
                if qapp := getattr(mod, "QApplication", None):
                    return qapp.instance() is not None
        return False

    @staticmethod
    def create_app() -> Any:
        from qtpy.QtWidgets import QApplication

        if (qapp := QApplication.instance()) is None:
            # if we're running in IPython
            # start the %gui qt magic if NDV_IPYTHON_GUI_QT!=0
            if (ipy_shell := _ipython_shell()) and (
                os.getenv(IPYTHON_GUI_QT, "true").lower() not in ("0", "false", "no")
            ):
                ipy_shell.enable_gui("qt")  # type: ignore [no-untyped-call]
            # otherwise create a new QApplication
            else:
                # must be stored in a class variable to prevent garbage collection
                QtProvider._APP_INSTANCE = qapp = QApplication(sys.argv)
                qapp.setOrganizationName("ndv")
                qapp.setApplicationName("ndv")

        _install_excepthook()
        return qapp

    @staticmethod
    def exec() -> None:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance() or QtProvider.create_app()

        for wdg in QApplication.topLevelWidgets():
            wdg.raise_()

        if ipy_shell := _ipython_shell():
            # if we're already in an IPython session with %gui qt, don't block
            if str(ipy_shell.active_eventloop).startswith("qt"):
                return

        app.exec()

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._qt.qt_view import QtArrayView

        return QtArrayView


class WxProvider(GuiProvider):
    """Provider for wxPython."""

    @staticmethod
    def is_running() -> bool:
        if wx := sys.modules.get("wx"):
            return wx.App.Get() is not None
        return False

    @staticmethod
    def create_app() -> Any:
        import wx

        if (wxapp := wx.App.Get()) is None:
            wxapp = wx.App()

        _install_excepthook()
        return wxapp

    @staticmethod
    def exec() -> None:
        import wx

        app = wx.App.Get() or WxProvider.create_app()
        app.MainLoop()
        _install_excepthook()

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._wx.wx_view import WxArrayView

        return WxArrayView


class JupyterProvider(GuiProvider):
    """Provider for Jupyter notebooks/lab (NOT ipython)."""

    @staticmethod
    def is_running() -> bool:
        if ipy_shell := _ipython_shell():
            return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
        return False

    @staticmethod
    def create_app() -> Any:
        if not JupyterProvider.is_running():  # pragma: no cover
            # if we got here, it probably means that someone used
            # NDV_GUI_FRONTEND=jupyter without actually being in a juptyer notebook
            raise RuntimeError(
                "Jupyter is not running a notebook shell.  Cannot create app."
            )
        return None

    @staticmethod
    def exec() -> None:
        pass

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._jupyter.jupyter_view import JupyterArrayView

        return JupyterArrayView


class VispyProvider(CanvasProvider):
    @staticmethod
    def is_imported() -> bool:
        return "vispy" in sys.modules

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("vispy") is not None

    @staticmethod
    def array_canvas_class() -> type[ArrayCanvas]:
        from vispy.app import use_app

        from ndv.views._vispy._vispy import VispyViewerCanvas

        # these may not be necessary, since we likely have already called
        # create_app by this point and vispy will autodetect that.
        # it's an extra precaution
        _frontend = gui_frontend()
        if _frontend == GuiFrontend.JUPYTER:
            use_app("jupyter_rfb")
        elif _frontend == GuiFrontend.WX:
            use_app("wx")
        # there is no `use_app('qt')`... it's all specific to pyqt/pyside, etc...
        # so we just let vispy autodetect it

        return VispyViewerCanvas

    @staticmethod
    def histogram_canvas_class() -> type[HistogramCanvas]:
        from ndv.views._vispy._histogram import VispyHistogramCanvas

        return VispyHistogramCanvas


class PygfxProvider(CanvasProvider):
    @staticmethod
    def is_imported() -> bool:
        return "pygfx" in sys.modules

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("pygfx") is not None

    @staticmethod
    def array_canvas_class() -> type[ArrayCanvas]:
        from ndv.views._pygfx._pygfx import GfxArrayCanvas

        return GfxArrayCanvas

    @staticmethod
    def histogram_canvas_class() -> type[HistogramCanvas]:
        raise RuntimeError("Histogram not supported for pygfx")


def _ipython_shell() -> InteractiveShell | None:
    if (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        return cast("InteractiveShell", shell)
    return None


# -------------------- Provider selection --------------------

# list of available GUI frontends and canvas backends, tried in order

GUI_PROVIDERS: dict[GuiFrontend, GuiProvider] = {
    GuiFrontend.QT: QtProvider,
    GuiFrontend.WX: WxProvider,
    GuiFrontend.JUPYTER: JupyterProvider,
}
CANVAS_PROVIDERS: dict[CanvasBackend, CanvasProvider] = {
    CanvasBackend.VISPY: VispyProvider,
    CanvasBackend.PYGFX: PygfxProvider,
}


@cache  # not allowed to change
def gui_frontend() -> GuiFrontend:
    """Return the preferred GUI frontend.

    This is determined first by the NDV_GUI_FRONTEND environment variable, after which
    GUI_PROVIDERS are tried in order until one is found that is either already running,
    or available.
    """
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    if requested:
        if requested not in (valid := {x.value for x in GuiFrontend}):
            raise ValueError(
                f"Invalid GUI frontend: {requested!r}. Valid options: {valid}"
            )
        key = GuiFrontend(requested)
        # ensure the app is created for explicitly requested frontends
        provider = GUI_PROVIDERS[key]
        if not provider.is_running():
            provider.create_app()
        return key

    for key, provider in GUI_PROVIDERS.items():
        if provider.is_running():
            return key

    errors: list[tuple[GuiFrontend, BaseException]] = []
    for key, provider in GUI_PROVIDERS.items():
        try:
            provider.create_app()
            return key
        except Exception as e:
            errors.append((key, e))

    raise RuntimeError(  # pragma: no cover
        f"Could not find an appropriate GUI frontend: {valid!r}. Tried:\n\n"
        + "\n".join(f"- {key.value}: {err}" for key, err in errors)
    )


def canvas_backend(requested: str | None) -> CanvasBackend:
    """Return the preferred canvas backend.

    This is determined first by the NDV_CANVAS_BACKEND environment variable, after which
    CANVAS_PROVIDERS are tried in order until one is found that is either already
    imported or available
    """
    backend = requested or os.getenv(CANVAS_ENV_VAR, "").lower()

    if backend:
        if backend not in (valid := {x.value for x in CanvasBackend}):
            raise ValueError(
                f"Invalid canvas backend: {backend!r}. Valid options: {valid}"
            )
        return CanvasBackend(backend)

    for key, provider in CANVAS_PROVIDERS.items():
        if provider.is_imported():
            return key
    errors: list[tuple[CanvasBackend, BaseException]] = []
    for key, provider in CANVAS_PROVIDERS.items():
        try:
            if provider.is_available():
                return key
        except Exception as e:
            errors.append((key, e))

    raise RuntimeError(  # pragma: no cover
        f"Could not find an appropriate canvas backend: {valid!r}. Tried:\n\n"
        + "\n".join(f"- {key.value}: {err}" for key, err in errors)
    )


# TODO: add a way to set the frontend via an environment variable
# (for example, it should be possible to use qt frontend in a jupyter notebook)
def get_array_view_class() -> type[ArrayView]:
    if (frontend := gui_frontend()) not in GUI_PROVIDERS:  # pragma: no cover
        raise NotImplementedError(f"No GUI frontend found for {frontend}")
    return GUI_PROVIDERS[frontend].array_view_class()


def get_array_canvas_class(backend: str | None = None) -> type[ArrayCanvas]:
    _backend = canvas_backend(backend)
    if _backend not in CANVAS_PROVIDERS:  # pragma: no cover
        raise NotImplementedError(f"No canvas backend found for {_backend}")
    return CANVAS_PROVIDERS[_backend].array_canvas_class()


def get_histogram_canvas_class(backend: str | None = None) -> type[HistogramCanvas]:
    _backend = canvas_backend(backend)
    if _backend not in CANVAS_PROVIDERS:  # pragma: no cover
        raise NotImplementedError(f"No canvas backend found for {_backend}")
    return CANVAS_PROVIDERS[_backend].histogram_canvas_class()


def run_app() -> None:
    """Start the GUI application event loop."""
    GUI_PROVIDERS[gui_frontend()].exec()


# -------------------- Exception handling --------------------


def _install_excepthook() -> None:
    """Install a custom excepthook that does not raise sys.exit().

    This is necessary to prevent the application from closing when an exception
    is raised.
    """
    if hasattr(sys, "_original_excepthook_"):
        # don't install the excepthook more than once
        return
    sys._original_excepthook_ = sys.excepthook  # type: ignore
    sys.excepthook = ndv_excepthook


def _print_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    try:
        import psygnal
        from rich.console import Console
        from rich.traceback import Traceback

        tb = Traceback.from_exception(
            exc_type, exc_value, exc_traceback, suppress=[psygnal], max_frames=10
        )
        Console(stderr=True).print(tb)
    except ImportError:
        traceback.print_exception(exc_type, value=exc_value, tb=exc_traceback)


def ndv_excepthook(
    exc_type: type[BaseException], exc_value: BaseException, tb: TracebackType | None
) -> None:
    _print_exception(exc_type, exc_value, tb)
    if not tb:
        return

    if (
        (debugpy := sys.modules.get("debugpy"))
        and debugpy.is_client_connected()
        and ("pydevd" in sys.modules)
    ):
        with suppress(Exception):
            import threading

            import pydevd

            py_db = pydevd.get_global_debugger()
            thread = threading.current_thread()
            additional_info = py_db.set_additional_thread_info(thread)
            additional_info.is_tracing += 1

            try:
                arg = (exc_type, exc_value, tb)
                py_db.stop_on_unhandled_exception(py_db, thread, additional_info, arg)
            finally:
                additional_info.is_tracing -= 1
    elif os.getenv(DEBUG_EXCEPTIONS):
        # Default to pdb if no better option is available
        import pdb

        pdb.post_mortem(tb)

    if os.getenv(EXIT_ON_EXCEPTION):
        print(f"\n{EXIT_ON_EXCEPTION} is set, exiting.")
        sys.exit(1)
