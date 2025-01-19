from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import suppress
from enum import Enum
from functools import cache, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol, cast

from ndv._types import MouseMoveEvent, MousePressEvent, MouseReleaseEvent

if TYPE_CHECKING:
    from collections.abc import Container
    from types import TracebackType

    from IPython.core.interactiveshell import InteractiveShell
    from typing_extensions import ParamSpec, TypeVar

    from ndv.views.bases import ArrayCanvas, ArrayView, HistogramCanvas
    from ndv.views.bases._graphics._mouseable import Mouseable

    T = TypeVar("T")
    P = ParamSpec("P")

GUI_ENV_VAR = "NDV_GUI_FRONTEND"
"""Preferred GUI frontend. If not set, the first available GUI frontend is used."""

CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
"""Preferred canvas backend. If not set, the first available canvas backend is used."""

DEBUG_EXCEPTIONS = "NDV_DEBUG_EXCEPTIONS"
"""Whether to drop into a debugger when an exception is raised. Default False."""

EXIT_ON_EXCEPTION = "NDV_EXIT_ON_EXCEPTION"
"""Whether to exit the application when an exception is raised. Default False."""

IPYTHON_GUI_MAGIC = "NDV_IPYTHON_MAGIC"
"""Whether to use %gui magic when running in IPython. Default True."""


class GuiFrontend(str, Enum):
    """Enum of available GUI frontends.

    Attributes
    ----------
    QT : str
        [PyQt5/PySide2/PyQt6/PySide6](https://doc.qt.io)
    JUPYTER : str
        [Jupyter notebook/lab](https://jupyter.org)
    WX : str
        [wxPython](https://wxpython.org)
    """

    QT = "qt"
    JUPYTER = "jupyter"
    WX = "wx"


class CanvasBackend(str, Enum):
    """Enum of available canvas backends.

    Attributes
    ----------
    VISPY : str
        [Vispy](https://vispy.org)
    PYGFX : str
        [Pygfx](https://github.com/pygfx/pygfx)
    """

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
    @staticmethod
    def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]: ...
    @staticmethod
    def call_in_main_thread(
        func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future: Future[T] = Future()
        future.set_result(func(*args, **kwargs))
        return future

    @staticmethod
    def get_executor() -> Executor:
        return _thread_pool_executor()


@cache
def _thread_pool_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


class CanvasProvider(Protocol):
    @staticmethod
    def is_imported() -> bool: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def array_canvas_class() -> type[ArrayCanvas]: ...
    @staticmethod
    def histogram_canvas_class() -> type[HistogramCanvas]: ...


# FIXME:
# the implementation below has a lot of nested imports that will be largely
# unnecessary after the GUI has been decided.  Consider alternative patterns.
# primarily, we need to avoid importing any frontends "accidentally".  But beyond
# that, it can be refactored as needed.


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
            # start the %gui qt magic if NDV_IPYTHON_MAGIC!=0
            if (ipy_shell := _ipython_shell()) and (
                os.getenv(IPYTHON_GUI_MAGIC, "true").lower() not in ("0", "false", "no")
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
    def call_in_main_thread(
        func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        from ._qt._main_thread import call_in_main_thread

        return call_in_main_thread(func, *args, **kwargs)

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._qt._array_view import QtArrayView

        return QtArrayView

    @staticmethod
    def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]:
        from qtpy.QtCore import QEvent, QObject
        from qtpy.QtGui import QMouseEvent

        if not isinstance(canvas, QObject):
            raise TypeError(f"Expected canvas to be QObject, got {type(canvas)}")

        class Filter(QObject):
            def eventFilter(self, obj: QObject | None, qevent: QEvent | None) -> bool:
                """Event filter installed on the canvas to handle mouse events.

                here is where we get a chance to intercept mouse events before allowing
                the canvas to respond to them. Return `True` to prevent the event from
                being passed to the canvas.
                """
                if qevent is None:
                    return False  # pragma: no cover

                try:
                    # use children in case backend has a subwidget stealing events.
                    children: Container = canvas.children()
                except RuntimeError:
                    # native is likely dead
                    return False

                intercept = False
                if obj is canvas or obj in children:
                    if isinstance(qevent, QMouseEvent):
                        pos = qevent.pos()
                        etype = qevent.type()
                        if etype == QEvent.Type.MouseMove:
                            mme = MouseMoveEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_move(mme)
                            receiver.mouseMoved.emit(mme)
                        elif etype == QEvent.Type.MouseButtonPress:
                            mpe = MousePressEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_press(mpe)
                            receiver.mousePressed.emit(mpe)
                        elif etype == QEvent.Type.MouseButtonRelease:
                            mre = MouseReleaseEvent(x=pos.x(), y=pos.y())
                            intercept |= receiver.on_mouse_release(mre)
                            receiver.mouseReleased.emit(mre)
                return intercept

        f = Filter()
        canvas.installEventFilter(f)
        return lambda: canvas.removeEventFilter(f)


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
        # if we're running in IPython
        # start the %gui qt magic if NDV_IPYTHON_MAGIC!=0
        if (ipy_shell := _ipython_shell()) and (
            os.getenv(IPYTHON_GUI_MAGIC, "true").lower() not in ("0", "false", "no")
        ):
            ipy_shell.enable_gui("wx")  # type: ignore [no-untyped-call]

        _install_excepthook()
        from ._wx._main_thread import call_in_main_thread  # noqa: F401

        return wxapp

    @staticmethod
    def exec() -> None:
        import wx

        app = cast("wx.App", wx.App.Get() or WxProvider.create_app())

        if ipy_shell := _ipython_shell():
            # if we're already in an IPython session with %gui qt, don't block
            if str(ipy_shell.active_eventloop).startswith("wx"):
                return

        app.MainLoop()

    @staticmethod
    def call_in_main_thread(
        func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        from ._wx._main_thread import call_in_main_thread

        return call_in_main_thread(func, *args, **kwargs)

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._wx._array_view import WxArrayView

        return WxArrayView

    @staticmethod
    def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]:
        from wx import EVT_LEFT_DOWN, EVT_LEFT_UP, EVT_MOTION, EvtHandler, MouseEvent

        if not isinstance(canvas, EvtHandler):
            raise TypeError(
                f"Expected vispy canvas to be wx EvtHandler, got {type(canvas)}"
            )

        # TIP: event.Skip() allows the event to propagate to other handlers.

        def on_mouse_move(event: MouseEvent) -> None:
            mme = MouseMoveEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_move(mme):
                receiver.mouseMoved.emit(mme)
                event.Skip()

        def on_mouse_press(event: MouseEvent) -> None:
            mpe = MousePressEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_press(mpe):
                receiver.mousePressed.emit(mpe)
                event.Skip()

        def on_mouse_release(event: MouseEvent) -> None:
            mre = MouseReleaseEvent(x=event.GetX(), y=event.GetY())
            if not receiver.on_mouse_release(mre):
                receiver.mouseReleased.emit(mre)
                event.Skip()

        canvas.Bind(EVT_MOTION, on_mouse_move)
        canvas.Bind(EVT_LEFT_DOWN, on_mouse_press)
        canvas.Bind(EVT_LEFT_UP, on_mouse_release)

        def _unbind() -> None:
            canvas.Unbind(EVT_MOTION, on_mouse_move)
            canvas.Unbind(EVT_LEFT_DOWN, on_mouse_press)
            canvas.Unbind(EVT_LEFT_UP, on_mouse_release)

        return _unbind


class JupyterProvider(GuiProvider):
    """Provider for Jupyter notebooks/lab (NOT ipython)."""

    @staticmethod
    def is_running() -> bool:
        if ipy_shell := _ipython_shell():
            return bool(ipy_shell.__class__.__name__ == "ZMQInteractiveShell")
        return False

    @staticmethod
    def create_app() -> Any:
        if not JupyterProvider.is_running() and not os.getenv("PYTEST_CURRENT_TEST"):
            # if we got here, it probably means that someone used
            # NDV_GUI_FRONTEND=jupyter without actually being in a juptyer notebook
            # we allow it in tests, but not in normal usage.
            raise RuntimeError(  # pragma: no cover
                "Jupyter is not running a notebook shell.  Cannot create app."
            )
        # No app creation needed...
        # but make sure we can actually import the stuff we need
        import ipywidgets  # noqa: F401
        import jupyter  # noqa: F401
        import jupyter_rfb  # noqa: F401

    @staticmethod
    def exec() -> None:
        pass

    @staticmethod
    def array_view_class() -> type[ArrayView]:
        from ._jupyter._array_view import JupyterArrayView

        return JupyterArrayView

    @staticmethod
    def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]:
        from jupyter_rfb import RemoteFrameBuffer

        if not isinstance(canvas, RemoteFrameBuffer):
            raise TypeError(
                f"Expected canvas to be RemoteFrameBuffer, got {type(canvas)}"
            )

        # patch the handle_event from _jupyter_rfb.CanvasBackend
        # to intercept various mouse events.
        super_handle_event = canvas.handle_event

        def handle_event(self: RemoteFrameBuffer, ev: dict) -> None:
            etype = ev["event_type"]
            if etype == "pointer_move":
                mme = MouseMoveEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_move(mme)
                receiver.mouseMoved.emit(mme)
            elif etype == "pointer_down":
                mpe = MousePressEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_press(mpe)
                receiver.mousePressed.emit(mpe)
            elif etype == "pointer_up":
                mre = MouseReleaseEvent(x=ev["x"], y=ev["y"])
                receiver.on_mouse_release(mre)
                receiver.mouseReleased.emit(mre)
            super_handle_event(ev)

        canvas.handle_event = MethodType(handle_event, canvas)
        return lambda: setattr(canvas, "handle_event", super_handle_event)


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

        from ndv.views._vispy._array_canvas import VispyArrayCanvas

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

        return VispyArrayCanvas

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
        from ndv.views._pygfx._array_canvas import GfxArrayCanvas

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
    """Return the active [`GuiFrontend`][ndv.views.GuiFrontend].

    This is determined first by the `NDV_GUI_FRONTEND` environment variable, after which
    known GUI providers are tried in order until one is found that is either already
    running, or available.
    """
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = {x.value for x in GuiFrontend}
    if requested:
        if requested not in valid:
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

    valid = {x.value for x in CanvasBackend}
    if backend:
        if backend not in valid:
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
    """Return [`ArrayView`][ndv.views.bases.ArrayView] class for current GUI frontend."""  # noqa: E501
    return GUI_PROVIDERS[gui_frontend()].array_view_class()


def get_array_canvas_class(backend: str | None = None) -> type[ArrayCanvas]:
    """Return [`ArrayCanvas`][ndv.views.bases.ArrayCanvas] class for current canvas backend."""  # noqa: E501
    _backend = canvas_backend(backend)
    if _backend not in CANVAS_PROVIDERS:  # pragma: no cover
        raise NotImplementedError(f"No canvas backend found for {_backend}")
    return CANVAS_PROVIDERS[_backend].array_canvas_class()


def get_histogram_canvas_class(backend: str | None = None) -> type[HistogramCanvas]:
    """Return [`HistogramCanvas`][ndv.views.bases.HistogramCanvas] class for current canvas backend."""  # noqa: E501
    _backend = canvas_backend(backend)
    if _backend not in CANVAS_PROVIDERS:  # pragma: no cover
        raise NotImplementedError(f"No canvas backend found for {_backend}")
    return CANVAS_PROVIDERS[_backend].histogram_canvas_class()


def filter_mouse_events(canvas: Any, receiver: Mouseable) -> Callable[[], None]:
    """Intercept mouse events on `scene_canvas` and forward them to `receiver`.

    Parameters
    ----------
    canvas : Any
        The front-end canvas widget to intercept mouse events from.
    receiver : Mouseable
        The object to forward mouse events to.

    Returns
    -------
    Callable[[], None]
        A function that can be called to remove the event filter.
    """
    return GUI_PROVIDERS[gui_frontend()].filter_mouse_events(canvas, receiver)


def run_app() -> None:
    """Start the active GUI application event loop."""
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
    elif os.getenv(DEBUG_EXCEPTIONS) in ("1", "true", "True"):
        # Default to pdb if no better option is available
        import pdb

        pdb.post_mortem(tb)

    if os.getenv(EXIT_ON_EXCEPTION) in ("1", "true", "True"):
        print(f"\n{EXIT_ON_EXCEPTION} is set, exiting.")
        sys.exit(1)


def ensure_main_thread(func: Callable[P, T]) -> Callable[P, Future[T]]:
    """Decorator that ensures a function is called in the main thread."""

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        fn = GUI_PROVIDERS[gui_frontend()].call_in_main_thread
        return fn(func, *args, **kwargs)

    return _wrapper


def submit_task(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:
    """Submit a task to the GUI event loop.

    Returns a `concurrent.futures.Future` or an `asyncio.Future` depending on the
    GUI frontend.
    """
    executor = GUI_PROVIDERS[gui_frontend()].get_executor()
    return executor.submit(func, *args, **kwargs)
