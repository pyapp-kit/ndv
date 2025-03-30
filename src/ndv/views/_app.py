from __future__ import annotations

import importlib.util
import os
import sys
from enum import Enum
from functools import cache, wraps
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

    from IPython.core.interactiveshell import InteractiveShell
    from typing_extensions import ParamSpec, TypeVar

    from ndv.views.bases import ArrayCanvas, ArrayView, HistogramCanvas
    from ndv.views.bases._app import NDVApp
    from ndv.views.bases._graphics._mouseable import Mouseable

    T = TypeVar("T")
    P = ParamSpec("P")

GUI_ENV_VAR = "NDV_GUI_FRONTEND"
"""Preferred GUI frontend. If not set, the first available GUI frontend is used."""

CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
"""Preferred canvas backend. If not set, the first available canvas backend is used."""


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


class CanvasProvider(Protocol):
    @staticmethod
    def is_imported() -> bool: ...
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def array_canvas_class() -> type[ArrayCanvas]: ...
    @staticmethod
    def histogram_canvas_class() -> type[HistogramCanvas]: ...


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
        elif _frontend == GuiFrontend.QT:
            from qtpy import API_NAME

            use_app(API_NAME.lower())

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
        from ndv.views._pygfx._histogram import PyGFXHistogramCanvas

        return PyGFXHistogramCanvas


# -------------------- Provider selection --------------------

# list of available GUI frontends and canvas backends, tried in order

GUI_PROVIDERS: dict[GuiFrontend, tuple[str, str]] = {
    GuiFrontend.QT: ("ndv.views._qt._app", "QtAppWrap"),
    GuiFrontend.WX: ("ndv.views._wx._app", "WxAppWrap"),
    GuiFrontend.JUPYTER: ("ndv.views._jupyter._app", "JupyterAppWrap"),
}
MOD_TO_KEY = {mod: key for key, (mod, _) in GUI_PROVIDERS.items()}
CANVAS_PROVIDERS: dict[CanvasBackend, CanvasProvider] = {
    CanvasBackend.VISPY: VispyProvider,
    CanvasBackend.PYGFX: PygfxProvider,
}


def _running_apps() -> Iterator[GuiFrontend]:
    """Return an iterator of running GUI applications."""
    for mod_name in ("PyQt5", "PySide2", "PySide6", "PyQt6"):
        if mod := sys.modules.get(f"{mod_name}.QtWidgets"):
            if (
                qapp := getattr(mod, "QApplication", None)
            ) and qapp.instance() is not None:
                yield GuiFrontend.QT

    if (wx := sys.modules.get("wx")) and wx.App.Get() is not None:
        yield GuiFrontend.WX

    if (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
        shell = cast("InteractiveShell", shell)
        if shell.__class__.__name__ == "ZMQInteractiveShell":
            yield GuiFrontend.JUPYTER


def _load_app(module: str, cls_name: str) -> NDVApp:
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    return cast("NDVApp", cls())


@cache  # not allowed to change
def ndv_app() -> NDVApp:
    """Return the active [`GuiFrontend`][ndv.views.GuiFrontend].

    This is determined first by the `NDV_GUI_FRONTEND` environment variable, after which
    known GUI providers are tried in order until one is found that is either already
    running, or available.
    """
    running = list(_running_apps())

    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = {x.value for x in GuiFrontend}
    if requested:
        if requested not in valid:
            raise ValueError(
                f"Invalid GUI frontend: {requested!r}. Valid options: {valid}"
            )
        # ensure the app is created for explicitly requested frontends
        app = _load_app(*GUI_PROVIDERS[GuiFrontend(requested)])
        app.create_app()
        return app

    for key, provider in GUI_PROVIDERS.items():
        if key in running:
            app = _load_app(*provider)
            app.create_app()
            return app

    errors: list[tuple[str, BaseException]] = []
    for key, provider in GUI_PROVIDERS.items():
        try:
            app = _load_app(*provider)
            app.create_app()
            return app
        except Exception as e:
            errors.append((key, e))

    raise RuntimeError(  # pragma: no cover
        f"Could not find an appropriate GUI frontend: {valid!r}. Tried:\n\n"
        + "\n".join(f"- {key}: {err}" for key, err in errors)
    )


def gui_frontend() -> GuiFrontend:
    # possibly temporary hack to get the current frontend as an enum, for back-compat
    return MOD_TO_KEY[ndv_app().__module__]


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


def get_array_view_class() -> type[ArrayView]:
    """Return [`ArrayView`][ndv.views.bases.ArrayView] class for current GUI frontend."""  # noqa: E501
    return ndv_app().array_view_class()


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
    return ndv_app().filter_mouse_events(canvas, receiver)


def call_later(msec: int, func: Callable[[], None]) -> None:
    """Call `func` after `msec` milliseconds.

    This can be used to enqueue a function to be called after the current event loop
    iteration.  For example, before calling `run_app()`, to ensure that the event
    loop is running before the function is called.

    Parameters
    ----------
    msec : int
        The number of milliseconds to wait before calling `func`.
    func : Callable[[], None]
        The function to call.
    """
    ndv_app().call_later(msec, func)


def run_app() -> None:
    """Start the active GUI application event loop."""
    ndv_app().run()


def ensure_main_thread(func: Callable[P, T]) -> Callable[P, Future[T]]:
    """Decorator that ensures a function is called in the main thread."""

    @wraps(func)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        fn = ndv_app().call_in_main_thread
        return fn(func, *args, **kwargs)

    return _wrapper


def submit_task(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T]:
    """Submit a task to the GUI event loop.

    Returns a `concurrent.futures.Future` or an `asyncio.Future` depending on the
    GUI frontend.
    """
    executor = ndv_app().get_executor()
    return executor.submit(func, *args, **kwargs)
