from __future__ import annotations

import importlib.util
import os
import sys
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, cast

from scenex.adaptors import use
from scenex.app import app

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from concurrent.futures import Future
    from typing import Literal

    from IPython.core.interactiveshell import InteractiveShell
    from typing_extensions import ParamSpec, TypeVar

    from ndv.views.bases import ArrayView
    from ndv.views.bases._app import NDVApp

    T = TypeVar("T")
    P = ParamSpec("P")

GUI_ENV_VAR = "NDV_GUI_FRONTEND"
"""Preferred GUI frontend. If not set, the first available GUI frontend is used."""

CANVAS_ENV_VAR = "NDV_CANVAS_BACKEND"
"""Preferred canvas backend. If not set, the first available canvas backend is used."""

_APP: NDVApp | None = None
"""Singleton instance of the current (GUI) application. Once set it shouldn't change."""


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


# -------------------- Provider selection --------------------

# list of available GUI frontends and canvas backends, tried in order

GUI_PROVIDERS: dict[GuiFrontend, tuple[str, str]] = {
    GuiFrontend.QT: ("ndv.views._qt._app", "QtAppWrap"),
    GuiFrontend.WX: ("ndv.views._wx._app", "WxAppWrap"),
    GuiFrontend.JUPYTER: ("ndv.views._jupyter._app", "JupyterAppWrap"),
}
MOD_TO_KEY = {mod: key for key, (mod, _) in GUI_PROVIDERS.items()}


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


def ndv_app() -> NDVApp:
    """Return the active [`GuiFrontend`][ndv.views.GuiFrontend].

    This is determined first by the `NDV_GUI_FRONTEND` environment variable, after which
    known GUI providers are tried in order until one is found that is either already
    running, or available.
    """
    global _APP
    if _APP is not None:
        return _APP

    running = list(_running_apps())

    # Try 1: Load a frontend explicitly requested by the user
    requested = os.getenv(GUI_ENV_VAR, "").lower()
    valid = {x.value for x in GuiFrontend}
    if requested:
        if requested not in valid:
            raise ValueError(
                f"Invalid GUI frontend: {requested!r}. Valid options: {valid}"
            )
        # ensure the app is created for explicitly requested frontends
        _APP = _load_app(*GUI_PROVIDERS[GuiFrontend(requested)])
        _APP.create_app()
        return _APP

    # Try 2: Utilize an existing, running app
    for key, provider in GUI_PROVIDERS.items():
        if key in running:
            _APP = _load_app(*provider)
            _APP.create_app()
            return _APP

    # Try 3: Load an existing app
    errors: list[tuple[str, BaseException]] = []
    for key, provider in GUI_PROVIDERS.items():
        try:
            _APP = _load_app(*provider)
            _APP.create_app()
            return _APP
        except Exception as e:
            errors.append((key, e))

    raise RuntimeError(  # pragma: no cover
        f"Could not find an appropriate GUI frontend: {valid!r}. Tried:\n\n"
        + "\n".join(f"- {key}: {err}" for key, err in errors)
    )


def set_canvas_backend(
    backend: Literal["vispy", "pygfx"] | None = None,
) -> None:
    """Sets the preferred canvas backend. Cannot be set after the GUI is running."""
    use(backend)


def set_gui_backend(backend: Literal["jupyter", "qt", "wx"] | None = None) -> None:
    """Sets the preferred GUI backend. Cannot be set after the GUI is running."""
    if _APP:
        raise RuntimeError("Cannot change the backend once the app is running")
    if backend is None:
        os.environ.pop(GUI_ENV_VAR)
    else:
        os.environ[GUI_ENV_VAR] = GuiFrontend(backend).value  # validate


def gui_frontend() -> GuiFrontend:
    # possibly temporary hack to get the current frontend as an enum, for back-compat
    return MOD_TO_KEY[ndv_app().__module__]


def get_array_view_class() -> type[ArrayView]:
    """Return [`ArrayView`][ndv.views.bases.ArrayView] class for current GUI frontend."""  # noqa: E501
    return ndv_app().array_view_class()


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
    app().call_later(msec, func)


def process_events() -> None:
    """Force processing of events for the application."""
    app().process_events()


def run_app() -> None:
    """Start the active GUI application event loop."""
    app().run()


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
