from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import suppress
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, cast

if TYPE_CHECKING:
    from types import TracebackType

    from IPython.core.interactiveshell import InteractiveShell
    from typing_extensions import ParamSpec, TypeVar

    from ndv.views.bases import ArrayView
    from ndv.views.bases._graphics._mouseable import Mouseable

    T = TypeVar("T")
    P = ParamSpec("P")

ENV_IPYTHON_GUI_MAGIC = os.getenv("NDV_IPYTHON_MAGIC", "true").lower()
"""Whether to use %gui magic when running in IPython. Default True."""


class NDVApp:
    """Base class for application wrappers."""

    USE_IPY_MAGIC = ENV_IPYTHON_GUI_MAGIC not in ("0", "false", "no")
    # must be valid key for %gui <magic> in IPython
    IPY_MAGIC_KEY: ClassVar[Literal["qt", "wx", None]] = None

    def create_app(self) -> Any:
        """Create the application instance, if not already created."""
        raise NotImplementedError

    def array_view_class(self) -> type[ArrayView]:
        raise NotImplementedError

    def run(self) -> None:
        """Run the application."""
        pass

    def filter_mouse_events(
        self, canvas: Any, receiver: Mouseable
    ) -> Callable[[], None]:
        """Install mouse event filter on `canvas`, redirecting events to `receiver`."""
        raise NotImplementedError

    def call_in_main_thread(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Call `func` in the main gui thread."""
        future: Future[T] = Future()
        future.set_result(func(*args, **kwargs))
        return future

    def get_executor(self) -> Executor:
        """Return an executor for running tasks in the background."""
        return _thread_pool_executor()

    @staticmethod
    def _ipython_shell() -> InteractiveShell | None:
        if (ipy := sys.modules.get("IPython")) and (shell := ipy.get_ipython()):
            return cast("InteractiveShell", shell)
        return None

    def _maybe_enable_ipython_gui(self) -> bool:
        if (
            (ipy_shell := self._ipython_shell())
            and self.USE_IPY_MAGIC
            and (key := self.IPY_MAGIC_KEY)
        ):
            ipy_shell.enable_gui(key)  # type: ignore [no-untyped-call]
            return True
        return False

    @staticmethod
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

    def process_events(self) -> None:
        """Process events for the application."""
        pass

    def call_later(self, msec: int, func: Callable[[], None]) -> None:
        """Call `func` after `msec` milliseconds."""
        # generic implementation using python threading

        from threading import Timer

        Timer(msec / 1000, func).start()


@cache
def _thread_pool_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)


# -------------------- Exception handling --------------------
DEBUG_EXCEPTIONS = "NDV_DEBUG_EXCEPTIONS"
"""Whether to drop into a debugger when an exception is raised. Default False."""

EXIT_ON_EXCEPTION = "NDV_EXIT_ON_EXCEPTION"
"""Whether to exit the application when an exception is raised. Default False."""


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
