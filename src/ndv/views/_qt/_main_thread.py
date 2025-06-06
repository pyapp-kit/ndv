from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable

from qtpy.QtCore import QCoreApplication, QMetaObject, QObject, Qt, QThread, Slot

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, TypeVar

    T = TypeVar("T")
    P = ParamSpec("P")


class MainThreadInvoker(QObject):
    _current_callable: Callable | None = None
    _moved: bool = False

    def invoke(
        self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """Invokes a function in the main thread and returns a Future."""
        future: Future[T] = Future()

        def wrapper() -> None:
            try:
                result = func(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        self._current_callable = wrapper
        QMetaObject.invokeMethod(
            self, "_invoke_current", Qt.ConnectionType.QueuedConnection
        )
        return future

    @Slot()  # type: ignore [misc]
    def _invoke_current(self) -> None:
        """Invokes the current callable."""
        if (cb := self._current_callable) is not None:
            cb()
            _INVOKERS.discard(self)


if (QAPP := QCoreApplication.instance()) is None:
    raise RuntimeError("QApplication must be created before this module is imported.")

_APP_THREAD = QAPP.thread()

_INVOKERS = set()


def call_in_main_thread(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Future[T]:
    if QThread.currentThread() is not _APP_THREAD:
        invoker = MainThreadInvoker()
        invoker.moveToThread(_APP_THREAD)
        _INVOKERS.add(invoker)
        return invoker.invoke(func, *args, **kwargs)

    future: Future[T] = Future()
    future.set_result(func(*args, **kwargs))
    return future
