from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Callable

import wx

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, TypeVar

    T = TypeVar("T")
    P = ParamSpec("P")


class MainThreadInvoker:
    def __init__(self) -> None:
        """Utility for invoking functions in the main thread."""
        # Ensure this is initialized from the main thread
        if not wx.IsMainThread():
            raise RuntimeError(
                "MainThreadInvoker must be initialized in the main thread"
            )

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

        wx.CallAfter(wrapper)
        return future


_MAIN_THREAD_INVOKER = MainThreadInvoker()


def call_in_main_thread(
    func: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Future[T]:
    if not wx.IsMainThread():
        return _MAIN_THREAD_INVOKER.invoke(func, *args, **kwargs)

    future: Future[T] = Future()
    future.set_result(func(*args, **kwargs))
    return future
