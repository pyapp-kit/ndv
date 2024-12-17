from contextlib import suppress
from typing import Any, Callable, overload

import wx
import wx.lib.newevent


class WxSignalInstance:
    def __init__(
        self,
        instance: wx.EvtHandler,
        binder: wx.PyEventBinder,
        event: wx.PyEvent,
        name: str,
    ) -> None:
        self.instance = instance
        self.binder = binder
        self.event = event
        self.name = name
        self._is_blocked = False

    def connect(self, callback: Callable) -> None:
        def _call_with_event_value(event: wx.Event) -> None:
            callback(*event.value)

        self.instance.Bind(self.binder, _call_with_event_value)

    def emit(self, *args: Any) -> None:
        if not self._is_blocked:
            evt = self.event(value=args)
            wx.PostEvent(self.instance, evt)

    def blocked(self) -> "BlockerContext":
        return BlockerContext(self)


class BlockerContext:
    def __init__(self, signal: WxSignalInstance) -> None:
        self.signal = signal

    def __enter__(self) -> WxSignalInstance:
        self.signal._is_blocked = True
        return self.signal

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        wx.Yield()  # Forces the event loop to process pending events
        self.signal._is_blocked = False


class WxSignal:
    event: wx.PyEvent
    binder: wx.PyEventBinder
    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.event, self.binder = wx.lib.newevent.NewEvent()

    @overload
    def __get__(self, obj: None, owner: type) -> "WxSignal": ...
    @overload
    def __get__(self, obj: wx.EvtHandler, owner: type) -> "WxSignalInstance": ...
    def __get__(
        self, obj: wx.EvtHandler | None, owner: type
    ) -> "WxSignalInstance | WxSignal":
        if obj is None:
            return self

        if not hasattr(self, "binder"):
            raise RuntimeError("This Signal has not been assigned to a class instance.")

        sig = WxSignalInstance(obj, self.binder, self.event, self.name)
        with suppress(AttributeError):
            setattr(obj, self.name, sig)  # try to cache it on the instance
        return sig
