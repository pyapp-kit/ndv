from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    SupportsIndex,
    TypeVar,
    get_args,
    overload,
)

from psygnal import Signal
from pydantic import (
    TypeAdapter,
)
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

_T = TypeVar("_T")


class ValidatedEventedList(MutableSequence[_T]):
    item_inserted = Signal(int, object)  # (idx, value)
    item_removed = Signal(int, object)  # (idx, value)
    item_changed = Signal(object, object, object)  # (int | slice, new, old)
    items_reordered = Signal()

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self,
        iterable: Iterable[_T],
        *,
        _item_adaptor: TypeAdapter | None = ...,
    ) -> None: ...
    def __init__(
        self,
        iterable: Iterable[_T] = (),
        *,
        _item_adaptor: TypeAdapter | None = None,
    ) -> None:
        self._item_adaptor = _item_adaptor
        if self._item_adaptor is not None:
            iterable = (self._item_adaptor.validate_python(v) for v in iterable)
        self._list = list(iterable)

    # ---------------- abstract interface ----------------

    @overload
    def __getitem__(self, i: SupportsIndex) -> _T: ...
    @overload
    def __getitem__(self, i: slice) -> list[_T]: ...
    def __getitem__(self, i: SupportsIndex | slice) -> _T | list[_T]:
        return self._list[i]

    @overload
    def __setitem__(self, key: SupportsIndex, value: _T) -> None: ...
    @overload
    def __setitem__(self, key: slice, value: Iterable[_T]) -> None: ...
    def __setitem__(self, key: slice | SupportsIndex, value: _T | Iterable[_T]) -> None:
        if isinstance(value, Iterable):
            value = (self._validate_item(v) for v in value)
        else:
            value = self._validate_item(value)

        # no-op if value is identical
        old = self._list[key]
        if value is old:
            return

        self._list[key] = value  # type: ignore [index,assignment]
        self.item_changed.emit(key, value, old)

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        item = self._list[key]
        del self._list[key]
        self.item_removed.emit(key, item)

    def insert(self, index: SupportsIndex, obj: _T) -> None:
        obj = self._validate_item(obj)
        self._list.insert(index, obj)
        self.item_inserted.emit(index, obj)

    def __len__(self) -> int:
        return len(self._list)

    def __eq__(self, value: object) -> bool:
        return self._list == value

    # -----------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._list)
        # return f"{type(self).__name__}({self._list!r})"

    @cached_property
    def _validate_item(self) -> Callable[[Any], _T]:
        if self._item_adaptor is None:
            # __orig_class__ is not available during __init__
            # https://discuss.python.org/t/runtime-access-to-type-parameters/37517
            cls = getattr(self, "__orig_class__", None) or type(self)
            if args := get_args(cls):
                self._item_adaptor = TypeAdapter(args[0])

        if self._item_adaptor is not None:
            return self._item_adaptor.validate_python

        return lambda x: x

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> Mapping[str, Any]:
        """Return the Pydantic core schema for this object."""
        item_type = args[0] if (args := get_args(source_type)) else Any

        def _validate(obj: Any, _item_type: Any = item_type) -> Any:
            # delayed instantiation of TypeAdapter to allow recursive models
            # time to rebuild
            adapter = TypeAdapter(_item_type)
            return cls(obj, _item_adaptor=adapter)

        def _serialize(obj: ValidatedEventedList[_T]) -> Any:
            return obj._list

        items_schema = handler.generate_schema(item_type)
        list_schema = core_schema.list_schema(items_schema=items_schema)
        return core_schema.no_info_plain_validator_function(
            function=_validate,
            json_schema_input_schema=list_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                return_schema=list_schema,
            ),
        )
