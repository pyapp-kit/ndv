from __future__ import annotations

from collections.abc import Iterable, Iterator, MutableMapping
from contextlib import suppress
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeVar,
    cast,
    get_args,
    overload,
)

from psygnal import Signal
from pydantic import TypeAdapter
from pydantic_core import SchemaValidator, core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)
_NULL = object()


class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]: ...
    def __getitem__(self, key: _KT, /) -> _VT_co: ...


class ValidatedEventedDict(MutableMapping[_KT, _VT]):
    item_added = Signal(str, object)  # key, new_value
    item_removed = Signal(str, object)  # key, old_value
    item_changed = Signal(str, object, object)  # key, new_value, old_value
    value_changed = Signal()

    # long ugly overloads to support all possible ways to initialize a ValidatedDict
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(  # type: ignore[misc]
        self: dict[str, _VT],
        key_validator: Callable[[Any], _KT] | None = None,
        value_validator: Callable[[Any], _VT] | None = None,
        **kwargs: _VT,
    ) -> None: ...
    @overload
    def __init__(
        self,
        map: SupportsKeysAndGetItem[_KT, _VT],
        /,
        key_validator: Callable[[Any], _KT] | None = None,
        value_validator: Callable[[Any], _VT] | None = None,
    ) -> None: ...
    @overload
    def __init__(  # type: ignore[misc]
        self: dict[str, _VT],
        map: SupportsKeysAndGetItem[str, _VT],
        /,
        key_validator: Callable[[Any], _KT] | None = ...,
        value_validator: Callable[[Any], _VT] | None = ...,
        validate_lookup: bool = ...,
        **kwargs: _VT,
    ) -> None: ...
    @overload
    def __init__(
        self,
        iterable: Iterable[tuple[_KT, _VT]],
        /,
        key_validator: Callable[[Any], _KT] | None = ...,
        value_validator: Callable[[Any], _VT] | None = ...,
        validate_lookup: bool = ...,
    ) -> None: ...
    @overload
    def __init__(  # type: ignore[misc]
        self: dict[str, _VT],
        iterable: Iterable[tuple[str, _VT]],
        /,
        key_validator: Callable[[Any], _KT] | None = ...,
        value_validator: Callable[[Any], _VT] | None = ...,
        validate_lookup: bool = ...,
        **kwargs: _VT,
    ) -> None: ...
    def __init__(  # type: ignore[misc] # does not accept all possible overloads
        self,
        *args: Any,
        key_validator: Callable[[Any], _KT] | None = None,
        value_validator: Callable[[Any], _VT] | None = None,
        validate_lookup: bool = False,
        **kwargs: Any,
    ) -> None:
        self._key_validator = key_validator
        self._value_validator = value_validator
        self._validate_lookup = validate_lookup
        _d = {}
        for k, v in dict(*args, **kwargs).items():
            if self._key_validator is not None:
                k = self._key_validator(k)
            if self._value_validator is not None:
                v = self._value_validator(v)
            _d[k] = v
        self._dict: dict[_KT, _VT] = _d

    def __missing__(self, key: _KT) -> _VT:
        raise KeyError(key)

    # ---------------- abstract interface ----------------

    def __getitem__(self, key: _KT) -> _VT:
        if self._validate_lookup:
            key = self._validate_key(key)
        try:
            return self._dict[key]
        except KeyError:
            return self.__missing__(key)

    # def __setitem__(self, key: _KT, value: _VT) -> None:
    # we allow Any here because validation may change the type of the value.
    def __setitem__(self, key: Any, value: Any) -> None:
        key = self._validate_key(key)
        value = self._validate_value(value)
        before = self._dict.get(key, _NULL)
        self._dict[key] = value
        # if the value is the same as before, try to exit early without emitting signals
        # but catch exceptions that may be raised during __eq__ (like numpy)
        if before is not _NULL:
            with suppress(Exception):
                if before == value:
                    return
            self.item_changed.emit(key, value, before)
        else:
            self.item_added.emit(key, value)
        self.value_changed.emit()

    def __delitem__(self, key: _KT) -> None:
        if self._validate_lookup:
            key = self._validate_key(key)
        # TODO: maybe add removing signal (before actual removal) if needed
        item = self._dict.pop(key)
        self.item_removed.emit(key, item)
        self.value_changed.emit()

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._dict)

    # batch operations, with a single value_changed  -------------------------------

    @overload
    def assign(self, m: SupportsKeysAndGetItem[_KT, _VT], /, **kwargs: _VT) -> None: ...
    @overload
    def assign(self, m: Iterable[tuple[_KT, _VT]], /, **kwargs: _VT) -> None: ...
    @overload
    def assign(self, **kwargs: _VT) -> None: ...
    def assign(self, *args: Any, **kwargs: _VT) -> None:  # type: ignore[misc]
        """Override state with the key/value pairs from the mapping or iterable.

        Similar to update, but clears the dictionary first (without signals), replacing
        the contents with the key/value pairs from the mapping or iterable, and then
        emitting a single value_changed signal at the end.
        """
        with self.value_changed.blocked():
            self.clear()
            self.update(*args, **kwargs)
        self.value_changed.emit()

    @overload
    def update(self, m: SupportsKeysAndGetItem[_KT, _VT], /, **kwargs: _VT) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[_KT, _VT]], /, **kwargs: _VT) -> None: ...
    @overload
    def update(self, **kwargs: _VT) -> None: ...
    def update(self, *args: Any, **kwargs: _VT) -> None:  # type: ignore[misc]
        """Update the dictionary with the key/value pairs from the mapping or iterable.

        only emit a single value_changed signal at the end.
        """
        with self.value_changed.blocked():
            super().update(*args, **kwargs)
        # TODO: only emit if anything was caught
        self.value_changed.emit()

    def clear(self) -> None:
        """Clear the dictionary.

        only emit a single value_changed signal at the end.
        """
        with self.value_changed.blocked():
            super().clear()
        # TODO: only emit if anything was caught
        self.value_changed.emit()

    # -----------------------------------------------------

    @cached_property
    def _validate_key(self) -> Callable[[Any], _KT]:
        """Return a function that validates keys."""
        if self._key_validator is not None:
            return self._key_validator
        # No key validator was provided during init.  Try to get the key type from the
        # class type hint and return a validator function for it.
        # __orig_class__ is not available during __init__
        # https://discuss.python.org/t/runtime-access-to-type-parameters/37517
        cls = getattr(self, "__orig_class__", None) or type(self)
        if args := get_args(cls):
            return TypeAdapter(args[0]).validator.validate_python
        # fall back to identity function
        return lambda x: x

    @cached_property
    def _validate_value(self) -> Callable[[Any], _VT]:
        """Return a function that validates values."""
        if self._value_validator is not None:
            return self._value_validator
        # No value validator was provided during init.  Try to get the value type from
        # the class type hint and return a validator function for it.
        # __orig_class__ is not available during __init__
        # https://discuss.python.org/t/runtime-access-to-type-parameters/37517
        cls = getattr(self, "__orig_class__", None) or type(self)
        if len(args := get_args(cls)) > 1:
            return TypeAdapter(args[1]).validator.validate_python
        # fall back to identity function
        return lambda x: x

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._dict!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Return the Pydantic core schema for this object.

        In this method, we parse the key and value types of the `source_type`, which
        be something like ValidatedDict[KT, VT]. And then create a validator function
        that creates a new instance of the ValidatedDict during assignment, passing in
        the key and value validator functions (from pydantic).

        Parameters
        ----------
        source_type : Any
            The source type.  This will usually be `cls`.
        handler : GetCoreSchemaHandler
            Handler to call into the next CoreSchema schema generation function.
        """
        # get key/value types from the source_type type hint.
        key_type = val_type = Any
        if args := get_args(source_type):
            key_type = args[0]
            if len(args) > 1:
                val_type = args[1]

        # get key/value core schemas for the key/value types.
        keys_schema = _get_schema(key_type, handler)
        values_schema = _get_schema(val_type, handler)
        validate_key = SchemaValidator(keys_schema).validate_python
        validate_value = SchemaValidator(values_schema).validate_python

        # define function that creates new instance during assignment
        # passing in the validator functions.
        def _new(*args: Any, **kwargs: Any) -> ValidatedEventedDict[_KT, _VT]:
            return cls(  # type: ignore
                *args,
                key_validator=validate_key,
                value_validator=validate_value,
                **kwargs,
            )

        # this schema for this validated dict
        dict_schema = core_schema.dict_schema(
            keys_schema=keys_schema,
            values_schema=values_schema,
        )
        # wrap the schema with a validator function that creates a new instance,
        # passing in the key/value validators.
        return core_schema.no_info_after_validator_function(
            function=_new,
            schema=dict_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x._dict, return_schema=dict_schema
            ),
        )


def _get_schema(hint: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    # check if the hint already has a core schema attached to it.
    if hasattr(hint, "__pydantic_core_schema__"):
        return cast("core_schema.CoreSchema", hint.__pydantic_core_schema__)
    # otherwise, call the handler to get the core schema.
    return handler.generate_schema(hint)
