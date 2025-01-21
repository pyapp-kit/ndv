from __future__ import annotations

import contextlib
import logging
from abc import abstractmethod
from contextlib import suppress
from importlib import import_module
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
)

from psygnal import EmissionInfo, SignalGroupDescriptor
from pydantic import BaseModel, ConfigDict
from pydantic.fields import Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ["Field", "ModelBase", "SupportsVisibility", "VisModel"]

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)
SETTER_METHOD = "_vis_set_{name}"


class ModelBase(BaseModel):
    """Base class for all evented pydantic-style models."""

    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        validate_default=True,
        validate_assignment=True,
        repr_exclude_defaults=True,  # type: ignore [typeddict-unknown-key]
    )

    # repr that excludes default values
    def __repr_args__(self) -> Iterable[tuple[str | None, Any]]:
        super_args = super().__repr_args__()
        if not self.model_config.get("repr_exclude_defaults"):
            yield from super_args
            return

        fields = self.model_fields
        for key, val in super_args:
            default = fields[key].get_default(  # type: ignore
                call_default_factory=True, validated_data={}
            )
            with suppress(Exception):
                if val == default:
                    continue
            yield key, val


F = TypeVar("F", covariant=True, bound="VisModel")


class BackendAdaptor(Protocol[F]):
    """Protocol for backend adaptor classes.

    An adaptor class is responsible for converting all of the ndv protocol methods
    into native calls for the given backend.
    """

    @abstractmethod
    def __init__(self, obj: F, **backend_kwargs: Any) -> None:
        """All backend adaptor objects receive the object they are adapting."""
        ...

    @abstractmethod
    def _vis_get_native(self) -> Any:
        """Return the native object for the backend."""

    # TODO: add a "detach" or "cleanup" method?


class SupportsVisibility(BackendAdaptor[F], Protocol):
    """Protocol for objects that support visibility (show/hide)."""

    @abstractmethod
    def _vis_set_visible(self, arg: bool) -> None:
        """Set the visibility of the object."""


AdaptorType = TypeVar("AdaptorType", bound=BackendAdaptor, covariant=True)


class VisModel(ModelBase, Generic[AdaptorType]):
    """Front end object driving a backend interface.

    This is an important class.  Most things subclass this.  It provides the event
    connection between the model object and a backend adaptor.

    A backend adaptor is a class that implements the BackendAdaptor protocol (of type
    `T`... for which this class is a generic). The backend adaptor is an object
    responsible for converting all of the ndv protocol methods (stuff like
    "_vis_set_width", "_vis_set_visible", etc...) into the appropriate calls for
    the given backend.
    """

    # Really, this should be `_backend_adaptors: ClassVar[dict[str, T]]``,
    # but thats a type error.
    # PEP 526 states that ClassVar cannot include any type variables...
    # but there is discussion that this might be too limiting.
    # dicsussion: https://github.com/python/mypy/issues/5144
    # _backend_adaptors: ClassVar[dict[str, BackendAdaptor]] = PrivateAttr({})

    # This is the set of all field names that must have setters in the backend adaptor.
    # set during the init
    # _evented_fields: ClassVar[set[str]] = PrivateAttr(set())

    # this is a cache of all adaptor classes that have been validated to implement
    # the correct methods (via validate_adaptor_class).
    _validated_adaptor_classes: ClassVar[set[type]] = set()

    def model_post_init(self, __context: Any) -> None:
        # if using this in an EventedModel, connect to the events
        self.events.connect(self._on_any_event)
        # determine fields that need setter methods in the backend adaptor
        # TODO:
        # this really shouldn't need to be in the init.  `__init_subclass__` would be
        # better, but that unfortunately gets called after EventedModel.__new__.
        # need to look into it
        signal_names = set(self.events)
        self._evented_fields = set(self.model_fields).intersection(signal_names)
        self._backend_adaptors: dict[str, BackendAdaptor] = {}

    def has_backend_adaptor(self, backend: str | None = None) -> bool:
        """Return True if the object has a backend adaptor.

        If None is passed, the returned bool indicates the presence of any
        adaptor class.
        """
        if backend is None:
            return bool(self._backend_adaptors)
        return backend in self._backend_adaptors

    def backend_adaptor(self, backend: str | None = None) -> AdaptorType:
        """Get the backend adaptor for this object. Creates one if it doesn't exist.

        Parameters
        ----------
        backend : str, optional
            The name of the backend to use, by default None.  If None, the default
            backend will be used.
        """
        backend = backend or _get_default_backend()
        if backend not in self._backend_adaptors:
            cls = self._get_adaptor_class(backend)
            self._backend_adaptors[backend] = self._create_adaptor(cls)
        return cast("AdaptorType", self._backend_adaptors[backend])

    @property
    def backend_adaptors(self) -> Iterable[AdaptorType]:
        """Convenient, public iterator for backend adaptor objects."""
        yield from self._backend_adaptors.values()  # type: ignore

    def dangerously_get_native_object(self, backend: str | None = None) -> Any:
        """Return the native object for a backend.

        NOTE! Directly modifying the backend objects is not supported.  This method
        is here as a convenience for debugging, development, and experimentation.
        Direct modification of the backend object may lead to desyncronization of
        the model and the backend object, or other unexpected behavior.
        """
        adaptor = self.backend_adaptor(backend=backend)
        return adaptor._vis_get_native()

    def _get_adaptor_class(
        self,
        backend: str,
        class_name: str | None = None,
    ) -> type[AdaptorType]:
        """Retrieve the adaptor class with the same name as the object class."""
        class_name = class_name or type(self).__name__
        backend_module = import_module(f"ndv.views._scene.{backend}")
        adaptor_class = getattr(backend_module, class_name)
        return self.validate_adaptor_class(adaptor_class)

    def _create_adaptor(self, cls: type[AdaptorType]) -> AdaptorType:
        """Instantiate the backend adaptor object.

        The purpose of this method is to allow subclasses to override the
        creation of the backend object. Or do something before/after.
        """
        logger.debug(f"Attaching {type(self)} to backend {cls}")
        adaptor = cls(self)
        sync_adaptor(adaptor, self)
        return adaptor

    def _sync_adaptors(self) -> None:
        for adaptor in self.backend_adaptors:
            sync_adaptor(adaptor, self)

    def _on_any_event(self, info: EmissionInfo) -> None:
        signal_name = info.signal.name
        if signal_name not in self._evented_fields:
            return

        # NOTE: this loop runs anytime any attribute on any model is changed...
        # so it has the potential to be a performance bottleneck.
        # It is the the apparent cost, however, for allowing a model object to have
        # multiple simultaneous backend adaptors. This should be re-evaluated often.
        for adaptor in self.backend_adaptors:
            try:
                name = SETTER_METHOD.format(name=signal_name)
                setter = getattr(adaptor, name)
            except AttributeError as e:
                logger.exception(e)
                return

            event_name = f"{type(self).__name__}.{signal_name}"
            logger.debug(f"{event_name}={info.args} emitting to backend")

            try:
                setter(info.args[0])
            except Exception as e:
                logger.exception(e)
                breakpoint()

    # TODO:
    # def detach(self) -> None:
    #     """Disconnect and destroy the backend adaptor from the object."""
    #     self._backend = None

    def validate_adaptor_class(self, adaptor_class: Any) -> type[AdaptorType]:
        """Validate that the adaptor class is appropriate for the core object."""
        # XXX: this could be a classmethod, but it's turning out to be difficult to
        # set _evented_fields on that class (see note in __init__)

        cls = type(self)
        if adaptor_class in cls._validated_adaptor_classes:
            return cast("type[AdaptorType]", adaptor_class)

        # logger.debug(f"Validating adaptor class {adaptor_class} for {cls}")
        if missing := {
            SETTER_METHOD.format(name=field)
            for field in self._evented_fields
            if not hasattr(adaptor_class, SETTER_METHOD.format(name=field))
        }:
            raise ValueError(
                f"{adaptor_class} cannot be used as a backend object for "
                f"{cls}: it is missing the following methods: {missing}"
            )
        cls._validated_adaptor_classes.add(adaptor_class)
        return cast("type[AdaptorType]", adaptor_class)


# XXX: the default behavior should be to
# pick the "right" backend for the current environment.
# i.e. ndv should work with no configuration in both jupyter and ipython desktop.)
def _get_default_backend() -> str:
    """Stub function for the concept of picking a backend when none is specified.

    This will likely be context dependent.
    """
    from ndv.views._app import canvas_backend

    return canvas_backend(None).value


def _update_blocker(adaptor: BackendAdaptor) -> contextlib.AbstractContextManager:
    from ndv.models._scene.nodes.node import NodeAdaptorProtocol

    if isinstance(adaptor, NodeAdaptorProtocol):

        @contextlib.contextmanager
        def blocker() -> Iterator[None]:
            adaptor._vis_block_updates()
            try:
                yield
            finally:
                adaptor._vis_unblock_updates()

        return blocker()
    return contextlib.nullcontext()


def sync_adaptor(adaptor: BackendAdaptor, model: VisModel) -> None:
    """Decorator to validate and cache adaptor classes."""
    with _update_blocker(adaptor):
        for field_name in model.model_fields:
            method_name = SETTER_METHOD.format(name=field_name)
            value = getattr(model, field_name)
            try:
                vis_set = getattr(adaptor, method_name)
                vis_set(value)
            except Exception as e:
                logger.error(
                    "Failed to set field %r on adaptor %r: %s", field_name, adaptor, e
                )
    force_update = getattr(adaptor, "_vis_force_update", lambda: None)
    force_update()
