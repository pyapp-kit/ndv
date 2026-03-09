from typing import ClassVar

from psygnal import SignalGroupDescriptor
from pydantic import BaseModel, ConfigDict


class NDVModel(BaseModel):
    """Base evented model for NDV models.

    Uses [pydantic.BaseModel][] and [psygnal.SignalGroupDescriptor][].
    """

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )
    events: ClassVar[SignalGroupDescriptor] = SignalGroupDescriptor()
