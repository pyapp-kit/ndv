"""Here temporarily to allow access to the original (legacy) version of NDViewer.

This module should not be used for new code.
"""

from ._old_data_wrapper import DataWrapper
from ._old_viewer import NDViewer
from .util import imshow

__all__ = ["DataWrapper", "NDViewer", "imshow"]
