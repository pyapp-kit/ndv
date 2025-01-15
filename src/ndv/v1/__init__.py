"""Here to allow access to the original (legacy) version of NDViewer.

!!! warning

    This module should not be used for new code.  It will be removed in a future
    release.
"""

from ._old_data_wrapper import DataWrapper
from ._old_viewer import NDViewer
from ._util import imshow

__all__ = ["DataWrapper", "NDViewer", "imshow"]
