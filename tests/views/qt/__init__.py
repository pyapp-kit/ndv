"""Tests pertaining to Qt components"""

from importlib.util import find_spec

import pytest

if find_spec("qtpy") is None:
    pytest.skip("Skipping Qt tests as Qt is not installed")
