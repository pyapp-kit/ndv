"""Tests pertaining to Qt components"""

import pytest

try:
    # NB: We could use importlib, but we'd have to search for qtpy
    # AND for one of the many bindings (PyQt5, PyQt6, PySide2)
    # This seems easier.
    from qtpy.QtCore import Qt  # noqa: F401
except ImportError:
    pytest.skip("Skipping Qt tests as Qt is not installed", allow_module_level=True)
