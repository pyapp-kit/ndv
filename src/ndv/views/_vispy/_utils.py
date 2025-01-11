from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from typing import TYPE_CHECKING

from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _opengl_context() -> Iterator[None]:
    """Assure we are running with a valid OpenGL context.

    Only create a Canvas if one doesn't exist. Creating and closing a
    Canvas causes vispy to process Qt events which can cause problems.
    Ideally call opengl_context() on start after creating your first
    Canvas. However it will work either way.
    """
    canvas = Canvas(show=False) if get_current_canvas() is None else None
    try:
        yield
    finally:
        if canvas is not None:
            canvas.close()


@cache
def get_gl_extensions() -> set[str]:
    """Get basic info about the Gl capabilities of this machine."""
    with _opengl_context():
        return set(filter(None, gl.glGetParameter(gl.GL_EXTENSIONS).split()))


FLOAT_EXT = {"GL_ARB_texture_float", "GL_ATI_texture_float", "GL_NV_float_buffer"}


@cache
def supports_float_textures() -> bool:
    """Check if the current OpenGL context supports float textures."""
    return bool(FLOAT_EXT.intersection(get_gl_extensions()))
