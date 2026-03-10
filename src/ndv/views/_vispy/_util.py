from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

from vispy.app import Canvas
from vispy.gloo import gl
from vispy.gloo.context import get_current_canvas

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def _opengl_context() -> Generator[None, None, None]:
    """Assure we are running with a valid OpenGL context.

    Only create a Canvas is one doesn't exist. Creating and closing a
    Canvas causes vispy to process Qt events which can cause problems.
    """
    canvas = Canvas(show=False) if get_current_canvas() is None else None
    try:
        yield
    finally:
        if canvas is not None:
            canvas.close()


@lru_cache
def get_max_texture_sizes() -> tuple[int | None, int | None]:
    """Return the maximum texture sizes for 2D and 3D rendering.

    Returns
    -------
    Tuple[int | None, int | None]
        The max textures sizes for (2d, 3d) rendering.
    """
    with _opengl_context():
        max_size_2d = gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)

    if not max_size_2d:
        max_size_2d = None

    # vispy/gloo doesn't provide the GL_MAX_3D_TEXTURE_SIZE location,
    # but it can be found in this list of constants
    # http://pyopengl.sourceforge.net/documentation/pydoc/OpenGL.GL.html
    with _opengl_context():
        GL_MAX_3D_TEXTURE_SIZE = 32883
        max_size_3d = gl.glGetParameter(GL_MAX_3D_TEXTURE_SIZE)

    if not max_size_3d:
        max_size_3d = None

    return max_size_2d, max_size_3d
