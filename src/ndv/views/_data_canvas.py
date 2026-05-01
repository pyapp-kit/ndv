from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psygnal import Signal

if TYPE_CHECKING:
    from ndv.models._viewer_model import ArrayViewerModel

import cmap
import numpy as np
import scenex.app.events as events
from pydantic import Field, field_validator
from scenex import (
    Camera,
    Canvas,
    Image,
    Letterbox,
    Line,
    Mesh,
    Orbit,
    PanZoom,
    Points,
    Scene,
    Transform,
    UniformColor,
    View,
    Volume,
    set_cursor,
)
from scenex.adaptors import get_adaptor_registry
from scenex.app import CursorType
from scenex.model import EventedBase
from scenex.utils import projections


class DataCanvas:
    eventCaptured = Signal(events.Event)

    def __init__(self, viewer_model: ArrayViewerModel) -> None:
        self.viewer_model = viewer_model
        # We have one view
        self.view = View(
            scene=Scene(interactive=True),
            camera=Camera(interactive=True),
            on_resize=Letterbox(),
        )
        # On a canvas
        # NOTE: Keep the canvas hidden until we're ready to show it.
        # More than anything else, this prevents the GL context from being created
        # on the vispy backends before we're ready, leading to some nasty segfaults
        self._canvas = Canvas(width=600, height=600, views=[self.view], visible=False)

        self.roi_view = RectangularROI()
        self.roi_view.rect_mesh.visible = False
        self.roi_view.rect_mesh.parent = self.view.scene

        # Showing two dimensions
        self.ndims = 2

        self._canvas.set_event_filter(self._on_event)

    def _on_event(self, event: events.Event) -> bool:
        """Filter events from the canvas and re-emit them as a signal."""
        if self.roi_view.handle_event(
            event,
            self.view,
            self._canvas,
            self.viewer_model,
        ):
            return True
        self.eventCaptured.emit(event)
        return False  # don't consume the event, allow normal processing to continue

    def widget(self) -> Any:
        return get_adaptor_registry().get_adaptor(self._canvas)._snx_get_native()

    @property
    def ndims(self) -> int:
        return 2 if isinstance(self.view.camera.controller, PanZoom) else 3

    @ndims.setter
    def ndims(self, ndim: int) -> None:
        if ndim == 2:
            if not isinstance(self.view.camera.controller, PanZoom):
                self.view.camera.controller = PanZoom()
        elif ndim == 3:
            if not isinstance(self.view.camera.controller, Orbit):
                # FIXME: This logs a warning because child events are still being
                # propagated to the parent Camera.
                self.view.camera.controller = Orbit()
        else:
            raise ValueError("n_axes must be 2 or 3")
        self.reset_zoom()

    def reset_zoom(self) -> None:
        controller = self.view.camera.controller
        if isinstance(controller, PanZoom):
            projections.zoom_to_fit(
                self.view,
                type="orthographic",
                zoom_factor=0.9,
                letterbox=True,
            )
        elif isinstance(controller, Orbit):
            projections.zoom_to_fit(
                self.view,
                type="perspective",
                zoom_factor=0.9,
                letterbox=True,
            )
            if bb := self.view.scene.bounding_box:
                controller.center = np.mean(bb, axis=0)
            else:
                controller.center = (0, 0, 0)

    # FIXME: This doesn't really belong here.
    def set_scales(self, scales: tuple[float, ...]) -> None:
        """Set per-visible-axis scale factors for rendering."""
        if not scales:
            return
        # scales are in data order (slowest-to-fastest, e.g. ZYX)
        # vispy images use row,col -> y,x mapping, so reverse for XY
        vis_scales = list(reversed(scales))
        # pad to 3 components
        while len(vis_scales) < 3:
            vis_scales.append(1.0)
        sx, sy, sz = vis_scales[0], vis_scales[1], vis_scales[2]
        for node in self.view.scene.children:
            if not isinstance(node, Image | Volume):
                continue
            # FIXME: This might ignore downsampling. We will have to test.
            node.transform = Transform().scaled((sx, sy, sz))


def _dummy_mesh() -> Mesh:
    return Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        color=UniformColor(color=cmap.Color("royalblue")),
        opacity=0.25,
        order=1,
    )


class RectangularROI(EventedBase):
    bb: tuple[tuple[float, float], tuple[float, float]] = ((0, 0), (0, 0))
    handle_color: cmap.Color = cmap.Color("white")

    outline_color: cmap.Color = cmap.Color("royalblue")
    fill_color: cmap.Color = cmap.Color("green")
    anchor: tuple[float, float] | None = None
    drag_start: tuple[float, float] | None = None

    rect_mesh: Mesh = Field(default_factory=_dummy_mesh)
    rect_line: Line = Field(default_factory=Line)
    handles: Points = Field(default_factory=Points)

    def __init__(self) -> None:
        super().__init__()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)

        self.rect_mesh = Mesh(
            vertices=vertices,
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
            color=UniformColor(color=cmap.Color((0, 0, 0, 0))),
            opacity=0.25,
            order=1,
        )

        self.rect_line = Line(
            parent=self.rect_mesh,
            vertices=vertices[[0, 1, 2, 3, 0]],
            color=UniformColor(color=cmap.Color("yellow")),
            width=2.0,
            order=2,
        )

        self.handles = Points(
            parent=self.rect_mesh,
            vertices=vertices,
            size=14,
            face_color=UniformColor(color=cmap.Color("white")),
            symbol="disc",
            scaling="fixed",
            order=3,
        )
        self.events.bb.connect(self._on_bounding_box_changed)
        self.events.handle_color.connect(self._on_handle_color_changed)
        self.events.outline_color.connect(self._on_outline_color_changed)
        self.events.fill_color.connect(self._on_fill_color_changed)

    @field_validator("bb", mode="before")
    @classmethod
    def _normalize_bb(cls, v: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        (x0, y0), (x1, y1) = v
        return ((min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)))

    def _on_bounding_box_changed(
        self, bb: tuple[tuple[float, float], tuple[float, float]]
    ) -> None:
        x1, y1 = bb[0]
        # avoid zero-size which can cause rendering issues
        x2 = max(bb[1][0], bb[0][0] + 1e-5)
        y2 = max(bb[1][1], bb[0][1] + 1e-5)
        vertices = np.array(
            [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0]], dtype=float
        )
        self.rect_mesh.vertices = vertices
        self.rect_line.vertices = vertices[[0, 1, 2, 3, 0]]
        self.handles.vertices = vertices

    def _on_handle_color_changed(self, color: cmap.Color) -> None:
        self.handles.face_color = UniformColor(color=color)

    def _on_outline_color_changed(self, color: cmap.Color) -> None:
        self.rect_line.color = UniformColor(color=color)

    def _on_fill_color_changed(self, color: cmap.Color) -> None:
        self.rect_mesh.color = UniformColor(color=color)

    def _nearest_corner(self, wx: float, wy: float) -> int:
        """Index of the corner handle nearest to world position (wx, wy)."""
        world = self.rect_mesh.transform.map(self.rect_mesh.vertices)[:, :2]
        return int(np.argmin(np.linalg.norm(world - [wx, wy], axis=1)))

    def _cursor_for_pos(self, wx: float, wy: float) -> CursorType:
        # Even corners (BL=0, TR=2) are on the main diagonal,
        # odd corners (BR=1, TL=3) on the anti-diagonal.
        return (
            CursorType.BDIAG_ARROW
            if self._nearest_corner(wx, wy) % 2 == 0
            else CursorType.FDIAG_ARROW
        )

    def handle_event(
        self,
        event: events.Event,
        view: View,
        canvas: Canvas,
        viewer_model: ArrayViewerModel,
    ) -> bool:
        """Handle ROI creation, dragging, and cursor updates.

        Returns True if the event was consumed.
        """
        from ndv.models._viewer_model import InteractionMode

        if isinstance(event, events.MouseMoveEvent):
            if not (ray := view.to_ray(event.pos)):
                return False
            pos = ray.origin[:2]
            # -- Dragging a handle -- #
            if self.anchor is not None:
                self.bb = (pos, self.anchor)
                set_cursor(canvas, self._cursor_for_pos(*pos))
                return True
            # -- Dragging the whole rectangle -- #
            if self.drag_start is not None:
                delta = np.subtract(pos, self.drag_start)
                # NOTE we just need two opposite corners, doesn't matter which two.
                v0 = self.rect_mesh.vertices[0, :2] + delta
                v2 = self.rect_mesh.vertices[2, :2] + delta
                self.bb = (v0, v2)
                self.drag_start = pos
                return True
            # -- Hover cursor -- #
            if viewer_model.interaction_mode == InteractionMode.CREATE_ROI:
                set_cursor(canvas, CursorType.CROSS)
            elif ray.intersections(self.handles):
                set_cursor(canvas, self._cursor_for_pos(*pos))
            elif ray.intersections(self.rect_mesh):
                set_cursor(canvas, CursorType.ALL_ARROW)
            else:
                set_cursor(canvas, CursorType.DEFAULT)

        elif isinstance(event, events.MousePressEvent):
            if not (ray := view.to_ray(event.pos)):
                return False
            if event.buttons & events.MouseButton.LEFT:
                pos = ray.origin[:2]
                if viewer_model.interaction_mode == InteractionMode.CREATE_ROI:
                    self.rect_mesh.visible = True
                    self.bb = (
                        (ray.origin[0], ray.origin[1]),
                        (ray.origin[0] + 1, ray.origin[1] + 1),
                    )
                    viewer_model.interaction_mode = InteractionMode.PAN_ZOOM
                # -- Start a handle drag -- #
                if ray.intersections(self.handles):
                    clicked = self._nearest_corner(*pos)
                    opp = (clicked + 2) % 4
                    self.anchor = self.rect_mesh.vertices[opp, :2]
                    return True
                # -- Start a rectangle drag -- #
                elif ray.intersections(self.rect_mesh):
                    self.drag_start = pos
                    return True

        elif isinstance(event, events.MouseReleaseEvent):
            self.anchor = None
            self.drag_start = None
            return True

        elif isinstance(event, events.MouseLeaveEvent):
            self.anchor = None
            self.drag_start = None
            set_cursor(canvas, CursorType.DEFAULT)

        return False
