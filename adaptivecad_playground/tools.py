"""Interactive tools for the AdaptiveCAD Playground canvas."""
from __future__ import annotations

import math
from typing import List, Tuple

from PySide6.QtCore import Qt

Point = Tuple[float, float]


class ToolBase:
    """Common interface every tool implements."""

    def __init__(self, canvas):
        self.canvas = canvas

    def mouse_press(self, event):  # pragma: no cover - GUI entry point
        pass

    def mouse_move(self, event):  # pragma: no cover - GUI entry point
        pass

    def mouse_release(self, event):  # pragma: no cover - GUI entry point
        pass

    def key_press(self, event):  # pragma: no cover - GUI entry point
        pass

    def deactivate(self):  # pragma: no cover - GUI entry point
        pass


class SelectTool(ToolBase):
    """Hit-test the scene and select shapes."""

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return
        point = (event.position().x(), event.position().y())
        self.canvas.select_shape_at(point)


class PiACircleTool(ToolBase):
    """Two-click adaptive circle creation tool."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self._center: Point | None = None

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return
        point = (event.position().x(), event.position().y())
        point = self.canvas.snap_point(point)
        if self._center is None:
            self._center = point
            preview = self.canvas.shape_from_circle(point, 0.0, self.canvas.params())
            self.canvas.set_temp_shape(preview, show_status=False)
            self.canvas.emit_status_from_shape(preview)
            return
        radius = math.hypot(point[0] - self._center[0], point[1] - self._center[1])
        if radius <= 1e-3:
            return
        self.canvas.add_circle(self._center, radius)
        self._center = None
        self.canvas.clear_temp_shape()

    def mouse_move(self, event):
        if self._center is None:
            return
        point = (event.position().x(), event.position().y())
        point = self.canvas.snap_point(point)
        radius = math.hypot(point[0] - self._center[0], point[1] - self._center[1])
        preview = self.canvas.shape_from_circle(self._center, radius, self.canvas.params())
        self.canvas.set_temp_shape(preview)

    def deactivate(self):
        self._center = None
        self.canvas.clear_temp_shape()


class PiACurveTool(ToolBase):
    """Bezier-like curve creation driven by adaptive π."""

    def __init__(self, canvas):
        super().__init__(canvas)
        self._control_points: List[Point] = []
        self._hover_point: Point | None = None

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return
        point = (event.position().x(), event.position().y())
        point = self.canvas.snap_point(point)
        self._control_points.append(point)
        self._hover_point = None
        self._update_preview()

    def mouse_move(self, event):
        if not self._control_points:
            return
        self._hover_point = (event.position().x(), event.position().y())
        self._hover_point = self.canvas.snap_point(self._hover_point)
        self._update_preview()

    def key_press(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if len(self._control_points) >= 2:
                self.canvas.add_curve(self._control_points)
            self._reset()
        elif key in (Qt.Key_Escape,):
            self._reset()
        elif key in (Qt.Key_Backspace, Qt.Key_Delete):
            if self._control_points:
                self._control_points.pop()
                self._update_preview()

    def deactivate(self):
        self._reset()

    def _reset(self):
        self._control_points.clear()
        self._hover_point = None
        self.canvas.clear_temp_shape()

    def _update_preview(self):
        candidate: List[Point] = list(self._control_points)
        if self._hover_point is not None:
            candidate.append(self._hover_point)
        if len(candidate) < 2:
            self.canvas.clear_temp_shape()
            return
        preview = self.canvas.shape_from_curve(candidate, self.canvas.params())
        self.canvas.set_temp_shape(preview)
