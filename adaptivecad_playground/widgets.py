"""Qt widgets for the AdaptiveCAD Playground UI."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from geometry import (
    arc_length,
    area,
    circle_points,
    curve_area,
    curve_arc_length,
    curve_points,
    pi_a,
    point_to_polyline_distance,
)
from tools import PiACircleTool, PiACurveTool, SelectTool

Point = Tuple[float, float]
Params = Dict[str, float]


@dataclass
class Shape:
    """Persisted shape data used for painting, export, and metrics."""

    type: str
    params: Params
    points: np.ndarray
    center: Optional[Point] = None
    radius: Optional[float] = None
    control_points: Optional[List[Point]] = None

    def copy(self) -> "Shape":
        return Shape(
            type=self.type,
            params=dict(self.params),
            points=self.points.copy(),
            center=None if self.center is None else (self.center[0], self.center[1]),
            radius=self.radius,
            control_points=None if self.control_points is None else list(self.control_points),
        )


class Controls:
    """Docked parameter sliders that feed the canvas."""

    _DESCRIPTIONS = {
        "alpha": "Reinforcement α: blend in external bias to πₐ.",
        "mu": "Decay μ: how fast πₐ relaxes back as radius increases.",
        "k0": "Curvature seed k₀: base curvature that bends circles/curves.",
    }

    def __init__(self, set_params_cb):
        self._set_params_cb = set_params_cb
        self.dock = QDockWidget("Parameters")
        self.dock.setObjectName("AdaptiveParamsDock")
        self.dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.dock.setToolTip("Parameters: adjust α, μ, and k₀ to reshape πₐ responses.")
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._value_labels: Dict[str, QLabel] = {}
        self._slider_widgets: Dict[str, QSlider] = {}

        self._add_slider(layout, "alpha", "α", 0, 100, 0)
        self._add_slider(layout, "mu", "μ", 0, 100, 5)
        self._add_slider(layout, "k0", "k₀", -100, 100, 10)

        layout.addStretch(1)
        self.dock.setWidget(host)
        self._emit()

    def _add_slider(self, layout: QVBoxLayout, key: str, label_text: str, mn: int, mx: int, value: int) -> None:
        label = QLabel(f"{label_text}: {value / 100:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.valueChanged.connect(lambda v, k=key, lbl=label, txt=label_text: self._on_value_changed(k, lbl, txt, v))
        desc = self._DESCRIPTIONS.get(key, "")
        if desc:
            label.setToolTip(desc)
            slider.setToolTip(desc + " Drag to adjust.")
        layout.addWidget(label)
        layout.addWidget(slider)
        self._value_labels[key] = label
        self._slider_widgets[key] = slider

    def _on_value_changed(self, key: str, label: QLabel, label_text: str, value: int) -> None:
        label.setText(f"{label_text}: {value / 100:.2f}")
        self._emit()

    def _emit(self) -> None:
        params = {
            "alpha": self._slider_widgets["alpha"].value() / 100.0,
            "mu": self._slider_widgets["mu"].value() / 100.0,
            "k0": self._slider_widgets["k0"].value() / 100.0,
        }
        self._set_params_cb(params)


class Canvas(QWidget):
    """Main drawing surface handling user interaction and rendering."""

    status_changed = Signal(dict)
    tool_changed = Signal(str)
    snap_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("AdaptiveCanvas")
        self.setMinimumSize(QSize(640, 480))
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(
            "Canvas: left-click with the active tool.\n"
            "PiA Circle: first click = center, second click = radius.\n"
            "PiA Curve: click points, press Enter to commit."
        )

        self._params: Params = {"alpha": 0.0, "mu": 0.05, "k0": 0.1}
        self._shapes: List[Shape] = []
        self._selected_index: Optional[int] = None
        self._temp_shape: Optional[Shape] = None

        self._tool_name: Optional[str] = None
        self._tool_cache = {}
        self._tool = None
        # Snap and grid
        self._snap_to_grid = True
        self._grid_size = 20

        # History stacks
        self._undo_stack = []
        self._redo_stack = []

        self._emit_default_status()

    # ------------------------------------------------------------------
    # Parameter management
    def set_params(self, params: Params) -> None:
        self._params = dict(params)

    def params(self) -> Params:
        return dict(self._params)

    # ------------------------------------------------------------------
    # Snap/grid control
    def set_snap_to_grid(self, enabled: bool) -> None:
        self._apply_snap_state(enabled, emit=True)

    def toggle_snap_to_grid(self) -> None:
        self.set_snap_to_grid(not self._snap_to_grid)

    def is_snap_enabled(self) -> bool:
        return self._snap_to_grid

    def set_grid_size(self, size: int) -> None:
        self._grid_size = max(2, int(size))
        self.update()

    def grid_size(self) -> int:
        return self._grid_size

    def snap_point(self, pt: Point) -> Point:
        if not self._snap_to_grid:
            return (float(pt[0]), float(pt[1]))
        g = float(self._grid_size)
        return (round(pt[0] / g) * g, round(pt[1] / g) * g)

    def _apply_snap_state(self, enabled: bool, emit: bool) -> None:
        enabled = bool(enabled)
        if self._snap_to_grid == enabled:
            return
        self._snap_to_grid = enabled
        self.update()
        if emit:
            self.snap_changed.emit(enabled)

    # ------------------------------------------------------------------
    # Tool management
    def available_tools(self) -> Tuple[str, ...]:
        return ("select", "piacircle", "piacurve")

    def current_tool(self) -> Optional[str]:
        return self._tool_name

    def set_tool(self, name: str) -> None:
        if name == self._tool_name:
            return
        if name not in self.available_tools():
            raise ValueError(f"Unknown tool '{name}'")
        if self._tool is not None:
            self._tool.deactivate()
        if name not in self._tool_cache:
            self._tool_cache[name] = self._instantiate_tool(name)
        self._tool = self._tool_cache[name]
        self._tool_name = name
        self.tool_changed.emit(name)
        self.setFocus()
        self.update()

    def _instantiate_tool(self, name: str):
        if name == "select":
            return SelectTool(self)
        if name == "piacircle":
            return PiACircleTool(self)
        if name == "piacurve":
            return PiACurveTool(self)
        raise ValueError(f"Unhandled tool '{name}'")

    # ------------------------------------------------------------------
    # Shape fabrication helpers
    def shape_from_circle(self, center: Point, radius: float, params: Optional[Params] = None) -> Shape:
        params = dict(params or self._params)
        circle_pts = circle_points(center, radius, params, samples=256)
        return Shape(
            type="piacircle",
            params=params,
            points=circle_pts,
            center=(float(center[0]), float(center[1])),
            radius=float(radius),
        )

    def shape_from_curve(self, control_points: Sequence[Point], params: Optional[Params] = None) -> Shape:
        params = dict(params or self._params)
        ctrl = [
            (float(pt[0]), float(pt[1]))
            for pt in control_points
        ]
        curve_pts = curve_points(ctrl, params, samples=256)
        return Shape(
            type="piacurve",
            params=params,
            points=curve_pts,
            control_points=ctrl,
        )

    def add_circle(self, center: Point, radius: float) -> None:
        self._push_history()
        shape = self.shape_from_circle(center, radius, self._params)
        self._shapes.append(shape)
        self._set_selected_index(len(self._shapes) - 1)
        self.update()

    def add_curve(self, control_points: Sequence[Point]) -> None:
        if len(control_points) < 2:
            return
        self._push_history()
        shape = self.shape_from_curve(control_points, self._params)
        self._shapes.append(shape)
        self._set_selected_index(len(self._shapes) - 1)
        self.update()

    def set_temp_shape(self, shape: Shape, show_status: bool = True) -> None:
        self._temp_shape = shape
        self.update()
        if show_status:
            self.emit_status_from_shape(shape)

    def clear_temp_shape(self) -> None:
        if self._temp_shape is None:
            return
        self._temp_shape = None
        self.update()
        self._emit_selection_status()

    # ------------------------------------------------------------------
    # Selection & status helpers
    def select_shape_at(self, point: Point) -> None:
        if not self._shapes:
            self._set_selected_index(None)
            return
        threshold = 12.0
        best_index = None
        best_distance = float("inf")
        for idx, shape in enumerate(self._shapes):
            dist = point_to_polyline_distance(point, shape.points)
            if dist < best_distance:
                best_distance = dist
                best_index = idx
        if best_index is not None and best_distance <= threshold:
            self._set_selected_index(best_index)
        else:
            self._set_selected_index(None)

    def emit_status_from_shape(self, shape: Shape) -> None:
        metrics = self._shape_metrics(shape)
        metrics["kind"] = shape.type
        self.status_changed.emit(metrics)

    def _emit_default_status(self) -> None:
        self.status_changed.emit({"kind": None, "radius": None, "pi_a": None, "arc_length": None, "area": None})

    def _emit_selection_status(self) -> None:
        if self._selected_index is None:
            self._emit_default_status()
            return
        self.emit_status_from_shape(self._shapes[self._selected_index])

    def _set_selected_index(self, index: Optional[int]) -> None:
        if index is not None and (index < 0 or index >= len(self._shapes)):
            index = None
        if index == self._selected_index:
            self._emit_selection_status()
            return
        self._selected_index = index
        self._emit_selection_status()
        self.update()

    def _shape_metrics(self, shape: Shape) -> Dict[str, Optional[float]]:
        if shape.type == "piacircle":
            radius = shape.radius or 0.0
            params = shape.params
            return {
                "radius": radius,
                "pi_a": pi_a(radius, **params),
                "arc_length": arc_length(radius, params),
                "area": area(radius, params),
            }
        if shape.type == "piacurve":
            control_points = shape.control_points or []
            if len(control_points) < 2:
                span = 0.0
            else:
                span = math.dist(control_points[0], control_points[-1])
            params = shape.params
            metrics = {
                "radius": span / 2.0 if span > 0.0 else 0.0,
                "pi_a": pi_a(max(span, 1e-6), **params),
                "arc_length": curve_arc_length(control_points, params),
                "area": None,
            }
            if len(control_points) >= 3:
                metrics["area"] = curve_area(control_points, params)
            return metrics
        return {"radius": None, "pi_a": None, "arc_length": None, "area": None}

    # ------------------------------------------------------------------
    # Painting
    def paintEvent(self, event):  # pragma: no cover - GUI entry point
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(250, 250, 250))

        # Draw grid if enabled
        if self._snap_to_grid:
            self._draw_grid(painter)

        for idx, shape in enumerate(self._shapes):
            if shape.points is None or len(shape.points) < 2:
                continue
            color = QColor(30, 30, 30)
            width = 2
            if idx == self._selected_index:
                color = QColor(50, 120, 215)
                width = 3
            self._draw_shape(painter, shape, color, width)

        if self._temp_shape and self._temp_shape.points is not None and len(self._temp_shape.points) >= 2:
            self._draw_shape(painter, self._temp_shape, QColor(200, 80, 80), 2, dashed=True)

    def _draw_shape(self, painter: QPainter, shape: Shape, color: QColor, width: int, dashed: bool = False) -> None:
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        points = [QPointF(float(x), float(y)) for x, y in shape.points]
        painter.drawPolyline(points)
        if shape.type == "piacurve" and shape.control_points:
            handle_pen = QPen(QColor(120, 120, 120), 1, Qt.DotLine)
            painter.setPen(handle_pen)
            controls = [QPointF(float(x), float(y)) for x, y in shape.control_points]
            painter.drawPolyline(controls)
            painter.setPen(QPen(QColor(120, 120, 120), 4))
            for pt in controls:
                painter.drawPoint(pt)

    def _draw_grid(self, painter: QPainter) -> None:
        g = self._grid_size
        if g <= 0:
            return
        rect = self.rect()
        left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
        pen_minor = QPen(QColor(235, 235, 235), 1)
        pen_major = QPen(QColor(220, 220, 220), 1)
        # vertical lines
        x = (left // g) * g
        while x <= right:
            painter.setPen(pen_major if (x % (g * 5) == 0) else pen_minor)
            painter.drawLine(int(x), top, int(x), bottom)
            x += g
        # horizontal lines
        y = (top // g) * g
        while y <= bottom:
            painter.setPen(pen_major if (y % (g * 5) == 0) else pen_minor)
            painter.drawLine(left, int(y), right, int(y))
            y += g

    # ------------------------------------------------------------------
    # History (Undo/Redo)
    def _snapshot(self) -> dict:
        return {
            "params": dict(self._params),
            "shapes": [self._serialize_shape(s) for s in self._shapes],
            "selected": self._selected_index,
            "snap": bool(self._snap_to_grid),
            "grid": int(self._grid_size),
        }

    def _restore(self, snap: dict) -> None:
        self._params = dict(snap.get("params", self._params))
        # Rehydrate shapes
        shapes = []
        for sd in snap.get("shapes", []):
            s = Shape(
                type=sd.get("type", "piacircle"),
                params=dict(sd.get("params", {})),
                points=np.array(sd.get("points", []), dtype=float),
                center=tuple(sd["center"]) if "center" in sd else None,
                radius=sd.get("radius"),
                control_points=[tuple(p) for p in sd.get("control_points", [])] if sd.get("control_points") else None,
            )
            shapes.append(s)
        self._shapes = shapes
        self._selected_index = snap.get("selected")
        self._apply_snap_state(snap.get("snap", self._snap_to_grid), emit=False)
        self._grid_size = int(snap.get("grid", self._grid_size))
        self.update()
        self._emit_selection_status()
        self.snap_changed.emit(self._snap_to_grid)

    def _push_history(self) -> None:
        # Cap history to avoid unbounded growth
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        current = self._snapshot()
        state = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore(state)

    def redo(self) -> None:
        if not self._redo_stack:
            return
        current = self._snapshot()
        state = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._restore(state)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    # ------------------------------------------------------------------
    # Event forwarding to the active tool
    def mousePressEvent(self, event):  # pragma: no cover - GUI entry point
        if self._tool is not None:
            self._tool.mouse_press(event)

    def mouseMoveEvent(self, event):  # pragma: no cover - GUI entry point
        if self._tool is not None:
            self._tool.mouse_move(event)

    def mouseReleaseEvent(self, event):  # pragma: no cover - GUI entry point
        if self._tool is not None:
            self._tool.mouse_release(event)

    def keyPressEvent(self, event):  # pragma: no cover - GUI entry point
        if self._tool is not None:
            self._tool.key_press(event)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Export helpers
    def export_png(self, parent=None) -> None:
        path, _ = QFileDialog.getSaveFileName(parent or self, "Export PNG", "adaptivecad_playground.png", "PNG Files (*.png)")
        if not path:
            return
        self.grab().save(path)

    def export_json(self, parent=None) -> None:
        path, _ = QFileDialog.getSaveFileName(parent or self, "Export JSON", "adaptivecad_playground.json", "JSON Files (*.json)")
        if not path:
            return
        payload = {
            "canvas_size": {"width": self.width(), "height": self.height()},
            "params": dict(self._params),
            "snap_to_grid": self._snap_to_grid,
            "grid_size": self._grid_size,
            "shapes": [self._serialize_shape(shape) for shape in self._shapes],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _serialize_shape(self, shape: Shape) -> dict:
        data = {
            "type": shape.type,
            "params": dict(shape.params),
            "points": shape.points.tolist() if shape.points is not None else [],
        }
        if shape.center is not None:
            data["center"] = list(shape.center)
        if shape.radius is not None:
            data["radius"] = shape.radius
        if shape.control_points is not None:
            data["control_points"] = [list(pt) for pt in shape.control_points]
        return data
