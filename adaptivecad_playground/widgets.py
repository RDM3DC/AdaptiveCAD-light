"""Qt widgets for the AdaptiveCAD Playground UI."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
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
from dimensions import (
    DimStyle,
    LinearDimension,
    RadialDimension,
    AngularDimension,
    linear_label,
    radial_label,
    angular_label,
    to_json as dims_to_json,
    from_json as dims_from_json,
)
from dim_draw import draw_linear_dim, draw_radial_dim, draw_angular_dim
from dim_tools import ToolContext, DimAngularTool, DimLinearTool, DimRadialTool, MeasureTool
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
    id: Optional[str] = None

    def copy(self) -> "Shape":
        return Shape(
            type=self.type,
            params=dict(self.params),
            points=self.points.copy(),
            center=None if self.center is None else (self.center[0], self.center[1]),
            radius=self.radius,
            control_points=None if self.control_points is None else list(self.control_points),
            id=self.id,
        )


class _DimensionPainter:
    """Adapter exposing primitive drawing hooks expected by dim_draw."""

    def __init__(self, painter: QPainter):
        self._painter = painter

    def _color(self, value: str) -> QColor:
        return QColor(value) if value else QColor(34, 34, 34)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float) -> None:
        self._painter.save()
        pen = QPen(self._color(color))
        pen.setWidthF(float(width))
        self._painter.setPen(pen)
        self._painter.drawLine(float(x1), float(y1), float(x2), float(y2))
        self._painter.restore()

    def polyline(self, points: List[Point], color: str, width: float) -> None:
        if not points:
            return
        self._painter.save()
        pen = QPen(self._color(color))
        pen.setWidthF(float(width))
        self._painter.setPen(pen)
        polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
        self._painter.drawPolyline(polygon)
        self._painter.restore()

    def text(
        self,
        label: str,
        x: float,
        y: float,
        px_height: float,
        color: str,
        anchor: str = "center",
        halo: bool = True,
    ) -> None:
        if not label:
            return
        self._painter.save()
        font = self._painter.font()
        font.setPixelSize(max(1, int(px_height)))
        self._painter.setFont(font)
        metrics = QFontMetrics(font)
        text_width = metrics.horizontalAdvance(label)
        text_height = metrics.height()
        if anchor == "center":
            x0 = float(x) - text_width / 2.0
            baseline = float(y) + text_height / 2.0 - metrics.descent()
        elif anchor == "right":
            x0 = float(x) - text_width
            baseline = float(y)
        else:  # left
            x0 = float(x)
            baseline = float(y)
        if halo:
            halo_pen = QPen(QColor(255, 255, 255))
            halo_pen.setWidth(max(1, int(px_height // 5 or 1)))
            self._painter.setPen(halo_pen)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    self._painter.drawText(x0 + dx, baseline + dy, label)
        text_pen = QPen(self._color(color))
        text_pen.setWidth(1)
        self._painter.setPen(text_pen)
        self._painter.drawText(x0, baseline, label)
        self._painter.restore()

    def arrow(self, x1: float, y1: float, x2: float, y2: float, size_px: float, color: str) -> None:
        dx = float(x1) - float(x2)
        dy = float(y1) - float(y2)
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        size = max(2.0, float(size_px))
        tip = QPointF(float(x2), float(y2))
        left = QPointF(tip.x() + ux * size + px * size * 0.5, tip.y() + uy * size + py * size * 0.5)
        right = QPointF(tip.x() + ux * size - px * size * 0.5, tip.y() + uy * size - py * size * 0.5)
        self._painter.save()
        pen = QPen(self._color(color))
        pen.setWidthF(1.0)
        self._painter.setPen(pen)
        self._painter.setBrush(self._color(color))
        polygon = QPolygonF([tip, left, right])
        self._painter.drawPolygon(polygon)
        self._painter.restore()
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
    dimensions_visibility_changed = Signal(bool)

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
        self._shape_counter: int = 1

        # Dimensions
        self._dimensions: List[object] = []
        self._show_dimensions: bool = True
        self._dim_style: DimStyle = DimStyle()

        self._tool_name: Optional[str] = None
        self._tool_cache = {}
        self._tool = None
        self._dynamic_tools = {"dim_linear", "dim_radial", "dim_angular", "measure"}
        # Snap and grid
        self._snap_to_grid = True
        self._grid_size = 20

        # History stacks
        self._undo_stack = []
        self._redo_stack = []
        self._status_state: Dict[str, Optional[float | str]] = {}

        self._emit_default_status()
        self.dimensions_visibility_changed.emit(self._show_dimensions)

    # ------------------------------------------------------------------
    # Parameter management
    def set_params(self, params: Params) -> None:
        self._params = dict(params)

    def params(self) -> Params:
        return dict(self._params)

    # ------------------------------------------------------------------
    # Status helpers
    def _emit_status(self, updates: Dict[str, Optional[float | str]]) -> None:
        self._status_state.update(updates)
        self.status_changed.emit(dict(self._status_state))

    def post_status_message(self, message: str) -> None:
        self._emit_status({"message": message})

    def dimension_style(self) -> DimStyle:
        return DimStyle(**self._dim_style.asdict())

    def set_dimension_style(self, style: DimStyle) -> None:
        self._push_history()
        self._dim_style = DimStyle(**style.asdict())
        self.update()

    def dimensions_visible(self) -> bool:
        return self._show_dimensions

    def set_dimensions_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._show_dimensions == visible:
            return
        self._push_history()
        self._show_dimensions = visible
        self.update()
        self.dimensions_visibility_changed.emit(self._show_dimensions)

    def toggle_dimensions_visible(self) -> None:
        self.set_dimensions_visible(not self._show_dimensions)

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

    def _world_from_event(self, event) -> Point:
        point = (event.position().x(), event.position().y())
        return self.snap_point(point)

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
        return (
            "select",
            "piacircle",
            "piacurve",
            "dim_linear",
            "dim_radial",
            "dim_angular",
            "measure",
        )

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
        self.post_status_message("")
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
        if name == "dim_linear":
            return DimLinearTool(self._build_dim_context())
        if name == "dim_radial":
            return DimRadialTool(self._build_dim_context())
        if name == "dim_angular":
            return DimAngularTool(self._build_dim_context())
        if name == "measure":
            return MeasureTool(self._build_dim_context())
        raise ValueError(f"Unhandled tool '{name}'")

    def _build_dim_context(self) -> ToolContext:
        return ToolContext(
            get_params=self.params,
            get_style=self.dimension_style,
            add_dimension=self.add_dimension,
            update_status=self.post_status_message,
            request_repaint=self.update,
            hit_test_circle=self.hit_test_circle,
            world_from_event=self._world_from_event,
            osnap_targets=self.osnap_targets,
        )

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

    def _next_shape_id(self) -> str:
        identifier = f"S{self._shape_counter:04d}"
        self._shape_counter += 1
        return identifier

    def add_circle(self, center: Point, radius: float) -> None:
        self._push_history()
        shape = self.shape_from_circle(center, radius, self._params)
        shape.id = self._next_shape_id()
        self._shapes.append(shape)
        self._set_selected_index(len(self._shapes) - 1)
        self.update()

    def add_curve(self, control_points: Sequence[Point]) -> None:
        if len(control_points) < 2:
            return
        self._push_history()
        shape = self.shape_from_curve(control_points, self._params)
        shape.id = self._next_shape_id()
        self._shapes.append(shape)
        self._set_selected_index(len(self._shapes) - 1)
        self.update()

    def add_dimension(self, dimension: object) -> None:
        self._push_history()
        self._dimensions.append(dimension)
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

    def osnap_targets(self) -> List[Tuple[str, Point]]:
        targets: List[Tuple[str, Point]] = []
        for shape in self._shapes:
            if shape.type == "piacircle" and shape.center is not None:
                cx, cy = shape.center
                targets.append(("center", shape.center))
                if shape.radius:
                    r = float(shape.radius)
                    targets.extend([
                        ("quad", (cx + r, cy)),
                        ("quad", (cx - r, cy)),
                        ("quad", (cx, cy + r)),
                        ("quad", (cx, cy - r)),
                    ])
            elif shape.type == "piacurve" and shape.control_points:
                ctrl = shape.control_points
                if ctrl:
                    targets.append(("end", ctrl[0]))
                    targets.append(("end", ctrl[-1]))
                    for mid in ctrl[1:-1]:
                        targets.append(("mid", mid))
        for dim in self._dimensions:
            if isinstance(dim, LinearDimension):
                targets.extend([("dim", dim.p1), ("dim", dim.p2), ("dim", dim.offset)])
            elif isinstance(dim, RadialDimension):
                targets.extend([("dim", dim.center), ("dim", dim.attach)])
            elif isinstance(dim, AngularDimension):
                targets.extend([
                    ("dim", dim.vtx),
                    ("dim", dim.p1),
                    ("dim", dim.p2),
                    ("dim", dim.attach),
                ])
        return targets

    def hit_test_circle(self, point: Point) -> Optional[Tuple[Point, float, str]]:
        best: Optional[Tuple[Point, float, str]] = None
        best_error = float("inf")
        for shape in self._shapes:
            if shape.type != "piacircle" or shape.center is None or shape.radius is None:
                continue
            cx, cy = shape.center
            radius = float(shape.radius)
            error = abs(math.hypot(point[0] - cx, point[1] - cy) - radius)
            if error < best_error and error <= 12.0:
                if not shape.id:
                    shape.id = self._next_shape_id()
                best = (shape.center, radius, shape.id)
                best_error = error
        return best

    def _shape_by_id(self, shape_id: Optional[str]) -> Optional[Shape]:
        if not shape_id:
            return None
        for shape in self._shapes:
            if shape.id == shape_id:
                return shape
        return None

    def _reseed_shape_counter(self) -> None:
        max_seen = 0
        for shape in self._shapes:
            if shape.id and shape.id.startswith("S"):
                try:
                    max_seen = max(max_seen, int(shape.id[1:]))
                except ValueError:
                    continue
        self._shape_counter = max_seen + 1

    def emit_status_from_shape(self, shape: Shape) -> None:
        metrics = self._shape_metrics(shape)
        metrics["kind"] = shape.type
        metrics["message"] = ""
        self._emit_status(metrics)

    def _emit_default_status(self) -> None:
        self._status_state = {
            "kind": None,
            "radius": None,
            "pi_a": None,
            "arc_length": None,
            "area": None,
            "message": "",
        }
        self.status_changed.emit(dict(self._status_state))

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

        self._paint_dimensions(painter)

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

    def _paint_dimensions(self, painter: QPainter) -> None:
        if not self._show_dimensions or not self._dimensions:
            return
        shim = _DimensionPainter(painter)
        to_screen = lambda x, y: (float(x), float(y))
        shape_lookup = {shape.id: shape for shape in self._shapes if shape.id}
        for dim in self._dimensions:
            style = dim.style if hasattr(dim, "style") else self._dim_style
            color = getattr(style, "color", "#222222")
            arrow_px = float(getattr(style, "arrow", 8.0))
            text_px = float(getattr(style, "text_height", 12.0))
            if isinstance(dim, LinearDimension):
                label = linear_label(dim.p1, dim.p2, style)
                draw_linear_dim(
                    shim,
                    to_screen,
                    dim.p1,
                    dim.p2,
                    dim.offset,
                    color=color,
                    arrow_px=arrow_px,
                    text=label,
                    text_px=text_px,
                )
            elif isinstance(dim, RadialDimension):
                params = dict(dim.params or {})
                if not params and getattr(dim, "shape_ref", None):
                    shape = shape_lookup.get(dim.shape_ref)
                    if shape is not None:
                        params = dict(shape.params)
                label = radial_label(dim.center, dim.radius, style, params)
                draw_radial_dim(
                    shim,
                    to_screen,
                    dim.center,
                    dim.attach,
                    color=color,
                    arrow_px=arrow_px,
                    text=label,
                    text_px=text_px,
                )
            elif isinstance(dim, AngularDimension):
                label = angular_label(dim.vtx, dim.p1, dim.p2, style)
                draw_angular_dim(
                    shim,
                    to_screen,
                    dim.vtx,
                    dim.p1,
                    dim.p2,
                    dim.attach,
                    color=color,
                    text=label,
                    text_px=text_px,
                )

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
            "dimensions": dims_to_json(self._dimensions),
            "dimension_style": self._dim_style.asdict(),
            "show_dimensions": self._show_dimensions,
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
                id=sd.get("id"),
            )
            shapes.append(s)
        self._shapes = shapes
        self._dimensions = dims_from_json(snap.get("dimensions", []))
        style_data = snap.get("dimension_style")
        if style_data:
            self._dim_style = DimStyle(**style_data)
        self._show_dimensions = bool(snap.get("show_dimensions", self._show_dimensions))
        self._reseed_shape_counter()
        self._selected_index = snap.get("selected")
        self._apply_snap_state(snap.get("snap", self._snap_to_grid), emit=False)
        self._grid_size = int(snap.get("grid", self._grid_size))
        self.update()
        self._emit_selection_status()
        self.snap_changed.emit(self._snap_to_grid)
        self.dimensions_visibility_changed.emit(self._show_dimensions)

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
            "dimensions": dims_to_json(self._dimensions),
            "dimension_style": self._dim_style.asdict(),
            "show_dimensions": self._show_dimensions,
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
        if shape.id is not None:
            data["id"] = shape.id
        return data
