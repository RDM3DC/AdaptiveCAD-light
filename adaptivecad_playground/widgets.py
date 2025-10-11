"""Qt widgets for the AdaptiveCAD Playground UI."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from adaptive_fields import AdaptiveMetricManager
from sketch_mode import SketchPlane, lift_polyline
from geometry import (
    arc_length,
    area,
    circle_points,
    curve_area,
    curve_arc_length,
    curve_points,
    polyline_length,
    polygon_area,
    pi_a,
    point_to_polyline_distance,
)
from intersect2d import (
    circle_circle_intersections,
    seg_circle_intersections,
    seg_seg_intersection,
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
from osnap import osnap_pick
from sora_background import BGMode, PanelTheme, SoraBackground, panel_theme_for
from sora_hud import HudTheme, SoraHud
from tools import PiACircleTool, PiACurveTool, SelectTool
from curved_widgets import CurvedViewportOverlay

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

    _OSNAP_GROUPS = [
        ("end", "Endpoints", "Snap to shape endpoints."),
        ("mid", "Midpoints", "Snap to segment midpoints."),
        ("center", "Centers", "Snap to circle centers."),
        ("quadrant", "Quadrants", "Snap to 90° points on circles."),
        ("intersection", "Intersections", "Snap to apparent intersections."),
        ("perp", "Perpendicular", "Snap perpendicular/nearest points on segments."),
        ("dimension", "Dimensions", "Snap to dimension anchors."),
    ]

    def __init__(
        self,
        set_params_cb,
        set_osnap_cb,
        get_osnap_cb,
        set_units_cb,
        get_units_cb,
        set_sketch_plane_cb,
        get_sketch_plane_cb,
    ):
        self._set_params_cb = set_params_cb
        self._set_osnap_cb = set_osnap_cb
        self._get_osnap_cb = get_osnap_cb
        self._set_units_cb = set_units_cb
        self._get_units_cb = get_units_cb
        self._set_sketch_plane_cb = set_sketch_plane_cb
        self._get_sketch_plane_cb = get_sketch_plane_cb
        self.dock = QDockWidget("Parameters")
        self.dock.setObjectName("AdaptiveParamsDock")
        self.dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.dock.setToolTip("Parameters: adjust α, μ, and k₀ to reshape πₐ responses.")
        host = QWidget()
        host.setObjectName("AdaptiveControlsHost")
        layout = QVBoxLayout(host)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._host_widget = host
        self._value_labels: Dict[str, QLabel] = {}
        self._slider_widgets: Dict[str, QSlider] = {}
        self._osnap_checks: Dict[str, QCheckBox] = {}
        self._units_combo: QComboBox = QComboBox()
        self._sketch_plane_combo: QComboBox = QComboBox()
        self._panel_theme: PanelTheme | None = None

        self._add_slider(layout, "alpha", "α", 0, 100, 0)
        self._add_slider(layout, "mu", "μ", 0, 100, 0)
        self._add_slider(layout, "k0", "k₀", -100, 100, 0)

        units_box = QGroupBox("Units")
        units_box.setToolTip("Choose how new dimensions format linear measurements.")
        units_layout = QVBoxLayout()
        units_layout.setContentsMargins(6, 6, 6, 6)
        units_layout.setSpacing(6)
        units_box.setLayout(units_layout)

        self._units_combo.addItem("Inches", "in")
        self._units_combo.addItem("Millimeters", "mm")
        self._units_combo.setToolTip("Switch between inch and millimeter outputs for dimensions.")
        self._units_combo.currentIndexChanged.connect(self._emit_units)

        units_layout.addWidget(self._units_combo)
        layout.addWidget(units_box)

        current_units = str(self._get_units_cb() or "in").lower()
        idx = self._units_combo.findData(current_units)
        if idx < 0:
            idx = self._units_combo.findData("in")
        blocked = self._units_combo.blockSignals(True)
        if idx >= 0:
            self._units_combo.setCurrentIndex(idx)
        else:
            self._units_combo.setCurrentIndex(0)
        self._units_combo.blockSignals(blocked)

        sketch_box = QGroupBox("Sketch Mode")
        sketch_box.setToolTip("Tiny sketch layer: plane selection for lathe/revolve workflows.")
        sketch_layout = QVBoxLayout()
        sketch_layout.setContentsMargins(6, 6, 6, 6)
        sketch_layout.setSpacing(6)
        sketch_box.setLayout(sketch_layout)

        self._sketch_plane_combo.addItem("World XY", SketchPlane.XY.value)
        self._sketch_plane_combo.addItem("World YZ", SketchPlane.YZ.value)
        self._sketch_plane_combo.addItem("World ZX", SketchPlane.ZX.value)
        self._sketch_plane_combo.setToolTip("Choose the sketch plane used when lifting profiles into 3D.")
        self._sketch_plane_combo.currentIndexChanged.connect(self._emit_sketch_plane)

        sketch_layout.addWidget(self._sketch_plane_combo)
        layout.addWidget(sketch_box)

        current_plane = str(self._get_sketch_plane_cb() or SketchPlane.XY.value)
        plane_idx = self._sketch_plane_combo.findData(current_plane)
        blocked_plane = self._sketch_plane_combo.blockSignals(True)
        if plane_idx >= 0:
            self._sketch_plane_combo.setCurrentIndex(plane_idx)
        else:
            self._sketch_plane_combo.setCurrentIndex(0)
        self._sketch_plane_combo.blockSignals(blocked_plane)

        snaps_box = QGroupBox("Object Snaps")
        snaps_layout = QVBoxLayout()
        snaps_layout.setSpacing(4)
        snaps_box.setLayout(snaps_layout)
        snaps_box.setToolTip("Toggle which object snaps are active when drawing.")
        osnap_state = dict(self._get_osnap_cb())
        for key, label, tip in self._OSNAP_GROUPS:
            checkbox = QCheckBox(label)
            checkbox.setToolTip(tip)
            checkbox.setChecked(bool(osnap_state.get(key, True)))
            checkbox.stateChanged.connect(lambda state, k=key: self._handle_osnap_changed(k, bool(state)))
            snaps_layout.addWidget(checkbox)
            self._osnap_checks[key] = checkbox
        layout.addWidget(snaps_box)

        layout.addStretch(1)
        self.dock.setWidget(host)
        self.apply_panel_theme(panel_theme_for(BGMode.GRADIENT))
        self._emit_params()
        self._emit_osnaps()
        self._emit_sketch_plane()

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
        self._emit_params()

    def _handle_osnap_changed(self, key: str, _enabled: bool) -> None:
        # stateChanged sends 0/2; bool(state) collapses to desired truthiness
        if key not in self._osnap_checks:
            return
        self._emit_osnaps()

    def _emit_units(self, *_args) -> None:
        units = self._units_combo.currentData()
        if units is None:
            return
        self._set_units_cb(str(units))

    def _emit_sketch_plane(self, *_args) -> None:
        plane = self._sketch_plane_combo.currentData()
        if plane is None:
            return
        self._set_sketch_plane_cb(str(plane))

    def _emit_params(self) -> None:
        params = {
            "alpha": self._slider_widgets["alpha"].value() / 100.0,
            "mu": self._slider_widgets["mu"].value() / 100.0,
            "k0": self._slider_widgets["k0"].value() / 100.0,
        }
        self._set_params_cb(params)

    def _emit_osnaps(self) -> None:
        states = {key: checkbox.isChecked() for key, checkbox in self._osnap_checks.items()}
        self._set_osnap_cb(states)

    def set_panel_mode(self, mode: BGMode) -> None:
        self.apply_panel_theme(panel_theme_for(mode))

    def apply_panel_theme(self, theme: PanelTheme) -> None:
        if theme is None:
            return
        if self._panel_theme == theme:
            return
        self._panel_theme = theme

        top = theme.top.name()
        bottom = theme.bottom.name()
        text = theme.text.name()
        border = theme.border.name()
        accent = theme.accent.name()
        gradient = f"qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bottom})"

        host_lines = [
            "#AdaptiveControlsHost {",
            f"    background: {gradient};",
            f"    color: {text};",
            "}",
            "#AdaptiveControlsHost QLabel,",
            "#AdaptiveControlsHost QCheckBox,",
            "#AdaptiveControlsHost QComboBox,",
            "#AdaptiveControlsHost QGroupBox,",
            "#AdaptiveControlsHost QSlider {",
            f"    color: {text};",
            "}",
            "#AdaptiveControlsHost QGroupBox {",
            f"    border: 1px solid {border};",
            "    margin-top: 10px;",
            "}",
            "#AdaptiveControlsHost QGroupBox::title {",
            f"    color: {text};",
            "    subcontrol-origin: margin;",
            "    left: 8px;",
            "    padding: 0 4px;",
            "}",
            "#AdaptiveControlsHost QCheckBox::indicator {",
            "    width: 14px;",
            "    height: 14px;",
            f"    border: 1px solid {border};",
            "    background: rgba(255, 255, 255, 25);",
            "}",
            "#AdaptiveControlsHost QCheckBox::indicator:checked {",
            f"    background: {accent};",
            "}",
            "#AdaptiveControlsHost QSlider::groove:horizontal {",
            f"    border: 1px solid {border};",
            "    height: 6px;",
            "    background: rgba(0, 0, 0, 60);",
            "}",
            "#AdaptiveControlsHost QSlider::handle:horizontal {",
            "    width: 14px;",
            "    margin: -4px 0;",
            f"    background: {accent};",
            f"    border: 1px solid {border};",
            "}",
            "#AdaptiveControlsHost QComboBox {",
            f"    border: 1px solid {border};",
            "    padding: 2px 6px;",
            "    background-color: rgba(255, 255, 255, 18);",
            "}",
        ]
        self._host_widget.setStyleSheet("\n".join(host_lines))

        dock_lines = [
            "QDockWidget#AdaptiveParamsDock {",
            f"    background: {gradient};",
            f"    color: {text};",
            "}",
            "QDockWidget#AdaptiveParamsDock::title {",
            f"    background: {top};",
            f"    color: {text};",
            "    padding-left: 8px;",
            "}",
        ]
        self.dock.setStyleSheet("\n".join(dock_lines))


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

        self._params: Params = {"alpha": 0.0, "mu": 0.0, "k0": 0.0}
        self._shapes: List[Shape] = []
        self._selection: List[str] = []
        self._selected_dimensions: List[str] = []
        self._temp_shape: Optional[Shape] = None
        self._shape_counter: int = 1

        self._metrics = AdaptiveMetricManager()
        self._preview_executor = ThreadPoolExecutor(max_workers=2)
        self._preview_future: Future | None = None
        self._preview_token: int = 0
        self._sketch_plane = SketchPlane.XY

        # Dimensions
        self._dimensions: List[object] = []
        self._show_dimensions: bool = True
        self._dim_style: DimStyle = DimStyle()

        self._tool_name: Optional[str] = None
        self._tool_cache = {}
        self._tool = None
        self._pointer_tool: object | None = None
        self._pointer_tool_name: Optional[str] = None
        self._dynamic_tools = {"dim_linear", "dim_radial", "dim_angular", "measure"}
        self._selection_drag_origin: Optional[Point] = None
        self._selection_rect: Optional[Tuple[float, float, float, float]] = None
        # Snap and grid
        self._snap_to_grid = True
        self._grid_size = 20
        self._osnap_tolerance = 12.0
        self._last_osnap: Optional[Tuple[str, Point]] = None
        self._osnap_groups: Dict[str, set[str]] = {
            "end": {"end"},
            "mid": {"mid"},
            "center": {"center"},
            "quadrant": {"quad", "90"},
            "intersection": {"appint"},
            "perp": {"perp", "closest"},
            "dimension": {"dim"},
        }
        self._enabled_osnap_groups: set[str] = set(self._osnap_groups.keys())
        self._osnap_kind_to_group: Dict[str, str] = {}
        for group, kinds in self._osnap_groups.items():
            for kind in kinds:
                self._osnap_kind_to_group[kind] = group
        # Viewport (pan/zoom)
        self._view_origin: List[float] = [0.0, 0.0]
        self._view_scale: float = 1.0
        self._view_min_scale: float = 0.2
        self._view_max_scale: float = 5.0

        # History stacks
        self._undo_stack = []
        self._redo_stack = []
        self._status_state: Dict[str, Optional[float | str]] = {}

        # Visual layers
        self.bg = SoraBackground()
        self.bg.set_mode(BGMode.GRID_SCAN)
        self.hud = SoraHud(HudTheme())
        self.hud.set_status("SORA OFF")
        self.hud.set_hints(
            [
                ("Tip: Press L for Linear Dim", 0.05),
                ("Hold Shift to snap", 0.28),
                ("PiA mode: Compare labels", 0.66),
            ]
        )
        self._anim_start = time.monotonic()
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self.update)
        self._anim_timer.start()
        self._load_visual_config()

        self._curved_ui_enabled = True
        self.overlay = CurvedViewportOverlay(self, get_params=self.params)
        self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()
        self.overlay.show()
        self._overlay_last_sample = None  # type: Optional[tuple[float, float, float]]

        self._emit_default_status()
        self.dimensions_visibility_changed.emit(self._show_dimensions)
        self._rebuild_adaptive_fields()

    def _load_visual_config(self) -> None:
        config_path = Path(__file__).with_name("demo_background_config.json")
        if not config_path.exists():
            return
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return

        hud_enabled = data.get("hud_enabled")
        if hud_enabled is not None:
            self.hud.set_enabled(bool(hud_enabled))

        hud_status = data.get("hud_status")
        if isinstance(hud_status, str) and hud_status.strip():
            self.hud.set_status(hud_status)

        hints = data.get("hints")
        if isinstance(hints, list):
            parsed: List[Tuple[str, float]] = []
            for entry in hints:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    try:
                        parsed.append((str(entry[0]), float(entry[1])))
                    except (ValueError, TypeError):
                        continue
            if parsed:
                self.hud.set_hints(parsed)

        mode_name = data.get("background_mode")
        if isinstance(mode_name, str):
            try:
                self.bg.set_mode(BGMode[mode_name.upper()])
            except KeyError:
                pass

    # ------------------------------------------------------------------
    # Parameter management
    def set_params(self, params: Params) -> None:
        self._params = dict(params)

    def params(self) -> Params:
        return dict(self._params)

    def units(self) -> str:
        return str(self._dim_style.units or "in")

    def sketch_plane(self) -> str:
        return self._sketch_plane.value

    def set_sketch_plane(self, plane: str | SketchPlane) -> None:
        try:
            target = plane if isinstance(plane, SketchPlane) else SketchPlane(str(plane).upper())
        except ValueError:
            return
        if self._sketch_plane == target:
            return
        self._sketch_plane = target
        self.post_status_message(f"Sketch plane set to {target.value}")

    def set_units(self, units: str) -> None:
        normalized = (units or "").strip().lower()
        if normalized not in {"in", "mm"}:
            return
        if self.units() == normalized:
            return
        self._push_history()
        self.cancel_curve_preview()
        self._dim_style.units = normalized
        for dim in self._dimensions:
            style = getattr(dim, "style", None)
            if isinstance(style, DimStyle):
                style.units = normalized
            elif hasattr(dim, "style") and hasattr(self._dim_style, "asdict"):
                copied = DimStyle(**self._dim_style.asdict())
                copied.units = normalized
                dim.style = copied
        message = "Units set to millimeters" if normalized == "mm" else "Units set to inches"
        self.post_status_message(message)
        self.update()

    def _rebuild_adaptive_fields(self) -> None:
        self._metrics.rebuild(self._shapes)

    def curvature_at(self, point: Point) -> float:
        return self._metrics.curvature(point)

    def memory_at(self, point: Point) -> float:
        return self._metrics.memory(point)

    def pi_a_at(self, point: Point, radius: float, params: Optional[Params] = None) -> float:
        return self._metrics.pi_a(point, radius, params or self._params)

    def adaptive_segment_length(
        self,
        p0: Point,
        p1: Point,
        params: Optional[Params] = None,
        *,
        samples: int = 16,
    ) -> float:
        return self._metrics.segment_length(p0, p1, params or self._params, samples=samples)

    def adaptive_polyline_length(
        self,
        points: Sequence[Point],
        params: Optional[Params] = None,
        *,
        samples_per_segment: int = 16,
    ) -> float:
        return self._metrics.polyline_length(points, params or self._params, samples_per_segment=samples_per_segment)

    def adaptive_curve_length(
        self,
        control_points: Sequence[Point],
        params: Optional[Params] = None,
        *,
        samples: int = 256,
        samples_per_segment: int = 16,
    ) -> float:
        return self._metrics.curve_length(
            control_points,
            params or self._params,
            samples=samples,
            samples_per_segment=samples_per_segment,
        )

    def adaptive_circle_circumference(
        self,
        center: Point,
        radius: float,
        params: Optional[Params] = None,
        *,
        samples: int = 512,
        samples_per_segment: int = 8,
    ) -> float:
        return self._metrics.circle_circumference(
            center,
            radius,
            params or self._params,
            samples=samples,
            samples_per_segment=samples_per_segment,
        )

    # ------------------------------------------------------------------
    # View transforms
    def world_to_screen(self, point: Point) -> Point:
        ox, oy = self._view_origin
        scale = self._view_scale
        return ((point[0] - ox) * scale, (point[1] - oy) * scale)

    def screen_to_world(self, point: Point) -> Point:
        ox, oy = self._view_origin
        scale = self._view_scale
        return (ox + point[0] / scale, oy + point[1] / scale)

    def _visible_world_rect(self) -> Tuple[float, float, float, float]:
        top_left = self.screen_to_world((0.0, 0.0))
        bottom_right = self.screen_to_world((float(self.width()), float(self.height())))
        x0 = min(top_left[0], bottom_right[0])
        x1 = max(top_left[0], bottom_right[0])
        y0 = min(top_left[1], bottom_right[1])
        y1 = max(top_left[1], bottom_right[1])
        return (x0, y0, x1, y1)

    def _apply_zoom(self, target_scale: float, pivot_screen: Optional[Point]) -> None:
        target_scale = max(self._view_min_scale, min(self._view_max_scale, target_scale))
        if abs(target_scale - self._view_scale) <= 1e-9:
            return
        if pivot_screen is None:
            pivot_screen = (self.width() / 2.0, self.height() / 2.0)
        pivot_world = self.screen_to_world(pivot_screen)
        self._view_scale = target_scale
        self._view_origin[0] = pivot_world[0] - pivot_screen[0] / target_scale
        self._view_origin[1] = pivot_world[1] - pivot_screen[1] / target_scale
        self.update()

    def zoom_by(self, factor: float, pivot_screen: Optional[Point] = None) -> None:
        self._apply_zoom(self._view_scale * factor, pivot_screen)

    def zoom_reset(self) -> None:
        self._view_scale = 1.0
        self._view_origin = [0.0, 0.0]
        self.update()

    def _handle_zoom_shortcut(self, event) -> bool:
        modifiers = event.modifiers()
        if not (modifiers & Qt.ControlModifier):
            return False
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.2)
            event.accept()
            return True
        if key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_by(1.0 / 1.2)
            event.accept()
            return True
        if key == Qt.Key_0:
            self.zoom_reset()
            event.accept()
            return True
        return False

    def osnap_groups(self) -> Dict[str, bool]:
        return {name: name in self._enabled_osnap_groups for name in self._osnap_groups}

    def set_osnap_groups(self, states: Dict[str, bool]) -> None:
        updated = set(self._enabled_osnap_groups)
        for name, enabled in states.items():
            if name not in self._osnap_groups:
                continue
            if enabled:
                updated.add(name)
            else:
                updated.discard(name)
        if updated == self._enabled_osnap_groups:
            return
        self._enabled_osnap_groups = updated
        if self._last_osnap is not None:
            self._last_osnap = None
        self.update()

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
    # Visual layers (HUD + Background)
    def hud_enabled(self) -> bool:
        return self.hud.is_enabled()

    def set_hud_enabled(self, enabled: bool) -> None:
        target = bool(enabled)
        if self.hud.is_enabled() == target:
            return
        self.hud.set_enabled(target)
        self.update()

    def set_hud_status(self, status: str) -> None:
        self.hud.set_status(status)
        self.update()

    def hud_status(self) -> str:
        return self.hud.status()

    def set_hud_hints(self, hints: Sequence[Tuple[str, float]]) -> None:
        self.hud.set_hints(hints)
        self.update()

    def background_mode(self) -> BGMode:
        return self.bg.mode()

    def set_background_mode(self, mode: BGMode) -> None:
        if not isinstance(mode, BGMode):
            raise TypeError("mode must be an instance of BGMode.")
        if self.bg.mode() is mode:
            return
        self.bg.set_mode(mode)
        self.update()

    def curved_ui_enabled(self) -> bool:
        return bool(self._curved_ui_enabled)

    def set_curved_ui_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._curved_ui_enabled == enabled:
            return
        self._curved_ui_enabled = enabled
        overlay = getattr(self, "overlay", None)
        if overlay is not None:
            overlay.setVisible(enabled)
            if enabled:
                overlay.raise_()
        if not enabled:
            self._overlay_last_sample = None
        self.update()

    # ------------------------------------------------------------------
    # Curved UI overlay helpers
    def _nudge_overlay_activity(self, input_energy: float) -> None:
        if not self._curved_ui_enabled:
            return
        overlay = getattr(self, "overlay", None)
        if overlay is None or not overlay.isVisible():
            return
        overlay.adapt(input_energy=max(0.0, min(1.0, float(input_energy))))

    def _update_overlay_activity(self, event) -> None:
        if not self._curved_ui_enabled:
            return
        overlay = getattr(self, "overlay", None)
        if overlay is None or not overlay.isVisible():
            return
        pos = event.position() if hasattr(event, "position") else event.posF()
        now = time.monotonic()
        x = float(pos.x()) if hasattr(pos, "x") else float(pos[0])
        y = float(pos.y()) if hasattr(pos, "y") else float(pos[1])
        last_sample = self._overlay_last_sample
        if last_sample is not None:
            prev_time, px, py = last_sample
            dt = max(now - prev_time, 1e-3)
            speed = math.hypot(x - px, y - py) / dt
            # Map pixel-per-second speed into [0,1]
            energy = max(0.0, min(1.0, speed / 600.0))
        else:
            energy = 0.2
        overlay.adapt(input_energy=energy)
        self._overlay_last_sample = (now, x, y)

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
        x, y = float(pt[0]), float(pt[1])
        tol_world = float(self._osnap_tolerance) / max(self._view_scale, 1e-9)
        snapped, kind = osnap_pick((x, y), self.osnap_targets(), tol=tol_world)
        snapped_pt = (float(snapped[0]), float(snapped[1]))
        state_changed = False
        if kind:
            if self._last_osnap != (kind, snapped_pt):
                self._last_osnap = (kind, snapped_pt)
                state_changed = True
        else:
            if self._last_osnap is not None:
                self._last_osnap = None
                state_changed = True
        if state_changed:
            self.update()
        if kind:
            return snapped_pt
        if not self._snap_to_grid:
            return (x, y)
        g = float(self._grid_size)
        return (round(x / g) * g, round(y / g) * g)

    def world_from_event(self, event) -> Point:
        screen_pt = (event.position().x(), event.position().y())
        world_pt = self.screen_to_world(screen_pt)
        return self.snap_point(world_pt)

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
        if name != "select":
            self.clear_pointer_tool()
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
        self._nudge_overlay_activity(0.35)
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
            world_from_event=self.world_from_event,
            osnap_targets=self.osnap_targets,
            sample_curvature=self.curvature_at,
            sample_memory=self.memory_at,
            sample_pi_a=lambda pt, radius, params=None: self.pi_a_at(pt, radius, params),
            adaptive_segment_length=lambda a, b, params=None: self.adaptive_segment_length(a, b, params),
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

    def shape_from_curve(
        self,
        control_points: Sequence[Point],
        params: Optional[Params] = None,
        *,
        samples: int = 256,
        prefer_gpu: Optional[bool] = None,
    ) -> Shape:
        params = dict(params or self._params)
        ctrl = [(float(pt[0]), float(pt[1])) for pt in control_points]
        use_gpu = self._metrics.gpu_enabled() if prefer_gpu is None else bool(prefer_gpu)
        device_hint = self._metrics.gpu_device_hint() if use_gpu else None
        curve_pts = curve_points(
            ctrl,
            params,
            samples=samples,
            prefer_gpu=use_gpu,
            gpu_device=device_hint,
        )
        return Shape(
            type="piacurve",
            params=params,
            points=curve_pts,
            control_points=ctrl,
        )

    def sketch_profile_world(self, control_points: Sequence[Point]) -> np.ndarray:
        """Lift 2D sketch points into world space using the active sketch plane."""

        return lift_polyline(control_points, self._sketch_plane)

    def cancel_curve_preview(self) -> None:
        self._preview_token += 1
        if self._preview_future is not None and not self._preview_future.done():
            self._preview_future.cancel()
        self._preview_future = None

    def request_curve_preview(self, control_points: Sequence[Point]) -> None:
        ctrl = [(float(pt[0]), float(pt[1])) for pt in control_points]
        if len(ctrl) < 2:
            self.cancel_curve_preview()
            self.clear_temp_shape()
            return

        params = dict(self._params)
        quick_pts = curve_points(ctrl, params, samples=64, prefer_gpu=False)
        quick_shape = Shape(
            type="piacurve",
            params=params,
            points=quick_pts,
            control_points=ctrl,
        )
        self.set_temp_shape(quick_shape, show_status=True)

        self._preview_token += 1
        token = self._preview_token
        if self._preview_future is not None and not self._preview_future.done():
            self._preview_future.cancel()

        prefer_gpu = self._metrics.gpu_enabled()
        device_hint = self._metrics.gpu_device_hint() if prefer_gpu else None

        def task() -> np.ndarray:
            return curve_points(
                ctrl,
                params,
                samples=256,
                prefer_gpu=prefer_gpu,
                gpu_device=device_hint,
            )

        future = self._preview_executor.submit(task)
        self._preview_future = future

        def _on_done(fut: Future) -> None:
            if fut.cancelled():
                return
            try:
                high_res = fut.result()
            except Exception:
                return

            def apply() -> None:
                if token != self._preview_token:
                    return
                shape = Shape(
                    type="piacurve",
                    params=params,
                    points=high_res,
                    control_points=ctrl,
                )
                self._preview_future = None
                self.set_temp_shape(shape, show_status=False)

            QTimer.singleShot(0, apply)

        future.add_done_callback(_on_done)

    def _next_shape_id(self) -> str:
        identifier = f"S{self._shape_counter:04d}"
        self._shape_counter += 1
        return identifier

    def add_circle(self, center: Point, radius: float) -> None:
        self._push_history()
        shape = self.shape_from_circle(center, radius, self._params)
        shape.id = self._next_shape_id()
        self._shapes.append(shape)
        self._rebuild_adaptive_fields()
        self._set_selection([shape.id])
        self.update()

    def add_curve(self, control_points: Sequence[Point]) -> None:
        if len(control_points) < 2:
            return
        self._push_history()
        shape = self.shape_from_curve(control_points, self._params)
        shape.id = self._next_shape_id()
        self._shapes.append(shape)
        self._rebuild_adaptive_fields()
        self._set_selection([shape.id])
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

    def closeEvent(self, event):  # pragma: no cover - GUI entry point
        try:
            self.cancel_curve_preview()
            self._preview_executor.shutdown(wait=False)
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Selection & status helpers
    def _hit_test_shape(self, point: Point, threshold: float = 12.0) -> Optional[str]:
        best_shape: Optional[Shape] = None
        best_distance = float("inf")
        for shape in self._shapes:
            if shape.points is None or len(shape.points) < 2:
                continue
            dist = point_to_polyline_distance(point, shape.points)
            if dist < best_distance:
                best_distance = dist
                best_shape = shape
        if best_shape is None or best_distance > threshold:
            return None
        if not best_shape.id:
            best_shape.id = self._next_shape_id()
        return best_shape.id

    def select_shape_at(self, point: Point, *, additive: bool = False, toggle: bool = False) -> None:
        self.clear_selection_rect()
        world_pt = self.screen_to_world(point)
        shape_id = self._hit_test_shape(world_pt)
        if shape_id is None:
            if not additive and not toggle:
                self._apply_selection_results([], [], additive=False, toggle=False)
            return
        self._apply_selection_results([shape_id], [], additive=additive, toggle=toggle)

    def select_items_at(self, point: Point, *, additive: bool = False, toggle: bool = False) -> None:
        self.clear_selection_rect()
        world_pt = self.screen_to_world(point)
        shape_id = self._hit_test_shape(world_pt)
        if shape_id:
            self._apply_selection_results([shape_id], [], additive=additive, toggle=toggle)
            return
        dim_id = self._dimension_hit_test(world_pt)
        if dim_id:
            self._apply_selection_results([], [dim_id], additive=additive, toggle=toggle)
            return
        if not additive and not toggle:
            self._apply_selection_results([], [], additive=False, toggle=False)

    def begin_selection_rect(self, origin: Point) -> None:
        self._selection_drag_origin = origin
        self._selection_rect = None

    def update_selection_rect(self, current: Point) -> None:
        if self._selection_drag_origin is None:
            return
        x0, y0 = self._selection_drag_origin
        x1, y1 = current
        rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self._selection_rect = rect
        self.update()

    def finalize_selection_rect(self) -> Optional[Tuple[float, float, float, float]]:
        rect = self._selection_rect
        self._selection_rect = None
        self._selection_drag_origin = None
        self.update()
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        if abs(x1 - x0) < 1e-3 or abs(y1 - y0) < 1e-3:
            return None
        world_a = self.screen_to_world((x0, y0))
        world_b = self.screen_to_world((x1, y1))
        return (world_a[0], world_a[1], world_b[0], world_b[1])

    def clear_selection_rect(self) -> None:
        if self._selection_rect is None and self._selection_drag_origin is None:
            return
        self._selection_rect = None
        self._selection_drag_origin = None
        self.update()

    def select_shapes_in_rect(
        self,
        rect: Tuple[float, float, float, float],
        *,
        additive: bool = False,
        toggle: bool = False,
    ) -> None:
        x0, y0, x1, y1 = rect
        x_min, x_max = (min(x0, x1), max(x0, x1))
        y_min, y_max = (min(y0, y1), max(y0, y1))
        hits: List[str] = []
        for shape in self._shapes:
            if shape.points is None or len(shape.points) < 2:
                continue
            if not shape.id:
                shape.id = self._next_shape_id()
            pts = shape.points
            shape_min_x = float(np.min(pts[:, 0]))
            shape_max_x = float(np.max(pts[:, 0]))
            shape_min_y = float(np.min(pts[:, 1]))
            shape_max_y = float(np.max(pts[:, 1]))
            if shape_max_x < x_min or shape_min_x > x_max or shape_max_y < y_min or shape_min_y > y_max:
                continue
            hits.append(shape.id)
        dim_hits = self._dimensions_in_rect(rect)
        if not hits and not dim_hits and not additive and not toggle:
            self._apply_selection_results([], [], additive=False, toggle=False)
            return
        self._apply_selection_results(hits, dim_hits, additive=additive, toggle=toggle)

    def select_items_in_rect(
        self,
        rect: Tuple[float, float, float, float],
        *,
        additive: bool = False,
        toggle: bool = False,
    ) -> None:
        self.select_shapes_in_rect(rect, additive=additive, toggle=toggle)

    def osnap_targets(self) -> List[Tuple[str, object]]:
        point_targets: List[Tuple[str, Point]] = []
        segments: List[Tuple[Point, Point]] = []
        circles: List[Tuple[Point, float]] = []

        allow_perp = "perp" in self._enabled_osnap_groups
        allow_intersections = "intersection" in self._enabled_osnap_groups

        def kind_enabled(kind: str) -> bool:
            group = self._osnap_kind_to_group.get(kind)
            if group is None:
                return True
            return group in self._enabled_osnap_groups

        def add_point(kind: str, pt: Point) -> None:
            if not kind_enabled(kind):
                return
            px, py = float(pt[0]), float(pt[1])
            key = (kind, round(px * 1e4), round(py * 1e4))
            if key in seen_points:
                return
            seen_points.add(key)
            point_targets.append((kind, (px, py)))

        def add_segment(a: Point, b: Point) -> None:
            if not (allow_perp or allow_intersections):
                return
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            if (abs(ax - bx) < 1e-9 and abs(ay - by) < 1e-9):
                return
            segments.append(((ax, ay), (bx, by)))

        seen_points: set[Tuple[str, int, int]] = set()

        for shape in self._shapes:
            if shape.type == "piacircle" and shape.center is not None and shape.radius:
                cx, cy = float(shape.center[0]), float(shape.center[1])
                r = float(shape.radius)
                add_point("center", (cx, cy))
                if allow_perp or allow_intersections:
                    circles.append(((cx, cy), r))
                for pt in [
                    (cx + r, cy),
                    (cx - r, cy),
                    (cx, cy + r),
                    (cx, cy - r),
                ]:
                    add_point("quad", pt)
                    add_point("90", pt)
            elif shape.points is not None and len(shape.points) >= 2:
                pts = [tuple(map(float, pt)) for pt in shape.points.tolist()]
                add_point("end", pts[0])
                add_point("end", pts[-1])
                for idx in range(len(pts) - 1):
                    a = pts[idx]
                    b = pts[idx + 1]
                    add_segment(a, b)
                    if shape.type != "piacurve":
                        mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
                        add_point("mid", mid)

                if shape.type == "piacurve":
                    if kind_enabled("mid"):
                        # Find the point halfway along the sampled curve length.
                        total = 0.0
                        seg_lengths: List[float] = []
                        for idx in range(len(pts) - 1):
                            a = pts[idx]
                            b = pts[idx + 1]
                            seg = math.hypot(b[0] - a[0], b[1] - a[1])
                            seg_lengths.append(seg)
                            total += seg
                        if total > 0.0:
                            target = total * 0.5
                            accum = 0.0
                            mid_pt = pts[0]
                            for idx, seg in enumerate(seg_lengths):
                                next_accum = accum + seg
                                if next_accum >= target:
                                    if seg > 0.0:
                                        t = (target - accum) / seg
                                        a = pts[idx]
                                        b = pts[idx + 1]
                                        mid_pt = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
                                    else:
                                        mid_pt = pts[idx]
                                    break
                                accum = next_accum
                            else:
                                mid_pt = pts[len(pts) // 2]
                            add_point("mid", mid_pt)

        for dim in self._dimensions:
            if isinstance(dim, LinearDimension):
                add_point("dim", dim.p1)
                add_point("dim", dim.p2)
                add_point("dim", dim.offset)
                add_segment(dim.p1, dim.p2)
            elif isinstance(dim, RadialDimension):
                add_point("dim", dim.center)
                add_point("dim", dim.attach)
                add_segment(dim.center, dim.attach)
            elif isinstance(dim, AngularDimension):
                add_point("dim", dim.vtx)
                add_point("dim", dim.p1)
                add_point("dim", dim.p2)
                add_point("dim", dim.attach)
                add_segment(dim.vtx, dim.p1)
                add_segment(dim.vtx, dim.p2)

        # Apparent intersections between segments and circles
        if allow_intersections:
            for i in range(len(segments)):
                a0, a1 = segments[i]
                for j in range(i + 1, len(segments)):
                    b0, b1 = segments[j]
                    pt = seg_seg_intersection(a0, a1, b0, b1)
                    if pt is not None:
                        add_point("appint", pt)

            for seg in segments:
                for circle in circles:
                    for pt in seg_circle_intersections(seg[0], seg[1], circle[0], circle[1]):
                        add_point("appint", pt)

            for i in range(len(circles)):
                for j in range(i + 1, len(circles)):
                    c0, r0 = circles[i]
                    c1, r1 = circles[j]
                    for pt in circle_circle_intersections(c0, r0, c1, r1):
                        add_point("appint", pt)

        out: List[Tuple[str, object]] = []
        out.extend(point_targets)
        if allow_perp:
            out.extend(("segment", seg) for seg in segments)
            out.extend(("circle", circ) for circ in circles)
        return out

    def hit_test_circle(self, point: Point) -> Optional[Tuple[Point, float, str]]:
        best: Optional[Tuple[Point, float, str]] = None
        best_error = float("inf")
        tol = 12.0 / max(self._view_scale, 1e-6)
        for shape in self._shapes:
            if shape.type != "piacircle" or shape.center is None or shape.radius is None:
                continue
            cx, cy = shape.center
            radius = float(shape.radius)
            error = abs(math.hypot(point[0] - cx, point[1] - cy) - radius)
            if error < best_error and error <= tol:
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

    def _dimension_by_id(self, dimension_id: Optional[str]):
        if not dimension_id:
            return None
        for dim in self._dimensions:
            if getattr(dim, "id", None) == dimension_id:
                return dim
        return None

    def _dimension_bounds(self, dimension) -> Tuple[float, float, float, float]:
        pts: List[Point] = []
        if isinstance(dimension, LinearDimension):
            pts = [dimension.p1, dimension.p2, dimension.offset]
        elif isinstance(dimension, RadialDimension):
            pts = [dimension.center, dimension.attach]
        elif isinstance(dimension, AngularDimension):
            pts = [dimension.vtx, dimension.p1, dimension.p2, dimension.attach]
        else:
            # Fallback: try generic attributes
            for name in ("p1", "p2", "offset", "center", "attach", "vtx"):
                value = getattr(dimension, name, None)
                if isinstance(value, tuple) and len(value) == 2:
                    pts.append(value)
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [float(pt[0]) for pt in pts]
        ys = [float(pt[1]) for pt in pts]
        padding = 8.0
        return (
            min(xs) - padding,
            min(ys) - padding,
            max(xs) + padding,
            max(ys) + padding,
        )

    def _dimension_hit_test(self, point: Point) -> Optional[str]:
        x, y = point
        best_id: Optional[str] = None
        best_area = float("inf")
        for dim in self._dimensions:
            dim_id = getattr(dim, "id", None)
            if not dim_id:
                continue
            x0, y0, x1, y1 = self._dimension_bounds(dim)
            if x < x0 or x > x1 or y < y0 or y > y1:
                continue
            area = (x1 - x0) * (y1 - y0)
            if area < best_area:
                best_area = area
                best_id = dim_id
        return best_id

    def _dimensions_in_rect(self, rect: Tuple[float, float, float, float]) -> List[str]:
        x0, y0, x1, y1 = rect
        x_min, x_max = (min(x0, x1), max(x0, x1))
        y_min, y_max = (min(y0, y1), max(y0, y1))
        hits: List[str] = []
        for dim in self._dimensions:
            dim_id = getattr(dim, "id", None)
            if not dim_id:
                continue
            dx0, dy0, dx1, dy1 = self._dimension_bounds(dim)
            if dx1 < x_min or dx0 > x_max or dy1 < y_min or dy0 > y_max:
                continue
            hits.append(dim_id)
        return hits

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
            "curvature": None,
            "memory": None,
            "message": "",
        }
        self.status_changed.emit(dict(self._status_state))

    def _emit_selection_status(self) -> None:
        shape_count = len(self._selection)
        dim_count = len(self._selected_dimensions)
        if shape_count == 0 and dim_count == 0:
            self._emit_default_status()
            return
        if shape_count == 1 and dim_count == 0:
            shape = self._shape_by_id(self._selection[0])
            if shape is not None:
                self.emit_status_from_shape(shape)
                return
        parts = []
        if shape_count:
            parts.append(f"{shape_count} shape{'s' if shape_count != 1 else ''}")
        if dim_count:
            parts.append(f"{dim_count} dimension{'s' if dim_count != 1 else ''}")
        message = ", ".join(parts) + " selected"
        self._emit_status(
            {
                "kind": None,
                "radius": None,
                "pi_a": None,
                "arc_length": None,
                "area": None,
                "curvature": None,
                "memory": None,
                "message": message,
            }
        )

    def _set_selection(self, ids: Sequence[Optional[str]]) -> None:
        filtered: List[str] = []
        seen = set()
        for sid in ids:
            if sid is None:
                continue
            sid_str = sid if isinstance(sid, str) else str(sid)
            if sid_str in seen:
                continue
            shape = self._shape_by_id(sid_str)
            if shape is None:
                continue
            filtered.append(sid_str)
            seen.add(sid_str)
        if filtered == self._selection:
            self._emit_selection_status()
            return
        self._selection = filtered
        self._emit_selection_status()
        self.update()

    def _set_dimension_selection(self, ids: Sequence[Optional[str]]) -> None:
        filtered: List[str] = []
        seen = set()
        for did in ids:
            if did is None:
                continue
            did_str = did if isinstance(did, str) else str(did)
            if did_str in seen:
                continue
            dim = self._dimension_by_id(did_str)
            if dim is None:
                continue
            filtered.append(did_str)
            seen.add(did_str)
        if filtered == self._selected_dimensions:
            self._emit_selection_status()
            return
        self._selected_dimensions = filtered
        self._emit_selection_status()
        self.update()

    def _apply_selection_results(
        self,
        shape_hits: Sequence[str],
        dimension_hits: Sequence[str],
        *,
        additive: bool = False,
        toggle: bool = False,
    ) -> None:
        shape_hits = [sid for sid in shape_hits if sid]
        dim_hits = [did for did in dimension_hits if did]
        shape_order = [shape.id for shape in self._shapes if shape.id]
        dim_order = [getattr(dim, "id", None) for dim in self._dimensions if getattr(dim, "id", None)]

        if toggle:
            shape_set = set(self._selection)
            for sid in shape_hits:
                if sid in shape_set:
                    shape_set.remove(sid)
                else:
                    shape_set.add(sid)
            dim_set = set(self._selected_dimensions)
            for did in dim_hits:
                if did in dim_set:
                    dim_set.remove(did)
                else:
                    dim_set.add(did)
            self._set_selection([sid for sid in shape_order if sid in shape_set])
            self._set_dimension_selection([did for did in dim_order if did in dim_set])
            return

        if additive:
            shape_set = set(self._selection)
            shape_set.update(shape_hits)
            dim_set = set(self._selected_dimensions)
            dim_set.update(dim_hits)
            self._set_selection([sid for sid in shape_order if sid in shape_set])
            self._set_dimension_selection([did for did in dim_order if did in dim_set])
            return

        target_shape_ids = set(shape_hits)
        target_dim_ids = set(dim_hits)
        self._set_selection([sid for sid in shape_order if sid in target_shape_ids])
        self._set_dimension_selection([did for did in dim_order if did in target_dim_ids])

    def get_selection(self) -> List[dict]:
        payloads: List[dict] = []
        for sid in self._selection:
            shape = self._shape_by_id(sid)
            if shape is None:
                continue
            payloads.append(self._shape_payload(shape))
        return payloads

    def get_selected_dimension_ids(self) -> List[str]:
        return list(self._selected_dimensions)

    def set_selection(self, payloads: Sequence[dict]) -> None:
        ids: List[str] = []
        for item in payloads:
            sid = item.get("id")
            if not sid:
                continue
            ids.append(str(sid))
        self._set_selection(ids)

    def update_scene(self, add_list: Sequence[dict], remove_list: Sequence[dict]) -> None:
        if not add_list and not remove_list:
            return
        self._push_history()
        prior_selection = list(self._selection)
        remove_ids = {str(item["id"]) for item in remove_list if item.get("id")}
        if remove_ids:
            self._shapes = [shape for shape in self._shapes if shape.id not in remove_ids]
        new_ids: List[str] = []
        for payload in add_list:
            shape = self._shape_from_payload(payload)
            if shape.id and any(existing.id == shape.id for existing in self._shapes):
                shape.id = None
            if not shape.id:
                shape.id = self._next_shape_id()
            self._shapes.append(shape)
            new_ids.append(shape.id)
        target_selection = new_ids if new_ids else [sid for sid in prior_selection if sid not in remove_ids]
        self._reseed_shape_counter()
        self._rebuild_adaptive_fields()
        self.update()
        self._set_selection(target_selection)

    def delete_items(
        self,
        remove_shapes: Sequence[dict],
        remove_dimensions: Sequence[str],
    ) -> Tuple[int, int]:
        shape_ids = {str(item["id"]) for item in remove_shapes if item and item.get("id")}
        dimension_ids = {str(did) for did in remove_dimensions if did}
        if not shape_ids and not dimension_ids:
            return (0, 0)
        self._push_history()
        if shape_ids:
            self._shapes = [shape for shape in self._shapes if shape.id not in shape_ids]
        if dimension_ids:
            self._dimensions = [dim for dim in self._dimensions if getattr(dim, "id", None) not in dimension_ids]
        self._reseed_shape_counter()
        self._rebuild_adaptive_fields()
        self.update()
        remaining_shapes = [sid for sid in self._selection if sid not in shape_ids]
        remaining_dims = [did for did in self._selected_dimensions if did not in dimension_ids]
        self._set_selection(remaining_shapes)
        self._set_dimension_selection(remaining_dims)
        return (len(shape_ids), len(dimension_ids))

    def _shape_payload(self, shape: Shape) -> dict:
        data = self._serialize_shape(shape)
        data["id"] = shape.id
        data["source_type"] = shape.type
        if shape.type == "piacircle":
            data["type"] = "piacircle"
            if shape.center is not None:
                data["center"] = [float(shape.center[0]), float(shape.center[1])]
            if shape.radius is not None:
                data["radius"] = float(shape.radius)
        elif shape.type == "piacurve":
            data["type"] = "curve"
            if shape.control_points:
                data["control_points"] = [list(pt) for pt in shape.control_points]
        elif shape.type == "polyline":
            data["type"] = "polyline"
        else:
            data["type"] = shape.type
        return data

    def _shape_from_payload(self, payload: dict) -> Shape:
        params = dict(payload.get("params") or self._params)
        payload_type = payload.get("type") or payload.get("source_type") or "polyline"
        source_type = payload.get("source_type")
        if payload_type == "piacircle" or source_type == "piacircle":
            center = tuple(payload.get("center", (0.0, 0.0)))
            radius = float(payload.get("radius", 0.0))
            points_data = payload.get("points")
            if not points_data:
                points_data = circle_points(center, radius, params, samples=256)
            points = np.array(points_data, dtype=float)
            return Shape(
                type="piacircle",
                params=params,
                points=points,
                center=(float(center[0]), float(center[1])),
                radius=float(radius),
                control_points=None,
                id=payload.get("id"),
            )
        if payload_type in ("curve", "piacurve"):
            pts = payload.get("points") or []
            points = np.array(pts, dtype=float)
            ctrl = payload.get("control_points")
            control_points = [tuple(pt) for pt in ctrl] if ctrl else None
            shape_type = source_type or "piacurve"
            return Shape(
                type=shape_type,
                params=params,
                points=points,
                control_points=control_points,
                id=payload.get("id"),
            )
        if payload_type == "segment":
            p1 = tuple(payload.get("p1", (0.0, 0.0)))
            p2 = tuple(payload.get("p2", (0.0, 0.0)))
            pts = [p1, p2]
        else:
            pts = payload.get("points") or []
        points = np.array(pts, dtype=float)
        shape_type = source_type or ("polyline" if payload_type in ("polyline", "segment") else payload_type)
        if not shape_type:
            shape_type = "polyline"
        return Shape(
            type=shape_type,
            params=params,
            points=points,
            control_points=None,
            id=payload.get("id"),
        )

    def pointer_tool_name(self) -> Optional[str]:
        return self._pointer_tool_name

    def set_pointer_tool(self, name: Optional[str], tool: Optional[object]) -> None:
        if self._pointer_tool is tool and self._pointer_tool_name == name:
            return
        if self._pointer_tool and hasattr(self._pointer_tool, "deactivate"):
            self._pointer_tool.deactivate()
        self._pointer_tool = tool
        self._pointer_tool_name = name
        if tool is not None:
            self.setFocus()

    def clear_pointer_tool(self) -> None:
        if self._pointer_tool and hasattr(self._pointer_tool, "deactivate"):
            self._pointer_tool.deactivate()
        self._pointer_tool = None
        self._pointer_tool_name = None

    def _dispatch_pointer_event(self, handler: str, event) -> bool:
        if self._pointer_tool is None:
            return False
        callback = getattr(self._pointer_tool, handler, None)
        if callback is None:
            return False
        callback(event)
        return True

    def _shape_centroid(self, shape: Shape) -> Optional[Point]:
        pts = shape.points
        if pts is None or len(pts) == 0:
            return None
        arr = np.asarray(pts, dtype=float)
        return (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))

    def _shape_metrics(self, shape: Shape) -> Dict[str, Optional[float]]:
        params = shape.params
        metrics: Dict[str, Optional[float]] = {
            "radius": None,
            "pi_a": None,
            "arc_length": None,
            "area": None,
            "curvature": None,
            "memory": None,
        }
        sample_point: Optional[Point] = None
        radius_for_pi: Optional[float] = None
        shape_id = getattr(shape, "id", None)
        pack = self._metrics.field_pack(shape_id)
        if pack.has_data():
            metrics["curvature"] = pack.average_curvature()
            metrics["memory"] = pack.average_memory()

        if shape.type == "piacircle":
            radius = shape.radius or 0.0
            metrics["radius"] = radius
            if shape.center is not None:
                metrics["arc_length"] = self.adaptive_circle_circumference(shape.center, radius, params)
            else:
                metrics["arc_length"] = arc_length(radius, params)
            metrics["area"] = area(radius, params)
            sample_point = shape.center if shape.center is not None else self._shape_centroid(shape)
            radius_for_pi = radius
        elif shape.type == "piacurve":
            control_points = shape.control_points or []
            if len(control_points) >= 2:
                span = math.dist(control_points[0], control_points[-1])
            else:
                span = 0.0
            metrics["radius"] = span / 2.0 if span > 0.0 else 0.0
            metrics["arc_length"] = self.adaptive_curve_length(control_points, params)
            if len(control_points) >= 3:
                metrics["area"] = curve_area(control_points, params)
            sample_point = self._shape_centroid(shape)
            radius_for_pi = max(span, 1e-6)
        elif shape.type == "polyline":
            pts = shape.points if shape.points is not None else np.zeros((0, 2), dtype=float)
            if pts.shape[0] >= 2:
                span = math.dist(pts[0], pts[-1])
            else:
                span = 0.0
            metrics["radius"] = span / 2.0 if span > 0.0 else 0.0
            metrics["arc_length"] = self.adaptive_polyline_length(pts, params)
            metrics["area"] = polygon_area(pts) if pts.shape[0] >= 3 else None
            sample_point = self._shape_centroid(shape)
            radius_for_pi = max(span, 1e-6) if span > 0.0 else 1.0

        if sample_point is None and shape.points is not None and len(shape.points) >= 1:
            sample_point = self._shape_centroid(shape)

        if sample_point is not None:
            if not pack.has_data():
                metrics["curvature"] = self.curvature_at(sample_point)
                metrics["memory"] = self.memory_at(sample_point)
            if radius_for_pi is None:
                radius_for_pi = max(metrics["radius"] or 0.0, 1.0)
            metrics["pi_a"] = self.pi_a_at(sample_point, radius_for_pi, params)
        else:
            radius_value = metrics["radius"]
            if radius_value is not None:
                metrics["pi_a"] = pi_a(max(radius_value, 1e-6), **params)
        return metrics

    # ------------------------------------------------------------------
    # Painting
    def paintEvent(self, event):  # pragma: no cover - GUI entry point
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        elapsed = max(0.0, time.monotonic() - self._anim_start)
        self.bg.draw_background(painter, self.rect(), elapsed)

        # Draw grid if enabled
        gridless_modes = {BGMode.GRID_SCAN, BGMode.SORA2, BGMode.VIDEO}
        if self._snap_to_grid and self.bg.mode() not in gridless_modes:
            self._draw_grid(painter)

        selected_ids = set(self._selection)
        for idx, shape in enumerate(self._shapes):
            if shape.points is None or len(shape.points) < 2:
                continue
            color = QColor(30, 30, 30)
            width = 2
            if shape.id and shape.id in selected_ids:
                color = QColor(50, 120, 215)
                width = 3
            self._draw_shape(painter, shape, color, width)

        if self._temp_shape and self._temp_shape.points is not None and len(self._temp_shape.points) >= 2:
            self._draw_shape(painter, self._temp_shape, QColor(200, 80, 80), 2, dashed=True)

        self._paint_dimensions(painter)
        if self._selection_rect is not None:
            x0, y0, x1, y1 = self._selection_rect
            rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            pen = QPen(QColor(50, 120, 215), 1, Qt.DashLine)
            brush = QColor(50, 120, 215, 40)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawRect(rect)

        self._draw_osnap_indicator(painter)
        self.hud.draw_hud(painter, self.rect(), elapsed)

    def _draw_shape(self, painter: QPainter, shape: Shape, color: QColor, width: int, dashed: bool = False) -> None:
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        points = []
        for x, y in shape.points:
            sx, sy = self.world_to_screen((float(x), float(y)))
            points.append(QPointF(sx, sy))
        painter.drawPolyline(points)
        if shape.type == "piacurve" and shape.control_points:
            handle_pen = QPen(QColor(120, 120, 120), 1, Qt.DotLine)
            painter.setPen(handle_pen)
            controls = []
            for x, y in shape.control_points:
                sx, sy = self.world_to_screen((float(x), float(y)))
                controls.append(QPointF(sx, sy))
            painter.drawPolyline(controls)
            painter.setPen(QPen(QColor(120, 120, 120), 4))
            for pt in controls:
                painter.drawPoint(pt)

    def _draw_osnap_indicator(self, painter: QPainter) -> None:
        if not self._last_osnap:
            return
        kind, point = self._last_osnap
        x, y = self.world_to_screen(point)
        painter.save()
        painter.translate(float(x), float(y))
        base_color = QColor(50, 120, 215)
        pen = QPen(base_color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        size = 6.0

        if kind == "end":
            painter.drawLine(-size, -size, size, size)
            painter.drawLine(-size, size, size, -size)
        elif kind == "mid":
            painter.setBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 60))
            painter.drawRect(QRectF(-size, -size, size * 2, size * 2))
        elif kind == "center":
            painter.drawEllipse(QPointF(0.0, 0.0), size, size)
            painter.drawEllipse(QPointF(0.0, 0.0), size * 0.35, size * 0.35)
        elif kind in {"quad", "90"}:
            points = [
                QPointF(0.0, -size),
                QPointF(size, 0.0),
                QPointF(0.0, size),
                QPointF(-size, 0.0),
            ]
            polygon = QPolygonF(points + [points[0]])
            painter.drawPolyline(polygon)
        elif kind == "appint":
            painter.drawLine(-size * 1.2, 0.0, size * 1.2, 0.0)
            painter.drawLine(0.0, -size * 1.2, 0.0, size * 1.2)
            painter.drawLine(-size, -size, size, size)
            painter.drawLine(-size, size, size, -size)
        elif kind in {"perp", "closest"}:
            painter.drawLine(-size, 0.0, size, 0.0)
            painter.drawLine(0.0, 0.0, 0.0, size)
            painter.drawLine(0.0, 0.0, size * 0.7, -size * 0.7)
        elif kind == "dim":
            painter.drawRect(QRectF(-size, -size, size * 2, size * 2))
        else:
            painter.drawEllipse(QPointF(0.0, 0.0), size * 0.8, size * 0.8)
        painter.restore()

    def _paint_dimensions(self, painter: QPainter) -> None:
        if not self._show_dimensions or not self._dimensions:
            return
        shim = _DimensionPainter(painter)
        to_screen = lambda x, y: self.world_to_screen((float(x), float(y)))
        shape_lookup = {shape.id: shape for shape in self._shapes if shape.id}
        for dim in self._dimensions:
            style = dim.style if hasattr(dim, "style") else self._dim_style
            color = getattr(style, "color", "#222222")
            arrow_px = float(getattr(style, "arrow", 8.0))
            text_px = float(getattr(style, "text_height", 12.0))
            dim_id = getattr(dim, "id", None)
            if dim_id and dim_id in self._selected_dimensions:
                color = "#3278d7"
            if isinstance(dim, LinearDimension):
                params = dict(getattr(dim, "params", {}) or self._params)
                adaptive_length = None
                mode = getattr(style, "mode", "both")
                if mode in {"pi_a", "both"}:
                    adaptive_length = self.adaptive_segment_length(dim.p1, dim.p2, params)
                label = linear_label(dim.p1, dim.p2, style, adaptive_length=adaptive_length)
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
        g = float(self._grid_size)
        if g <= 0.0:
            return
        scale = self._view_scale
        if scale <= 0.0:
            return
        pen_minor = QPen(QColor(235, 235, 235), 1)
        pen_major = QPen(QColor(220, 220, 220), 1)
        pen_minor.setCosmetic(True)
        pen_major.setCosmetic(True)

        x0, y0, x1, y1 = self._visible_world_rect()
        width = self.width()
        height = self.height()
        base_spacing = g * scale
        skip_minor = base_spacing < 6.0
        skip_major = base_spacing < 3.0

        start_idx_x = int(math.floor(x0 / g))
        end_idx_x = int(math.ceil(x1 / g))
        for idx in range(start_idx_x, end_idx_x + 1):
            if skip_major and idx % 10 != 0:
                continue
            if skip_minor and idx % 5 != 0:
                continue
            world_x = idx * g
            screen_x, _ = self.world_to_screen((world_x, 0.0))
            if screen_x < -2.0 or screen_x > width + 2.0:
                continue
            painter.setPen(pen_major if idx % 5 == 0 else pen_minor)
            painter.drawLine(int(round(screen_x)), 0, int(round(screen_x)), height)

        start_idx_y = int(math.floor(y0 / g))
        end_idx_y = int(math.ceil(y1 / g))
        for idx in range(start_idx_y, end_idx_y + 1):
            if skip_major and idx % 10 != 0:
                continue
            if skip_minor and idx % 5 != 0:
                continue
            world_y = idx * g
            _, screen_y = self.world_to_screen((0.0, world_y))
            if screen_y < -2.0 or screen_y > height + 2.0:
                continue
            painter.setPen(pen_major if idx % 5 == 0 else pen_minor)
            painter.drawLine(0, int(round(screen_y)), width, int(round(screen_y)))

    # ------------------------------------------------------------------
    # History (Undo/Redo)
    def _snapshot(self) -> dict:
        return {
            "params": dict(self._params),
            "shapes": [self._serialize_shape(s) for s in self._shapes],
            "selection": list(self._selection),
            "snap": bool(self._snap_to_grid),
            "grid": int(self._grid_size),
            "dimensions": dims_to_json(self._dimensions),
            "dimension_style": self._dim_style.asdict(),
            "show_dimensions": self._show_dimensions,
            "dimension_selection": list(self._selected_dimensions),
            "background_mode": self.bg.mode().name,
            "hud_enabled": self.hud.is_enabled(),
            "hud_status": self.hud.status(),
            "hud_hints": self.hud.hints(),
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
        self._rebuild_adaptive_fields()
        self._reseed_shape_counter()
        mode_name = snap.get("background_mode")
        if isinstance(mode_name, str):
            try:
                self.bg.set_mode(BGMode[mode_name.upper()])
            except KeyError:
                pass
        hud_enabled = snap.get("hud_enabled")
        if hud_enabled is not None:
            self.hud.set_enabled(bool(hud_enabled))
        hud_status = snap.get("hud_status")
        if isinstance(hud_status, str) and hud_status.strip():
            self.hud.set_status(hud_status)
        hud_hints = snap.get("hud_hints")
        if isinstance(hud_hints, list):
            parsed_hints: List[Tuple[str, float]] = []
            for entry in hud_hints:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    try:
                        parsed_hints.append((str(entry[0]), float(entry[1])))
                    except (ValueError, TypeError):
                        continue
            if parsed_hints:
                self.hud.set_hints(parsed_hints)
            else:
                self.hud.set_hints([])
        raw_selection = snap.get("selection")
        if raw_selection is None:
            legacy = snap.get("selected")
            if legacy is None:
                raw_selection = []
            elif isinstance(legacy, list):
                raw_selection = legacy
            else:
                raw_selection = [legacy]
        if not isinstance(raw_selection, list):
            raw_selection = [raw_selection]
        self._set_selection(raw_selection)
        dim_selection = snap.get("dimension_selection", [])
        if not isinstance(dim_selection, list):
            dim_selection = []
        self._set_dimension_selection(dim_selection)
        self._apply_snap_state(snap.get("snap", self._snap_to_grid), emit=False)
        self._grid_size = int(snap.get("grid", self._grid_size))
        self.update()
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
    def resizeEvent(self, event):  # pragma: no cover - GUI layout handling
        super().resizeEvent(event)
        overlay = getattr(self, "overlay", None)
        if overlay is not None:
            overlay.setGeometry(self.rect())
            if self._curved_ui_enabled:
                overlay.raise_()

    def wheelEvent(self, event):  # pragma: no cover - GUI entry point
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta == 0:
                event.accept()
                return
            steps = delta / 120.0
            factor = 1.2 ** steps
            pivot = (event.position().x(), event.position().y())
            self.zoom_by(factor, pivot)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # pragma: no cover - GUI entry point
        self._nudge_overlay_activity(0.5)
        if self._dispatch_pointer_event("mouse_press", event):
            return
        if self._tool is not None:
            self._tool.mouse_press(event)

    def mouseMoveEvent(self, event):  # pragma: no cover - GUI entry point
        self._nudge_overlay_activity(0.5)
        self._update_overlay_activity(event)
        if self._dispatch_pointer_event("mouse_move", event):
            return
        if self._tool is not None:
            self._tool.mouse_move(event)

        self._update_overlay_activity(event)
    def mouseReleaseEvent(self, event):  # pragma: no cover - GUI entry point
        self._nudge_overlay_activity(0.2)
        if self._dispatch_pointer_event("mouse_release", event):
            return
        if self._tool is not None:
            self._tool.mouse_release(event)
        self._nudge_overlay_activity(0.2)

    def keyPressEvent(self, event):  # pragma: no cover - GUI entry point
        if self._handle_zoom_shortcut(event):
            return
        if self._dispatch_pointer_event("key_press", event):
            return
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
            "background_mode": self.bg.mode().name,
            "hud_enabled": self.hud.is_enabled(),
            "hud_status": self.hud.status(),
            "hud_hints": self.hud.hints(),
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
