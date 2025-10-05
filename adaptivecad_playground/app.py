"""Application bootstrap for the AdaptiveCAD Playground."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QStatusBar, QToolBar

from widgets import Canvas, Controls


class Main(QMainWindow):
    """Top-level window wiring together the canvas, controls, and chrome."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AdaptiveCAD Playground (πₐ)")

        self.canvas = Canvas()
        self.controls = Controls(self.canvas.set_params)

        self.setCentralWidget(self.canvas)
        self.addDockWidget(Qt.RightDockWidgetArea, self.controls.dock)

        self._status_labels: dict[str, QLabel] = {}
        self._tool_actions: dict[str, QAction] = {}
        self._snap_action: QAction | None = None
        self._dims_action: QAction | None = None
        self._setup_status_bar()
        self._make_toolbar()
        self._make_menu()

        self.canvas.status_changed.connect(self._on_status_changed)
        self.canvas.tool_changed.connect(self._on_tool_changed)
        self.canvas.snap_changed.connect(self._on_snap_changed)
        self.canvas.dimensions_visibility_changed.connect(self._on_dimensions_visibility_changed)

        self.resize(1200, 800)
        self.canvas.set_tool("select")
        self.canvas.set_snap_to_grid(True)

    # ------------------------------------------------------------------
    # UI scaffolding
    def _setup_status_bar(self) -> None:
        bar = QStatusBar()
        bar.setSizeGripEnabled(False)
        self.setStatusBar(bar)

        self._mode_label = QLabel("Tool: Select")
        self._mode_label.setToolTip("Active tool. Change using the toolbar on the left.")
        bar.addPermanentWidget(self._mode_label)

        self._status_labels = {
            "radius": QLabel("r: --"),
            "pi_a":     QLabel("pi_a: --"),
            "arc_length": QLabel("arc: --"),
            "area":    QLabel("area: --"),
        }
        status_help = {
            "radius": "Radius: distance for selected πₐ circle or proxy span.",
            "pi_a": "Local πₐ value computed with current parameters.",
            "arc_length": "Arc length of the adaptive circle/curve.",
            "area": "Area enclosed (when defined for the selection).",
        }
        for key, label in self._status_labels.items():
            label.setToolTip(status_help.get(key, ""))
            bar.addPermanentWidget(label)

    def _make_toolbar(self) -> None:
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.LeftToolBarArea, toolbar)

        action_group = QActionGroup(self)
        action_group.setExclusive(True)

        self._tool_actions = {}
        definitions = (
            ("select", "Select", "Select tool: click shapes to inspect metrics.", None),
            ("piacircle", "πₐ Circle", "πₐ Circle: first click center, second click sets radius.", None),
            ("piacurve", "πₐ Curve", "πₐ Curve: click control points, Enter to finish.", None),
            ("dim_linear", "Linear Dim", "Linear dimension: three clicks to place an aligned measurement.", "dim_linear.svg"),
            ("dim_radial", "Radial Dim", "Radial dimension: click a πₐ circle, then place the label.", "dim_radial.svg"),
            ("dim_angular", "Angular Dim", "Angular dimension: vertex, ray A, ray B, then label.", "dim_angular.svg"),
            ("measure", "Measure", "Measure tool: hover to read world coordinates.", "measure.svg"),
        )
        for name, text, tip, icon_name in definitions:
            if icon_name:
                action = QAction(self._load_icon(icon_name), text, self)
                toolbar.addAction(action)
            else:
                action = QAction(text, self)
                toolbar.addAction(action)
            action.setCheckable(True)
            action.setActionGroup(action_group)
            action.triggered.connect(lambda checked, n=name: self._activate_tool(n, checked))
            action.setToolTip(tip)
            action.setStatusTip(tip)
            self._tool_actions[name] = action

    def _make_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        export_png_action = file_menu.addAction("Export PNG")
        export_png_action.triggered.connect(lambda: self.canvas.export_png(self))
        export_png_action.setToolTip("Export a PNG snapshot of the current canvas.")
        export_png_action.setStatusTip("Export a PNG snapshot of the current canvas.")

        export_json_action = file_menu.addAction("Export JSON")
        export_json_action.triggered.connect(lambda: self.canvas.export_json(self))
        export_json_action.setToolTip("Export a JSON file with shapes, parameters, and settings.")
        export_json_action.setStatusTip("Export a JSON file with shapes, parameters, and settings.")

        # Edit menu: Undo/Redo
        edit_menu = menu_bar.addMenu("&Edit")
        undo_action = edit_menu.addAction("Undo")
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.canvas.undo)
        undo_action.setToolTip("Undo the last drawing or parameter change.")
        undo_action.setStatusTip("Undo the last drawing or parameter change.")
        redo_action = edit_menu.addAction("Redo")
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.canvas.redo)
        redo_action.setToolTip("Redo the last undone step.")
        redo_action.setStatusTip("Redo the last undone step.")

        # View menu: Snap to Grid toggle
        view_menu = menu_bar.addMenu("&View")
        self._snap_action = view_menu.addAction("Snap to Grid")
        self._snap_action.setCheckable(True)
        self._snap_action.triggered.connect(lambda checked: self.canvas.set_snap_to_grid(bool(checked)))
        self._snap_action.setChecked(self.canvas.is_snap_enabled())
        self._snap_action.setToolTip("Toggle snapping drawing points to the grid.")
        self._snap_action.setStatusTip("Toggle snapping drawing points to the grid.")

        self._dims_action = view_menu.addAction("Show Dimensions")
        self._dims_action.setCheckable(True)
        self._dims_action.setChecked(self.canvas.dimensions_visible())
        self._dims_action.setToolTip("Toggle dimension annotations on the canvas.")
        self._dims_action.setStatusTip("Toggle dimension annotations on the canvas.")
        self._dims_action.triggered.connect(
            lambda checked: self.canvas.set_dimensions_visible(bool(checked))
        )

    # ------------------------------------------------------------------
    # Event handlers
    def _activate_tool(self, name: str, checked: bool) -> None:
        if not checked:
            return
        self.canvas.set_tool(name)

    def _on_tool_changed(self, name: str) -> None:
        text = {
            "select": "Select",
            "piacircle": "PiA Circle",
            "piacurve": "PiA Curve",
            "dim_linear": "Linear Dim",
            "dim_radial": "Radial Dim",
            "dim_angular": "Angular Dim",
            "measure": "Measure",
        }.get(name, name.title())
        self._mode_label.setText(f"Tool: {text}")
        action = self._tool_actions.get(name)
        if action:
            blocked = action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(blocked)

    def _on_status_changed(self, payload: dict) -> None:
        message = payload.get("message")
        if message is not None:
            if message:
                self.statusBar().showMessage(message, 4000)
            else:
                self.statusBar().clearMessage()
        radius = payload.get("radius")
        pi_value = payload.get("pi_a")
        arc = payload.get("arc_length")
        area_value = payload.get("area")

        self._status_labels["radius"].setText(self._format_value("r", radius, precision=2))
        self._status_labels["pi_a"].setText(self._format_value("pi_a", pi_value, precision=4))
        self._status_labels["arc_length"].setText(self._format_value("arc", arc, precision=2))
        self._status_labels["area"].setText(self._format_value("area", area_value, precision=2))

    def _format_value(self, label: str, value: float | None, precision: int) -> str:
        if value is None:
            return f"{label}: --"
        return f"{label}: {value:.{precision}f}"

    def _on_snap_changed(self, enabled: bool) -> None:
        if self._snap_action is None:
            return
        blocked = self._snap_action.blockSignals(True)
        self._snap_action.setChecked(bool(enabled))
        self._snap_action.blockSignals(blocked)

    def _on_dimensions_visibility_changed(self, visible: bool) -> None:
        if self._dims_action is None:
            return
        blocked = self._dims_action.blockSignals(True)
        self._dims_action.setChecked(bool(visible))
        self._dims_action.blockSignals(blocked)

    def _load_icon(self, filename: str) -> QIcon:
        icon_path = Path(__file__).with_name("icons") / filename
        if icon_path.exists():
            return QIcon(str(icon_path))
        return QIcon()


def main() -> int:
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
