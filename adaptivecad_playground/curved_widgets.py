# adaptivecad_playground/curved_widgets.py
"""
CurvedUI v0.1 widgets (PySide6):
- CurvedViewportOverlay: draws a superellipse panel and a curved ribbon of buttons
- CurvedPanel: reusable rounded panel (superellipse)
- CurvedButton: pill/orb button (curved only)
All strokes are paper-space so the UI has no straight, pixel-snap lines and stays smooth while zooming.

Note: These are lightweight visual components; wire your existing actions to them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import math, time

try:
    from PySide6.QtCore import Qt, QTimer, QRectF, QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
    from PySide6.QtWidgets import QWidget
except Exception:
    QWidget = object

try:
    from .curved_ui import CurvatureState, rounded_path, arc_positions
except ImportError:  # When executed as a loose script without a package context
    from curved_ui import CurvatureState, rounded_path, arc_positions

@dataclass
class CurvedTheme:
    bg1: str = "#081C28"
    bg2: str = "#0C2A3A"
    edge: str = "#00D1FF"
    tick: str = "#00FFC2"
    text: str = "#E6FCFF"
    btn_fill: str = "#0F3347"
    btn_fill_hi: str = "#10485F"

class CurvedPanel(QWidget):
    def __init__(self, parent=None, theme: CurvedTheme = CurvedTheme()):
        super().__init__(parent)
        self.theme = theme
        self.n = 3.8
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w,h = self.width(), self.height()
        path = rounded_path((2,2,w-4,h-4), self.n, 128)
        # background gradient
        g = QPainterPath()
        if isinstance(path, QPainterPath):
            qp.fillPath(path, QColor(self.theme.bg1))
            pen = QPen(QColor(self.theme.edge)); pen.setWidthF(1.8); pen.setCosmetic(True); qp.setPen(pen)
            qp.drawPath(path)
        else:
            # fallback (no Qt path)
            pass

class CurvedButton(QWidget):
    def __init__(self, label: str, on_click: Optional[Callable]=None, parent=None, theme: CurvedTheme = CurvedTheme()):
        super().__init__(parent)
        self.label = label
        self.theme = theme
        self.on_click = on_click
        self.setFixedSize(96, 36)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def mousePressEvent(self, ev):
        if self.on_click: self.on_click()

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w,h = self.width(), self.height()
        r = h/2.0
        path = rounded_path((1,1,w-2,h-2), n=2.2, m=96)  # close to a capsule, still curved
        if isinstance(path, QPainterPath):
            qp.fillPath(path, QColor(self.theme.btn_fill))
            pen = QPen(QColor(self.theme.edge)); pen.setWidthF(1.5); pen.setCosmetic(True); qp.setPen(pen)
            qp.drawPath(path)
            qp.setPen(QPen(QColor(self.theme.text), 1))
            f = qp.font(); f.setPointSizeF(10.5); qp.setFont(f)
            qp.drawText(self.rect(), Qt.AlignCenter, self.label)

class CurvedViewportOverlay(QWidget):
    """
    Drop this as a child overlay of your Canvas. It animates a curved ribbon of buttons and a rounded border panel.
    """
    def __init__(self, parent=None, theme: CurvedTheme = CurvedTheme(), get_params: Optional[Callable]=None):
        super().__init__(parent)
        self.theme = theme
        self.curv = CurvatureState()
        self.get_params = get_params or (lambda: {"α":0.02,"μ":0.01,"k0":0.1})
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._t0 = time.time()
        self.timer = QTimer(self); self.timer.timeout.connect(self.update); self.timer.start(33)  # ~30 FPS

    def adapt(self, input_energy=0.0):
        # external hook if you want to pump activity (mouse velocity, clicks, etc.)
        self.curv.activity = max(0.0, min(1.0, self.curv.activity + input_energy))

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w,h = self.width(), self.height()
        t = time.time() - self._t0
        n = self.curv.tick(1/30.0, input_energy=0.0)  # passive decay; call adapt() elsewhere

        # rounded outer border (superellipse)
        path = rounded_path((6,6,w-12,h-12), n, 160)
        if isinstance(path, QPainterPath):
            # translucent inner wash so shapes remain visible beneath the overlay
            fill = QColor(self.theme.bg2)
            fill.setAlphaF(0.32)
            qp.fillPath(path, fill)
            pen = QPen(QColor(self.theme.edge))
            pen.setWidthF(2.0)
            pen.setCosmetic(True)
            qp.setPen(pen)
            qp.drawPath(path)

        # Curved ribbon buttons along a πₐ-warped arc near the top
        rect = (0.0, 0.0, float(w), float(h))
        params = self.get_params()
        arc_pts = arc_positions(6, rect, start_deg=200, sweep_deg=140, params=params)
        # draw beads
        pen = QPen(QColor(self.theme.tick))
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        qp.setPen(pen)
        for (x,y) in arc_pts:
            r = 12.0 + 4.0*math.sin(2*math.pi*(t*0.25 + x*0.001))
            # draw a small circular bead (curved only)
            bead_fill = QColor(self.theme.btn_fill)
            bead_fill.setAlphaF(0.85)
            qp.setBrush(bead_fill)
            qp.drawEllipse(QPointF(x, y), r, r)
