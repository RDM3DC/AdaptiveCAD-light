"""Animated background modes for the AdaptiveCAD playground."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen


class BGMode(Enum):
    """Background rendering modes."""

    OFF = auto()
    GRADIENT = auto()
    GRID_SCAN = auto()
    VIDEO = auto()
    SORA2 = auto()
    BLUEPRINT = auto()
    SUNSET = auto()
    CARBON = auto()


@dataclass(frozen=True)
class PanelTheme:
    top: QColor
    bottom: QColor
    text: QColor
    border: QColor
    accent: QColor


_PANEL_THEME_MAP: dict[BGMode, PanelTheme] = {
    BGMode.OFF: PanelTheme(
        top=QColor(245, 245, 245),
        bottom=QColor(230, 230, 230),
        text=QColor(32, 32, 32),
        border=QColor(200, 200, 200),
        accent=QColor(90, 140, 220),
    ),
    BGMode.GRADIENT: PanelTheme(
        top=QColor(46, 58, 82),
        bottom=QColor(28, 34, 48),
        text=QColor(235, 244, 255),
        border=QColor(70, 110, 170),
        accent=QColor(110, 180, 255),
    ),
    BGMode.GRID_SCAN: PanelTheme(
        top=QColor(32, 56, 70),
        bottom=QColor(18, 32, 44),
        text=QColor(225, 245, 250),
        border=QColor(0, 150, 190),
        accent=QColor(0, 200, 255),
    ),
    BGMode.VIDEO: PanelTheme(
        top=QColor(28, 40, 52),
        bottom=QColor(12, 18, 26),
        text=QColor(230, 240, 250),
        border=QColor(70, 110, 150),
        accent=QColor(130, 180, 255),
    ),
    BGMode.SORA2: PanelTheme(
        top=QColor(22, 40, 68),
        bottom=QColor(12, 22, 44),
        text=QColor(230, 242, 255),
        border=QColor(40, 120, 220),
        accent=QColor(110, 180, 255),
    ),
    BGMode.BLUEPRINT: PanelTheme(
        top=QColor(24, 48, 92),
        bottom=QColor(12, 28, 60),
        text=QColor(220, 240, 255),
        border=QColor(90, 140, 220),
        accent=QColor(140, 200, 255),
    ),
    BGMode.SUNSET: PanelTheme(
        top=QColor(88, 28, 72),
        bottom=QColor(40, 12, 36),
        text=QColor(255, 235, 240),
        border=QColor(190, 90, 150),
        accent=QColor(255, 170, 120),
    ),
    BGMode.CARBON: PanelTheme(
        top=QColor(46, 46, 48),
        bottom=QColor(22, 22, 24),
        text=QColor(225, 235, 240),
        border=QColor(90, 90, 96),
        accent=QColor(120, 200, 255),
    ),
}


def panel_theme_for(mode: BGMode) -> PanelTheme:
    return _PANEL_THEME_MAP.get(mode, _PANEL_THEME_MAP[BGMode.GRADIENT])


class SoraBackground:
    """Handles animated backgrounds and Sora placeholders."""

    def __init__(self) -> None:
        self._mode: BGMode = BGMode.GRADIENT
        self._video_frame: Optional[QImage] = None
        self._sora_message: str = "Sora 2 background pending public API availability."
        self._video_message: str = "Video background stub: supply frames via set_video_frame(...)."

    # ------------------------------------------------------------------
    def mode(self) -> BGMode:
        return self._mode

    def set_mode(self, mode: BGMode) -> None:
        if not isinstance(mode, BGMode):
            raise TypeError("mode must be a BGMode value.")
        self._mode = mode

    def set_video_frame(self, frame: Optional[QImage]) -> None:
        self._video_frame = frame

    def clear_video_frame(self) -> None:
        self._video_frame = None

    def set_sora_message(self, message: str) -> None:
        if message:
            self._sora_message = message

    # ------------------------------------------------------------------
    def draw_background(self, painter: QPainter, rect, elapsed: float) -> None:
        if not isinstance(rect, QRectF):
            rect = QRectF(rect)

        if self._mode is BGMode.OFF:
            painter.fillRect(rect, QColor(250, 250, 250))
            return
        if self._mode is BGMode.GRADIENT:
            self._draw_gradient_background(painter, rect, elapsed)
            return
        if self._mode is BGMode.GRID_SCAN:
            self._draw_grid_scan_background(painter, rect, elapsed)
            return
        if self._mode is BGMode.VIDEO:
            self._draw_video_background(painter, rect)
            return
        if self._mode is BGMode.SORA2:
            self._draw_sora_stub(painter, rect, elapsed)
            return
        if self._mode is BGMode.BLUEPRINT:
            self._draw_blueprint_background(painter, rect, elapsed)
            return
        if self._mode is BGMode.SUNSET:
            self._draw_sunset_background(painter, rect, elapsed)
            return
        if self._mode is BGMode.CARBON:
            self._draw_carbon_background(painter, rect, elapsed)
            return

    # ------------------------------------------------------------------
    def _draw_gradient_background(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        phase = 0.5 + 0.5 * math.sin(elapsed * 0.2)
        base = QColor(10, 16, 32)
        accent = QColor(0, 110, 190)
        accent2 = QColor(0, 180, 210)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, base)
        gradient.setColorAt(phase * 0.4, accent)
        gradient.setColorAt(0.9, accent2)
        painter.fillRect(rect, gradient)

    def _draw_blueprint_background(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor(10, 34, 78))
        gradient.setColorAt(1.0, QColor(4, 18, 40))
        painter.fillRect(rect, gradient)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)

        spacing = 36.0
        offset = (elapsed * 12.0) % spacing
        major = QPen(QColor(150, 200, 255, 190), 1.6)
        major.setCosmetic(True)
        minor = QPen(QColor(90, 140, 210, 110), 1)
        minor.setCosmetic(True)

        y = rect.top() - spacing * 2 + offset
        band = 0
        while y <= rect.bottom() + spacing * 2:
            painter.setPen(major if band % 5 == 0 else minor)
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += spacing
            band += 1

        x = rect.left() - spacing * 2 + offset
        column = 0
        while x <= rect.right() + spacing * 2:
            painter.setPen(major if column % 5 == 0 else minor)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += spacing
            column += 1

        painter.restore()

    def _draw_sunset_background(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        top = QColor(255, 120, 60)
        mid = QColor(255, 70, 120)
        bottom = QColor(40, 10, 60)
        phase = 0.5 + 0.5 * math.sin(elapsed * 0.18)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, top)
        gradient.setColorAt(0.45 + phase * 0.1, mid)
        gradient.setColorAt(1.0, bottom)
        painter.fillRect(rect, gradient)

        glow_height = rect.height() * (0.18 + 0.04 * math.sin(elapsed * 0.22))
        glow_rect = QRectF(rect.left(), rect.bottom() - glow_height, rect.width(), glow_height)
        painter.fillRect(glow_rect, QColor(255, 190, 120, 80))
        painter.setPen(QPen(QColor(255, 210, 160, 90), 2))
        painter.drawLine(rect.left(), glow_rect.top(), rect.right(), glow_rect.top())

    def _draw_carbon_background(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor(20, 20, 24))
        gradient.setColorAt(1.0, QColor(10, 10, 12))
        painter.fillRect(rect, gradient)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)
        spacing = 14.0
        offset = (elapsed * 20.0) % spacing
        pen_light = QPen(QColor(140, 140, 140, 90), 1)
        pen_light.setCosmetic(True)
        pen_dark = QPen(QColor(0, 0, 0, 120), 1)
        pen_dark.setCosmetic(True)

        start = rect.left() - rect.height() - spacing
        end = rect.right() + rect.height() + spacing
        pos = start + offset
        toggle = 0
        while pos <= end:
            painter.setPen(pen_light if toggle % 2 == 0 else pen_dark)
            painter.drawLine(pos, rect.top(), pos + rect.height(), rect.bottom())
            pos += spacing * 0.5
            toggle += 1

        pos = start - offset
        toggle = 0
        while pos <= end:
            painter.setPen(pen_dark if toggle % 2 == 0 else pen_light)
            painter.drawLine(pos, rect.bottom(), pos + rect.height(), rect.top())
            pos += spacing * 0.5
            toggle += 1

        painter.restore()

    def _draw_grid_scan_background(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        self._draw_gradient_background(painter, rect, elapsed)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, False)

        grid_color = QColor(0, 190, 210, 60)
        major_color = QColor(0, 210, 255, 90)
        spacing = 48.0
        start_x = int(rect.left() // spacing - 2) * spacing
        end_x = int(rect.right() // spacing + 2) * spacing
        start_y = int(rect.top() // spacing - 2) * spacing
        end_y = int(rect.bottom() // spacing + 2) * spacing

        pen = QPen(grid_color, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)

        x = start_x
        idx = 0
        while x <= end_x:
            if idx % 3 == 0:
                painter.setPen(QPen(major_color, 1.4))
            else:
                painter.setPen(pen)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += spacing
            idx += 1

        pen = QPen(grid_color, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        y = start_y
        idx = 0
        while y <= end_y:
            if idx % 3 == 0:
                painter.setPen(QPen(major_color, 1.4))
            else:
                painter.setPen(pen)
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += spacing
            idx += 1

        sweep_height = rect.height() * 0.16
        sweep_y = rect.top() + (rect.height() + sweep_height) * ((elapsed * 0.25) % 1.0) - sweep_height
        sweep_rect = QRectF(rect.left(), sweep_y, rect.width(), sweep_height)
        sweep_color = QColor(0, 255, 230, 26)
        painter.fillRect(sweep_rect, sweep_color)

        painter.restore()

    def _draw_video_background(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor(6, 20, 35))
        if self._video_frame is None or self._video_frame.isNull():
            self._draw_placeholder_label(
                painter,
                rect,
                "Video background awaiting frames.",
            )
            return

        frame = self._video_frame.scaled(
            int(rect.width()),
            int(rect.height()),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        target = QRectF(rect)
        # Center the frame crop
        dx = (frame.width() - rect.width()) / 2.0
        dy = (frame.height() - rect.height()) / 2.0
        source = QRectF(dx, dy, rect.width(), rect.height())
        painter.drawImage(target, frame, source)

        overlay_color = QColor(6, 20, 35, 90)
        painter.fillRect(rect, overlay_color)

        self._draw_placeholder_label(
            painter,
            rect.adjusted(0, rect.height() * 0.35, 0, -rect.height() * 0.2),
            self._video_message,
        )

    def _draw_sora_stub(self, painter: QPainter, rect: QRectF, elapsed: float) -> None:
        self._draw_gradient_background(painter, rect, elapsed)
        painter.save()

        overlay = QColor(0, 130, 255, 35)
        painter.fillRect(rect, overlay)

        self._draw_placeholder_label(
            painter,
            rect.adjusted(rect.width() * 0.15, rect.height() * 0.32, -rect.width() * 0.15, -rect.height() * 0.3),
            self._sora_message,
        )

        painter.restore()

    def _draw_placeholder_label(self, painter: QPainter, rect: QRectF, message: str) -> None:
        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        painter.setPen(QColor(255, 255, 255, 210))
        painter.setBrush(Qt.NoBrush)
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, message)

        painter.restore()
