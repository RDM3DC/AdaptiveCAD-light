"""Animated background modes for the AdaptiveCAD playground."""
from __future__ import annotations

import math
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

        # Scanner sweep
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
