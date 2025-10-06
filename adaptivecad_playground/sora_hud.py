"""Animated HUD overlay for the AdaptiveCAD playground."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen

Hint = Tuple[str, float]


@dataclass
class HudTheme:
    """Theme colors and sizing for the Sora HUD."""

    base_ring: QColor = field(default_factory=lambda: QColor(0, 210, 255, 180))
    glow: QColor = field(default_factory=lambda: QColor(0, 180, 220, 90))
    ticks: QColor = field(default_factory=lambda: QColor(0, 250, 255, 130))
    tick_highlight: QColor = field(default_factory=lambda: QColor(255, 255, 255, 220))
    hint_text: QColor = field(default_factory=lambda: QColor(200, 255, 255, 200))
    hint_marker: QColor = field(default_factory=lambda: QColor(0, 240, 255, 180))
    badge_bg: QColor = field(default_factory=lambda: QColor(6, 36, 60, 210))
    badge_border: QColor = field(default_factory=lambda: QColor(0, 210, 255, 160))
    status_ready: QColor = field(default_factory=lambda: QColor(0, 245, 190))
    status_idle: QColor = field(default_factory=lambda: QColor(255, 120, 90))
    hud_text: QColor = field(default_factory=lambda: QColor(220, 248, 255))
    margin: float = 32.0


class SoraHud:
    """Jarvis-style heads-up display rendered around the viewport."""

    _TICK_COUNT = 96

    def __init__(self, theme: HudTheme | None = None) -> None:
        self._theme = theme or HudTheme()
        self._enabled = True
        self._status_text = "SORA OFF"
        self._hints: List[Hint] = []
        self._font_family = "Orbitron"

    # ------------------------------------------------------------------
    # Configuration
    def set_theme(self, theme: HudTheme) -> None:
        self._theme = theme

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        return self._enabled

    def set_status(self, status: str) -> None:
        if not status:
            status = "SORA OFF"
        self._status_text = status.upper()

    def status(self) -> str:
        return self._status_text

    def set_hints(self, hints: Sequence[Hint]) -> None:
        cleaned: List[Hint] = []
        for text, pos in hints:
            if not text.strip():
                continue
            cleaned.append((text.strip(), float(pos) % 1.0))
        self._hints = cleaned

    def append_hint(self, text: str, position: float) -> None:
        if not text.strip():
            return
        self._hints.append((text.strip(), float(position) % 1.0))

    def hints(self) -> List[Hint]:
        return list(self._hints)

    # ------------------------------------------------------------------
    # Drawing
    def draw_hud(self, painter: QPainter, rect, elapsed: float) -> None:
        if not self._enabled:
            return
        if not isinstance(rect, QRectF):
            rect = QRectF(rect)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        radius = max(
            80.0,
            min(rect.width(), rect.height()) * 0.5 - self._theme.margin,
        )
        center = rect.center()

        self._draw_glow(painter, center, radius, elapsed)
        self._draw_ring(painter, center, radius, elapsed)
        self._draw_ticks(painter, center, radius, elapsed)
        self._draw_hints(painter, center, radius)
        self._draw_badge(painter, center, radius)

        painter.restore()

    def _draw_glow(self, painter: QPainter, center: QPointF, radius: float, elapsed: float) -> None:
        pulse = 0.6 + 0.4 * math.sin(elapsed * 0.7)
        glow_color = QColor(self._theme.glow)
        glow_color.setAlphaF(min(1.0, max(0.0, pulse)))
        pen = QPen(glow_color, radius * 0.18)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

    def _draw_ring(self, painter: QPainter, center: QPointF, radius: float, elapsed: float) -> None:
        # Outer ring
        pen = QPen(self._theme.base_ring, 3.2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)

        # Inner accent ring
        accent_radius = radius * 0.86
        accent_pen = QPen(self._theme.base_ring, 1.4)
        accent_pen.setCosmetic(True)
        accent_pen.setStyle(Qt.DashLine)
        painter.setPen(accent_pen)
        painter.drawEllipse(center, accent_radius, accent_radius)

    def _draw_ticks(self, painter: QPainter, center: QPointF, radius: float, elapsed: float) -> None:
        highlight_idx = int((elapsed * 48.0) % self._TICK_COUNT)
        painter.save()
        painter.translate(center)

        for idx in range(self._TICK_COUNT):
            angle = 2.0 * math.pi * (idx / self._TICK_COUNT)
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            if idx % 12 == 0:
                inner = radius - 24.0
                outer = radius + 10.0
                width = 2.6
            elif idx % 3 == 0:
                inner = radius - 16.0
                outer = radius + 6.0
                width = 1.8
            else:
                inner = radius - 10.0
                outer = radius + 4.0
                width = 1.2
            color = QColor(self._theme.ticks)
            if idx == highlight_idx:
                color = QColor(self._theme.tick_highlight)
            pen = QPen(color, width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(cos_a * inner, sin_a * inner),
                QPointF(cos_a * outer, sin_a * outer),
            )

        painter.restore()

    def _draw_hints(self, painter: QPainter, center: QPointF, radius: float) -> None:
        if not self._hints:
            return

        font = QFont(self._font_family, 9)
        font.setCapitalization(QFont.AllUppercase)
        metrics = QFontMetrics(font)

        for text, pos in self._hints:
            angle = (pos * 2.0 * math.pi) - math.pi / 2.0
            direction = QPointF(math.cos(angle), math.sin(angle))
            marker_inner = radius + 8.0
            marker_outer = radius + 16.0
            marker_pen = QPen(self._theme.hint_marker, 2.0)
            marker_pen.setCosmetic(True)
            painter.setPen(marker_pen)
            painter.drawLine(
                QPointF(center.x() + direction.x() * marker_inner, center.y() + direction.y() * marker_inner),
                QPointF(center.x() + direction.x() * marker_outer, center.y() + direction.y() * marker_outer),
            )

            text_rect = QRectF(0.0, 0.0, 260.0, metrics.height() + 6.0)
            text_rect.moveCenter(
                QPointF(
                    center.x() + direction.x() * (radius + 60.0),
                    center.y() + direction.y() * (radius + 60.0),
                )
            )

            # Bias alignment for readability
            align_flags = Qt.AlignCenter
            if direction.x() > 0.25:
                align_flags = Qt.AlignLeft | Qt.AlignVCenter
                text_rect.setLeft(text_rect.left() - 70.0)
                text_rect.setRight(text_rect.right() - 70.0)
            elif direction.x() < -0.25:
                align_flags = Qt.AlignRight | Qt.AlignVCenter
                text_rect.setLeft(text_rect.left() + 70.0)
                text_rect.setRight(text_rect.right() + 70.0)

            painter.save()
            painter.setFont(font)
            painter.setPen(self._theme.hint_text)
            painter.drawText(text_rect, align_flags | Qt.TextWordWrap, text)
            painter.restore()

    def _draw_badge(self, painter: QPainter, center: QPointF, radius: float) -> None:
        badge_width = max(160.0, radius * 0.9)
        badge_height = 34.0
        badge_rect = QRectF(
            center.x() - badge_width / 2.0,
            center.y() - radius - badge_height - 18.0,
            badge_width,
            badge_height,
        )

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        border_pen = QPen(self._theme.badge_border, 1.2)
        border_pen.setCosmetic(True)
        painter.setPen(border_pen)
        painter.setBrush(self._theme.badge_bg)
        painter.drawRoundedRect(badge_rect, 10.0, 10.0)

        status_color = self._theme.status_ready if "READY" in self._status_text else self._theme.status_idle
        indicator_radius = 5.0
        indicator_center = QPointF(badge_rect.left() + 16.0, badge_rect.center().y())
        painter.setBrush(status_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(indicator_center, indicator_radius, indicator_radius)

        painter.setFont(QFont(self._font_family, 10))
        painter.setPen(self._theme.hud_text)
        text_rect = QRectF(
            indicator_center.x() + indicator_radius + 8.0,
            badge_rect.top(),
            badge_rect.width() - (indicator_radius + 24.0),
            badge_rect.height(),
        )
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._status_text)

        painter.restore()
