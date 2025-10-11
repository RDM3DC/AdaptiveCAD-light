# adaptivecad_playground/curved_ui.py
"""
CurvedUI v0.1 — geometry + helpers for a UI with no straight lines and adaptive curvature.

This module provides:
- superellipse/squircle path generation (no straight edges)
- arc layout utilities (positions along a warped arc)
- adaptive curvature scheduler (responds to user interaction/zoom/time)
- πₐ-aware angle warp (optional) so arc spacing can reflect your adaptive π

All functions are Qt-agnostic; QPainterPath is optional.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]

# ---- πₐ helpers (optional) -------------------------------------------------
def pi_a_value(r: float, params: dict) -> float:
    """Example πₐ kernel; replace with your Playground's kernel if available."""
    α = params.get("α", params.get("alpha", 0.0))
    μ = params.get("μ", params.get("mu", 0.0))
    k0 = params.get("k0", 0.0)
    return math.pi * (1.0 + k0 * r) * math.exp(-μ * r)

def warp_angle_by_pi_a(theta: float, r: float, params: dict) -> float:
    """Warp a Euclidean angle by local πₐ — light visual connection to your math."""
    return theta * (pi_a_value(r, params) / math.pi)

# ---- Curvature scheduler ----------------------------------------------------
@dataclass
class CurvatureState:
    base_n: float = 3.5    # superellipse exponent (2=ellipse, higher -> squarer, stay curved)
    boost_n: float = 1.0   # added when 'active'
    decay_mu: float = 1.5  # s^-1 exponential decay of activity
    activity: float = 0.0  # [0..1] internal state

    def tick(self, dt: float, input_energy: float = 0.0) -> float:
        """
        Update activity with input_energy (e.g., |mouse velocity|, tool change impulses).
        Returns current exponent n for superellipse corners: n = base + boost*activity.
        """
        # normalize input a bit
        a = max(0.0, min(1.0, input_energy))
        self.activity = max(0.0, min(1.0, self.activity*(math.exp(-self.decay_mu*dt)) + (1.0-self.activity)*a))
        return self.base_n + self.boost_n * self.activity

# ---- Superellipse / Squircle -----------------------------------------------
def superellipse_points(rect: Rect, n: float = 4.0, m: int = 128) -> List[Point]:
    """
    Parametric superellipse (Lamé curve): |x/a|^n + |y/b|^n = 1
    Returns a list of (x,y) around the boundary; always curved (no straight edges).
    """
    x0,y0,w,h = rect
    a,b = w/2.0, h/2.0
    cx, cy = x0 + a, y0 + b
    pts = []
    for i in range(m):
        t = 2*math.pi * (i / m)
        ct, st = math.cos(t), math.sin(t)
        x = math.copysign(abs(ct)**(2.0/n), ct) * a + cx
        y = math.copysign(abs(st)**(2.0/n), st) * b + cy
        pts.append((x,y))
    return pts

def rounded_path(rect: Rect, n: float = 4.0, m: int = 128):
    """Optionally create a QPainterPath if Qt is available."""
    try:
        from PySide6.QtGui import QPainterPath
        path = QPainterPath()
        pts = superellipse_points(rect, n, m)
        if not pts: return QPainterPath()
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(p[0], p[1])
        path.closeSubpath()
        return path
    except Exception:
        return superellipse_points(rect, n, m)

# ---- Arc layout -------------------------------------------------------------
def arc_positions(n_items: int, rect: Rect, start_deg: float, sweep_deg: float,
                  params: Optional[dict] = None) -> List[Point]:
    """
    Place n_items along an arc of a circle inscribed in rect, using optional πₐ angle warp.
    """
    x0,y0,w,h = rect
    r = min(w,h)/2.2
    cx, cy = x0 + w/2.0, y0 + h/2.0
    out = []
    for i in range(n_items):
        u = i/(max(1,n_items-1)) if n_items>1 else 0.5
        θ = math.radians(start_deg + u*sweep_deg)
        if params is not None:
            θ = warp_angle_by_pi_a(θ, r, params)
        x = cx + r*math.cos(θ)
        y = cy + r*math.sin(θ)
        out.append((x,y))
    return out
