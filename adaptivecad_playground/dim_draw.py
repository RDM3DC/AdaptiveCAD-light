# adaptivecad_playground/dim_draw.py
"""
Qt-agnostic drawing helpers for dimensions.
You provide "to_screen: (x,y)->(X,Y)" and a basic "Painter" shim with:
- line(x1,y1,x2,y2,color,width)
- polyline([(x,y),...], color, width)
- text(label, x, y, px_height, color, anchor="center", halo=True)
- arrow(x1,y1,x2,y2, size_px, color)
This keeps Canvas code small. Implement the shim in your widgets.py using QPainter.
"""

from __future__ import annotations
from typing import Tuple, List
import math

Point = Tuple[float, float]

def _unit(vx, vy):
    n = math.hypot(vx, vy)
    if n == 0: return 0.0, 0.0
    return vx/n, vy/n

def draw_linear_dim(painter, to_screen, p1: Point, p2: Point, offset: Point, color="#222", arrow_px=8, text="", text_px=12):
    # Compute projection line at "offset" (label line) using a perpendicular
    import math
    # Screen positions
    X1, Y1 = to_screen(*p1); X2, Y2 = to_screen(*p2); XO, YO = to_screen(*offset)
    # Direction along dimension
    dx, dy = (X2-X1, Y2-Y1)
    ux, uy = _unit(dx, dy)
    # Perp
    px, py = -uy, ux
    # Project offset point onto dimension line to build label segment symmetric around offset
    # Build a small segment +/- s around offset for the text baseline
    seg_half = 30  # px
    A = (XO - seg_half*ux, YO - seg_half*uy)
    B = (XO + seg_half*ux, YO + seg_half*uy)
    # Extension lines from p1,p2 perpendicular to label baseline
    # Feet points
    t1 = ((X1 - XO)*ux + (Y1 - YO)*uy)
    t2 = ((X2 - XO)*ux + (Y2 - YO)*uy)
    F1 = (XO + t1*ux, YO + t1*uy)
    F2 = (XO + t2*ux, YO + t2*uy)
    # Draw
    painter.line(F1[0], F1[1], X1, Y1, color, 1.0)
    painter.line(F2[0], F2[1], X2, Y2, color, 1.0)
    painter.line(A[0], A[1], B[0], B[1], color, 1.5)
    # Arrowheads
    painter.arrow(A[0], A[1], F1[0], F1[1], arrow_px, color)
    painter.arrow(B[0], B[1], F2[0], F2[1], arrow_px, color)
    # Text
    painter.text(text, XO, YO - 8, text_px, color, anchor="center", halo=True)

def draw_radial_dim(painter, to_screen, center: Point, attach: Point, color="#222", arrow_px=8, text="", text_px=12):
    CX, CY = to_screen(*center); AX, AY = to_screen(*attach)
    painter.line(CX, CY, AX, AY, color, 1.5)
    painter.arrow(AX, AY, CX, CY, arrow_px, color)
    painter.text(text, AX, AY - 8, text_px, color, anchor="center", halo=True)

def draw_angular_dim(painter, to_screen, vtx: Point, p1: Point, p2: Point, attach: Point, color="#222", text="", text_px=12):
    VX, VY = to_screen(*vtx); X1, Y1 = to_screen(*p1); X2, Y2 = to_screen(*p2); AX, AY = to_screen(*attach)
    painter.line(VX, VY, X1, Y1, color, 1.0)
    painter.line(VX, VY, X2, Y2, color, 1.0)
    painter.text(text, AX, AY - 8, text_px, color, anchor="center", halo=True)
