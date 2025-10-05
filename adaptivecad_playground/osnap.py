# adaptivecad_playground/osnap.py
"""
Object-snap helpers covering common drafting aids (endpoint, midpoint,
quadrant, apparent intersection, closest/perpendicular projections).

The canvas advertises snap candidates via ``ctx.osnap_targets()`` which now
includes both literal points and lightweight geometry descriptors such as
segments or circles. ``osnap_pick`` evaluates these candidates against the
requested cursor location and returns the best snap point within a tolerance.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import math

Point = Tuple[float,float]
Target = Tuple[str, object]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _project_point(p: Point, a: Point, b: Point) -> Tuple[Point, float, float]:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return (ax, ay), 0.0, 0.0
    raw = ((p[0] - ax) * abx + (p[1] - ay) * aby) / denom
    t = max(0.0, min(1.0, raw))
    proj = (ax + abx * t, ay + aby * t)
    return proj, t, raw


def osnap_pick(p: Point, targets: Sequence[Target], tol: float = 8.0) -> Tuple[Point, Optional[str]]:
    """
    Return the closest snap point to ``p`` from the supplied targets.

    Targets can be simple tuples ``("end", (x, y))`` or geometry descriptors
    like ``("segment", ((x1, y1), (x2, y2)))`` and ``("circle", ((cx, cy), r))``.
    A snap is accepted only if it falls within ``tol`` units of ``p``.
    """
    if not targets:
        return (p, None)
    px, py = float(p[0]), float(p[1])
    best_point: Point | None = None
    best_dist = float(tol)

    best_kind: Optional[str] = None

    def consider(candidate: Point, kind: Optional[str]) -> None:
        nonlocal best_point, best_dist, best_kind
        dist = _distance((px, py), candidate)
        if dist <= best_dist:
            best_point = candidate
            best_dist = dist
            best_kind = kind

    for kind, payload in targets:
        if payload is None:
            continue
        if kind in {"end", "mid", "center", "quad", "dim", "appint", "closest", "perp", "90"}:
            qx, qy = payload  # type: ignore[misc]
            consider((float(qx), float(qy)), kind)
        elif kind == "segment":
            a, b = payload  # type: ignore[misc]
            proj, t, raw = _project_point((px, py), a, b)
            if raw < -0.05 or raw > 1.05:
                # Too far outside the segment to be useful.
                continue
            kind_label = "perp" if 1e-3 < t < 1 - 1e-3 else "closest"
            consider((float(proj[0]), float(proj[1])), kind_label)
        elif kind == "circle":
            center, radius = payload  # type: ignore[misc]
            cx, cy = float(center[0]), float(center[1])
            r = float(radius)
            if r <= 0.0:
                continue
            dx, dy = px - cx, py - cy
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                # Degenerate: fallback to positive-x quadrant point
                consider((cx + r, cy), "90")
            else:
                scale = r / length
                consider((cx + dx * scale, cy + dy * scale), "perp")
        else:
            # Unknown descriptor; ignore
            continue

    if best_point is None:
        return ((px, py), None)
    return (best_point, best_kind)
