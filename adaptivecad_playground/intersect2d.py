"""2D intersection helpers for AdaptiveCAD edit operations."""
from __future__ import annotations

from typing import List, Optional, Tuple
import math

Point = Tuple[float, float]
EPS = 1e-9


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def seg_seg_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Optional[Point]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < EPS:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / den
    if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def project_point_to_segment(p: Point, a: Point, b: Point):
    ab = sub(b, a)
    ab2 = dot(ab, ab)
    if ab2 < EPS:
        return a, 0.0
    t = dot(sub(p, a), ab) / ab2
    t = max(0.0, min(1.0, t))
    return add(a, mul(ab, t)), t


def line_circle_intersections(p1: Point, p2: Point, c: Point, r: float) -> List[Point]:
    x1, y1 = p1[0] - c[0], p1[1] - c[1]
    x2, y2 = p2[0] - c[0], p2[1] - c[1]
    dx, dy = x2 - x1, y2 - y1
    dr2 = dx * dx + dy * dy
    D = x1 * y2 - x2 * y1
    disc = r * r * dr2 - D * D
    pts: List[Point] = []
    if disc < -EPS:
        return pts
    sqrt_disc = math.sqrt(max(0.0, disc))
    sgn = 1.0 if dy >= 0 else -1.0
    for sd in (+sqrt_disc, -sqrt_disc):
        x = (D * dy + sgn * dx * sd) / dr2
        y = (-D * dx + abs(dy) * sd) / dr2
        pts.append((x + c[0], y + c[1]))
    if len(pts) == 2 and abs(pts[0][0] - pts[1][0]) < 1e-12 and abs(pts[0][1] - pts[1][1]) < 1e-12:
        pts = [pts[0]]
    return pts


def seg_circle_intersections(p1: Point, p2: Point, c: Point, r: float) -> List[Point]:
    pts: List[Point] = []
    for s in line_circle_intersections(p1, p2, c, r):
        proj, t = project_point_to_segment(s, p1, p2)
        if (proj[0] - s[0]) ** 2 + (proj[1] - s[1]) ** 2 < 1e-14 and -EPS <= t <= 1 + EPS:
            pts.append(s)
    out: List[Point] = []
    for p in pts:
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 1e-18 for q in out):
            out.append(p)
    return out


def circle_circle_intersections(c1: Point, r1: float, c2: Point, r2: float) -> List[Point]:
    d = norm(sub(c2, c1))
    if d < EPS or d > r1 + r2 + EPS or d < abs(r1 - r2) - EPS:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h2 = r1 * r1 - a * a
    if h2 < -EPS:
        return []
    h = math.sqrt(max(0.0, h2))
    x0 = c1[0] + a * (c2[0] - c1[0]) / d
    y0 = c1[1] + a * (c2[1] - c1[1]) / d
    rx = -(c2[1] - c1[1]) * (h / d)
    ry = (c2[0] - c1[0]) * (h / d)
    p_a = (x0 + rx, y0 + ry)
    p_b = (x0 - rx, y0 - ry)
    return [p_a] if h < 1e-12 else [p_a, p_b]
