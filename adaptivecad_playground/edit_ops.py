"""Core edit operations for AdaptiveCAD (join / break / trim / extend)."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Any
import math

from intersect2d import seg_seg_intersection, project_point_to_segment, seg_circle_intersections

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def join_polylines(paths: List[List[Point]], tol: float = 1e-6) -> List[List[Point]]:
    """Greedy merge of path endpoints within tolerance."""

    def endpoints(path):
        return path[0], path[-1]

    unused = [list(p) for p in paths if len(p) >= 2]
    out: List[List[Point]] = []
    while unused:
        cur = unused.pop(0)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(unused):
                p = unused[i]
                a0, a1 = endpoints(cur)
                b0, b1 = endpoints(p)
                if _dist(a1, b0) <= tol:
                    cur = cur + p[1:]
                    unused.pop(i)
                    changed = True
                    continue
                if _dist(a1, b1) <= tol:
                    cur = cur + list(reversed(p[:-1]))
                    unused.pop(i)
                    changed = True
                    continue
                if _dist(a0, b1) <= tol:
                    cur = p + cur[1:]
                    unused.pop(i)
                    changed = True
                    continue
                if _dist(a0, b0) <= tol:
                    cur = list(reversed(p)) + cur[1:]
                    unused.pop(i)
                    changed = True
                    continue
                i += 1
        out.append(cur)
    return out


def break_segment_at_point(seg: Segment, pt: Point, tol: float = 1e-6) -> Optional[List[Segment]]:
    """Split segment at the click point if projection falls within tolerance."""
    a, b = seg
    proj, t = project_point_to_segment(pt, a, b)
    if _dist(proj, pt) > tol or t <= 0.0 or t >= 1.0:
        return None
    return [(a, proj), (proj, b)]


def _iter_segments_of(target: Any) -> Iterable[Segment]:
    if isinstance(target, tuple) and len(target) == 2 and isinstance(target[0], tuple):
        # Raw segment tuple
        yield target
        return
    if isinstance(target, dict) and target.get("type") in ("polyline", "curve"):
        pts = target["points"]
        for i in range(len(pts) - 1):
            yield (tuple(pts[i]), tuple(pts[i + 1]))
        return
    if isinstance(target, dict) and target.get("type") == "segment":
        yield (tuple(target["p1"]), tuple(target["p2"]))
        return


def trim_segment_to(seg: Segment, cutter: Any, tol: float = 1e-6) -> Optional[Segment]:
    """Shorten ``seg`` to the nearest intersection with ``cutter``."""
    a, b = seg
    hits: List[Point] = []
    for s in _iter_segments_of(cutter):
        h = seg_seg_intersection(a, b, s[0], s[1])
        if h is not None:
            hits.append(h)
    if isinstance(cutter, dict) and cutter.get("type") == "piacircle":
        hits += seg_circle_intersections(a, b, tuple(cutter["center"]), float(cutter["radius"]))
    if not hits:
        return None
    dists = [(_dist(a, h), "a", h) for h in hits] + [(_dist(b, h), "b", h) for h in hits]
    _, which, h = min(dists, key=lambda t: t[0])
    return (h, b) if which == "a" else (a, h)


def extend_segment_to(seg: Segment, target: Any, tol: float = 1e-6) -> Optional[Segment]:
    """Extend ``seg`` until it meets ``target``."""
    a, b = seg
    hits: List[Point] = []
    for s in _iter_segments_of(target):
        h = seg_seg_intersection(a, b, s[0], s[1])
        if h is not None:
            hits.append(h)
    if isinstance(target, dict) and target.get("type") == "piacircle":
        hits += seg_circle_intersections(a, b, tuple(target["center"]), float(target["radius"]))
    if not hits:
        return None
    dists = [(_dist(a, h), "a", h) for h in hits] + [(_dist(b, h), "b", h) for h in hits]
    _, which, h = min(dists, key=lambda t: t[0])
    return (h, b) if which == "a" else (a, h)
