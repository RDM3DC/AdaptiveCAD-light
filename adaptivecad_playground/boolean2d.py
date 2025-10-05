"""2D polygon boolean helpers (union / difference / intersection)."""
from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]


try:  # shapely is optional; hold a reference if available
    import shapely.geometry as _SG  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency path
    _SG = None


def has_shapely() -> bool:
    return _SG is not None


def poly_union(a: List[Point], b: List[Point]) -> List[List[Point]]:
    if has_shapely():
        g = _SG.Polygon(a).buffer(0.0).union(_SG.Polygon(b).buffer(0.0))
        return _explode_polygons(g)
    raise NotImplementedError("Union requires shapely (fallback planned for v0.2).")


def poly_difference(a: List[Point], b: List[Point]) -> List[List[Point]]:
    if has_shapely():
        g = _SG.Polygon(a).buffer(0.0).difference(_SG.Polygon(b).buffer(0.0))
        return _explode_polygons(g)
    raise NotImplementedError("Difference requires shapely (fallback planned for v0.2).")


def poly_intersection(a: List[Point], b: List[Point]) -> List[List[Point]]:
    if has_shapely():
        g = _SG.Polygon(a).buffer(0.0).intersection(_SG.Polygon(b).buffer(0.0))
        return _explode_polygons(g)
    return _suthodg_clip(a, b)


def _explode_polygons(geom) -> List[List[Point]]:
    sg = _SG
    if sg is None:
        return []
    polys: List[List[Point]] = []
    if isinstance(geom, sg.Polygon):
        polys.append(list(geom.exterior.coords)[:-1])
    else:
        for part in geom.geoms:
            if isinstance(part, sg.Polygon):
                polys.append(list(part.exterior.coords)[:-1])
    return [[(float(x), float(y)) for (x, y) in poly] for poly in polys]


# --- Convex-clip fallback for intersection ---

def _inside(p: Point, a: Point, b: Point) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0


def _isect(s: Point, e: Point, a: Point, b: Point) -> Point:
    x1, y1 = s
    x2, y2 = e
    x3, y3 = a
    x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den if abs(den) > 1e-12 else 1.0
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _suthodg_clip(subject: List[Point], clip: List[Point]) -> List[List[Point]]:
    out = subject[:]
    cp1 = clip[-1]
    for cp2 in clip:
        inp = out
        out = []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    out.append(_isect(s, e, cp1, cp2))
                out.append(e)
            elif _inside(s, cp1, cp2):
                out.append(_isect(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    return [out] if out else []
