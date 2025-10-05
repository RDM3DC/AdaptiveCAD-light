# adaptivecad_playground/osnap.py
"""
Minimal object-snap helpers (end/mid/center/quad).

Your Canvas should expose osnap targets via ctx.osnap_targets() returning
[("end", (x,y)), ("mid",(x,y)), ("center",(x,y)), ...].
"""

from __future__ import annotations
from typing import List, Tuple
import math

Point = Tuple[float,float]

def osnap_pick(p: Point, targets: List[Tuple[str, Point]], tol: float = 8.0) -> Point:
    """Return the closest snap target in screen/world space hybrid.
    For v0.1 we assume 'targets' are in world coords and the Canvas pre-filters by proximity.
    """
    if not targets:
        return p
    best = None; best_d = 1e9
    for name, q in targets:
        d = math.hypot(p[0]-q[0], p[1]-q[1])
        if d < best_d:
            best_d = d; best = q
    return best if best is not None else p
