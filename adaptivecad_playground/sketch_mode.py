"""Lightweight sketching helpers shared by sketch-driven tools.

The playground remains fundamentally 2D, but revolve/lathe workflows benefit
from knowing which canonical plane a profile lives in. This module provides a
simple enum plus utilities to convert between 2D sketch coordinates and 3D
world points aligned to the selected sketch plane.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class SketchPlane(str, Enum):
    XY = "XY"
    YZ = "YZ"
    ZX = "ZX"


@dataclass(frozen=True)
class SketchFrame:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    normal: np.ndarray


_CANONICAL_FRAMES: dict[SketchPlane, SketchFrame] = {
    SketchPlane.XY: SketchFrame(
        origin=np.zeros(3, dtype=float),
        x_axis=np.array([1.0, 0.0, 0.0], dtype=float),
        y_axis=np.array([0.0, 1.0, 0.0], dtype=float),
        normal=np.array([0.0, 0.0, 1.0], dtype=float),
    ),
    SketchPlane.YZ: SketchFrame(
        origin=np.zeros(3, dtype=float),
        x_axis=np.array([0.0, 1.0, 0.0], dtype=float),
        y_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        normal=np.array([1.0, 0.0, 0.0], dtype=float),
    ),
    SketchPlane.ZX: SketchFrame(
        origin=np.zeros(3, dtype=float),
        x_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        y_axis=np.array([1.0, 0.0, 0.0], dtype=float),
        normal=np.array([0.0, 1.0, 0.0], dtype=float),
    ),
}


def frame_for_plane(plane: SketchPlane) -> SketchFrame:
    return _CANONICAL_FRAMES[plane]


def lift_point(point_2d: Sequence[float], plane: SketchPlane, *, origin: Sequence[float] | None = None) -> np.ndarray:
    frame = frame_for_plane(plane)
    o = frame.origin if origin is None else np.asarray(origin, dtype=float)
    x_axis = frame.x_axis
    y_axis = frame.y_axis
    x, y = float(point_2d[0]), float(point_2d[1])
    return o + x_axis * x + y_axis * y


def lift_polyline(points_2d: Iterable[Sequence[float]], plane: SketchPlane, *, origin: Sequence[float] | None = None) -> np.ndarray:
    pts = [lift_point(pt, plane, origin=origin) for pt in points_2d]
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def canonical_sketch_profiles() -> dict[str, dict[str, object]]:
    return {
        "vase_profile": {
            "plane": SketchPlane.ZX,
            "points": [
                (0.0, 0.0),
                (0.18, 0.1),
                (0.28, 0.4),
                (0.32, 0.75),
                (0.26, 1.15),
                (0.14, 1.4),
                (0.05, 1.55),
            ],
        }
    }


__all__ = [
    "SketchPlane",
    "SketchFrame",
    "frame_for_plane",
    "lift_point",
    "lift_polyline",
    "canonical_sketch_profiles",
]
