"""Geometry helpers for the AdaptiveCAD Playground.

The numerical routines in this module are lightweight approximations that keep
interactivity snappy while still reflecting the adaptive π behaviour that drives
the demo. They can be swapped out for higher fidelity kernels later on.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _normalize_params(params: dict | None) -> dict:
    """Return a params dict with ASCII keys and default values."""
    params = params or {}
    alpha = params.get("alpha", params.get("α", 0.0))
    mu = params.get("mu", params.get("μ", 0.0))
    k0 = params.get("k0", params.get("k₀", 0.0))
    return {"alpha": float(alpha or 0.0), "mu": float(mu or 0.0), "k0": float(k0 or 0.0)}


def pi_a(radius: float, alpha: float = 0.0, mu: float = 0.0, k0: float = 0.0) -> float:
    """Adaptive π kernel placeholder.

    The model gently scales the baseline π by the curvature seed ``k0`` and
    damps it with an exponential decay controlled by ``mu``. ``alpha`` is wired
    in for future reinforcement feedback terms.
    """
    base = 1.0 + k0 * radius
    decay = math.exp(-mu * radius)
    return math.pi * base * decay + alpha * 0.0


def circle_points(center: Sequence[float], radius: float, params: dict | None, samples: int = 256) -> np.ndarray:
    """Sample the warped circle as a polyline."""
    params = _normalize_params(params)
    angle = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=True)
    local_pi = pi_a(radius, **params)
    warped_angle = angle * (local_pi / math.pi)
    cx, cy = float(center[0]), float(center[1])
    x = cx + radius * np.cos(warped_angle)
    y = cy + radius * np.sin(warped_angle)
    return np.column_stack((x, y))


def arc_length(radius: float, params: dict | None) -> float:
    """Approximate the arc length of the warped circle."""
    samples = circle_points((0.0, 0.0), radius, params, samples=2048)
    return polyline_length(samples)


def area(radius: float, params: dict | None) -> float:
    """Approximate the area enclosed by the warped circle."""
    samples = circle_points((0.0, 0.0), radius, params, samples=2048)
    return polygon_area(samples)


def _bezier_sample(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a Bezier curve of arbitrary degree at parameter values ``t``."""
    if control_points.shape[0] == 0:
        return np.zeros((len(t), 2))
    if control_points.shape[0] == 1:
        return np.repeat(control_points, len(t), axis=0)
    pts = np.broadcast_to(control_points, (len(t),) + control_points.shape).copy()
    for _ in range(1, control_points.shape[0]):
        pts = (1.0 - t)[:, None, None] * pts[:, :-1, :] + t[:, None, None] * pts[:, 1:, :]
    return pts[:, 0, :]


def curve_points(control_points: Iterable[Sequence[float]], params: dict | None, samples: int = 256) -> np.ndarray:
    """Sample a Bezier-like adaptive curve defined by ``control_points``."""
    control_points = np.asarray(list(control_points), dtype=float)
    if control_points.shape[0] < 2:
        return control_points
    params_norm = _normalize_params(params)
    span = np.linalg.norm(control_points[-1] - control_points[0])
    local_pi = max(pi_a(span if span > 1e-6 else 1.0, **params_norm), 1e-6)
    ratio = local_pi / math.pi
    exponent = float(np.clip(ratio, 0.5, 1.5))
    t = np.linspace(0.0, 1.0, samples)
    warped_t = np.clip(t ** exponent, 0.0, 1.0)
    return _bezier_sample(control_points, warped_t)


def curve_arc_length(control_points: Iterable[Sequence[float]], params: dict | None, samples: int = 1024) -> float:
    """Approximate the arc length of the adaptive curve."""
    pts = curve_points(control_points, params, samples=samples)
    return polyline_length(pts)


def curve_area(control_points: Iterable[Sequence[float]], params: dict | None, samples: int = 1024) -> float:
    """Approximate the signed area enclosed by closing the curve."""
    pts = curve_points(control_points, params, samples=samples)
    return polygon_area(pts)


def polyline_length(points: np.ndarray) -> float:
    """Return the cumulative length of a polyline."""
    if points.size == 0:
        return 0.0
    delta = np.diff(points, axis=0)
    seg = np.hypot(delta[:, 0], delta[:, 1])
    return float(np.sum(seg))


def polygon_area(points: np.ndarray) -> float:
    """Return the absolute area spanned by a closed polygon."""
    if points.size == 0:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_to_polyline_distance(point: Sequence[float], polyline: np.ndarray) -> float:
    """Compute the minimum distance from ``point`` to the given ``polyline``."""
    if polyline.shape[0] == 0:
        return float("inf")
    if polyline.shape[0] == 1:
        return float(np.hypot(point[0] - polyline[0, 0], point[1] - polyline[0, 1]))
    seg_vec = polyline[1:] - polyline[:-1]
    seg_len_sq = np.sum(seg_vec ** 2, axis=1)
    to_point = np.asarray(point, dtype=float) - polyline[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sum(to_point * seg_vec, axis=1) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    projection = polyline[:-1] + seg_vec * t[:, None]
    dist = np.hypot(point[0] - projection[:, 0], point[1] - projection[:, 1])
    return float(np.min(dist))


# ---------------------------------------------------------------------------
# Dimension helpers


def euclid_len(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Return the Euclidean distance between two points."""
    return float(math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])))


def angle_deg(vertex: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> float:
    """Return the absolute angle between the rays (vertex->p1) and (vertex->p2)."""
    v1x = float(p1[0]) - float(vertex[0])
    v1y = float(p1[1]) - float(vertex[1])
    v2x = float(p2[0]) - float(vertex[0])
    v2y = float(p2[1]) - float(vertex[1])
    a1 = math.atan2(v1y, v1x)
    a2 = math.atan2(v2y, v2x)
    delta = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(delta))


def format_length(length: float, units: str, precision: int) -> str:
    suffix = f" {units}" if units else ""
    return f"{length:.{precision}f}{suffix}"


def radial_label(radius: float, params: dict | None, style) -> str:
    """Return a formatted radial label string honoring the style mode."""
    precision = int(getattr(style, "precision", 2))
    units = str(getattr(style, "units", ""))
    mode = getattr(style, "mode", "both")
    suffix = f" {units}" if units else ""

    parts: list[str] = [f"R={radius:.{precision}f}{suffix}"]

    local_params = _normalize_params(params)
    circ_pi_a = arc_length(radius, local_params)
    circ_euclid = 2.0 * math.pi * radius

    if mode == "pi_a":
        parts.append(f"Cₐ={circ_pi_a:.{precision}f}{suffix}")
    elif mode == "both":
        parts.append(f"C={circ_euclid:.{precision}f}{suffix}")
        if abs(circ_pi_a - circ_euclid) > 10 ** (-precision):
            parts.append(f"Cₐ={circ_pi_a:.{precision}f}{suffix}")
    # Euclid mode omits circumference extras
    return " · ".join(parts)
