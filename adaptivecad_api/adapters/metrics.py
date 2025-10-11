"""Metrics helpers that surface core measurements through the API."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

try:
    from adaptivecad_core.metrics import arc_len_adaptive
    from adaptivecad_core.fields import FieldPack as _FieldPack
except ModuleNotFoundError:  # pragma: no cover
    arc_len_adaptive = None  # type: ignore[assignment]
    _FieldPack = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from adaptivecad_core.fields import FieldPack
else:  # pragma: no cover
    FieldPack = Any  # type: ignore[assignment]


def _ensure_curve(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("Curve must contain at least two points")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[1] == 2:
        zeros = np.zeros((arr.shape[0], 1), dtype=float)
        return np.hstack([arr, zeros])
    raise ValueError("Curve points must be 2D or 3D")


def _build_field(field_spec: Dict[str, Any] | None):
    if _FieldPack is None or not field_spec:
        return None
    vertices = field_spec.get("vertices")
    faces = field_spec.get("faces", [])
    if vertices is None:
        return None
    beta = field_spec.get("beta", 0.1)
    gamma = field_spec.get("gamma", 0.05)
    return _FieldPack(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int32), beta=beta, gamma=gamma)


def measure(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return requested measurements for curve selections using adaptive metrics."""

    if arc_len_adaptive is None:
        raise RuntimeError("adaptivecad_core.metrics.arc_len_adaptive unavailable")

    curves = payload.get("curves") or payload.get("selection") or []
    if not isinstance(curves, list):
        raise ValueError("Payload 'curves' must be a list")

    eta_k = float(payload.get("eta_k", 0.5))
    eta_m = float(payload.get("eta_m", 0.3))
    field = _build_field(payload.get("field"))

    measurements: Dict[str, Dict[str, float]] = {}
    for idx, entry in enumerate(curves):
        if isinstance(entry, dict):
            pts = entry.get("points") or entry.get("vertices")
            identifier = entry.get("id") or entry.get("name") or f"curve-{idx}"
        else:
            pts = entry
            identifier = f"curve-{idx}"
        if pts is None:
            raise ValueError(f"Curve entry {idx} missing 'points'")
        curve_arr = _ensure_curve(pts)
        length = arc_len_adaptive(curve_arr, field, eta_k=eta_k, eta_m=eta_m)
        measurements[identifier] = {"arc_length": float(length)}

    return {"status": "ok", "measurements": measurements}
