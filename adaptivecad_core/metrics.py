"""Metric helpers shared across headless and UI pipelines."""
from __future__ import annotations

from typing import Iterable, Protocol, Tuple

import numpy as np


class SampleProtocol(Protocol):
    def sample(self, point: Iterable[float]) -> Tuple[float, float]:
        ...


def arc_len_adaptive(curve_points: Iterable[Iterable[float]], fields: SampleProtocol | None, *, eta_k: float = 0.5, eta_m: float = 0.3) -> float:
    """Return a curvature and memory weighted arc length for a 3D polyline."""

    pts = np.asarray(list(curve_points), dtype=float)
    if pts.shape[0] < 2:
        return 0.0

    total = 0.0
    for i in range(pts.shape[0] - 1):
        p = pts[i]
        q = pts[i + 1]
        mid = 0.5 * (p + q)
        seg_len = float(np.linalg.norm(q - p))
        if seg_len == 0.0:
            continue
        w = 1.0
        if fields is not None:
            sampler = getattr(fields, "sample", None)
            if callable(sampler):
                kappa, memory = sampler(mid)
                w += float(eta_k) * float(kappa) + float(eta_m) * float(memory)
        total += w * seg_len
    return float(total)


__all__ = ["arc_len_adaptive"]
