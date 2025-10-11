"""Placeholder adaptive cut/add/join helpers from the revolve bundle.

These routines are intentionally lightweight and serve as scaffolding for
future π-aware boolean and stitching workflows. They operate purely on numpy
arrays so they can be exercised in notebooks or headless demos without pulling
in Blender.
"""
from __future__ import annotations

import numpy as np


def adaptive_cut(surface_vertices, surface_faces, seed_a, seed_b, fields, *, lam_k: float = 0.4, lam_m: float = 0.2, steps: int = 64) -> np.ndarray:
    """Return a memory-weighted straight-line seam between two seeds."""

    pa = np.asarray(seed_a, dtype=float)
    pb = np.asarray(seed_b, dtype=float)
    path = [pa + t * (pb - pa) for t in np.linspace(0.0, 1.0, steps)]
    weights = [1.0 + lam_k * fields.sample(p)[0] + lam_m * fields.sample(p)[1] for p in path]
    seam = [p / max(w, 1e-6) for p, w in zip(path, weights)]
    for point in seam:
        idx = fields.nearest_vertex_index(point)
        kappa = fields.kappa[idx]
        fields.memory[idx] += 0.05 * (1.0 + abs(kappa))
    return np.asarray(seam, dtype=float)


def adaptive_add(vertices, direction, fields, *, rho: float = 0.5, t0: float = 1.0) -> np.ndarray:
    """Offset vertices along ``direction`` inversely proportional to curvature."""

    V = np.asarray(vertices, dtype=float)
    direction = np.asarray(direction, dtype=float)
    out = []
    for v in V:
        kappa, _ = fields.sample(v)
        t = t0 / (1.0 + rho * abs(kappa))
        out.append(v + direction * t)
    return np.asarray(out, dtype=float)


def adaptive_join(vertices_a, vertices_b, fields, *, lam: float = 0.5) -> np.ndarray:
    """Blend two vertex sets using curvature and memory weighting."""

    VA = np.asarray(vertices_a, dtype=float)
    VB = np.asarray(vertices_b, dtype=float)
    seam = []
    for a in VA:
        d2 = np.einsum("ij,ij->i", VB - a, VB - a)
        j = int(np.argmin(d2))
        b = VB[j]
        kappa, memory = fields.sample(0.5 * (a + b))
        weight = 1.0 / (1.0 + lam * (kappa + memory))
        seam.append(a * (1.0 - weight) + b * weight)
    for point in seam:
        idx = fields.nearest_vertex_index(point)
        fields.memory[idx] += 0.05
    return np.asarray(seam, dtype=float)


__all__ = ["adaptive_cut", "adaptive_add", "adaptive_join"]
