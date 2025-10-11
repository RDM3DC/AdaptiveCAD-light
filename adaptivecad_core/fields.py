"""Curvature and memory field helpers for adaptive kernels.

This module mirrors the lightweight ``FieldPack`` shipping in the standalone
revolve bundle, but adapts the API to integrate with the rest of the
AdaptiveCAD-light codebase. The class supplies ``sample`` / ``accumulate``
hooks so callers such as :func:`adaptivecad_core.revolve.revolve_adaptive`
can tighten the seam and persist memory feedback without depending on the
playground's heavier GPU field stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

EPS = 1e-12


def _cotangent(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the cotangent of the angle at ``a`` for triangle ``(a, b, c)``."""

    u = b - a
    v = c - a
    cross_n = np.cross(u, v)
    area2 = np.linalg.norm(cross_n)
    if area2 < EPS:
        return 0.0
    dot_uv = float(np.dot(u, v))
    return dot_uv / (area2 + EPS)


def _per_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return barycentric vertex areas for the mesh."""

    count = vertices.shape[0]
    accum = np.zeros(count, dtype=float)
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        area = 0.5 * np.linalg.norm(np.cross(vj - vi, vk - vi))
        if not np.isfinite(area):
            area = 0.0
        share = area / 3.0
        accum[i] += share
        accum[j] += share
        accum[k] += share
    return accum


def compute_mean_curvature_magnitude(vertices: Iterable[Iterable[float]], faces: Iterable[Iterable[int]]) -> np.ndarray:
    """Discrete mean-curvature magnitude using the cotangent Laplacian."""

    V = np.asarray(vertices, dtype=float)
    F = np.asarray(list(faces), dtype=np.int32)
    if F.size == 0:
        return np.zeros(len(V), dtype=float)

    L = np.zeros((len(V), 3), dtype=float)
    A = _per_vertex_areas(V, F) + EPS

    for tri in F:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        vi, vj, vk = V[i], V[j], V[k]

        cot_i = _cotangent(vi, vj, vk)
        cot_j = _cotangent(vj, vk, vi)
        cot_k = _cotangent(vk, vi, vj)

        w_ij = cot_k
        L[i] += w_ij * (vj - vi)
        L[j] += w_ij * (vi - vj)

        w_jk = cot_i
        L[j] += w_jk * (vk - vj)
        L[k] += w_jk * (vj - vk)

        w_ki = cot_j
        L[k] += w_ki * (vi - vk)
        L[i] += w_ki * (vk - vi)

    mean_curv_normal = L / (2.0 * A[:, None])
    kappa = np.linalg.norm(mean_curv_normal, axis=1)

    if np.max(kappa) > 0.0:
        p95 = float(np.percentile(kappa, 95.0))
        scale = p95 if p95 > EPS else float(np.max(kappa) + EPS)
        kappa = np.clip(kappa / (scale + EPS), 0.0, 1.0)
    return kappa.astype(float, copy=False)


@dataclass
class FieldPack:
    """Lightweight curvature and memory container used by headless tools."""

    vertices: np.ndarray
    faces: np.ndarray
    beta: float = 0.1
    gamma: float = 0.05

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        self.beta = float(self.beta)
        self.gamma = float(self.gamma)
        self.kappa = compute_mean_curvature_magnitude(self.vertices, self.faces)
        self.memory = np.zeros(len(self.vertices), dtype=float)

    def nearest_vertex_index(self, point: Iterable[float]) -> int:
        diff = self.vertices - np.asarray(point, dtype=float)
        dist2 = np.einsum("ij,ij->i", diff, diff)
        return int(np.argmin(dist2))

    def sample(self, point: Iterable[float]) -> tuple[float, float]:
        idx = self.nearest_vertex_index(point)
        return float(self.kappa[idx]), float(self.memory[idx])

    def pi_a(self, point: Iterable[float]) -> float:
        kappa, mem = self.sample(point)
        return float(np.pi * (1.0 + self.beta * kappa + self.gamma * mem))

    def write_memory(self, indices: Iterable[int] | int, delta: float = 0.05) -> None:
        if indices is None:
            return
        if isinstance(indices, (list, tuple, np.ndarray)):
            for idx in indices:
                self._bump_index(int(idx), delta)
        else:
            self._bump_index(int(indices), delta)

    def _bump_index(self, idx: int, delta: float) -> None:
        if 0 <= idx < len(self.memory):
            self.memory[idx] += float(delta)

    def accumulate(self, point: Iterable[float], amount: float) -> None:
        idx = self.nearest_vertex_index(point)
        self._bump_index(idx, amount)

    def decay_memory(self, rate: float = 0.0) -> None:
        if rate <= 0.0:
            return
        self.memory *= (1.0 - float(rate))


__all__ = ["FieldPack", "compute_mean_curvature_magnitude"]
