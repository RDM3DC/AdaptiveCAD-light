"""Adaptive field utilities supporting curvature, memory, and πₐ evaluation.

The playground renders lightweight approximations so tools can query spatial
metrics without running full PDE solvers. These classes synthesize continuous
fields from the discrete shapes stored on the canvas. The fields are rebuilt
whenever geometry changes, allowing downstream consumers (dimensions, tools,
status readouts) to sample curvature-adaptive behaviour at arbitrary points.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from geometry import pi_a as _pi_a

Point = Tuple[float, float]


@dataclass
class CurvatureSample:
    position: Point
    curvature: float
    weight: float
    source_id: Optional[str] = None


@dataclass
class MemorySample:
    position: Point
    magnitude: float
    weight: float
    source_id: Optional[str] = None


def _as_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(points, np.ndarray):
        return np.asarray(points, dtype=float)
    return np.array(points, dtype=float)


class CurvatureField:
    """Aggregate discrete curvature samples from canvas shapes."""

    def __init__(self, smoothing_radius: float = 60.0):
        self._sigma = max(1.0, float(smoothing_radius))
        self._samples: List[CurvatureSample] = []
        self._samples_by_shape: dict[str | None, List[CurvatureSample]] = {}

    @property
    def samples(self) -> List[CurvatureSample]:
        return list(self._samples)

    def rebuild(self, shapes: Iterable[object]) -> None:
        self._samples.clear()
        self._samples_by_shape.clear()
        for shape in shapes:
            points = getattr(shape, "points", None)
            if points is None:
                continue
            array = _as_array(points)
            if array.shape[0] < 3:
                continue
            closed = bool(array.shape[0] >= 4 and np.linalg.norm(array[0] - array[-1]) < 1e-6)
            if closed:
                array = array[:-1]
                if array.shape[0] < 3:
                    continue
            padded = np.vstack((array[0], array, array[-1])) if not closed else np.vstack((array[-1], array, array[0]))
            forward = padded[2:] - padded[1:-1]
            backward = padded[1:-1] - padded[:-2]
            len_f = np.linalg.norm(forward, axis=1)
            len_b = np.linalg.norm(backward, axis=1)
            mask = (len_f > 1e-6) & (len_b > 1e-6)
            curvature = np.zeros(len(array), dtype=float)
            if np.any(mask):
                t1 = np.zeros_like(forward)
                t2 = np.zeros_like(forward)
                t1[mask] = backward[mask] / len_b[mask][:, None]
                t2[mask] = forward[mask] / len_f[mask][:, None]
                cross = t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0]
                dot = np.clip((t1 * t2).sum(axis=1), -1.0, 1.0)
                angle = np.arctan2(cross, dot)
                avg_len = (len_f + len_b) * 0.5
                denom = np.maximum(avg_len, 1e-6)
                curvature[mask] = angle[mask] / denom[mask]
            shape_id = getattr(shape, "id", None)
            weights = (len_f + len_b) * 0.5
            weights = np.where(np.isfinite(weights), np.maximum(weights, 1e-6), 1.0)
            for idx, pos in enumerate(array):
                sample = CurvatureSample(position=(float(pos[0]), float(pos[1])), curvature=float(curvature[idx]), weight=float(weights[min(idx, len(weights) - 1)]), source_id=shape_id)
                self._samples.append(sample)
                self._samples_by_shape.setdefault(shape_id, []).append(sample)

    def sample(self, point: Point, radius: Optional[float] = None) -> float:
        if not self._samples:
            return 0.0
        sigma = float(radius) if radius is not None else self._sigma
        sigma = max(1.0, sigma)
        denom = 0.0
        accum = 0.0
        px, py = float(point[0]), float(point[1])
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        for sample in self._samples:
            dx = px - sample.position[0]
            dy = py - sample.position[1]
            weight = math.exp(-(dx * dx + dy * dy) * inv_two_sigma_sq) * sample.weight
            denom += weight
            accum += sample.curvature * weight
        if denom <= 1e-9:
            return 0.0
        return accum / denom

    def aggregate_for_shape(self, shape_id: Optional[str]) -> float:
        entries = self._samples_by_shape.get(shape_id)
        if not entries:
            return 0.0
        total_weight = sum(sample.weight for sample in entries)
        if total_weight <= 1e-9:
            return 0.0
        return sum(sample.curvature * sample.weight for sample in entries) / total_weight


class MaterialMemoryField:
    """Diffuse accumulated curvature magnitude for downstream reinforcement."""

    def __init__(self, smoothing_radius: float = 90.0):
        self._sigma = max(1.0, float(smoothing_radius))
        self._samples: List[MemorySample] = []

    def rebuild_from_curvature(self, curvature: CurvatureField) -> None:
        self._samples = [
            MemorySample(position=sample.position, magnitude=abs(sample.curvature), weight=sample.weight, source_id=sample.source_id)
            for sample in curvature.samples
        ]

    def sample(self, point: Point, radius: Optional[float] = None) -> float:
        if not self._samples:
            return 0.0
        sigma = float(radius) if radius is not None else self._sigma
        sigma = max(1.0, sigma)
        denom = 0.0
        accum = 0.0
        px, py = float(point[0]), float(point[1])
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        for sample in self._samples:
            dx = px - sample.position[0]
            dy = py - sample.position[1]
            weight = math.exp(-(dx * dx + dy * dy) * inv_two_sigma_sq) * sample.weight
            denom += weight
            accum += sample.magnitude * weight
        if denom <= 1e-9:
            return 0.0
        return accum / denom


class AdaptiveMetricManager:
    """Owns curvature/material fields and exposes πₐ sampling helpers."""

    def __init__(self, curvature_radius: float = 60.0, memory_radius: float = 90.0):
        self._curvature = CurvatureField(curvature_radius)
        self._memory = MaterialMemoryField(memory_radius)

    def rebuild(self, shapes: Iterable[object]) -> None:
        self._curvature.rebuild(shapes)
        self._memory.rebuild_from_curvature(self._curvature)

    def curvature(self, point: Point, radius: Optional[float] = None) -> float:
        return self._curvature.sample(point, radius)

    def memory(self, point: Point, radius: Optional[float] = None) -> float:
        return self._memory.sample(point, radius)

    def pi_a(self, point: Point, radius: float, params: Optional[dict] = None) -> float:
        params = params or {}
        alpha = float(params.get("alpha", params.get("α", 0.0)) or 0.0)
        mu = float(params.get("mu", params.get("μ", 0.0)) or 0.0)
        k0 = float(params.get("k0", params.get("k₀", 0.0)) or 0.0)
        base = _pi_a(radius, alpha=alpha, mu=mu, k0=k0)
        curvature = self.curvature(point)
        memory_term = self.memory(point)
        reinforcement = 1.0 + alpha * curvature
        relaxation = math.exp(-abs(mu) * memory_term)
        return base * reinforcement * relaxation

    def aggregate_curvature_for_shape(self, shape_id: Optional[str]) -> float:
        return self._curvature.aggregate_for_shape(shape_id)
