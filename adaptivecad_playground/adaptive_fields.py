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
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:  # Optional GPU acceleration
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - fallback when CuPy unavailable
    cp = None

from geometry import pi_a as _pi_a
from geometry import circle_points as _circle_points, curve_points as _curve_points

Point = Tuple[float, float]


@dataclass
class FieldPack:
    """Compact cache of sampled curvature/memory data living on CPU or GPU."""

    positions: np.ndarray
    curvature: np.ndarray
    memory: np.ndarray
    weights: np.ndarray
    device: Optional[int] = None

    @classmethod
    def empty(cls) -> "FieldPack":
        zeros_vec = np.zeros((0, 2), dtype=float)
        zeros = np.zeros(0, dtype=float)
        return cls(zeros_vec, zeros, zeros.copy(), zeros.copy(), None)

    def sample_count(self) -> int:
        return int(self.positions.shape[0]) if self.positions.size else 0

    def has_data(self) -> bool:
        return self.sample_count() > 0

    def average_curvature(self) -> float:
        if not self.has_data():
            return 0.0
        xp = self._module()
        denom = xp.sum(self.weights)
        denom_f = self._scalar_to_float(denom, xp)
        if denom_f <= 1e-9:
            return 0.0
        value = xp.dot(self.curvature, self.weights)
        return self._scalar_to_float(value, xp) / denom_f

    def average_memory(self) -> float:
        if not self.has_data():
            return 0.0
        xp = self._module()
        denom = xp.sum(self.weights)
        denom_f = self._scalar_to_float(denom, xp)
        if denom_f <= 1e-9:
            return 0.0
        value = xp.dot(self.memory, self.weights)
        return self._scalar_to_float(value, xp) / denom_f

    # ------------------------------------------------------------------
    # Helpers

    def _module(self):
        if cp is not None and isinstance(self.positions, cp.ndarray):  # type: ignore[arg-type]
            return cp
        return np

    @staticmethod
    def _scalar_to_float(value, module) -> float:
        if module is cp:  # type: ignore
            return float(cp.asnumpy(value))
        return float(value)


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
        self._arrays_by_shape: dict[str | None, tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = {}
        self._use_gpu = False
        self._gpu_devices: List[int] = []

    def configure_gpu(self, devices: Sequence[int], use_gpu: bool) -> None:
        if cp is None or not devices:
            self._use_gpu = False
            self._gpu_devices = []
            return
        self._use_gpu = bool(use_gpu)
        self._gpu_devices = list(devices) if self._use_gpu else []

    @property
    def samples(self) -> List[CurvatureSample]:
        return list(self._samples)

    def rebuild(self, shapes: Iterable[object]) -> None:
        self._samples.clear()
        self._samples_by_shape.clear()
        self._arrays_by_shape.clear()
        shape_list = list(shapes)
        if not shape_list:
            return
        prepared: List[tuple[int, Optional[str], np.ndarray, bool]] = []
        for idx_shape, shape in enumerate(shape_list):
            prepared_shape = self._prepare_shape_array(shape)
            if prepared_shape is None:
                continue
            shape_id, array, closed = prepared_shape
            prepared.append((idx_shape, shape_id, array, closed))
        if not prepared:
            return

        gpu_devices = self._gpu_devices if (self._use_gpu and self._gpu_devices) else []
        gpu_count = len(gpu_devices)

        if gpu_count <= 1:
            device_id = gpu_devices[0] if gpu_count == 1 else None
            for _, shape_id, array, closed in prepared:
                positions, curvature, weights = self._compute_curvature(array, closed, device_id)
                self._store_shape_samples(shape_id, positions, curvature, weights, device_id)
            return

        assignments: dict[int, List[tuple[int, Optional[str], np.ndarray, bool]]] = {device: [] for device in gpu_devices}
        for entry in prepared:
            idx_shape, shape_id, array, closed = entry
            device_id = gpu_devices[idx_shape % gpu_count]
            assignments[device_id].append(entry)

        results: List[tuple[int, Optional[str], np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            futures = [
                executor.submit(self._process_batch, batch, device_id)
                for device_id, batch in assignments.items()
                if batch
            ]
            for future in futures:
                results.extend(future.result())

        for idx_shape, shape_id, positions, curvature, weights, device_id in sorted(results, key=lambda item: item[0]):
            self._store_shape_samples(shape_id, positions, curvature, weights, device_id)

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
        arrays = self._arrays_by_shape.get(shape_id)
        if not arrays:
            return 0.0
        _, curvature, weights, device_id = arrays
        xp = cp if (cp is not None and isinstance(curvature, cp.ndarray)) else np  # type: ignore[arg-type]
        total_weight = xp.sum(weights)
        total_weight_f = float(cp.asnumpy(total_weight)) if xp is cp else float(total_weight)
        if total_weight_f <= 1e-9:
            return 0.0
        value = xp.dot(curvature, weights)
        value_f = float(cp.asnumpy(value)) if xp is cp else float(value)
        return value_f / total_weight_f

    def shape_ids(self) -> List[str | None]:
        return list(self._samples_by_shape.keys())

    def arrays_for_shape(self, shape_id: Optional[str]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]]:
        return self._arrays_by_shape.get(shape_id)

    # ------------------------------------------------------------------
    # Internal helpers

    def _prepare_shape_array(self, shape: object) -> Optional[tuple[Optional[str], np.ndarray, bool]]:
        points = getattr(shape, "points", None)
        if points is None:
            return None
        array = _as_array(points)
        if array.shape[0] < 3:
            return None
        closed = bool(array.shape[0] >= 4 and np.linalg.norm(array[0] - array[-1]) < 1e-6)
        if closed:
            array = array[:-1]
            if array.shape[0] < 3:
                return None
        shape_id = getattr(shape, "id", None)
        return shape_id, array, closed

    def _process_batch(
        self,
        batch: List[tuple[int, Optional[str], np.ndarray, bool]],
        device_id: int,
    ) -> List[tuple[int, Optional[str], np.ndarray, np.ndarray, np.ndarray, Optional[int]]]:
        results: List[tuple[int, Optional[str], np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = []
        for idx_shape, shape_id, array, closed in batch:
            positions, curvature, weights = self._compute_curvature(array, closed, device_id)
            if positions.size == 0:
                continue
            results.append((idx_shape, shape_id, positions, curvature, weights, device_id))
        return results

    def _store_shape_samples(
        self,
        shape_id: Optional[str],
        positions: np.ndarray,
        curvature: np.ndarray,
        weights: np.ndarray,
        device_id: Optional[int],
    ) -> None:
        if positions.size == 0:
            return
        for idx_pt, pos in enumerate(positions):
            weight_idx = min(idx_pt, len(weights) - 1)
            sample = CurvatureSample(
                position=(float(pos[0]), float(pos[1])),
                curvature=float(curvature[idx_pt]),
                weight=float(weights[weight_idx]),
                source_id=shape_id,
            )
            self._samples.append(sample)
            self._samples_by_shape.setdefault(shape_id, []).append(sample)
        self._arrays_by_shape[shape_id] = (positions, curvature, weights, device_id)

    def _compute_curvature(
        self,
        array: np.ndarray,
        closed: bool,
        device_id: Optional[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if device_id is None or cp is None or not self._use_gpu:
            return self._compute_curvature_cpu(array, closed)
        return self._compute_curvature_gpu(array, closed, device_id)

    @staticmethod
    def _compute_curvature_cpu(array: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        weights = (len_f + len_b) * 0.5
        weights = np.where(np.isfinite(weights), np.maximum(weights, 1e-6), 1.0)
        return array.astype(float, copy=False), curvature.astype(float, copy=False), weights.astype(float, copy=False)

    @staticmethod
    def _compute_curvature_gpu(array: np.ndarray, closed: bool, device_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cp is None:  # pragma: no cover - safety fallback if CuPy missing
            return CurvatureField._compute_curvature_cpu(array, closed)
        with cp.cuda.Device(device_id):
            arr = cp.asarray(array, dtype=cp.float64)
            padded = cp.vstack((arr[0], arr, arr[-1])) if not closed else cp.vstack((arr[-1], arr, arr[0]))
            forward = padded[2:] - padded[1:-1]
            backward = padded[1:-1] - padded[:-2]
            len_f = cp.linalg.norm(forward, axis=1)
            len_b = cp.linalg.norm(backward, axis=1)
            mask = (len_f > 1e-6) & (len_b > 1e-6)
            curvature = cp.zeros(arr.shape[0], dtype=cp.float64)
            if int(cp.count_nonzero(mask)):
                t1 = cp.zeros_like(forward)
                t2 = cp.zeros_like(forward)
                t1[mask] = backward[mask] / len_b[mask][:, None]
                t2[mask] = forward[mask] / len_f[mask][:, None]
                cross = t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0]
                dot = cp.clip((t1 * t2).sum(axis=1), -1.0, 1.0)
                angle = cp.arctan2(cross, dot)
                avg_len = (len_f + len_b) * 0.5
                denom = cp.maximum(avg_len, 1e-6)
                curvature = cp.where(mask, angle / denom, curvature)
            weights = (len_f + len_b) * 0.5
            weights = cp.where(cp.isfinite(weights), cp.maximum(weights, 1e-6), 1.0)
            return (
                cp.asnumpy(arr),
                cp.asnumpy(curvature),
                cp.asnumpy(weights),
            )


class MaterialMemoryField:
    """Diffuse accumulated curvature magnitude for downstream reinforcement."""

    def __init__(self, smoothing_radius: float = 90.0):
        self._sigma = max(1.0, float(smoothing_radius))
        self._samples: List[MemorySample] = []
        self._samples_by_shape: dict[str | None, List[MemorySample]] = {}
        self._arrays_by_shape: dict[str | None, tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]] = {}
        self._use_gpu = False
        self._gpu_devices: List[int] = []

    def configure_gpu(self, devices: Sequence[int], use_gpu: bool) -> None:
        if cp is None or not devices:
            self._use_gpu = False
            self._gpu_devices = []
            return
        self._use_gpu = bool(use_gpu)
        self._gpu_devices = list(devices) if self._use_gpu else []

    def rebuild_from_curvature(self, curvature: CurvatureField) -> None:
        self._samples = []
        self._samples_by_shape.clear()
        self._arrays_by_shape.clear()
        for shape_id in curvature.shape_ids():
            arrays = curvature.arrays_for_shape(shape_id)
            if not arrays:
                continue
            positions, curvature_vals, weights, device_id = arrays
            if positions.size == 0:
                self._arrays_by_shape[shape_id] = (
                    np.zeros((0, 2), dtype=float),
                    np.zeros(0, dtype=float),
                    np.zeros(0, dtype=float),
                    device_id,
                )
                continue
            magnitudes = np.abs(curvature_vals)
            for idx, pos in enumerate(positions):
                weight_idx = min(idx, len(weights) - 1)
                entry = MemorySample(
                    position=(float(pos[0]), float(pos[1])),
                    magnitude=float(magnitudes[idx]),
                    weight=float(weights[weight_idx]),
                    source_id=shape_id,
                )
                self._samples.append(entry)
                self._samples_by_shape.setdefault(shape_id, []).append(entry)
            self._arrays_by_shape[shape_id] = (positions, magnitudes, weights, device_id)

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

    def aggregate_for_shape(self, shape_id: Optional[str]) -> float:
        arrays = self._arrays_by_shape.get(shape_id)
        if not arrays:
            return 0.0
        _, magnitudes, weights, _ = arrays
        xp = cp if (cp is not None and isinstance(magnitudes, cp.ndarray)) else np  # type: ignore[arg-type]
        total_weight = xp.sum(weights)
        total_weight_f = float(cp.asnumpy(total_weight)) if xp is cp else float(total_weight)
        if total_weight_f <= 1e-9:
            return 0.0
        value = xp.dot(magnitudes, weights)
        value_f = float(cp.asnumpy(value)) if xp is cp else float(value)
        return value_f / total_weight_f

    def shape_ids(self) -> List[str | None]:
        return list(self._samples_by_shape.keys())

    def arrays_for_shape(self, shape_id: Optional[str]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, Optional[int]]]:
        return self._arrays_by_shape.get(shape_id)


class AdaptiveMetricManager:
    """Owns curvature/material fields and exposes πₐ sampling helpers."""

    def __init__(
        self,
        curvature_radius: float = 60.0,
        memory_radius: float = 90.0,
        *,
        prefer_gpu: bool = True,
    ) -> None:
        self._curvature = CurvatureField(curvature_radius)
        self._memory = MaterialMemoryField(memory_radius)
        self._field_packs: dict[str | None, FieldPack] = {}
        disable_env = os.environ.get("ADAPTIVECAD_DISABLE_GPU", "").strip().lower()
        env_disables_gpu = disable_env in {"1", "true", "yes", "on"}
        gpu_requested = bool(prefer_gpu) and not env_disables_gpu and cp is not None
        self._gpu_devices: List[int] = []
        if gpu_requested and cp is not None:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
            except Exception:  # pragma: no cover - CUDA runtime missing
                device_count = 0
            if device_count > 0:
                self._gpu_devices = list(range(device_count))
                self._curvature.configure_gpu(self._gpu_devices, True)
                self._memory.configure_gpu(self._gpu_devices, True)
            else:
                self._curvature.configure_gpu([], False)
                self._memory.configure_gpu([], False)
        else:
            self._curvature.configure_gpu([], False)
            self._memory.configure_gpu([], False)

    def gpu_enabled(self) -> bool:
        return bool(self._gpu_devices)

    def gpu_device_hint(self) -> Optional[int]:
        return self._gpu_devices[0] if self._gpu_devices else None

    def rebuild(self, shapes: Iterable[object]) -> None:
        self._curvature.rebuild(shapes)
        self._memory.rebuild_from_curvature(self._curvature)
        self._field_packs.clear()
        shape_ids = set(self._curvature.shape_ids()) | set(self._memory.shape_ids())
        for shape_id in shape_ids:
            cur_arrays = self._curvature.arrays_for_shape(shape_id)
            mem_arrays = self._memory.arrays_for_shape(shape_id)
            device_id: Optional[int] = None
            if cur_arrays:
                positions, curvature, weights, device_id = cur_arrays
            elif mem_arrays:
                positions, _, weights, device_id = mem_arrays
                curvature = np.zeros(mem_arrays[0].shape[0], dtype=float)
            else:
                positions = np.zeros((0, 2), dtype=float)
                curvature = np.zeros(0, dtype=float)
                weights = np.zeros(0, dtype=float)
            if mem_arrays:
                memory = mem_arrays[1]
                if not cur_arrays and weights.size == 0:
                    weights = mem_arrays[2]
            else:
                memory = np.zeros(positions.shape[0], dtype=float)
            if positions.size == 0:
                pack = FieldPack.empty()
            else:
                final_weights = weights if weights.size else np.ones(positions.shape[0], dtype=float)
                pack = FieldPack(positions, curvature, memory, final_weights, device_id)
            self._field_packs[shape_id] = pack

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
        pack = self.field_pack(shape_id)
        return pack.average_curvature()

    def aggregate_memory_for_shape(self, shape_id: Optional[str]) -> float:
        pack = self.field_pack(shape_id)
        return pack.average_memory()

    def field_pack(self, shape_id: Optional[str]) -> FieldPack:
        return self._field_packs.get(shape_id, FieldPack.empty())

    # ------------------------------------------------------------------
    # Adaptive metric helpers

    def segment_length(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        params: Optional[dict] = None,
        *,
        samples: int = 16,
    ) -> float:
        p0_arr = np.asarray(p0, dtype=float)
        p1_arr = np.asarray(p1, dtype=float)
        delta = p1_arr - p0_arr
        length = float(np.hypot(delta[0], delta[1]))
        if length <= 1e-9:
            return 0.0
        span = max(length * 0.5, 1e-6)
        steps = max(1, int(samples))
        ts = np.linspace(0.0, 1.0, steps + 1)
        mids = p0_arr + (ts[:-1] + ts[1:])[:, None] * 0.5 * delta
        ratios: List[float] = []
        for mid in mids:
            ratios.append(self.pi_a((float(mid[0]), float(mid[1])), span, params) / math.pi)
        avg_ratio = float(np.mean(ratios)) if ratios else 1.0
        return length * avg_ratio

    def polyline_length(
        self,
        points: Sequence[Sequence[float]],
        params: Optional[dict] = None,
        *,
        samples_per_segment: int = 16,
    ) -> float:
        array = np.asarray(points, dtype=float)
        if array.shape[0] < 2:
            return 0.0
        total = 0.0
        for idx in range(array.shape[0] - 1):
            total += self.segment_length(array[idx], array[idx + 1], params, samples=samples_per_segment)
        return total

    def curve_length(
        self,
        control_points: Sequence[Sequence[float]],
        params: Optional[dict] = None,
        *,
        samples: int = 256,
        samples_per_segment: int = 16,
    ) -> float:
        pts = _curve_points(control_points, params, samples=samples)
        return self.polyline_length(pts, params, samples_per_segment=samples_per_segment)

    def circle_circumference(
        self,
        center: Sequence[float],
        radius: float,
        params: Optional[dict] = None,
        *,
        samples: int = 512,
        samples_per_segment: int = 8,
    ) -> float:
        pts = _circle_points(center, radius, params, samples=samples)
        return self.polyline_length(pts, params, samples_per_segment=samples_per_segment)
