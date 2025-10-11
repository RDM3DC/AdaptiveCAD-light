"""Adaptive revolve/lathe helper with πₐ-aware sampling.

This module exposes a single entry point, :func:`revolve_adaptive`, which
revolves a polyline profile around an axis while adapting the θ segmentation
based on curvature and (optionally) external memory fields. The implementation
is derived from the design brief provided for AdaptiveCAD-light v0.2 but is
packaged as a pure-python utility so both the playground and the Blender core
can reuse it.

The function accepts either pure numpy inputs or lightweight field samplers. If
no field feedback is provided the algorithm falls back to a deterministic chord
error rule so it can be used in environments that do not yet expose πₐ fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

EPS = 1e-9


class FieldSampler(Protocol):
    """Protocol for optional curvature/memory sampling hooks."""

    def sample(self, point: Sequence[float]) -> tuple[float, float]:
        ...

    def accumulate(self, point: Sequence[float], amount: float) -> None:
        ...


@dataclass
class RevolveResult:
    vertices: np.ndarray
    faces: np.ndarray
    seam_ring_indices: np.ndarray
    theta0: float
    segments: int


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + EPS)


def _project_point_to_axis(p: np.ndarray, a0: np.ndarray, a_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point on the axis and the radial vector."""

    ap = p - a0
    t = float(np.dot(ap, a_dir))
    foot = a0 + t * a_dir
    radial = p - foot
    return foot, radial


def _rotate_about_axis(v: np.ndarray, a_dir: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula."""

    k = _unit(a_dir)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    cross = np.cross(k, v)
    return v * c + cross * s + k * np.dot(k, v) * (1.0 - c)


def _discrete_curve_curvature(polyline: np.ndarray) -> np.ndarray:
    """Estimate curvature for each vertex of a 3D polyline."""

    n = polyline.shape[0]
    kappa = np.zeros(n, dtype=float)
    if n < 3:
        return kappa
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    seg = np.concatenate([[seg[0]], seg, [seg[-1]]])
    for i in range(1, n - 1):
        v1 = polyline[i] - polyline[i - 1]
        v2 = polyline[i + 1] - polyline[i]
        l1 = float(np.linalg.norm(v1) + EPS)
        l2 = float(np.linalg.norm(v2) + EPS)
        cosang = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        ang = float(np.arccos(cosang))
        ds = 0.5 * (l1 + l2)
        kappa[i] = ang / (ds + EPS)
    if n >= 3:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
    return kappa


def _resolve_sampler(fields: Optional[object | Callable[[Sequence[float]], tuple[float, float]]]) -> tuple[Optional[Callable[[Sequence[float]], tuple[float, float]]], Optional[Callable[[Sequence[float], float], None]]]:
    """Return (sample_fn, accumulate_fn) based on provided fields object."""

    if fields is None:
        return None, None
    if callable(fields):
        return fields, None
    sample_fn = getattr(fields, "sample", None)
    accumulate_fn = getattr(fields, "accumulate", None)
    if sample_fn is None:
        return None, None
    return sample_fn, accumulate_fn


def _theta_segments(
    profile: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    angle: float,
    tol: float,
    eta_k: float,
    eta_m: float,
    sampler: Optional[Callable[[Sequence[float]], tuple[float, float]]],
    profile_curv: np.ndarray,
    min_seg: int = 16,
    max_seg: int = 720,
) -> int:
    a0, ad = axis_point, _unit(axis_dir)
    thetas: list[float] = []
    Mbar = 0.0
    if sampler is not None:
        samples = []
        for p in profile:
            curv, mem = sampler(p)
            samples.append(mem)
        if samples:
            Mbar = float(np.mean(samples))
    for idx, point in enumerate(profile):
        _, radial = _project_point_to_axis(point, a0, ad)
        r = float(np.linalg.norm(radial))
        if r < 1e-6:
            continue
        val = 1.0 - (tol / max(r, tol))
        val = float(np.clip(val, -1.0, 1.0))
        dtheta = 2.0 * float(np.arccos(val))
        tighten = 1.0 + eta_k * profile_curv[idx] + eta_m * Mbar
        dtheta /= max(tighten, 1e-3)
        if dtheta <= 0.0 or not np.isfinite(dtheta):
            dtheta = np.deg2rad(2.0)
        thetas.append(dtheta)
    if not thetas:
        S = max(min_seg, int(np.ceil(angle / np.deg2rad(10.0))))
        return int(np.clip(S, min_seg, max_seg))
    dtheta_min = max(min(thetas), np.deg2rad(0.5))
    segments = int(np.ceil(angle / dtheta_min))
    return int(np.clip(segments, min_seg, max_seg))


def _select_seam_theta(
    profile: np.ndarray,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    eta_m: float,
    sampler: Optional[Callable[[Sequence[float]], tuple[float, float]]],
    candidate_count: int = 12,
) -> float:
    if sampler is None:
        return 0.0
    a0, ad = axis_point, _unit(axis_dir)
    candidates = np.linspace(0.0, 2.0 * np.pi, candidate_count, endpoint=False)
    best_theta = 0.0
    best_energy = float("inf")
    for theta0 in candidates:
        energy = 0.0
        for p in profile:
            _, radial = _project_point_to_axis(p, a0, ad)
            seam_point = _rotate_about_axis(radial, ad, theta0) + (a0 + np.dot(p - a0, ad) * ad)
            curv, mem = sampler(seam_point)
            energy += abs(curv) + eta_m * mem
        if energy < best_energy:
            best_energy = energy
            best_theta = theta0
    return float(best_theta)


def revolve_adaptive(
    profile_pts: Iterable[Sequence[float]],
    axis_point: Sequence[float],
    axis_dir: Sequence[float],
    *,
    angle: float | None = None,
    theta0: float = 0.0,
    tol: float = 0.001,
    eta_k: float = 0.4,
    eta_m: float = 0.3,
    fields: Optional[object | Callable[[Sequence[float]], tuple[float, float]]] = None,
    smart_seam: bool = False,
    cap_start: bool = True,
    cap_end: bool = True,
    min_segments: int = 16,
    max_segments: int = 720,
) -> RevolveResult:
    """Revolve a polyline profile around an axis with adaptive sampling."""

    profile = np.asarray(list(profile_pts), dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 3:
        raise ValueError("profile_pts must be an iterable of 3D points")
    if profile.shape[0] < 2:
        raise ValueError("profile must contain at least two points")

    a0 = np.asarray(axis_point, dtype=float)
    ad = _unit(np.asarray(axis_dir, dtype=float))
    sweep_angle = float(angle if angle is not None else 2.0 * np.pi)

    sampler, accumulator = _resolve_sampler(fields)

    kappa = _discrete_curve_curvature(profile)

    if smart_seam:
        theta0 = _select_seam_theta(profile, a0, ad, eta_m, sampler)

    segments = _theta_segments(
        profile,
        a0,
        ad,
        sweep_angle,
        tol,
        eta_k,
        eta_m,
        sampler,
        kappa,
        min_seg=min_segments,
        max_seg=max_segments,
    )

    verts: list[np.ndarray] = []
    rings: list[list[int]] = []
    for point in profile:
        foot, radial = _project_point_to_axis(point, a0, ad)
        height = np.dot(point - a0, ad)
        ring: list[int] = []
        for j in range(segments + 1):
            theta = theta0 + sweep_angle * (j / segments)
            pos = _rotate_about_axis(radial, ad, theta) + (a0 + height * ad)
            ring.append(len(verts))
            verts.append(pos)
        rings.append(ring)

    vertices = np.vstack(verts)
    faces: list[list[int]] = []
    for i in range(profile.shape[0] - 1):
        row0, row1 = rings[i], rings[i + 1]
        for j in range(segments):
            a = row0[j]
            b = row0[j + 1]
            c = row1[j + 1]
            d = row1[j]
            faces.append([a, b, c])
            faces.append([a, c, d])

    seam_ring_indices = np.array([ring[0] for ring in rings], dtype=np.int32)

    def _touches_axis(pt: np.ndarray) -> bool:
        _, radial = _project_point_to_axis(pt, a0, ad)
        return float(np.linalg.norm(radial)) < 1e-6

    if cap_start and _touches_axis(profile[0]):
        center = a0 + np.dot(profile[0] - a0, ad) * ad
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center])
        row = rings[0]
        for j in range(segments):
            faces.append([center_idx, row[j + 1], row[j]])

    if cap_end and _touches_axis(profile[-1]):
        center = a0 + np.dot(profile[-1] - a0, ad) * ad
        center_idx = len(vertices)
        vertices = np.vstack([vertices, center])
        row = rings[-1]
        for j in range(segments):
            faces.append([center_idx, row[j], row[j + 1]])

    faces_arr = np.asarray(faces, dtype=np.int32)

    if accumulator is not None:
        for idx in seam_ring_indices:
            pt = vertices[idx]
            accumulator(pt, 0.05)

    return RevolveResult(
        vertices=vertices,
        faces=faces_arr,
        seam_ring_indices=seam_ring_indices,
        theta0=float(theta0),
        segments=int(segments),
    )


__all__ = ["revolve_adaptive", "RevolveResult"]
