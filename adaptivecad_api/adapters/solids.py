"""Solid modeling adapters backed by AdaptiveCAD core utilities."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

import numpy as np

try:
    from adaptivecad_core.revolve import revolve_adaptive
    from adaptivecad_core.prototype_cut_join import adaptive_add, adaptive_join, adaptive_cut
    from adaptivecad_core.fields import FieldPack as _FieldPack
except ModuleNotFoundError:  # pragma: no cover
    revolve_adaptive = None  # type: ignore[assignment]
    adaptive_add = None  # type: ignore[assignment]
    adaptive_join = None  # type: ignore[assignment]
    adaptive_cut = None  # type: ignore[assignment]
    _FieldPack = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from adaptivecad_core.fields import FieldPack
else:  # pragma: no cover
    FieldPack = Any  # type: ignore[assignment]


def _ensure_points(points: Iterable[Iterable[float]], *, dim: int = 3) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.ndim != 2:
        raise ValueError("Profile data must be a 2D array")
    if arr.shape[1] == dim:
        return arr
    if arr.shape[1] > dim:
        return arr[:, :dim]
    # Pad with zeros for missing trailing coordinates
    pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=float)
    return np.hstack([arr, pad])


def _build_field(field_spec: Optional[Dict[str, Any]]):
    if _FieldPack is None or not field_spec:
        return None
    vertices = field_spec.get("vertices")
    faces = field_spec.get("faces", [])
    if vertices is None:
        return None
    beta = field_spec.get("beta", 0.1)
    gamma = field_spec.get("gamma", 0.05)
    return _FieldPack(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int32),
        beta=beta,
        gamma=gamma,
    )


def _default_axis(payload: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis_point = np.asarray(payload.get("axis_point", [0.0, 0.0, 0.0]), dtype=float)
    axis_dir = np.asarray(payload.get("axis_dir", [0.0, 0.0, 1.0]), dtype=float)
    if np.linalg.norm(axis_dir) == 0.0:
        axis_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    return axis_point, axis_dir


async def revolve(payload: Dict[str, Any], emit) -> Dict[str, Any]:
    """Revolve a 3D profile using the adaptive π-aware kernel."""

    if revolve_adaptive is None:
        raise RuntimeError("adaptivecad_core.revolve is unavailable")

    profile = payload.get("profile")
    if not profile:
        raise ValueError("Payload missing 'profile' points")

    profile_pts = _ensure_points(profile)
    axis_point, axis_dir = _default_axis(payload)
    field = _build_field(payload.get("field"))

    settings = payload.get("settings", {})
    angle = settings.get("angle")
    theta0 = settings.get("theta0", payload.get("theta0", 0.0))
    tol = settings.get("tolerance", payload.get("tolerance", 0.001))
    eta_k = settings.get("eta_k", payload.get("eta_k", 0.4))
    eta_m = settings.get("eta_m", payload.get("eta_m", 0.3))
    smart_seam = bool(settings.get("smart_seam", payload.get("smart_seam", False)))
    cap_start = bool(settings.get("cap_start", payload.get("cap_start", True)))
    cap_end = bool(settings.get("cap_end", payload.get("cap_end", True)))
    min_segments = int(settings.get("min_segments", payload.get("min_segments", 16)))
    max_segments = int(settings.get("max_segments", payload.get("max_segments", 720)))

    await emit("progress", {"message": "Running revolve"})
    await asyncio.sleep(0)
    result = revolve_adaptive(
        profile_pts,
        axis_point,
        axis_dir,
        angle=angle,
        theta0=theta0,
        tol=tol,
        eta_k=eta_k,
        eta_m=eta_m,
        fields=field,
        smart_seam=smart_seam,
        cap_start=cap_start,
        cap_end=cap_end,
        min_segments=min_segments,
        max_segments=max_segments,
    )
    artifact = {
        "vertices": result.vertices.tolist(),
        "faces": result.faces.tolist(),
        "theta0": float(result.theta0),
        "segments": int(result.segments),
        "seam_ring_indices": result.seam_ring_indices.tolist(),
    }
    await emit("progress", {"message": "Revolve completed"})
    return {"status": "ok", "artifact": artifact}


async def extrude(payload: Dict[str, Any], emit) -> Dict[str, Any]:
    await emit("progress", {"message": "Preparing extrude"})
    await asyncio.sleep(0)

    vertices_data = payload.get("vertices")
    if vertices_data is None:
        raise ValueError("Extrude payload requires 'vertices'")

    base = _ensure_points(vertices_data)
    direction = np.asarray(payload.get("direction", [0.0, 0.0, 1.0]), dtype=float)
    distance = float(payload.get("distance", 1.0))
    direction = direction * distance

    faces = payload.get("faces") or []
    field = _build_field(payload.get("field") or {"vertices": base, "faces": faces})

    if adaptive_add is not None and field is not None:
        rho = float(payload.get("rho", 0.5))
        t0 = float(payload.get("t0", distance))
        top = adaptive_add(base, direction, field, rho=rho, t0=t0)
    else:
        top = base + direction

    vertices = np.vstack([base, top])
    face_list = []
    base_offset = 0
    top_offset = base.shape[0]

    def _triangulate_loop(loop: np.ndarray, offset: int, reverse: bool = False) -> list[list[int]]:
        tris: list[list[int]] = []
        if loop.shape[0] < 3:
            return tris
        order = range(1, loop.shape[0] - 1)
        for idx in order:
            tri = [offset, offset + idx, offset + idx + 1]
            if reverse:
                tri[1], tri[2] = tri[2], tri[1]
            tris.append(tri)
        return tris

    if not faces and base.shape[0] >= 3:
        base_faces = _triangulate_loop(base, base_offset, reverse=True)
        top_faces = _triangulate_loop(top, top_offset, reverse=False)
        face_list.extend(base_faces)
        face_list.extend(top_faces)
    else:
        for face in faces:
            face_list.append([base_offset + int(i) for i in face])
            face_list.append([top_offset + int(i) for i in face])

    for idx in range(base.shape[0]):
        nxt = (idx + 1) % base.shape[0]
        face_list.append(
            [
                base_offset + idx,
                base_offset + nxt,
                top_offset + nxt,
            ]
        )
        face_list.append(
            [
                base_offset + idx,
                top_offset + nxt,
                top_offset + idx,
            ]
        )

    await emit("progress", {"message": "Extrude completed"})
    return {
        "status": "ok",
        "artifact": {
            "vertices": vertices.tolist(),
            "faces": face_list,
        },
    }


async def boolean(payload: Dict[str, Any], emit) -> Dict[str, Any]:
    await emit("progress", {"message": "Preparing boolean"})
    await asyncio.sleep(0)

    operation = payload.get("operation", "join")

    if operation == "cut":
        if adaptive_cut is None:
            raise RuntimeError("adaptivecad_core.prototype_cut_join.adaptive_cut unavailable")
        surface = payload.get("surface") or {}
        vertices = surface.get("vertices")
        faces = surface.get("faces", [])
        if vertices is None:
            raise ValueError("Cut operation requires 'surface.vertices'")
        field = _build_field(payload.get("field") or surface)
        seed_a = payload.get("seed_a")
        seed_b = payload.get("seed_b")
        if seed_a is None or seed_b is None:
            raise ValueError("Cut operation requires 'seed_a' and 'seed_b'")
        lam_k = float(payload.get("lam_k", 0.4))
        lam_m = float(payload.get("lam_m", 0.2))
        steps = int(payload.get("steps", 64))
        seam = adaptive_cut(vertices, faces, seed_a, seed_b, field, lam_k=lam_k, lam_m=lam_m, steps=steps)
        result = {"seam": seam.tolist()}
    elif operation == "add":
        if adaptive_add is None:
            raise RuntimeError("adaptivecad_core.prototype_cut_join.adaptive_add unavailable")
        vertices = payload.get("vertices")
        direction = payload.get("direction")
        if vertices is None or direction is None:
            raise ValueError("Add operation requires 'vertices' and 'direction'")
        field = _build_field(payload.get("field"))
        rho = float(payload.get("rho", 0.5))
        t0 = float(payload.get("t0", 1.0))
        displaced = adaptive_add(vertices, direction, field, rho=rho, t0=t0) if field is not None else np.asarray(vertices, dtype=float) + np.asarray(direction, dtype=float)
        result = {"vertices": displaced.tolist()}
    else:  # join/union-style seam
        if adaptive_join is None:
            raise RuntimeError("adaptivecad_core.prototype_cut_join.adaptive_join unavailable")
        a = payload.get("vertices_a")
        b = payload.get("vertices_b")
        if a is None or b is None:
            raise ValueError("Join operation requires 'vertices_a' and 'vertices_b'")
        field = _build_field(payload.get("field"))
        lam = float(payload.get("lam", 0.5))
        seam = adaptive_join(a, b, field, lam=lam) if field is not None else np.asarray(a, dtype=float)
        result = {"seam": seam.tolist()}

    await emit("progress", {"message": "Boolean completed"})
    return {"status": "ok", "artifact": result}
