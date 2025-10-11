"""Export utilities bridging to AdaptiveCAD core curvature fields."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

try:
    from adaptivecad_core.fields import FieldPack as _FieldPack
except ModuleNotFoundError:  # pragma: no cover
    _FieldPack = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from adaptivecad_core.fields import FieldPack
else:  # pragma: no cover
    FieldPack = Any  # type: ignore[assignment]


def _build_field(mesh: Dict[str, Any]) -> Any:
    if _FieldPack is None or not mesh:
        return None
    vertices = mesh.get("vertices")
    faces = mesh.get("faces", [])
    if vertices is None:
        return None
    beta = mesh.get("beta", 0.1)
    gamma = mesh.get("gamma", 0.05)
    return _FieldPack(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int32),
        beta=beta,
        gamma=gamma,
    )

async def export_scene(payload: Dict[str, Any]) -> Dict[str, str]:
    """Export mesh data to a JSON artifact enriched with curvature metrics."""

    mesh = payload.get("mesh")
    if mesh is None:
        raise ValueError("Export payload requires 'mesh'")

    field = _build_field(mesh)
    if field is None:
        raise RuntimeError("adaptivecad_core.fields.FieldPack unavailable or mesh incomplete")

    artifact = {
        "vertices": field.vertices.tolist(),
        "faces": field.faces.tolist(),
        "kappa": field.kappa.tolist(),
        "memory": field.memory.tolist(),
        "beta": field.beta,
        "gamma": field.gamma,
    }

    output_path = payload.get("path")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, temp_path = tempfile.mkstemp(prefix="adaptivecad_export_", suffix=".json")
        os.close(fd)
        path = Path(temp_path)
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    return {"status": "ok", "artifact": str(path)}
