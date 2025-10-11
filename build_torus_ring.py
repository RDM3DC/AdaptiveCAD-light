"""Generate a torus-like ring with optional spiral twist."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

OUTPUT_PATH = Path("torus_ring.obj")
MAJOR_RADIUS_MM = 20.0   # Distance from origin to the centre of the tube (outer radius)
MINOR_RADIUS_MM = 5.0    # Radius of the tube cross-section
TWIST_REVOLUTIONS = 1.5  # Number of additional rotations applied along the major loop
MAJOR_SAMPLES = 256
MINOR_SAMPLES = 96


def parametric_torus(
    major_radius: float,
    minor_radius: float,
    twist_turns: float,
    major_samples: int,
    minor_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangular faces for a twisted torus."""

    u = np.linspace(0.0, 2.0 * math.pi, major_samples, endpoint=False)
    v = np.linspace(0.0, 2.0 * math.pi, minor_samples, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")

    twist = twist_turns * 2.0 * math.pi * (U / (2.0 * math.pi))
    # Equivalent to twist_turns * U but keeps intent explicit
    angle = V + twist

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cos_u = np.cos(U)
    sin_u = np.sin(U)

    x = (major_radius + minor_radius * cos_angle) * cos_u
    y = (major_radius + minor_radius * cos_angle) * sin_u
    z = minor_radius * sin_angle

    vertices = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))

    faces: List[Tuple[int, int, int]] = []
    for i in range(major_samples):
        inext = (i + 1) % major_samples
        for j in range(minor_samples):
            jnext = (j + 1) % minor_samples
            a = i * minor_samples + j
            b = inext * minor_samples + j
            c = inext * minor_samples + jnext
            d = i * minor_samples + jnext
            faces.append((a, b, c))
            faces.append((a, c, d))

    return vertices, np.asarray(faces, dtype=np.int32)


def mesh_to_obj(vertices: Iterable[tuple[float, float, float]], faces: Iterable[tuple[int, int, int]]) -> str:
    """Convert a triangle mesh into ASCII OBJ format."""

    lines = ["o TorusRing"]
    for vx, vy, vz in vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    return "\n".join(lines) + "\n"


def build_torus_ring() -> None:
    """Generate the twisted torus mesh and export it as OBJ."""

    vertices_arr, faces_arr = parametric_torus(
        MAJOR_RADIUS_MM,
        MINOR_RADIUS_MM,
        TWIST_REVOLUTIONS,
        MAJOR_SAMPLES,
        MINOR_SAMPLES,
    )

    vertices = [(float(row[0]), float(row[1]), float(row[2])) for row in vertices_arr]
    faces = [(int(tri[0]), int(tri[1]), int(tri[2])) for tri in faces_arr]

    OUTPUT_PATH.write_text(mesh_to_obj(vertices, faces), encoding="utf-8")
    print(f"Wrote torus OBJ to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    build_torus_ring()
