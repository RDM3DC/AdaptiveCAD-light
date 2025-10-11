"""Generate a simplified Benchy-style hull using AdaptiveCAD revolve utilities."""
from __future__ import annotations

import math
from pathlib import Path

from adaptivecad_core.revolve import revolve_adaptive
from adaptivecad_playground.sketch_mode import SketchPlane, lift_polyline

OUTPUT_PATH = Path("benchy_hull.obj")
SCALE_TO_MM = 100.0  # scale factor applied to bring unit model into millimetre space

# Simplified side profile inspired by the classic Benchy hull silhouette.
# Points are defined in the ZX sketch plane with Z as "up" and X pointing outward.
HULL_PROFILE: list[tuple[float, float]] = [
    (0.0, -0.05),   # keel tip
    (0.02, -0.045),
    (0.045, -0.035),
    (0.07, -0.02),
    (0.09, 0.0),
    (0.10, 0.03),
    (0.09, 0.06),
    (0.07, 0.09),
    (0.05, 0.12),
    (0.03, 0.16),
    (0.02, 0.20),
    (0.015, 0.24),
    (0.012, 0.27),
    (0.010, 0.30),  # deck taper
    (0.008, 0.33),
    (0.006, 0.35),  # bow cap
]


def mesh_to_obj(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> str:
    """Convert the revolve output mesh into ASCII OBJ text."""

    lines = ["o BenchyHull"]
    for vx, vy, vz in vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    return "\n".join(lines) + "\n"


def build_benchy_hull() -> None:
    """Revolve the 2D profile and write an OBJ mesh."""

    profile_pts = lift_polyline(HULL_PROFILE, SketchPlane.ZX)
    # Shift the profile so it revolves around the keel line when sweeping 180 degrees.
    profile_pts[:, 1] += 0.08
    result = revolve_adaptive(
        profile_pts,
        axis_point=(0.0, 0.0, 0.0),
        axis_dir=(0.0, 1.0, 0.0),
        angle=math.radians(180.0),
        tol=0.0005,
        eta_k=0.4,
        eta_m=0.2,
        smart_seam=True,
        cap_start=True,
        cap_end=True,
    )

    vertices: list[tuple[float, float, float]] = []
    for row in result.vertices.tolist():
        if len(row) != 3:
            raise ValueError("Unexpected vertex dimensionality; expected 3 components")
        x, y, z = (float(coord) * SCALE_TO_MM for coord in row)
        vertices.append((x, y, z))

    faces: list[tuple[int, int, int]] = []
    for tri in result.faces.tolist():
        if len(tri) != 3:
            raise ValueError("Revolve output should be triangles only")
        i0, i1, i2 = (int(idx) for idx in tri)
        faces.append((i0, i1, i2))

    OUTPUT_PATH.write_text(mesh_to_obj(vertices, faces), encoding="utf-8")
    print(f"Wrote Benchy hull OBJ to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    build_benchy_hull()
