"""Generate a Möbius strip mesh."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Tuple
import struct

import numpy as np

OUTPUT_OBJ = Path("mobius_strip.obj")
OUTPUT_STL = Path("mobius_strip.stl")
RADIUS_MM = 30.0         # Radius of the strip centreline
WIDTH_MM = 10.0          # Half-width of the strip
TWIST_TURNS = 0.5        # Number of twists (0.5 = single Möbius twist)
MAJOR_SAMPLES = 512      # Samples along the centreline
WIDTH_SAMPLES = 32       # Samples across the strip width
THICKNESS_MM = 2.0       # Total wall thickness of the Möbius band


def parametric_mobius(
    radius: float,
    half_width: float,
    twist_turns: float,
    major_samples: int,
    width_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangular faces for a Möbius strip parameterization."""

    u = np.linspace(0.0, 2.0 * math.pi, major_samples, endpoint=False)
    v = np.linspace(-half_width, half_width, width_samples, endpoint=True)
    U, V = np.meshgrid(u, v, indexing="ij")

    twist = twist_turns * 2.0 * math.pi * (U / (2.0 * math.pi))
    cos_u = np.cos(U)
    sin_u = np.sin(U)
    cos_twist = np.cos(twist)
    sin_twist = np.sin(twist)

    x = (radius + V * cos_twist) * cos_u
    y = (radius + V * cos_twist) * sin_u
    z = V * sin_twist

    vertices = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))

    faces: list[tuple[int, int, int]] = []
    for i in range(major_samples):
        inext = (i + 1) % major_samples
        for j in range(width_samples - 1):
            a = i * width_samples + j
            b = inext * width_samples + j
            c = inext * width_samples + (j + 1)
            d = i * width_samples + (j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    return vertices, np.asarray(faces, dtype=np.int32)


def mesh_to_obj(vertices: Iterable[tuple[float, float, float]], faces: Iterable[tuple[int, int, int]]) -> str:
    """Convert a triangle mesh into ASCII OBJ format."""

    lines = ["o MobiusStrip"]
    for vx, vy, vz in vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for a, b, c in faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    return "\n".join(lines) + "\n"


def write_binary_stl(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    """Write the mesh to a binary STL file."""

    with path.open("wb") as fh:
        fh.write(b"AdaptiveCAD Mobius".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            ax, ay, az = vertices[a]
            bx, by, bz = vertices[b]
            cx, cy, cz = vertices[c]
            # Compute facet normal
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm != 0.0:
                nx /= norm
                ny /= norm
                nz /= norm
            fh.write(struct.pack("<3f", nx, ny, nz))
            # Write vertex coordinates for triangle
            fh.write(struct.pack("<3f", ax, ay, az))
            fh.write(struct.pack("<3f", bx, by, bz))
            fh.write(struct.pack("<3f", cx, cy, cz))
            fh.write(struct.pack("<H", 0))


def build_mobius_strip() -> None:
    """Generate the Möbius strip mesh and export it as OBJ."""

    # Base Möbius surface
    vertices_arr, faces_arr = parametric_mobius(
        radius=RADIUS_MM,
        half_width=WIDTH_MM / 2.0,
        twist_turns=TWIST_TURNS,
        major_samples=MAJOR_SAMPLES,
        width_samples=WIDTH_SAMPLES,
    )

    vertices = [(float(row[0]), float(row[1]), float(row[2])) for row in vertices_arr]
    faces = [(int(tri[0]), int(tri[1]), int(tri[2])) for tri in faces_arr]

    # Extrude into a thick band by offsetting along vertex normals and stitching sides
    import collections
    # Compute per-vertex normals
    normals = [np.zeros(3) for _ in vertices]
    for a, b, c in faces:
        v0 = np.array(vertices[a])
        v1 = np.array(vertices[b])
        v2 = np.array(vertices[c])
        fn = np.cross(v1 - v0, v2 - v0)
        norm_val = np.linalg.norm(fn)
        if norm_val != 0.0:
            fn = fn / norm_val
        normals[a] += fn
        normals[b] += fn
        normals[c] += fn
    # Normalize accumulated normals
    normals = [n / np.linalg.norm(n) if np.linalg.norm(n) != 0.0 else n for n in normals]
    half_t = THICKNESS_MM / 2.0
    # Create inner and outer vertex sets
    inner = [(x - normals[i][0] * half_t,
              y - normals[i][1] * half_t,
              z - normals[i][2] * half_t)
             for i, (x, y, z) in enumerate(vertices)]
    outer = [(x + normals[i][0] * half_t,
              y + normals[i][1] * half_t,
              z + normals[i][2] * half_t)
             for i, (x, y, z) in enumerate(vertices)]
    all_vertices = inner + outer
    base_n = len(inner)
    # Surface faces: outer preserved, inner reversed
    outer_faces = [(a + base_n, b + base_n, c + base_n) for a, b, c in faces]
    inner_faces = [(c, b, a) for a, b, c in faces]
    # Build side walls along boundary edges
    edge_count = collections.Counter()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[tuple(sorted((u, v)))] += 1
    wall_faces = []
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            if edge_count[tuple(sorted((u, v)))] == 1:
                # flipped ordering for outward normals
                wall_faces.append((v + base_n, v, u))
                wall_faces.append((u + base_n, v + base_n, u))
    faces = inner_faces + outer_faces + wall_faces
    # Export thick band
    OUTPUT_OBJ.write_text(mesh_to_obj(all_vertices, faces), encoding="utf-8")
    write_binary_stl(OUTPUT_STL, all_vertices, faces)
    print(f"Wrote Möbius strip OBJ to {OUTPUT_OBJ.resolve()}")
    print(f"Wrote Möbius strip STL to {OUTPUT_STL.resolve()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Möbius strip with custom parameters.")
    parser.add_argument("--twists", type=float, default=TWIST_TURNS,
                        help="Number of half-twists (0.5=single Möbius, 1.5=1½ twists, 2=orientable double loop)")
    parser.add_argument("--metric", choices=["euclidean","lorentz"], default="euclidean",
                        help="Metric signature for embedding (euclidean or lorentz)")
    parser.add_argument("--gauge-field", choices=["none","u1","su2"], default="none",
                        help="Gauge field type to annotate mesh (none, u1, su2)")
    parser.add_argument("--dim", type=int, default=3,
                        help="Embedded dimension (3 or 4 for time-like coordinate)")
    args = parser.parse_args()
    # Override global twist
    TWIST = args.twists
    print(f"Generating Möbius strip: twists={args.twists}, metric={args.metric}, gauge={args.gauge_field}, dim={args.dim}")
    # For now, only map twists into geometry; metric and gauge are conceptual
    # Update global twist parameter
    from __main__ import TWIST_TURNS as _old
    TWIST_TURNS = args.twists
    build_mobius_strip()
