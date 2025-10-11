import argparse
from pathlib import Path
import struct
import numpy as np

DEFAULT_OBJ = Path("adaptive_mobius_unity.obj")
DEFAULT_STL = Path("adaptive_mobius_unity.stl")


def adaptive_mobius_unity(
    radius_mm: float = 40.0,
    half_width_mm: float = 8.0,
    twists: float = 1.5,
    gamma: float = 0.25,
    samples_major: int = 480,
    samples_width: int = 48,
    tau: float = 0.5,
    proj_mode: str = "hybrid",
    thickness_mm: float = 2.0,
    kappa: float = 0.0,
):
    """
    Generate a (optionally thickened) adaptive Möbius band with π-adaptive curvature.

    Returns:
      vertices: (N,3) float32 array in mm
      faces: list[tuple[int,int,int]] wound CCW for outward normals
    Notes:
      - u is periodic; we wrap across the seam to avoid gaps.
      - thickness_mm=0.0 produces a single-sided sheet (non-manifold).
      - kappa=0.0 produces standard geometry; kappa>0 adds π-adaptive warping.
    """
    # Periodic along u; endpoint=False avoids duplicate seam column
    u = np.linspace(0.0, 2.0 * np.pi, samples_major, endpoint=False, dtype=np.float64)
    v = np.linspace(-half_width_mm, half_width_mm, samples_width, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # (M,N)

    # 4D time-like coordinate for projections
    t4 = gamma * np.sin(uu * twists / 2.0)

    # π-adaptive radius modulation
    Rk = radius_mm * (1.0 + kappa * np.sin(uu))

    # Euclidean base
    cos_tw = np.cos(twists * uu / 2.0)
    sin_tw = np.sin(twists * uu / 2.0)
    r_eff = Rk + vv * cos_tw
    x = r_eff * np.cos(uu)
    y = r_eff * np.sin(uu)
    z = vv * sin_tw

    # Projection blend
    if proj_mode == "euclidean":
        x3, y3, z3 = x, y, z
    elif proj_mode == "lorentz":
        # Simple Lorentz-like mixing between x and z (toy model)
        ch, sh = np.cosh(t4), np.sinh(t4)
        x3 = x * ch - z * sh
        y3 = y
        z3 = z * ch - x * sh
    elif proj_mode == "complex":
        phase = np.exp(1j * uu * twists)
        x3 = np.real(x + phase * 0.4)
        y3 = np.imag(y + phase * 0.4)
        z3 = z + np.real(phase) * 0.2
    else:
        # hybrid: blend Euclidean, Lorentz, Complex using tau in [0,1]
        phase = np.exp(1j * uu * twists)
        ch, sh = np.cosh(t4), np.sinh(t4)
        xE, yE, zE = x, y, z
        xL, yL, zL = x * ch - z * sh, y, z * ch - x * sh
        xC, yC, zC = np.real(x + phase * 0.4), np.imag(y + phase * 0.4), z + np.real(phase) * 0.2
        x3 = (1.0 - tau) * xE + tau * 0.5 * (xL + xC)
        y3 = (1.0 - tau) * yE + tau * 0.5 * (yL + yC)
        z3 = (1.0 - tau) * zE + tau * 0.5 * (zL + zC)

    M, N = samples_major, samples_width
    P = np.stack([x3, y3, z3], axis=-1)  # (M,N,3)

    # Compute per-vertex normals from param derivatives (periodic in u)
    def finite_diff_periodic(arr, axis):
        if axis == 0:  # along u, wrap
            return (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) * 0.5
        if axis == 1:  # along v, clamp ends (one-sided)
            result = np.zeros_like(arr)
            result[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) * 0.5
            result[:, 0, :] = arr[:, 1, :] - arr[:, 0, :]
            result[:, -1, :] = arr[:, -1, :] - arr[:, -2, :]
            return result
        raise ValueError

    Pu = finite_diff_periodic(P, axis=0)
    Pv = finite_diff_periodic(P, axis=1)
    Nrm = np.cross(Pu, Pv)  # (M,N,3)
    nlen = np.linalg.norm(Nrm, axis=-1, keepdims=True) + 1e-12
    Nn = Nrm / nlen  # unit normals

    verts = []
    faces = []

    def idx_outer(i, j):
        return i * N + j

    def idx_inner(i, j):
        return M * N + i * N + j

    if thickness_mm > 0.0:
        # Outer and inner shells
        outer = P + (0.5 * thickness_mm) * Nn
        inner = P - (0.5 * thickness_mm) * Nn
        verts = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)], axis=0)

        # Surface quads -> triangles (wrap i+1 with modulo)
        for i in range(M):
            i2 = (i + 1) % M
            for j in range(N - 1):
                # Outer (CCW outward)
                a, b, c, d = idx_outer(i, j), idx_outer(i, j + 1), idx_outer(i2, j), idx_outer(i2, j + 1)
                faces.append((a, b, d))
                faces.append((a, d, c))
                # Inner (flip winding so outward is still exterior)
                a, b, c, d = idx_inner(i, j), idx_inner(i2, j), idx_inner(i, j + 1), idx_inner(i2, j + 1)
                faces.append((a, b, d))
                faces.append((a, d, c))

        # Side walls at v edges (j=0 and j=N-1)
        for i in range(M):
            i2 = (i + 1) % M

            # j = 0 side
            o_a, o_b = idx_outer(i, 0), idx_outer(i2, 0)
            i_a, i_b = idx_inner(i, 0), idx_inner(i2, 0)
            # two triangles, oriented outward
            faces.append((o_a, i_b, i_a))
            faces.append((o_a, o_b, i_b))

            # j = N-1 side
            o_a, o_b = idx_outer(i, N - 1), idx_outer(i2, N - 1)
            i_a, i_b = idx_inner(i, N - 1), idx_inner(i2, N - 1)
            faces.append((o_a, i_a, i_b))
            faces.append((o_a, i_b, o_b))
    else:
        # Single-sided (non-manifold) sheet
        verts = P.reshape(-1, 3)
        for i in range(M):
            i2 = (i + 1) % M
            for j in range(N - 1):
                a = i * N + j
                b = a + 1
                c = i2 * N + j
                d = c + 1
                faces.append((a, b, d))
                faces.append((a, d, c))

    return verts.astype(np.float32, copy=False), faces


def save_obj(path: Path, vertices: np.ndarray, faces: list[tuple[int, int, int]]):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")


def write_binary_stl(path: Path, vertices: np.ndarray, faces: list[tuple[int, int, int]]):
    path = Path(path)
    with path.open("wb") as f:
        f.write(b"AdaptiveCAD Mobius".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        V = vertices
        for a, b, c in faces:
            A, B, C = V[a], V[b], V[c]
            n = np.cross(B - A, C - A)
            nl = np.linalg.norm(n) + 1e-20
            n = n / nl
            f.write(struct.pack("<3f", float(n[0]), float(n[1]), float(n[2])))
            for P in (A, B, C):
                f.write(struct.pack("<3f", float(P[0]), float(P[1]), float(P[2])))
            f.write(struct.pack("<H", 0))


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive π Möbius generator (OBJ/STL). Units: mm.")
    p.add_argument("--radius-mm", type=float, default=40.0)
    p.add_argument("--half-width-mm", type=float, default=8.0)
    p.add_argument("--twists", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--samples-major", type=int, default=480)
    p.add_argument("--samples-width", type=int, default=48)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--proj-mode", choices=["euclidean", "lorentz", "complex", "hybrid"], default="hybrid")
    p.add_argument("--thickness-mm", type=float, default=2.0, help="0 for sheet; >0 for manifold band")
    p.add_argument("--kappa", type=float, default=0.0, help="π-adaptive curvature (0=standard, >0=warped)")
    p.add_argument("--format", choices=["obj", "stl", "both"], default="both")
    p.add_argument("--out", type=Path, default=None, help="Output basename without extension")
    return p.parse_args()


def main():
    args = parse_args()
    V, F = adaptive_mobius_unity(
        radius_mm=args.radius_mm,
        half_width_mm=args.half_width_mm,
        twists=args.twists,
        gamma=args.gamma,
        samples_major=args.samples_major,
        samples_width=args.samples_width,
        tau=args.tau,
        proj_mode=args.proj_mode,
        thickness_mm=args.thickness_mm,
        kappa=args.kappa,
    )
    base = args.out if args.out else (DEFAULT_OBJ.with_suffix(""))
    if args.format in ("obj", "both"):
        save_obj(base.with_suffix(".obj"), V, F)
    if args.format in ("stl", "both"):
        write_binary_stl(base.with_suffix(".stl"), V, F)
    print(f"Wrote: {', '.join(str(p) for p in [base.with_suffix('.obj') if args.format in ('obj','both') else None, base.with_suffix('.stl') if args.format in ('stl','both') else None] if p)}")


if __name__ == "__main__":
    main()