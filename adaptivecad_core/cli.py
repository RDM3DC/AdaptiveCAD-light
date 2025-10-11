"""Command line interface for AdaptiveCAD workflows."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from adaptivecad_core.revolve import revolve_adaptive  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when executed as script
    from revolve import revolve_adaptive  # type: ignore

try:
    from adaptivecad_playground.sketch_mode import (
        SketchPlane,
        canonical_sketch_profiles,
        lift_polyline,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when executed as script
    from sketch_mode import SketchPlane, canonical_sketch_profiles, lift_polyline  # type: ignore


def _write_obj(path: Path, vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for vx, vy, vz in vertices:
            handle.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for face in faces:
            if len(face) != 3:
                raise ValueError("OBJ export expects triangulated faces")
            a, b, c = (int(f) + 1 for f in face)
            handle.write(f"f {a} {b} {c}\n")


def _coerce_plane(value: str | SketchPlane | None) -> SketchPlane:
    if isinstance(value, SketchPlane):
        return value
    if value is None:
        return SketchPlane.ZX
    try:
        return SketchPlane[str(value).upper()]
    except KeyError as exc:  # pragma: no cover - input validation
        raise ValueError(f"Unknown sketch plane '{value}'") from exc


def _load_profile_points(profile: str | None, plane: SketchPlane, params: dict | None) -> List[List[float]]:
    if profile is None:
        raise ValueError("Profile name is required when no polyline is provided.")
    bank = canonical_sketch_profiles()
    if profile not in bank:
        raise ValueError(f"Profile '{profile}' not found. Use 'list-profiles' to inspect options.")
    spec = bank[profile]
    spec_plane = spec.get("plane", plane)
    if isinstance(spec_plane, str):
        spec_plane = _coerce_plane(spec_plane)
    pts_2d = spec.get("points", [])
    if not pts_2d:
        raise ValueError(f"Profile '{profile}' has no point data.")
    return [list(pt) for pt in lift_polyline(pts_2d, spec_plane)]


def _read_polyline(path: Path) -> List[List[float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "points" in data:
            data = data["points"]
        elif "polyline" in data:
            data = data["polyline"]
    if not isinstance(data, list):
        raise ValueError("Polyline file must contain a list of point coordinates.")
    pts: List[List[float]] = []
    for item in data:
        if not isinstance(item, (list, tuple)):
            raise ValueError("Each point must be a list or tuple of coordinates.")
        coords = [float(v) for v in item]
        if len(coords) == 2:
            coords.append(0.0)
        if len(coords) != 3:
            raise ValueError("Points must be 2D or 3D.")
        pts.append(coords)
    return pts


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cmd_list_profiles(_args: argparse.Namespace) -> None:
    bank = canonical_sketch_profiles()
    if not bank:
        print("No canonical profiles registered.")
        return
    print("Available profiles:")
    for name, spec in bank.items():
        count = len(spec.get("points", []))
        plane = spec.get("plane", "ZX")
        print(f"  - {name} (points={count}, plane={plane})")


def _cmd_revolve(args: argparse.Namespace) -> None:
    plane = _coerce_plane(args.plane)
    params = {"alpha": args.alpha, "mu": args.mu, "k0": args.k0}
    if args.polyline:
        pts = _read_polyline(Path(args.polyline))
    else:
        pts = _load_profile_points(args.profile, plane, params)
    axis_map = {
        "X": (1.0, 0.0, 0.0),
        "Y": (0.0, 1.0, 0.0),
        "Z": (0.0, 0.0, 1.0),
    }
    axis = axis_map[args.axis.upper()]
    result = revolve_adaptive(
        profile_pts=pts,
        axis_point=(0.0, 0.0, 0.0),
        axis_dir=axis,
        angle=math.radians(args.angle),
        theta0=math.radians(args.theta0),
        tol=args.tolerance,
        eta_k=args.eta_k,
        eta_m=args.eta_m,
        smart_seam=args.smart_seam,
        cap_start=not args.no_cap_start,
        cap_end=not args.no_cap_end,
    )
    out_path = Path(args.output or "adaptive_revolve.obj")
    _ensure_dir(out_path)
    _write_obj(out_path, result.vertices, result.faces)
    print(
        f"Wrote {out_path} | verts={len(result.vertices)} tris={len(result.faces)} segments={result.segments}"  # noqa: E501
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptivecad",
        description="AdaptiveCAD command line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-profiles", help="List canonical sketch profiles")
    list_parser.set_defaults(func=_cmd_list_profiles)

    revolver = sub.add_parser("revolve", help="Generate a revolve mesh and export OBJ")
    revolver.add_argument("--profile", help="Name of the canonical sketch profile to revolve")
    revolver.add_argument("--polyline", help="Path to JSON file containing 2D/3D points")
    revolver.add_argument("--plane", default="ZX", help="Sketch plane for 2D profiles (XY, YZ, ZX)")
    revolver.add_argument("--axis", default="Z", choices=["X", "Y", "Z"], help="Axis of revolution")
    revolver.add_argument("--angle", type=float, default=360.0, help="Sweep angle in degrees")
    revolver.add_argument("--theta0", type=float, default=0.0, help="Starting seam angle in degrees")
    revolver.add_argument("--tolerance", type=float, default=0.001, help="Chord tolerance")
    revolver.add_argument("--eta-k", dest="eta_k", type=float, default=0.4, help="Curvature weighting")
    revolver.add_argument("--eta-m", dest="eta_m", type=float, default=0.3, help="Memory weighting")
    revolver.add_argument("--alpha", type=float, default=0.0, help="πₐ alpha parameter")
    revolver.add_argument("--mu", type=float, default=0.0, help="πₐ mu parameter")
    revolver.add_argument("--k0", type=float, default=0.0, help="πₐ k0 parameter")
    revolver.add_argument("--smart-seam", action="store_true", help="Enable smart seam search")
    revolver.add_argument("--no-cap-start", action="store_true", help="Do not cap the start vertex")
    revolver.add_argument("--no-cap-end", action="store_true", help="Do not cap the end vertex")
    revolver.add_argument("--output", help="Output OBJ path")
    revolver.set_defaults(func=_cmd_revolve)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI guard
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
