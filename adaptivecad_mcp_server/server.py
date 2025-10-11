"""Model Context Protocol server exposing AdaptiveCAD-lite utilities."""
from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence, cast

from mcp.server.fastmcp import FastMCP

from adaptivecad_core.revolve import revolve_adaptive
from adaptivecad_playground.geometry import curve_points, polygon_area, polyline_length
from adaptivecad_playground.sketch_mode import SketchPlane, canonical_sketch_profiles, lift_polyline

from .schemas import (
    AdaptiveMobiusRequest,
    AdaptiveMobiusResponse,
    CurveRequest,
    CurveResponse,
    CurveParams,
    HealthResponse,
    Mesh,
    ObjExportRequest,
    ObjExportResponse,
    PlaneLiteral,
    ProfileListResponse,
    ProfileSummary,
    RevolveRequest,
    RevolveResponse,
)

SERVER_VERSION = "0.1.0"

server = FastMCP(
    name="adaptivecad-lite-mcp",
    instructions=(
        "Expose AdaptiveCAD Lite geometry helpers as MCP tools. "
        "Available tools cover adaptive curve sampling, profile revolves, and OBJ export."
    ),
    website_url="https://github.com/RDM3DC/AdaptiveCAD-light",
)


def _params_to_dict(params: CurveParams | None) -> dict | None:
    if params is None:
        return None
    return params.model_dump()


def _coerce_polyline(points: Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    coerced: list[tuple[float, float]] = []
    for pair in points:
        seq = tuple(pair)
        if len(seq) != 2:
            raise ValueError("Polyline points must be 2D sequences")
        coerced.append((float(seq[0]), float(seq[1])))
    return coerced


@server.tool(name="health_check", title="Health Check", description="Validate the AdaptiveCAD MCP server is available.")
def health_check() -> HealthResponse:
    """Return a lightweight service status payload."""

    return HealthResponse(status="ok", version=SERVER_VERSION)


@server.tool(
    name="list_profiles",
    title="List Canonical Profiles",
    description="Return the bundled revolve profile sketches shipped with AdaptiveCAD Lite.",
)
def list_profiles() -> ProfileListResponse:
    profiles = canonical_sketch_profiles()
    summaries: list[ProfileSummary] = []
    for name, spec in profiles.items():
        plane_raw = spec.get("plane", "ZX")
        plane = SketchPlane(plane_raw) if not isinstance(plane_raw, SketchPlane) else plane_raw
        points_raw = cast(Iterable[Sequence[float]], spec.get("points", []))
        plane_literal: PlaneLiteral = cast(PlaneLiteral, plane.value if isinstance(plane, SketchPlane) else str(plane))
        summaries.append(
            ProfileSummary(
                name=str(name),
                plane=plane_literal,
                points=_coerce_polyline(points_raw),
            )
        )
    return ProfileListResponse(profiles=summaries)


@server.tool(
    name="generate_curve",
    title="Generate Adaptive Curve",
    description="Sample an adaptive πₐ curve defined by control points and kernel parameters.",
)
def generate_curve(payload: CurveRequest) -> CurveResponse:
    if len(payload.control_points) < 2:
        raise ValueError("At least two control points are required to sample a curve")

    params_dict = _params_to_dict(payload.params)
    polyline = curve_points(payload.control_points, params_dict, samples=payload.samples)
    sampled = [(float(p[0]), float(p[1])) for p in polyline.tolist()]
    length = polyline_length(polyline)
    area = polygon_area(polyline)
    return CurveResponse(polyline=sampled, length=float(length), area=float(area))


@server.tool(
    name="revolve_profile",
    title="Revolve Profile",
    description="Revolve a 2D sketch profile into a triangle mesh using the adaptive πₐ kernel.",
)
def revolve_profile(payload: RevolveRequest) -> RevolveResponse:
    if len(payload.profile.points) < 2:
        raise ValueError("Profile must contain at least two points")

    plane = SketchPlane(payload.profile.plane)
    profile_3d = lift_polyline(payload.profile.points, plane)
    angle_rad = math.radians(payload.angle_degrees)
    theta0 = math.radians(payload.theta_offset_deg)

    result = revolve_adaptive(
        profile_3d,
        payload.axis_point,
        payload.axis_dir,
        angle=angle_rad,
        theta0=theta0,
        tol=payload.tolerance,
        eta_k=payload.eta_k,
        eta_m=payload.eta_m,
        smart_seam=payload.smart_seam,
        cap_start=payload.cap_start,
        cap_end=payload.cap_end,
        min_segments=payload.min_segments,
        max_segments=payload.max_segments,
    )

    vertices: list[tuple[float, float, float]] = []
    for row in result.vertices.tolist():
        if len(row) != 3:
            raise ValueError("Mesh vertex rows must contain 3 floats")
        x, y, z = (float(coord) for coord in row)
        vertices.append((x, y, z))

    faces: list[tuple[int, int, int]] = []
    for face in result.faces.tolist():
        if len(face) != 3:
            raise ValueError("Mesh faces must be triangles")
        i0, i1, i2 = (int(index) for index in face)
        faces.append((i0, i1, i2))

    mesh = Mesh(vertices=vertices, faces=faces)
    return RevolveResponse(mesh=mesh, segments=result.segments, theta0=float(result.theta0))


@server.tool(
    name="export_obj",
    title="Export OBJ",
    description="Convert a triangle mesh into ASCII OBJ format for downstream tools.",
)
def export_obj(payload: ObjExportRequest) -> ObjExportResponse:
    lines = [f"o {payload.object_name}"]
    for vx, vy, vz in payload.mesh.vertices:
        lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")
    for tri in payload.mesh.faces:
        if len(tri) != 3:
            raise ValueError("OBJ export expects triangles only")
        a, b, c = (int(index) + 1 for index in tri)
        lines.append(f"f {a} {b} {c}")
    obj_data = "\n".join(lines) + "\n"
    return ObjExportResponse(obj=obj_data, triangle_count=len(payload.mesh.faces))


@server.tool(
    name="generate_adaptive_mobius",
    title="Generate Adaptive Möbius",
    description="Create a π-adaptive Möbius band with projection morphing and curvature modulation.",
)
def generate_adaptive_mobius(payload: AdaptiveMobiusRequest) -> AdaptiveMobiusResponse:
    """Generate an adaptive Möbius band using the π-adaptive generator."""
    import sys
    from pathlib import Path
    
    # Import the adaptive_mobius_unity function
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from adaptive_mobius_unity import adaptive_mobius_unity
    
    vertices_arr, faces_list = adaptive_mobius_unity(
        radius_mm=payload.radius_mm,
        half_width_mm=payload.half_width_mm,
        twists=payload.twists,
        gamma=payload.gamma,
        samples_major=payload.samples_major,
        samples_width=payload.samples_width,
        tau=payload.tau,
        proj_mode=payload.proj_mode,
        thickness_mm=payload.thickness_mm,
        kappa=payload.kappa,
    )
    
    vertices = [(float(row[0]), float(row[1]), float(row[2])) for row in vertices_arr]
    faces = [(int(tri[0]), int(tri[1]), int(tri[2])) for tri in faces_list]
    
    mesh = Mesh(vertices=vertices, faces=faces)
    return AdaptiveMobiusResponse(mesh=mesh, parameters=payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AdaptiveCAD MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport used by the server. VS Code typically uses 'stdio', remote hosts use HTTP.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for HTTP transports.")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports.")
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path used when running under SSE transport (e.g. '/adaptivecad').",
    )
    args = parser.parse_args()

    server.settings.host = args.host
    server.settings.port = int(args.port)

    if args.transport == "stdio":
        server.run("stdio")
    elif args.transport == "sse":
        server.run("sse", mount_path=args.mount_path)
    else:
        server.settings.mount_path = args.mount_path
        server.run("streamable-http")


if __name__ == "__main__":
    main()
