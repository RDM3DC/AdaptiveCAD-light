"""Pydantic schemas used by the AdaptiveCAD MCP server tools."""
from __future__ import annotations

from typing import Iterable, Literal, Sequence

from pydantic import BaseModel, Field, validator

PlaneLiteral = Literal["XY", "YZ", "ZX"]


class CurveParams(BaseModel):
    alpha: float = Field(0.0, description="Adaptive feedback gain controlling πₐ reinforcement.")
    mu: float = Field(0.0, description="Exponential decay applied to the πₐ radius response.")
    k0: float = Field(0.0, description="Seed curvature used to scale the baseline π value.")


class CurveRequest(BaseModel):
    control_points: list[tuple[float, float]] = Field(
        ..., description="Bezier-like control points in 2D canvas space (x, y)."
    )
    params: CurveParams | None = Field(
        default=None, description="Optional adaptive πₐ parameters applied to the curve sampling."
    )
    samples: int = Field(256, ge=8, le=4096, description="Number of samples to generate along the curve.")


class CurveResponse(BaseModel):
    polyline: list[tuple[float, float]] = Field(
        ..., description="Sampled polyline points representing the adaptive curve."
    )
    length: float = Field(..., ge=0.0, description="Approximate arc length of the sampled curve.")
    area: float = Field(..., description="Signed area enclosed by closing the sampled polyline.")


class ProfileSummary(BaseModel):
    name: str = Field(..., description="Identifier for the canonical revolve profile.")
    plane: PlaneLiteral = Field(..., description="Sketch plane associated with the profile.")
    points: list[tuple[float, float]] = Field(
        ..., description="2D profile coordinates (x, y) ordered along the sketch polyline."
    )


class ProfileListResponse(BaseModel):
    profiles: list[ProfileSummary] = Field(..., description="Canonical profiles bundled with AdaptiveCAD.")


class RevolveProfile(BaseModel):
    plane: PlaneLiteral = Field(
        "ZX",
        description="Sketch plane the 2D profile uses before lifting into 3D. Matches playground defaults.",
    )
    points: list[tuple[float, float]] = Field(
        default_factory=list,
        description="2D profile points (x, y) ordered along the sketch polyline.",
    )

    @validator("points", pre=True)
    def _coerce_points(
        cls, value: Iterable[Sequence[float]] | None
    ) -> list[tuple[float, float]]:  # noqa: N805 - pydantic validator signature
        pts: list[tuple[float, float]] = []
        if value is None:
            return pts
        for pair in value:
            if len(pair) != 2:
                raise ValueError("Profile points must be 2D sequences")
            pts.append((float(pair[0]), float(pair[1])))
        return pts


class RevolveRequest(BaseModel):
    profile: RevolveProfile = Field(..., description="2D sketch profile to revolve.")
    axis_point: tuple[float, float, float] = Field(
        (0.0, 0.0, 0.0), description="Point on the revolution axis in 3D space."
    )
    axis_dir: tuple[float, float, float] = Field(
        (0.0, 1.0, 0.0), description="Direction vector of the revolution axis.")
    angle_degrees: float = Field(360.0, gt=0.0, le=720.0, description="Sweep angle for the revolve in degrees.")
    theta_offset_deg: float = Field(0.0, ge=0.0, le=360.0, description="Initial seam rotation in degrees.")
    tolerance: float = Field(0.0005, gt=0.0, description="Chord error tolerance controlling segment density.")
    eta_k: float = Field(0.4, ge=0.0, description="Curvature-based tightening factor.")
    eta_m: float = Field(0.3, ge=0.0, description="Memory field tightening factor.")
    smart_seam: bool = Field(True, description="Enable adaptive seam selection based on memory sampling.")
    cap_start: bool = Field(True, description="Cap the start of the mesh when the profile touches the axis.")
    cap_end: bool = Field(True, description="Cap the end of the mesh when the profile touches the axis.")
    min_segments: int = Field(24, ge=8, le=2048, description="Minimum number of angular segments.")
    max_segments: int = Field(720, ge=32, le=4096, description="Maximum number of angular segments.")


class Mesh(BaseModel):
    vertices: list[tuple[float, float, float]] = Field(
        ..., description="Mesh vertices as 3D coordinates (x, y, z)."
    )
    faces: list[tuple[int, int, int]] = Field(
        ..., description="Triangle indices referencing the vertices list."
    )


class RevolveResponse(BaseModel):
    mesh: Mesh = Field(..., description="Triangle mesh generated from revolving the supplied profile.")
    segments: int = Field(..., ge=1, description="Number of angular segments used in the revolve.")
    theta0: float = Field(..., description="Seam offset used by the revolve in radians.")


class ObjExportRequest(BaseModel):
    mesh: Mesh = Field(..., description="Triangle mesh to convert into OBJ format.")
    object_name: str = Field(
        "AdaptiveCADMesh",
        min_length=1,
        max_length=128,
        description="Name used for the OBJ object declaration.",
    )


class ObjExportResponse(BaseModel):
    obj: str = Field(..., description="OBJ payload as a UTF-8 encoded string.")
    triangle_count: int = Field(..., ge=0, description="Number of faces exported to OBJ.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field("ok", description="Service status indicator.")
    version: str = Field(..., description="Version string reported by the MCP server.")


class AdaptiveMobiusRequest(BaseModel):
    """Request to generate an adaptive Möbius band with π-curvature."""
    radius_mm: float = Field(default=40.0, description="Ring radius in millimeters")
    half_width_mm: float = Field(default=8.0, description="Half-width of the band")
    twists: float = Field(default=1.5, description="Number of half-twists (0.5=single Möbius)")
    gamma: float = Field(default=0.25, description="4D time-like coordinate amplitude")
    tau: float = Field(default=0.5, description="Projection morph parameter [0,1]")
    proj_mode: Literal["euclidean", "lorentz", "complex", "hybrid"] = Field(default="hybrid")
    kappa: float = Field(default=0.0, description="π-adaptive curvature parameter")
    thickness_mm: float = Field(default=2.0, description="Band thickness (0=sheet)")
    samples_major: int = Field(default=480, description="Samples along ring")
    samples_width: int = Field(default=48, description="Samples across width")


class AdaptiveMobiusResponse(BaseModel):
    """Response containing the generated adaptive Möbius mesh."""
    mesh: Mesh
    parameters: AdaptiveMobiusRequest