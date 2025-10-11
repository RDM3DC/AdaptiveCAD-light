"""FastAPI entry point for AdaptiveCAD-Lite."""
from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "0.1.0"
APP_BUILD = "lite"

app = FastAPI(
    title="AdaptiveCAD Lite API",
    version=APP_VERSION,
    description="Generate adaptive pi_a and ARP geometry via lightweight endpoints.",
)


class ShapeRequest(BaseModel):
    """Request payload for geometry generation."""

    type: str = Field(..., description="Template identifier for the shape to generate.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameter overrides for the template (dimension, twist, etc.).",
    )
    output: str = Field(
        default="stl",
        description="Requested artifact format. Supported: stl, svg, png.",
    )


class ShapeDescription(BaseModel):
    """Metadata returned by /describe_shape."""

    type: str
    curvature_profile: str
    pi_a_ratio: str
    notes: str


AVAILABLE_SHAPES: Dict[str, ShapeDescription] = {
    "adaptive_disk": ShapeDescription(
        type="adaptive_disk",
        curvature_profile="Radially adaptive disk with thickness falloff.",
        pi_a_ratio="1.618 baseline with adaptive weighting for rim control.",
        notes="Use params.radius and params.taper to fine tune the silhouette.",
    ),
    "pi_a_torus": ShapeDescription(
        type="pi_a_torus",
        curvature_profile="Smooth torus with pi_a guided major/minor radii.",
        pi_a_ratio="Major:Minor ratio locks to golden-derived pi_a scaling.",
        notes="params.major_radius and params.minor_radius affect final curvature.",
    ),
    "arp_klein_bottle": ShapeDescription(
        type="arp_klein_bottle",
        curvature_profile="Non-orientable ARP field with self-intersection safeguards.",
        pi_a_ratio="Derived from ARP kernel sampling; stabilized for export.",
        notes="Requires params.frequency for wave oscillations.",
    ),
    "curvature_wave": ShapeDescription(
        type="curvature_wave",
        curvature_profile="Longitudinal wave with adaptive amplitude envelope.",
        pi_a_ratio="Oscillation respects pi_a cadence across segments.",
        notes="params.length and params.amplitude control the wave envelope.",
    ),
}

SUPPORTED_OUTPUTS = {"stl", "svg", "png"}


def _validate_request(req: ShapeRequest) -> None:
    """Ensure the caller requests a known template and artifact type."""

    if req.type not in AVAILABLE_SHAPES:
        raise HTTPException(status_code=400, detail=f"Unknown shape type '{req.type}'.")
    if req.output not in SUPPORTED_OUTPUTS:
        raise HTTPException(status_code=400, detail=f"Unsupported output '{req.output}'.")


@app.get("/version")
async def get_version() -> Dict[str, str]:
    """Expose lightweight build and status metadata."""

    return {
        "version": APP_VERSION,
        "build": APP_BUILD,
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/list_shapes")
async def list_shapes() -> Dict[str, List[str]]:
    """Return identifiers for currently available shape templates."""

    return {"available_shapes": list(AVAILABLE_SHAPES)}


@app.post("/generate_shape")
async def generate_shape(req: ShapeRequest) -> Dict[str, Any]:
    """Stubbed geometry generator; returns deterministic metadata for now."""

    _validate_request(req)
    shape_id = str(uuid.uuid4())
    artifact_path = f"https://adaptivecad.com/assets/{shape_id}.{req.output}"

    return {
        "id": shape_id,
        "type": req.type,
        "params": req.params,
        "artifact": artifact_path,
        "output": req.output,
        "status": "generated",
    }


@app.get("/describe_shape")
async def describe_shape(shape_type: str) -> ShapeDescription:
    """Return qualitative metadata about the selected shape template."""

    description = AVAILABLE_SHAPES.get(shape_type)
    if description is None:
        raise HTTPException(status_code=404, detail=f"No description for '{shape_type}'.")
    return description
