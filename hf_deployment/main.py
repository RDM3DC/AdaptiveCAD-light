"""FastAPI entry point for AdaptiveCAD-Lite."""
from __future__ import annotations

from datetime import datetime
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_VERSION = "0.2.0"
APP_BUILD = "lite-pi"

app = FastAPI(
    title="AdaptiveCAD Lite API",
    version=APP_VERSION,
    description="Generate adaptive π and ARP geometry via lightweight endpoints.",
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
        description="Requested artifact format. Supported: obj, stl.",
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
    "adaptive_mobius": ShapeDescription(
        type="adaptive_mobius",
        curvature_profile="π-adaptive Möbius band with Lorentz/complex projections.",
        pi_a_ratio="Kappa parameter controls π-adaptive curvature warping.",
        notes="Use params.tau, params.kappa, params.proj_mode for morphing.",
    ),
}

SUPPORTED_OUTPUTS = {"obj", "stl"}


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
    """Generate geometry using π-adaptive algorithms."""
    
    _validate_request(req)
    shape_id = str(uuid.uuid4())
    
    if req.type == "adaptive_mobius":
        from adaptive_mobius_unity import adaptive_mobius_unity, save_obj, write_binary_stl
        
        # Extract parameters with defaults
        params = req.params
        vertices, faces = adaptive_mobius_unity(
            radius_mm=params.get("radius_mm", 40.0),
            half_width_mm=params.get("half_width_mm", 8.0),
            twists=params.get("twists", 1.5),
            gamma=params.get("gamma", 0.25),
            tau=params.get("tau", 0.5),
            proj_mode=params.get("proj_mode", "hybrid"),
            thickness_mm=params.get("thickness_mm", 2.0),
            kappa=params.get("kappa", 0.0),
            samples_major=params.get("samples_major", 480),
            samples_width=params.get("samples_width", 48),
        )
        
        # Export to temp file based on format
        with tempfile.NamedTemporaryFile(suffix=f".{req.output}", delete=False, mode='wb') as tmp:
            tmp_path = Path(tmp.name)
        
        if req.output == "obj":
            save_obj(tmp_path, vertices, faces)
        elif req.output == "stl":
            write_binary_stl(tmp_path, vertices, faces)
        
        # For HF deployment, return file path for download
        # In production you'd upload to blob storage
        artifact_path = f"/download/{shape_id}.{req.output}"
        
        return {
            "id": shape_id,
            "type": req.type,
            "params": req.params,
            "artifact": artifact_path,
            "output": req.output,
            "status": "generated",
            "vertices": len(vertices),
            "faces": len(faces),
            "file_path": str(tmp_path),  # temporary local path
        }
    else:
        # Fallback for other shapes (still stubbed)
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
