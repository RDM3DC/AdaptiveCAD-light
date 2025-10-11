from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Optional: uncomment if you want SSE helper
# from sse_starlette.sse import EventSourceResponse

APP_VERSION = "0.9.0"
APP_BUILD = "full"

app = FastAPI(title="AdaptiveCAD API (Full)", version=APP_VERSION)

# ---------------------------
# Simple in-memory stores
# ---------------------------
SKETCHES: Dict[str, Dict[str, Any]] = {}
JOBS: Dict[str, Dict[str, Any]] = {}
ARTIFACTS: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# Security (very simple API key bearer for demo purposes)
# ---------------------------
API_KEY = os.getenv("ADAPTIVECAD_API_KEY", "dev-key")
def require_auth(authorization: Optional[str] = None):
    # Expect "Bearer <token>"
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True

# ---------------------------
# Models (mirror OpenAPI)
# ---------------------------
class VersionInfo(BaseModel):
    app: str = "adaptivecad"
    version: str = APP_VERSION
    build: str = APP_BUILD
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CreateSketchRequest(BaseModel):
    name: str = "sketch-1"
    units: str = "mm"
    grid: float = 1.0

class Entity(BaseModel):
    kind: str  # line|arc|bezier|circle|polyline
    p: List[List[float]]

class Sketch(BaseModel):
    id: str
    name: str
    units: str
    entities: List[Entity] = Field(default_factory=list)

class GenerateShapeRequest(BaseModel):
    type: str  # pi_a_circle|pi_a_curve
    params: Dict[str, Any]
    output: str = "json"  # json|obj|png

class Artifact(BaseModel):
    id: str
    kind: str = "file"
    filename: str
    url: Optional[str] = None
    size: Optional[int] = None
    mime: Optional[str] = None

class Job(BaseModel):
    id: str
    kind: str
    status: str = "queued"  # queued|running|completed|failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ---------------------------
# Helpers
# ---------------------------
def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# Placeholder adapters — wire these to your actual kernels:
def generate_pi_a_circle(params: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: call adaptivecad_playground.geometry / adaptive_fields here
    r = float(params.get("radius", 10.0))
    center = params.get("center", [0.0, 0.0])
    return {"kind": "circle", "p": [[center[0]-r, center[1]], [center[0]+r, center[1]]]}

def export_artifact(selection: List[str], fmt: str) -> Artifact:
    # TODO: write files to disk or object storage, hook Blender headless if needed
    art_id = _new_id("art")
    fname = f"{art_id}.{fmt}"
    path = os.path.join("/tmp", fname)
    with open(path, "wb") as f:
        f.write(b"")  # placeholder
    return Artifact(id=art_id, filename=fname, mime="application/octet-stream")

async def _run_job(job_id: str, kind: str, payload: Dict[str, Any]) -> None:
    job = JOBS[job_id]
    job["status"] = "running"
    for step in range(1, 6):
        await asyncio.sleep(0.1)
        job["progress"] = step * 20.0
    # TODO: dispatch to your real kernels by kind
    if kind in {"revolve", "extrude", "solid_boolean"}:
        artifact = export_artifact(payload.get("selection", []), payload.get("format", "obj"))
        job["result"] = {"artifact": artifact.model_dump()}
    job["status"] = "completed"

# ---------------------------
# Routes
# ---------------------------
@app.get("/version", response_model=VersionInfo)
def version():
    return VersionInfo()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/shapes/generate")
def generate_shape(req: GenerateShapeRequest, authorized: bool = Depends(require_auth)):
    if req.type == "pi_a_circle":
        entity = generate_pi_a_circle(req.params)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {req.type}")
    art = None
    if req.output != "json":
        art = export_artifact([_new_id("tmp")], req.output)
    return {
        "id": _new_id("shape"),
        "type": req.type,
        "params": req.params,
        "artifact": art.model_dump() if art else None,
    }

@app.post("/sketches", response_model=Sketch, status_code=201)
def create_sketch(req: CreateSketchRequest, authorized: bool = Depends(require_auth)):
    sid = _new_id("sk")
    sketch = Sketch(id=sid, name=req.name, units=req.units, entities=[])
    SKETCHES[sid] = sketch.model_dump()
    return sketch

@app.get("/sketches")
def list_sketches():
    return {"items": list(SKETCHES.values()), "nextCursor": None}

@app.get("/sketches/{sketch_id}", response_model=Sketch)
def get_sketch(sketch_id: str):
    sk = SKETCHES.get(sketch_id)
    if not sk:
        raise HTTPException(status_code=404, detail="Sketch not found")
    return sk

class AddEntitiesRequest(BaseModel):
    entities: List[Entity]

@app.post("/sketches/{sketch_id}/entities", response_model=Sketch)
def add_entities(sketch_id: str, req: AddEntitiesRequest):
    sk = SKETCHES.get(sketch_id)
    if not sk:
        raise HTTPException(status_code=404, detail="Sketch not found")
    sk["entities"].extend([e.model_dump() for e in req.entities])
    return sk

# --- Jobs ---
class CreateJobRequest(BaseModel):
    kind: str
    payload: Dict[str, Any] = Field(default_factory=dict)

@app.post("/jobs", response_model=Job, status_code=202)
async def create_job(req: CreateJobRequest):
    jid = _new_id("job")
    job = Job(id=jid, kind=req.kind).model_dump()
    JOBS[jid] = job
    # kick off task
    asyncio.create_task(_run_job(jid, req.kind, req.payload))
    return job

@app.get("/jobs", response_model=List[Job])
def list_jobs():
    return [Job(**j) for j in JOBS.values()]

@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return j

# Streaming via simple generator (SSE-like over text/plain for demo)
@app.get("/jobs/{job_id}/events")
def job_events(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    def event_stream():
        last = -1
        while True:
            job = JOBS[job_id]
            pct = job.get("progress", 0)
            if pct != last:
                yield f"event: progress\ndata: {{\"pct\": {pct}}}\n\n"
                last = pct
            if job.get("status") in {"completed", "failed"}:
                yield f"event: done\ndata: {json.dumps(job)}\n\n"
                break
            time.sleep(0.1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")