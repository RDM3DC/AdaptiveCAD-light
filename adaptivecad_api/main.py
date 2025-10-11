from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import FastAPI

from .adapters import metrics as metrics_adapter
from .adapters import sketch as sketch_adapter
from .adapters import solids as solids_adapter
from .adapters import exporter as exporter_adapter
from .jobs import JobContext, JobManager
from .routers import sketches as sketches_router
from .routers.jobs import build as build_jobs_router

app = FastAPI(title="AdaptiveCAD API", version="0.1.0", description="Async job router for AdaptiveCAD operations")
job_manager = JobManager()


async def solids_worker(kind: str, payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    if kind == "revolve":
        return await solids_adapter.revolve(payload, context.emit)
    if kind == "extrude":
        return await solids_adapter.extrude(payload, context.emit)
    if kind == "boolean":
        return await solids_adapter.boolean(payload, context.emit)
    raise ValueError(f"Unsupported solids worker '{kind}'")


async def revolve_worker(payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    return await solids_worker("revolve", payload, context)


async def extrude_worker(payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    return await solids_worker("extrude", payload, context)


async def boolean_worker(payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    return await solids_worker("boolean", payload, context)


async def measure_worker(payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    await context.emit("progress", {"message": "Computing measurements"})
    await asyncio.sleep(0)
    return metrics_adapter.measure(payload)


async def export_worker(payload: Dict[str, Any], context: JobContext) -> Dict[str, Any]:
    await context.emit("progress", {"message": "Preparing export"})
    await asyncio.sleep(0)
    return await exporter_adapter.export_scene(payload)


def _register_workers() -> None:
    job_manager.register_worker("revolve", revolve_worker)
    job_manager.register_worker("extrude", extrude_worker)
    job_manager.register_worker("boolean", boolean_worker)
    job_manager.register_worker("measure", measure_worker)
    job_manager.register_worker("export", export_worker)


_register_workers()

app.include_router(sketches_router.router)
app.include_router(build_jobs_router(job_manager))


@app.get("/")
async def index() -> Dict[str, Any]:
    return {
        "name": "adaptivecad-api",
        "version": app.version,
        "routes": [
            {"path": "/sketches", "methods": ["GET", "POST"]},
            {"path": "/jobs", "methods": ["GET", "POST"]},
        ],
        "sketch_count": len(sketch_adapter.list_sketches()),
    }
