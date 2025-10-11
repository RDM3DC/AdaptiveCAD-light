"""Simple in-memory job manager with progress streaming."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import uuid4

JobWorker = Callable[[Dict[str, Any], "JobContext"], Awaitable[Dict[str, Any]]]


@dataclass
class JobContext:
    """Runtime context supplied to job workers."""

    job_id: str
    _queue: "asyncio.Queue[str]"

    async def emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Push a structured event to the SSE stream."""

        payload = json.dumps({"type": event_type, "data": data})
        await self._queue.put(payload)


@dataclass
class JobRecord:
    """Internal job record state."""

    id: str
    kind: str
    status: str = "queued"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    _queue: "asyncio.Queue[str]" = field(default_factory=asyncio.Queue)

    @property
    def done(self) -> bool:
        return self.status in {"completed", "failed"}


class JobManager:
    """Manage async job execution and streaming events."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._workers: Dict[str, JobWorker] = {}
        self._lock = asyncio.Lock()

    def register_worker(self, kind: str, worker: JobWorker) -> None:
        self._workers[kind] = worker

    async def submit(self, kind: str, payload: Dict[str, Any]) -> JobRecord:
        if kind not in self._workers:
            raise ValueError(f"No worker registered for job kind '{kind}'")

        job_id = str(uuid4())
        job = JobRecord(id=job_id, kind=kind, payload=payload)
        async with self._lock:
            self._jobs[job_id] = job

        context = JobContext(job_id=job_id, _queue=job._queue)
        worker = self._workers[kind]
        asyncio.create_task(self._run_job(job, worker, context))
        return job

    async def _run_job(self, job: JobRecord, worker: JobWorker, context: JobContext) -> None:
        try:
            await context.emit("status", {"state": "started"})
            job.status = "running"
            job.updated_at = datetime.utcnow()
            result = await worker(job.payload, context)
            job.result = result
            job.status = "completed"
            await context.emit("status", {"state": "completed"})
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "failed"
            await context.emit("status", {"state": "failed", "message": job.error})
        finally:
            job.updated_at = datetime.utcnow()
            await context.emit("status", {"state": job.status})

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    async def event_generator(self, job_id: str):
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)

        queue = job._queue
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield message
                if job.done and queue.empty():
                    break
            except asyncio.TimeoutError:
                if job.done:
                    break
                continue

    def serialize(self, job: JobRecord) -> Dict[str, Any]:
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "created_at": job.created_at.isoformat() + "Z",
            "updated_at": job.updated_at.isoformat() + "Z",
            "payload": job.payload,
            "result": job.result,
            "error": job.error,
        }
