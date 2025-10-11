from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..jobs import JobManager, JobRecord


class JobSubmission(BaseModel):
    worker: str = Field(..., description="Registered worker name")
    payload: Dict[str, Any] = Field(default_factory=dict)


class JobSummary(BaseModel):
    id: str
    kind: str
    status: str
    payload: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobSummary":
        return cls(
            id=record.id,
            kind=record.kind,
            status=record.status,
            payload=record.payload,
            result=record.result,
            error=record.error,
        )


def build(job_manager: JobManager) -> APIRouter:
    router = APIRouter(prefix="/jobs", tags=["jobs"])

    @router.post("/", response_model=JobSummary, status_code=status.HTTP_202_ACCEPTED)
    async def submit_job(submission: JobSubmission) -> JobSummary:
        try:
            record = await job_manager.submit(submission.worker, submission.payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JobSummary.from_record(record)

    @router.get("/{job_id}", response_model=JobSummary)
    async def get_job(job_id: str) -> JobSummary:
        record = job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return JobSummary.from_record(record)

    @router.get("/{job_id}/stream")
    async def stream_job(job_id: str):
        record = job_manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

        async def event_source():
            async for message in job_manager.event_generator(job_id):
                data = json.loads(message)
                yield "event: {0}\n".format(data["type"]) + f"data: {json.dumps(data['data'])}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return router
