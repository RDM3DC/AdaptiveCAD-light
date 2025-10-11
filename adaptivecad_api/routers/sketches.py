from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ..adapters import sketch as sketch_adapter


class SketchCreate(BaseModel):
    name: str = Field(default="Untitled", description="Sketch name")
    units: str = Field(default="mm", description="Unit system")


class SketchResponse(BaseModel):
    id: str
    name: str
    units: str
    created_at: str
    updated_at: str
    entities: List[Dict[str, Any]]


class EntityCreate(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class EditRequest(BaseModel):
    operation: str
    payload: Dict[str, Any] = Field(default_factory=dict)


router = APIRouter(prefix="/sketches", tags=["sketches"])


@router.get("/", response_model=List[SketchResponse])
async def list_sketches() -> List[SketchResponse]:
    return [SketchResponse(**item) for item in sketch_adapter.list_sketches()]


@router.post("/", response_model=SketchResponse, status_code=status.HTTP_201_CREATED)
async def create_sketch(body: SketchCreate) -> SketchResponse:
    item = sketch_adapter.create_sketch(body.model_dump())
    return SketchResponse(**item)


@router.get("/{sketch_id}", response_model=SketchResponse)
async def get_sketch(sketch_id: str) -> SketchResponse:
    try:
        data = sketch_adapter.get_sketch(sketch_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sketch not found") from exc
    return SketchResponse(**data)


@router.delete("/{sketch_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sketch(sketch_id: str) -> None:
    try:
        sketch_adapter.delete_sketch(sketch_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sketch not found") from exc


@router.post("/{sketch_id}/entities", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def add_entity(sketch_id: str, body: EntityCreate) -> Dict[str, Any]:
    try:
        return sketch_adapter.add_entity(sketch_id, body.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sketch not found") from exc


@router.post("/{sketch_id}/edit")
async def run_edit(sketch_id: str, body: EditRequest) -> Dict[str, Any]:
    try:
        return sketch_adapter.run_edit(sketch_id, body.operation, body.payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sketch not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
