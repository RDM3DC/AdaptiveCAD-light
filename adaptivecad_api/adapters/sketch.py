"""In-memory sketch store with hooks into AdaptiveCAD playground modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    from adaptivecad_playground import edit_ops
except ModuleNotFoundError:  # pragma: no cover
    edit_ops = None  # type: ignore[assignment]


@dataclass
class SketchEntity:
    """Representation of a sketch entity."""

    type: str
    params: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class Sketch:
    """Stored sketch with metadata."""

    id: str
    name: str
    units: str
    created_at: datetime
    updated_at: datetime
    entities: List[SketchEntity] = field(default_factory=list)


class SketchStore:
    """Simple store backing the sketch routes."""

    def __init__(self) -> None:
        self._items: Dict[str, Sketch] = {}

    def create(self, name: str, units: str) -> Sketch:
        sketch_id = str(uuid4())
        now = datetime.utcnow()
        sketch = Sketch(id=sketch_id, name=name, units=units, created_at=now, updated_at=now)
        self._items[sketch_id] = sketch
        return sketch

    def list(self) -> List[Sketch]:
        return list(self._items.values())

    def get(self, sketch_id: str) -> Optional[Sketch]:
        return self._items.get(sketch_id)

    def delete(self, sketch_id: str) -> bool:
        return self._items.pop(sketch_id, None) is not None

    def add_entity(self, sketch_id: str, entity_type: str, params: Dict[str, Any]) -> SketchEntity:
        sketch = self._items[sketch_id]
        sketch.updated_at = datetime.utcnow()
        entity = SketchEntity(type=entity_type, params=params)
        sketch.entities.append(entity)
        return entity

    def replace(self, sketch_id: str, entities: List[Dict[str, Any]]) -> Sketch:
        sketch = self._items[sketch_id]
        sketch.entities = [SketchEntity(type=item["type"], params=item.get("params", {})) for item in entities]
        sketch.updated_at = datetime.utcnow()
        return sketch


_store = SketchStore()


def create_sketch(payload: Dict[str, Any]) -> Dict[str, Any]:
    sketch = _store.create(name=payload.get("name", "Untitled"), units=payload.get("units", "mm"))
    return serialize_sketch(sketch)


def list_sketches() -> List[Dict[str, Any]]:
    return [serialize_sketch(item) for item in _store.list()]


def get_sketch(sketch_id: str) -> Dict[str, Any]:
    sketch = _store.get(sketch_id)
    if sketch is None:
        raise KeyError(sketch_id)
    return serialize_sketch(sketch)


def delete_sketch(sketch_id: str) -> None:
    if not _store.delete(sketch_id):
        raise KeyError(sketch_id)


def add_entity(sketch_id: str, entity_payload: Dict[str, Any]) -> Dict[str, Any]:
    entity_type = entity_payload.get("type", "unknown")
    params = entity_payload.get("params", {})
    entity = _store.add_entity(sketch_id, entity_type, params)
    return serialize_entity(entity)


def run_edit(sketch_id: str, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke playground edit ops when available, otherwise return a stub payload."""

    if edit_ops is None:
        return {"status": "no-op", "operation": operation, "payload": payload}

    handler = getattr(edit_ops, operation, None)
    if handler is None:
        raise ValueError(f"Unsupported edit operation '{operation}'")

    sketch = _store.get(sketch_id)
    if sketch is None:
        raise KeyError(sketch_id)

    result = handler(sketch.entities, **payload)
    return {"status": "ok", "result": result}


def serialize_sketch(sketch: Sketch) -> Dict[str, Any]:
    return {
        "id": sketch.id,
        "name": sketch.name,
        "units": sketch.units,
        "created_at": sketch.created_at.isoformat() + "Z",
        "updated_at": sketch.updated_at.isoformat() + "Z",
        "entities": [serialize_entity(entity) for entity in sketch.entities],
    }


def serialize_entity(entity: SketchEntity) -> Dict[str, Any]:
    return {
        "id": entity.id,
        "type": entity.type,
        "params": entity.params,
    }
