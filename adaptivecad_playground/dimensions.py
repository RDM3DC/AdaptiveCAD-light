"""Dimension data models and helpers for AdaptiveCAD."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

Point = Tuple[float, float]
Mode = Literal["euclid", "pi_a", "both"]


@dataclass
class DimStyle:
    """Visual and formatting style for dimensions."""

    units: str = "in"
    precision: int = 2
    mode: Mode = "both"
    text_height: float = 12.0  # screen-space pixels
    arrow: float = 8.0         # screen-space pixels
    color: str = "#222222"
    locked: bool = False

    def to_json(self) -> dict:
        data = asdict(self)
        # JSON keeps a string for color and the rest as numbers/strings
        return data

    @staticmethod
    def from_json(payload: Optional[dict]) -> "DimStyle":
        if not payload:
            return DimStyle()
        return DimStyle(
            units=str(payload.get("units", "in")),
            precision=int(payload.get("precision", 2)),
            mode=payload.get("mode", "both"),
            text_height=float(payload.get("text_height", 12.0)),
            arrow=float(payload.get("arrow", 8.0)),
            color=str(payload.get("color", "#222222")),
            locked=bool(payload.get("locked", False)),
        )


@dataclass
class LinearDimension:
    id: str
    p1: Point
    p2: Point
    offset: Point
    style: DimStyle

    def to_json(self) -> dict:
        return {
            "type": "linear",
            "id": self.id,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "offset": list(self.offset),
            "style": self.style.to_json(),
        }

    @staticmethod
    def from_json(payload: dict) -> "LinearDimension":
        return LinearDimension(
            id=str(payload.get("id", "")),
            p1=_point(payload.get("p1", (0.0, 0.0))),
            p2=_point(payload.get("p2", (0.0, 0.0))),
            offset=_point(payload.get("offset", (0.0, 0.0))),
            style=DimStyle.from_json(payload.get("style")),
        )


@dataclass
class RadialDimension:
    id: str
    center: Point
    radius: float
    attach: Point
    style: DimStyle
    shape_ref: Optional[str] = None

    def to_json(self) -> dict:
        data = {
            "type": "radial",
            "id": self.id,
            "center": list(self.center),
            "radius": float(self.radius),
            "attach": list(self.attach),
            "style": self.style.to_json(),
        }
        if self.shape_ref:
            data["shape_ref"] = self.shape_ref
        return data

    @staticmethod
    def from_json(payload: dict) -> "RadialDimension":
        return RadialDimension(
            id=str(payload.get("id", "")),
            center=_point(payload.get("center", (0.0, 0.0))),
            radius=float(payload.get("radius", 0.0)),
            attach=_point(payload.get("attach", (0.0, 0.0))),
            style=DimStyle.from_json(payload.get("style")),
            shape_ref=payload.get("shape_ref"),
        )


@dataclass
class AngularDimension:
    id: str
    vtx: Point
    p1: Point
    p2: Point
    attach: Point
    style: DimStyle

    def to_json(self) -> dict:
        return {
            "type": "angular",
            "id": self.id,
            "vtx": list(self.vtx),
            "p1": list(self.p1),
            "p2": list(self.p2),
            "attach": list(self.attach),
            "style": self.style.to_json(),
        }

    @staticmethod
    def from_json(payload: dict) -> "AngularDimension":
        return AngularDimension(
            id=str(payload.get("id", "")),
            vtx=_point(payload.get("vtx", (0.0, 0.0))),
            p1=_point(payload.get("p1", (1.0, 0.0))),
            p2=_point(payload.get("p2", (0.0, 1.0))),
            attach=_point(payload.get("attach", (0.0, 0.0))),
            style=DimStyle.from_json(payload.get("style")),
        )


Dimension = Union[LinearDimension, RadialDimension, AngularDimension]


def dimensions_to_json(dimensions: Iterable[Dimension]) -> list[dict]:
    return [dim.to_json() for dim in dimensions]


def dimensions_from_json(payload: Optional[Sequence[dict]]) -> list[Dimension]:
    if not payload:
        return []
    result: list[Dimension] = []
    for item in payload:
        dim_type = item.get("type")
        if dim_type == "linear":
            result.append(LinearDimension.from_json(item))
        elif dim_type == "radial":
            result.append(RadialDimension.from_json(item))
        elif dim_type == "angular":
            result.append(AngularDimension.from_json(item))
    return result


def _point(value: Sequence[float]) -> Point:
    if len(value) >= 2:
        return (float(value[0]), float(value[1]))
    # Fallback to origin for malformed data
    return (0.0, 0.0)
