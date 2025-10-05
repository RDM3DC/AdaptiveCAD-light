# adaptivecad_playground/dimensions.py
"""
Dimensions v0.1 — dataclasses, formatting, (de)serialization, πₐ-aware labels.
Drop into adaptivecad_playground/ and import where needed.

Public API (minimal):
- DimStyle, LinearDimension, RadialDimension, AngularDimension
- to_json(scene_dims), from_json(json_dims)
- format_length(value, style)
- linear_label(p1,p2,style)
- radial_label(center,radius,style, params)
- angular_label(vtx,p1,p2,style)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, Literal, Dict, Any, List

from geometry import angle_deg, euclid_len, radial_label as geom_radial_label, format_length as geom_format_length

Point = Tuple[float, float]
Mode = Literal["euclid", "pi_a", "both"]

# ---- Style & Units ---------------------------------------------------------

@dataclass
class DimStyle:
    units: str = "in"           # "in","mm","ft"
    precision: int = 2
    mode: Mode = "both"         # what to display in labels
    text_height: float = 12.0   # pixels (paper-space)
    arrow: float = 8.0          # pixels
    color: str = "#222222"

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["text_height"] = float(self.text_height)
        d["arrow"] = float(self.arrow)
        return d

def format_length(L_model: float, style: DimStyle) -> str:
    return geom_format_length(L_model, style.units, style.precision)

# ---- Dimension dataclasses -------------------------------------------------

@dataclass
class LinearDimension:
    id: str = ""
    p1: Point = field(default_factory=lambda: (0.0, 0.0))
    p2: Point = field(default_factory=lambda: (0.0, 0.0))
    offset: Point = field(default_factory=lambda: (0.0, 0.0))  # label anchor position in world coords
    style: DimStyle = field(default_factory=DimStyle)

    def asdict(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "id": self.id,
            "p1": list(self.p1),
            "p2": list(self.p2),
            "offset": list(self.offset),
            "style": self.style.asdict(),
        }

@dataclass
class RadialDimension:
    id: str = ""
    center: Point = field(default_factory=lambda: (0.0, 0.0))
    radius: float = 0.0
    attach: Point = field(default_factory=lambda: (0.0, 0.0))  # label anchor position in world coords
    style: DimStyle = field(default_factory=DimStyle)
    shape_ref: Optional[str] = None  # optional link to a circle id
    params: Optional[Dict[str, float]] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        data = {
            "type": "radial",
            "id": self.id,
            "center": list(self.center),
            "radius": float(self.radius),
            "attach": list(self.attach),
            "style": self.style.asdict(),
            "shape_ref": self.shape_ref,
        }
        if self.params is not None:
            data["params"] = dict(self.params)
        return data

@dataclass
class AngularDimension:
    id: str = ""
    vtx: Point = field(default_factory=lambda: (0.0, 0.0))         # vertex
    p1: Point = field(default_factory=lambda: (0.0, 0.0))          # point on ray 1
    p2: Point = field(default_factory=lambda: (0.0, 0.0))          # point on ray 2
    attach: Point = field(default_factory=lambda: (0.0, 0.0))      # label anchor
    style: DimStyle = field(default_factory=DimStyle)

    def asdict(self) -> Dict[str, Any]:
        return {
            "type": "angular",
            "id": self.id,
            "vtx": list(self.vtx),
            "p1": list(self.p1),
            "p2": list(self.p2),
            "attach": list(self.attach),
            "style": self.style.asdict(),
        }

# ---- Serialization helpers -------------------------------------------------

def to_json(dimensions: List[object]) -> List[Dict[str, Any]]:
    out = []
    for d in dimensions:
        if hasattr(d, "asdict"):
            out.append(d.asdict())
    return out

def from_json(items: List[Dict[str, Any]]) -> List[object]:
    dims = []
    for it in items or []:
        style = DimStyle(**it.get("style", {}))
        typ = it.get("type")
        if typ == "linear":
            dims.append(LinearDimension(
                id=it["id"],
                p1=tuple(it["p1"]),
                p2=tuple(it["p2"]),
                offset=tuple(it["offset"]),
                style=style
            ))
        elif typ == "radial":
            dims.append(RadialDimension(
                id=it["id"],
                center=tuple(it["center"]),
                radius=float(it["radius"]),
                attach=tuple(it["attach"]),
                style=style,
                shape_ref=it.get("shape_ref"),
                params=it.get("params"),
            ))
        elif typ == "angular":
            dims.append(AngularDimension(
                id=it["id"],
                vtx=tuple(it["vtx"]),
                p1=tuple(it["p1"]),
                p2=tuple(it["p2"]),
                attach=tuple(it["attach"]),
                style=style
            ))
    return dims

# ---- Label builders --------------------------------------------------------

def linear_label(p1: Point, p2: Point, style: DimStyle) -> str:
    L = euclid_len(p1, p2)
    lab = format_length(L, style)
    return lab

def radial_label(_center: Point, radius: float, style: DimStyle, params: Optional[Dict[str, float]]) -> str:
    return geom_radial_label(radius, params or {}, style)

def angular_label(vtx: Point, p1: Point, p2: Point, style: DimStyle) -> str:
    a = angle_deg(vtx, p1, p2)
    fmt = f"{{:.{style.precision}f}}°"
    return fmt.format(a)
