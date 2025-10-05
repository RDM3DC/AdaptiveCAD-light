# adaptivecad_playground/dim_tools.py
"""
Dimension tools scaffolds. These are framework-light and expose simple callbacks.
Integrate by forwarding mouse events from your Canvas to the active tool.
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
from .dimensions import LinearDimension, RadialDimension, AngularDimension, DimStyle, linear_label, radial_label, angular_label
from .osnap import osnap_pick

Point = Tuple[float, float]

@dataclass
class ToolContext:
    params: dict                    # {"α":..., "μ":..., "k0":...}
    style: DimStyle
    add_dimension: Callable[[object], None]
    update_status: Callable[[str], None]
    request_repaint: Callable[[], None]
    hit_test_circle: Callable[[Point], Optional[Tuple[Point, float, str]]]  # returns (center, radius, shape_id) if a circle is under cursor
    world_from_event: Callable[[object], Point]  # mouse event -> world (x,y)
    osnap_targets: Callable[[], List[Tuple[str, Point]]]  # named points for snapping

class DimLinearTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.p1: Optional[Point] = None
        self.p2: Optional[Point] = None
        self.offset: Optional[Point] = None

    def on_mouse_press(self, ev):
        p = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
        if self.p1 is None:
            self.p1 = p
            self.ctx.update_status("LinearDim: pick second point")
        elif self.p2 is None:
            self.p2 = p
            self.ctx.update_status("LinearDim: place label")
        else:
            self.offset = p
            # Create dimension
            lab = linear_label(self.p1, self.p2, self.ctx.style)
            did = f"D{abs(hash((self.p1,self.p2,self.offset)))%10**6}"
            dim = LinearDimension(id=did, p1=self.p1, p2=self.p2, offset=self.offset, style=self.ctx.style)
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"LinearDim: {lab}")
            # reset
            self.p1 = self.p2 = self.offset = None
        self.ctx.request_repaint()

class DimRadialTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.center: Optional[Point] = None
        self.radius: Optional[float] = None
        self.attach: Optional[Point] = None
        self.shape_id: Optional[str] = None

    def on_mouse_press(self, ev):
        p = self.ctx.world_from_event(ev)
        if self.center is None:
            res = self.ctx.hit_test_circle(p)
            if res is None:
                self.ctx.update_status("RadialDim: click a πₐ circle")
                return
            self.center, self.radius, self.shape_id = res
            self.ctx.update_status("RadialDim: click to place label")
        else:
            self.attach = p
            lab = radial_label(self.center, self.radius, self.ctx.style, self.ctx.params)
            did = f"D{abs(hash((self.center,self.radius,self.attach)))%10**6}"
            dim = RadialDimension(id=did, center=self.center, radius=self.radius, attach=self.attach, style=self.ctx.style, shape_ref=self.shape_id)
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"RadialDim: {lab}")
            self.center = self.radius = self.attach = self.shape_id = None
        self.ctx.request_repaint()

class DimAngularTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.vtx: Optional[Point] = None
        self.p1: Optional[Point] = None
        self.p2: Optional[Point] = None
        self.attach: Optional[Point] = None

    def on_mouse_press(self, ev):
        p = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
        if self.vtx is None:
            self.vtx = p
            self.ctx.update_status("AngularDim: pick first ray point")
        elif self.p1 is None:
            self.p1 = p
            self.ctx.update_status("AngularDim: pick second ray point")
        elif self.p2 is None:
            self.p2 = p
            self.ctx.update_status("AngularDim: place label")
        else:
            self.attach = p
            lab = angular_label(self.vtx, self.p1, self.p2, self.ctx.style)
            did = f"D{abs(hash((self.vtx,self.p1,self.p2,self.attach)))%10**6}"
            dim = AngularDimension(id=did, vtx=self.vtx, p1=self.p1, p2=self.p2, attach=self.attach, style=self.ctx.style)
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"AngularDim: {lab}")
            self.vtx = self.p1 = self.p2 = self.attach = None
        self.ctx.request_repaint()

class MeasureTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.last: Optional[str] = None

    def on_mouse_move(self, ev):
        p = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
        # Basic measure: nearest segment length or angle if 3 picks are retained; minimal demo
        # For v0.1, just show cursor world coords (extend later to arc/area as needed)
        self.ctx.update_status(f"X={p[0]:.3f}, Y={p[1]:.3f}")
