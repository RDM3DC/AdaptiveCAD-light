# adaptivecad_playground/dim_tools.py
"""
Dimension tools scaffolds. These are framework-light and expose simple callbacks.
Integrate by forwarding mouse events from your Canvas to the active tool.
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable, List, Any
from dataclasses import dataclass
from dimensions import LinearDimension, RadialDimension, AngularDimension, DimStyle, linear_label, radial_label, angular_label
from osnap import osnap_pick

Point = Tuple[float, float]

@dataclass
class ToolContext:
    get_params: Callable[[], dict]
    get_style: Callable[[], DimStyle]
    add_dimension: Callable[[object], None]
    update_status: Callable[[str], None]
    request_repaint: Callable[[], None]
    hit_test_circle: Callable[[Point], Optional[Tuple[Point, float, str]]]  # returns (center, radius, shape_id) if a circle is under cursor
    world_from_event: Callable[[object], Point]  # mouse event -> world (x,y)
    osnap_targets: Callable[[], List[Tuple[str, Any]]]  # named points or descriptors for snapping
    sample_curvature: Callable[[Point], float]
    sample_memory: Callable[[Point], float]
    sample_pi_a: Callable[[Point, float, Optional[dict]], float]

class DimLinearTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.p1: Optional[Point] = None
        self.p2: Optional[Point] = None
        self.offset: Optional[Point] = None

    def mouse_press(self, ev):
        p, _ = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
        if self.p1 is None:
            self.p1 = p
            self.ctx.update_status("LinearDim: pick second point")
        elif self.p2 is None:
            self.p2 = p
            self.ctx.update_status("LinearDim: place label")
        else:
            self.offset = p
            style = self.ctx.get_style()
            lab = linear_label(self.p1, self.p2, style)
            did = f"D{abs(hash((self.p1,self.p2,self.offset)))%10**6}"
            dim = LinearDimension(id=did, p1=self.p1, p2=self.p2, offset=self.offset, style=DimStyle(**style.asdict()))
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"LinearDim: {lab}")
            # reset
            self.p1 = self.p2 = self.offset = None
        self.ctx.request_repaint()

    def deactivate(self):
        self.p1 = self.p2 = self.offset = None
        self.ctx.update_status("")

    def mouse_move(self, ev):  # unused but keeps interface parity
        return None

    def mouse_release(self, ev):
        return None

    def key_press(self, ev):
        return None

class DimRadialTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.center: Optional[Point] = None
        self.radius: Optional[float] = None
        self.attach: Optional[Point] = None
        self.shape_id: Optional[str] = None

    def mouse_press(self, ev):
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
            style = self.ctx.get_style()
            params = self.ctx.get_params()
            params_copy = dict(params)
            lab = radial_label(self.center, self.radius, style, params_copy)
            did = f"D{abs(hash((self.center,self.radius,self.attach)))%10**6}"
            dim = RadialDimension(
                id=did,
                center=self.center,
                radius=self.radius,
                attach=self.attach,
                style=DimStyle(**style.asdict()),
                shape_ref=self.shape_id,
                params=params_copy,
            )
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"RadialDim: {lab}")
            self.center = self.radius = self.attach = self.shape_id = None
        self.ctx.request_repaint()

    def deactivate(self):
        self.center = None
        self.radius = None
        self.attach = None
        self.shape_id = None
        self.ctx.update_status("")

    def mouse_move(self, ev):  # placeholder for potential previews
        return None

    def mouse_release(self, ev):
        return None

    def key_press(self, ev):
        return None

class DimAngularTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.vtx: Optional[Point] = None
        self.p1: Optional[Point] = None
        self.p2: Optional[Point] = None
        self.attach: Optional[Point] = None

    def mouse_press(self, ev):
        p, _ = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
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
            style = self.ctx.get_style()
            lab = angular_label(self.vtx, self.p1, self.p2, style)
            did = f"D{abs(hash((self.vtx,self.p1,self.p2,self.attach)))%10**6}"
            dim = AngularDimension(id=did, vtx=self.vtx, p1=self.p1, p2=self.p2, attach=self.attach, style=DimStyle(**style.asdict()))
            self.ctx.add_dimension(dim)
            self.ctx.update_status(f"AngularDim: {lab}")
            self.vtx = self.p1 = self.p2 = self.attach = None
        self.ctx.request_repaint()

    def deactivate(self):
        self.vtx = self.p1 = self.p2 = self.attach = None
        self.ctx.update_status("")

    def mouse_move(self, ev):
        return None

    def mouse_release(self, ev):
        return None

    def key_press(self, ev):
        return None

class MeasureTool:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.last: Optional[str] = None

    def mouse_move(self, ev):
        p, _ = osnap_pick(self.ctx.world_from_event(ev), self.ctx.osnap_targets())
        params = self.ctx.get_params()
        curvature = self.ctx.sample_curvature(p)
        memory = self.ctx.sample_memory(p)
        pia_value = self.ctx.sample_pi_a(p, 1.0, params)
        self.ctx.update_status(
            f"X={p[0]:.3f}, Y={p[1]:.3f}, kappa≈{curvature:.4f}, pi_a≈{pia_value:.4f}, memory≈{memory:.4f}"
        )

    def mouse_press(self, ev):
        return None

    def mouse_release(self, ev):
        return None

    def key_press(self, ev):
        return None

    def deactivate(self):
        self.last = None
        self.ctx.update_status("")
