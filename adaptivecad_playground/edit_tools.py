"""UI helpers wiring edit operations to the canvas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from edit_ops import break_segment_at_point, extend_segment_to, join_polylines, trim_segment_to
from intersect2d import project_point_to_segment

Point = Tuple[float, float]


@dataclass
class EditCtx:
    tol: float
    get_selection: Callable[[], List[Any]]
    set_selection: Callable[[List[Any]], None]
    update_scene: Callable[[List[Any], List[Any]], None]
    world_from_event: Callable[[object], Point]
    osnap_targets: Callable[[], List[Tuple[str, Point]]]
    update_status: Callable[[str], None]
    request_repaint: Callable[[], None]
    clear_pointer: Optional[Callable[[], None]] = None


class JoinTool:
    def __init__(self, ctx: EditCtx) -> None:
        self.ctx = ctx

    def on_trigger(self) -> None:
        sel = self.ctx.get_selection()
        paths: List[List[Point]] = []
        for sh in sel:
            if sh.get("type") == "segment":
                paths.append([tuple(sh["p1"]), tuple(sh["p2"])])
            elif sh.get("type") in ("polyline", "curve"):
                paths.append([tuple(p) for p in sh["points"]])
        if not paths:
            self.ctx.update_status("Join: select segments or polylines first")
            return
        merged = join_polylines(paths, tol=self.ctx.tol)
        new_shapes = [{"type": "polyline", "points": path} for path in merged]
        self.ctx.update_scene(new_shapes, sel)
        self.ctx.update_status(f"Join: {len(paths)} → {len(merged)}")
        self.ctx.request_repaint()


class BreakTool:
    def __init__(self, ctx: EditCtx) -> None:
        self.ctx = ctx

    def mouse_press(self, ev) -> None:
        sel = self.ctx.get_selection()
        if not sel:
            self.ctx.update_status("Break: select a segment or polyline")
            return
        target = sel[0]
        p = self.ctx.world_from_event(ev)
        if target.get("type") == "segment":
            seg = (tuple(target["p1"]), tuple(target["p2"]))
            parts = break_segment_at_point(seg, p, tol=self.ctx.tol)
            if parts:
                add = [{"type": "segment", "p1": a, "p2": b} for (a, b) in parts]
                self.ctx.update_scene(add, [target])
                self.ctx.update_status("Break: segment split")
        elif target.get("type") in ("polyline", "curve"):
            pts = [tuple(q) for q in target["points"]]
            best_i: Optional[int] = None
            best_d = float("inf")
            best_proj: Optional[Point] = None
            for i in range(len(pts) - 1):
                proj, _ = project_point_to_segment(p, pts[i], pts[i + 1])
                d = ((p[0] - proj[0]) ** 2 + (p[1] - proj[1]) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_i = i
                    best_proj = proj
            if best_i is None or best_proj is None or best_d > self.ctx.tol:
                self.ctx.update_status("Break: click near a segment")
                return
            part_a = pts[: best_i + 1] + [best_proj]
            part_b = [best_proj] + pts[best_i + 1 :]
            add = [
                {"type": "polyline", "points": part_a},
                {"type": "polyline", "points": part_b},
            ]
            self.ctx.update_scene(add, [target])
            self.ctx.update_status("Break: polyline split")
        self.ctx.request_repaint()

    def deactivate(self) -> None:
        return


class TrimTool:
    def __init__(self, ctx: EditCtx) -> None:
        self.ctx = ctx
        self.stage = 0
        self.target: Optional[Any] = None

    def mouse_press(self, ev) -> None:
        sel = self.ctx.get_selection()
        if self.stage == 0:
            if not sel or sel[0].get("type") not in ("segment", "polyline", "curve"):
                self.ctx.update_status("Trim: select target segment first")
                return
            self.target = sel[0]
            self.stage = 1
            self.ctx.update_status("Trim: now select cutter")
            return
        if len(sel) < 2:
            self.ctx.update_status("Trim: select cutter as second item")
            return
        cutter = sel[1]
        if self.target and self.target.get("type") == "segment":
            seg = (tuple(self.target["p1"]), tuple(self.target["p2"]))
            new_seg = trim_segment_to(seg, cutter, tol=self.ctx.tol)
            if new_seg:
                self.ctx.update_scene(
                    [{"type": "segment", "p1": new_seg[0], "p2": new_seg[1]}],
                    [self.target],
                )
                self.ctx.update_status("Trim: done")
        self.stage = 0
        self.target = None
        self.ctx.request_repaint()

    def deactivate(self) -> None:
        self.stage = 0
        self.target = None
        return


class ExtendTool:
    def __init__(self, ctx: EditCtx) -> None:
        self.ctx = ctx
        self.stage = 0
        self.target: Optional[Any] = None

    def mouse_press(self, ev) -> None:
        sel = self.ctx.get_selection()
        if self.stage == 0:
            if not sel or sel[0].get("type") not in ("segment", "polyline", "curve"):
                self.ctx.update_status("Extend: select target segment first")
                return
            self.target = sel[0]
            self.stage = 1
            self.ctx.update_status("Extend: now select boundary")
            return
        if len(sel) < 2:
            self.ctx.update_status("Extend: select boundary as second item")
            return
        boundary = sel[1]
        if self.target and self.target.get("type") == "segment":
            seg = (tuple(self.target["p1"]), tuple(self.target["p2"]))
            new_seg = extend_segment_to(seg, boundary, tol=self.ctx.tol)
            if new_seg:
                self.ctx.update_scene(
                    [{"type": "segment", "p1": new_seg[0], "p2": new_seg[1]}],
                    [self.target],
                )
                self.ctx.update_status("Extend: done")
        self.stage = 0
        self.target = None
        self.ctx.request_repaint()

    def deactivate(self) -> None:
        self.stage = 0
        self.target = None
        return
